import polars as pl
import polars.selectors as cs
import numpy as np
import os

from sklearn.metrics import roc_auc_score
from utils import get_binary_label
from omegaconf import DictConfig
import hydra
from paths import get_config_path
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

custom_functions = {
    "get_binary_label": get_binary_label,
}

def align_to_columns(df_dummies: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    cols = df_dummies.columns
    missing = [c for c in columns if c not in cols]
    extra = [c for c in cols if c not in columns]
    if missing:
        df_dummies = df_dummies.with_columns([pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing])
    if extra:
        df_dummies = df_dummies.drop(extra)
    return df_dummies.select(columns)

def one_hot_encode_data(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot non-numeric columns only; ints/floats/bools pass through as Float32.

    Integer columns (e.g. age, 0/1 flags) stay single features so importances use base names.
    """
    nb_names = df.select(cs.numeric() | cs.boolean()).columns
    nb_df = df.select(nb_names).select(pl.all().cast(pl.Float32)) if nb_names else None
    rest_names = [c for c in df.columns if c not in set(nb_names)]
    if not rest_names:
        assert nb_df is not None
        return nb_df
    dummies = df.select(rest_names).to_dummies()
    if nb_df is None:
        return dummies
    return pl.concat([nb_df, dummies], how="horizontal")


def float_feature_matrix(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.all().cast(pl.Float32))


def impute_train_medians(X_train: pl.DataFrame, X_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fill nulls with per-column train medians (0.0 if train column is all-null). Needed for MLP and stable X."""
    med_row = X_train.select(pl.all().median()).row(0)
    fills: list[float] = []
    for v in med_row:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            fills.append(0.0)
        else:
            fills.append(float(v))
    train_nc = sum(X_train.null_count().row(0))
    if train_nc:
        print(f"Imputing {train_nc:,} null feature cells in train (train medians → test uses same fills)")
    X_train_i = X_train.with_columns([pl.col(c).fill_null(fills[i]) for i, c in enumerate(X_train.columns)])
    X_test_i = X_test.with_columns([pl.col(c).fill_null(fills[i]) for i, c in enumerate(X_test.columns)])
    return X_train_i, X_test_i


def prepare_data(paths_cfg: dict, data_cfg: dict) -> tuple[list[str], list[int]]:
    df = pl.read_csv(paths_cfg.tabular_ehr_path, null_values=[".", ""], try_parse_dates=True, infer_schema_length=100000)
    id_col = data_cfg.id_col
    label_col = data_cfg.label_col
    all_discards = []

    if data_cfg.label_func is not None:
        func = custom_functions[data_cfg.label_func["func"]]
        df = func(df, **data_cfg.label_func["args"])
        discards = df.filter(pl.col(label_col).is_null()).get_column(id_col).drop_nulls().unique().to_list()
        print(f"Discarded {len(discards):,} rows due to null label")
        df = df.filter(~pl.col(id_col).is_in(discards))
        all_discards.extend(discards)
    df = df.drop([c for c in data_cfg.drop_feature_cols if c in df.columns])

    # Get train and test data
    df_ids = set(df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list())
    if paths_cfg.train_ids_path is not None or paths_cfg.test_ids_path is not None:
        initial_train_ids = (
            set(json.load(open(paths_cfg.train_ids_path))) if paths_cfg.train_ids_path is not None else None
        )
        initial_test_ids = (
            set(json.load(open(paths_cfg.test_ids_path))) if paths_cfg.test_ids_path is not None else None
        )
        n_initial_train = len(initial_train_ids) if initial_train_ids is not None else None
        n_initial_test = len(initial_test_ids) if initial_test_ids is not None else None
        print(f"initial_train_ids={n_initial_train} initial_test_ids={n_initial_test}")
        if initial_train_ids is not None:
            train_ids = list(initial_train_ids & df_ids)
        else:
            train_ids = list(df_ids - (initial_test_ids or set()))
        if initial_test_ids is not None:
            test_ids = list(initial_test_ids & df_ids)
        else:
            test_ids = list(df_ids - set(train_ids))
        print(f"train_ids={len(train_ids)} test_ids={len(test_ids)}")
    else:
        print("No train/test ids provided, using random split")
        train_ids = df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()
        test_ids = df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()
        train_ids, test_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    df_train = df.filter(pl.col(id_col).is_in(train_ids))
    df_test = df.filter(pl.col(id_col).is_in(test_ids))
    print(f"train_rows={df_train.height:,} test_rows={df_test.height:,}")

    # One-hot encode data
    X_train = float_feature_matrix(one_hot_encode_data(df_train.drop([id_col, label_col])))
    train_cols = X_train.columns
    y_train = df_train.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    X_test = float_feature_matrix(align_to_columns(
        one_hot_encode_data(df_test.drop([id_col, label_col])),
        train_cols,
    ))
    y_test = df_test.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    train_row_ids = df_train.get_column(id_col).cast(pl.String, strict=False).to_list()
    test_row_ids = df_test.get_column(id_col).cast(pl.String, strict=False).to_list()

    return X_train, y_train, X_test, y_test, all_discards, test_row_ids, train_row_ids


def save_encoding_json(
    path_prefix: str,
    split: str,
    row_ids: list[str],
    encodings: np.ndarray,
    suffix: str,
) -> None:
    out_path = f"{path_prefix}_{split}_{suffix}.json"
    payload = {str(row_id): encodings[i].tolist() for i, row_id in enumerate(row_ids)}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} with {len(payload):,} patients, {encodings.shape[1]} dims each")


def get_model(model_cfg: dict):
    # Import only one backend per run (XGBoost before Torch in xgb_model; no XGBoost in mlp_model).
    if model_cfg.name == "xgboost":
        from models.xgb_model import XGBModel
        return XGBModel(model_cfg.params)
    if model_cfg.name == "mlp":
        from models.mlp_model import MLPModel
        return MLPModel(model_cfg.params)
    raise ValueError(f"Invalid model name: {model_cfg.name}")

@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    X_train, y_train, X_test, y_test, discards, test_ids, train_row_ids = prepare_data(cfg.paths, cfg.data)

    # train + predict
    model = get_model(cfg.model)
    if cfg.model.name == "xgboost":
        print(f"xgboost_device={model.device}")
    if cfg.model.name == "mlp":
        print(f"mlp_device={model.device}")
        X_train, X_test = impute_train_medians(X_train, X_test)
    model.fit(X_train.to_numpy(), y_train)
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_score_train = model.predict_proba(X_train_np)
    y_score = model.predict_proba(X_test_np)
    auc = roc_auc_score(y_test, y_score)
    import torch
    from torchmetrics.classification import BinarySensitivityAtSpecificity, BinarySpecificityAtSensitivity

    sens_at_spec_metric = BinarySensitivityAtSpecificity(min_specificity=0.85)
    sens_at_spec, thr = sens_at_spec_metric(
        torch.tensor(y_score, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int64),
    )
    spec_at_sens_metric = BinarySpecificityAtSensitivity(min_sensitivity=0.70)
    spec_at_sens, thr = spec_at_sens_metric(
        torch.tensor(y_score, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int64),
    )
    print(f"auc={auc:.4f} sens_at_spec={float(sens_at_spec.item()):.4f} thr={float(thr.item()):.6g} spec_at_sens={float(spec_at_sens.item()):.4f} thr={float(thr.item()):.6g}")

    # get important features
    if cfg.model.name == "xgboost":
        important_features = model.get_important_features(feature_names=X_train.columns)
        print(f"important_features={important_features}")

    # save predictions (test + train; same row order as X_test / X_train)
    os.makedirs(Path(cfg.paths.predictions_path).parent, exist_ok=True)
    os.makedirs(Path(cfg.paths.discards_path).parent, exist_ok=True)
    id_key = cfg.data.id_col
    save_encoding = cfg.paths.get("save_encoding", cfg.paths.get("save_xgb_encoding", True))
    margin_train = margin_test = None
    if cfg.model.name == "xgboost" and save_encoding:
        margin_train = model.predict_margin(X_train_np)
        margin_test = model.predict_margin(X_test_np)

    out_test = pl.DataFrame({
        id_key: test_ids,
        "ehr_pred": y_score,
    })
    if margin_test is not None:
        out_test = out_test.with_columns(pl.Series("ehr_margin", margin_test))
    out_test.write_csv(f"{cfg.paths.predictions_path}_test.csv")
    print(f"Wrote {cfg.paths.predictions_path}_test.csv with {out_test.height:,} rows (test)")

    train_pred_path = f"{cfg.paths.predictions_path}_train.csv"
    out_train = pl.DataFrame({
        id_key: train_row_ids,
        "ehr_pred": y_score_train,
    })
    if margin_train is not None:
        out_train = out_train.with_columns(pl.Series("ehr_margin", margin_train))
    out_train.write_csv(train_pred_path)
    print(f"Wrote {train_pred_path} with {out_train.height:,} rows (train)")

    if cfg.model.name == "xgboost" and save_encoding:
        save_encoding_json(cfg.paths.predictions_path, "train", train_row_ids, model.predict_leaf(X_train_np), "leaf")
        save_encoding_json(cfg.paths.predictions_path, "test", test_ids, model.predict_leaf(X_test_np), "leaf")

    if cfg.model.name == "mlp" and save_encoding:
        save_encoding_json(
            cfg.paths.predictions_path, "train", train_row_ids, model.encode_last_hidden(X_train_np), "encoding"
        )
        save_encoding_json(
            cfg.paths.predictions_path, "test", test_ids, model.encode_last_hidden(X_test_np), "encoding"
        )

    # save discards
    with open(cfg.paths.discards_path, "w") as f:
        json.dump(discards, f)
    print(f"Wrote {cfg.paths.discards_path} with {len(discards):,} rows")

if __name__ == "__main__":
    main()
