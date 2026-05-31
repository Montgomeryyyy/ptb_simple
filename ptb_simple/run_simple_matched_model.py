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
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

custom_functions = {
    "get_binary_label": get_binary_label,
}


@dataclass
class MatchedPrep:
    """Tabular + image-subset matrices and id sets from ``prepare_data``."""

    X_train: pl.DataFrame
    y_train: np.ndarray
    X_test: pl.DataFrame
    y_test: np.ndarray
    X_train_img: pl.DataFrame
    y_train_img: np.ndarray
    X_test_img: pl.DataFrame
    y_test_img: np.ndarray
    img_train_ids: set[str]
    img_test_ids: set[str]
    train_row_ids: list[str]
    test_row_ids: list[str]
    train_img_row_ids: list[str]
    test_img_row_ids: list[str]
    img_pred_test_img: np.ndarray
    discards: list[str]


def unpack_img_preds(img_preds_data: dict, agg_func: str, id_col: str) -> pl.DataFrame:
    rows: list[dict] = []
    for cpr_child, patient_data in img_preds_data.items():
        for img in patient_data.get("imgs", []):
            rows.append({id_col: cpr_child, "img_pred": img.get("pred")})
    df = pl.DataFrame(rows)
    agg_func = agg_func.lower()
    if agg_func == "no_agg":
        return df
    if agg_func == "mean":
        return df.group_by(id_col).agg(pl.col("img_pred").mean().alias("img_pred"))
    if agg_func == "max":
        return df.group_by(id_col).agg(pl.col("img_pred").max().alias("img_pred"))
    if agg_func == "min":
        return df.group_by(id_col).agg(pl.col("img_pred").min().alias("img_pred"))
    raise ValueError(f"Invalid agg_func: {agg_func}. Expected one of: mean, max, min, no_agg")


def print_metrics(name: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    import torch
    from torchmetrics.classification import BinarySensitivityAtSpecificity, BinarySpecificityAtSensitivity

    auc = roc_auc_score(y_true, y_score)
    prevalence = float(np.mean(y_true))
    y_score_t = torch.tensor(y_score, dtype=torch.float32)
    y_true_t = torch.tensor(y_true, dtype=torch.int64)
    sens_at_spec, _ = BinarySensitivityAtSpecificity(min_specificity=0.85)(y_score_t, y_true_t)
    spec_at_sens, _ = BinarySpecificityAtSensitivity(min_sensitivity=0.70)(y_score_t, y_true_t)
    print(
        f"{name} auc={auc:.4f} prevalence={prevalence:.4f} "
        f"sens_at_spec={float(sens_at_spec.item()):.4f} "
        f"spec_at_sens={float(spec_at_sens.item()):.4f} n={y_true.shape[0]:,}"
    )


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


def prepare_data(paths_cfg: dict, data_cfg: dict) -> MatchedPrep:
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

    # Get img train and test data
    img_train_data = json.load(open(paths_cfg.img_pred_train_path))
    img_test_data = json.load(open(paths_cfg.img_pred_test_path))
    img_train_ids = set(img_train_data.keys())
    img_test_ids = set(img_test_data.keys())
    agg_func = str(data_cfg.get("agg_img_preds", "mean"))
    img_train_preds = unpack_img_preds(img_train_data, agg_func, id_col)
    img_test_preds = unpack_img_preds(img_test_data, agg_func, id_col)
    df_train_img = df.filter(pl.col(id_col).is_in(img_train_ids)).join(img_train_preds, on=id_col, how="left")
    df_test_img = df.filter(pl.col(id_col).is_in(img_test_ids)).join(img_test_preds, on=id_col, how="left")
    print(f"train_img_rows={df_train_img.height:,} test_img_rows={df_test_img.height:,}")

    # Get train and test data
    if paths_cfg.train_ids_path is not None:
        initial_train_ids = set(json.load(open(paths_cfg.train_ids_path)))
        initial_test_ids = set(json.load(open(paths_cfg.test_ids_path)))
        train_ids = list(initial_train_ids & set(df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()))
        test_ids = list(initial_test_ids & set(df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()))
    else:
        print("No train/test ids provided, using random split")
        train_ids = df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()
        test_ids = df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()
        train_ids, test_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    overlap_img_ids = img_test_ids & set(train_ids)
    if overlap_img_ids:
        print(f"WARNING: {len(overlap_img_ids)} overlapping ids between img test and ehr train!!! Removing them from train ids")
        train_ids = list(set(train_ids) - overlap_img_ids)

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

    img_pred_test_img = (
        df_test_img.get_column("img_pred").fill_null(0.0).cast(pl.Float32, strict=False).to_numpy()
    )

    # Same column order / width as tabular X_* (image subsets see fewer raw categories otherwise).
    X_train_img = float_feature_matrix(align_to_columns(
        one_hot_encode_data(df_train_img.drop([id_col, label_col, "img_pred"])),
        train_cols,
    ))
    y_train_img = df_train_img.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    X_test_img = float_feature_matrix(align_to_columns(
        one_hot_encode_data(df_test_img.drop([id_col, label_col, "img_pred"])),
        train_cols,
    ))
    y_test_img = df_test_img.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    train_img_row_ids = df_train_img.get_column(id_col).cast(pl.String, strict=False).to_list()
    test_img_row_ids = df_test_img.get_column(id_col).cast(pl.String, strict=False).to_list()

    return MatchedPrep(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_img=X_train_img,
        y_train_img=y_train_img,
        X_test_img=X_test_img,
        y_test_img=y_test_img,
        img_train_ids=img_train_ids,
        img_test_ids=img_test_ids,
        train_row_ids=train_row_ids,
        test_row_ids=test_row_ids,
        train_img_row_ids=train_img_row_ids,
        test_img_row_ids=test_img_row_ids,
        img_pred_test_img=img_pred_test_img,
        discards=all_discards,
    )

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
    p = prepare_data(cfg.paths, cfg.data)

    # train + predict
    model = get_model(cfg.model)
    if cfg.model.name == "xgboost":
        print(f"xgboost_device={model.device}")
    if cfg.model.name == "mlp":
        print(f"mlp_device={model.device}")
        p.X_train, p.X_test = impute_train_medians(p.X_train, p.X_test)
        _, p.X_train_img = impute_train_medians(p.X_train, p.X_train_img)
        _, p.X_test_img = impute_train_medians(p.X_train, p.X_test_img)
    model.fit(p.X_train.to_numpy(), p.y_train)
    y_score = model.predict_proba(p.X_test.to_numpy())
    auc = roc_auc_score(p.y_test, y_score)
    import torch
    from torchmetrics.classification import BinarySensitivityAtSpecificity, BinarySpecificityAtSensitivity

    sens_at_spec_metric = BinarySensitivityAtSpecificity(min_specificity=0.85)
    sens_at_spec, _ = sens_at_spec_metric(
        torch.tensor(y_score, dtype=torch.float32),
        torch.tensor(p.y_test, dtype=torch.int64),
    )
    spec_at_sens_metric = BinarySpecificityAtSensitivity(min_sensitivity=0.70)
    spec_at_sens, _ = spec_at_sens_metric(
        torch.tensor(y_score, dtype=torch.float32),
        torch.tensor(p.y_test, dtype=torch.int64),
    )
    test_prevalence = float(np.mean(p.y_test))
    print(
        f"auc={auc:.4f} prevalence={test_prevalence:.4f} "
        f"sens_at_spec={float(sens_at_spec.item()):.4f} spec_at_sens={float(spec_at_sens.item()):.4f}"
    )

    # get important features
    if cfg.model.name == "xgboost":
        important_features = model.get_important_features(feature_names=p.X_train.columns)
        print(f"important_features={important_features}")

    # save predictions (test + train; same row order as X_test / X_train)
    os.makedirs(Path(cfg.paths.predictions_path).parent, exist_ok=True)
    os.makedirs(Path(cfg.paths.discards_path).parent, exist_ok=True)

    y_img_train_score = model.predict_proba(p.X_train_img.to_numpy())
    y_img_test_score = model.predict_proba(p.X_test_img.to_numpy())

    out_train = pl.DataFrame({
        cfg.data.id_col: p.train_img_row_ids,
        "ehr_pred": y_img_train_score,
    })
    out_train.write_csv(f"{cfg.paths.predictions_path}_train.csv")

    out_test = pl.DataFrame({
        cfg.data.id_col: p.test_img_row_ids,
        "ehr_pred": y_img_test_score,
    })
    out_test.write_csv(f"{cfg.paths.predictions_path}_test.csv")

    print_metrics("auc_img", p.y_test_img, y_img_test_score)

    print_metrics("raw_img_pred test_img", p.y_test_img, p.img_pred_test_img)

    model_img = get_model(cfg.model)
    X_train_img = p.X_train_img
    X_test_img = p.X_test_img
    if cfg.model.name == "mlp":
        X_train_img, X_test_img = impute_train_medians(X_train_img, X_test_img)
    model_img.fit(X_train_img.to_numpy(), p.y_train_img)
    y_img_train_only_score = model_img.predict_proba(X_test_img.to_numpy())
    print_metrics("train_img_only test_img", p.y_test_img, y_img_train_only_score)

    with open(cfg.paths.discards_path, "w") as f:
        json.dump(p.discards, f)
    print(f"Wrote {cfg.paths.discards_path} with {len(p.discards):,} discards")

if __name__ == "__main__":
    main()
