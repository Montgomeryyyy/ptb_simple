import json

import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from paths import get_config_path
from sklearn.metrics import roc_auc_score
from utils import get_binary_label

FEATURE_COLS = ("ehr_pred", "img_pred")

custom_functions = {
    "get_binary_label": get_binary_label,
}


def _aggregate_img_preds(df: pl.DataFrame, agg_func: str, id_col: str) -> pl.DataFrame:
    if not df.is_empty() and isinstance(df["img_pred"][0], list):
        df = df.explode("img_pred")
    df = df.with_columns(
        pl.col("img_pred").cast(pl.Float64, strict=False),
        pl.col("GA_days").cast(pl.Float64, strict=False),
    )
    agg_func = agg_func.lower()
    if agg_func == "no_agg":
        return df
    if agg_func == "mean":
        return df.group_by(id_col).agg(
            pl.col("img_pred").mean().alias("img_pred"),
            pl.col("GA_days").first().alias("GA_days"),
        )
    if agg_func == "max":
        return df.group_by(id_col).agg(
            pl.col("img_pred").max().alias("img_pred"),
            pl.col("GA_days").first().alias("GA_days"),
        )
    if agg_func == "min":
        return df.group_by(id_col).agg(
            pl.col("img_pred").min().alias("img_pred"),
            pl.col("GA_days").first().alias("GA_days"),
        )
    raise ValueError(f"Invalid agg_func: {agg_func}. Expected one of: mean, max, min, no_agg")


def unpack_img_preds(img_preds_data: dict, agg_func: str, id_col: str) -> pl.DataFrame:
    rows: list[dict] = []
    for cpr_child, patient_data in img_preds_data.items():
        scan_date = patient_data["study_date"]
        for preds in patient_data["pred"]:
            rows.append({
                id_col: str(cpr_child),
                "m_cpr": patient_data["CPR_MOTHER"],
                "GA_days": patient_data["GA"],
                "pregnancy_end": patient_data["BIRTHDAY"],
                "scan_date": scan_date,
                "img_pred": preds,
            })
    return _aggregate_img_preds(pl.DataFrame(rows), agg_func, id_col)


def load_img_preds_parquet(parquet_path: str, agg_func: str, id_col: str) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    df = df.select(
        pl.col("CPR_CHILD").cast(pl.String, strict=False).alias(id_col),
        pl.col("CPR_MOTHER").alias("m_cpr"),
        pl.col("GA").alias("GA_days"),
        pl.col("BIRTHDAY").alias("pregnancy_end"),
        pl.col("study_date"),
        pl.col("pred").alias("img_pred"),
    )
    return _aggregate_img_preds(df, agg_func, id_col)


def load_img_preds(path: str, agg_func: str, id_col: str) -> pl.DataFrame:
    if path.lower().endswith(".parquet"):
        return load_img_preds_parquet(path, agg_func, id_col)
    with open(path) as f:
        return unpack_img_preds(json.load(f), agg_func, id_col)


def get_label(df: pl.DataFrame, data_cfg: DictConfig) -> pl.DataFrame:
    if data_cfg.label_func is None:
        raise ValueError("data.label_func is required (e.g. get_binary_label on GA_days).")
    spec = data_cfg.label_func
    func = custom_functions[str(spec["func"])]
    args = OmegaConf.to_container(spec["args"], resolve=True)
    return func(df, **args)


def get_model(model_cfg: dict):
    if model_cfg.name == "xgboost":
        from models.xgb_model import XGBModel
        return XGBModel(model_cfg.params)
    if model_cfg.name == "mlp":
        from models.mlp_model import MLPModel
        return MLPModel(model_cfg.params)
    raise ValueError(f"Invalid model name: {model_cfg.name}")


def prepare_data(paths_cfg: DictConfig, data_cfg: DictConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    id_col = data_cfg.id_col
    label_col = data_cfg.label_col

    img_train_df = load_img_preds(
        paths_cfg.img_pred_train_path,
        str(data_cfg.agg_img_preds),
        id_col,
    )
    img_test_df = load_img_preds(
        paths_cfg.img_pred_test_path,
        str(data_cfg.agg_img_preds),
        id_col,
    )

    ehr_train_df = pl.read_csv(paths_cfg.ehr_pred_train_path)
    ehr_test_df = pl.read_csv(paths_cfg.ehr_pred_test_path)

    train_df = img_train_df.join(ehr_train_df, on=id_col, how="left")
    test_df = img_test_df.join(ehr_test_df, on=id_col, how="left")

    train_df = get_label(train_df, data_cfg)
    test_df = get_label(test_df, data_cfg)

    n_drop_tr = train_df.filter(pl.col(label_col).is_null()).height
    n_drop_te = test_df.filter(pl.col(label_col).is_null()).height
    if n_drop_tr or n_drop_te:
        print(f"Dropped rows with null {label_col}: train={n_drop_tr:,} test={n_drop_te:,}")
    train_df = train_df.filter(pl.col(label_col).is_not_null())
    test_df = test_df.filter(pl.col(label_col).is_not_null())

    return train_df, test_df


def _peek(train_df: pl.DataFrame, test_df: pl.DataFrame, *, title: str) -> None:
    print(f"=== {title} ===")
    for split, df in ("train", train_df), ("test", test_df):
        print(df.head())
        print(f"n {split} rows: {df.height:,}")


def filter_input_data(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    label_col: str,
    *,
    feature_cols: tuple[str, ...],
    drop_ehr_null: bool,
    ehr_fill: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def one(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        d = df.filter(pl.col(label_col).is_not_null())
        if drop_ehr_null:
            d = d.filter(pl.col("ehr_pred").is_not_null())
        elif "ehr_pred" in feature_cols:
            d = d.with_columns(pl.col("ehr_pred").fill_null(ehr_fill))
        if "img_pred" in feature_cols:
            d = d.with_columns(pl.col("img_pred").fill_null(0.0))
        X = d.select(*feature_cols).cast(pl.Float32).to_numpy()
        y = d.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
        return X, y

    X_tr, y_tr = one(train_df)
    X_te, y_te = one(test_df)
    return X_tr, y_tr, X_te, y_te


def print_metrics(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    train_n: int | None = None,
    test_n: int | None = None,
) -> None:
    import torch
    from torchmetrics.classification import BinarySensitivityAtSpecificity, BinarySpecificityAtSensitivity

    auc = roc_auc_score(y_true, y_score)
    prevalence = float(np.mean(y_true))
    y_score_t = torch.tensor(y_score, dtype=torch.float32)
    y_true_t = torch.tensor(y_true, dtype=torch.int64)
    sens_at_spec, _ = BinarySensitivityAtSpecificity(min_specificity=0.85)(y_score_t, y_true_t)
    spec_at_sens, _ = BinarySpecificityAtSensitivity(min_sensitivity=0.70)(y_score_t, y_true_t)
    counts = ""
    if train_n is not None:
        counts = f" train_n={train_n:,} test_n={test_n:,}"
    elif test_n is not None:
        counts = f" test_n={test_n:,}"
    print(
        f"[{name}] sens@spec={float(sens_at_spec.item()):.4f} spec@sens={float(spec_at_sens.item()):.4f} "
        f"auc={auc:.4f} prevalence={prevalence:.4f}{counts}"
    )


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_df, test_df = prepare_data(cfg.paths, cfg.data)
    _peek(train_df, test_df, title="joined img + EHR + label")

    label_col = cfg.data.label_col
    id_col = cfg.data.id_col

    for split, df in ("train", train_df), ("test", test_df):
        null_img = df.filter(pl.col("img_pred").is_null())
        n_rows = null_img.height
        n_ids = null_img.get_column(id_col).drop_nulls().n_unique()
        print(f"{split}: dropping img_pred null → {n_ids:,} unique {id_col} ({n_rows:,} rows)")

    train_df = train_df.filter(pl.col("img_pred").is_not_null())
    test_df = test_df.filter(pl.col("img_pred").is_not_null())

    variants: tuple[tuple[str, tuple[str, ...], bool, float], ...] = (
        ("ehr_nonnull", FEATURE_COLS, True, 0.0),
        ("fill_null_ehr", FEATURE_COLS, False, 0.0),
        ("img_only", ("img_pred",), False, 0.0),
        ("img_only_ehr_nonnull", ("img_pred",), True, 0.0),
    )

    for name, feature_cols, drop_ehr_null, ehr_fill in variants:
        X_tr, y_tr, X_te, y_te = filter_input_data(
            train_df,
            test_df,
            label_col,
            feature_cols=feature_cols,
            drop_ehr_null=drop_ehr_null,
            ehr_fill=ehr_fill,
        )
        if X_tr.shape[0] < 2 or X_te.shape[0] < 1:
            print(f"[{name}] skip: insufficient rows (train={X_tr.shape[0]}, test={X_te.shape[0]})")
            continue

        if feature_cols == ("img_pred",):
            print_metrics(f"{name} raw_img_pred", y_te, X_te[:, 0], test_n=X_te.shape[0])
        if "ehr_pred" in feature_cols:
            ehr_idx = feature_cols.index("ehr_pred")
            print_metrics(f"{name} raw_ehr_pred", y_te, X_te[:, ehr_idx], test_n=X_te.shape[0])

        model = get_model(cfg.model)
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_te)
        print_metrics(name, y_te, y_pred, train_n=X_tr.shape[0], test_n=X_te.shape[0])


if __name__ == "__main__":
    main()
