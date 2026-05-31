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


def unpack_img_preds(img_preds_data: dict, agg_func: str, id_col: str) -> pl.DataFrame:
    rows: list[dict] = []
    for cpr_child, patient_data in img_preds_data.items():
        cpr_mother = patient_data.get("CPR_MOTHER")
        ga = patient_data.get("GA")
        birthday = patient_data.get("BIRTHDAY")
        imgs = patient_data.get("imgs", [])
        for img in imgs:
            rows.append({
                id_col: cpr_child,
                "m_cpr": cpr_mother,
                "GA_days": ga,
                "pregnancy_end": birthday,
                "scan_date": img.get("study_date"),
                "img_pred": img.get("pred"),
            })
    df = pl.DataFrame(rows)
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

    img_train_df = unpack_img_preds(
        json.load(open(paths_cfg.img_pred_train_path)),
        str(data_cfg.agg_img_preds),
        id_col,
    )
    img_test_df = unpack_img_preds(
        json.load(open(paths_cfg.img_pred_test_path)),
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

        # Baseline sanity-check: treat raw img_pred as the score (no model fit)
        if feature_cols == ("img_pred",):
            raw_score = X_te[:, 0]
            raw_auc = roc_auc_score(y_te, raw_score)
            import torch
            from torchmetrics.classification import BinarySensitivityAtSpecificity, BinarySpecificityAtSensitivity

            raw_score_t = torch.tensor(raw_score, dtype=torch.float32)
            y_true_t = torch.tensor(y_te, dtype=torch.int64)
            raw_sens_at_spec, _ = BinarySensitivityAtSpecificity(min_specificity=0.85)(raw_score_t, y_true_t)
            raw_spec_at_sens, _ = BinarySpecificityAtSensitivity(min_sensitivity=0.70)(raw_score_t, y_true_t)
            print(
                f"[{name}] raw_img_pred auc={raw_auc:.4f} sens@spec={float(raw_sens_at_spec.item()):.4f} "
                f"spec@sens={float(raw_spec_at_sens.item()):.4f} test_n={X_te.shape[0]:,}"
            )

        model = get_model(cfg.model)
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_te)
        
        auc = roc_auc_score(y_te, y_pred)

        import torch
        from torchmetrics.classification import BinarySensitivityAtSpecificity, BinarySpecificityAtSensitivity

        y_score_t = torch.tensor(y_pred, dtype=torch.float32)
        y_true_t = torch.tensor(y_te, dtype=torch.int64)
        sens_at_spec, thr_s = BinarySensitivityAtSpecificity(min_specificity=0.85)(y_score_t, y_true_t)
        spec_at_sens, thr_e = BinarySpecificityAtSensitivity(min_sensitivity=0.70)(y_score_t, y_true_t)
        print(
            f"[{name}] auc={auc:.4f} sens@spec={float(sens_at_spec.item()):.4f} thr={float(thr_s.item()):.6g} "
            f"spec@sens={float(spec_at_sens.item()):.4f} thr={float(thr_e.item()):.6g} train_n={X_tr.shape[0]:,} test_n={X_te.shape[0]:,}"
        )


if __name__ == "__main__":
    main()
