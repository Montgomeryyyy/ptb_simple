import json

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from paths import get_config_path
from sklearn.metrics import roc_auc_score
from utils import get_binary_label

FEATURE_COLS = ("img_pred",)

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

    train_df = unpack_img_preds(
        json.load(open(paths_cfg.img_pred_train_path)),
        str(data_cfg.agg_img_preds),
        id_col,
    )
    test_df = unpack_img_preds(
        json.load(open(paths_cfg.img_pred_test_path)),
        str(data_cfg.agg_img_preds),
        id_col,
    )

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


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_df, test_df = prepare_data(cfg.paths, cfg.data)
    _peek(train_df, test_df, title="img risk + label")

    label_col = cfg.data.label_col
    id_col = cfg.data.id_col

    for split, df in ("train", train_df), ("test", test_df):
        null_img = df.filter(pl.col("img_pred").is_null())
        n_rows = null_img.height
        n_ids = null_img.get_column(id_col).drop_nulls().n_unique()
        print(f"{split}: img_pred null -> {n_ids:,} unique {id_col} ({n_rows:,} rows)")

    # Two views:
    # - keep_null_img: keep all labeled rows; fill missing img_pred with 0.0
    # - drop_null_img: drop rows where img_pred is null
    train_keep = train_df.with_columns(pl.col("img_pred").fill_null(0.0))
    test_keep = test_df.with_columns(pl.col("img_pred").fill_null(0.0))
    train_drop = train_df.filter(pl.col("img_pred").is_not_null())
    test_drop = test_df.filter(pl.col("img_pred").is_not_null())

    import torch
    from torchmetrics.classification import BinarySensitivityAtSpecificity, BinarySpecificityAtSensitivity

    def eval_view(name: str, tr: pl.DataFrame, te: pl.DataFrame) -> None:
        y_tr = tr.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
        y_te = te.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
        prevalence = float(np.mean(y_te))

        # Baseline: use raw image risk score directly (no MLP fit)
        raw_score = te.get_column("img_pred").cast(pl.Float32).to_numpy()
        raw_auc = roc_auc_score(y_te, raw_score)
        raw_score_t = torch.tensor(raw_score, dtype=torch.float32)
        y_true_t = torch.tensor(y_te, dtype=torch.int64)
        raw_sens_at_spec, _ = BinarySensitivityAtSpecificity(min_specificity=0.85)(raw_score_t, y_true_t)
        raw_spec_at_sens, _ = BinarySpecificityAtSensitivity(min_sensitivity=0.70)(raw_score_t, y_true_t)
        print(
            f"[{name}] raw_img_pred auc={raw_auc:.4f} prevalence={prevalence:.4f} "
            f"sens@spec={float(raw_sens_at_spec.item()):.4f} "
            f"spec@sens={float(raw_spec_at_sens.item()):.4f} "
            f"train_n={tr.height:,} test_n={te.height:,}"
        )

        # MLP on the (possibly imputed) image risk score
        X_tr = tr.get_column("img_pred").cast(pl.Float32).to_numpy().reshape(-1, 1)
        X_te = te.get_column("img_pred").cast(pl.Float32).to_numpy().reshape(-1, 1)
        model = get_model(cfg.model)
        if cfg.model.name == "xgboost":
            print(f"xgboost_device={model.device}")
        if cfg.model.name == "mlp":
            print(f"mlp_device={model.device}")
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_te)

        auc = roc_auc_score(y_te, y_pred)
        y_score_t = torch.tensor(y_pred, dtype=torch.float32)
        sens_at_spec, _ = BinarySensitivityAtSpecificity(min_specificity=0.85)(y_score_t, y_true_t)
        spec_at_sens, _ = BinarySpecificityAtSensitivity(min_sensitivity=0.70)(y_score_t, y_true_t)
        print(
            f"[{name}] mlp auc={auc:.4f} prevalence={prevalence:.4f} "
            f"sens@spec={float(sens_at_spec.item()):.4f} "
            f"spec@sens={float(spec_at_sens.item()):.4f} "
            f"train_n={tr.height:,} test_n={te.height:,}"
        )

    eval_view("drop_null_img", train_drop, test_drop)
    eval_view("keep_null_img_fill0", train_keep, test_keep)


if __name__ == "__main__":
    main()
