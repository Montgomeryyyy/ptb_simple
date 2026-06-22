import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from paths import get_config_path
from sklearn.metrics import roc_auc_score
from utils import get_binary_label

from merge_img_ehr_csv import load_ehr_csv, load_img_csv

FEATURE_COLS = ("ehr_pred", "img_pred")

custom_functions = {
    "get_binary_label": get_binary_label,
}


def aggregate_img_preds(img_df: pl.DataFrame, agg_func: str, id_col: str) -> pl.DataFrame:
    df = img_df.with_columns(pl.col("img_pred").cast(pl.Float64, strict=False))
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
    spec = data_cfg.label_func
    func = custom_functions[str(spec["func"])]
    args = OmegaConf.to_container(spec["args"], resolve=True)
    return func(df, **args)


def print_img_scores(stage: str, img_df: pl.DataFrame, id_col: str) -> None:
    print(f"\n=== {stage} ===")
    print(f"rows={img_df.height:,} patients={img_df[id_col].n_unique():,}")
    print(img_df.select(id_col, "img_pred").head())
    preds = img_df.get_column("img_pred").drop_nulls()
    if preds.len():
        print(
            f"img_pred: min={preds.min():.4f} max={preds.max():.4f} "
            f"mean={preds.mean():.4f} median={preds.median():.4f}"
        )


def merge_labeled(
    ehr_df: pl.DataFrame,
    img_df: pl.DataFrame,
    id_col: str,
    data_cfg: DictConfig,
    label_col: str,
) -> pl.DataFrame:
    df = ehr_df.join(img_df, on=id_col, how="inner")
    df = get_label(df, data_cfg)
    return df.filter(
        pl.col(label_col).is_not_null()
        & pl.col("ehr_pred").is_not_null()
        & pl.col("img_pred").is_not_null()
    )


def prepare_data(
    paths_cfg: DictConfig,
    data_cfg: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    id_col = data_cfg.id_col
    label_col = data_cfg.label_col
    agg_func = str(data_cfg.agg_img_preds)

    img_train_raw = load_img_csv(paths_cfg.img_pred_train_path, id_col)
    img_test_raw = load_img_csv(paths_cfg.img_pred_test_path, id_col)
    print_img_scores("train img before agg", img_train_raw, id_col)
    print_img_scores("test img before agg", img_test_raw, id_col)

    img_train_df = aggregate_img_preds(img_train_raw, agg_func, id_col)
    img_test_df = aggregate_img_preds(img_test_raw, agg_func, id_col)
    print_img_scores(f"train img after agg ({agg_func})", img_train_df, id_col)
    print_img_scores(f"test img after agg ({agg_func})", img_test_df, id_col)

    ehr_train_df = load_ehr_csv(paths_cfg.ehr_pred_train_path, id_col)
    ehr_test_df = load_ehr_csv(paths_cfg.ehr_pred_test_path, id_col)

    train_df = merge_labeled(ehr_train_df, img_train_df, id_col, data_cfg, label_col)
    test_df = merge_labeled(ehr_test_df, img_test_df, id_col, data_cfg, label_col)
    test_img_before_df = merge_labeled(ehr_test_df, img_test_raw, id_col, data_cfg, label_col)

    print(f"\ntrain rows: {train_df.height:,} test rows: {test_df.height:,}")
    return train_df, test_df, test_img_before_df


def to_xy(df: pl.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(*FEATURE_COLS).cast(pl.Float32).to_numpy()
    y = df.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    return X, y


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
        f"[{name}] sens@spec={float(sens_at_spec.item()):.4f} spec@sens={float(spec_at_sens.item()):.4f} "
        f"auc={auc:.4f} prevalence={prevalence:.4f} n={len(y_true):,}"
    )


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_df, test_df, test_img_before_df = prepare_data(cfg.paths, cfg.data)
    X_tr, y_tr = to_xy(train_df, cfg.data.label_col)
    X_te, y_te = to_xy(test_df, cfg.data.label_col)

    y_img_before = test_img_before_df.get_column(cfg.data.label_col).cast(pl.Float32, strict=False).to_numpy()
    img_before = test_img_before_df.get_column("img_pred").cast(pl.Float32, strict=False).to_numpy()

    print_metrics("raw_ehr_pred", y_te, X_te[:, FEATURE_COLS.index("ehr_pred")])
    print_metrics("img_before_agg", y_img_before, img_before)
    print_metrics("img_after_agg", y_te, X_te[:, FEATURE_COLS.index("img_pred")])

    from models.mlp_model import MLPModel

    model = MLPModel(cfg.model.params)
    print(f"mlp_device={model.device}")
    model.fit(X_tr, y_tr)
    y_pred = model.predict_proba(X_te)
    print_metrics("mlp_fusion", y_te, y_pred)


if __name__ == "__main__":
    main()
