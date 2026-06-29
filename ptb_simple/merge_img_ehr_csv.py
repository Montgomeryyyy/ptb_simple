import polars as pl
from omegaconf import DictConfig
import hydra
from paths import get_config_path


def load_img_csv(path: str, id_col: str) -> pl.DataFrame:
    return (
        pl.read_csv(path, infer_schema_length=1000000)
        .with_columns(pl.col("CPR_CHILD").cast(pl.String, strict=False).alias(id_col))
        .drop("CPR_CHILD")
        .rename({"preterm_pred": "img_pred"})
    )


def _unique_ids(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return df.select(pl.col(id_col).cast(pl.String, strict=False).alias(id_col)).unique()


def print_id_overlap(label: str, ehr_df: pl.DataFrame, img_df: pl.DataFrame, id_col: str) -> None:
    ehr_ids = _unique_ids(ehr_df, id_col)
    img_ids = _unique_ids(img_df, id_col)
    overlap = ehr_ids.join(img_ids, on=id_col, how="inner")
    n_ehr = ehr_ids.height
    n_img = img_ids.height
    n_overlap = overlap.height
    ehr_pct = 100.0 * n_overlap / n_ehr if n_ehr else 0.0
    img_pct = 100.0 * n_overlap / n_img if n_img else 0.0
    print(
        f"{label}: ehr_ids={n_ehr:,} img_ids={n_img:,} overlap={n_overlap:,} "
        f"({ehr_pct:.1f}% of ehr, {img_pct:.1f}% of img)"
    )


def load_ehr_csv(path: str, id_col: str) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=1000000).with_columns(pl.col(id_col).cast(pl.String, strict=False))


def merge_split(
    split: str,
    img_df: pl.DataFrame,
    ehr_df: pl.DataFrame,
    id_col: str,
) -> pl.DataFrame:
    print(f"{split} img rows: {img_df.height:,}")
    print(f"{split} ehr rows: {ehr_df.height:,}")

    merged = ehr_df.join(img_df, on=id_col, how="left")
    print(f"{split} merged rows: {merged.height:,}")
    return merged


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    id_col = cfg.data.id_col

    img_train_df = load_img_csv(cfg.paths.img_pred_train_path, id_col)
    img_test_df = load_img_csv(cfg.paths.img_pred_test_path, id_col)
    ehr_train_df = load_ehr_csv(cfg.paths.ehr_pred_train_path, id_col)
    ehr_test_df = load_ehr_csv(cfg.paths.ehr_pred_test_path, id_col)

    print("=== ID overlap (unique ids) ===")
    print_id_overlap("ehr_train ∩ img_train", ehr_train_df, img_train_df, id_col)
    print_id_overlap("ehr_test ∩ img_test", ehr_test_df, img_test_df, id_col)
    print_id_overlap("ehr_train ∩ img_test", ehr_train_df, img_test_df, id_col)
    print_id_overlap("ehr_test ∩ img_train", ehr_test_df, img_train_df, id_col)
    print()

    train_df = merge_split("train", img_train_df, ehr_train_df, id_col)
    test_df = merge_split("test", img_test_df, ehr_test_df, id_col)

    non_null_img_pred_train = train_df.filter(pl.col("img_pred").is_not_null())
    non_null_img_pred_test = test_df.filter(pl.col("img_pred").is_not_null())

    print(f"non_null_img_pred_train: {non_null_img_pred_train.height:,}")
    print(f"non_null_img_pred_test: {non_null_img_pred_test.height:,}")

    print(non_null_img_pred_train.head())
    print(non_null_img_pred_test.head())

if __name__ == "__main__":
    main()
