import polars as pl
from omegaconf import DictConfig
import hydra
from paths import get_config_path


def load_img_csv(path: str, id_col: str) -> pl.DataFrame:
    return (
        pl.read_csv(path)
        .with_columns(pl.col("CPR_CHILD").cast(pl.String, strict=False).alias(id_col))
        .drop("CPR_CHILD")
        .rename({"pred": "img_pred"})
    )


def merge_split(
    split: str,
    img_path: str,
    ehr_path: str,
    id_col: str,
) -> pl.DataFrame:
    img_df = load_img_csv(img_path, id_col)
    ehr_df = pl.read_csv(ehr_path)

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

    train_df = merge_split(
        "train",
        cfg.paths.img_pred_train_path,
        cfg.paths.ehr_pred_train_path,
        id_col,
    )
    test_df = merge_split(
        "test",
        cfg.paths.img_pred_test_path,
        cfg.paths.ehr_pred_test_path,
        id_col,
    )


if __name__ == "__main__":
    main()
