import json

import polars as pl
from omegaconf import DictConfig
import hydra
from paths import get_config_path

def unpack_img_preds(img_preds_data: dict, agg_func: str = "mean") -> pl.DataFrame:
    rows: list[dict] = []
    for cpr_child, patient_data in img_preds_data.items():
        cpr_mother = patient_data.get("CPR_MOTHER")
        ga = patient_data.get("GA")
        birthday = patient_data.get("BIRTHDAY")
        imgs = patient_data.get("imgs", [])
        for img in imgs:
            rows.append({
                "b_cpr": cpr_child,
                "m_cpr": cpr_mother,
                "GA_days": ga,
                "pregnancy_end": birthday,
                "scan_date": img.get("study_date"),
                "img_pred": img.get("pred"),
            })
    df = pl.DataFrame(rows)
    if agg_func == "mean":
        return df.group_by("b_cpr").agg(pl.col("img_pred").mean().alias("img_pred"))
    elif agg_func == "max":
        return df.group_by("b_cpr").agg(pl.col("img_pred").max().alias("img_pred"))
    elif agg_func == "min":
        return df.group_by("b_cpr").agg(pl.col("img_pred").min().alias("img_pred"))
    else:
        raise ValueError(f"Invalid agg_func: {agg_func}")

def prepare_data(paths_cfg, data_cfg) -> pl.DataFrame:
    img_train_data = json.load(open(paths_cfg.img_pred_train_path))
    img_test_data = json.load(open(paths_cfg.img_pred_test_path))
    img_train_df = unpack_img_preds(img_train_data, data_cfg.agg_img_preds)
    img_test_df = unpack_img_preds(img_test_data, data_cfg.agg_img_preds)

    ehr_train_df = pl.read_csv(paths_cfg.ehr_pred_train_path)
    ehr_test_df = pl.read_csv(paths_cfg.ehr_pred_test_path)

    train_df = img_train_df.join(ehr_train_df, on=data_cfg.id_col, how="left")
    test_df = img_test_df.join(ehr_test_df, on=data_cfg.id_col, how="left")
    return train_df, test_df

@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_df, test_df = prepare_data(cfg.paths, cfg.data)
    print(train_df.head())
    print("n train rows:", train_df.height)
    print(test_df.head())
    print("n test rows:", test_df.height)


if __name__ == "__main__":
    main()
