import json

import polars as pl
from omegaconf import DictConfig
import hydra
from paths import get_config_path


def prepare_data(paths_cfg) -> pl.DataFrame:
    img_train_data = json.load(open(paths_cfg.img_pred_train_path))
    img_test_data = json.load(open(paths_cfg.img_pred_test_path))
    rows: list[dict] = []

    for data in (img_train_data, img_test_data):
        for cpr_child, patient_data in data.items():
            cpr_mother = patient_data.get("CPR_MOTHER")
            ga = patient_data.get("GA")
            birthday = patient_data.get("BIRTHDAY")
            imgs = patient_data.get("imgs", [])
            for img in imgs:
                rows.append({
                    "CPR_CHILD": cpr_child,
                    "CPR_MOTHER": cpr_mother,
                    "GA": ga,
                    "BIRTHDAY": birthday,
                    "STUDY_DATE": img.get("study_date"),
                    "PRED": img.get("pred"),
                })

    df = pl.DataFrame(rows)
    per_child = df.group_by("CPR_CHILD").len().rename({"len": "n_rows"})
    avg_rows = float(per_child["n_rows"].mean())
    n_children = per_child.height
    print(f"CPR_CHILD count={n_children:,} total_img_rows={df.height:,} avg_rows_per_CPR_CHILD={avg_rows:.4f}")
    print(df.head())
    return df


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    _ = prepare_data(cfg.paths)


if __name__ == "__main__":
    main()
