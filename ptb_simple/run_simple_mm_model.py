import polars as pl
import polars.selectors as cs
import numpy as np

from sklearn.metrics import roc_auc_score
from utils import get_binary_label
from omegaconf import DictConfig
import hydra
from paths import get_config_path
import json
from sklearn.model_selection import train_test_split

custom_functions = {
    "get_binary_label": get_binary_label,
}


def prepare_data(paths_cfg) -> None:

    img_train_data = json.load(open(paths_cfg.img_pred_train_path))
    img_test_data = json.load(open(paths_cfg.img_pred_test_path))
    rows = []

    for data in [img_train_data, img_test_data]:
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
                    "PRED": img.get("pred")
                })

            df = pl.DataFrame(rows)

            print(df.head())
            return None


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    _ = prepare_data(cfg.paths, cfg.data)

    

if __name__ == "__main__":
    main()
