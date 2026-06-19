import json

import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from paths import get_config_path
from sklearn.metrics import roc_auc_score
from utils import get_binary_label

@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_ehr_df = pl.read_csv(cfg.paths.ehr_pred_train_path)
    test_ehr_df = pl.read_csv(cfg.paths.ehr_pred_test_path)
    train_img_df = pl.read_csv(cfg.paths.img_pred_train_path)
    test_img_df = pl.read_csv(cfg.paths.img_pred_test_path)

    train_df = train_ehr_df.join(train_img_df, on=cfg.data.id_col, how="left")
    test_df = test_ehr_df.join(test_img_df, on=cfg.data.id_col, how="left")

    print(len(train_df), len(test_df))
    print(f"train_df.head(): {train_df.head()}")
    print(f"test_df.head(): {test_df.head()}")

if __name__ == "__main__":
    main()
