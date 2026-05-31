import json
import os
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from paths import get_config_path
from sklearn.model_selection import train_test_split

from run_simple_matched_model import (
    align_to_columns,
    custom_functions,
    float_feature_matrix,
    one_hot_encode_data,
    print_metrics,
)


@dataclass
class MatchedMMPrep:
    X_train: pl.DataFrame
    y_train: np.ndarray
    X_test: pl.DataFrame
    y_test: np.ndarray
    X_train_img: pl.DataFrame
    y_train_img: np.ndarray
    X_test_img: pl.DataFrame
    y_test_img: np.ndarray
    train_row_ids: list[str]
    test_row_ids: list[str]
    train_img_row_ids: list[str]
    test_img_row_ids: list[str]
    img_pred_test_img: np.ndarray
    discards: list[str]


def unpack_img_preds(img_preds_data: dict, agg_func: str, id_col: str) -> pl.DataFrame:
    rows: list[dict] = []
    for cpr_child, patient_data in img_preds_data.items():
        imgs = patient_data.get("imgs", [])
        for img in imgs:
            rows.append({id_col: cpr_child, "img_pred": img.get("pred")})
    df = pl.DataFrame(rows)
    agg_func = agg_func.lower()
    if agg_func == "no_agg":
        return df
    if agg_func == "mean":
        return df.group_by(id_col).agg(pl.col("img_pred").mean().alias("img_pred"))
    if agg_func == "max":
        return df.group_by(id_col).agg(pl.col("img_pred").max().alias("img_pred"))
    if agg_func == "min":
        return df.group_by(id_col).agg(pl.col("img_pred").min().alias("img_pred"))
    raise ValueError(f"Invalid agg_func: {agg_func}. Expected one of: mean, max, min, no_agg")


def attach_img_pred(df: pl.DataFrame, img_df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    out = df.join(img_df, on=id_col, how="left")
    return out.with_columns(pl.col("img_pred").fill_null(0.0))


def feature_matrix(df: pl.DataFrame, id_col: str, label_col: str, train_cols: list[str] | None = None) -> pl.DataFrame:
    X = float_feature_matrix(one_hot_encode_data(df.drop([id_col, label_col])))
    if train_cols is None:
        return X
    return float_feature_matrix(align_to_columns(X, train_cols))


def prepare_data(paths_cfg: dict, data_cfg: dict) -> MatchedMMPrep:
    df = pl.read_csv(paths_cfg.tabular_ehr_path, null_values=[".", ""], try_parse_dates=True, infer_schema_length=100000)
    id_col = data_cfg.id_col
    label_col = data_cfg.label_col
    all_discards: list[str] = []

    if data_cfg.label_func is not None:
        func = custom_functions[data_cfg.label_func["func"]]
        df = func(df, **data_cfg.label_func["args"])
        discards = df.filter(pl.col(label_col).is_null()).get_column(id_col).drop_nulls().unique().to_list()
        print(f"Discarded {len(discards):,} rows due to null label")
        df = df.filter(~pl.col(id_col).is_in(discards))
        all_discards.extend(discards)
    df = df.drop([c for c in data_cfg.drop_feature_cols if c in df.columns])

    img_train_data = json.load(open(paths_cfg.img_pred_train_path))
    img_test_data = json.load(open(paths_cfg.img_pred_test_path))
    img_train_ids = set(img_train_data.keys())
    img_test_ids = set(img_test_data.keys())
    img_train_df = unpack_img_preds(img_train_data, str(data_cfg.agg_img_preds), id_col)
    img_test_df = unpack_img_preds(img_test_data, str(data_cfg.agg_img_preds), id_col)

    if paths_cfg.train_ids_path is not None:
        initial_train_ids = set(json.load(open(paths_cfg.train_ids_path)))
        initial_test_ids = set(json.load(open(paths_cfg.test_ids_path)))
        train_ids = list(
            initial_train_ids
            & set(df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list())
        )
        test_ids = list(
            initial_test_ids
            & set(df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list())
        )
    else:
        print("No train/test ids provided, using random split")
        all_ids = df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()
        train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    overlap_img_ids = img_test_ids & set(train_ids)
    if overlap_img_ids:
        print(
            f"WARNING: {len(overlap_img_ids)} overlapping ids between img test and ehr train!!! "
            "Removing them from train ids"
        )
        train_ids = list(set(train_ids) - overlap_img_ids)

    df_train = attach_img_pred(df.filter(pl.col(id_col).is_in(train_ids)), img_train_df, id_col)
    df_test = attach_img_pred(df.filter(pl.col(id_col).is_in(test_ids)), img_test_df, id_col)
    df_train_img = attach_img_pred(df.filter(pl.col(id_col).is_in(img_train_ids)), img_train_df, id_col)
    df_test_img = attach_img_pred(df.filter(pl.col(id_col).is_in(img_test_ids)), img_test_df, id_col)
    print(f"train_rows={df_train.height:,} test_rows={df_test.height:,}")
    print(f"train_img_rows={df_train_img.height:,} test_img_rows={df_test_img.height:,}")

    X_train = feature_matrix(df_train, id_col, label_col)
    train_cols = X_train.columns
    y_train = df_train.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    X_test = feature_matrix(df_test, id_col, label_col, train_cols)
    y_test = df_test.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()

    X_train_img = feature_matrix(df_train_img, id_col, label_col, train_cols)
    y_train_img = df_train_img.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    X_test_img = feature_matrix(df_test_img, id_col, label_col, train_cols)
    y_test_img = df_test_img.get_column(label_col).cast(pl.Float32, strict=False).to_numpy()
    img_pred_test_img = df_test_img.get_column("img_pred").cast(pl.Float32, strict=False).to_numpy()

    return MatchedMMPrep(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_img=X_train_img,
        y_train_img=y_train_img,
        X_test_img=X_test_img,
        y_test_img=y_test_img,
        train_row_ids=df_train.get_column(id_col).cast(pl.String, strict=False).to_list(),
        test_row_ids=df_test.get_column(id_col).cast(pl.String, strict=False).to_list(),
        train_img_row_ids=df_train_img.get_column(id_col).cast(pl.String, strict=False).to_list(),
        test_img_row_ids=df_test_img.get_column(id_col).cast(pl.String, strict=False).to_list(),
        img_pred_test_img=img_pred_test_img,
        discards=all_discards,
    )


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    p = prepare_data(cfg.paths, cfg.data)

    from models.xgb_model import XGBModel

    model = XGBModel(cfg.model.params)
    print(f"xgboost_device={model.device}")
    model.fit(p.X_train.to_numpy(), p.y_train)

    y_score = model.predict_proba(p.X_test.to_numpy())
    print_metrics("test", p.y_test, y_score)

    y_img_test_score = model.predict_proba(p.X_test_img.to_numpy())
    print_metrics("test_img", p.y_test_img, y_img_test_score)

    print_metrics("raw_img_pred test_img", p.y_test_img, p.img_pred_test_img)

    model_img = XGBModel(cfg.model.params)
    model_img.fit(p.X_train_img.to_numpy(), p.y_train_img)
    y_img_train_only_score = model_img.predict_proba(p.X_test_img.to_numpy())
    print_metrics("train_img_only test_img", p.y_test_img, y_img_train_only_score)

    important_features = model.get_important_features(feature_names=p.X_train.columns)
    print(f"important_features={important_features}")

    os.makedirs(Path(cfg.paths.predictions_path).parent, exist_ok=True)
    os.makedirs(Path(cfg.paths.discards_path).parent, exist_ok=True)

    y_img_train_score = model.predict_proba(p.X_train_img.to_numpy())
    pl.DataFrame({cfg.data.id_col: p.train_img_row_ids, "ehr_pred": y_img_train_score}).write_csv(
        f"{cfg.paths.predictions_path}_train.csv"
    )
    pl.DataFrame({cfg.data.id_col: p.test_img_row_ids, "ehr_pred": y_img_test_score}).write_csv(
        f"{cfg.paths.predictions_path}_test.csv"
    )

    with open(cfg.paths.discards_path, "w") as f:
        json.dump(p.discards, f)
    print(f"Wrote {cfg.paths.discards_path} with {len(p.discards):,} discards")


if __name__ == "__main__":
    main()
