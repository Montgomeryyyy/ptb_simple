import polars as pl

from sklearn.metrics import roc_auc_score
from utils import get_binary_label
from omegaconf import DictConfig
import hydra
from paths import get_config_path
import json

custom_functions = {
    "get_binary_label": get_binary_label,
}

def align_to_columns(df_dummies: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    cols = df_dummies.columns
    missing = [c for c in columns if c not in cols]
    extra = [c for c in cols if c not in columns]
    if missing:
        df_dummies = df_dummies.with_columns([pl.lit(0).cast(pl.Int8).alias(c) for c in missing])
    if extra:
        df_dummies = df_dummies.drop(extra)
    return df_dummies.select(columns)

def one_hot_encode_data(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col(pl.Boolean).cast(pl.Int8)).to_dummies()


def prepare_data(paths_cfg: dict, data_cfg: dict) -> tuple[list[str], list[int]]:
    df = pl.read_csv(paths_cfg.tabular_ehr_path, null_values=[".", ""], try_parse_dates=True, infer_schema_length=10000)
    id_col = data_cfg.id_col
    label_col = data_cfg.label_col
    all_discards = []

    if data_cfg.label_func is not None:
        func = custom_functions[data_cfg.label_func["func"]]
        df = func(df, **data_cfg.label_func["args"])
        discards = df.filter(pl.col(label_col).is_null()).get_column(id_col).drop_nulls().unique().to_list()
        print(f"Discarded {len(discards):,} rows due to null label")
        df = df.filter(~pl.col(id_col).is_in(discards))
        all_discards.extend(discards)
    df = df.drop([c for c in data_cfg.drop_feature_cols if c in df.columns])

    # Get train and test data

    # df_train_tmp = pl.read_csv(paths_cfg.train_data_path, columns=[id_col])
    # df_test_tmp = pl.read_csv(paths_cfg.test_data_path, columns=[id_col])
    initial_train_ids = set(json.load(open(paths_cfg.train_ids_path)))
    initial_test_ids = set(json.load(open(paths_cfg.test_ids_path)))
    train_ids = list(initial_train_ids & set(df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()))
    test_ids = list(initial_test_ids & set(df.get_column(id_col).drop_nulls().cast(pl.String, strict=False).unique().to_list()))
    df_train = df.filter(pl.col(id_col).is_in(train_ids))
    df_test = df.filter(pl.col(id_col).is_in(test_ids))
    print(f"train_rows={df_train.height:,} test_rows={df_test.height:,}")

    # One-hot encode data
    X_train = one_hot_encode_data(df_train.drop([id_col, label_col]))
    train_cols = X_train.columns
    y_train = df_train.get_column(label_col).cast(pl.Int64, strict=False).to_numpy()
    X_test = one_hot_encode_data(df_test.drop([id_col, label_col]))
    X_test = align_to_columns(X_test, train_cols)
    y_test = df_test.get_column(label_col).cast(pl.Int64, strict=False).to_numpy()

    return X_train, y_train, X_test, y_test, all_discards, test_ids

def get_model(model_cfg: dict):
    # Import only one backend per run (XGBoost before Torch in xgb_model; no XGBoost in mlp_model).
    if model_cfg.name == "xgboost":
        from models.xgb_model import XGBModel
        return XGBModel(model_cfg.params)
    if model_cfg.name == "mlp":
        from models.mlp_model import MLPModel
        return MLPModel(model_cfg.params)
    raise ValueError(f"Invalid model name: {model_cfg.name}")

@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    X_train, y_train, X_test, y_test, discards, test_ids = prepare_data(cfg.paths, cfg.data)

    # train + predict
    model = get_model(cfg.model)
    if cfg.model.name == "xgboost":
        print(f"xgboost_device={model.device}")
    if cfg.model.name == "mlp":
        print(f"mlp_device={model.device}")
    model.fit(X_train.to_numpy(), y_train)
    y_score = model.predict_proba(X_test.to_numpy())
    auc = roc_auc_score(y_test, y_score)
    import torch
    from torchmetrics.classification import BinarySensitivityAtSpecificity

    sens_at_spec_metric = BinarySensitivityAtSpecificity(min_specificity=0.85)
    sens_at_spec, thr = sens_at_spec_metric(
        torch.tensor(y_score, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int64),
    )
    print(f"auc={auc:.4f} sens_at_spec={float(sens_at_spec.item()):.4f} thr={float(thr.item()):.6g}")

    # get important features
    if cfg.model.name == "xgboost":
        important_features = model.get_important_features(feature_names=X_train.columns)
        print(f"important_features={important_features}")

    # save predictions
    out = pl.DataFrame({
        "b_cpr": test_ids,
        "proba": y_score,
    })
    out.write_csv(cfg.paths.predictions_path)
    print(f"Wrote {cfg.paths.predictions_path} with {out.height:,} rows")

    # save discards
    with open(cfg.paths.discards_path, "w") as f:
        json.dump(discards, f)
    print(f"Wrote {cfg.paths.discards_path} with {len(discards):,} rows")

if __name__ == "__main__":
    main()
