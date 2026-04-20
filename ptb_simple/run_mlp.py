import polars as pl

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PATH = "../EHR_extract/outputs/table_test_make_main_table.yaml.csv"
IMG_PREDS = "outputs/test_img_pred_scores.csv"
LABEL_COL = "current_ptb"
ID_COL = "b_cpr"
OUT_PATH = "outputs/ptb_ehr_mlp_risk_scores.csv"
DROP_FEATURE_COLS = {
    "m_cpr",
    "pregnancy_start",
    "pregnancy_end",
    "GA_days",
    "GA_weeks",
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

def main(tabular_ehr_path: str, img_preds_path: str):
    df = pl.read_csv(tabular_ehr_path)
    img_preds = pl.read_csv(img_preds_path)
    img_b_cprs = img_preds.get_column(ID_COL).drop_nulls().cast(pl.String, strict=False).unique().to_list()

    # Prepare data
    kept_cols = [c for c in df.columns if c not in (set(DROP_FEATURE_COLS) | {LABEL_COL, ID_COL})]
    clean_df = df.select([ID_COL, LABEL_COL, *kept_cols])

    # Test set: labeled rows whose b_cpr appears in IMG_PREDS
    df_test = clean_df.filter(pl.col(ID_COL).is_in(list(img_b_cprs)))
    df_train = clean_df.filter(~pl.col(ID_COL).is_in(list(img_b_cprs)))

    X_train_pl = one_hot_encode_data(df_train.drop([ID_COL, LABEL_COL]))
    train_cols = X_train_pl.columns

    y_train = df_train.get_column(LABEL_COL).cast(pl.Int64, strict=False).to_numpy()

    X_test_pl = one_hot_encode_data(df_test.drop([ID_COL, LABEL_COL]))
    X_test_pl = align_to_columns(X_test_pl, train_cols)
    y_test = df_test.get_column(LABEL_COL).cast(pl.Int64, strict=False).to_numpy()

    X_train_np = X_train_pl.to_numpy()
    X_test_np = X_test_pl.to_numpy()

    # MLP trains best with standardized numeric inputs
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train_np, y_train)

    proba = model.predict_proba(X_test_np)[:, 1]
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")

    print(
        "split="
        f"train(labeled_not_in_img_preds)={df_train.height:,} "
        f"test(labeled_in_img_preds)={df_test.height:,} "
        f"n_features_model={X_train_np.shape[1]:,}"
    )
    print(f"auc={auc:.4f}")

    # Save probabilities for the TEST set only
    out = df_test.select([pl.col(ID_COL)]).with_columns(
        pl.Series("ptb_risk", proba)
    )
    out.write_csv(OUT_PATH)
    print(f"Wrote {OUT_PATH} with {out.height:,} rows")

if __name__ == "__main__":
    main(PATH, IMG_PREDS)
