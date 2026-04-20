import polars as pl
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EHR_PREDS =  "outputs/ptb_ehr_mlp_risk_scores.csv"
IMG_PREDS = "outputs/test_img_pred_scores.csv"
LABEL_PATH = "../EHR_extract/outputs/table_test_make_main_table.yaml.csv"
LABEL_COL = "current_ptb"
ID_COL = "b_cpr"
OUT_PATH = "outputs/ptb_combined_risk_scores.csv"
DISCARDS_PATH = "outputs/ptb_combined_risk_scores_discards.json"

def maintain_discards(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    # Track rows we drop (and why) as Python lists
    dropped = (
        df.with_columns(
            [
                pl.col("img_pred").is_null().alias("missing_img_pred"),
                pl.col("ehr_risk").is_null().alias("missing_ehr_risk"),
                pl.col(LABEL_COL).is_null().alias("missing_label"),
            ]
        )
        .filter(pl.col("missing_img_pred") | pl.col("missing_ehr_risk") | pl.col("missing_label"))
    )
    if dropped.height > 0:
        discards_img = (
            dropped.filter(pl.col("missing_img_pred"))
            .get_column(ID_COL)
            .drop_nulls()
            .unique()
            .to_list()
        )
        discards_ehr = (
            dropped.filter(pl.col("missing_ehr_risk"))
            .get_column(ID_COL)
            .drop_nulls()
            .unique()
            .to_list()
        )
    else:
        discards_img = []
        discards_ehr = []
    return discards_img, discards_ehr

def main(ehr_preds_path: str, img_preds_path: str, label_path: str):
    ehr_preds = pl.read_csv(ehr_preds_path)
    label_df = pl.read_csv(label_path)
    img_preds = pl.read_csv(img_preds_path)

    label_pairs = (
        label_df.select([pl.col(ID_COL), pl.col(LABEL_COL).cast(pl.Int64, strict=False)])
        .filter(pl.col(LABEL_COL).is_not_null())
    )

    # Use the two prediction values available per row after join:
    # - `pred` from image model (may have multiple rows per `b_cpr`)
    # - `ptb_risk` from tabular EHR model (one row per `b_cpr`)
    combined_pre = (
        img_preds.join(ehr_preds, on=ID_COL, how="left")
        .join(label_pairs, on=ID_COL, how="inner")
        .with_columns(
            [
                pl.col("pred").cast(pl.Float64, strict=False).alias("img_pred"),
                pl.col("ptb_risk").cast(pl.Float64, strict=False).alias("ehr_risk"),
            ]
        )
    )
    discards_img, discards_ehr = maintain_discards(combined_pre)
    combined = combined_pre.drop_nulls(["img_pred", "ehr_risk", LABEL_COL])

    X = combined.select(["ehr_risk", "img_pred"]).to_numpy()
    y = combined.get_column(LABEL_COL).to_numpy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.2,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    print(f"n_rows={combined.height:,} auc={auc:.4f}")

    out = combined.select(
        [
            pl.col(ID_COL),
            pl.col("ehr_risk"),
            pl.col("img_pred"),
            pl.Series("ptb_risk_combined", proba),
            pl.col(LABEL_COL).alias("label"),
        ]
    )
    out.write_csv(OUT_PATH)
    print(f"Wrote {OUT_PATH} with {out.height:,} rows")

    all_discards = {
        "img": 
            {
                "n_rows": len(discards_img),
                "b_cprs": discards_img,
            },
        "ehr": {
            "n_rows": len(discards_ehr),
            "b_cprs": discards_ehr,
        },
    }
    with open(DISCARDS_PATH, "w") as f:
        json.dump(all_discards, f)

if __name__ == "__main__":
    main(EHR_PREDS, IMG_PREDS, LABEL_PATH)
