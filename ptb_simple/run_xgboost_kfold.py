import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from xgboost.core import XGBoostError


def detect_xgboost_device() -> str:
    """
    Return "cuda" if this xgboost build/runtime can train with GPU, else "cpu".
    We probe by fitting a tiny model once (fast) and catching CUDA-related failures.
    """
    X_probe = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y_probe = np.array([0, 0, 1, 1], dtype=int)
    probe = XGBClassifier(
        n_estimators=2,
        max_depth=2,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=0,
        n_jobs=-1,
    )
    try:
        probe.fit(X_probe, y_probe)
        return "cuda"
    except XGBoostError:
        return "cpu"

PATH = "../EHR_extract/outputs/table_test_make_main_table.yaml.csv"
LABEL_COL = "current_ptb"
ID_COL = "b_cpr"
OUT_PATH = "outputs/ptb_ehr_xgboost_risk_scores_kfold_oof.csv"
N_SPLITS = 5
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

def make_xgb_classifier(device: str, random_state: int = 42) -> XGBClassifier:
    """
    Prefer CUDA if available; fall back to CPU if not.
    We'll still retry-fit on CPU if the CUDA build/runtime isn't present.
    """
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=random_state,
        n_jobs=-1,
    )

def main(tabular_ehr_path: str = PATH) -> None:
    df = pl.read_csv(
        tabular_ehr_path,
        null_values=[".", ""],
        try_parse_dates=True,
        infer_schema_length=10000,
    )
    device = detect_xgboost_device()
    print(f"Using device: {device}")

    # Prepare data
    kept_cols = [c for c in df.columns if c not in (set(DROP_FEATURE_COLS) | {LABEL_COL, ID_COL})]
    clean_df = df.select([ID_COL, LABEL_COL, *kept_cols])

    # K-fold over everyone with a label
    df_labeled = clean_df.filter(pl.col(LABEL_COL).is_not_null()).with_row_index("_row")
    if df_labeled.height == 0:
        raise ValueError(f"No non-null labels found in {LABEL_COL!r}.")

    y = df_labeled.get_column(LABEL_COL).cast(pl.Int64, strict=False).to_numpy()
    n = df_labeled.height
    oof_proba = np.full(n, np.nan, dtype=float)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
        train_idx_s = pl.Series(train_idx).implode()
        val_idx_s = pl.Series(val_idx).implode()
        df_train = df_labeled.filter(pl.col("_row").is_in(train_idx_s))
        df_val = df_labeled.filter(pl.col("_row").is_in(val_idx_s))

        X_train_pl = one_hot_encode_data(df_train.drop([ID_COL, LABEL_COL, "_row"]))
        train_cols = X_train_pl.columns
        X_val_pl = one_hot_encode_data(df_val.drop([ID_COL, LABEL_COL, "_row"]))
        X_val_pl = align_to_columns(X_val_pl, train_cols)

        X_train_np = X_train_pl.to_numpy()
        X_val_np = X_val_pl.to_numpy()
        y_train = df_train.get_column(LABEL_COL).cast(pl.Int64, strict=False).to_numpy()
        y_val = df_val.get_column(LABEL_COL).cast(pl.Int64, strict=False).to_numpy()

        model = make_xgb_classifier(device=device, random_state=42)
        try:
            model.fit(X_train_np, y_train)
        except XGBoostError:
            # In case GPU was detected but fails at runtime for this fold, fall back.
            model = make_xgb_classifier(device="cpu", random_state=42)
            model.fit(X_train_np, y_train)
        proba_val = model.predict_proba(X_val_np)[:, 1]
        oof_proba[val_idx] = proba_val

        fold_auc = roc_auc_score(y_val, proba_val) if len(np.unique(y_val)) > 1 else float("nan")
        print(f"fold={fold_idx}/{N_SPLITS} n_train={len(train_idx):,} n_val={len(val_idx):,} auc={fold_auc:.4f}")

    overall_auc = roc_auc_score(y, oof_proba) if len(np.unique(y)) > 1 else float("nan")
    print(f"overall_oof_auc={overall_auc:.4f}")

    out = df_labeled.select([pl.col(ID_COL), pl.col(LABEL_COL)]).with_columns(
        [
            pl.Series("ptb_risk_oof", oof_proba),
            pl.col(LABEL_COL).alias("label"),
        ]
    )
    out.write_csv(OUT_PATH)
    print(f"Wrote {OUT_PATH} with {out.height:,} rows")

if __name__ == "__main__":
    main()
