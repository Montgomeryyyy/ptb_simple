import json
import warnings

import polars as pl


# -------------------------
# Paths
# -------------------------

json_path = "../../data/img_pop/train_first_preg.json"
parquet_path = "../../data/img_pop/train_first_preg.parquet"
out_path = "../../data/img_pop/train_first_preg_with_metadata.csv"

# -------------------------
# Load JSON
# -------------------------

with open(json_path, "r") as f:
    data = json.load(f)


# -------------------------
# Load metadata from parquet
# -------------------------

df_meta = (
    pl.read_parquet(parquet_path)
    .select([
        "CPR_CHILD",
        "CPR_MOTHER",
        "GA",
        "study_date",
        "BIRTHDAY",
    ])
    .with_columns([
        pl.col("CPR_CHILD").cast(pl.Utf8),
        pl.col("study_date").cast(pl.Utf8),
        pl.col("BIRTHDAY").cast(pl.Utf8),
    ])
)


# -------------------------
# Create metadata lookup
# -------------------------

meta_lookup = {
    row["CPR_CHILD"]: {
        "CPR_MOTHER": row["CPR_MOTHER"],
        "GA": row["GA"],
        "study_date": row["study_date"],
        "BIRTHDAY": row["BIRTHDAY"],
    }
    for row in df_meta.iter_rows(named=True)
}


# -------------------------
# Combine JSON and metadata
# -------------------------
# Assumes JSON structure:
#
# {
#     "child_id_1": {
#         "pred_imgs": [[...], ...],
#         "embeddings": [...]
#     },
#     ...
# }

rows: list[dict] = []

for id_child, values in data.items():
    metadata = meta_lookup.get(str(id_child))

    if metadata is None:
        warnings.warn(f"No metadata match found for id_child={id_child}. Skipping.")
        continue

    for preds in values["pred"]:
        for pred in preds:
            rows.append({
                "CPR_CHILD": str(id_child),
                **metadata,
                "pred": pred,
            })


# -------------------------
# Save CSV
# -------------------------

df_out = pl.DataFrame(rows).with_columns(pl.col("pred").cast(pl.Float64, strict=False))
df_out.write_csv(out_path)

print(f"Saved CSV to: {out_path}")
print(f"Original JSON entries: {len(data)}")
print(f"Rows saved: {df_out.height:,}")
print(f"Unique CPR_CHILD: {df_out['CPR_CHILD'].n_unique():,}")
