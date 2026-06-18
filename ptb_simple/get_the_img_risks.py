import json
import warnings
import polars as pl


# -------------------------
# Paths
# -------------------------

json_path = "../../data/img_pop/train_first_preg.json"
parquet_path = "../../data/img_pop/train_first_preg.parquet"
out_path = "../data/img_pop/train_first_preg_with_metadata.json"

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
#     "child_id_1": [
#         pred_imgs,
#         embeddings
#     ],
#     "child_id_2": [
#         pred_imgs,
#         embeddings
#     ]
# }

enriched = {}

for id_child, values in data.items():
    metadata = meta_lookup.get(str(id_child))

    if metadata is None:
        warnings.warn(f"No metadata match found for id_child={id_child}. Skipping.")
        continue

    enriched[id_child] = {
        "pred_imgs": values[0],
        "embeddings": values[1],
        **metadata,
    }


# -------------------------
# Save enriched JSON
# -------------------------

with open(out_path, "w") as f:
    json.dump(enriched, f, indent=2)


print(f"Saved enriched JSON to: {out_path}")
print(f"Original JSON entries: {len(data)}")
print(f"Matched entries saved: {len(enriched)}")
print(f"Skipped entries: {len(data) - len(enriched)}")