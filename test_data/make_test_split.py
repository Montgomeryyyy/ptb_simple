import polars as pl
import numpy as np
import random

df = pl.read_csv(
    "../EHR_extract/outputs/table_test_make_main_table.yaml.csv",
    null_values=[".", ""],
    try_parse_dates=True,
    infer_schema_length=10000,
)

ID_COL = "b_cpr"
TEST_FRAC = 0.2
SEED = 42
TRAIN_OUT = "test_data/train.csv"
TEST_OUT = "test_data/test.csv"

random.seed(SEED)
unique_ids = df.get_column(ID_COL).drop_nulls().unique().to_list()

n_test = max(1, int(round(len(unique_ids) * TEST_FRAC)))
test_ids = set(random.sample(unique_ids, k=n_test))

df_test = df.filter(pl.col(ID_COL).is_in(list(test_ids)))
df_train = df.filter(~pl.col(ID_COL).is_in(list(test_ids)))

df_train.write_csv(TRAIN_OUT)
df_test.write_csv(TEST_OUT)

print(f"unique_ids={len(unique_ids):,} test_ids={len(test_ids):,}")
print(f"train_rows={df_train.height:,} test_rows={df_test.height:,}")
print(f"Wrote {TRAIN_OUT} and {TEST_OUT}")

