import polars as pl
import numpy as np
import random 

df = pl.read_csv(
    "../EHR_extract/outputs/table_test_make_main_table.yaml.csv",
)
random.seed(42)
sample_b_cprs = df.get_column("b_cpr").sample(n=1000, with_replacement=True).to_list()

rows_b_cpr: list[str] = []
rows_pred: list[int] = []
for b in sample_b_cprs:
    k = random.randint(1, 3)  # 1-3 entries per b_cpr
    rows_b_cpr.extend([b] * k)
    rows_pred.extend(random.choices(range(1, 13), k=k))  # 1-12 inclusive

random_out = pl.DataFrame(
    {
        "b_cpr": rows_b_cpr,
        "pred": rows_pred,
    }
)
random_out.write_csv("test_img_pred_scores.csv")