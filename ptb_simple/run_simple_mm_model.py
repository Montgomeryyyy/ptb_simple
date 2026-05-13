import json

import polars as pl
from omegaconf import DictConfig
import hydra
from paths import get_config_path


def _log_img_ehr_id_overlap(
    img_df: pl.DataFrame,
    ehr_df: pl.DataFrame,
    id_col: str,
    split: str,
) -> None:
    if id_col not in img_df.columns:
        raise ValueError(f"{split}: image frame missing id_col {id_col!r}; columns={img_df.columns}")
    if id_col not in ehr_df.columns:
        raise ValueError(f"{split}: EHR frame missing id_col {id_col!r}; columns={ehr_df.columns}")
    img_ids = set(
        img_df.get_column(id_col).cast(pl.String, strict=False).drop_nulls().unique().to_list()
    )
    ehr_ids = set(
        ehr_df.get_column(id_col).cast(pl.String, strict=False).drop_nulls().unique().to_list()
    )
    overlap = img_ids & ehr_ids
    only_img = img_ids - ehr_ids
    only_ehr = ehr_ids - img_ids
    print(
        f"[{split}] id_col={id_col!r} img_unique={len(img_ids):,} ehr_unique={len(ehr_ids):,} "
        f"intersection={len(overlap):,} img_only={len(only_img):,} ehr_only={len(only_ehr):,}"
    )
    if only_img:
        sample = sorted(only_img)[:5]
        print(f"[{split}] img-only ids (first 5 of {len(only_img):,}): {sample}")
    if only_ehr:
        sample = sorted(only_ehr)[:5]
        print(f"[{split}] ehr-only ids (first 5 of {len(only_ehr):,}): {sample}")


def _log_test_ids_in_test_dfs(
    test_ids: list,
    img_test_df: pl.DataFrame,
    ehr_test_df: pl.DataFrame,
    id_col: str,
) -> None:
    """Report how many JSON test_ids appear in test-side img / EHR tables."""
    test_set = {str(x) for x in test_ids}
    img_test_ids = set(
        img_test_df.get_column(id_col).cast(pl.String, strict=False).drop_nulls().unique().to_list()
    )
    ehr_test_ids = set(
        ehr_test_df.get_column(id_col).cast(pl.String, strict=False).drop_nulls().unique().to_list()
    )
    in_img = test_set & img_test_ids
    in_ehr = test_set & ehr_test_ids
    in_both = in_img & in_ehr
    missing_img = test_set - img_test_ids
    missing_ehr = test_set - ehr_test_ids
    print(
        f"[test_ids_json] n={len(test_set):,} in img_test={len(in_img):,} in ehr_test={len(in_ehr):,} "
        f"in_both={len(in_both):,} missing_img={len(missing_img):,} missing_ehr={len(missing_ehr):,}"
    )
    if missing_img:
        print(f"[test_ids_json] not in img_test (first 5 of {len(missing_img):,}): {sorted(missing_img)[:5]}")
    if missing_ehr:
        print(f"[test_ids_json] not in ehr_test (first 5 of {len(missing_ehr):,}): {sorted(missing_ehr)[:5]}")


def _log_train_test_id_overlap(
    img_train_df: pl.DataFrame,
    img_test_df: pl.DataFrame,
    ehr_train_df: pl.DataFrame,
    ehr_test_df: pl.DataFrame,
    id_col: str,
) -> None:
    def _ids(df: pl.DataFrame) -> set[str]:
        return set(
            df.get_column(id_col).cast(pl.String, strict=False).drop_nulls().unique().to_list()
        )

    o_img = _ids(img_train_df) & _ids(img_test_df)
    o_ehr = _ids(ehr_train_df) & _ids(ehr_test_df)
    print(f"[train∩test] id_col={id_col!r} img overlap n={len(o_img):,} ehr overlap n={len(o_ehr):,}")
    if o_img:
        print(f"[train∩test] img ids in both splits (first 5 of {len(o_img):,}): {sorted(o_img)[:5]}")
    if o_ehr:
        print(f"[train∩test] ehr ids in both splits (first 5 of {len(o_ehr):,}): {sorted(o_ehr)[:5]}")


def unpack_img_preds(img_preds_data: dict, agg_func: str = "mean") -> pl.DataFrame:
    rows: list[dict] = []
    for cpr_child, patient_data in img_preds_data.items():
        cpr_mother = patient_data.get("CPR_MOTHER")
        ga = patient_data.get("GA")
        birthday = patient_data.get("BIRTHDAY")
        imgs = patient_data.get("imgs", [])
        for img in imgs:
            rows.append({
                "b_cpr": cpr_child,
                "m_cpr": cpr_mother,
                "GA_days": ga,
                "pregnancy_end": birthday,
                "scan_date": img.get("study_date"),
                "img_pred": img.get("pred"),
            })
    df = pl.DataFrame(rows)
    if agg_func == "mean":
        return df.group_by("b_cpr").agg(pl.col("img_pred").mean().alias("img_pred"))
    elif agg_func == "max":
        return df.group_by("b_cpr").agg(pl.col("img_pred").max().alias("img_pred"))
    elif agg_func == "min":
        return df.group_by("b_cpr").agg(pl.col("img_pred").min().alias("img_pred"))
    else:
        raise ValueError(f"Invalid agg_func: {agg_func}")

def prepare_data(paths_cfg, data_cfg) -> pl.DataFrame:
    img_train_data = json.load(open(paths_cfg.img_pred_train_path))
    img_test_data = json.load(open(paths_cfg.img_pred_test_path))
    img_train_df = unpack_img_preds(img_train_data, data_cfg.agg_img_preds)
    img_test_df = unpack_img_preds(img_test_data, data_cfg.agg_img_preds)

    ehr_train_df = pl.read_csv(paths_cfg.ehr_pred_train_path)
    ehr_test_df = pl.read_csv(paths_cfg.ehr_pred_test_path)

    id_col = data_cfg.id_col
    _log_img_ehr_id_overlap(img_train_df, ehr_train_df, id_col, "train")
    _log_img_ehr_id_overlap(img_test_df, ehr_test_df, id_col, "test")

    test_ids_path = paths_cfg.get("test_ids_path") if hasattr(paths_cfg, "get") else getattr(paths_cfg, "test_ids_path", None)
    if test_ids_path not in (None, ""):
        test_ids = json.load(open(test_ids_path))
        _log_test_ids_in_test_dfs(test_ids, img_test_df, ehr_test_df, id_col)

    _log_train_test_id_overlap(img_train_df, img_test_df, ehr_train_df, ehr_test_df, id_col)

    train_df = img_train_df.join(ehr_train_df, on=id_col, how="left")
    test_df = img_test_df.join(ehr_test_df, on=id_col, how="left")
    return train_df, test_df

@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_df, test_df = prepare_data(cfg.paths, cfg.data)
    print(train_df.head())
    print("n train rows:", train_df.height)
    print(test_df.head())
    print("n test rows:", test_df.height)


if __name__ == "__main__":
    main()
