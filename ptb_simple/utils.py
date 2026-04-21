import polars as pl

def get_python_operator(operator_str):
    if operator_str == "==":
        return lambda col, val: col.cast(pl.String) == val
    elif operator_str == "!=":
        return lambda col, val: col.cast(pl.String) != val
    elif operator_str == ">":
        return lambda col, val: col.cast(pl.Float64, strict=False) > val
    elif operator_str == "<":
        return lambda col, val: col.cast(pl.Float64, strict=False) < val
    elif operator_str == ">=":
        return lambda col, val: col.cast(pl.Float64, strict=False) >= val
    elif operator_str == "<=":
        return lambda col, val: col.cast(pl.Float64, strict=False) <= val
    else:
        raise NotImplementedError(f"Unknown operator: {operator_str}")


def get_binary_label(
    df: pl.DataFrame,
    col: str,
    operator: str,
    value: float,
    new_col: str,
) -> pl.DataFrame:
    operator_func = get_python_operator(operator)
    return df.with_columns(operator_func(pl.col(col), value).alias(new_col))