from typing import Literal

from TensorProvUtils import *

def merge_func(df1: pd.DataFrame, df2: pd.DataFrame, on_merge: str, how_merge: Literal["left", "right", "inner", "outer", "cross"] = "inner") -> pd.DataFrame:
    return df1.merge(df2, on=on_merge, how=how_merge, suffixes=('_x', '_y'))

if __name__ == '__main__':
    df_lengths_joins = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    function_args = {
        "on_merge": "name",
        "how_merge": "inner"
    }
    execute_and_log(df_lengths_joins, merge_func, function_args, 2)
    exit(0)