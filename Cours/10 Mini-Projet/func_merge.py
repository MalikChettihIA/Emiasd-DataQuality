from typing import Literal

from tensor_prov_utils import *

def merge_func(df1: pd.DataFrame, df2: pd.DataFrame, on_merge: str, how_merge: Literal["left", "right", "inner", "outer", "cross"] = "inner") -> pd.DataFrame:
    """
    Join is a sample of Data Fusion operation.
    The join of the datasets Dl and Dr, implemented using the Merge operation in the
    Pandas library, and denoted by Dl ont
    C Dr, produces a dataset Dj as a result of joining
    Dl and Dr on a boolean condition C, where t represents the join type (inner, left outer,
    right outer, or full outer).
    """
    return df1.merge(df2, on=on_merge, how=how_merge, suffixes=('_x', '_y'))

if __name__ == '__main__':
    #df_lengths_joins = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_joins = [25]
    function_args = {
        "on_merge": "name",
        "how_merge": "inner"
    }
    execute_and_log(df_lengths_joins, merge_func, function_args, 2)
    exit(0)