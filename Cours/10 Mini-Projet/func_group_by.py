from tensor_prov_utils import *


def group_by_func(df: pd.DataFrame, group_by_column: str, agg_column: str, agg_func: str, prov_column: str):
    aggregated_df = df.groupby(group_by_column).agg(
        **{agg_column: (agg_column, agg_func),  # Apply the aggregation to the specified column
           prov_column: (prov_column, lambda x: list(x))}  # Capture the provenance (hashes or ids)
    ).reset_index()

    return aggregated_df

if __name__ == '__main__':
    #df_lengths_group_by = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_group_by = [25]

    function_args = {
        "group_by_column": "city",
        "agg_column": "age",
        "agg_func": "mean"
    }
    execute_and_log(df_lengths_group_by, group_by_func, function_args)
    exit(0)