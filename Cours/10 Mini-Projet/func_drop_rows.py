from tensor_prov_utils import *

def drop_rows_func(df: pd.DataFrame, indexes: list):
    valid_indexes = [i for i in indexes if i in df.index]
    return df.copy().drop(index=valid_indexes)

if __name__ == '__main__':
    #df_lengths_query = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_query = [25]

    function_args = {
        "indexes": [1,7]
    }
    execute_and_log(df_lengths_query, drop_rows_func, function_args)
    exit(0)