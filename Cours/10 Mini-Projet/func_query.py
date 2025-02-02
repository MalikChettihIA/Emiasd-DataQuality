from tensor_prov_utils import *

def query_func(df: pd.DataFrame, condition: str):
    return df.query(condition)

if __name__ == '__main__':
    #df_lengths_query = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_query = [25]

    function_args = {
        "condition": "city == 'NY' and age > 20"
    }
    execute_and_log(df_lengths_query, query_func, function_args)
    exit(0)