from tensor_prov_utils import *

def drop_columns_func(df: pd.DataFrame, dropped_columns: list):
    """
    Drop function is a sample of Vertical Data Reduction - Drop columns.
    The operation, remove some of the attributes characterizing the data records in the input dataset Din and produce
    a next dataset Dout that reflects the dataset obtained as a result.
    """
    return df.drop(dropped_columns, axis=1)

if __name__ == '__main__':
    # Test is done len(df_lengths_reduce) iteration.
    #df_lengths_reduce = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_reduce = [25]
    function_args = {
        "dropped_columns": ["name", "city"]
    }
    execute_and_log(df_lengths_reduce, drop_columns_func, function_args)
    exit(0)