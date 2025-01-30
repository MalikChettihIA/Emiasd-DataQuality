from TensorProvUtils import *

def reduce_func(df: pd.DataFrame, reduced_columns: list):
    """
    Reduce function is a sample of Vertical Data Reduction - Feature Selection.
    It takes a pandas dataframe as input, a list of feature as parameter and return the dataframe after reduction.
    """
    return df[reduced_columns]

if __name__ == '__main__':
    # Test is done len(df_lengths_reduce) iteration.
    #df_lengths_reduce = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_reduce = [25]
    function_args = {
        "reduced_columns": ["name", "city"]
    }
    execute_and_log(df_lengths_reduce, reduce_func, function_args)
    exit(0)