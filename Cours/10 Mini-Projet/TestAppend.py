import pandas as pd
from TensorProvUtils import *

def append_func(df1: pd.DataFrame, df2: pd.DataFrame, ignore_index:bool=True) -> pd.DataFrame:
    return pd.concat([df1,df2], ignore_index=ignore_index)

if __name__ == '__main__':
    #df_lengths_append = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_append = [10]
    function_args = {
        "ignore_index": True
    }
    execute_and_log(df_lengths_append, append_func, function_args, 2)
    exit(0)