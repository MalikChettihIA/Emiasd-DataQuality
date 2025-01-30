import pandas as pd
from TensorProvUtils import *

def data_transformation_func(df1: pd.DataFrame) -> pd.DataFrame:
    return df1['age'].fillna(df1['age'].mean())

if __name__ == '__main__':
    df_lengths = [10, 100, 1000]
    execute_and_log(df_lengths, data_transformation_func, None)
    exit(0)