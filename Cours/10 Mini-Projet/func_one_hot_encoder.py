import pandas as pd
from tensor_prov_utils import *

def one_hot_encoder_func(df: pd.DataFrame) -> pd.DataFrame:

    one_hot_encoded = pd.get_dummies(df['city'], prefix='city')
    return pd.concat([df,one_hot_encoded], axis=1)

if __name__ == '__main__':
    #df_lengths_one_hot_encoder = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_one_hot_encoder = [25]
    function_args = {}
    execute_and_log(df_lengths_one_hot_encoder, one_hot_encoder_func, function_args)
    exit(0)