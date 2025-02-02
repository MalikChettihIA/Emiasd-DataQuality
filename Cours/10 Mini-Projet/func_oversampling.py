from tensor_prov_utils import *

def oversampling_func(df: pd.DataFrame, owersapling_times: int):
    df_duplicated = pd.concat([df] * owersapling_times, ignore_index=True)
    return df_duplicated

if __name__ == '__main__':
    #df_lengths_owersapling = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_owersapling = [25]
    function_args = {
        "owersapling_times": 2
    }
    execute_and_log(df_lengths_owersapling, oversampling_func, function_args)
    exit(0)