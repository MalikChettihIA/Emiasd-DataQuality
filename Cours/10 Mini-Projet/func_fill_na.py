from tensor_prov_utils import *

def fill_na_func(df: pd.DataFrame, column_to_fill: str):
    output_df = df.copy()
    mean_value = output_df[column_to_fill].mean()
    output_df[column_to_fill] = output_df[column_to_fill].fillna(mean_value)

    # Retourner le DataFrame complet
    return output_df

if __name__ == '__main__':
    #df_lengths_fill_na = [10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    df_lengths_fill_na = [25]

    function_args = {
        "column_to_fill": "age"
    }
    execute_and_log(df_lengths_fill_na, fill_na_func, function_args)
    exit(0)