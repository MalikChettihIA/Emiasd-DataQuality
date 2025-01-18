import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import hashlib
import time

class TensorProv:

    def __init__(self, function_name, method='hash'):
        self.function_name = function_name
        self.method = method

    def __call__(self, *args, **kwargs):
        if self.function_name == 'query':
            if self.method == 'ids':
                return TensorProv.query_with_ids_provenance(*args, **kwargs)
            else:
                return TensorProv.query_with_hash_provenance(*args, **kwargs)
        elif self.function_name == 'merge'
            if self.method == 'ids':
                return TensorProv.merge_with_ids_provenance(*args, **kwargs)
            else:
                return TensorProv.merge_with_hash_provenance(*args, **kwargs)
        else:
            raise ValueError(f"Unknown method '{self.method}' for query.")

    @staticmethod
    def hash_row_content(row: pd.Series) -> str:
        row_str = row.to_json(date_format="iso", orient="columns")
        return hashlib.md5(row_str.encode("utf-8")).hexdigest()

    @staticmethod
    def query_with_hash_provenance(df: pd.DataFrame, query_expr: str):

        df["_hash_"] = df.apply(TensorProv.hash_row_content, axis=1)
        filtered_df = df.query(query_expr).copy()

        start_time = time.time()
        N = len(df)  # total rows in original df
        M = len(filtered_df)  # total rows in filtered df

        hash_to_position = {
            h: i for i, h in enumerate(df["_hash_"].values)
        }

        row_positions = np.arange(M, dtype=np.int32)
        col_positions = np.empty(M, dtype=np.int32)
        data = np.ones(M, dtype=np.int8)

        # For each row i in filtered_df, find the original row position via the hash
        for i in range(M):
            row_hash = filtered_df.iloc[i]["_hash_"]
            if row_hash not in hash_to_position:
                raise ValueError(f"Hash {row_hash} in filtered_df not found in original_df. "
                                 f"Possible data mismatch or missing row?")
            orig_pos = hash_to_position[row_hash]
            col_positions[i] = orig_pos

        provenance_matrix = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(M, N)
        )
        end_time = time.time()

        return df, filtered_df, (end_time-start_time), provenance_matrix

    @staticmethod
    def query_with_ids_provenance(df: pd.DataFrame, query_expr: str):

        df["_id_"] = df.index
        filtered_df = df.query(query_expr).copy()

        start_time = time.time()
        N = len(df)
        M = len(filtered_df)

        orig_indices = filtered_df["_id_"].to_numpy()
        row_positions = np.arange(M)
        col_positions = orig_indices

        # The data array simply holds ones for each match
        data = np.ones(M, dtype=np.int8)

        # Build the sparse COO matrix
        provenance_matrix = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(M, N)
        )
        end_time = time.time()

        return df, filtered_df, (end_time - start_time),

    @staticmethod
    def merge_with_hash_provenance(df1: pd.DataFrame, df2: pd.DataFrame, key: str, how: str):


    @staticmethod
    def merge_with_ids_provenance(df1: pd.DataFrame, df2: pd.DataFrame, key: str, how: str):

