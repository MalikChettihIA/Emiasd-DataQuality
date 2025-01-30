import pandas as pd
import numpy as np
from scipy.sparse import (coo_matrix)
import hashlib
import time

class TensorProv:
    """
    Infer the provenance of data transformations.
    Given the input data frame(s), output data frame and the kind of the operation
    (vertical reduction, horizontal reduction, etc.), construct a tensor that informs
    on the provenance of the data records of the output data frames and how they depend
    on the input data frames.

    """
    def __init__(self, function, method='hash'):

        self.function = function
        self.function_name = self.function.__name__
        self.method = method

        self.prov_type1_list = ['query_func','reduce_func', 'oversampling_func']
        self.prov_type2_list = ['merge_func']
        self.prov_type3_list = ['append_func']
        self.prov_type4_list = ['one_hot_encoder_func']

    def __call__(self, *args, **kwargs):

        if self.function_name in self.prov_type1_list:
            return self.prov_type1(*args, **kwargs)
        elif self.function_name in self.prov_type2_list:
            return self.prov_type2(*args, **kwargs)
        elif self.function_name in self.prov_type3_list:
            return self.prov_type3(*args, **kwargs)
        elif self.function_name in self.prov_type4_list:
            return self.prov_type4(*args, **kwargs)
        else:
            raise ValueError(f"Cannot capture provenance for more than 2 dataframes.")

    @staticmethod
    def hash_row_content(row: pd.Series) -> str:
        row_str = row.to_json(date_format="iso", orient="columns")
        return hashlib.md5(row_str.encode("utf-8")).hexdigest()

    def check_call_args(self, *args, **kwargs):

        if "reduced_columns" in kwargs:
            reduced_columns = kwargs["reduced_columns"]
            if isinstance(reduced_columns, list):
                if self.method == "hash" and "_hash_" not in reduced_columns:
                    reduced_columns.append("_hash_")
                if self.method == "ids" and "_id_" not in reduced_columns:
                    reduced_columns.append("_id_")
        if "hda_x_columns" in kwargs:
            if self.method == "hash":
                indices_args = {'hash_col':'_hash_'}
                kwargs = {k: v for d in (kwargs, indices_args) for k, v in d.items()}
            if self.method == "ids":
                indices_args = {'hash_col': '_id_'}
                kwargs = {k: v for d in (kwargs, indices_args) for k, v in d.items()}

        return args, kwargs


    def prov_type1 (self, *args, **kwargs):
        """
        La capture de provenance de type 1 capture
        """
        if self.method == 'ids':
            return self.ids_prov_type1(*args, **kwargs)
        else:
            return self.hash_prov_type1(*args, **kwargs)

    def hash_prov_type1(self, *args, **kwargs):

        start_time = time.time()
        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df = kwargs.get('df')
        if df is None:
            raise ValueError(f"Call argument 'df' is required.")

        df["_hash_"] = df.apply(TensorProv.hash_row_content, axis=1)
        result_df = self.function(**kwargs).copy()


        n:int = len(df)  # total rows in original df
        m: int = len(result_df)  # total rows in filtered df

        hash_to_position = {
            h: i for i, h in enumerate(df["_hash_"].values)
        }

        row_positions = np.arange(m, dtype=np.int32)
        col_positions = np.empty(m, dtype=np.int32)
        data = np.ones(m, dtype=np.int8)

        # For each row i in result_df, find the original row position via the hash
        for i in range(m):
            row_hash = result_df.iloc[i]["_hash_"]
            if row_hash not in hash_to_position:
                raise ValueError(f"Hash {row_hash} in result_df not found in original_df. "
                                 f"Possible data mismatch or missing row?")
            orig_pos = hash_to_position[row_hash]
            col_positions[i] = orig_pos

        provenance_matrix = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m, n)
        )
        end_time = time.time()

        return result_df, (end_time-start_time), provenance_matrix

    def ids_prov_type1(self, *args, **kwargs):

        start_time = time.time()
        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df = kwargs.get('df')
        if df is None:
            raise ValueError(f"Call argument 'df' is required.")

        df["_id_"] = df.index
        result_df = self.function(**kwargs).copy()


        n = len(df)
        m = len(result_df)

        orig_indices = result_df["_id_"].to_numpy()
        row_positions = np.arange(m)
        col_positions = orig_indices

        # The data array simply holds ones for each match
        data = np.ones(m, dtype=np.int8)

        # Build the sparse COO matrix
        provenance_matrix = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m,n)
        )
        end_time = time.time()

        return result_df, (end_time - start_time), provenance_matrix

    def prov_type2 (self, *args, **kwargs):
        if self.method == 'ids':
            return self.ids_prov_type2(*args, **kwargs)
        else:
            return self.hash_prov_type2(*args, **kwargs)

    def hash_prov_type2(self, *args, **kwargs):
        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df1 = kwargs.get('df1')
        if df1 is None:
            raise ValueError(f"Call argument 'df1' is required.")

        df2 = kwargs.get('df2')
        if df2 is None:
            raise ValueError(f"Call argument 'df2' is required.")

        df1["_hash_1"] = df1.apply(TensorProv.hash_row_content, axis=1)
        df2["_hash_2"] = df2.apply(TensorProv.hash_row_content, axis=1)

        result_df = self.function(**kwargs).copy()

        start_time = time.time()

        n1 = len(df1)
        n2 = len(df2)
        m = len(result_df)

        t = []

        for i in range(m):
            hash1 = result_df.iloc[i].get("_hash_1", None)
            hash2 = result_df.iloc[i].get("_hash_2", None)

            row_positions = []
            col_positions = []
            data = []

            row_idx = df1[df1["_hash_1"] == hash1].index[0]
            row_positions.append(row_idx)
            row_idx = df2[df2["_hash_2"] == hash2].index[0]
            col_positions.append(row_idx)
            data.append(i)

            provenance_matrix = coo_matrix(
                (data, (row_positions, col_positions)),
                shape=(n1, n2)
            )
            t.append(provenance_matrix)

        end_time = time.time()
        elapsed_time = end_time - start_time

        return result_df, elapsed_time, t

    def ids_prov_type2(self, *args, **kwargs):

        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df1 = kwargs.get('df1')
        if df1 is None:
            raise ValueError(f"Call argument 'df1' is required.")

        df2 = kwargs.get('df2')
        if df2 is None:
            raise ValueError(f"Call argument 'df2' is required.")

        df1["_id_1"] = np.arange(len(df1))
        df2["_id_2"] = np.arange(len(df2))

        result_df = self.function(**kwargs).copy()

        start_time = time.time()

        n1 = len(df1)
        n2 = len(df2)
        m = len(result_df)


        t = []
        for i in range(m):

            id1 = result_df.iloc[i].get("_id_1", np.nan)
            id2 = result_df.iloc[i].get("_id_2", np.nan)

            row_positions = []
            row_positions.append(int(id1))
            col_positions = []
            col_positions.append(int(id2))
            data = [i]

            # Create a sparse matrix for the i-th row in T
            provenance_matrix = coo_matrix(
                (data, (row_positions, col_positions)),
                shape=(n1,n2)
            )
            t.append(provenance_matrix)

        end_time = time.time()
        elapsed_time = end_time - start_time

        return result_df, elapsed_time, t

    def prov_type3 (self, *args, **kwargs):
        if self.method == 'ids':
            return self.ids_prov_type3(*args, **kwargs)
        else:
            return self.hash_prov_type3(*args, **kwargs)

    def hash_prov_type3(self, *args, **kwargs):

        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df1 = kwargs.get('df1')
        if df1 is None:
            raise ValueError(f"Call argument 'df1' is required.")

        df2 = kwargs.get('df2')
        if df2 is None:
            raise ValueError(f"Call argument 'df2' is required.")

        df1["_hash_1"] = df1.apply(TensorProv.hash_row_content, axis=1)
        df2["_hash_2"] = df2.apply(TensorProv.hash_row_content, axis=1)

        result_df = self.function(**kwargs).copy()
        t = []

        start_time = time.time()

        # Gestion du cas ou la ligne provient du df1
        n1:int = len(df1)  # total rows in original df
        m: int = len(result_df)  # total rows in filtered df

        hash_to_position1 = {
            h: i for i, h in enumerate(df1["_hash_1"].values)
        }

        row_positions = np.arange(m, dtype=np.int32)
        col_positions = np.empty(m, dtype=np.int32)

        for i in range(m):
            row_hash = result_df.iloc[i]["_hash_1"]
            if pd.isna(row_hash) or row_hash not in hash_to_position1:
                orig_pos = 0  # Valeur par défaut
            else:
                orig_pos = hash_to_position1[row_hash]
            col_positions[i] = orig_pos

        data = TensorProv.prov_type3_data(df1, df2, result_df, True)

        provenance_matrix_df1 = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m, n1)
        )
        t.append(provenance_matrix_df1)

        # Gestion du cas ou la ligne provient du df2
        n2: int = len(df2)  # total rows in original df
        m: int = len(result_df)  # total rows in filtered df

        hash_to_position2 = {
            h: i for i, h in enumerate(df2["_hash_2"].values)
        }

        row_positions = np.arange(m, dtype=np.int32)
        col_positions = np.empty(m, dtype=np.int32)

        for i in range(m):
            row_hash = result_df.iloc[i]["_hash_2"]
            if pd.isna(row_hash) or row_hash not in hash_to_position2 :
                orig_pos = 0  # Valeur par défaut
            else:
                orig_pos = hash_to_position2[row_hash]
            col_positions[i] = orig_pos

        data = TensorProv.prov_type3_data(df1, df2, result_df, False)

        provenance_matrix_df2 = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m, n2)
        )
        t.append(provenance_matrix_df2)

        end_time = time.time()

        return result_df, (end_time-start_time), t

    def ids_prov_type3(self, *args, **kwargs):

        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df1 = kwargs.get('df1')
        if df1 is None:
            raise ValueError(f"Call argument 'df1' is required.")

        df2 = kwargs.get('df2')
        if df2 is None:
            raise ValueError(f"Call argument 'df2' is required.")

        df1["_id_1"] = np.arange(len(df1))
        df2["_id_2"] = np.arange(len(df2))

        result_df = self.function(**kwargs).copy()
        t = []
        start_time = time.time()

        # Gestion du cas ou la ligne provient du df1
        n1 = len(df1)
        m = len(result_df)

        orig_indices1 = result_df["_id_1"].fillna(0).to_numpy().astype(int)
        row_positions = np.arange(m)
        col_positions = orig_indices1

        data = TensorProv.prov_type3_data(df1, df2, result_df, True)

        # Build the sparse COO matrix
        provenance_matrix1 = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m,n1)
        )
        t.append(provenance_matrix1)

        # Gestion du cas ou la ligne provient du df2
        n2 = len(df2)
        m = len(result_df)

        orig_indices2 = result_df["_id_2"].fillna(0).to_numpy().astype(int)
        row_positions = np.arange(m)
        col_positions = orig_indices2

        data = TensorProv.prov_type3_data(df1, df2, result_df, False)

        # Build the sparse COO matrix
        provenance_matrix2 = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m,n2)
        )
        t.append(provenance_matrix2)
        end_time = time.time()

        return result_df, (end_time - start_time), t

    @staticmethod
    def prov_type3_data(df1: pd.DataFrame, df2:pd.DataFrame, result_df:pd.DataFrame, df1_is_first: bool) -> np.ndarray:
        if df1_is_first:
            df1_data = data = np.ones(len(df1), dtype=np.int8)
            df2_data = np.zeros(len(result_df)-len(df1), dtype=np.int8)
            return np.concatenate((df1_data, df2_data))
        else:
            df1_data = data = np.zeros(len(result_df)-len(df2), dtype=np.int8)
            df2_data = np.ones(len(df2), dtype=np.int8)
            return np.concatenate((df1_data, df2_data))

    def prov_type4 (self, *args, **kwargs):
        if self.method == 'ids':
            return self.ids_prov_type4(*args, **kwargs)
        else:
            return self.hash_prov_type4(*args, **kwargs)

    def hash_prov_type4(self, *args, **kwargs):

        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df = kwargs.get('df')
        if df is None:
            raise ValueError(f"Call argument 'df' is required.")

        df["_hash_"] = df.apply(TensorProv.hash_row_content, axis=1)

        result_df = self.function(**kwargs).copy()

        start_time = time.time()

        # Gestion du cas ou la ligne provient du df1
        n1:int = len(df)  # total rows in original df
        m: int = len(result_df)  # total rows in filtered df

        hash_to_position1 = {
            h: i for i, h in enumerate(df["_hash_"].values)
        }

        row_positions = np.arange(m, dtype=np.int32)
        col_positions = np.empty(m, dtype=np.int32)

        for i in range(m):
            row_hash = result_df.iloc[i]["_hash_"]
            if pd.isna(row_hash) or row_hash not in hash_to_position1:
                orig_pos = 0  # Valeur par défaut
            else:
                orig_pos = hash_to_position1[row_hash]
            col_positions[i] = orig_pos

        data = np.ones(m, dtype=np.int8)

        provenance_matrix_df = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m, m)
        )

        end_time = time.time()

        return result_df, (end_time-start_time), provenance_matrix_df

    def ids_prov_type4(self, *args, **kwargs):

        args, kwargs = TensorProv.check_call_args(self, *args, **kwargs)

        df = kwargs.get('df')
        if df is None:
            raise ValueError(f"Call argument 'df' is required.")

        df["_id_"] = np.arange(len(df))

        result_df = self.function(**kwargs).copy()
        start_time = time.time()

        # Gestion du cas ou la ligne provient du df1
        n1 = len(df)
        m = len(result_df)

        orig_indices1 = result_df["_id_"].fillna(0).to_numpy().astype(int)
        row_positions = np.arange(m)
        col_positions = orig_indices1

        data = np.ones(m, dtype=np.int8)

        # Build the sparse COO matrix
        provenance_matrix_df = coo_matrix(
            (data, (row_positions, col_positions)),
            shape=(m,m)
        )
        end_time = time.time()

        return result_df, (end_time - start_time), provenance_matrix_df