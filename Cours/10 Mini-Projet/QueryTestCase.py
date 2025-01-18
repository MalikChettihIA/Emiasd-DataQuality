import unittest
import pandas as pd
import numpy as np
import random
import string
import TensorProv


class QueryTestCase(unittest.TestCase):
    FILTER_CONDITION = "city == 'NY' and age > 20"
    DEBUG = True

    def setUp(self):
        self.personDataFrame = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Dave"],
            "age":  [25, 32, 45, 23],
            "city": ["NY", "SF", "LA", "NY"]
        })

    def test_hash_query(self):
        tensor_query = TensorProv.TensorProv('query','hash')
        df, filtered_df, execution_time, provenance_matrix = tensor_query(self.personDataFrame, self.FILTER_CONDITION)
        if self.DEBUG :
            print('\n.......................................................')
            print('\033[31m.test_hash_filter Done ...\033[0m')
            print("Original DataFrame:\n", df)
            print("Filtered DataFrame:\n", filtered_df)
            print("\033[32mExecution Time:", execution_time, "\033[0m")
            print("Provenance Matrix (COO format):")
            print("Shape:", provenance_matrix.shape)
            print("Row indices (filtered_df):", provenance_matrix.row)
            print("Column indices (original_df):", provenance_matrix.col)
            print("Data Provenance:\n", provenance_matrix)
            print('........................................................')

    def test_ids_query(self):
        tensor_query = TensorProv.TensorProv('query','ids')
        df, filtered_df, execution_time, provenance_matrix = tensor_query(self.personDataFrame, self.FILTER_CONDITION)
        if self.DEBUG:
            print('\n........................................................')
            print('\033[31m.test_ids_filter Done ...\033[0m')
            print("Original DataFrame:\n", df)
            print("Filtered DataFrame:\n", filtered_df)
            print("\033[32mExecution Time:", execution_time, "\033[0m")
            print("Provenance Matrix (COO format):")
            print("Shape:", provenance_matrix.shape)
            print("Row indices (filtered_df):", provenance_matrix.row)
            print("Column indices (original_df):", provenance_matrix.col)
            print("Data Provenance:\n", provenance_matrix)
            print('........................................................')

if __name__ == '__main__':
    unittest.main()
