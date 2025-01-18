import unittest
import pandas as pd
import numpy as np
import random
import string
import TensorProv


class MergeTestCase(unittest.TestCase):
    MERGE_KEY = "name"
    MERGE_HOW = "inner"
    DEBUG = False

    def setUp(self):
        self.personDataFrame = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Dave"],
            "age":  [25, 32, 45, 23],
            "city": ["NY", "SF", "LA", "NY"]
        })
        self.hobbiesDataFrame = pd.DataFrame({
            "name": ["Alice", "Alice", "Bob", "Charlie", "Charlie", "Dave"],
            "hobbies": ["Reading", "Painting", "Cycling", "Cooking", "Gardening", "Hiking"]
        })


    def test_hash_merge(self):
        tensor_filter = TensorProv.TensorProv('merge','hash')
        df1, df2, merged_df, execution_time, provenance_matrix = tensor_filter(self.personDataFrame, self.hobbiesDataFrame, self.MERGE_KEY, self.MERGE_HOW)
        print('\n.......................................................')
        print('\033[31m.test_hash_merge Done ...\033[0m')
        print("Original DataFrame:\n", df1)
        print("Original DataFrame:\n", df2)
        print("Filtered DataFrame:\n", merged_df)
        print("\033[32m[Execution Time:", execution_time, "\033[0m")
        print("Provenance Matrix (COO format):")
        print("Shape:", provenance_matrix.shape)
        print("Row indices (filtered_df):", provenance_matrix.row)
        print("Column indices (original_df):", provenance_matrix.col)
        print("Data:", provenance_matrix.data)


    def test_ids_merge(self):
        tensor_filter = TensorProv.TensorProv('merge','ids')
        df1, df2, merged_df, execution_time, provenance_matrix = tensor_filter(self.personDataFrame, self.hobbiesDataFrame, self.MERGE_KEY, self.MERGE_HOW)
        print('\n.......................................................')
        print('\033[31m.test_hash_merge Done ...\033[0m')
        print("Original DataFrame:\n", df1)
        print("Original DataFrame:\n", df2)
        print("Filtered DataFrame:\n", merged_df)
        print("\033[32m[Execution Time:", execution_time, "\033[0m")
        print("Provenance Matrix (COO format):")
        print("Shape:", provenance_matrix.shape)
        print("Row indices (filtered_df):", provenance_matrix.row)
        print("Column indices (original_df):", provenance_matrix.col)
        print("Data:", provenance_matrix.data)

if __name__ == '__main__':
    unittest.main()
