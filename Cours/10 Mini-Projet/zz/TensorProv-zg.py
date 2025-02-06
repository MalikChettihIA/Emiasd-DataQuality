import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt

class TensorProv:
    def __init__(self, method='record_id'):
        self.method = method
        self.record_id_counter = 0

    def _generate_hash(self, row, max_value):
        """Génère un hash unique pour une ligne et le normalise."""
        row_str = ','.join(map(str, row))
        return abs(hash(row_str)) % max_value

    def _add_record_ids(self, df):
        """Ajoute un identifiant ou un hash selon la méthode choisie."""
        df = df.copy()  # Copie pour éviter des modifications accidentelles
        if self.method == 'record_id':
            if '_record_id' not in df.columns:
                df['_record_id'] = range(self.record_id_counter, self.record_id_counter + len(df))
                self.record_id_counter += len(df)
                print("Colonne '_record_id' ajoutée avec succès.")
        elif self.method == 'hashing':
            if '_hash_id' not in df.columns:
                df['_hash_id'] = df.apply(lambda row: self._generate_hash(row, len(df)), axis=1)
                print("Colonne '_hash_id' ajoutée avec succès.")
        return df

    def horizontal_data_reduction(self, df, condition):
        """Filtre les lignes selon une condition (réduction horizontale)."""
        df = self._add_record_ids(df)
        df_out = df.query(condition).reset_index(drop=True)

        # Début du chronométrage
        start_time = time.time()

        ids = df_out['_record_id'].values if self.method == 'record_id' else df_out['_hash_id'].values
        rows = np.arange(len(df_out))

        # Vérification des indices
        print(f"IDs générés (max: {len(df) - 1}): {ids.max()}")

        tensor = csr_matrix((np.ones(len(rows)), (rows, ids)), shape=(len(df_out), len(df)))

        # Temps écoulé pour constituer le tenseur
        elapsed_time = time.time() - start_time
        print(f"Tenseur constitué en {elapsed_time:.4f} secondes.")

        return df_out.drop(columns=['_record_id', '_hash_id'], errors='ignore'), tensor

    def vertical_data_reduction(self, df, columns_to_keep):
        """Réduit les colonnes selon une liste donnée (réduction verticale)."""
        df = self._add_record_ids(df)
        print(f"Colonnes actuelles après ajout des identifiants : {df.columns.tolist()}")

        id_column = '_record_id' if self.method == 'record_id' else '_hash_id'
        if id_column not in df.columns:
            raise KeyError(f"'{id_column}' n'est pas dans les colonnes du DataFrame.")

        df_out = df[columns_to_keep + [id_column]]

        # Début du chronométrage
        start_time = time.time()

        tensor = csr_matrix((np.ones(len(df)), (np.arange(len(df)), np.arange(len(df)))), shape=(len(df), len(df)))

        # Temps écoulé pour constituer le tenseur
        elapsed_time = time.time() - start_time
        print(f"Tenseur constitué en {elapsed_time:.4f} secondes.")

        return df_out.drop(columns=[id_column], errors='ignore'), tensor

    def horizontal_data_augmentation(self, df, n_samples):
        """Ajoute des exemples synthétiques (augmentation horizontale)."""
        df = self._add_record_ids(df)
        indices = np.random.choice(len(df), size=n_samples, replace=True)
        synthetic_data = df.iloc[indices].copy()
        numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != '_record_id':
                synthetic_data[col] += np.random.normal(0, 0.1, size=len(synthetic_data))
        df_out = pd.concat([df, synthetic_data]).reset_index(drop=True)

        # Début du chronométrage
        start_time = time.time()

        rows = np.arange(len(df_out))
        cols = np.concatenate((np.arange(len(df)), indices))
        tensor = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df_out), len(df)))

        # Temps écoulé pour constituer le tenseur
        elapsed_time = time.time() - start_time
        print(f"Tenseur constitué en {elapsed_time:.4f} secondes.")

        return df_out.drop(columns=['_record_id', '_hash_id'], errors='ignore'), tensor

    def vertical_data_augmentation(self, df, new_columns):
        """Ajoute de nouvelles colonnes au DataFrame (augmentation verticale)."""
        df = self._add_record_ids(df)

        # Ajout des nouvelles colonnes
        for col_name, col_func in new_columns.items():
            df[col_name] = col_func(df)

        # Début du chronométrage
        start_time = time.time()

        tensor = csr_matrix((np.ones(len(df)), (np.arange(len(df)), np.arange(len(df)))), shape=(len(df), len(df)))

        # Temps écoulé pour constituer le tenseur
        elapsed_time = time.time() - start_time
        print(f"Tenseur constitué en {elapsed_time:.4f} secondes.")

        return df.drop(columns=['_record_id', '_hash_id'], errors='ignore'), tensor

    def join(self, df_left, df_right, on, how='inner'):
        """Jointure de deux DataFrames avec suivi de provenance."""
        df_left = self._add_record_ids(df_left)
        df_right = self._add_record_ids(df_right)
        df_out = pd.merge(df_left, df_right, on=on, how=how)

        # Sélectionner les colonnes d'identifiant appropriées
        id_left = '_record_id' if self.method == 'record_id' else '_hash_id'
        id_right = '_record_id' if self.method == 'record_id' else '_hash_id'

        left_ids = df_out[f'{id_left}_x'].values
        right_ids = df_out[f'{id_right}_y'].values

        # Réindexer les IDs pour qu'ils correspondent aux dimensions des DataFrames
        left_ids = np.clip(left_ids, 0, len(df_left) - 1)
        right_ids = np.clip(right_ids, 0, len(df_right) - 1)

        # Début du chronométrage
        start_time = time.time()

        left_tensor = csr_matrix((np.ones(len(left_ids)), (np.arange(len(left_ids)), left_ids)),
                                 shape=(len(df_out), len(df_left)))
        right_tensor = csr_matrix((np.ones(len(right_ids)), (np.arange(len(right_ids)), right_ids)),
                                  shape=(len(df_out), len(df_right)))

        # Temps écoulé pour constituer les tenseurs
        elapsed_time = time.time() - start_time
        print(f"Tenseurs constitués en {elapsed_time:.4f} secondes.")

        return df_out.drop(columns=[f'{id_left}_x', f'{id_right}_y', '_hash_id_x', '_hash_id_y'], errors='ignore'), (left_tensor, right_tensor)



class TensorProvTests:
    def __init__(self, data_path):
        """Classe de test pour TensorProv"""
        self.data_path = data_path
        self.tp = None
        self.customer_data = None

    def load_data(self):
        """Charge les données client."""
        self.customer_data = pd.read_csv(self.data_path)
        print(f"Dataset chargé : {self.customer_data.shape} lignes, {self.customer_data.shape[1]} colonnes.")

    def initialize_tensorprov(self, method):
        """Initialise TensorProv avec la méthode spécifiée."""
        self.tp = TensorProv(method=method)
        print(f"Instance TensorProv initialisée avec la méthode '{method}'.")

    def test_operation(self, method, operation_name, operation_func):
        """Teste une opération et affiche les résultats."""
        print(f"\n=== Test {operation_name} avec la méthode '{method}' ===")
        result = operation_func()
        result_df = result[0]
        result_tensor = result[1]

        print(f"Taille du DataFrame résultant : {result_df.shape}")
        print(result_df.head())

        # Vérification des tenseurs
        if isinstance(result_tensor, tuple):
            print(f"Les tenseurs retournés pour '{operation_name}' :")
            for i, tensor in enumerate(result_tensor):
                print(f"Tenseur {i + 1} (forme : {tensor.shape})")
        else:
            print(f"Tenseur obtenu ({operation_name}) - forme : {result_tensor.shape}")

    def run_all_tests(self):
        """Exécute tous les tests pour les deux méthodes."""
        operations = {
            "Horizontal Data Reduction": lambda: self.tp.horizontal_data_reduction(self.customer_data.copy(), "Age > 50"),
            "Vertical Data Reduction": lambda: self.tp.vertical_data_reduction(self.customer_data.copy(), ["CustomerID", "Age", "City"]),
            "Horizontal Data Augmentation": lambda: self.tp.horizontal_data_augmentation(self.customer_data.copy(), n_samples=10),
            "Vertical Data Augmentation": lambda: self.tp.vertical_data_augmentation(self.customer_data.copy(), {
                "NewFeature": lambda df: np.random.rand(len(df))
            }),
            "Join": lambda: self.tp.join(self.customer_data.copy(),
                                         pd.DataFrame({
                                             "CustomerID": np.random.randint(1, 31, size=10),
                                             "PurchaseAmount": np.random.uniform(10, 100, size=10)
                                         }),
                                         on="CustomerID", how="inner")
        }

        for method in ['record_id', 'hashing']:
            self.initialize_tensorprov(method)
            for operation_name, operation_func in operations.items():
                self.test_operation(method, operation_name, operation_func)

    def compare_execution_times(self):
        """Compare les temps d'exécution des méthodes 'record_id' et 'hashing'."""

        methods = ["record_id", "hashing"]
        operations = {
            "Horizontal Data Reduction": lambda tp, df: tp.horizontal_data_reduction(df.copy(), "Age > 50"),
            "Vertical Data Reduction": lambda tp, df: tp.vertical_data_reduction(df.copy(), ["CustomerID", "Age", "City"]),
            "Horizontal Data Augmentation": lambda tp, df: tp.horizontal_data_augmentation(df.copy(), n_samples=10),
            "Vertical Data Augmentation": lambda tp, df: tp.vertical_data_augmentation(df.copy(), {
                "NewFeature": lambda df: np.random.rand(len(df))
            }),
            "Join": lambda tp, df: tp.join(df.copy(),
                                           pd.DataFrame({
                                               "CustomerID": np.random.randint(1, 31, size=10),
                                               "PurchaseAmount": np.random.uniform(10, 100, size=10)
                                           }),
                                           on="CustomerID", how="inner")
        }

        execution_times = {op: {method: None for method in methods} for op in operations.keys()}

        for method in methods:
            self.initialize_tensorprov(method)
            tp = self.tp
            df = self.customer_data.copy()

            for operation, func in operations.items():
                start_time = time.time()
                func(tp, df)
                execution_times[operation][method] = time.time() - start_time

        # Création du graphique
        labels = list(operations.keys())
        record_id_times = [execution_times[op]["record_id"] for op in labels]
        hashing_times = [execution_times[op]["hashing"] for op in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, record_id_times, width, label='record_id', alpha=0.7)
        ax.bar(x + width / 2, hashing_times, width, label='hashing', alpha=0.7)

        ax.set_xlabel('Opérations')
        ax.set_ylabel('Temps d\'exécution (secondes)')
        ax.set_title('Comparaison des temps d\'exécution entre record_id et hashing')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    # Chemin vers le fichier client
    data_path = "_light.csv"

    # Exécution des tests et de la comparaison des performances
    tester = TensorProvTests(data_path)
    tester.load_data()
    tester.run_all_tests()
    tester.compare_execution_times()