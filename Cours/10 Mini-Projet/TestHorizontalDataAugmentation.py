from TensorProvUtils import *

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def hda_function(df: pd.DataFrame, feature_cols: list, target_col: str, hash_col: str = "_hash_"):
    """
    Applique SMOTE et retourne un DataFrame avec la provenance des lignes.
    """
    # Vérifier la présence de la colonne hash
    if hash_col not in df.columns:
        raise ValueError(f"La colonne '{hash_col}' est manquante.")

    # Séparer les features et la target
    X = df[feature_cols].values  # Convertir en tableau NumPy
    y = df[target_col]
    hash_data = df[hash_col]

    # Appliquer SMOTE
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Identifier les indices synthétiques
    n_original = len(X)
    synthetic_indices = np.arange(n_original, len(X_resampled))

    # Entraîner un modèle NearestNeighbors sur les données d'origine
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X)  # Entraînement sur les données originales

    # Trouver les voisins des instances synthétiques (en utilisant X_resampled comme tableau NumPy)
    _, origin_indices = nn.kneighbors(X_resampled[synthetic_indices])

    # Créer le DataFrame de sortie
    df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
    df_resampled[target_col] = y_resampled

    # Ajouter la colonne de provenance
    provenance = []
    for i in range(len(df_resampled)):
        if i < n_original:
            provenance.append([hash_data.iloc[i]])
        else:
            idx = origin_indices[i - n_original][0]
            provenance.append([hash_data.iloc[idx]])

    df_resampled[hash_col] = provenance

    return df_resampled

if __name__ == '__main__':
    function_args = {
        "feature_cols": ['age'],
        "target_col": 'city'
    }
    execute_and_log(df_counts, hda_function, function_args)
    exit(0)