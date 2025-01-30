import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import os
import copy
from TensorProv import TensorProv

######### ---------------------------------------------------------------------------------------------------- #########
######### Méthodes utilitaires pour la génération de datafarme de tests                                        #########
######### ---------------------------------------------------------------------------------------------------- #########

# Fixer la graine pour les générateurs de nombres aléatoires
random.seed(42)  # Pour le module random
np.random.seed(42)  # Pour numpy

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_large_df(n_persons=1000):

    # Generate the person DataFrame
    persons_df = pd.DataFrame({
        "name": [random_string() for _ in range(n_persons)],
        "age": np.random.randint(18, 70, size=n_persons),
        "city": np.random.choice(["NY", "SF", "LA", "Berlin", "London"], size=n_persons)
    })

    hobbies_list = []
    for name in persons_df["name"]:
        n_hobbies = random.randint(0, 3)  # Randomly choose between 0 and 3 hobbies
        hobbies = np.random.choice(
            ["Reading", "Painting", "Cycling", "Cooking", "Gardening", "Hiking"],
            size=n_hobbies,
            replace=False
        )
        for hobby in hobbies:
            hobbies_list.append({"name": name, "hobbies": hobby})
    hobbies_df = pd.DataFrame(hobbies_list)

    return persons_df, hobbies_df

######### ---------------------------------------------------------------------------------------------------- #########
######### Méthode utilitaire permettant l'exécution des tests et l'affichage d'un tableau récapitulatif        #########
######### ---------------------------------------------------------------------------------------------------- #########

def generate_and_save_datasets(df_counts: list, base_path="datasets"):
    """
    Génère des datasets et les stocke en fichiers CSV.
    """
    os.makedirs(base_path, exist_ok=True)
    for count in df_counts:
        persons_df, hobbies_df = generate_large_df(n_persons=count)

        persons_path = os.path.join(base_path, f"persons_{count}.csv")
        hobbies_path = os.path.join(base_path, f"hobbies_{count}.csv")

        persons_df.to_csv(persons_path, index=False)
        hobbies_df.to_csv(hobbies_path, index=False)
        print(f"Datasets sauvegardés : {persons_path}, {hobbies_path}")


def execute_and_log(df_counts: list, function, function_args, nb_df: int = 1, base_path="datasets"):
    """
    Charge les datasets depuis les fichiers CSV et exécute les tests.
    """
    test_logs = []
    first_iteration_results_is_to_store = True
    first_iteration_results = []

    for count in df_counts:
        # Charger les datasets depuis les fichiers CSV
        persons_path = os.path.join(base_path, f"persons_{count}.csv")
        hobbies_path = os.path.join(base_path, f"hobbies_{count}.csv")

        persons_df = pd.read_csv(persons_path)
        hobbies_df = pd.read_csv(hobbies_path) if nb_df == 2 else None

        print(f"Running {function.__name__} tests with {count} rows")

        df_args = {"df": persons_df} if nb_df == 1 else {"df1": persons_df, "df2": hobbies_df}
        function_args_copy = copy.deepcopy(function_args)
        all_args = {k: v for d in (df_args, function_args_copy) for k, v in d.items()}

        # Exécution avec la méthode Hash
        hash_tensor_filter = TensorProv(function=function, method='hash')
        hash_result_df, execution_time, hash_provenance_matrix_list = hash_tensor_filter(**all_args)

        test_logs.append({
            "Test Name": f"Test {function.__name__} with {count} rows",
            "Capture Method": "Hash",
            "Original (Person) Rows": len(persons_df),
            "Original (Hobbies) Rows": len(hobbies_df) if nb_df == 2 else "N/A",
            "Transformation Function": function.__name__,
            "Result Rows": len(hash_result_df),
            "Execution Time (s)": execution_time
        })

        # Exécution avec la méthode Ids
        ids_tensor_filter = TensorProv(function=function, method='ids')
        ids_result_df, execution_time, ids_provenance_matrix_list = ids_tensor_filter(**all_args)

        test_logs.append({
            "Test Name": f"Test {function.__name__} with {count} rows",
            "Capture Method": "Ids",
            "Original (Person) Rows": len(persons_df),
            "Original (Hobbies) Rows": len(hobbies_df) if nb_df == 2 else "N/A",
            "Transformation Function": function.__name__,
            "Result Rows": len(ids_result_df),
            "Execution Time (s)": execution_time
        })

        if first_iteration_results_is_to_store:
            first_iteration_results_is_to_store = False
            first_iteration_results.append(persons_df)
            if nb_df == 2:
                first_iteration_results.append(hobbies_df)
            first_iteration_results.append(hash_result_df)
            first_iteration_results.append(hash_provenance_matrix_list)
            first_iteration_results.append(ids_result_df)
            first_iteration_results.append(ids_provenance_matrix_list)

    print("\nFirst Iteration Details:")
    if len(first_iteration_results) == 5:
        print("\nInput Person dataframe is :")
        print(first_iteration_results[0])
        print("\nTransformation Result dataframe is :")
        print(first_iteration_results[1])

        print("\nHash Capture Result :")
        hash_cco_matrix = first_iteration_results[2]
        print("  Row Indices:", hash_cco_matrix.row)
        print("  Col Indices:", hash_cco_matrix.col)
        print("  Data:", hash_cco_matrix.data)
        print("  Shape:", hash_cco_matrix.shape)
        print("  Matrix:")
        print(hash_cco_matrix.toarray())

        print("\nIds Capture Result :")
        ids_cco_matrix = first_iteration_results[4]
        print("  Row Indices:", ids_cco_matrix.row)
        print("  Col Indices:", ids_cco_matrix.col)
        print("  Data:", ids_cco_matrix.data)
        print("  Shape:", ids_cco_matrix.shape)
        print("  Matrix:")
        print(ids_cco_matrix.toarray())
    else:
        print("\nInput Person dataframe is :")
        print(first_iteration_results[0])
        print("\nInput Hobbies dataframe is :")
        print(first_iteration_results[1])
        print("\nTransformation Result dataframe is :")
        print(first_iteration_results[2])

        print("\nHash Capture Result :")
        for iteration_count, hash_cco_matrix in enumerate(first_iteration_results[3]):
            print(f"  Matrice {iteration_count + 1}")

            print("    Row Indices:", hash_cco_matrix.row)
            print("    Col Indices:", hash_cco_matrix.col)
            print("    Data:", hash_cco_matrix.data)
            print("    Shape:", hash_cco_matrix.shape)
            print("    Matrix:")
            print(hash_cco_matrix.toarray())

        print("\nIds Capture Result :")
        for iteration_count, ids_cco_matrix in enumerate(first_iteration_results[5]):
            print(f"  Matrice {iteration_count + 1}")

            print("    Row Indices:", ids_cco_matrix.row)
            print("    Col Indices:", ids_cco_matrix.col)
            print("    Data:", ids_cco_matrix.data)
            print("    Shape:", ids_cco_matrix.shape)
            print("    Matrix:")
            print(ids_cco_matrix.toarray())

    print("\nTest Summary:")
    df_logs = pd.DataFrame(test_logs)
    print(df_logs.to_string(index=False))
    plot_results(df_counts, function, df_logs)

def plot_results(df_counts: list, function, df_logs: pd.DataFrame):

    # Extraire les données pour le graphique
    hash_times = df_logs[df_logs["Capture Method"] == "Hash"]["Execution Time (s)"]
    ids_times = df_logs[df_logs["Capture Method"] == "Ids"]["Execution Time (s)"]

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(df_counts, hash_times, label="Hash Method", marker="o")
    plt.plot(df_counts, ids_times, label="Ids Method", marker="o")

    # Ajouter des labels et un titre
    plt.xlabel("Number of Rows")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Execution Time Comparison: Hash vs Ids Methods for {function.__name__}")
    #plt.xscale("log")  # Échelle logarithmique pour l'axe des x
    #plt.yscale("log")  # Échelle logarithmique pour l'axe des y
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Afficher le graphique
    plt.show()