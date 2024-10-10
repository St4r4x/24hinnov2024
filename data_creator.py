import glob

import pandas as pd

# Chemin vers le dossier contenant les fichiers CSV
path = "./data_brut/*.csv"

# Liste pour stocker les données
data = []

# Parcourir tous les fichiers CSV dans le dossier
for csv_file in glob.glob(path):
    # Extraire l'année depuis le nom du fichier
    year = csv_file.split('_')[2].split('.')[0]

    # Lire le fichier CSV
    df = pd.read_csv(csv_file)

    # Ajouter la colonne "Année"
    df['Année'] = year

    # Renommer les colonnes pour correspondre au format souhaité
    df.columns = ['Semaine', 'Volume (MWh)', 'Année']

    # Ajouter les données à la liste
    data.append(df)

# Concatenation de toutes les données
final_df = pd.concat(data)

# Sauvegarder les données concaténées dans un fichier CSV
final_df.to_csv('data_concatenee.csv', index=False)
