import pandas as pd

# Chemin vers le fichier Excel
csv_file = "./data_brut/StockHydraulique_2015.csv"

# Lire le fichier Excel avec le moteur approprié
df = pd.read_csv(csv_file, sep=';', encoding='utf-8')

# Afficher le contenu du DataFrame
print(df)
