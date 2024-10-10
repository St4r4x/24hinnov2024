import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Charger le modèle et le scaler
model = load_model('model_lstm.h5')
scaler = joblib.load('scaler.pkl')

def preparer_donnees_inference(remplissage_actuel, pluie_future, temperature_future):
    assert len(pluie_future) == 48 and len(temperature_future) == 48, "Les prévisions météo doivent couvrir 48 heures"
    
    data = pd.DataFrame({
        'remplissage': [remplissage_actuel] + [0] * 48,
        'pluie': [0] + pluie_future,
        'temperature': [0] + temperature_future
    })
    
    data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns)
    
    X = np.concatenate(([data_scaled['remplissage'].iloc[0]], data_scaled.iloc[1:][['pluie', 'temperature']].values.flatten()))
    X = X.reshape(1, 1, -1)
    
    return X

def predire_remplissage_horaire(remplissage_actuel, pluie_future, temperature_future):
    X = preparer_donnees_inference(remplissage_actuel, pluie_future, temperature_future)
    prediction_scaled = model.predict(X)
    
    # Inverser la normalisation
    prediction = scaler.inverse_transform(np.concatenate((np.zeros((prediction_scaled.shape[0], 1)), prediction_scaled.reshape(-1, 48)), axis=1))[:, 1:]
    
    return prediction[0]  # Retourner les 48 valeurs horaires

# Exemple d'utilisation
remplissage_actuel = 60  # Taux de remplissage actuel (%)
pluie_future = np.random.uniform(0, 100, 48)  # Prévisions de pluie pour les 48 prochaines heures
temperature_future = np.random.uniform(10, 30, 48)  # Prévisions de température pour les 48 prochaines heures

prediction_horaire = predire_remplissage_horaire(remplissage_actuel, pluie_future, temperature_future)

# Afficher les résultats
plt.figure(figsize=(12, 6))
plt.plot(range(48), prediction_horaire, label='Taux de remplissage prédit', marker='o')
plt.axhline(y=remplissage_actuel, color='r', linestyle='--', label='Taux de remplissage actuel')
plt.title("Prédiction horaire des fluctuations du taux de remplissage sur 48h")
plt.xlabel("Heures")
plt.ylabel("Taux de remplissage (%)")
plt.legend()
plt.grid(True)
plt.xticks(range(0, 49, 6))  # Afficher les heures par intervalles de 6
plt.tight_layout()
plt.show()

# Afficher les prédictions heure par heure
print(f"Taux de remplissage actuel : {remplissage_actuel}%")
print("\nPrédictions heure par heure :")
for heure, prediction in enumerate(prediction_horaire, 1):
    print(f"Heure {heure:2d} : {prediction:.2f}%")

# Calculer et afficher quelques statistiques
print(f"\nTaux de remplissage minimum prédit : {np.min(prediction_horaire):.2f}%")
print(f"Taux de remplissage maximum prédit : {np.max(prediction_horaire):.2f}%")
print(f"Taux de remplissage moyen prédit : {np.mean(prediction_horaire):.2f}%")
