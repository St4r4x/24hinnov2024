import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
import joblib

# Fonction pour générer des données simulées
def generer_donnees(N):
    taux_remplissage = np.sin(np.linspace(0, 10*np.pi, N)) * 50 + 50
    pluie = np.random.uniform(0, 100, N)
    temperature = np.random.uniform(10, 30, N)
    return taux_remplissage, pluie, temperature

# Fonction pour créer des séquences
def creer_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(np.concatenate(([data['remplissage'].iloc[i]], data.iloc[i+1:i+sequence_length+1][['pluie', 'temperature']].values.flatten())))
        y.append(data['remplissage'].iloc[i+1:i+sequence_length+1].values)
    return np.array(X), np.array(y)

# Générer et préparer les données
N = 1000
sequence_length = 48
taux_remplissage, pluie, temperature = generer_donnees(N)
data = pd.DataFrame({
    'remplissage': taux_remplissage,
    'pluie': pluie,
    'temperature': temperature
})

# Normaliser les données
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

X, y = creer_sequences(data_scaled, sequence_length)

# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape pour LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Construire le modèle
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    RepeatVector(sequence_length),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(1))
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Sauvegarder le modèle et le scaler
model.save('model_lstm.h5')
joblib.dump(scaler, 'scaler.pkl')

# Afficher l'historique d'entraînement
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.title('Historique d\'entraînement du modèle')
plt.ylabel('Perte')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print("Modèle entraîné et sauvegardé avec succès.")
