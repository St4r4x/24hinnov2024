import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Vérifier si CUDA est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger et préparer les données
data = pd.read_csv('data_concatenee.csv')
data['Date'] = pd.to_datetime(data['Année'].astype(
    str) + '-W' + data['Semaine'].astype(str) + '-1', format='%Y-W%W-%w')
data = data.sort_values('Date')

# Normaliser les données
scaler = MinMaxScaler()
data['Volume_scaled'] = scaler.fit_transform(data[['Volume (MWh)']])

# Fonction pour créer des séquences


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)]['Volume_scaled'].values
        y = data.iloc[i+seq_length]['Volume_scaled']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Paramètres
sequence_length = 52  # Une année de données hebdomadaires
X, y = create_sequences(data, sequence_length)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# Convertir en tenseurs PyTorch et déplacer vers le device
X_train = torch.FloatTensor(X_train).unsqueeze(2).to(device)
X_test = torch.FloatTensor(X_test).unsqueeze(2).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Définir le modèle LSTM


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Initialiser le modèle, la fonction de perte et l'optimiseur
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Entraîner le modèle
num_epochs = 100
batch_size = 32
train_loader = DataLoader(list(zip(X_train, y_train)),
                          batch_size=batch_size, shuffle=True)

losses = []
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

    # Évaluer sur l'ensemble de test
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs.squeeze(), y_test)

    losses.append(test_loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {test_loss.item():.4f}')

# Sauvegarder le modèle
torch.save(model.state_dict(), 'model_lstm_pytorch.pth')

# Tracer la courbe de perte
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Courbe de perte du modèle')
plt.xlabel('Époques')
plt.ylabel('Perte (MSE)')
plt.show()

print("Modèle entraîné et sauvegardé avec succès.")
