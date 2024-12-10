import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Data pelatihan (masukkan data dari tabel)
data_train = [
    [57.4, 60.2, 34.1, 2.8, 13.4, 110.4, 56.6, 110.3, 0.00, 17.5, 22.8, 0.4, 66.0, 2.6, 17.3, 4.6, 21.3, 3.8, 0.4, 0.5, 3.0, 3.2, 30.5, 3.10, 0.2, 1.9, 0.4, 0.1, 0.0, 2.0],
    [65.2, 65.9, 22.8, 2.8, 13.4, 120.8, 57.0, 115.5, 0.0, 17.7, 18.9, 0.5, 1.9, 0.4, 0.3, 3.30, 2.9, 31.9, 2.7, 0.2, 3.2, 0.4, 0.1, 0.1, 2.3 ],
    [],
    [0.3927, 0.3953, 0.1765, 0.1641, 0.1000, 0.5573, 0.1872, 0.3667, 0.3112, 0.2078, 0.1111, 0.1121, 0.1211],
    [0.3906, 0.3952, 0.1727, 0.1663, 0.1123, 0.1000, 0.1820, 0.3722, 0.2781, 0.2113, 0.1112, 0.1123, 0.1269],
    [0.4212, 0.3942, 0.4105, 0.1727, 0.1995, 0.2099, 0.1800, 0.4236, 0.3956, 0.2272, 0.1693, 0.2125, 0.1256]]
     
df_train = {
    "X1": data_train[0],
    "X2": data_train[1],
    "X3": data_train[2],
    "X4": data_train[3],
    "X5": data_train[4],
    "target": data_train[5]
}
pd.DataFrame(data_train)

# Normalisasi data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train.drop("target", axis=1))
y_train = df_train["target"].values

# Membuat model neural network
model = Sequential([
    Dense(5, activation='sigmoid', input_dim=X_train.shape[1]),
    Dense(2, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Latih model
history = model.fit(X_train, y_train, epochs=489, verbose=0)

# Visualisasikan Mean Squared Error
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.title("Best Training Performance")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (mse)")

# Set nilai pada sumbu Y
plt.yscale('log')  # Mengatur skala logaritmik pada sumbu Y
plt.yticks([1, 0.1, 0.01, 0.001, 0.0001], labels=["1", "0.1", "0.01", "0.001", "0.0001"])

# Menampilkan legenda dan plot
plt.legend()
plt.show()