import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Function to install pip if not already installed
def install_pip():
    try:
        import pip
        print("pip is already installed.")
    except ImportError:
        print("pip is not installed. Installing pip...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])
        print("pip installation completed.")

# Function to install required packages using pip
def install_requirements(requirements_file):
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]  # Remove empty lines and leading/trailing whitespaces
    
    print("Installing required packages...")
    for req in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    print("Installation completed.")

# Install pip if not already installed
install_pip()

# Install required packages from requirements.txt
install_requirements('requirements.txt')

def load_data(stock_file):
    df = pd.read_csv(stock_file)
    return df

def preprocess_data(df):
    df['Daily_Return'] = (df['Close'] / df['Close'].shift(1)) - 1
    df = df.dropna()  
    df['Close'].fillna(method='bfill', inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)
    df['Daily_Return'].fillna(method='bfill', inplace=True)
    df['Daily_Return'].fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_model(seq_length):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2)
    return history

def predict_future(model, data, scaler, seq_length, future_steps):
    predictions = []
    current_sequence = data[-seq_length:].reshape(1, seq_length, 1)

    for i in range(future_steps):
        next_pred = model.predict(current_sequence)[0]
        predictions.append(next_pred)
        current_sequence = np.append(current_sequence[:,1:,:], [[next_pred]], axis=1)

    predictions = scaler.inverse_transform(predictions)
    return predictions

def plot_predictions(df, predictions, future_steps):
    plt.plot(df['Date'], df['Close'], label='Actual Stock Price')
    plt.plot(np.arange(len(df)-1,len(df)-1+future_steps), predictions, label='Predicted Stock Price', color='r')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()

# Load or build model
def load_or_build_model(seq_length, filename):
    if os.path.exists(filename):
        return load_model(filename)
    else:
        model = build_model(seq_length)
        return model

def main():
    df = load_data("NVDA.csv")
    data, scaler = preprocess_data(df)

    seq_length = 145
    epochs = 150
    batch_size = 128
    future_steps = 30

    X, y = create_sequences(data, seq_length)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = load_or_build_model(seq_length, "pretrained_lstm_model.h5")

    history = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)  # Provide epochs and batch_size arguments here

    # Save model
    model.save("pretrained_lstm_model.h5")

    plot_loss(history)

    predictions = predict_future(model, data, scaler, seq_length, future_steps)

    plot_predictions(df, predictions, future_steps)

if __name__ == "__main__":
    main()
