import subprocess
import sys

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

import subprocess
import sys

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

def load_data(stock_file):
    df = pd.read_csv(stock_file)
    return df

def preprocess_data(df):
    # Calculate daily returns
    df['Daily_Return'] = (df['Close'] / df['Close'].shift(1)) - 1
    df = df.dropna()  # Remove the first row with NaN values
    # Fill null values with the previous non-null value for 'Close' and 'Daily_Return' columns
    df['Close'].fillna(method='bfill', inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)
    df['Daily_Return'].fillna(method='bfill', inplace=True)
    df['Daily_Return'].fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Create sequences of data for training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_model(seq_length):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))  # Add dropout layer
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))  # Add dropout layer
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))  # Add dropout layer
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)
    return history

# Predict future stock prices
def predict_future(model, data, scaler, seq_length, future_steps):
    predictions = []

    # Take the last sequence from the data
    current_sequence = data[-seq_length:].reshape(1, seq_length, 1)

    for i in range(future_steps):
        next_pred = model.predict(current_sequence)[0]
        predictions.append(next_pred)
        current_sequence = np.append(current_sequence[:,1:,:], [[next_pred]], axis=1)

    predictions = scaler.inverse_transform(predictions)
    return predictions

# Plot predictions
def plot_predictions(df, predictions, future_steps):
    plt.plot(df['Date'], df['Close'], label='Actual Stock Price')
    plt.plot(np.arange(len(df)-1,len(df)-1+future_steps), predictions, label='Predicted Stock Price', color='r')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Plot training vs test loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Test Loss')
    plt.legend()
    plt.show()

# Main function
def main():
    # Load and preprocess the data
    df = load_data("NVDA.csv")
    data, scaler = preprocess_data(df)

    # Define hyperparameters
    seq_length = 145
    epochs = 7800
    batch_size = 128
    future_steps = 30

    # Create sequences for training
    X, y = create_sequences(data, seq_length)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build the model
    model = build_model(seq_length)

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)

    # Plot training vs test loss
    plot_loss(history)

    # Make predictions
    predictions = predict_future(model, data, scaler, seq_length, future_steps)

    # Plot predictions
    plot_predictions(df, predictions, future_steps)

if __name__ == "__main__":
    main()