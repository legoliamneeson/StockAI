import subprocess
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

# Function to verify GPU availability
def verify_gpu_availability():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("GPUs detected:")
        for gpu in gpus:
            print("  -", gpu)
    else:
        print("No GPUs detected.")

# Call the function to verify GPU availability
verify_gpu_availability()

# Function to configure TensorFlow session for GPU
def configure_gpu_memory():
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to allocate only as much GPU memory as needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth is configured successfully!")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs available, running on CPU.")

# Call the function to configure GPU memory
configure_gpu_memory()

# Load the stock data
def load_data(stock_file):
    df = pd.read_csv(stock_file)
    return df

def preprocess_data(df):
    # Fill null values with the previous non-null value
    df['Close'].fillna(method='ffill', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
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
def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

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

# Main function
def main():
    # Load and preprocess the data
    df = load_data("C:/Users/Cole/Downloads/NVDA.csv")
    data, scaler = preprocess_data(df)

    # Define hyperparameters
    seq_length = 245
    future_steps = 30
    epochs = 10000  # Increase the number of epochs
    batch_size = 128

    # Create sequences for training
    X, y = create_sequences(data, seq_length)

    # Split data into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train the model
    model = build_model(seq_length)
    train_model(model, X_train, y_train, epochs, batch_size)

    # Make predictions
    predictions = predict_future(model, data, scaler, seq_length, future_steps)

    # Plot predictions
    plt.plot(df['Date'], df['Close'], label='Actual Stock Price')
    plt.plot(np.arange(len(df)-1,len(df)-1+future_steps), predictions, label='Predicted Stock Price', color='r')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
