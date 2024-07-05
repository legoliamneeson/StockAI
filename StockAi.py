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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

def load_data(stock_file):
    df = pd.read_csv(stock_file)
    return df

def preprocess_data(df):
    # Interpolate missing values
    df.interpolate(method='linear', inplace=True)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)
    
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_model(seq_length):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, callbacks):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2, callbacks=callbacks)
    return history

def predict_future(model, data, scaler, seq_length, future_steps):
    predictions = []

    # Adding last x and y point of the stock history
    last_x_point = data[-seq_length:]
    last_y_point = data[-1]

    for i in range(future_steps):
        if i == 0:
            current_sequence = last_x_point.reshape(1, seq_length, 1)
        else:
            current_sequence = np.append(current_sequence[:, 1:, :], [[next_pred]], axis=1)

        next_pred = model.predict(current_sequence)[0]
        predictions.append(next_pred)

    predictions = np.insert(predictions, 0, last_y_point)  # Inserting the last y point at the start
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return predictions

def plot_predictions(df, predictions, future_steps,predictions2,future_steps2,predictions3,future_steps3,predictions4,future_steps4):
    plt.plot(df['Date'], df['Close'], label='Actual Stock Price')
    plt.plot(np.arange(len(df)-1,len(df)-1+future_steps4+1), predictions4, label='Predicted Stock Price4', color='c')
    plt.plot(np.arange(len(df)-1,len(df)-1+future_steps3+1), predictions3, label='Predicted Stock Price3', color='m')
    plt.plot(np.arange(len(df)-1,len(df)-1+future_steps2+1), predictions2, label='Predicted Stock Price2', color='g')
    plt.plot(np.arange(len(df)-1,len(df)-1+future_steps+1), predictions, label='Predicted Stock Price', color='r')
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
    df = load_data("SPX.csv")
    trainName = "SPXtrain.h5"
    data, scaler = preprocess_data(df)
    seq_length = 50
    epochs = 1000
    batch_size = 256
    future_steps = 1
    future_steps2 = 5
    future_steps3 = 10
    future_steps4 = 15
    X, y = create_sequences(data, seq_length)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Recreate the model instance
    model = build_model(seq_length)
    
    # Define early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, callbacks=[early_stopping])

    # Save model
    model.save(trainName)

    plot_loss(history)

    predictions = predict_future(model, data, scaler, seq_length, future_steps)
    predictions2 = predict_future(model, data, scaler, seq_length, future_steps2)
    predictions3 = predict_future(model, data, scaler, seq_length, future_steps3)
    predictions4 = predict_future(model, data, scaler, seq_length, future_steps4)
    plot_predictions(df, predictions, future_steps,predictions2,future_steps2,predictions3,future_steps3,predictions4,future_steps4)
    # Calculate MSE on validation data
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    print("Validation MSE:", mse)

if __name__ == "__main__":
    main()
