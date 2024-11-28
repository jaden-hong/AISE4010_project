from data_load import *

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# LSTM model definition
def create_lstm_model(input_shape,lr):
    model = models.Sequential()
    model.add(layers.LSTM(128, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(1))  # Change based on your output

    optimizer = Adam(learning_rate=lr) #setting 

    model.compile(optimizer=optimizer, metrics = ['mae'], loss='mse')  #use mae and mse since regression
    return model

def getXandY(type="bus",start_year=2014,end_year=2015):
    # type = "bus"
    # Get the directory of the current script (for relative path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # File path relative to the script's folder
    file_path = os.path.join(script_dir, 'ttc_{0}_data-{1}-{2}.pkl'.format(type,start_year,end_year))

    if os.path.exists(file_path):
        # If file exists, load the data
        with open(file_path, 'rb') as file:
            df = pickle.load(file)
            print("File exists. Loaded data:\n", df)
    else:
        # If file does not exist, save the data
        with open(file_path, 'wb') as file:

            df = loadRawData(type=type,start_year=2014,end_year=2024)

            pickle.dump(df, file)
            print("File did not exist. Created and saved data:\n", df)
    
    targets = ["Min Delay"]
    features = ["Date", "Time","Direction"]

    # df = loadRawData(start_year=2014,end_year=2016,type="bus")

    df,scaler = process_data(df,targets=targets,features=features)

    print(df.head())

    n_steps = 10
    n_outputs = 1
    
    
    X, y= create_sliding_windows(df,n_steps,n_outputs,target_column=targets[0])
    # print("X shape:",X.shape)
    # print("y shape:",y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(np.isnan(X_train).any(), np.isinf(X_train).any())
    print(np.isnan(y_train).any(), np.isinf(y_train).any())

    return X_train, X_test, y_train, y_test, scaler

def create_sliding_windows(df, n_steps, n_outputs, target_column):
    """
    Converts a DataFrame into overlapping sliding windows.

    Parameters:
    - df: Input DataFrame with features and target variable.
    - n_steps: Number of time steps in the input sequence.
    - n_outputs: Number of time steps in the output sequence.
    - target_column: Name or index of the target column.

    Returns:
    - X: Numpy array of shape (num_samples, n_steps, num_features)
    - y: Numpy array of shape (num_samples, n_outputs)
    """
    X, y = [], []
    if isinstance(target_column, str):
        target_index = df.columns.get_loc(target_column)  # Get column index
    else:
        target_index = target_column

    data = df.to_numpy()  # Convert to NumPy for efficiency
    for i in range(len(data) - n_steps - n_outputs + 1):
        # Include all columns except the target in X
        X.append(data[i:i + n_steps, :])
        # Use only the target column for y
        y.append(data[i + n_steps:i + n_steps + n_outputs, target_index])
    X = np.array(X)
    y = np.array(y)

    # Exclude the target column from X (optional if the target is among features)
    # X = np.delete(X, target_index, axis=-1)
    return X, y

def train_lstm_model(model, X_train, X_test, y_train, y_test):
    # creating model
    model.summary()

    #fitting model
    history = model.fit(X_train,
                        y_train,
                        epochs = 10,
                        batch_size = 128,
                        validation_split=0.2,
                        verbose=1)

    return history, model



    
def test_model(history,lstm_model,X_train,X_test,y_train,y_test):

    # testing model:

    y_pred_test = lstm_model.predict(X_test)
   

    n_elements = len(y_pred_test)
    pad_size = 10 - (n_elements % 10)  # Calculate how much to pad
    y_pred_test_padded = np.pad(y_pred_test, (0, pad_size), mode='constant', constant_values=0)
    y_pred_test = y_pred_test_padded.reshape(-1, 10, 1)

    y_pred_train = lstm_model.predict(X_train)
    n_elements = len(y_pred_train)
    pad_size = 10 - (n_elements % 10)  # Calculate how much to pad
    y_pred_train_padded = np.pad(y_pred_train, (0, pad_size), mode='constant', constant_values=0)
    y_pred_train = y_pred_train_padded.reshape(-1, 10, 1) #since its not divisble by 10 ?


    test_loss = lstm_model.evaluate(y_test, y_pred_test)
    train_loss = lstm_model.evaluate(y_train,y_pred_train)

    # y_test_pred = lstm_model.predict(X_test)
    # y_train_pred = lstm_model.predict(X_train)
    # # print("Predictions:", y_pred)

    print(f"MSE training: {train_loss[0]}")
    print(f"MSE testing: {test_loss[0]}")

    print(f'MAE train: {train_loss[1]}')
    print(f'MAE test: {test_loss[1]}')


    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def getForecastData(end_year,type):
    forecast_year = end_year + 1

    targets = ["Min Delay"]
    features = ["Date", "Time","Direction"]

    df = loadRawData(type=type,start_year=forecast_year,end_year=forecast_year)
    
    df,scaler = process_data(df,targets=targets,features=features)

    print(df.head())

    datetime = df.index #for use in plotting

    # datetime = pd.date_range(start=df.index[-365], periods=365, freq='D')
    n_steps = 10
    n_outputs = 1
    

    
    X, y = create_sliding_windows(df,n_steps,n_outputs,target_column=targets[0])

    return X, y, datetime[:-n_steps], scaler #this is data to be used for forecasting


def main():

    type = 'bus' #the dataset type we will be testing
    start_year = 2014
    end_year = 2023

    #function will create the split the dataset into training and testing 
    X_train, X_test, y_train, y_test, scaler = getXandY(type,start_year,end_year) # will store the data as pickle 
    
    # creating model
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print(X_train[0])
    print(y_train)

    input_shape = (X_train.shape[0],X_train.shape[2]) #num samples, num features in each sample
    lr = 0.00001


    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'lstm-model-{0}-{1}.pkl'.format(start_year,end_year))

    #this block loads the trained model + history if it is saved, otherwise reloading it
    if os.path.exists(file_path):
        # If file exists, load the data
        with open(file_path, 'rb') as file:
            
            lstm_history, lstm_model = pickle.load(file)
            print("File exists, reading \n")

    else:
        # If file does not exist, save the data
        with open(file_path, 'wb') as file:

            lstm_model = create_lstm_model(input_shape,lr)
            lstm_history, lstm_model = train_lstm_model(lstm_model, X_train, X_test, y_train, y_test)

            pickle.dump((lstm_history,lstm_model), file)
            print("File did not exist. Created and saved data\n")
    
    # lstm_model = create_lstm_model(input_shape,lr)
    # lstm_history, lstm_model = train_lstm_model(lstm_model, X_train, X_test, y_train, y_test)

    # getting model results / accuracy
    test_model(lstm_history, lstm_model, X_train, X_test, y_train, y_test)


    # forecasting on future years
    X_for, y_for, datetime, scaler_for = getForecastData(end_year,type)
    
    ## Getting forecasting predictions:
    y_pred_for = lstm_model.predict(X_for) 

    y_pred_for_unscaled = scaler_for.inverse_transform(y_pred_for) #unscaled based off standard scaler as defined before
    y_for_unscaled = scaler_for.inverse_transform(y_for) 

    # print("Predictions of the next year:",y_pred_for_unscaled)
    # print("Actual delays of the next year:",y_pred_unscaled)

    #plotting:
    # Plot the original delay (min delay or target delay) in the original dataset
    plt.plot(datetime, y_for, label='Original Delay (Actual)', color='blue', alpha=0.7)

    # Plot predicted delay for the last year (aligned with datetime index)
    plt.plot(datetime, y_pred_for, label='Predicted Delay', color='red', linestyle='--',alpha=0.7)

    # Customize plot
    plt.title('Original vs Predicted Delay')
    plt.xlabel('Date')
    plt.ylabel('Delay')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()