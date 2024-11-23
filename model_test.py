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
    model.add(layers.LSTM(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(1))  # Change based on your output

    optimizer = Adam(learning_rate=lr) #setting 

    model.compile(optimizer=optimizer, metrics = ['mae'], loss='mse')  #use mae and mse since regression
    return model



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
    X = np.delete(X, target_index, axis=-1)
    return X, y

def main():
    transit_type = "bus"
    # Get the directory of the current script (for relative path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # File path relative to the script's folder
    file_path = os.path.join(script_dir, 'ttc_{0}_data-2014-2024.pkl'.format(transit_type))

    # if os.path.exists(file_path):
    #     # If file exists, load the data
    #     with open(file_path, 'rb') as file:
    #         df = pickle.load(file)
    #         print("File exists. Loaded data:\n", df)
    # else:
    #     # If file does not exist, save the data
    #     with open(file_path, 'wb') as file:

    #         df = loadRawData(start_year=2014,end_year=2024)

    #         pickle.dump(df, file)
    #         print("File did not exist. Created and saved data:\n", df)

    
    targets = ["Min Delay"]
    features = ["Date", "Time","Direction"]

    df = loadRawData(start_year=2014,end_year=2016,type="bus")

    df = process_data(df,targets=targets,features=features)

    print(df.head())
    # df = df.reset_index() #including the Datetime as a feature
    


    # X = df.drop(targets,axis=1)
    # y = df[targets]
    
    n_steps = 10
    n_outputs = 1
    
    
    X, y= create_sliding_windows(df,n_steps,n_outputs,target_column=targets[0])
    print("X shape:",X.shape)
    print("y shape:",y.shape)
    # print(X[0][0][0],y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features)) 
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))

    print("HELOOOOOOOOOOOO")
    print(np.isnan(X_train).any(), np.isinf(X_train).any())
    print(np.isnan(y_train).any(), np.isinf(y_train).any())
    print("HELOOOOOOOOOOOO")


    # creating model
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print(X_train[0])
    print(y_train)

    input_shape = (X_train.shape[0],X_train.shape[2]) #num samples, num features in each sample
    lstm_model = create_lstm_model(input_shape = input_shape, lr = 0.0001)
    lstm_model.summary()



    #fitting model
    history = lstm_model.fit(X_train, y_train, epochs = 10, batch_size = 128, verbose=1)

    # testing model:
    test_loss = lstm_model.evaluate(X_test, y_test)
    train_loss = lstm_model.evaluate(X_train, y_train)

    # y_test_pred = lstm_model.predict(X_test)
    # y_train_pred = lstm_model.predict(X_train)
    # # print("Predictions:", y_pred)

    print(f"MSE training: {train_loss[0]}")
    print(f"MSE testing: {test_loss[0]}")

    print(f'MAE train: {train_loss[1]}')
    print(f'MAE test: {test_loss[1]}')


    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()