"""
AISE 4010
This code is used to load the data, + process and other stuff idk

"""
from config import data_path

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split



def loadRawData(type="bus",start_year = 2014, end_year = 2015,targets = [], features = []):
    """
    type = bus, subway, streetcar
    start_year = start of year range
    end_year = end of year range
    targets = targets of dataset
    features = features of dataset

    
    loads data, based off given parameters
    """

    subfolder_path = os.path.join(data_path, type)

    if not os.path.isdir(subfolder_path):
            raise ValueError(f"Subfolder '{type}' does not exist in {data_path}.") #making sure path is correct
    
    all_data = pd.DataFrame()

    for filename in os.listdir(subfolder_path):
        print("On filename:",filename)
        if (
            filename.endswith(".xlsx") and
            filename.startswith(f"ttc-{type}-delay-data") and
            start_year <= int(filename.split("-")[-1].split(".")[0]) <= end_year
        ):
                file_path = os.path.join(subfolder_path, filename)
                data = pd.read_excel(file_path)
                all_data = pd.concat([all_data, data], ignore_index=True)    

    # all_data.info()
    print(all_data.describe())
    print("\n")
    return all_data

def process_data(df,targets,features):
    '''
    Takes in dataframe and preprocesses based off arguments 
    '''
    # targets = "min_delay"
    # features = ["","",""]

    df = df.sort_index()
    df = df[targets+features] #only using necessary data
    #drop empty rows:
    df.dropna(axis=0, how='all', inplace=True)
    # print(df['Time'].head())
    



    # combines the time with the date
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
    df['Datetime'] = pd.to_datetime(df['Report Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time']) #combining into one column

    #dropping the unecessary columns:
    df.drop(columns = ['Time','Report Date'], inplace = True)
    

    # preprocessing the direction to make consistent 4 + 1 directions 
    valid_directions = ['n','s','e','w','b'] #should only have n,e,s,w, b - both ways

    df['Direction'] = df['Direction'].str[0].str.lower()
    df['Direction'] = df['Direction'].apply(lambda x: x if x in valid_directions else 'unknown')
    
    unique_directions = df['Direction'].unique() 
    # print(unique_directions)

    #one hot encoding
    # categorical_features = df.select_dtypes(include=['object']).columns # only categorical features selected
    categorical_features = ['Route','Direction']
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_features])
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df, encoded_df], axis=1)
    df.set_index('Datetime',inplace=True)
    
    # print(df.columns)
    print(df.head())
    # scaling
     
    # scaler = StandardScaler() #standard because we expect standard deviation
    scaler = MinMaxScaler() #min max because ...

    df[['Min Delay']] = scaler.fit_transform(df[['Min Delay']])


    # Check for missing values in 'Min Delay'
    print(df['Min Delay'].isnull().sum())


    # plot the delay

    # print(df['Min Delay'].describe())
    monthly_delays = df['Min Delay'].resample('H').mean()
    print(monthly_delays.head(20))  # Check the first few rows of the resampled data
    monthly_delays.dropna(inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_delays, label='Minimum delay')
    plt.title('Min delay time series')
    plt.xlabel('Date')
    plt.ylabel('Min Delay')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    df = loadRawData()

    targets = ["Min Delay"]
    features = ["Report Date", "Route", "Time","Direction"]

    process_data(df,targets=targets,features=features)

if __name__ == "__main__":
    main()