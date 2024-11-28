"""
AISE 4010
This code is used to load the data, + process and other stuff idk

"""
from config import data_path

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dateutil.parser import parse

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import pickle


def standardize_time(time_str):
    # Add seconds if missing
    if len(time_str.split(':')) == 2:  # Format is h:m
        time_str += ':00'
    return time_str

def loadRawData(type="bus",start_year = 2014, end_year = 2015,targets = [], features = [],file_path = 'a'):
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
        # print("On filename:",filename)
        if (
            filename.endswith(".xlsx") and
            filename.startswith(f"ttc-{type}-delay-data") and
            start_year <= int(filename.split("-")[-1].split(".")[0]) <= end_year
        ):
                file_path = os.path.join(subfolder_path, filename)
                sheet_names = pd.ExcelFile(file_path).sheet_names
                for month in sheet_names:
                    data = pd.read_excel(file_path,sheet_name=month)

                    # accounting for inconsistent data formatting
                    if 'Report Date' in data.columns:
                        data.rename(columns={'Report Date': 'Date'}, inplace=True)
                    elif 'Date' in data.columns:
                        pass  # Column is already named "date"

                    if 'Delay' in data.columns:
                        data.rename(columns={'Delay': 'Min Delay'}, inplace=True)
                    elif 'Min Delay' in data.columns:
                        pass  # Column is already named "date"
                    
                    all_data = pd.concat([all_data, data], ignore_index=True)    

    # all_data.info()
    # print(all_data.describe())
    # print("\n")
    return all_data

def standardize_time_format(time_str):
    try:
        # Parse the time string
        parsed_time = parse(str(time_str)).time()  # Extract only the time
        # Format to HH:MM:SS
        return parsed_time.strftime('%H:%M:%S')
    except Exception as e:
        print(f"Could not parse '{time_str}': {e}")
        return None
    
def standardize_date_format(date_str):
    try:
        # Parse the date string
        parsed_date = parse(str(date_str)) 
        # Format to YYYY-MM-DD
        return parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Could not parse '{date_str}': {e}")
        return None

def process_data(df,targets,features,start="-01-01"):
    '''
    Takes in dataframe and preprocesses based off arguments 
    '''
    # targets = "min_delay"
    # features = ["","",""]
    # print("Using features:\n",features,"\nTargets:",targets)
    df = df.sort_index()
    df = df[targets+features] #only using necessary data
    #drop empty rows:
    df.dropna(axis=0, how='all', inplace=True) #drops where all are null
    # print(df['Time'].head())
    
    df['Time'] = df['Time'].apply(standardize_time_format)

    # combines the time with the date
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
    df['Datetime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time']) #combining into one column

    #dropping the unecessary columns:
    df.drop(columns = ['Time','Date'], inplace = True)
    # print(df.columns)

    # preprocessing the direction to make consistent 4 + 1 directions 
    valid_directions = ['n','s','e','w','b'] #should only have n,e,s,w, b - both ways

    df['Direction'] = df['Direction'].str[0].str.lower()
    df['Direction'] = df['Direction'].apply(lambda x: x if x in valid_directions else 'unknown')
    
    unique_directions = df['Direction'].unique() 
    # print(unique_directions)

    #one hot encoding
    categorical_features = df.select_dtypes(include=['object']).columns # only categorical features selected
    # categorical_features = ['Route','Direction']
    for features in categorical_features:
        df[features] = df[features].astype(str)
    # df['Route'] = df['Route'].astype(str)
    # df['Direction'] = df['Direction'].astype(str)

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_features])
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df, encoded_df], axis=1)

    df.drop(categorical_features, axis=1, inplace=True)

    df.set_index('Datetime',inplace=True)

    # Extract year, month, day, hour, and minute from the Datetime index
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
     
    scaler = StandardScaler() #standard because we expect standard deviation
    # scaler = MinMaxScaler() #min max because ...

    df[['Min Delay']] = scaler.fit_transform(df[['Min Delay']])
    df.dropna(axis=0, how='any', inplace=True)
    
    # plot the delay

    # monthly_delays = df['Min Delay'].resample('W').mean()

    # plt.figure(figsize=(10, 6))
    # plt.plot(monthly_delays, label='Minimum delay')
    # plt.title('Min delay time series')
    # plt.xlabel('Date')
    # plt.ylabel('Min Delay')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return df,scaler

def main():
    transit_type = "streetcar"
    # Get the directory of the current script (for relative path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # File path relative to the script's folder
    file_path = os.path.join(script_dir, 'ttc_{0}_data-2014-2024.pkl'.format(transit_type))

    if os.path.exists(file_path):
        # If file exists, load the data
        with open(file_path, 'rb') as file:
            df = pickle.load(file)
            print("File exists. Loaded data:\n", df)
    else:
        # If file does not exist, save the data
        with open(file_path, 'wb') as file:

            df = loadRawData(start_year=2014,end_year=2024)

            pickle.dump(df, file)
            print("File did not exist. Created and saved data:\n", df)


    
    # print(df.columns)

    targets = ["Min Delay"]
    features = ["Date", "Route", "Time","Direction"]

    process_data(df,targets=targets,features=features)

if __name__ == "__main__":
    main()