"""
AISE 4010
This code is used to load the data, + process and other stuff idk

"""

import os
import pandas as pd
from config import data_path

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

    all_data.info()

def main():
    loadRawData()

if __name__ == "__main__":
    main()