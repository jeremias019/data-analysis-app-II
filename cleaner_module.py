# In this file I will load and clean the dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataCleaner:
    
    def __init__(self, file_path, num_columns=None, cat_columns=None):
        # Define column names directly within the class
        self.column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                             'marital-status', 'occupation', 'relationship', 'race', 
                             'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                             'native-country', 'income']
        
        self.raw_data = pd.read_csv(file_path, names=self.column_names, header=0)
        
        # Default selection if columns are not provided
        if num_columns is None:
            num_columns = self.raw_data.select_dtypes(include=[np.number]).columns[:2].tolist()
        if cat_columns is None:
            cat_columns = self.raw_data.select_dtypes(include=['object']).columns[:2].tolist()
        
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.cleaned_data = self.raw_data.copy()
    
    def load_data(self):
        print("First few rows of the raw dataset:")
        print(self.raw_data.head())
        
        print("\nSummary statistics before cleaning:")
        print(self.raw_data.describe())
        
        print("\nData types:")
        print(self.raw_data.dtypes)
        
        print("\nUnique values per column:")
        for col in self.raw_data.columns:
            print(f"{col}: {self.raw_data[col].nunique()}")
    
    def check_missing_data(self):
        missing_data = self.raw_data.isnull().sum()
        
        
        print("\nMissing data points per column:")
        print(missing_data[missing_data > 0])
        
    
    def clean_data(self):
        # Impute missing values
        num_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        self.cleaned_data[num_cols] = imputer.fit_transform(self.cleaned_data[num_cols])
        
        cat_cols = self.cleaned_data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.cleaned_data.loc[:, col] = self.cleaned_data[col].fillna(self.cleaned_data[col].mode()[0])
        
        # Standardize numerical columns
        scaler = StandardScaler()
        self.cleaned_data[num_cols] = scaler.fit_transform(self.cleaned_data[num_cols])
        
        print("\nData after cleaning and pre-processing:")
        print(self.cleaned_data.head())
        
        print("\nSummary statistics after cleaning:")
        print(self.cleaned_data.describe())
    
    def get_raw_data(self):
        return self.raw_data
    
    def get_cleaned_data(self):
        return self.cleaned_data
    
    def get_selected_data(self, num_columns, cat_columns):
        return self.cleaned_data[num_columns + cat_columns]
    
    def save_to_csv(self, data, filename):
        data.to_csv(filename, index=False)
    
    def run_cleaning(self):
        self.load_data()
        self.check_missing_data()
        self.clean_data()

