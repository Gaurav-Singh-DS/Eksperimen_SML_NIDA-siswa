# automated_nida.py

import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_housing_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Convert range sqft values like "2100 - 2850" to their average
    def convert_sqft(x):
        try:
            if isinstance(x, str) and '-' in x:
                low, high = map(float, x.split('-'))
                return (low + high) / 2
            return float(x)
        except:
            return np.nan

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

    # Extract number of bedrooms from "2 BHK", "4 Bedroom", etc.
    df['bedrooms'] = df['size'].str.extract(r'(\d+)').astype(float)

    # Handle missing values
    num_cols = ['total_sqft', 'bath', 'balcony']
    cat_cols = ['area_type', 'availability', 'location', 'size', 'society']

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Feature engineering
    df['price_per_sqft'] = df['price'] / df['total_sqft']

    # Remove outliers based on price_per_sqft
    df = df[(df['price_per_sqft'] >= df['price_per_sqft'].quantile(0.01)) & 
            (df['price_per_sqft'] <= df['price_per_sqft'].quantile(0.99))]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the processed data
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    preprocess_housing_data(
        'data/Bengaluru_House_Data.csv',                       
        'housing_preprocessed/processed_data.csv'              
    )
