import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
    def rename_variable_value_name(self):
        self.df['Gender'] = self.df['Gender'].replace({'Not specified': 'not_specified', np.nan: 'not_specified'})
        self.df['MaritalStatus'] = self.df['MaritalStatus'].replace({'Not specified': np.nan})
    def handle_missing_data(self):
        self.rename_variable_value_name()
        """Handle missing data by dropping columns with excessive missing values and imputing remaining values."""
        # Calculate missing percentage for each column
        missing_percentage = self.df.isnull().mean() * 100
        
        # Drop columns with > 50% missing data
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        if high_missing_cols:
            self.df.drop(columns=high_missing_cols, inplace=True)
            print(f"Dropped columns with > 50% missing data: {high_missing_cols}")

        # Impute numerical columns with mean
        num_imputer = SimpleImputer(strategy='mean')
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns  # Include int64 as well
        if numerical_cols.any():
            self.df[numerical_cols] = num_imputer.fit_transform(self.df[numerical_cols])
        else:
            print("No numerical columns found for imputation.")

        # Impute categorical columns with the most frequent value
        cat_imputer = SimpleImputer(strategy='most_frequent')
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if categorical_cols.any():
            self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])
        else:
            print("No categorical columns found for imputation.")

        print("Preprocessing completed. Missing values handled.")

    def feature_engineering(self):
        """Extract features from datetime columns and perform additional feature engineering."""
        # Standardize and extract features from 'TransactionMonth'
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
            self.df['TransactionYear'] = self.df['TransactionMonth'].dt.year
            self.df['TransactionMonthNum'] = self.df['TransactionMonth'].dt.month
            self.df['TransactionDay'] = self.df['TransactionMonth'].dt.day
            self.df.drop(columns=['TransactionMonth'], inplace=True)

        # Standardize and extract features from 'VehicleIntroDate'
        if 'VehicleIntroDate' in self.df.columns:
            self.df['VehicleIntroDate'] = pd.to_datetime(self.df['VehicleIntroDate'], errors='coerce')
            self.df['VehicleIntroYear'] = self.df['VehicleIntroDate'].dt.year
            self.df['VehicleIntroMonth'] = self.df['VehicleIntroDate'].dt.month
            self.df.drop(columns=['VehicleIntroDate'], inplace=True)

        # Feature engineering example
        if 'RegistrationYear' in self.df.columns:
            self.df['VehicleAge'] = 2023 - self.df['RegistrationYear']
        if 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
            self.df['PremiumPerClaim'] = self.df['TotalPremium'] / (self.df['TotalClaims'] + 1)  # Avoid division by zero
        if 'NewVehicle' in self.df.columns:
            self.df['IsNewVehicle'] = self.df['NewVehicle'].apply(lambda x: 1 if x == 'Yes' else 0)

        print("Feature engineering completed.")

    def encode_categorical_data(self):
        """Encode categorical variables in the DataFrame using label encoding."""
        label_encoders = {}
        value_mappings = {}
        categorical_cols = self.df.select_dtypes(include=['object']).columns

        # Specify columns to exclude from encoding
        exclude_cols = ['RegistrationYear', 'TransactionYear', 'VehicleIntroYear']
        
        for col in categorical_cols:
            # Skip excluded columns
            if col in exclude_cols:
                continue

            # Convert the column to string to avoid mixed types
            self.df[col] = self.df[col].astype(str)

            le = LabelEncoder()  # Create a new LabelEncoder instance
            self.df[col] = le.fit_transform(self.df[col])  # Fit and transform the data
            
            # Store the encoder and the mapping of original values to encoded values
            label_encoders[col] = le
            value_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        print("Categorical data encoded using label encoding.")
        
        # Convert value_mappings to a DataFrame
        mappings_df = pd.DataFrame(
            [(col, orig, enc) for col, mapping in value_mappings.items() for orig, enc in mapping.items()],
            columns=['Column', 'Original Value', 'Encoded Value']
        )

        return value_mappings, mappings_df  # Return both the mappings dictionary and DataFrame
    def standardize_numerical_data(self):
        scaler = StandardScaler()
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        if not numerical_cols.empty:
            self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
            print("Numerical data standardized.")
        else:
            print("No numerical columns found for standardization.")
    

    def get_processed_data(self):
        """Return the processed DataFrame."""
        return self.df

    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        self.handle_missing_data()
        self.feature_engineering()
        value_mappings, mappings_df = self.encode_categorical_data()  # Capture the value mappings
        print("Preprocessing pipeline completed.")
        return value_mappings, mappings_df  # Return the mappings and DataFrame