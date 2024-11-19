import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class InsuranceBasicStast:
    def __init__(self, df):
        self.df = df
    def Describtive_ana(self):
        print(self.df.describe())
    def variability_spcf_col(self):
        result = self.df[['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured', 'CalculatedPremiumPerTerm']].describe()
        return result
    def data_types(self):
        data_typs = self.df.dtypes
        missing_df = pd.DataFrame({
            'Column': self.df.columns,  
            'data_typs': data_typs
        }).sort_values(by='data_typs', ascending=False) 
        return missing_df
    
    def missing_percentage(self):
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)
        
        return missing_df
    def fill_numeri_by_mean(self):
        self.df['NumberOfDoors'] = self.df['NumberOfDoors'].fillna(self.df['NumberOfDoors'].mean())
        self.df['SumInsured'] = self.df['SumInsured'].fillna(self.df['SumInsured'].mean())
        self.df['CalculatedPremiumPerTerm'] = self.df['CalculatedPremiumPerTerm'].fillna(self.df['CalculatedPremiumPerTerm'].mean())
        self.df['TotalClaims'] = self.df['TotalClaims'].fillna(self.df['TotalClaims'].mean())
        self.df['TotalClaims'] = self.df['TotalClaims'].fillna(self.df['TotalClaims'].mean())
    def fill_catagor_by_mode(self):
        self.df['Gender'] = self.df['Gender'].fillna(self.df['Gender'].mode()[0])
        self.df['MaritalStatus'] = self.df['MaritalStatus'].fillna(self.df['MaritalStatus'].mode()[0])
        self.df['Title'] = self.df['Title'].fillna(self.df['Title'].mode()[0])
    def rename_variable_value_name(self):
        self.df['Gender'] = self.df['Gender'].replace({'Not specified': 'not_specified', np.nan: 'not_specified'})
        self.df['MaritalStatus'] = self.df['MaritalStatus'].replace({'Not specified': np.nan})
    def univariate_analysis_plot(self):
        self.fill_numeri_by_mean()
        numerical_cols = ['NumberOfDoors','SumInsured', 'CalculatedPremiumPerTerm', 'TotalClaims', 'TotalClaims']
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_cols):
            plt.subplot(3, 3, i + 1)  # Adjust the subplot grid as needed
            sns.histplot(self.df[col], bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.xticks(rotation = 45, ha='right')
        plt.tight_layout()
        plt.show()
    def univariant_catagor_attri(self):
        self.rename_variable_value_name()
        categorical_cols = ['Product', 'TermFrequency', 'NewVehicle','VehicleType', 'Gender','MaritalStatus', 'Title']
        plt.figure(figsize=(15, 15))
        for i, col in enumerate(categorical_cols):
            plt.subplot(3, 3, i + 1)  # Adjust the subplot grid as needed
            sns.countplot(x=self.df[col])
            plt.title(f'Count of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    def bivariate_analysis(self):
        self.fill_numeri_by_mean()
        res_corr = self.df[['TotalPremium', 'TotalClaims', 'PostalCode']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(res_corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    def bivariate_scatter_plot(self):
        self.fill_numeri_by_mean()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='TotalPremium', y='TotalClaims', hue='PostalCode', alpha=0.6)
        plt.title('Scatter Plot of TotalPremium vs TotalClaims')
        plt.xlabel('TotalPremium')
        plt.ylabel('TotalClaims')
        plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    def analyze_insurance_trends(self):
        """
        Analyzes insurance trends by year using the provided DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing insurance data with a 'TransactionMonth' column.
        """
        
        # Convert TransactionMonth to datetime format
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        
        # Extract the year from TransactionMonth
        self.df['Year'] = self.df['TransactionMonth'].dt.year
        
        # 1. Trends in Insurance Cover Type by Year
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.df, x='CoverType', hue='Year')
        plt.title('Distribution of Insurance Cover Types by Year')
        plt.xlabel('Insurance Cover Type')
        plt.ylabel('Count')
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # 2. Average Premium Trends by Year
        avg_premium_by_year = self.df.groupby(['Year'])['TotalPremium'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=avg_premium_by_year, x='Year', y='TotalPremium', marker='o')
        plt.title('Average Premium Trends Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Premium')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # 3. Vehicle Makes by Year
        top_makes = self.df['make'].value_counts().nlargest(10).index
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df[self.df['make'].isin(top_makes)], x='make', y='TotalPremium', hue='Year')
        plt.title('Premiums for Top Vehicle Makes by Year')
        plt.xlabel('Vehicle Make')
        plt.ylabel('Total Premium')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    def plot_Vehicle_premium(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='VehicleType', y='TotalPremium')
        plt.title('Total Premium Distribution by Vehicle Type')
        plt.xticks(rotation=45)
        plt.xlabel('Vehicle Type')
        plt.ylabel('Total Premium')
        plt.tight_layout()
        plt.show()
    def outlier_all_num_co(self):
        # Updated list of numeric columns
        numeric_columns = [
            'TotalPremium', 'CalculatedPremiumPerTerm', 'TotalClaims', 
            'CustomValueEstimate', 'NumberOfDoors', 'kilowatts', 'cubiccapacity', 'Cylinders'
        ]
        
        # Filter numeric columns and check for non-numeric or missing columns
        available_columns = [col for col in numeric_columns if col in self.df.columns and np.issubdtype(self.df[col].dtype, np.number)]
        
        if len(available_columns) == 0:
            print("No numeric columns found to plot.")
            return
        
        # Fill missing values with the mean of each column
        df_clean = self.df[available_columns].fillna(self.df[available_columns].mean())
        
        # Filter out columns where all values are constant or NaN
        filtered_columns = [col for col in available_columns if df_clean[col].nunique() > 1]

        if len(filtered_columns) == 0:
            print("No numeric columns with variability found to plot.")
            return

        num_cols = len(filtered_columns)
        num_rows = int(np.ceil(num_cols / 2))  # Calculate number of rows required
        plt.figure(figsize=(15, num_rows * 5))  # Adjust figure size based on number of rows
        
        for i, col in enumerate(filtered_columns):
            plt.subplot(num_rows, 2, i + 1)
            sns.boxplot(data=df_clean, y=col)
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Value')
        
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()