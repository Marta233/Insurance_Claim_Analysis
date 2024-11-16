import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set(style="whitegrid")
class InsuranceBasicStast:
    def __init__(self, df):
        self.df = df
        logging.info("Initialized InsuranceBasicStast with dataframe of shape %s", df.shape)

    def Describtive_ana(self):
        logging.info("Entering Descriptive Analysis")
        result = self.df.describe().T.round(2)
        print(result)
        logging.info("Exiting Descriptive Analysis")

    def variability_spcf_col(self):
        logging.info("Calculating variability for specific columns")
        result = self.df[['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured']].describe().T.round(2)
        logging.info("Exiting variability_spcf_col")
        return result

    def data_types(self):
        logging.info("Getting data types and missing data")
        data_typs = self.df.dtypes
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'data_typs': data_typs
        }).sort_values(by='data_typs', ascending=False)
        logging.info("Exiting data_types")
        return missing_df

    def missing_percentage(self):
        logging.info("Calculating missing percentage")
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)
        logging.info("Exiting missing_percentage")
        return missing_df

    def fill_numeri_by_mean(self):
        logging.info("Filling numeric columns with mean values")
        numeric_columns = ['NumberOfDoors', 'SumInsured', 'CalculatedPremiumPerTerm', 'TotalClaims']
        for column in numeric_columns:
            self.df[column] = self.df[column].fillna(self.df[column].mean())
        logging.info("Exiting fill_numeri_by_mean")

    def fill_catagor_by_mode(self):
        logging.info("Filling categorical columns with mode values")
        categorical_columns = ['Gender', 'MaritalStatus', 'Title']
        for column in categorical_columns:
            self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        logging.info("Exiting fill_catagor_by_mode")

    def rename_variable_value_name(self):
        logging.info("Renaming variable values")
        self.df['Gender'] = self.df['Gender'].fillna('Not specified')
        self.df['MaritalStatus'] = self.df['MaritalStatus'].fillna('Not specified')
    def univariate_analysis_plot(self):
        logging.info("Generating univariate analysis plots for numeric")
        self.fill_numeri_by_mean()
        numerical_cols = ['NumberOfDoors','PostalCode','kilowatts', 'SumInsured', 'CalculatedPremiumPerTerm', 'TotalClaims']
        
        # Calculate the number of rows needed (2 plots per row)
        num_rows = (len(numerical_cols) + 1) // 2  # Use integer division to determine rows needed
        
        plt.figure(figsize=(15, num_rows * 5))  # Adjust figure height based on number of rows
        for i, col in enumerate(numerical_cols):
            plt.subplot(num_rows, 2, i + 1)  # 2 plots per row
            sns.histplot(self.df[col], bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        logging.info("Exiting univariate_analysis_plot")

    def univariant_catagor_attri(self):
        logging.info("Generating univariate categorical attribute plots")
        self.rename_variable_value_name()
        categorical_cols = ['Product', 'TermFrequency', 'CoverCategory','NewVehicle', 'VehicleType', 'Gender','Language', 'MaritalStatus', 'Title']
        plt.figure(figsize=(10, 8))
        for i, col in enumerate(categorical_cols):
            sns.countplot(x=self.df[col])
            plt.title(f'Count of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        logging.info("Exiting univariant_catagor_attri")

    def bivariate_analysis(self):
        logging.info("Performing bivariate analysis")
        self.fill_numeri_by_mean()
        res_corr = self.df[['TotalPremium', 'TotalClaims', 'PostalCode']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(res_corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
        logging.info("Exiting bivariate_analysis")

    def bivariate_scatter_plot(self):
        logging.info("Generating bivariate scatter plot")
        self.fill_numeri_by_mean()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='TotalPremium', y='TotalClaims', hue='PostalCode', alpha=0.6)
        plt.title('Scatter Plot of TotalPremium vs TotalClaims')
        plt.xlabel('TotalPremium')
        plt.ylabel('TotalClaims')
        plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        logging.info("Exiting bivariate_scatter_plot")

    def analyze_insurance_trends(self):
        logging.info("Analyzing insurance trends")
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        self.df['Year'] = self.df['TransactionMonth'].dt.year
        
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.df, x='CoverType', hue='Year')
        plt.title('Distribution of Insurance Cover Types by Year')
        plt.xlabel('Insurance Cover Type')
        plt.ylabel('Count')
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        avg_premium_by_year = self.df.groupby(['Year'])['TotalPremium'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=avg_premium_by_year, x='Year', y='TotalPremium', marker='o')
        plt.title('Average Premium Trends Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Premium')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        top_makes = self.df['make'].value_counts().nlargest(10).index
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df[self.df['make'].isin(top_makes)], x='make', y='TotalPremium', hue='Year')
        plt.title('Premiums for Top Vehicle Makes by Year')
        plt.xlabel('Vehicle Make')
        plt.ylabel('Total Premium')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        logging.info("Exiting analyze_insurance_trends")

    def plot_Vehicle_premium(self):
        logging.info("Plotting vehicle premium distribution")
        
        # Box plot for Total Premium by Vehicle Type
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='VehicleType', y='TotalPremium', palette='Set2')
        plt.title('Total Premium Distribution by Vehicle Type', fontsize=16)
        plt.xticks(rotation=45)
        plt.xlabel('Vehicle Type', fontsize=14)
        plt.ylabel('Total Premium', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Bar plot for Total Premium by Vehicle Type and Country
        plt.figure(figsize=(14, 7))
        sns.barplot(data=self.df, x='VehicleType', y='TotalPremium', hue='Province', ci=None, palette='pastel')
        plt.title('Average Total Premium by Vehicle Type and Country', fontsize=16)
        plt.xticks(rotation=45)
        plt.xlabel('Vehicle Type', fontsize=14)
        plt.ylabel('Average Total Premium', fontsize=14)
        plt.legend(title='Country', fontsize=12)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        
        logging.info("Exiting plot_Vehicle_premium")

    def plot_Vehicle_TotalClaims(self):
        logging.info("Plotting vehicle total claims distribution")
        
        # Box plot for Total Claims by Vehicle Type
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='VehicleType', y='TotalClaims', palette='Set2')
        plt.title('Total Claims Distribution by Vehicle Type', fontsize=16)
        plt.xticks(rotation=45)
        plt.xlabel('Vehicle Type', fontsize=14)
        plt.ylabel('Total Claims', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Bar plot for Total Claims by Vehicle Type and Country
        plt.figure(figsize=(14, 7))
        sns.barplot(data=self.df, x='VehicleType', y='TotalClaims', hue='Province', ci=None, palette='pastel')
        plt.title('Average Total Claims by Vehicle Type and Country', fontsize=16)
        plt.xticks(rotation=45)
        plt.xlabel('Vehicle Type', fontsize=14)
        plt.ylabel('Average Total Claims', fontsize=14)
        plt.legend(title='Country', fontsize=12)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        
        logging.info("Exiting plot_Vehicle_TotalClaims")

    def outlier_all_num_co(self):
        logging.info("Checking for outliers in numeric columns")
        numeric_columns = [
            'TotalPremium', 'CalculatedPremiumPerTerm', 'TotalClaims', 
            'CustomValueEstimate', 'NumberOfDoors', 'kilowatts', 'cubiccapacity', 'Cylinders'
        ]
        
        available_columns = [col for col in numeric_columns if col in self.df.columns and np.issubdtype(self.df[col].dtype, np.number)]
        
        if len(available_columns) == 0:
            logging.warning("No numeric columns found to plot.")
            return
        
        df_clean = self.df[available_columns].fillna(self.df[available_columns].mean())
        filtered_columns = [col for col in available_columns if df_clean[col].nunique() > 1]

        if len(filtered_columns) == 0:
            logging.warning("No numeric columns with variability found to plot.")
            return

        num_cols = len(filtered_columns)
        num_rows = int(np.ceil(num_cols / 2))
        plt.figure(figsize=(15, num_rows * 5))
        
        for i, col in enumerate(filtered_columns):
            plt.subplot(num_rows, 2, i + 1)
            sns.boxplot(data=df_clean, y=col)
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Value')
        
        plt.tight_layout()
        plt.show()
        logging.info("Exiting outlier_all_num_co")