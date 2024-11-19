import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceByPostalCode:
    def __init__(self, df):
        self.df = df
        """
        Initialize the class with the dataset.
        :param df: Pandas DataFrame containing insurance data
        """

    def preprocess_data(self):
        """
        Preprocess the data by imputing missing values and scaling features.
        """
        # Feature Scaling
        scaler = StandardScaler()
        features = self.df.drop(['TotalClaims', 'PostalCode'], axis=1)
        scaled_features = scaler.fit_transform(features)
        self.df[features.columns] = scaled_features

    def plot_predictions_vs_actuals_all(self, models):
        """
        Plot the predicted vs actual total claims for all PostalCodes.
        """
        plt.figure(figsize=(10, 6))
        all_y = []
        all_predictions = []
        
        for PostalCode, model in models.items():
            PostalCode_data = self.df[self.df['PostalCode'] == PostalCode]
            X = PostalCode_data.drop(['TotalClaims', 'PostalCode'], axis=1)
            y = PostalCode_data['TotalClaims']
            predictions = model.predict(X)
            
            all_y.extend(y)
            all_predictions.extend(predictions)
        
        plt.scatter(all_y, all_predictions, alpha=0.5, label='Predictions')
        plt.plot([min(all_y), max(all_y)], [min(all_y), max(all_y)], 'k--', lw=2, label='Ideal Fit')
        plt.xlabel('Actual Total Claims')
        plt.ylabel('Predicted Total Claims')
        plt.title('Predicted vs Actual Total Claims (All PostalCodes)')
        plt.legend()
        plt.show()
    
    def plot_residuals_all(self, models):
        """
        Plot the residuals for all PostalCodes.
        """
        plt.figure(figsize=(10, 6))
        all_residuals = []
        
        for PostalCode, model in models.items():
            PostalCode_data = self.df[self.df['PostalCode'] == PostalCode]
            X = PostalCode_data.drop(['TotalClaims', 'PostalCode'], axis=1)
            y = PostalCode_data['TotalClaims']
            predictions = model.predict(X)
            residuals = y - predictions
            
            all_residuals.extend(residuals)
        
        sns.histplot(all_residuals, kde=True)
        plt.xlabel('Residuals')
        plt.title('Residuals Distribution (All PostalCodes)')
        plt.show()
    
    def plot_feature_importance_all(self, models, feature_names):
        """
        Plot the feature importance based on model coefficients averaged across all models.
        """
        average_coefficients = pd.Series(0, index=feature_names)
        
        for PostalCode, model in models.items():
            coefficients = model.coef_
            average_coefficients += pd.Series(coefficients, index=feature_names)
        
        average_coefficients /= len(models)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': average_coefficients
        }).sort_values(by='Coefficient', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
        plt.title('Average Feature Importance Across All PostalCodes')
        plt.show()
    
    def fit_linear_regression_per_PostalCode(self):
        """
        Fit a linear regression model for each PostalCode and return the models.
        """
        self.preprocess_data()
        models = {}
        
        for PostalCode in self.df['PostalCode'].unique():
            PostalCode_data = self.df[self.df['PostalCode'] == PostalCode]
            
            # Check if there are enough samples
            if len(PostalCode_data) < 5:  # Adjust the threshold as needed
                print(f"Not enough data for PostalCode {PostalCode}. Skipping.")
                continue
            
            X = PostalCode_data.drop(['TotalClaims', 'PostalCode'], axis=1)
            y = PostalCode_data['TotalClaims']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            print(f"PostalCode {PostalCode} - R^2 Score: {r2_score(y_test, predictions)}")
            
            models[PostalCode] = model
        
        return models

# Example usage:
# df = pd.read_csv('your_data.csv')  # Replace with your actual data loading
# insurance_modeler = InsuranceByPostalCode(df)
# model_dict = insurance_modeler.fit_linear_regression_per_PostalCode()
# insurance_modeler.plot_predictions_vs_actuals_all(model_dict)
# insurance_modeler.plot_residuals_all(model_dict)
# insurance_modeler.plot_feature_importance_all(model_dict, df.drop(['TotalClaims', 'PostalCode'], axis=1).columns)
