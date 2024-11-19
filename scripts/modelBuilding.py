import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import time
import shap
from lime.lime_tabular import LimeTabularExplainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InsuranceModeling:
    def __init__(self, data):
        self.data = data
        self.results = []  # Store model results here
        self.model = None  # Store the trained model
        self.X_train = None  # Store training data
        self.X_test = None  # Store test data
        self.y_test = None  # Store test labels
        
        # Set the experiment name (create it if it doesn't exist)
        mlflow.set_experiment("Insurance_Claim_Analysis_Experiment")
    def prepare_data(self, premium_features):
        """Prepare the dataset for modeling."""
        X = self.data[premium_features]
        y = self.data['TotalPremium']
        
        # One-hot encoding for categorical variables
        X = pd.get_dummies(X, drop_first=True).fillna(0)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def fit_linear_regression(self, X_train, y_train, X_test, y_test):
        """Fit a Linear Regression model."""
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log the model with MLflow
        with mlflow.start_run():
            mlflow.log_param("model_type", "Linear Regression")
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R²", r2)
            mlflow.sklearn.log_model(self.model, "model")

        self.results.append(('Linear Regression', rmse, r2))
        logging.info(f'Linear Regression: RMSE = {rmse:.4f}, R² = {r2:.4f}')

        # Store data for LIME and SHAP
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test

    def run_linear_regression(self):
        """Run Linear Regression model."""
        premium_features = self.get_premium_features()
        X_train, X_test, y_train, y_test = self.prepare_data(premium_features)
        self.fit_linear_regression(X_train, y_train, X_test, y_test)

    def analyze_lime(self, instance_index=0):
        """Run LIME analysis on a specific instance."""
        if self.model is None:
            raise ValueError("You must run a model first to analyze LIME.")
        
        lime_explainer = LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, mode='regression')
        exp = lime_explainer.explain_instance(self.X_test.values[instance_index], self.model.predict, num_features=10)
        
        # Display LIME results
        try:
            exp.show_in_notebook(show_table=True)  # If using Jupyter notebook
        except Exception:
            fig = exp.as_pyplot_figure()
            plt.show()

    def analyze_shap(self):
        """Run SHAP analysis."""
        if self.model is None:
            raise ValueError("You must run a model first to analyze SHAP.")
        
        shap.initjs()  # Initialize SHAP JavaScript visualization
        shap_explainer = shap.Explainer(self.model, self.X_train)
        shap_values = shap_explainer(self.X_test)
        
        # Summary plot
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X_train.columns)

    # Other existing methods remain unchanged...

    def get_premium_features(self):
        return [
            'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 
            'AccountType', 'Gender', 'Country', 'Province', 'PostalCode', 
            'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType', 'MaritalStatus',
            'RegistrationYear', 'Model', 'Cylinders', 'NumberOfDoors', 
            'VehicleAge', 'AlarmImmobiliser', 'TrackingDevice', 
            'CapitalOutstanding', 'IsNewVehicle'
        ]

    def display_results_comparison(self):
        """Display a comparison table of results."""
        results_df = pd.DataFrame(self.results, columns=['Model', 'RMSE', 'R²'])
        print("\nModel Comparison:")
        print(results_df)