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
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InsuranceModeling:
    def __init__(self, data):
        self.data = data
        self.results = []  # Store model results here

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
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log the model with MLflow
        with mlflow.start_run():
            mlflow.log_param("model_type", "Linear Regression")
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R²", r2)
            mlflow.sklearn.log_model(lr_model, "model")

        self.results.append(('Linear Regression', rmse, r2))
        logging.info(f'Linear Regression: RMSE = {rmse:.4f}, R² = {r2:.4f}')

        # Analyze feature importance (coefficients)
        self.analyze_linear_regression_importance(lr_model, X_train.columns)

    def analyze_linear_regression_importance(self, model, feature_names):
        """Analyze feature importance for Linear Regression."""
        importance = model.coef_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plotting the feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance (Coefficient)')
        plt.title('Feature Importance from Linear Regression')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.show()

    def fit_random_forest(self, X_train, y_train, X_test, y_test):
        """Fit a Random Forest model with hyperparameter tuning."""
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Simplified hyperparameter tuning
        param_dist = {
            'n_estimators': [100],  # Reduced to a single value
            'max_depth': [10]       # Reduced to a single value
        }
        random_search = RandomizedSearchCV(rf_model, param_dist, n_iter=1, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        logging.info(f"Random Forest fitting time: {time.time() - start_time:.2f} seconds")
        
        best_rf_model = random_search.best_estimator_
        y_pred = best_rf_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log the model with MLflow
        with mlflow.start_run():
            mlflow.log_param("model_type", "Random Forest")
            mlflow.log_param("n_estimators", random_search.best_params_['n_estimators'])
            mlflow.log_param("max_depth", random_search.best_params_['max_depth'])
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R²", r2)
            mlflow.sklearn.log_model(best_rf_model, "model")

        self.results.append(('Random Forest', rmse, r2))
        logging.info(f'Random Forest: RMSE = {rmse:.4f}, R² = {r2:.4f}')

        # Analyze feature importance for Random Forest
        self.analyze_feature_importance(best_rf_model, X_train.columns)

    def fit_xgboost(self, X_train, y_train, X_test, y_test):
        """Fit an XGBoost model with hyperparameter tuning."""
        xgb_model = XGBRegressor(random_state=42, n_jobs=-1)
        
        # Simplified hyperparameter tuning
        param_dist = {
            'n_estimators': [100],  # Reduced to a single value
            'max_depth': [3]        # Reduced to a single value
        }
        random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=1, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        best_xgb_model = random_search.best_estimator_
        y_pred = best_xgb_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log the model with MLflow
        with mlflow.start_run():
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("n_estimators", random_search.best_params_['n_estimators'])
            mlflow.log_param("max_depth", random_search.best_params_['max_depth'])
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R²", r2)
            mlflow.xgboost.log_model(best_xgb_model, "model")

        self.results.append(('XGBoost', rmse, r2))
        logging.info(f'XGBoost: RMSE = {rmse:.4f}, R² = {r2:.4f}')
        
        # Analyze feature importance for XGBoost
        self.analyze_feature_importance(best_xgb_model, X_train.columns)

    def analyze_feature_importance(self, model, feature_names):
        """Analyze feature importance using Random Forest and XGBoost."""
        importance = model.feature_importances_

        # Create a DataFrame for visualization
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plotting the feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance from Model')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.show()

    def run_linear_regression(self):
        """Run Linear Regression model."""
        premium_features = self.get_premium_features()
        X_train, X_test, y_train, y_test = self.prepare_data(premium_features)
        self.fit_linear_regression(X_train, y_train, X_test, y_test)

    def run_random_forest(self):
        """Run Random Forest model."""
        premium_features = self.get_premium_features()
        X_train, X_test, y_train, y_test = self.prepare_data(premium_features)
        self.fit_random_forest(X_train, y_train, X_test, y_test)

    def run_xgboost(self):
        """Run XGBoost model."""
        premium_features = self.get_premium_features()
        X_train, X_test, y_train, y_test = self.prepare_data(premium_features)
        self.fit_xgboost(X_train, y_train, X_test, y_test)

    def get_premium_features(self):
        return [
            'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 
            'AccountType', 'Gender', 'Country', 'Province', 'PostalCode', 
            'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType', 
            'RegistrationYear', 'Model', 'Cylinders', 'NumberOfDoors', 
            'VehicleAge', 'AlarmImmobiliser', 'TrackingDevice', 
            'CapitalOutstanding', 'IsNewVehicle'
        ]

    def display_results_comparison(self):
        """Display a comparison table of results."""
        results_df = pd.DataFrame(self.results, columns=['Model', 'RMSE', 'R²'])
        print("\nModel Comparison:")
        print(results_df)