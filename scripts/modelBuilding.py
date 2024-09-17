import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class InsuranceModeling:
    def __init__(self, data):
        self.data = data

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
        
        return 'Linear Regression', rmse, r2

    def fit_random_forest(self, X_train, y_train, X_test, y_test):
        """Fit a Random Forest model with hyperparameter tuning."""
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Hyperparameter tuning with RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None]
        }
        random_search = RandomizedSearchCV(rf_model, param_dist, n_iter=4, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        best_rf_model = random_search.best_estimator_
        y_pred = best_rf_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return 'Random Forest', rmse, r2

    def fit_xgboost(self, X_train, y_train, X_test, y_test):
        """Fit an XGBoost model with hyperparameter tuning."""
        xgb_model = XGBRegressor(random_state=42, n_jobs=-1)
        
        # Hyperparameter tuning with RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10]
        }
        random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=4, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        best_xgb_model = random_search.best_estimator_
        y_pred = best_xgb_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return 'XGBoost', rmse, r2

    def analyze_feature_importance(self, model, feature_names):
        """Analyze feature importance using XGBoost."""
        importance = model.feature_importances_

        # Create a DataFrame for visualization
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plotting the feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance from XGBoost')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.show()

    def run_models(self):
        """Run all models and evaluate their performance."""
        premium_features = [
            'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 
            'AccountType', 'Gender', 'Country', 'Province', 'PostalCode', 
            'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType', 
            'RegistrationYear', 'Model', 'Cylinders', 'NumberOfDoors', 
            'VehicleAge', 'AlarmImmobiliser', 'TrackingDevice', 
            'CapitalOutstanding', 'IsNewVehicle'
        ]
        
        X_train, X_test, y_train, y_test = self.prepare_data(premium_features)

        # Store results
        results = []
        results.append(self.fit_linear_regression(X_train, y_train, X_test, y_test))
        results.append(self.fit_random_forest(X_train, y_train, X_test, y_test))
        best_model_name, rmse, r2 = self.fit_xgboost(X_train, y_train, X_test, y_test)
        results.append((best_model_name, rmse, r2))

        # Analyze feature importance
        self.analyze_feature_importance(results[-1][0], X_train.columns)

        # Compare models
        self.compare_models(results)

    def compare_models(self, results):
        """Compare model performance and plot results."""
        print("\nModel Comparison:")
        print("{:<15} {:<10} {:<10}".format("Model", "RMSE", "R²"))
        
        models = []
        rmses = []
        r2s = []

        for model_name, rmse, r2 in results:
            print("{:<15} {:<10.4f} {:<10.4f}".format(model_name, rmse, r2))
            models.append(model_name)
            rmses.append(rmse)
            r2s.append(r2)

        # Plot RMSE
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(models, rmses, color='skyblue')
        plt.title('Model RMSE Comparison')
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.xticks(rotation=15)

        # Plot R²
        plt.subplot(1, 2, 2)
        plt.bar(models, r2s, color='lightgreen')
        plt.title('Model R² Comparison')
        plt.xlabel('Models')
        plt.ylabel('R²')
        plt.xticks(rotation=15)

        plt.tight_layout()
        plt.show()

# Example usage
# df = pd.read_csv('your_data.csv')  # Load your data
# insurance_modeling = InsuranceModeling(df)
# insurance_modeling.run_models()