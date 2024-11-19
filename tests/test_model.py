import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import sys
sys.path.insert(0, 'D:/10 A KAI 2/week 3/Insurance_Claim_Analysis/')
from scripts.modelBuilding import InsuranceModeling 

class TestInsuranceModeling(unittest.TestCase):

    def setUp(self):
        # Create mock data for testing
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
        self.data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        self.data['TotalPremium'] = y
        
        self.model = InsuranceModeling(self.data)

    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.log_model')
    def test_fit_linear_regression(self, mock_log_model, mock_start_run):
        premium_features = [f'feature_{i}' for i in range(10)]
        X_train, X_test, y_train, y_test = self.model.prepare_data(premium_features)

        self.model.fit_linear_regression(X_train, y_train, X_test, y_test)

        # Check if results are logged
        self.assertIn(('Linear Regression', unittest.mock.ANY, unittest.mock.ANY), self.model.results)
        mock_start_run.assert_called_once()
        mock_log_model.assert_called_once()

    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.log_model')
    @patch('matplotlib.pyplot.show')  # Mock plt.show to prevent displaying the plot
    def test_analyze_linear_regression_importance(self, mock_show, mock_log_model, mock_start_run):
        lr_model = MagicMock(coef=np.array([0.1, 0.3, 0.5]))
        feature_names = ['feature_0', 'feature_1', 'feature_2']
        
        self.model.analyze_linear_regression_importance(lr_model, feature_names)

        # Check if the plot was created
        mock_show.assert_called_once()

    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.log_model')
    def test_fit_random_forest(self, mock_log_model, mock_start_run):
        premium_features = [f'feature_{i}' for i in range(10)]
        X_train, X_test, y_train, y_test = self.model.prepare_data(premium_features)

        self.model.fit_random_forest(X_train, y_train, X_test, y_test)

        # Check if results are logged
        self.assertIn(('Random Forest', unittest.mock.ANY, unittest.mock.ANY), self.model.results)
        mock_start_run.assert_called_once()
        mock_log_model.assert_called_once()

    @patch('mlflow.start_run')
    @patch('mlflow.xgboost.log_model')
    def test_fit_xgboost(self, mock_log_model, mock_start_run):
        premium_features = [f'feature_{i}' for i in range(10)]
        X_train, X_test, y_train, y_test = self.model.prepare_data(premium_features)

        self.model.fit_xgboost(X_train, y_train, X_test, y_test)

        # Check if results are logged
        self.assertIn(('XGBoost', unittest.mock.ANY, unittest.mock.ANY), self.model.results)
        mock_start_run.assert_called_once()
        mock_log_model.assert_called_once()

    @patch('matplotlib.pyplot.show')  # Mock plt.show to prevent displaying the plot
    def test_analyze_feature_importance(self, mock_show):
        model = MagicMock(feature_importances=np.array([0.2, 0.5, 0.3]))
        feature_names = ['feature_0', 'feature_1', 'feature_2']

        self.model.analyze_feature_importance(model, feature_names)

        # Check if the plot was created
        mock_show.assert_called_once()

    def test_prepare_data(self):
        premium_features = [f'feature_{i}' for i in range(10)]
        X_train, X_test, y_train, y_test = self.model.prepare_data(premium_features)

        # Check the shape of the training and testing sets
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)

    def test_get_premium_features(self):
        expected_features = [
            'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 
            'AccountType', 'Gender', 'Country', 'Province', 'PostalCode', 
            'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType', 
            'RegistrationYear', 'Model', 'Cylinders', 'NumberOfDoors', 
            'VehicleAge', 'AlarmImmobiliser', 'TrackingDevice', 
            'CapitalOutstanding', 'IsNewVehicle'
        ]
        self.assertListEqual(self.model.get_premium_features(), expected_features)

    def test_display_results_comparison(self):
        self.model.results = [('Linear Regression', 0.5, 0.8), ('Random Forest', 0.4, 0.85)]
        with patch('builtins.print') as mock_print:
            self.model.display_results_comparison()
            mock_print.assert_called_once_with("\nModel Comparison:\n", pd.DataFrame(self.model.results, columns=['Model', 'RMSE', 'R²']))

if __name__ == '__main__':
    unittest.main()