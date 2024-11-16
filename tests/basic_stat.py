import unittest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'D:/10 A KAI 2/week 3/Insurance_Claim_Analysis/')
from scripts.Basic_stat import InsuranceBasicStast  # Adjust the import according to your module name
class TestInsuranceBasicStast(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing with the structure you provided
        data = {
            'UnderwrittenCoverID': [145249, 145249, 145249, 145255, 145255, 145247],
            'PolicyID': ['12827', '12827', '12827', '12827', '12827', '12827'],
            'TransactionMonth': pd.to_datetime(['2015-03-01', '2015-05-01', '2015-07-01', 
                                                 '2015-05-01', '2015-07-01', '2015-01-01']),
            'IsVATRegistered': [True, True, True, True, True, True],
            'Citizenship': ['AF, ZA'] * 6,  # Repeat the same value to match the row count
            'LegalType': ['Individual'] * 6,
            'Title': ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr'],
            'Language': ['English'] * 6,
            'Bank': ['First National Bank'] * 6,
            'AccountType': ['Current account'] * 6,
            'MaritalStatus': ['Married'] * 6,
            'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Male'],
            'Country': ['South Africa'] * 6,
            'Province': ['Gauteng'] * 6,
            'PostalCode': [1459] * 6,
            'MainCrestaZone': ['Rand East'] * 6,
            'SubCrestaZone': ['Rand East'] * 6,
            'ItemType': ['Mobility - Motor'] * 6,
            'mmcode': [44069150] * 6,
            'VehicleType': ['Passenger Vehicle'] * 6,
            'RegistrationYear': [2004] * 6,
            'make': ['MERCEDES-BENZ'] * 6,
            'Model': ['E 240'] * 6,
            'Cylinders': [6] * 6,
            'cubiccapacity': [2597] * 6,
            'kilowatts': [130] * 6,
            'bodytype': ['S/D'] * 6,
            'NumberOfDoors': [4] * 6,
            'VehicleIntroDate': ['Jun-02'] * 6,
            'CustomValueEstimate': [119300] * 6,
            'AlarmImmobiliser': ['Yes'] * 6,
            'TrackingDevice': ['No'] * 6,
            'CapitalOutstanding': [119300, 119300, 119300, 119300, 119300, 500000],
            'NewVehicle': ['More than 6 months'] * 6,
            'WrittenOff': [np.nan] * 6,
            'Rebuilt': [np.nan] * 6,
            'Converted': [np.nan] * 6,
            'CrossBorder': [np.nan] * 6,
            'NumberOfVehiclesInFleet': [np.nan] * 6,
            'SumInsured': [np.nan, np.nan, np.nan, 119300, 119300, 500000],
            'TermFrequency': ['Monthly'] * 6,
            'CalculatedPremiumPerTerm': [25, 584.6468, 57.5412, 584.6468, 584.6468, 57.5412],
            'ExcessSelected': ['Windscreen'] * 6,
            'CoverCategory': ['Comprehensive - Taxi'] * 6,
            'CoverType': ['Motor Comprehensive'] * 6,
            'CoverGroup': ['Mobility Metered Taxis: Monthly'] * 6,
            'Section': ['Commercial'] * 6,
            'Product': ['IFRS Constant'] * 6,
            'StatutoryClass': [21.92982456] * 6,
            'StatutoryRiskType': [0] * 6,
            'TotalPremium': [0.0, 512.8480702, 21.92982456, 0.0, 512.8480702, 3.256434635],
            'TotalClaims': [0, 0, 0, 0, 0, 0],
            'Year': [2015] * 6
        }
        self.df = pd.DataFrame(data)
        self.insurance = InsuranceBasicStast(self.df)
    def test_descriptive_ana(self):
        expected = self.df.describe().T.round(2)
        result = self.insurance.Describtive_ana()
        self.assertIsNone(result)  # Since the method prints, we check for None

    def test_variability_spcf_col(self):
        expected = self.df[['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured']].describe().T.round(2)
        result = self.insurance.variability_spcf_col()
        pd.testing.assert_frame_equal(result, expected)

    def test_data_types(self):
        expected = pd.DataFrame({
            'Column': self.df.columns,
            'data_typs': self.df.dtypes
        }).sort_values(by='data_typs', ascending=False)
        result = self.insurance.data_types()
        pd.testing.assert_frame_equal(result, expected)

if __name__ == '__main__':
    unittest.main()