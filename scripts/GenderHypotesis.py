import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

class ABHypothesisTest:
    def __init__(self, df, feature_col, metric_col, group_a_values, group_b_values, alpha=0.05):
        """
        Initialize the ABHypothesisTest class with the DataFrame and relevant values for hypothesis testing.

        Parameters:
        df : pd.DataFrame : The dataset containing the data
        feature_col : str : The column used to segment the data (e.g., 'Gender', 'Province')
        metric_col : str : The KPI to evaluate (e.g., 'TotalClaims', 'SumInsured')
        group_a_values : list : The values in feature_col corresponding to Group A (e.g., ['Male'])
        group_b_values : list : The values in feature_col corresponding to Group B (e.g., ['Female'])
        alpha : float : Significance level (default 0.05)
        """
        self.df = df
        self.feature_col = feature_col
        self.metric_col = metric_col
        self.group_a_values = group_a_values
        self.group_b_values = group_b_values
        self.alpha = alpha

    def ab_hypothesis_test(self):
        """
        Perform A/B hypothesis testing to check for significant differences between two groups.
        This function assumes the metric is numerical and uses a t-test.
        """
        # Step 1: Data Segmentation
        group_A = self.df[self.df[self.feature_col].isin(self.group_a_values)][self.metric_col]
        group_B = self.df[self.df[self.feature_col].isin(self.group_b_values)][self.metric_col]

        # Ensure both groups have data to avoid errors in t-test
        if len(group_A) == 0 or len(group_B) == 0:
            print("One or both groups have no data. Cannot perform the test.")
            return

        # Step 2: Hypothesis Testing (t-test)
        t_stat, p_value = ttest_ind(group_A, group_B)

        # Step 3: Analyze the p-value and test result
        if p_value < self.alpha:
            print(f"Reject the null hypothesis. Significant difference between the groups for {self.metric_col}.")
        else:
            print(f"Fail to reject the null hypothesis. No significant difference between the groups for {self.metric_col}.")
        
        print(f"T-statistic: {t_stat}, p-value: {p_value}")

        # Step 4: Report Mean Values
        mean_A = group_A.mean()
        mean_B = group_B.mean()
        print(f"Mean {self.metric_col} for Group A: {mean_A}")
        print(f"Mean {self.metric_col} for Group B: {mean_B}")

    def chi_squared_test(self):
        """
        Perform a Chi-Squared test for categorical data.
        Use this method when the metric is categorical.
        """
        # Step 1: Create a contingency table
        contingency_table = pd.crosstab(self.df[self.feature_col], self.df[self.metric_col])

        # Step 2: Perform Chi-Squared test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Step 3: Analyze the p-value and test result
        if p_value < self.alpha:
            print(f"Reject the null hypothesis. Significant association between {self.feature_col} and {self.metric_col}.")
        else:
            print(f"Fail to reject the null hypothesis. No significant association between {self.feature_col} and {self.metric_col}.")

        print(f"Chi-Squared statistic: {chi2}, p-value: {p_value}")

    def run_test(self):
        """
        Runs the appropriate test based on the data type of the metric column.
        Automatically chooses between t-test (numerical data) and Chi-Squared test (categorical data).
        """
        # Filter the DataFrame to include only the relevant groups
        filtered_df = self.df[self.df[self.feature_col].isin(self.group_a_values + self.group_b_values)]
        self.df = filtered_df  # Update the dataframe with the filtered one

        # Count the number of data points in each group
        group_a_counts = self.df[self.df[self.feature_col].isin(self.group_a_values)].shape[0]
        group_b_counts = self.df[self.df[self.feature_col].isin(self.group_b_values)].shape[0]

        print(f"Number of data points in Group A: {group_a_counts}")
        print(f"Number of data points in Group B: {group_b_counts}")

        # If the metric column is numerical, run a t-test
        if pd.api.types.is_numeric_dtype(self.df[self.metric_col]):
            if group_a_counts > 0 and group_b_counts > 0:
                self.ab_hypothesis_test()
            else:
                print("Not enough data points for both groups.")
        # If the metric column is categorical, run a Chi-Squared test
        else:
            if len(self.df[self.feature_col].unique()) == 2:
                self.chi_squared_test()
            else:
                print("Not enough data points for both groups.")
