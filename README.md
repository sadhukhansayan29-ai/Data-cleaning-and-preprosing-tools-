# Import required libraries
import pandas as pd
import numpy as np


# -------------------------------
# Data Cleaning & Preprocessing Tool
# -------------------------------


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        # keep a copy to avoid mutating caller's DataFrame unexpectedly
        self.data = data.copy()

    # 1) Handling Missing Values
    def handle_missing_values(self, method: str = "mean") -> pd.DataFrame:
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if method == "mean" and pd.api.types.is_numeric_dtype(
                    self.data[column]
                ):
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif method == "median" and pd.api.types.is_numeric_dtype(
                    self.data[column]
                ):
                    self.data[column].fillna(self.data[column].median(), inplace=True)
                elif method == "mode":
                    # mode() may return multiple values; take first
                    mode_vals = self.data[column].mode()
                    if not mode_vals.empty:
                        self.data[column].fillna(mode_vals[0], inplace=True)
                else:
                    # fallback: drop rows where this column is missing
                    self.data.dropna(subset=[column], inplace=True)
        return self.data

    # 2) Removing Duplicates
    def remove_duplicates(self) -> pd.DataFrame:
        before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        after = len(self.data)
        print(f"Removed {before - after} duplicate rows.")
        return self.data

    # 3) Outlier detection & removal using IQR method
    def remove_outliers(self) -> pd.DataFrame:
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before = len(self.data)
            self.data = self.data[
                (self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)
            ]
            after = len(self.data)
            print(f"Removed {before - after} outliers from column '{col}'.")
        return self.data

    # 4) Run full cleaning pipeline
    def clean_data(self, missing_method: str = "mean") -> pd.DataFrame:
        print("Handling missing values...")
        self.handle_missing_values(method=missing_method)
        print("Removing duplicates...")
        self.remove_duplicates()
        print("Removing outliers...")
        self.remove_outliers()
        print("✅ Data cleaning completed successfully!")
        return self.data


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example dataset
    data = {
        "Age": [25, 27, 29, np.nan, 22, 120, 25, 25],
        "Salary": [50000, 54000, np.nan, 58000, 62000, 300000, 50000, 50000],
        "City": [
            "Kolkata",
            "Delhi",
            "Mumbai",
            np.nan,
            "Delhi",
            "Kolkata",
            "Kolkata",
            "Kolkata",
        ],
    }

    df = pd.DataFrame(data)
    print("Original Data:")
    print(df)

    cleaner = DataPreprocessor(df)
    cleaned_df = cleaner.clean_data(missing_method="median")

    print("\nCleaned Data:")
    print(cleaned_df)
