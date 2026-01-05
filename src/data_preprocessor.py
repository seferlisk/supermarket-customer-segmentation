import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataPreprocessor:
    def __init__(self, file_path, scaler_type='standard'):
        self.file_path = file_path
        self.data = None
        self.df_clean = None
        # Select scaler logic
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

    def load_data(self):
        """Loads data from the CSV file."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def get_features(self, columns):
        """Selects specific columns for clustering."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.X = self.data[columns]
        return self.X

    def inspect_data(self):
        """Returns head and info for inspection."""
        if self.data is None:
            return None, None
        return self.data.head(), self.data.info()

    def remove_outliers(self, columns, threshold=1.5):
        """
        Removes outliers based on the IQR method.
        Returns the cleaned dataframe and the number of rows removed.
        """
        if self.data is None: raise ValueError("Data not loaded.")

        # CRITICAL: We perform the cleaning on a copy
        cleaned_data = self.data.copy()

        for col in columns:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]

        # CRITICAL: We assign the result back to the class attribute
        self.df_clean = cleaned_data

        return self.df_clean

    def plot_boxplots(self, columns):
        """Visualizes distribution to spot outliers."""
        plt.figure(figsize=(10, 5))
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.boxplot(y=self.data[col])
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    def get_scaled_features(self, columns):
        """Scales the selected features and returns the scaled array. If data isn't cleaned yet,
        it cleans it automatically."""
        # Check if df_clean exists
        if self.df_clean is None:
            print("Warning: remove_outliers() was not called. Calling it automatically now...")
            self.remove_outliers(columns)

        X = self.df_clean[columns].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def check_nulls(self):
        """Checks for missing values in the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        null_counts = self.data.isnull().sum()
        has_nulls = null_counts.sum() > 0
        return null_counts, has_nulls

