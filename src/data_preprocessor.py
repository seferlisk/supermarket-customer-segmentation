import pandas as pd

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None

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

    def check_nulls(self):
        """Checks for missing values in the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        null_counts = self.data.isnull().sum()
        has_nulls = null_counts.sum() > 0
        return null_counts, has_nulls