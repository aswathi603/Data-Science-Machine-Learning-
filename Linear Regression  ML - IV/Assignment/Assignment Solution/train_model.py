import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

class TrainHousingModel:
    """
    Class to load Boston Housing data, train a Linear Regression model with scaling, and save artifacts.
    """

    def __init__(self):
        # Get directory where this file exists
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Correct relative path to train.csv
        self.data_path = os.path.join(self.base_dir, "boston-housing", "train.csv")

        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load data from CSV, detect target column, and return X, y.
        """
        print(" Loading data from:", self.data_path)
        df = pd.read_csv(self.data_path)
        print(" Columns:", df.columns.tolist())

        # Drop unnamed columns (usually index columns)
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]

        # Drop ID column if present (it's an index, not a feature)
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)

        # Detect target column
        if "MEDV" in df.columns:
            target_col = "MEDV"
        elif "medv" in df.columns:
            target_col = "medv"
        elif "target" in df.columns:
            target_col = "target"
        else:
            target_col = df.columns[-1]

        print(" Target column used:", target_col)

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        return X, y

    def train_and_save(self):
        """
        Train the model on scaled data and save model + scaler using pickle.
        """
        X, y = self.load_data()
        
        # Ensure X and y are numpy arrays (no DataFrame remnants)
        X = X.values if hasattr(X, 'values') else X
        y = y.values if hasattr(y, 'values') else y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Save model + scaler using pickle
        with open(os.path.join(self.base_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(self.base_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        
        print(" Model and scaler saved successfully")

if __name__ == "__main__":
    trainer = TrainHousingModel()
    trainer.train_and_save()