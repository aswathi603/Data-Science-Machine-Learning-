import pickle
import numpy as np
import os

class predict_housing_medv:
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        # Load the trained model and scaler using pickle
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, model_path), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(base_dir, scaler_path), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Feature names and their min-max ranges (updated to match your CSV: lowercase, no 'ID', 'B' as 'black')
        self.features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
        self.ranges = {
            'crim': (0.00632, 88.9762),
            'zn': (0.0, 100.0),
            'indus': (0.46, 27.74),
            'chas': (0.0, 1.0),
            'nox': (0.385, 0.871),
            'rm': (3.561, 8.78),
            'age': (2.9, 100.0),
            'dis': (1.1296, 12.1265),
            'rad': (1.0, 24.0),
            'tax': (187.0, 711.0),
            'ptratio': (12.6, 22.0),
            'black': (0.32, 396.9),  # 'B' is 'black' in your CSV
            'lstat': (1.73, 37.97)
        }

    def predict(self, features):
        """
        Predict MEDV given a list of feature values (scales inputs first).
        :param features: List of 13 float values for the features.
        :return: Predicted MEDV (float).
        """
        if len(features) != 13:
            raise ValueError("Exactly 13 feature values required.")
        # Scale using numpy array (matches how scaler was fitted)
        scaled_features = self.scaler.transform(np.array(features).reshape(1, -1))
        prediction = self.model.predict(scaled_features)
        return prediction[0]
    
    def get_feature_importance(self):
        """
        Returns feature importance as a dictionary
        """
        return dict(zip(self.features, self.model.coef_))


    def run_terminal(self):
        print("🏠 Boston Housing MEDV Predictor")
        print("Enter feature values within the specified ranges.\n")

        while True:
            inputs = []
            try:
                for feature in self.features:
                    min_val, max_val = self.ranges[feature]
                    value = float(
                        input(f"Enter {feature} ({min_val} - {max_val}): ")
                    )

                    if not (min_val <= value <= max_val):
                        raise ValueError(f"{feature} out of range!")

                    inputs.append(value)

                # Predict once
                prediction = self.predict(inputs)
                print(f"\n✅ Predicted MEDV: ${prediction:.2f} (in thousands)\n")

                # Ask user if they want to continue
                choice = input("Do you want to predict again? (y/n): ").lower()
                if choice != 'y':
                    print("👋 Exiting predictor. Thank you!")
                    break

            except ValueError as e:
                print(f"❌ Invalid input: {e}\n")


# Example usage for terminal (uncomment to run directly)
if __name__ == "__main__":
    predictor = predict_housing_medv()
    predictor.run_terminal()