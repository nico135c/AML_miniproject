from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

class Classifier:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = self.preds = None
        self.load_data()
    
    def load_data(self):
        """
        Load the dataset
        """
        # fetch dataset 
        steel_plates_faults = fetch_ucirepo(id=198) 
        
        # data (as pandas dataframes) 
        self.X = steel_plates_faults.data.features 
        self.y = steel_plates_faults.data.targets 
    
    def split_data(self,test_split=0.2):
        """
        Split the dataset into train and test data (80/20 split by default)
        
        :param test_split: Size of test split
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_split, random_state=42)

    def scale_data(self):
        """
        Standardize features so all variables are on the same scale
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def apply_smote(self):
        """
        Apply SMOTE oversampling to the training data only.
        """
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)

    def _create_report(self):
        """
        Create Classification Report for the Classifier
        """
        # Create figures/MODEL_NAME directory if it doesn't exist
        path = os.path.join("figures", self.name.replace(" ", "_"))
        
        if not os.path.exists(path):
            os.makedirs(path) 
        
        report = classification_report(self.y_test, self.preds)

        # Title text
        title = f"{self.name}\nClassification Report"

        # Split into lines to size the figure
        lines = report.split("\n")
        max_line_length = max(len(line) for line in lines)

        # Dynamic figure size
        fig_width = max(6, max_line_length * 0.12)
        fig_height = len(lines) * 0.27 + 1.2  # extra space for title

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Title at top
        plt.text(0.5, 1.05, title,
                fontsize=14, fontweight="bold",
                ha="center", va="bottom",
                fontfamily="sans-serif",
                transform=plt.gca().transAxes)

        # Report text
        plt.text(0.01, 1.0, report,
                fontsize=10, fontfamily="monospace",
                va="top")

        plt.axis("off")

        filename = f"{self.name.replace(' ', '_')}_classification_report.png"
        plt.savefig(f"{path}/{filename}", dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close()

        print(f"Classification report image saved as: figures/{filename}")

    def _create_cmatrix(self):
        """
        Create Confusion Matrix for the Classifier
        """
        # Create figures directory + model subfolder
        path = os.path.join("figures", self.name.replace(" ", "_"))
        os.makedirs(path, exist_ok=True)

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, self.preds)
        disp = ConfusionMatrixDisplay(cm)

        # Plot with green-yellow colormap
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="YlGn", colorbar=False)

        # Title
        ax.set_title(f"{self.name}\nConfusion Matrix", 
                    fontsize=14, fontweight="bold", pad=20)

        # Save
        filename = f"{self.name.replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(os.path.join(path, filename),
                    dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close()

        print(f"Confusion matrix image saved as: {path}/{filename}")
    
    def _create_feature_importance_plot(self):
        """
        Create and save a Feature Importance bar plot.
        """
        # Check model supports feature_importances_
        if not hasattr(self.model, "feature_importances_"):
            print(f"Model for {self.name} does not support feature importances.")
            return

        # Create figures directory if it doesn't exist
        path = os.path.join("figures", self.name.replace(" ", "_"))
        
        if not os.path.exists(path):
            os.makedirs(path) 

        # Extract importances
        importances = self.model.feature_importances_
        feature_names = list(self.X.columns)

        # Sort importances
        sorted_idx = importances.argsort()
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]

        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(sorted_features, sorted_importances, color="steelblue")
        
        plt.title(f"{self.name}\nFeature Importance", fontsize=14, fontweight="bold")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        filename = f"{self.name.replace(' ', '_')}_feature_importance.png"
        plt.savefig(f"{path}/{filename}", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Feature importance plot saved as: figures/{filename}")

    def create_plots(self):
        self._create_cmatrix()
        self._create_feature_importance_plot()
        self._create_report()


class BinaryClassifier(Classifier):
    def __init__(self):
        super().__init__(name="Binary Classifier")

    def train(self):
        """
        Train Binary Classifier
        """
        print("Training Binary Classifier...")
        
        # Defining fault splits (surface and structural faults)
        surface = ["Z_Scratch", "K_Scratch", "Stains", "Dirtiness"]
        structural = ["Pastry", "Bumps", "Other_Faults"]


        # Create binary labels: 1 for surface faults, 0 for structural faults
        self.y = (self.y[surface].sum(axis=1) > 0).astype(int)

        # Split the data
        self.split_data()

        # Scale the data
        self.scale_data()

        # Fit the model
        self.model = LogisticRegression(class_weight="balanced", max_iter=5000)
        self.model.fit(self.X_train, self.y_train)

        self.preds = self.model.predict(self.X_test)
        print("Binary Classifier training completed.")

class MultiClassClassifier(Classifier):
    def __init__(self):
        super().__init__(name="Multi-Class Classifier")

    def train(self):
        """
        Train Multiclass Classifier
        """
        print("Training Multi-Class Classifier...")
        #Converting multi-label to single-label by taking the index of the maximum fault
        self.y = self.y.idxmax(axis=1)

        # Split the data
        self.split_data()

        # Apply SMOTE
        self.apply_smote()

        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)

        self.preds = self.model.predict(self.X_test)
        print("Multi-Class Classifier training completed.")

        