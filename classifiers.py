from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

class Classifier:
    def __init__(self, name):
        self.name = name
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = self.preds = None
        self.load_data()
    
    def load_data(self):
        # fetch dataset 
        steel_plates_faults = fetch_ucirepo(id=198) 
        
        # data (as pandas dataframes) 
        self.X = steel_plates_faults.data.features 
        self.y = steel_plates_faults.data.targets 
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def get_report(self):
        # Create figures directory if it doesn't exist
        if not os.path.exists("figures"):
            os.makedirs("figures") 
        
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
        plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close()

        print(f"Classification report image saved as: figures/{filename}")

    def get_cmatrix(self):
        # Create figures directory if it doesn't exist
        if not os.path.exists("figures"):
            os.makedirs("figures") 
            
        cm = confusion_matrix(self.y_test, self.preds)
        disp = ConfusionMatrixDisplay(cm)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        # Title
        ax.set_title(f"{self.name}\nConfusion Matrix", fontsize=14, fontweight="bold", pad=20)

        filename = f"{self.name.replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close()

        print(f"Confusion matrix image saved as: figures/{filename}")

class BinaryClassifier(Classifier):
    def __init__(self):
        super().__init__(name="Binary Classifier")

    def train(self):
        print("Training Binary Classifier...")
        # Defining fault splits (surface faults and structural faults)
        surface = ["Z_Scratch", "K_Scratch", "Stains", "Dirtiness"]
        structural = ["Pastry", "Bumps", "Other_Faults"]

        # Create binary labels: 1 for surface faults, 0 for structural faults
        self.y = (self.y[surface].sum(axis=1) > 0).astype(int)

        # Split the data
        self.split_data()

        model = LogisticRegression(max_iter=500)
        model.fit(self.X_train, self.y_train)

        self.preds = model.predict(self.X_test)
        print("Binary Classifier training completed.")

class MultiClassClassifier(Classifier):
    def __init__(self):
        super().__init__(name="Multi-Class Classifier")

    def train(self):
        print("Training Multi-Class Classifier...")
        #Converting multi-label to single-label by taking the index of the maximum fault
        self.y = self.y.idxmax(axis=1)

        # Split the data
        self.split_data()

        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)

        self.preds = model.predict(self.X_test)
        print("Multi-Class Classifier training completed.")

        