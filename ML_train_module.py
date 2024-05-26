
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

class ModelTraining:
    
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        self.scaler = StandardScaler()
    
    def train_models(self, X_train, y_train):
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
        return trained_models
    
    def evaluate_model(self, model, X_test, y_test):
        # Scale the test data
        X_test_scaled = self.scaler.transform(X_test)
        
        y_pred = model.predict(X_test_scaled)
        
        if len(set(y_test)) > 2:  # Multiclass classification
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            else:
                y_prob = model.decision_function(X_test_scaled)
                if y_prob.ndim == 1:
                    y_prob = y_prob[:, np.newaxis]
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        else:  # Binary classification
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                y_prob = model.decision_function(X_test_scaled)
                roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "roc_auc": roc_auc
        }
        
        print(f"\nEvaluation Metrics for {type(model).__name__}:\n", classification_report(y_test, y_pred))
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for {type(model).__name__}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
        return metrics
    
    def save_model(self, model, filename):
        joblib.dump(model, filename)


