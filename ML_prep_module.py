import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

class DataPreparation:
    
    def __init__(self, data, sample_size=None):
        self.data = data.sample(n=sample_size) if sample_size else data
    
    def plot_class_distribution(self, target_column):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_column, data=self.data)
        plt.title(f"Class Distribution for {target_column}")
        plt.show()
    
    def encode_categorical_features(self):
        label_encoders = {}
        for column in self.data.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            self.data[column] = label_encoders[column].fit_transform(self.data[column])
        return self.data
    
    def handle_class_imbalance(self, X, y):
        min_class_count = y.value_counts().min()
        if min_class_count <= 1:
            print("Skipping SMOTE due to insufficient samples in some classes.")
            return X, y
        
        k_neighbors = min(5, max(1, min_class_count - 1))  # Ensure k_neighbors is at least 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def prepare_data(self, target_column, test_size=0.2, random_state=42, handle_imbalance=False):
        self.data = self.encode_categorical_features()
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        if handle_imbalance:
            X, y = self.handle_class_imbalance(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
