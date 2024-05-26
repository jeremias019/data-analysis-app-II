# in this file I will apply the relevant statistics of the cleaned data

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

class DataStatistics:
    
    def __init__(self, data):
        self.data = data
    
    def descriptive_statistics(self):
        stats_summary = []
        print("\nDescriptive Statistics:")
        for col in self.data.select_dtypes(include=[np.number]).columns:
            stats = {
                'Feature': col,
                'Mean': self.data[col].mean(),
                'Median': self.data[col].median(),
                'Standard Deviation': self.data[col].std(),
                'Variance': self.data[col].var(),
                'Minimum': self.data[col].min(),
                'Maximum': self.data[col].max(),
                'Skewness': skew(self.data[col]),
                'Kurtosis': kurtosis(self.data[col])
            }
            stats_summary.append(stats)
            print(f"\nStatistics for {col}:")
            print(f"Mean: {stats['Mean']}")
            print(f"Median: {stats['Median']}")
            print(f"Standard Deviation: {stats['Standard Deviation']}")
            print(f"Variance: {stats['Variance']}")
            print(f"Minimum: {stats['Minimum']}")
            print(f"Maximum: {stats['Maximum']}")
            print(f"Skewness: {stats['Skewness']}")
            print(f"Kurtosis: {stats['Kurtosis']}")
        
        for col in self.data.select_dtypes(include=['object']).columns:
            stats = {
                'Feature': col,
                'Frequencies': self.data[col].value_counts().to_dict()
            }
            stats_summary.append(stats)
            print(f"\nFrequencies for {col}:")
            print(self.data[col].value_counts())
        
        return pd.DataFrame(stats_summary)
    
    def visualize_data_separate_windows(self):
        print("\nVisualizing Data in Separate Windows:")
        
        # Histograms for numerical features
        for col in self.data.select_dtypes(include=[np.number]).columns:
            print(f"Plotting histogram for {col}")
            plt.figure()
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Histogram for {col}")
            plt.show()
        
        # Bar plots and pie charts for categorical features
        for col in self.data.select_dtypes(include=['object']).columns:
            print(f"Plotting bar plot for {col}")
            plt.figure()
            sns.countplot(x=col, data=self.data)
            plt.title(f"Bar plot for {col}")
            plt.show()
            
            print(f"Plotting pie chart for {col}")
            plt.figure()
            self.data[col].value_counts().plot.pie(autopct='%1.1f%%')
            plt.title(f"Pie chart for {col}")
            plt.ylabel('')
            plt.show()
    
    def visualize_data_grouped_windows(self):
        print("\nVisualizing Data Grouped by Type in Separate Windows:")
        
        # Histograms for numerical features
        num_numeric_cols = len(self.data.select_dtypes(include=[np.number]).columns)
        fig, axes = plt.subplots(nrows=num_numeric_cols, ncols=1, figsize=(10, 5 * num_numeric_cols))
        axes = axes.flatten() if num_numeric_cols > 1 else [axes]
        
        for ax, col in zip(axes, self.data.select_dtypes(include=[np.number]).columns):
            print(f"Plotting histogram for {col}")
            sns.histplot(self.data[col], kde=True, ax=ax)
            ax.set_title(f"Histogram for {col}")
        
        plt.tight_layout()
        plt.show()
        
        # Bar plots and pie charts for categorical features
        num_categorical_cols = len(self.data.select_dtypes(include=['object']).columns)
        fig, axes = plt.subplots(nrows=num_categorical_cols, ncols=2, figsize=(15, 5 * num_categorical_cols))
        axes = axes.flatten() if num_categorical_cols > 1 else [axes]
        
        for idx, col in enumerate(self.data.select_dtypes(include=['object']).columns):
            print(f"Plotting bar plot for {col}")
            sns.countplot(x=col, data=self.data, ax=axes[idx * 2])
            axes[idx * 2].set_title(f"Bar plot for {col}")
            
            print(f"Plotting pie chart for {col}")
            self.data[col].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[idx * 2 + 1])
            axes[idx * 2 + 1].set_title(f"Pie chart for {col}")
            axes[idx * 2 + 1].set_ylabel('')
        
        plt.tight_layout()
        plt.show()
    
    def run_statistics(self, separate_windows=False):
        stats_df = self.descriptive_statistics()
        if separate_windows:
            self.visualize_data_separate_windows()
        else:
            self.visualize_data_grouped_windows()
        return stats_df
    
    def save_statistics_to_csv(self, stats_df, filename):
        stats_df.to_csv(filename, index=False)

