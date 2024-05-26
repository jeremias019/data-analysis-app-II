import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
from cleaner_module import DataCleaner
from statistics_module import DataStatistics
from ML_prep_module import DataPreparation
from metrics_saver import save_metrics_to_file

# Import the test functions
from income_pred_test import run_income_prediction
from marital_pred_test import run_marital_status_prediction
from workclass_pred_test import run_workclass_prediction

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processing and Analysis GUI")
        
        self.file_path = ""
        self.cleaned_data = None
        self.result_queue = queue.Queue()
        
        # Step 1: Select and load the dataset
        self.select_button = tk.Button(root, text="Select Dataset", command=self.select_dataset)
        self.select_button.pack(pady=10)
        
        # Step 2: Clean the data
        self.clean_button = tk.Button(root, text="Clean Data", command=self.clean_data)
        self.clean_button.pack(pady=10)
        
        # Step 3: Data analysis
        self.analysis_button = tk.Button(root, text="Data Analysis", command=self.open_analysis_window)
        self.analysis_button.pack(pady=10)
        
        # Step 4: Run ML models
        self.ml_button = tk.Button(root, text="Run ML Models", command=self.open_ml_window)
        self.ml_button.pack(pady=10)
        
        # Exit button
        self.exit_button = tk.Button(root, text="Exit", command=self.exit_application)
        self.exit_button.pack(pady=10)
        
    def select_dataset(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if self.file_path:
            messagebox.showinfo("Dataset Selected", f"Selected dataset: {self.file_path}")
    
    def clean_data(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a dataset first.")
            return
        cleaner = DataCleaner(self.file_path)
        cleaner.run_cleaning()
        self.cleaned_data = cleaner.get_cleaned_data()
        messagebox.showinfo("Cleaning Completed", "Data cleaning is completed.")
    
    def open_analysis_window(self):
        if self.cleaned_data is None:
            messagebox.showerror("Error", "Please clean the data first.")
            return
        
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title("Data Analysis")
        
        self.num_var_label = tk.Label(self.analysis_window, text="Select Numerical Variables:")
        self.num_var_label.pack(pady=5)
        
        self.num_listbox = tk.Listbox(self.analysis_window, selectmode=tk.MULTIPLE, exportselection=0)
        for col in self.cleaned_data.select_dtypes(include=['number']).columns:
            self.num_listbox.insert(tk.END, col)
        self.num_listbox.pack(pady=5)
        
        self.cat_var_label = tk.Label(self.analysis_window, text="Select Categorical Variables:")
        self.cat_var_label.pack(pady=5)
        
        self.cat_listbox = tk.Listbox(self.analysis_window, selectmode=tk.MULTIPLE, exportselection=0)
        for col in self.cleaned_data.select_dtypes(include=['object']).columns:
            self.cat_listbox.insert(tk.END, col)
        self.cat_listbox.pack(pady=5)
        
        self.plot_button = tk.Button(self.analysis_window, text="Get Plots and Statistics", command=self.get_plots_and_statistics)
        self.plot_button.pack(pady=10)
    
    def get_plots_and_statistics(self):
        num_vars = [self.num_listbox.get(i) for i in self.num_listbox.curselection()]
        cat_vars = [self.cat_listbox.get(i) for i in self.cat_listbox.curselection()]
        
        if len(num_vars) != 2 or len(cat_vars) != 2:
            messagebox.showerror("Error", "Please select exactly two numerical and two categorical variables.")
            return
        
        self.root.after(100, self.run_data_statistics, num_vars, cat_vars)
    
    def run_data_statistics(self, num_vars, cat_vars):
        prep = DataPreparation(self.cleaned_data)
        selected_data = self.cleaned_data[num_vars + cat_vars]
        selected_data.to_csv("selected_data.csv", index=False)
        
        stats_module = DataStatistics(selected_data)
        stats_df = stats_module.run_statistics()
        stats_df.to_csv("data_statistics.csv", index=False)
        
        messagebox.showinfo("Task Completed", "Plots and statistics generated. Check 'data_statistics.csv'.")
    
    def open_ml_window(self):
        if self.cleaned_data is None:
            messagebox.showerror("Error", "Please clean the data first.")
            return
        
        self.ml_window = tk.Toplevel(self.root)
        self.ml_window.title("Run ML Models")
        
        self.income_button = tk.Button(self.ml_window, text="Run Income Prediction", command=lambda: self.run_ml_model(run_income_prediction))
        self.income_button.pack(pady=5)
        
        self.marital_status_button = tk.Button(self.ml_window, text="Run Marital Status Prediction", command=lambda: self.run_ml_model(run_marital_status_prediction))
        self.marital_status_button.pack(pady=5)
        
        self.workclass_button = tk.Button(self.ml_window, text="Run Workclass Prediction", command=lambda: self.run_ml_model(run_workclass_prediction))
        self.workclass_button.pack(pady=5)
    
    def run_ml_model(self, model_function):
        thread = threading.Thread(target=self.run_ml_model_thread, args=(model_function,))
        thread.start()
        self.root.after(100, self.process_queue)
    
    def run_ml_model_thread(self, model_function):
        try:
            model_function()
            self.result_queue.put("Task Completed. Check model_metrics.csv for details.")
        except Exception as e:
            self.result_queue.put(f"Error: {e}")
    
    def process_queue(self):
        try:
            result = self.result_queue.get_nowait()
            messagebox.showinfo("Task Completed", result)
        except queue.Empty:
            self.root.after(100, self.process_queue)

    def exit_application(self):
        self.root.quit()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

