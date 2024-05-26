
import time
from cleaner_module import DataCleaner
from ML_prep_module import DataPreparation
from ML_train_module import ModelTraining
from metrics_saver import save_metrics_to_file

def run_marital_status_prediction():
    # Define the path to your dataset
    file_path = 'people.data'
    metrics_file = 'model_metrics.csv'
    sample_size = 5000  # Set sample size to reduce computation time

    # Step 1: Clean the data
    print("Testing DataCleaner...")
    start_time = time.time()
    cleaner = DataCleaner(file_path)
    cleaner.run_cleaning()

    # Get the cleaned data
    cleaned_data = cleaner.get_cleaned_data()
    end_time = time.time()
    print(f"Data cleaning took {end_time - start_time:.2f} seconds.")

    # Step 2: Data Preparation for Marital Status Prediction
    print("\nPreparing Data for Marital Status Prediction...")
    start_time = time.time()
    prep = DataPreparation(cleaned_data, sample_size=sample_size)
    prep.plot_class_distribution(target_column='marital-status')
    X_train, X_test, y_train, y_test = prep.prepare_data(target_column='marital-status', handle_imbalance=True)
    end_time = time.time()
    print(f"Data preparation took {end_time - start_time:.2f} seconds.")

    # Step 3: Train and Evaluate Models for Marital Status Prediction
    print("\nTraining and Evaluating Models for Marital Status Prediction...")
    model_trainer = ModelTraining()
    start_time = time.time()
    trained_models = model_trainer.train_models(X_train, y_train)
    end_time = time.time()
    print(f"Model training took {end_time - start_time:.2f} seconds.")

    # Evaluate each model and store the metrics
    marital_status_metrics = {}
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        start_time = time.time()
        metrics = model_trainer.evaluate_model(model, X_test, y_test)
        end_time = time.time()
        print(f"Evaluation for {name} took {end_time - start_time:.2f} seconds.")
        marital_status_metrics[name] = metrics
        model_trainer.save_model(model, f'{name}_marital_status_model.joblib')

    # Save metrics to file
    save_metrics_to_file(metrics_file, 'Marital Status', marital_status_metrics)

    # Print the collected metrics
    for name, metrics in marital_status_metrics.items():
        print(f"\nMetrics for {name}:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    run_marital_status_prediction()
