import time

from cleaner_module import DataCleaner
from ML_prep_module import DataPreparation
from ML_train_module import ModelTraining
from metrics_saver import save_metrics_to_file

def run_workclass_prediction():
    # Define the path to your dataset
    file_path = 'people.data'
    metrics_file = 'model_metrics.csv'
    sample_size = 5000  # i set sample size to reduce computation time

    # Step 1: Clean the data
    print("Testing DataCleaner...")
    start_time = time.time()
    cleaner = DataCleaner(file_path)
    cleaner.run_cleaning()

    # Get the cleaned data
    cleaned_data = cleaner.get_cleaned_data()
    end_time = time.time()
    print(f"Data cleaning took {end_time - start_time:.2f} seconds.")

    # Step 2: Data Preparation for Workclass Prediction
    print("\nPreparing Data for Workclass Prediction...")
    start_time = time.time()
    prep = DataPreparation(cleaned_data, sample_size=sample_size)
    prep.plot_class_distribution(target_column='workclass')
    X_train, X_test, y_train, y_test = prep.prepare_data(target_column='workclass', handle_imbalance=True)
    end_time = time.time()
    print(f"Data preparation took {end_time - start_time:.2f} seconds.")

    # Step 3: Train and Evaluate Models for Workclass Prediction
    print("\nTraining and Evaluating Models for Workclass Prediction...")
    model_trainer = ModelTraining()
    start_time = time.time()
    trained_models = model_trainer.train_models(X_train, y_train)
    end_time = time.time()
    print(f"Model training took {end_time - start_time:.2f} seconds.")

    # Evaluate each model and store the metrics
    workclass_metrics = {}
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        start_time = time.time()
        metrics = model_trainer.evaluate_model(model, X_test, y_test)
        end_time = time.time()
        print(f"Evaluation for {name} took {end_time - start_time:.2f} seconds.")
        workclass_metrics[name] = metrics
        model_trainer.save_model(model, f'{name}_workclass_model.joblib')

    # Save metrics to file
    save_metrics_to_file(metrics_file, 'Workclass', workclass_metrics)

    # Print the collected metrics
    for name, metrics in workclass_metrics.items():
        print(f"\nMetrics for {name}:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    run_workclass_prediction()

