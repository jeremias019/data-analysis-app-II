import csv

def save_metrics_to_file(filename, task, metrics):
    header = ['Task', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writerow(header)
        
        for model_name, metric_values in metrics.items():
            writer.writerow([task, model_name, 
                             metric_values['accuracy'], 
                             metric_values['precision'], 
                             metric_values['recall'], 
                             metric_values['f1'], 
                             metric_values['roc_auc']])
