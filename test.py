# test.py

import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dataset import CustomDataset
from model import CNN_BiGRU
from config import GESTURE

class TestConfig:
    def __init__(self):
        self.window_size = 30
        self.batch_size = 64
        self.dataset_dir = './test_data/'
        self.model_path = './model/model.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device):
    """Load the trained model"""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model with various metrics"""
    all_predictions = []
    all_labels = []
    total_latency = 0
    num_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_batch = y_batch.type(torch.LongTensor).to(device).squeeze()
            x_batch = x_batch.to(device)

            # Measure inference time
            start_time = time.time()
            pred = model(x_batch)
            end_time = time.time()

            # Calculate latency
            batch_latency = (end_time - start_time) * 1000  # Convert to milliseconds
            total_latency += batch_latency
            num_samples += x_batch.size(0)

            # Get predictions
            predictions = pred.argmax(dim=1).cpu().numpy()
            labels = y_batch.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Calculate false positive rate for each class
    fp_rates = []
    for i in range(len(GESTURE)):
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fp_rates.append(fp_rate)

    # Calculate average latency per sample
    avg_latency = total_latency / num_samples

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'false_positive_rates': fp_rates,
        'average_latency_ms': avg_latency
    }

def print_results(results):
    """Print evaluation results in a formatted way"""
    print("\n=== Model Evaluation Results ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Average Latency: {results['average_latency_ms']:.2f} ms per sample")

    print("\nFalse Positive Rates per class:")
    for i, gesture in enumerate(GESTURE):
        print(f"{gesture}: {results['false_positive_rates'][i]:.4f}")

    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

def main():
    # Initialize configuration
    config = TestConfig()

    # Load test data
    test_dataset = CustomDataset(window_size=config.window_size, folder_dir=config.dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load model
    model = load_model(config.model_path, config.device)

    # Evaluate model
    results = evaluate_model(model, test_loader, config.device)

    # Print results
    print_results(results)

if __name__ == "__main__":
    main()