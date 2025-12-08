from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt


from utils import format_time, save_json


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


def train_linear_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    test_size: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
    hidden_dim: int = 512,
    early_stopping_patience: int = 10
) -> Tuple[LinearClassifier, Dict[str, Any]]:
    print(f"Training linear classifier (dim={features.shape[1]}, classes={num_classes})")
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LinearClassifier(features.shape[1], num_classes, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}/{val_loss:.4f}, Acc: {train_acc:.4f}/{val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    print(f"Training completed in {format_time(training_time)}")
    
    return model, history


def train_knn_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    n_neighbors: int = 5
) -> Tuple[KNeighborsClassifier, float]:
    print(f"Training k-NN (k={n_neighbors})")
    
    start_time = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"k-NN training completed in {format_time(training_time)}")
    
    return knn, training_time

def evaluate_classifier(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    model_type: str = 'sklearn',
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    print("Evaluating classifier...")
    
    start_time = time.time()
    
    if model_type == 'pytorch':
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features).to(device)
            outputs = model(X_tensor)
            _, predictions = outputs.max(1)
            predictions = predictions.cpu().numpy()
    else:
        predictions = model.predict(features)
    
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    cm = confusion_matrix(labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'inference_time': inference_time,
        'predictions': predictions.tolist()
    }
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return metrics


def plot_training_curves(
    history: Dict[str, List],
    output_path: Path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_classification_pipeline(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    device: torch.device,
    output_dir: Path,
    methods: list = ['linear', 'knn']
) -> Dict[str, Any]:
    num_classes = len(class_names)
    results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    if 'linear' in methods:
        print("\n1. Linear Classifier")
        linear_model, history = train_linear_classifier(
            X_train, y_train, num_classes, device
        )
        
        linear_metrics = evaluate_classifier(
            linear_model, X_test, y_test, class_names, 'pytorch', device
        )
        
        plot_training_curves(history, output_dir / 'figures' / 'linear_training_curves.png')
        
        results['linear'] = {
            'metrics': linear_metrics,
            'history': history
        }
    
    if 'knn' in methods:
        print("\n2. k-NN Classifier")
        knn_model, knn_train_time = train_knn_classifier(X_train, y_train)
        
        knn_metrics = evaluate_classifier(
            knn_model, X_test, y_test, class_names, 'sklearn'
        )
        
        results['knn'] = {
            'metrics': knn_metrics,
            'training_time': knn_train_time
        }
    
    
    print("\n4. Comparison")
    
    comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    comparison_data = {}
    
    for method in methods:
        if method in results:
            comparison_data[method] = {
                metric: results[method]['metrics'][metric] for metric in comparison_metrics
            }
    
    fig, axes = plt.subplots(1, len(comparison_metrics), figsize=(18, 5))
    
    for idx, metric in enumerate(comparison_metrics):
        method_names = list(comparison_data.keys())
        values = [comparison_data[method][metric] for method in method_names]
        
        bars = axes[idx].bar(method_names, values, alpha=0.7)
        
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f'{value:.3f}', ha='center', va='bottom')
        
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'classification_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    save_json(results, output_dir / 'results', 'classification_metrics.json')
    
    return results


