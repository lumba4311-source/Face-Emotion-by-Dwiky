"""
Evaluasi dan visualisasi performa model
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_data_and_model():
    """
    Load test data dan trained model
    """
    print("Loading data and model...")
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    model = keras.models.load_model('best_model.keras')
    
    return X_test, y_test, model

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

def plot_per_class_accuracy(y_true, y_pred):
    """
    Plot akurasi per kelas emosi
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotion_labels, per_class_acc * 100, color='steelblue')
    
    # Warnai bar berdasarkan performa
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        if acc >= 0.7:
            bar.set_color('green')
        elif acc >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        
        # Tambahkan nilai di atas bar
        plt.text(i, acc * 100 + 1, f'{acc*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300)
    print("Per-class accuracy plot saved as 'per_class_accuracy.png'")
    plt.show()
    
    return per_class_acc

def plot_sample_predictions(X_test, y_true, y_pred, num_samples=20):
    """
    Visualisasi prediksi pada sample test data
    """
    # Pilih sample secara random
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    
    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        img = X_test[idx].squeeze()
        true_label = emotion_labels[y_true[idx]]
        pred_label = emotion_labels[y_pred[idx]]
        
        ax.imshow(img, cmap='gray')
        
        # Warna berdasarkan benar/salah
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        title = f'True: {true_label}\nPred: {pred_label}'
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300)
    print("Sample predictions saved as 'sample_predictions.png'")
    plt.show()

def plot_misclassified_examples(X_test, y_true, y_pred, num_examples=10):
    """
    Visualisasi contoh yang salah diklasifikasi
    """
    # Cari indices yang salah
    misclassified = np.where(y_true != y_pred)[0]
    
    if len(misclassified) == 0:
        print("No misclassified examples found!")
        return
    
    # Pilih sample
    num_examples = min(num_examples, len(misclassified))
    indices = np.random.choice(misclassified, num_examples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
    
    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        if i >= num_examples:
            ax.axis('off')
            continue
        
        img = X_test[idx].squeeze()
        true_label = emotion_labels[y_true[idx]]
        pred_label = emotion_labels[y_pred[idx]]
        
        ax.imshow(img, cmap='gray')
        title = f'True: {true_label}\nPred: {pred_label}'
        ax.set_title(title, fontsize=9, color='red', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=300)
    print("Misclassified examples saved as 'misclassified_examples.png'")
    plt.show()

def generate_classification_report(y_true, y_pred):
    """
    Generate dan simpan classification report
    """
    report = classification_report(y_true, y_pred,
                                   target_names=emotion_labels,
                                   digits=4)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    # Save ke file
    with open('classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    print("\nClassification report saved as 'classification_report.txt'")
    
    # Create DataFrame untuk visualisasi
    report_dict = classification_report(y_true, y_pred,
                                       target_names=emotion_labels,
                                       output_dict=True)
    
    df = pd.DataFrame(report_dict).transpose()
    df = df.round(4)
    
    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['precision', 'recall', 'f1-score']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for ax, metric, color in zip(axes, metrics, colors):
        data = df.loc[emotion_labels, metric] * 100
        bars = ax.bar(emotion_labels, data, color=color, alpha=0.7)
        
        # Tambahkan nilai
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300)
    print("Metrics comparison saved as 'metrics_comparison.png'")
    plt.show()

def evaluate_model():
    """
    Main evaluation function
    """
    # Load data dan model
    X_test, y_test, model = load_data_and_model()
    
    # Predict
    print("\nMaking predictions on test set...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # 2. Per-class accuracy
    per_class_acc = plot_per_class_accuracy(y_test, y_pred)
    
    # 3. Classification report
    generate_classification_report(y_test, y_pred)
    
    # 4. Sample predictions
    plot_sample_predictions(X_test, y_test, y_pred)
    
    # 5. Misclassified examples
    plot_misclassified_examples(X_test, y_test, y_pred)
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total test samples: {len(y_test)}")
    print(f"Correct predictions: {np.sum(y_pred == y_test)}")
    print(f"Wrong predictions: {np.sum(y_pred != y_test)}")
    print(f"Overall accuracy: {accuracy*100:.2f}%")
    print(f"\nPer-class accuracy:")
    for label, acc in zip(emotion_labels, per_class_acc):
        print(f"  {label:10s}: {acc*100:.2f}%")
    print(f"{'='*60}")
    
    print("\nAll evaluation results have been saved!")

if __name__ == "__main__":
    evaluate_model()