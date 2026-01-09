# Face Emotion Recognition dengan OpenCV dan CNN

Project ini mengimplementasikan sistem deteksi emosi wajah menggunakan:
- Dataset: FER2013 dari Kaggle
- Model: CNN dengan Spatial Transformer Network
- Real-time Detection: OpenCV untuk deteksi wajah dan inference

## 📋 Daftar Emosi yang Dikenali

1. Angry (Marah)
2. Disgust (Jijik)
3. Fear (Takut)
4. Happy (Senang)
5. Sad (Sedih)
6. Surprise (Terkejut)
7. Neutral (Netral)

## 🛠️ Requirements

### Software
- Python 3.8 atau lebih baru
- Webcam (untuk real-time detection)

### Libraries
```bash
pip install -r requirements.txt
```

Atau install manual:
```bash
pip install opencv-python numpy pandas tensorflow keras matplotlib scikit-learn seaborn
```

## 📦 Download Dataset

1. Download dataset FER2013 dari Kaggle:
   https://www.kaggle.com/datasets/msambare/fer2013

2. Extract dan letakkan file `fer2013.csv` di folder project

## 🚀 Cara Menggunakan

### 1. Persiapan Dataset
```bash
python 1_prepare_dataset.py
```
Script ini akan:
- Load dataset dari CSV
- Preprocessing dan normalisasi
- Split data training dan testing
- Visualisasi sample data
- Save processed data ke folder `processed_data/`

### 2. Training Model
```bash
python 2_build_model.py
```
Script ini akan:
- Build CNN model dengan Spatial Transformer Network
- Training model dengan callbacks (early stopping, learning rate reduction)
- Save best model sebagai `best_model.keras`
- Plot training history

**Catatan**: Training memerlukan waktu (tergantung GPU/CPU). Dengan GPU, sekitar 1-2 jam untuk 50 epochs.

### 3. Evaluasi Model
```bash
python 4_evaluate_model.py
```
Script ini akan generate:
- Confusion matrix
- Per-class accuracy
- Classification report
- Sample predictions
- Misclassified examples

### 4. Real-time Detection
```bash
python 3_realtime_detection.py
```

**Kontrol:**
- Pilih mode 1 untuk webcam real-time
- Pilih mode 2 untuk deteksi dari file gambar
- Tekan 'q' untuk quit
- Tekan 's' untuk toggle tampilan probability

## 📊 Arsitektur Model

### CNN Architecture:
```
Input (48x48x1)
↓
Spatial Transformer Network (Optional)
↓
Conv2D (64) + BatchNorm + Conv2D (64) + BatchNorm + MaxPool + Dropout
↓
Conv2D (128) + BatchNorm + Conv2D (128) + BatchNorm + MaxPool + Dropout
↓
Conv2D (256) + BatchNorm + Conv2D (256) + BatchNorm + MaxPool + Dropout
↓
Conv2D (512) + BatchNorm + Conv2D (512) + BatchNorm + MaxPool + Dropout
↓
Flatten
↓
Dense (512) + BatchNorm + Dropout
↓
Dense (256) + BatchNorm + Dropout
↓
Dense (7, softmax)
```

### Key Features:
- **Spatial Transformer Network**: Membantu model fokus pada bagian penting wajah
- **Batch Normalization**: Stabilkan training
- **Dropout**: Mencegah overfitting
- **Data Augmentation**: Random rotation dan zoom

## 📈 Expected Performance

Dengan training yang proper, model diharapkan mencapai:
- Overall Accuracy: 60-70%
- Happy emotion: ~80-85% (paling mudah dideteksi)
- Disgust emotion: ~50-60% (paling sulit, karena sample sedikit)

## 📁 Struktur File

```
project/
│
├── 1_prepare_dataset.py       # Preprocessing dataset
├── 2_build_model.py            # Build dan training model
├── 3_realtime_detection.py    # Real-time detection dengan OpenCV
├── 4_evaluate_model.py         # Evaluasi dan visualisasi
├── README.md                   # Dokumentasi ini
├── requirements.txt            # Dependencies
│
├── fer2013.csv                 # Dataset (download dari Kaggle)
├── best_model.keras            # Trained model (generated)
│
├── processed_data/             # Processed dataset (generated)
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
│
└── output/                     # Hasil visualisasi (generated)
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    ├── training_history.png
    └── sample_predictions.png
```

## 🎯 Tips untuk Hasil Lebih Baik

1. **Training lebih lama**: Coba epochs 75-100
2. **Learning rate scheduling**: Adjust learning rate callback
3. **Data augmentation**: Tambah variasi augmentasi
4. **Ensemble methods**: Kombinasikan beberapa model
5. **Focal loss**: Untuk handle imbalanced dataset

## 🐛 Troubleshooting

### Model tidak bisa load
```python
# Gunakan legacy optimizer
from tensorflow.keras.optimizers.legacy import Adam
```

### Webcam tidak terdeteksi
- Coba ganti `camera_index` dari 0 ke 1 atau 2
- Pastikan permission webcam sudah diizinkan

### Out of Memory saat training
- Kurangi batch_size dari 64 ke 32 atau 16
- Kurangi ukuran model (jumlah filters)

## 📚 Referensi

1. FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
2. Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
3. OpenCV Documentation: https://docs.opencv.org/

## 👨‍💻 Untuk Review Jurnal

### Point-point yang bisa di-highlight:

1. **Dataset**: FER2013 dengan 35,887 gambar grayscale 48x48
2. **Preprocessing**: Normalisasi, data augmentation
3. **Architecture**: CNN dengan STN untuk invariance terhadap transformasi geometrik
4. **Evaluation metrics**: Accuracy, precision, recall, F1-score, confusion matrix
5. **Real-time application**: Implementasi dengan OpenCV untuk praktis usage

### Kelebihan approach ini:
- STN membantu handle variasi pose dan alignment wajah
- Batch normalization dan dropout untuk generalisasi lebih baik
- Real-time capable dengan webcam

### Limitasi:
- Dataset FER2013 memiliki noise dan label yang tidak sempurna
- Imbalanced classes (Disgust sangat sedikit)
- Performa turun pada kondisi lighting ekstrem

## 📝 License

Project ini untuk tujuan edukasi dan review jurnal.

---
**Dibuat untuk tugas review jurnal deteksi emosi wajah**