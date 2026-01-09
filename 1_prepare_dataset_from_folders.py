"""
Script LITE untuk mempersiapkan dataset FER2013 dengan jumlah gambar yang dibatasi
COCOK untuk laptop dengan spek terbatas!
"""

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Definisi label emosi
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ========================================
# SETTING JUMLAH GAMBAR (EDIT DI SINI!)
# ========================================
MAX_IMAGES_PER_EMOTION_TRAIN = 7215  # Maksimal gambar per emosi untuk training
MAX_IMAGES_PER_EMOTION_TEST = 1800   # Maksimal gambar per emosi untuk testing

# Kalau mau lebih ringan lagi, ubah jadi:
# MAX_IMAGES_PER_EMOTION_TRAIN = 300
# MAX_IMAGES_PER_EMOTION_TEST = 100

# Atau kalau mau SUPER LITE (cepet tapi akurasi turun):
# MAX_IMAGES_PER_EMOTION_TRAIN = 200
# MAX_IMAGES_PER_EMOTION_TEST = 50

print("=" * 70)
print("🚀 FER2013 LITE - Dataset Preparation")
print("=" * 70)
print(f"📦 Max images per emotion (train): {MAX_IMAGES_PER_EMOTION_TRAIN}")
print(f"📦 Max images per emotion (test):  {MAX_IMAGES_PER_EMOTION_TEST}")
print(f"💾 Estimated total: ~{MAX_IMAGES_PER_EMOTION_TRAIN * 7 + MAX_IMAGES_PER_EMOTION_TEST * 7} images")
print("=" * 70)

def load_images_from_folder(base_folder, max_per_emotion):
    """
    Load images dari struktur folder dengan batasan jumlah
    """
    images = []
    labels = []
    
    print(f"\nLoading images from {base_folder}...")
    print(f"Max per emotion: {max_per_emotion}")
    
    total_skipped = 0
    
    for emotion_idx, emotion in enumerate(emotion_labels):
        emotion_folder = os.path.join(base_folder, emotion)
        
        if not os.path.exists(emotion_folder):
            print(f"  ⚠️  {emotion}: Folder not found!")
            continue
        
        image_files = [f for f in os.listdir(emotion_folder) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        total_available = len(image_files)
        
        # Batasi jumlah gambar
        if total_available > max_per_emotion:
            image_files = image_files[:max_per_emotion]
            skipped = total_available - max_per_emotion
            total_skipped += skipped
            print(f"  📁 {emotion:10s}: Using {max_per_emotion}/{total_available} (skipped {skipped})")
        else:
            print(f"  📁 {emotion:10s}: Using {total_available}/{total_available}")
        
        for img_file in image_files:
            img_path = os.path.join(emotion_folder, img_file)
            
            # Load image as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize ke 48x48 jika belum
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(emotion_idx)
    
    if total_skipped > 0:
        print(f"  ✂️  Total skipped: {total_skipped} images")
    
    return np.array(images), np.array(labels)

def load_fer2013_lite(train_folder='train', test_folder='test'):
    """
    Load dataset dari folder structure dengan batasan
    """
    print("\n" + "=" * 70)
    print("📂 LOADING FER2013 DATASET (LITE VERSION)")
    print("=" * 70)
    
    # Load training data
    print("\n1️⃣  Loading training data...")
    X_train, y_train = load_images_from_folder(train_folder, MAX_IMAGES_PER_EMOTION_TRAIN)
    
    # Load testing data
    print("\n2️⃣  Loading testing data...")
    X_test, y_test = load_images_from_folder(test_folder, MAX_IMAGES_PER_EMOTION_TEST)
    
    print("\n" + "=" * 70)
    print("📊 DATASET SUMMARY")
    print("=" * 70)
    print(f"✓ Training images: {len(X_train):,}")
    print(f"✓ Testing images:  {len(X_test):,}")
    print(f"✓ Total images:    {len(X_train) + len(X_test):,}")
    print(f"✓ Image shape:     {X_train[0].shape}")
    
    # Estimasi ukuran memory
    train_size_mb = (X_train.nbytes) / (1024 * 1024)
    test_size_mb = (X_test.nbytes) / (1024 * 1024)
    total_size_mb = train_size_mb + test_size_mb
    
    print(f"\n💾 Memory usage:")
    print(f"   Training data: ~{train_size_mb:.1f} MB")
    print(f"   Testing data:  ~{test_size_mb:.1f} MB")
    print(f"   Total:         ~{total_size_mb:.1f} MB")
    
    print("\n📈 Emotion distribution in training set:")
    for i, label in enumerate(emotion_labels):
        count = np.sum(y_train == i)
        percentage = (count/len(y_train)*100) if len(y_train) > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"   {label.capitalize():10s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    print("\n📈 Emotion distribution in test set:")
    for i, label in enumerate(emotion_labels):
        count = np.sum(y_test == i)
        percentage = (count/len(y_test)*100) if len(y_test) > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"   {label.capitalize():10s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    """
    Preprocessing data untuk training
    """
    print("\n" + "=" * 70)
    print("⚙️  PREPROCESSING DATA")
    print("=" * 70)
    
    # Normalize pixel values ke range [0, 1]
    print("   Normalizing pixel values...")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape untuk CNN (tambah channel dimension)
    print("   Reshaping for CNN...")
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    
    print(f"   ✓ Training shape: {X_train.shape}")
    print(f"   ✓ Testing shape:  {X_test.shape}")
    
    return X_train, X_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir='processed_data'):
    """
    Save processed data ke file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "=" * 70)
    print("💾 SAVING PROCESSED DATA")
    print("=" * 70)
    
    print("   Saving to numpy files...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Hitung ukuran file
    total_size = 0
    for fname in ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']:
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath) / (1024 * 1024)
        total_size += size
        print(f"   ✓ {fname:15s}: {size:6.2f} MB")
    
    print(f"   📦 Total size: {total_size:.2f} MB")
    print(f"   📁 Location: {output_dir}/")

def visualize_samples(X_train, y_train, num_samples=10):
    """
    Visualisasi beberapa sample dari dataset
    """
    print("\n" + "=" * 70)
    print("🎨 CREATING SAMPLE VISUALIZATION")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images from FER2013 (LITE)', fontsize=16, fontweight='bold')
    
    # Pilih random samples
    indices = np.random.choice(len(X_train), min(num_samples, len(X_train)), replace=False)
    
    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        img = X_train[idx].squeeze()
        emotion = emotion_labels[y_train[idx]]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(emotion.capitalize(), fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples_lite.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved as 'dataset_samples_lite.png'")
    plt.close()

def visualize_distribution(y_train, y_test):
    """
    Visualisasi distribusi kelas
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set distribution
    train_counts = [np.sum(y_train == i) for i in range(len(emotion_labels))]
    colors_train = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    bars1 = ax1.bar(emotion_labels, train_counts, color=colors_train, alpha=0.8)
    ax1.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Images', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, count) in enumerate(zip(bars1, train_counts)):
        ax1.text(i, count + 10, str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Test set distribution
    test_counts = [np.sum(y_test == i) for i in range(len(emotion_labels))]
    bars2 = ax2.bar(emotion_labels, test_counts, color=colors_train, alpha=0.8)
    ax2.set_title('Test Set Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Images', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, count) in enumerate(zip(bars2, test_counts)):
        ax2.text(i, count + 3, str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('class_distribution_lite.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved as 'class_distribution_lite.png'")
    plt.close()

if __name__ == "__main__":
    # Cek struktur folder
    if not os.path.exists('train') or not os.path.exists('test'):
        print("\n❌ ERROR: Folder 'train' atau 'test' tidak ditemukan!")
        print("\nPastikan struktur folder seperti ini:")
        print("FaceEmotion/")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── disgust/")
        print("│   └── ...")
        print("└── test/")
        print("    ├── angry/")
        print("    └── ...")
        exit(1)
    
    print("\n" + "=" * 70)
    print("⚡ Starting dataset preparation (LITE version)...")
    print("=" * 70)
    
    # Load dataset
    X_train, X_test, y_train, y_test = load_fer2013_lite('train', 'test')
    
    # Visualize samples
    visualize_samples(X_train, y_train)
    
    # Visualize distribution
    visualize_distribution(y_train, y_test)
    
    # Preprocess
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # Save
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETED!")
    print("=" * 70)
    print("\n💡 Tips untuk training:")
    print("   - Kalau masih berat, edit MAX_IMAGES_PER_EMOTION jadi lebih kecil")
    print("   - Kurangi epochs di 2_build_model.py jadi 25-30")
    print("   - Kurangi batch_size jadi 32 atau 16")
    print("\n▶️  Selanjutnya jalankan: python 2_build_model.py")
    print("=" * 70)