"""
Model CNN SIMPLE untuk Face Emotion Recognition
Versi stabil tanpa Spatial Transformer - cocok untuk laptop biasa
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ========================================
# SETTING TRAINING (EDIT DI SINI!)
# ========================================
EPOCHS = 30              # Jumlah epoch (kurangi jadi 20 kalau laptop lemot)
BATCH_SIZE = 32          # Batch size (kurangi jadi 16 kalau RAM kecil)
LEARNING_RATE = 0.001    # Learning rate

print("=" * 70)
print("🤖 CNN Model Builder for Emotion Recognition")
print("=" * 70)
print(f"⚙️  Epochs: {EPOCHS}")
print(f"⚙️  Batch size: {BATCH_SIZE}")
print(f"⚙️  Learning rate: {LEARNING_RATE}")
print("=" * 70)

def build_simple_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Build CNN model sederhana tapi efektif
    Tanpa Spatial Transformer untuk stabilitas
    """
    print("\n📐 Building CNN model...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    print("✅ Model built successfully!")
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train model dengan callbacks
    """
    # Convert labels ke categorical
    print("\n📊 Converting labels to categorical...")
    y_train_cat = keras.utils.to_categorical(y_train, 7)
    y_test_cat = keras.utils.to_categorical(y_test, 7)
    
    # Compile model
    print("⚙️  Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Training
    print("\n" + "=" * 70)
    print("🚀 Starting training...")
    print("=" * 70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    print(f"Steps per epoch: {len(X_train) // BATCH_SIZE}")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training history
    """
    print("\n📈 Creating training visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Add max accuracy text
    max_val_acc = max(history.history['val_accuracy'])
    max_epoch = history.history['val_accuracy'].index(max_val_acc)
    axes[0].axhline(y=max_val_acc, color='r', linestyle='--', alpha=0.5)
    axes[0].text(max_epoch, max_val_acc, f'  Max: {max_val_acc:.4f}', 
                fontsize=10, color='red', fontweight='bold')
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Add min loss text
    min_val_loss = min(history.history['val_loss'])
    min_epoch = history.history['val_loss'].index(min_val_loss)
    axes[1].axhline(y=min_val_loss, color='r', linestyle='--', alpha=0.5)
    axes[1].text(min_epoch, min_val_loss, f'  Min: {min_val_loss:.4f}', 
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved as 'training_history.png'")
    plt.close()

def main():
    """
    Main function
    """
    # Check processed data
    if not os.path.exists('processed_data'):
        print("\n❌ ERROR: Folder 'processed_data' tidak ditemukan!")
        print("Jalankan dulu: python 1_prepare_dataset_LITE.py")
        return
    
    # Load processed data
    print("\n📂 Loading processed data...")
    try:
        X_train = np.load('processed_data/X_train.npy')
        X_test = np.load('processed_data/X_test.npy')
        y_train = np.load('processed_data/y_train.npy')
        y_test = np.load('processed_data/y_test.npy')
        
        print(f"   ✓ Training data: {X_train.shape}")
        print(f"   ✓ Testing data: {X_test.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Build model
    model = build_simple_cnn_model()
    
    # Print model summary
    print("\n" + "=" * 70)
    print("📋 MODEL ARCHITECTURE")
    print("=" * 70)
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\n💾 Total parameters: {total_params:,}")
    print("=" * 70)
    
    # Train model
    try:
        history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Plot results
        plot_training_history(history)
        
        # Final evaluation
        print("\n" + "=" * 70)
        print("🎯 FINAL EVALUATION")
        print("=" * 70)
        
        y_test_cat = keras.utils.to_categorical(y_test, 7)
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
        print(f"✅ Test Loss: {test_loss:.4f}")
        print("=" * 70)
        
        print("\n💾 Model saved as 'best_model.keras'")
        print("\n▶️  Selanjutnya jalankan:")
        print("   - Evaluasi: python 4_evaluate_model.py")
        print("   - Real-time: python 3_realtime_detection.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Model terakhir disimpan sebagai 'best_model.keras'")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()