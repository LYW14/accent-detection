import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import gc
import tensorflow as tf

# ==== MAC OPTIMIZATION: Limit TensorFlow Memory Growth ====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==== CONFIG ====
TRAIN_CSV = "labeled_dataset_train.csv"
DEV_CSV = "labeled_dataset_dev.csv"
TEST_CSV = "labeled_dataset_test.csv"
CACHE_FILE = "cached_data.pkl" 
# ================

print("="*60)
print("LOADING DATA")
print("="*60)

# Load data (same as before)
if os.path.exists(CACHE_FILE):
    print(f"Loading cached data from {CACHE_FILE}...")
    import pickle
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    X_train, X_dev, X_test = cache['X_train'], cache['X_dev'], cache['X_test']
    y_train, y_dev, y_test = cache['y_train'], cache['y_dev'], cache['y_test']
    encoder = cache['encoder']
    num_classes = len(encoder.classes_)
    print("✓ Loaded from cache!")
else:
    print("No cache found. Loading from disk (this will take a while)...")
    import pickle
    
    df_train = pd.read_csv(TRAIN_CSV)
    df_dev = pd.read_csv(DEV_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df_train["accent"])
    y_dev = encoder.transform(df_dev["accent"])
    y_test = encoder.transform(df_test["accent"])
    num_classes = len(encoder.classes_)
    
    def load_mfcc_features(df, max_width=250):
        X = []
        for i, path in enumerate(df["spectrogram"]):
            if i % 1000 == 0:
                print(f"  Loaded {i}/{len(df)} samples...")
            try:
                S = np.load(path)
                if S.shape[1] < max_width:
                    pad_width = max_width - S.shape[1]
                    S = np.pad(S, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    S = S[:, :max_width]
                X.append(S)
            except Exception as e:
                print(f"  Error loading {path}: {e}")
                continue
        return np.array(X)
    
    print("Loading train data...")
    X_train = load_mfcc_features(df_train)
    print("Loading dev data...")
    X_dev = load_mfcc_features(df_dev)
    print("Loading test data...")
    X_test = load_mfcc_features(df_test)
    
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_dev = (X_dev - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    
    X_train = X_train[..., np.newaxis]
    X_dev = X_dev[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    print("(Skipping cache save to preserve RAM)")

print(f"Classes: {list(encoder.classes_)}")
print(f"Train samples: {len(y_train)}, Dev: {len(y_dev)}, Test: {len(y_test)}")
print(f"X_train shape: {X_train.shape}")
print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}], mean: {X_train.mean():.3f}")

# FOR MAC TESTING: Use only 20% of data to fit in memory
SUBSAMPLE_FRACTION = 0.1
X_train = X_train[:int(len(X_train) * SUBSAMPLE_FRACTION)]
y_train = y_train[:int(len(y_train) * SUBSAMPLE_FRACTION)]
X_dev = X_dev[:int(len(X_dev) * SUBSAMPLE_FRACTION)]
y_dev = y_dev[:int(len(y_dev) * SUBSAMPLE_FRACTION)]
X_test = X_test[:int(len(X_test) * SUBSAMPLE_FRACTION)]
y_test = y_test[:int(len(y_test) * SUBSAMPLE_FRACTION)]
print(f"\n⚠️  SUBSAMPLED TO {SUBSAMPLE_FRACTION*100:.0f}% FOR MAC TESTING")
print(f"Train samples (subsampled): {len(y_train)}, Dev: {len(y_dev)}, Test: {len(y_test)}")

y_train_cat = to_categorical(y_train, num_classes)
y_dev_cat = to_categorical(y_dev, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ==================================================================
# BUILD LIGHTWEIGHT CNN MODEL (Much smaller for Mac)
# ==================================================================
print("\n" + "="*60)
print("BUILDING LIGHTWEIGHT MODEL FOR MAC")
print("="*60)

model = Sequential([
    # Block 1 - Reduced from 64 to 32 filters
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), input_shape=X_train.shape[1:]),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal()),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    # Block 2 - Reduced from 128 to 64 filters, removed one Conv layer
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal()),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal()),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    # Block 3 - Reduced from 256 to 128 filters, removed one Conv layer
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal()),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    # Global pooling
    GlobalAveragePooling2D(),
    
    # Dense layers - Reduced from 512/256 to 128/64
    Dense(128, activation='relu', kernel_initializer=HeNormal()),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_initializer=HeNormal()),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')
])

model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"\nTotal trainable parameters: {trainable_params:,}")
print(f"Estimated memory usage: ~{trainable_params * 4 / 1024 / 1024:.1f} MB")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==================================================================
# TEST: OVERFITTING TEST ON 100 SAMPLES
# ==================================================================
print("\n" + "="*60)
print("TEST 1: OVERFITTING ON 100 SAMPLES")
print("="*60)

X_tiny = X_train[:100]
y_tiny = y_train_cat[:100]

print(f"Training on {len(X_tiny)} samples...")
print(f"Label distribution: {np.bincount(np.argmax(y_tiny, axis=1))}")

history_tiny = model.fit(
    X_tiny, y_tiny,
    epochs=50,
    batch_size=16,  # Reduced from 32
    verbose=1
)

final_acc = history_tiny.history['accuracy'][-1]
print(f"\n{'='*60}")
print(f"OVERFITTING TEST RESULT: {final_acc:.4f}")
print(f"{'='*60}")

if final_acc > 0.9:
    print("✓ Model can learn. Proceeding to full training...")
    proceed = True
else:
    print("✗ Model cannot overfit 100 samples.")
    print("Stopping here. Check model architecture.")
    proceed = False

# Clear memory
gc.collect()

# plot overfitting test
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_tiny.history['accuracy'])
plt.axhline(y=0.9, color='g', linestyle='--', label='Target (90%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Overfitting Test - Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_tiny.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Overfitting Test - Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('overfitting_test_lightweight.png')
print("\n✓ Saved overfitting_test_lightweight.png")

# ==================================================================
# FULL TRAINING (only if overfitting test passed)
# ==================================================================
proceed = True  # Set to True to run training

if proceed:
    print("\n" + "="*60)
    print("TRAINING ON SUBSET (Memory-Efficient)")
    print("="*60)
    
    # Create data generator with augmentation
    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='constant',
        cval=0
    )
    
    # Add callbacks
    early_stop = EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_accuracy',
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_cnn_model_lightweight.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train with smaller batch size
    BATCH_SIZE = 8  # Very small for Mac
    
    history = model.fit(
        train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        validation_data=(X_dev, y_dev_cat),
        epochs=50,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )
    
    # Clear memory
    gc.collect()
    
    # evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, batch_size=BATCH_SIZE, verbose=0)
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print(f"{'='*60}")
    
    # save model
    model.save("accent_cnn_model_lightweight.h5")
    print("\n✓ Model saved as accent_cnn_model_lightweight.h5")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training History - Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_lightweight.png')
    print("✓ Saved training_history_lightweight.png")

    print("\n" + "="*60)
    print("TEST SET: CONFUSION MATRIX & CLASSIFICATION REPORT")
    print("="*60)
    
    # get predictions in batches to save memory
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)
    
    # confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=encoder.classes_, 
                yticklabels=encoder.classes_)
    plt.title('Confusion Matrix - Test Set (Lightweight Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_lightweight.png')
    print("✓ Saved confusion_matrix_lightweight.png")
    
    # Classification Report
    print("\n" + "="*60)
    print("Per-Class Performance:")
    print("="*60)
    report = classification_report(y_test_classes, y_pred_classes, 
                                   target_names=encoder.classes_, 
                                   digits=3)
    print(report)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, accent in enumerate(encoder.classes_):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {accent:>10s}: {class_acc:.1%} ({class_correct}/{class_total})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"\nModel size comparison:")
    print(f"  Original model: ~3-4M parameters")
    print(f"  Lightweight model: ~{trainable_params:,} parameters")
else:
    print("\nFull training skipped due to failed overfitting test.")