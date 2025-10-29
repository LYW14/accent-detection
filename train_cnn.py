import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# ==== CONFIG ====
CSV_PATH = "labeled_dataset.csv"
# ================

# Load data
df = pd.read_csv(CSV_PATH)

# Encode accent labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["accent"])
num_classes = len(encoder.classes_)
print("Classes:", list(encoder.classes_))

# Load and pad/trim spectrograms
X = []
max_width = 500  # adjust based on your dataset

for path in df["spectrogram"]:
    S = np.load(path)
    if S.shape[1] < max_width:
        # pad with zeros on the right
        pad_width = max_width - S.shape[1]
        S = np.pad(S, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # trim if too long
        S = S[:, :max_width]
    X.append(S)

X = np.array(X)
X = X[..., np.newaxis]  # add channel dimension

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Model definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat),
                    epochs=15, batch_size=8)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"\n Test accuracy: {test_acc:.2f}")

# Save model
model.save("accent_cnn_model.h5")
print("Model saved as accent_cnn_model.h5")

import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over epochs')

plt.tight_layout()
plt.show()

