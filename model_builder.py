import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_CSV_PATH = "labels.csv"
TRAIN_IMAGES_PATH = "traffic_Data/DATA"
TEST_IMAGES_PATH = "traffic_Data/TEST"
MODEL_SAVE_PATH = "best_traffic_sign_model.keras"
IMAGE_HEIGHT, IMAGE_WIDTH = 32, 32
EPOCHS = 15
BATCH_SIZE = 64

print("Starting traffic sign classification training...")
print("Loading labels from labels.csv...")
labels_df = pd.read_csv(LABELS_CSV_PATH)
num_classes = labels_df.shape[0]
print(f"Found {num_classes} classes.")

def load_images(base_path):
    images, labels = [], []
    for label in range(num_classes):
        class_dir = os.path.join(base_path, str(label))
        if not os.path.isdir(class_dir):
            continue
        image_files = os.listdir(class_dir)
        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

print(f"Loading training images from {TRAIN_IMAGES_PATH}...")
x_train, y_train = load_images(TRAIN_IMAGES_PATH)
print(f"Loaded {x_train.shape[0]} training images.")

# Load test images
print(f"Loading test images from {TEST_IMAGES_PATH}...")
x_test, y_test = load_images(TEST_IMAGES_PATH)
if len(x_test) == 0:
    print("No test images found, using train/validation split...")
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(f"Training set size: {x_train.shape[0]}")
print(f"Test set size: {x_test.shape[0]}")

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define multiple models
def build_basic_cnn():
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_deeper_cnn():
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_vgg_like_cnn():
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mobilenetv2_small():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        include_top=False,
        weights=None
    )
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet_like():
    inputs = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    
    # First conv block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Shortcut path
    shortcut = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same')(inputs)

    # Residual connection
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# List of models to train
models = {
    "Basic_CNN": build_basic_cnn(),
    "Deeper_CNN": build_deeper_cnn(),
    "VGG_Like_CNN": build_vgg_like_cnn(),
    "MobileNetV2_Small": build_mobilenetv2_small(),
    "ResNet_Like_CNN": build_resnet_like()
}

# Train and evaluate all models
model_accuracies = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    val_accuracy = max(history.history['val_accuracy'])
    model_accuracies[name] = val_accuracy
    print(f"{name} Validation Accuracy: {val_accuracy:.4f}")

# Print all model accuracies
print("\n==== Summary of all models ====")
for name, acc in model_accuracies.items():
    print(f"{name}: {acc:.4f}")

# Select best model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_val_accuracy = model_accuracies[best_model_name]
best_model = models[best_model_name]

# Save best model
if best_model:
    best_model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Best Model: {best_model_name} with Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
else:
    print("No model trained.")
