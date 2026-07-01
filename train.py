import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV3Large
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

def train(dataset_dir, epochs=20, batch_size=32, seed=42):
    print(f"Loading dataset from: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        shuffle=True,
        image_size=(124, 124),
        batch_size=batch_size
    )

    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        shuffle=True,
        image_size=(124, 124),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Split validation dataset into test and validation sets
    val_batches = tf.data.experimental.cardinality(val_ds)  
    test_ds = val_ds.take(val_batches // 2)  
    val_dat = val_ds.skip(val_batches // 2)  

    # Optimize datasets
    test_ds_eval = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    # Compute class weights
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())
    
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(class_names)),
        y=all_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weights_array)}
    print(f"Class weights computed: {class_weights}")

    # Define Data Augmentation
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    # Load pre-trained MobileNetV3Large base model
    base_model = MobileNetV3Large(
        include_top=False,
        input_shape=(124, 124, 3),
        weights='imagenet',
        include_preprocessing=True
    )
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Define model architecture
    model = Sequential([
        layers.Input(shape=(124, 124, 3)),
        data_augmentation,
        base_model,
        GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping callback (monitors pure validation split 'val_dat')
    early = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_dat,  # Fixed data leakage bug
        epochs=epochs,
        class_weight=class_weights,
        batch_size=batch_size,
        callbacks=[early]
    )

    # Evaluate on test set
    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_ds_eval)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Generate classification metrics
    y_true = np.concatenate([y.numpy() for _, y in test_ds_eval], axis=0)
    y_pred_probs = model.predict(test_ds_eval)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save model
    model_name = "MobileNetV3Large.keras"
    model.save(model_name)
    print(f"Model saved successfully to {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3Large for Garbage Classification")
    parser.add_argument("--data_dir", type=str, default="./TrashType_Image_Dataset", help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size)
