"""
🛬 Phase 1: Fast U-Net Training (Smaller Dataset) ✈️

This is an optimized version for quick testing with a smaller subset of data.
Perfect for validating the pipeline and getting results quickly.

Key optimizations:
- Uses only 200 training samples (instead of 3987)
- Smaller model with fewer parameters
- Reduced epochs and faster convergence
- CPU-optimized settings
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from glob import glob
import albumentations as A
import random

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Optimize TensorFlow for CPU
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

class FastRunwayDataGenerator(tf.keras.utils.Sequence):
    """Optimized data generator for fast training"""
    
    def __init__(self, image_paths, mask_paths, batch_size=16, img_size=(128, 128), 
                 shuffle=True, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.image_paths))
        
        # Simplified augmentation for speed
        if self.augment:
            self.augmentor = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ])
        
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_masks = []
        
        for idx in batch_indexes:
            # Load and preprocess image
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            
            # Load and preprocess mask
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            
            # Apply augmentation
            if self.augment:
                augmented = self.augmentor(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Normalize image to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert mask to binary [0, 1] (mask values are 0 and 38)
            mask = (mask > 19).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            
            batch_images.append(image)
            batch_masks.append(mask)
        
        return np.array(batch_images), np.array(batch_masks)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for binary segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined Dice + Binary Cross-Entropy loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def create_fast_unet_model(input_shape=(128, 128, 3), num_classes=1):
    """
    Create a faster, smaller U-Net model optimized for quick training
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (simplified)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)  # 64x64
    
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)  # 32x32
    
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)  # 16x16
    
    # Bridge
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = layers.UpSampling2D((2, 2))(conv4)  # 32x32
    merge5 = layers.Concatenate()([conv3, up5])
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = layers.UpSampling2D((2, 2))(conv5)  # 64x64
    merge6 = layers.Concatenate()([conv2, up6])
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = layers.UpSampling2D((2, 2))(conv6)  # 128x128
    merge7 = layers.Concatenate()([conv1, up7])
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    
    # Final output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(conv7)
    
    model = Model(inputs, outputs, name='FastUNet_Runway')
    return model

def load_sample_data_paths(data_dir, max_samples=200):
    """Load a subset of data for fast training"""
    print(f"📂 Loading sample data (max {max_samples} samples)...")
    
    # Training image paths
    train_img_dir = os.path.join(data_dir, "1920x1080", "1920x1080", "train")
    train_mask_dir = os.path.join(data_dir, "labels", "labels", "areas", "train_labels_1920x1080")
    
    # Get all training images
    train_images = glob(os.path.join(train_img_dir, "*.png"))
    
    # Randomly sample a subset
    random.shuffle(train_images)
    train_images = train_images[:max_samples]
    
    print(f"Selected {len(train_images)} random training images")
    
    # Match images with masks
    matched_pairs = []
    for img_path in train_images:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(train_mask_dir, img_name)
        
        if os.path.exists(mask_path):
            matched_pairs.append((img_path, mask_path))
        else:
            print(f"Warning: No mask found for {img_name}")
    
    print(f"Successfully matched {len(matched_pairs)} image-mask pairs")
    
    # Separate paths
    image_paths = [pair[0] for pair in matched_pairs]
    mask_paths = [pair[1] for pair in matched_pairs]
    
    return image_paths, mask_paths

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Dice coefficient plot
    ax2.plot(history.history['dice_coefficient'], label='Training Dice')
    ax2.plot(history.history['val_dice_coefficient'], label='Validation Dice')
    ax2.set_title('Dice Coefficient')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Training plots saved to: {save_path}")

def visualize_predictions(model, generator, num_samples=4):
    """Visualize model predictions on validation data"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    
    # Get a batch of data
    images, true_masks = generator[0]
    
    # Make predictions
    pred_masks = model.predict(images[:num_samples])
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # True mask
        axes[i, 1].imshow(true_masks[i, :, :, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(pred_masks[i, :, :, 0], cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fast_validation_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("🔍 Prediction visualizations saved to: fast_validation_predictions.png")

def main():
    """Main fast training function"""
    print("🚀 Starting Phase 1: Fast U-Net Runway Segmentation Training ✈️")
    print("=" * 70)
    print("🎯 FAST MODE: Using smaller dataset for quick validation")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = r"c:\Users\Om Raj\Desktop\arch"
    IMG_SIZE = (128, 128)  # Smaller image size for speed
    BATCH_SIZE = 16        # Larger batch size for efficiency
    EPOCHS = 20            # Fewer epochs
    LEARNING_RATE = 1e-3   # Higher learning rate for faster convergence
    MAX_SAMPLES = 200      # Small dataset
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load sample data paths
    print("\n📂 Loading sample data...")
    image_paths, mask_paths = load_sample_data_paths(DATA_DIR, MAX_SAMPLES)
    
    if len(image_paths) < 20:
        print("❌ Not enough samples found. Need at least 20 samples.")
        return
    
    # Split data (80% train, 20% validation)
    print("\n🔄 Splitting data...")
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Create data generators
    print("\n🔧 Creating data generators...")
    train_generator = FastRunwayDataGenerator(
        train_images, train_masks, 
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE, 
        shuffle=True, 
        augment=True
    )
    
    val_generator = FastRunwayDataGenerator(
        val_images, val_masks, 
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE, 
        shuffle=False, 
        augment=False
    )
    
    # Create model
    print("\n🏗️ Building Fast U-Net model...")
    model = create_fast_unet_model(input_shape=(*IMG_SIZE, 3))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coefficient, 'binary_accuracy']
    )
    
    print(f"Model created with {model.count_params():,} parameters")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            'models/best_fast_runway_unet.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🚀 Starting fast training ({EPOCHS} epochs max)...")
    print("=" * 70)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/fast_runway_unet_final.h5')
    print("\n✅ Fast training completed!")
    print(f"Models saved in 'models/' directory")
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_training_history(history, 'plots/fast_training_history.png')
    
    # Visualize predictions
    print("\n🔍 Visualizing predictions...")
    visualize_predictions(model, val_generator)
    
    # Final evaluation
    print("\n📈 Final Model Evaluation:")
    val_loss, val_dice, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Dice Score: {val_dice:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("\n🎉 Fast Phase 1 Complete!")
    print("💡 This model can now be used for Phase 2: Inference and Line Extraction")
    print("=" * 70)

if __name__ == "__main__":
    main()