"""
🛬 Data Verification Script ✈️

This script verifies the dataset structure and displays sample images and masks
to ensure everything is correctly set up before training.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random

def verify_data_structure(data_dir):
    """Verify the dataset structure and file counts"""
    print("📂 Verifying dataset structure...")
    
    # Check main directories
    train_img_dir = os.path.join(data_dir, "1920x1080", "1920x1080", "train")
    train_mask_dir = os.path.join(data_dir, "labels", "labels", "areas", "train_labels_1920x1080")
    test_img_dir = os.path.join(data_dir, "1920x1080", "1920x1080", "test")
    
    print(f"Training images directory: {train_img_dir}")
    print(f"Training masks directory: {train_mask_dir}")
    print(f"Test images directory: {test_img_dir}")
    
    # Count files
    train_images = glob(os.path.join(train_img_dir, "*.png"))
    train_masks = glob(os.path.join(train_mask_dir, "*.png"))
    test_images = glob(os.path.join(test_img_dir, "*.png"))
    
    print(f"\n📊 File Counts:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Training masks: {len(train_masks)}")
    print(f"  Test images: {len(test_images)}")
    
    # Check matching pairs
    matched_pairs = 0
    sample_pairs = []
    
    for img_path in train_images[:100]:  # Check first 100 for speed
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(train_mask_dir, img_name)
        
        if os.path.exists(mask_path):
            matched_pairs += 1
            if len(sample_pairs) < 6:
                sample_pairs.append((img_path, mask_path))
    
    print(f"  Matched pairs (first 100 checked): {matched_pairs}/100")
    
    return sample_pairs

def display_samples(sample_pairs):
    """Display sample image-mask pairs"""
    print("\n🖼️ Displaying sample image-mask pairs...")
    
    fig, axes = plt.subplots(len(sample_pairs), 3, figsize=(12, 3*len(sample_pairs)))
    if len(sample_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img_path, mask_path) in enumerate(sample_pairs):
        # Load original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize for display
        img_resized = cv2.resize(img, (256, 256))
        mask_resized = cv2.resize(mask, (256, 256))
        
        # Display original image
        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f'Original Image\n{os.path.basename(img_path)}')
        axes[i, 0].axis('off')
        
        # Display mask
        axes[i, 1].imshow(mask_resized, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth Mask\n{os.path.basename(mask_path)}')
        axes[i, 1].axis('off')
        
        # Display overlay
        overlay = img_resized.copy()
        runway_pixels = mask_resized > 19  # Mask values are 0 and 38
        overlay[runway_pixels, 0] = 255  # Red channel for runway
        overlay[runway_pixels, 1] = 0
        overlay[runway_pixels, 2] = 0
        blended = cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0)
        
        axes[i, 2].imshow(blended)
        axes[i, 2].set_title('Image + Mask Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_verification_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print mask statistics
    print("\n📈 Mask Statistics:")
    for i, (img_path, mask_path) in enumerate(sample_pairs):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        runway_pixels = np.sum(mask > 19)  # Mask values are 0 and 38
        total_pixels = mask.shape[0] * mask.shape[1]
        runway_percentage = (runway_pixels / total_pixels) * 100
        
        print(f"  Sample {i+1}: {runway_percentage:.2f}% runway pixels")

def check_image_formats(sample_pairs):
    """Check image formats and properties"""
    print("\n🔍 Checking image properties...")
    
    for i, (img_path, mask_path) in enumerate(sample_pairs[:3]):
        # Check image
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"\nSample {i+1}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Mask unique values: {np.unique(mask)}")

def main():
    """Main verification function"""
    print("🛬 Runway Dataset Verification ✈️")
    print("=" * 50)
    
    # Set data directory
    DATA_DIR = r"c:\Users\Om Raj\Desktop\arch"
    
    # Verify structure and get samples
    sample_pairs = verify_data_structure(DATA_DIR)
    
    if sample_pairs:
        # Display samples
        display_samples(sample_pairs)
        
        # Check formats
        check_image_formats(sample_pairs)
        
        print("\n✅ Data verification complete!")
        print("💡 If everything looks good, run: python phase1_unet_training.py")
    else:
        print("\n❌ No matching image-mask pairs found!")
        print("Please check your dataset structure.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()