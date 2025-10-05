#!/usr/bin/env python3
"""
Create Ground Truth Label for YSSY16R1_2LDImage1.png
Based on the runway segmentation shown in the uploaded visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_runway_label():
    """Create ground truth label for YSSY16R1_2LDImage1.png"""
    
    # Load the original image to get dimensions
    image_path = "1920x1080/1920x1080/test/YSSY16R1_2LDImage1.png"
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    height, width = original_image.shape[:2]
    print(f"📏 Image dimensions: {width}x{height}")
    
    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Based on the visualization, create a more accurate runway mask
    # The runway should be much larger - it occupies most of the central area
    # Looking at your uploaded image, the runway is the main trapezoidal area
    
    # Scale coordinates to full image dimensions (128x128 -> 1920x1080)
    scale_x = width / 128
    scale_y = height / 128
    
    # Runway polygon points scaled to full image size
    runway_points = np.array([
        # Bottom wide part of runway (scaled)
        [int(12 * scale_x), int(127 * scale_y)],   # Bottom left
        [int(87 * scale_x), int(127 * scale_y)],   # Bottom right
        
        # Top narrow part of runway (scaled, vanishing point perspective)
        [int(55 * scale_x), int(63 * scale_y)],    # Top right  
        [int(50 * scale_x), int(63 * scale_y)],    # Top left
    ], dtype=np.int32)
    
    print(f"🛬 Runway polygon points:")
    for i, point in enumerate(runway_points):
        print(f"   Point {i+1}: ({point[0]}, {point[1]})")
    
    # Fill the runway polygon
    cv2.fillPoly(mask, [runway_points], 255)
    
    # Create output directory
    output_dir = Path("labels/labels/areas/test_labels_1920x1080")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the label
    output_path = output_dir / "YSSY16R1_2LDImage1.png"
    cv2.imwrite(str(output_path), mask)
    
    print(f"✅ Ground truth label created: {output_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Created label
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Created Ground Truth Label")
    axes[1].axis('off')
    
    # Overlay
    overlay = original_rgb.copy()
    mask_colored = np.zeros_like(original_rgb)
    mask_colored[mask == 255] = [255, 0, 0]  # Red for runway
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Original + Created Label Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("runway_label_creation.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved: runway_label_creation.png")
    
    # Verify the created label
    print(f"📈 Label statistics:")
    print(f"   • Total pixels: {mask.size}")
    print(f"   • Runway pixels: {np.sum(mask == 255)}")
    print(f"   • Background pixels: {np.sum(mask == 0)}")
    print(f"   • Runway coverage: {np.sum(mask == 255) / mask.size * 100:.2f}%")

if __name__ == "__main__":
    print("🎯 CREATING GROUND TRUTH LABEL FOR YSSY16R1_2LDImage1.png")
    print("=" * 60)
    create_runway_label()
    print("=" * 60)
    print("🎉 LABEL CREATION COMPLETE!")