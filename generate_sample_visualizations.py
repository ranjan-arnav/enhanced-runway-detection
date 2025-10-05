#!/usr/bin/env python3
"""
Generate sample analysis visualization images for testing the web interface
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2

def create_sample_runway_image():
    """Create a synthetic runway image"""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Sky gradient (blue to light blue)
    for i in range(180):
        color = int(135 + (120 * i / 180))
        img[i, :] = [min(255, color), min(255, color + 20), 255]
    
    # Ground (green/brown)
    img[180:, :] = [34, 139, 34]
    
    # Runway (gray rectangle)
    runway_start_x = 250
    runway_width = 140
    runway_start_y = 200
    runway_height = 120
    
    # Main runway surface
    img[runway_start_y:runway_start_y+runway_height, 
        runway_start_x:runway_start_x+runway_width] = [80, 80, 80]
    
    # Runway centerline (white)
    center_x = runway_start_x + runway_width // 2
    for y in range(runway_start_y + 10, runway_start_y + runway_height - 10, 20):
        img[y:y+10, center_x-2:center_x+2] = [255, 255, 255]
    
    # Runway edges (white lines)
    img[runway_start_y:runway_start_y+runway_height, runway_start_x:runway_start_x+2] = [255, 255, 255]
    img[runway_start_y:runway_start_y+runway_height, runway_start_x+runway_width-2:runway_start_x+runway_width] = [255, 255, 255]
    
    return img

def create_heatmap_visualization():
    """Create a heatmap-style visualization"""
    heatmap = np.zeros((360, 640))
    
    # Create runway heatmap
    runway_start_x = 250
    runway_width = 140
    runway_start_y = 200
    runway_height = 120
    
    # Strong activation on runway
    heatmap[runway_start_y:runway_start_y+runway_height, 
            runway_start_x:runway_start_x+runway_width] = 0.8
    
    # Medium activation around runway
    margin = 20
    heatmap[runway_start_y-margin:runway_start_y+runway_height+margin, 
            runway_start_x-margin:runway_start_x+runway_width+margin] = 0.4
    
    # Add some noise
    noise = np.random.normal(0, 0.1, heatmap.shape)
    heatmap = np.clip(heatmap + noise, 0, 1)
    
    return heatmap

def create_ground_truth_mask():
    """Create a clean ground truth mask"""
    mask = np.zeros((360, 640))
    
    # Clean runway rectangle
    runway_start_x = 250
    runway_width = 140
    runway_start_y = 200
    runway_height = 120
    
    mask[runway_start_y:runway_start_y+runway_height, 
         runway_start_x:runway_start_x+runway_width] = 1
    
    return mask

def create_edge_visualization(img):
    """Create edge detection visualization"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Create colored edge image
    edge_img = img.copy()
    edge_img[edges > 0] = [0, 255, 0]  # Green edges
    
    return edge_img

def create_polygon_overlay(img):
    """Create polygon overlay visualization"""
    overlay_img = img.copy()
    
    # Define runway polygon
    runway_points = np.array([
        [250, 200],   # Top-left
        [390, 200],   # Top-right
        [390, 320],   # Bottom-right
        [250, 320]    # Bottom-left
    ])
    
    # Draw polygon outline
    cv2.polylines(overlay_img, [runway_points], True, (0, 255, 255), 3)  # Cyan outline
    
    # Fill polygon with semi-transparent overlay
    overlay = overlay_img.copy()
    cv2.fillPoly(overlay, [runway_points], (0, 255, 255))
    cv2.addWeighted(overlay_img, 0.7, overlay, 0.3, 0, overlay_img)
    
    return overlay_img

def generate_enhanced_analysis_visualization(image_name, iou_score, anchor_score, boolean_score):
    """Generate a complete enhanced analysis visualization"""
    
    # Create the figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Enhanced Exact Procedure Analysis: {image_name}', fontsize=16, fontweight='bold')
    
    # Create grid layout
    gs = GridSpec(3, 4, figure=fig, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 0.8])
    
    # Generate sample data
    original_img = create_sample_runway_image()
    heatmap = create_heatmap_visualization()
    gt_mask = create_ground_truth_mask()
    edge_img = create_edge_visualization(original_img)
    polygon_img = create_polygon_overlay(original_img)
    
    # 1. Input RGB Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img)
    ax1.set_title('Input RGB Image', fontweight='bold')
    ax1.axis('off')
    
    # 2. Segmented Image (GT-Enhanced Display)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(heatmap, cmap='hot', alpha=0.8)
    ax2.set_title('Segmented Image (GT-Enhanced Display)', fontweight='bold')
    ax2.axis('off')
    
    # 3. Ground Truth Segment Mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(gt_mask, cmap='gray')
    ax3.set_title('Ground Truth Segment Mask', fontweight='bold')
    ax3.axis('off')
    
    # 4. Enhanced Edge Visualization
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(edge_img)
    ax4.set_title('Enhanced Edge Visualization', fontweight='bold')
    ax4.axis('off')
    
    # 5. Runway Polygon Overlay
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(polygon_img)
    ax5.set_title('Runway Polygon Overlay', fontweight='bold')
    ax5.axis('off')
    
    # 6. Results Panel
    ax6 = fig.add_subplot(gs[:, 3])
    ax6.text(0.1, 0.9, 'ENHANCED ANALYSIS RESULTS', fontsize=14, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax6.text(0.1, 0.75, f'📊 IoU SCORE: {iou_score:.4f}', fontsize=12, fontweight='bold')
    ax6.text(0.15, 0.70, f'• Intersection: {int(iou_score * 1000)} pixels', fontsize=10)
    ax6.text(0.15, 0.66, f'• Union: {int(1000)} pixels', fontsize=10)
    
    ax6.text(0.1, 0.55, f'🎯 ANCHOR: {anchor_score:.4f}', fontsize=12, fontweight='bold')
    ax6.text(0.15, 0.50, f'• Polygons: 0 pixels', fontsize=10)
    ax6.text(0.15, 0.46, f'• Total: 4,447 pixels', fontsize=10)
    
    ax6.text(0.1, 0.35, f'✅ BOOLEAN: {boolean_score}', fontsize=12, fontweight='bold')
    
    ax6.text(0.1, 0.20, '🔧 Enhanced GT-Aligned Analysis', fontsize=11, fontweight='bold')
    ax6.text(0.15, 0.15, '• Ground truth edge prioritization', fontsize=9)
    ax6.text(0.15, 0.11, '• Spatial alignment detection', fontsize=9)
    ax6.text(0.15, 0.07, '• Visual GT-based heatmap display', fontsize=9)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def main():
    """Generate sample visualizations"""
    
    # Create output directory
    output_dir = "exact_procedure_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Generating sample enhanced analysis visualizations...")
    
    # Sample data
    samples = [
        {
            'name': 'LPPD30_3_5FNLImage5',
            'iou': 0.5789,
            'anchor': 0.6047,
            'boolean': True
        },
        {
            'name': '4AK606_1_5LDImage2', 
            'iou': 0.4321,
            'anchor': 0.3976,
            'boolean': True
        },
        {
            'name': 'EDDF07R1_4LDImage1',
            'iou': 0.6234,
            'anchor': 0.5567,
            'boolean': True
        },
        {
            'name': 'KJAC01_1_10LDImage1',
            'iou': 0.4892,
            'anchor': 0.4123,
            'boolean': False
        },
        {
            'name': 'LGAV03R3_2LDImage2',
            'iou': 0.5445,
            'anchor': 0.4789,
            'boolean': True
        }
    ]
    
    for sample in samples:
        print(f"📊 Generating visualization for {sample['name']}...")
        
        # Generate the visualization
        fig = generate_enhanced_analysis_visualization(
            sample['name'], 
            sample['iou'], 
            sample['anchor'], 
            sample['boolean']
        )
        
        # Save the image
        output_path = os.path.join(output_dir, f"{sample['name']}_ENHANCED_GT_ALIGNED_exact_procedure.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"✅ Saved: {output_path}")
    
    print(f"\n🎉 Generated {len(samples)} sample visualizations in {output_dir}/")
    print("🌐 Now open http://localhost:5000 to see them in your web interface!")

if __name__ == "__main__":
    main()