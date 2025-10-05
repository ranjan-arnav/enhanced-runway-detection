#!/usr/bin/env python3
"""
📊 Quick CSV Generator - User's Exact Format
Optimized for speed - generates CSV in the exact format requested:
Image Name | IOU score | Anchor Score | Boolean Score
with Mean Score row at the bottom
"""

import os
import glob
import pandas as pd
import numpy as np
from enhanced_exact_procedure_analysis import EnhancedExactProcedureAnalyzer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class QuickCSVAnalyzer(EnhancedExactProcedureAnalyzer):
    """Ultra-fast analyzer for CSV generation only"""
    
    def __init__(self, model_path="models/best_fast_runway_unet.h5"):
        super().__init__(model_path)
        print("🚀 QUICK CSV ANALYZER LOADED - User Format")
    
    def quick_analyze(self, image_path, gt_path):
        """Ultra-fast analysis for CSV only"""
        try:
            # Load and preprocess
            original_image, processed_image, original_size = self.load_and_preprocess_image(image_path)
            gt_original, gt_processed = self.load_ground_truth(gt_path)
            
            # Get model prediction
            pred_input = np.expand_dims(processed_image, axis=0)
            prediction = self.model.predict(pred_input, verbose=0)[0, :, :, 0]
            
            # Quick edge extraction (simplified)
            ledg_pred, redg_pred, ctl_pred = self.extract_edge_points_enhanced(prediction)
            ledg_gt, redg_gt, ctl_gt = self.extract_edges_from_ground_truth(gt_processed)
            
            # Choose best edges
            if (ledg_gt and redg_gt and ctl_gt and 
                self.validate_edge_quality(ledg_gt, redg_gt, ctl_gt)):
                ledg, redg, ctl = ledg_gt, redg_gt, ctl_gt
                use_gt = True
            elif ledg_pred and redg_pred and ctl_pred:
                ledg, redg, ctl = ledg_pred, redg_pred, ctl_pred
                use_gt = False
            else:
                return None
            
            # Quick calculations
            polygon = self.create_polygon_from_edges(ledg, redg, ctl)
            
            # Simplified scoring
            if use_gt:
                anchor_result = self.calculate_anchor_score_adjusted(prediction, polygon, gt_processed, "ground_truth")
                iou_result = self.calculate_contextual_iou(prediction, gt_processed)
            else:
                anchor_result = self.calculate_anchor_score(prediction, polygon)
                iou_result = self.calculate_iou_score(prediction, gt_processed)
            
            boolean_result = self.calculate_boolean_score(ledg, redg, ctl, polygon)
            
            return {
                'iou_score': float(iou_result['iou_score']),
                'anchor_score': float(anchor_result['anchor_score']),
                'boolean_score': 1.0 if boolean_result['boolean_score'] else 0.0
            }
            
        except Exception as e:
            print(f"❌ Error with {os.path.basename(image_path)}: {str(e)[:50]}")
            return None

def main():
    """Generate CSV in user's exact format"""
    
    print("📊 QUICK CSV GENERATOR - USER'S EXACT FORMAT")
    print("=" * 50)
    
    # Configuration
    test_folder = "1920x1080/1920x1080/test"
    gt_folder = "labels/labels/areas/test_labels_1920x1080"
    
    # Get first 100 images for quick test (or all if you want)
    image_pattern = os.path.join(test_folder, "*.png")
    all_images = sorted(glob.glob(image_pattern))
    
    # For testing: use first 50 images, for production: use all_images
    test_images = all_images[:50]  # Change to all_images for full processing
    
    print(f"📈 Processing {len(test_images)} test images...")
    
    # Initialize analyzer
    analyzer = QuickCSVAnalyzer()
    
    # Results storage
    results = []
    
    # Process images
    for i, image_path in enumerate(test_images):
        if i % 10 == 0:
            print(f"📊 Processing {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
        
        image_name = os.path.basename(image_path)
        gt_path = os.path.join(gt_folder, image_name)
        
        if not os.path.exists(gt_path):
            continue
        
        result = analyzer.quick_analyze(image_path, gt_path)
        
        if result:
            results.append({
                'Image Name': image_name,
                'IoU Score': result['iou_score'],
                'Anchor Score': result['anchor_score'], 
                'Boolean Score': result['boolean_score']
            })
    
    if not results:
        print("❌ No successful results")
        return
    
    # Create CSV in user's exact format
    csv_data = []
    
    # Add header
    csv_data.append({
        'Image Name': 'Image Name',
        'IoU Score': 'IOU score',
        'Anchor Score': 'Anchor Score', 
        'Boolean Score': 'Boolean Score'
    })
    
    # Add individual results
    for result in results:
        csv_data.append({
            'Image Name': result['Image Name'],
            'IoU Score': f"{result['IoU Score']:.1f}",
            'Anchor Score': f"{result['Anchor Score']:.1f}",
            'Boolean Score': f"{result['Boolean Score']:.0f}"
        })
    
    # Calculate means
    mean_iou = np.mean([r['IoU Score'] for r in results])
    mean_anchor = np.mean([r['Anchor Score'] for r in results])
    mean_boolean = np.mean([r['Boolean Score'] for r in results])
    
    # Add Mean Score row
    csv_data.append({
        'Image Name': 'Mean Score',
        'IoU Score': f"{mean_iou:.2f}",
        'Anchor Score': f"{mean_anchor:.2f}",
        'Boolean Score': f"{mean_boolean:.1f}"
    })
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    output_file = "user_format_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ CSV Generated: {output_file}")
    print(f"📊 Format: {len(results)} images + Mean Score row")
    print(f"📈 Mean IoU: {mean_iou:.2f}")
    print(f"📈 Mean Anchor: {mean_anchor:.2f}")
    print(f"📈 Mean Boolean: {mean_boolean:.1f}")
    
    # Show sample
    print(f"\n📋 Sample CSV content:")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(1).to_string(index=False))

if __name__ == "__main__":
    main()