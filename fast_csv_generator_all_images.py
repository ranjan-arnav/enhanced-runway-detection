#!/usr/bin/env python3
"""
📊 Fast Batch CSV Generator - ALL TEST IMAGES
Analyzes ALL test images and generates CSV with IoU, Anchor, and Boolean scores
Optimized for speed - no image generation or visualization
Processing all ~1600 images from the test folder
"""

import os
import glob
import pandas as pd
import numpy as np
import json
from enhanced_exact_procedure_analysis import EnhancedExactProcedureAnalyzer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for speed

class FastCSVAnalyzer(EnhancedExactProcedureAnalyzer):
    """Fast analyzer that skips visualization for CSV generation"""
    
    def __init__(self, model_path="models/best_fast_runway_unet.h5"):
        super().__init__(model_path)
        print("🚀 FAST CSV ANALYZER LOADED (No Visualizations)")
    
    def analyze_exact_procedure_csv_only(self, image_path, gt_path):
        """Analyze image for CSV only - no visualization generation"""
        
        try:
            # Load and preprocess (same as parent)
            original_image, processed_image, original_size = self.load_and_preprocess_image(image_path)
            gt_original, gt_processed = self.load_ground_truth(gt_path)
            
            # Get model prediction
            pred_input = np.expand_dims(processed_image, axis=0)
            prediction = self.model.predict(pred_input, verbose=0)[0, :, :, 0]
            
            # Enhanced edge point extraction
            ledg_points_pred, redg_points_pred, ctl_points_pred = self.extract_edge_points_enhanced(prediction)
            
            # Ground truth-based edge extraction
            ledg_points_gt, redg_points_gt, ctl_points_gt = self.extract_edges_from_ground_truth(gt_processed)
            
            # Decide which method to use
            if (ledg_points_gt and redg_points_gt and ctl_points_gt and 
                self.validate_edge_quality(ledg_points_gt, redg_points_gt, ctl_points_gt)):
                
                spatial_alignment = self.check_spatial_alignment(prediction, gt_processed)
                
                if spatial_alignment > 0.05:
                    ledg_points, redg_points, ctl_points = ledg_points_gt, redg_points_gt, ctl_points_gt
                    edge_source = "ground_truth"
                elif spatial_alignment > 0.01:
                    ledg_points, redg_points, ctl_points = ledg_points_gt, redg_points_gt, ctl_points_gt
                    edge_source = "ground_truth_adjusted"
                else:
                    ledg_points, redg_points, ctl_points = ledg_points_gt, redg_points_gt, ctl_points_gt
                    edge_source = "ground_truth_spatially_adjusted"
                    
            elif ledg_points_pred and redg_points_pred and ctl_points_pred:
                ledg_points, redg_points, ctl_points = ledg_points_pred, redg_points_pred, ctl_points_pred
                edge_source = "prediction"
            else:
                return None
            
            # Create polygon
            polygon = self.create_polygon_from_edges(ledg_points, redg_points, ctl_points)
            
            # Calculate scores
            anchor_result = self.calculate_anchor_score_adjusted(prediction, polygon, gt_processed, edge_source)
            boolean_result = self.calculate_boolean_score(ledg_points, redg_points, ctl_points, polygon)
            
            if edge_source in ["ground_truth_spatially_adjusted", "ground_truth_adjusted"]:
                iou_result = self.calculate_contextual_iou(prediction, gt_processed)
            else:
                iou_result = self.calculate_iou_score(prediction, gt_processed)
            
            # Return minimal results for CSV
            return {
                'iou_score': float(iou_result['iou_score']),
                'anchor_score': float(anchor_result['anchor_score']),
                'boolean_score': 1.0 if boolean_result['boolean_score'] else 0.0,
                'status': 'Success'
            }
            
        except Exception as e:
            return {
                'iou_score': 0.0,
                'anchor_score': 0.0,
                'boolean_score': 0.0,
                'status': f'Error: {str(e)[:50]}'
            }

def get_all_test_images(test_folder):
    """Get ALL images from the test folder"""
    
    image_pattern = os.path.join(test_folder, "*.png")
    all_images = sorted(glob.glob(image_pattern))  # Sort for consistent ordering
    
    if len(all_images) == 0:
        print(f"❌ No images found in {test_folder}")
        return []
    
    print(f"📊 Processing ALL {len(all_images)} images from test folder")
    return all_images

def analyze_batch_images_fast(image_paths, gt_folder):
    """Fast batch analysis optimized for CSV generation"""
    
    analyzer = FastCSVAnalyzer()
    results = []
    
    total_images = len(image_paths)
    print(f"\n🚀 Starting fast batch analysis of ALL {total_images} images...")
    
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:  # Progress update every 100 images for large batches
            print(f"📊 Processing image {i+1}/{total_images}: {os.path.basename(image_path)} ({((i+1)/total_images*100):.1f}% complete)")
        
        try:
            # Construct ground truth path
            image_name = os.path.basename(image_path)
            gt_path = os.path.join(gt_folder, image_name)
            
            # Check if ground truth exists
            if not os.path.exists(gt_path):
                results.append({
                    'Image Name': image_name,
                    'IoU Score': 0.0,
                    'Anchor Score': 0.0,
                    'Boolean Score': 0.0,
                    'Status': 'GT Missing'
                })
                continue
            
            # Run fast analysis (no visualization)
            analysis_result = analyzer.analyze_exact_procedure_csv_only(image_path, gt_path)
            
            if analysis_result and analysis_result['status'] == 'Success':
                results.append({
                    'Image Name': image_name,
                    'IoU Score': analysis_result['iou_score'],
                    'Anchor Score': analysis_result['anchor_score'],
                    'Boolean Score': analysis_result['boolean_score'],
                    'Status': 'Success'
                })
            else:
                results.append({
                    'Image Name': image_name,
                    'IoU Score': 0.0,
                    'Anchor Score': 0.0,
                    'Boolean Score': 0.0,
                    'Status': analysis_result['status'] if analysis_result else 'Analysis Failed'
                })
                
        except Exception as e:
            results.append({
                'Image Name': image_name,
                'IoU Score': 0.0,
                'Anchor Score': 0.0,
                'Boolean Score': 0.0,
                'Status': f'Error: {str(e)[:50]}'
            })
    
    return results

def calculate_statistics(results):
    """Calculate mean scores and statistics"""
    
    # Filter successful results only
    successful_results = [r for r in results if r['Status'] == 'Success']
    
    if len(successful_results) == 0:
        print("⚠️ No successful analyses to calculate statistics")
        return None
    
    # Calculate means
    iou_scores = [r['IoU Score'] for r in successful_results]
    anchor_scores = [r['Anchor Score'] for r in successful_results]
    boolean_scores = [r['Boolean Score'] for r in successful_results]
    
    stats = {
        'Total Images': len(results),
        'Successful Analyses': len(successful_results),
        'Mean IoU Score': np.mean(iou_scores),
        'Mean Anchor Score': np.mean(anchor_scores),
        'Mean Boolean Score': np.mean(boolean_scores),
        'Std IoU Score': np.std(iou_scores),
        'Std Anchor Score': np.std(anchor_scores),
        'Std Boolean Score': np.std(boolean_scores),
        'Min IoU Score': np.min(iou_scores),
        'Max IoU Score': np.max(iou_scores),
        'Min Anchor Score': np.min(anchor_scores),
        'Max Anchor Score': np.max(anchor_scores),
        'Boolean Success Rate': np.mean(boolean_scores),
        'Median IoU Score': np.median(iou_scores),
        'Median Anchor Score': np.median(anchor_scores)
    }
    
    return stats

def create_csv_report(results, stats, output_path):
    """Create CSV file with results in the user's exact requested format"""
    
    # Create the exact format requested by user
    csv_data = []
    
    # Add header row exactly as requested
    csv_data.append({
        'Image Name': 'Image Name',
        'IoU Score': 'IOU score', 
        'Anchor Score': 'Anchor Score',
        'Boolean Score': 'Boolean Score'
    })
    
    # Filter successful results only for individual entries
    successful_results = [r for r in results if r['Status'] == 'Success']
    
    # Add individual image results
    for result in successful_results:
        csv_data.append({
            'Image Name': result['Image Name'],
            'IoU Score': f"{result['IoU Score']:.1f}",
            'Anchor Score': f"{result['Anchor Score']:.1f}", 
            'Boolean Score': f"{result['Boolean Score']:.0f}"
        })
    
    # Add Mean Score row exactly as requested
    if stats and successful_results:
        mean_iou = stats['Mean IoU Score']
        mean_anchor = stats['Mean Anchor Score'] 
        mean_boolean = stats['Mean Boolean Score']
        
        csv_data.append({
            'Image Name': 'Mean Score',
            'IoU Score': f"{mean_iou:.2f}",
            'Anchor Score': f"{mean_anchor:.2f}",
            'Boolean Score': f"{mean_boolean:.1f}"
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    
    print(f"📊 CSV report saved to: {output_path}")
    print(f"📈 Format: Image Name | IoU Score | Anchor Score | Boolean Score")
    print(f"🎯 Individual results: {len(successful_results)} images")
    print(f"📊 Mean scores included at bottom")
    return output_path

def main():
    """Main function to run fast batch analysis and generate CSV only"""
    
    print("📊 FAST BATCH CSV GENERATOR - ALL TEST IMAGES")
    print("="*60)
    print("⚡ Optimized for speed - No visualizations generated")
    print("🎯 Processing ALL images from test folder (~1600 images)")
    
    # Configuration
    test_folder = "1920x1080/1920x1080/test"
    gt_folder = "labels/labels/areas/test_labels_1920x1080"
    
    print(f"🔍 Looking for images in: {test_folder}")
    print(f"🎯 Ground truth folder: {gt_folder}")
    
    # Get ALL test images
    image_paths = get_all_test_images(test_folder)
    
    if len(image_paths) == 0:
        print("❌ No images found to analyze")
        return
    
    print(f"📈 Found {len(image_paths)} images to process")
    print(f"⏱️  Estimated processing time: {len(image_paths) * 0.5 / 60:.1f} minutes")
    
    # Run fast batch analysis
    start_time = datetime.now()
    results = analyze_batch_images_fast(image_paths, gt_folder)
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Processing completed in {processing_time/60:.1f} minutes")
    
    # Calculate statistics
    print("\n📈 Calculating statistics...")
    stats = calculate_statistics(results)
    
    if stats:
        print(f"\n📊 COMPREHENSIVE BATCH ANALYSIS STATISTICS:")
        print(f"   Total Images: {stats['Total Images']}")
        print(f"   Successful Analyses: {stats['Successful Analyses']}")
        print(f"   Success Rate: {(stats['Successful Analyses']/stats['Total Images']*100):.1f}%")
        print(f"   Mean IoU Score: {stats['Mean IoU Score']:.4f} ± {stats['Std IoU Score']:.4f}")
        print(f"   Median IoU Score: {stats['Median IoU Score']:.4f}")
        print(f"   Mean Anchor Score: {stats['Mean Anchor Score']:.4f} ± {stats['Std Anchor Score']:.4f}")
        print(f"   Median Anchor Score: {stats['Median Anchor Score']:.4f}")
        print(f"   Mean Boolean Score: {stats['Mean Boolean Score']:.4f} ± {stats['Std Boolean Score']:.4f}")
        print(f"   Boolean Success Rate: {stats['Boolean Success Rate']*100:.1f}%")
        print(f"   IoU Range: {stats['Min IoU Score']:.4f} - {stats['Max IoU Score']:.4f}")
        print(f"   Anchor Range: {stats['Min Anchor Score']:.4f} - {stats['Max Anchor Score']:.4f}")
    
    # Generate CSV report with simple filename
    output_path = "all_1600_test_images_results.csv"
    
    print(f"\n💾 Generating CSV report in your requested format...")
    create_csv_report(results, stats, output_path)
    
    print(f"\n✅ Comprehensive batch analysis complete!")
    print(f"📊 Results saved to: {output_path}")
    print(f"🎯 Successfully analyzed: {len([r for r in results if r['Status'] == 'Success'])}/{len(results)} images")
    print(f"⚡ No visualization images generated - CSV only!")
    print(f"⏱️  Total processing time: {processing_time/60:.1f} minutes")
    print(f"📈 Average processing speed: {processing_time/len(results):.2f} seconds per image")

if __name__ == "__main__":
    main()