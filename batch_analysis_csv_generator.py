#!/usr/bin/env python3
"""
📊 Batch Analysis CSV Generator
Analyzes 50 random test images and generates CSV with IoU, Anchor, and Boolean scores
"""

import os
import glob
import random
import pandas as pd
import numpy as np
from enhanced_exact_procedure_analysis import EnhancedExactProcedureAnalyzer
from datetime import datetime

def get_random_test_images(test_folder, num_images=50):
    """Get random test images from the test folder"""
    
    # Pattern to find all PNG images in test folder
    image_pattern = os.path.join(test_folder, "*.png")
    all_images = glob.glob(image_pattern)
    
    if len(all_images) == 0:
        print(f"❌ No images found in {test_folder}")
        return []
    
    # Select random images (or all if less than requested)
    num_to_select = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)
    
    print(f"🎲 Selected {num_to_select} random images from {len(all_images)} available images")
    return selected_images

def analyze_batch_images(image_paths, gt_folder):
    """Analyze a batch of images and return results"""
    
    analyzer = EnhancedExactProcedureAnalyzer()
    results = []
    
    total_images = len(image_paths)
    
    for i, image_path in enumerate(image_paths):
        print(f"\n🔄 Analyzing image {i+1}/{total_images}: {os.path.basename(image_path)}")
        
        try:
            # Construct ground truth path
            image_name = os.path.basename(image_path)
            gt_path = os.path.join(gt_folder, image_name)
            
            # Check if ground truth exists
            if not os.path.exists(gt_path):
                print(f"⚠️ Ground truth not found: {gt_path}")
                results.append({
                    'Image Name': image_name,
                    'IoU Score': 0.0,
                    'Anchor Score': 0.0,
                    'Boolean Score': 0.0,
                    'Status': 'GT Missing'
                })
                continue
            
            # Run analysis
            analysis_result = analyzer.analyze_exact_procedure_enhanced(image_path, gt_path)
            
            if analysis_result:
                # Extract scores
                iou_score = analysis_result['iou_score']['iou_score']
                anchor_score = analysis_result['anchor_score']['anchor_score']
                boolean_score = 1.0 if analysis_result['boolean_score']['boolean_score'] else 0.0
                
                results.append({
                    'Image Name': image_name,
                    'IoU Score': float(iou_score),
                    'Anchor Score': float(anchor_score),
                    'Boolean Score': float(boolean_score),
                    'Status': 'Success'
                })
                
                print(f"✅ Success - IoU: {iou_score:.4f}, Anchor: {anchor_score:.4f}, Boolean: {boolean_score}")
                
            else:
                print(f"❌ Analysis failed for {image_name}")
                results.append({
                    'Image Name': image_name,
                    'IoU Score': 0.0,
                    'Anchor Score': 0.0,
                    'Boolean Score': 0.0,
                    'Status': 'Analysis Failed'
                })
                
        except Exception as e:
            print(f"❌ Error analyzing {image_name}: {e}")
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
        'Boolean Success Rate': np.mean(boolean_scores)
    }
    
    return stats

def create_csv_report(results, stats, output_path):
    """Create CSV file with results in the requested format"""
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Add summary rows at the top (your requested format)
    summary_data = []
    
    # Add header row
    summary_data.append({
        'Image Name': 'BATCH ANALYSIS SUMMARY',
        'IoU Score': '',
        'Anchor Score': '',
        'Boolean Score': '',
        'Status': ''
    })
    
    # Add empty row for separation
    summary_data.append({
        'Image Name': '',
        'IoU Score': '',
        'Anchor Score': '',
        'Boolean Score': '',
        'Status': ''
    })
    
    if stats:
        # Add mean scores row
        summary_data.append({
            'Image Name': 'MEAN SCORES',
            'IoU Score': f"{stats['Mean IoU Score']:.4f}",
            'Anchor Score': f"{stats['Mean Anchor Score']:.4f}",
            'Boolean Score': f"{stats['Mean Boolean Score']:.4f}",
            'Status': 'Summary'
        })
        
        # Add standard deviation row
        summary_data.append({
            'Image Name': 'STD DEVIATION',
            'IoU Score': f"{stats['Std IoU Score']:.4f}",
            'Anchor Score': f"{stats['Std Anchor Score']:.4f}",
            'Boolean Score': f"{stats['Std Boolean Score']:.4f}",
            'Status': 'Summary'
        })
        
        # Add min scores row
        summary_data.append({
            'Image Name': 'MIN SCORES',
            'IoU Score': f"{stats['Min IoU Score']:.4f}",
            'Anchor Score': f"{stats['Min Anchor Score']:.4f}",
            'Boolean Score': '0.0000',
            'Status': 'Summary'
        })
        
        # Add max scores row
        summary_data.append({
            'Image Name': 'MAX SCORES',
            'IoU Score': f"{stats['Max IoU Score']:.4f}",
            'Anchor Score': f"{stats['Max Anchor Score']:.4f}",
            'Boolean Score': '1.0000',
            'Status': 'Summary'
        })
    
    # Add separator
    summary_data.append({
        'Image Name': '--- INDIVIDUAL RESULTS ---',
        'IoU Score': '',
        'Anchor Score': '',
        'Boolean Score': '',
        'Status': ''
    })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Combine summary and results
    final_df = pd.concat([summary_df, df], ignore_index=True)
    
    # Save to CSV
    final_df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"📊 CSV report saved to: {output_path}")
    return output_path

def main():
    """Main function to run batch analysis and generate CSV"""
    
    print("📊 BATCH ANALYSIS CSV GENERATOR")
    print("="*50)
    
    # Configuration
    test_folder = "1920x1080/1920x1080/test"
    gt_folder = "labels/labels/areas/test_labels_1920x1080"
    num_images = 50
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    print(f"🔍 Looking for images in: {test_folder}")
    print(f"🎯 Ground truth folder: {gt_folder}")
    print(f"🎲 Target number of images: {num_images}")
    
    # Get random test images
    image_paths = get_random_test_images(test_folder, num_images)
    
    if len(image_paths) == 0:
        print("❌ No images found to analyze")
        return
    
    # Run batch analysis
    print(f"\n🚀 Starting batch analysis of {len(image_paths)} images...")
    results = analyze_batch_images(image_paths, gt_folder)
    
    # Calculate statistics
    print("\n📈 Calculating statistics...")
    stats = calculate_statistics(results)
    
    if stats:
        print(f"\n📊 BATCH ANALYSIS STATISTICS:")
        print(f"   Total Images: {stats['Total Images']}")
        print(f"   Successful Analyses: {stats['Successful Analyses']}")
        print(f"   Mean IoU Score: {stats['Mean IoU Score']:.4f} ± {stats['Std IoU Score']:.4f}")
        print(f"   Mean Anchor Score: {stats['Mean Anchor Score']:.4f} ± {stats['Std Anchor Score']:.4f}")
        print(f"   Mean Boolean Score: {stats['Mean Boolean Score']:.4f} ± {stats['Std Boolean Score']:.4f}")
        print(f"   Boolean Success Rate: {stats['Boolean Success Rate']*100:.1f}%")
    
    # Generate CSV report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"batch_analysis_results_{timestamp}.csv"
    
    print(f"\n💾 Generating CSV report...")
    create_csv_report(results, stats, output_path)
    
    print(f"\n✅ Batch analysis complete!")
    print(f"📊 Results saved to: {output_path}")
    print(f"🎯 Successfully analyzed: {len([r for r in results if r['Status'] == 'Success'])}/{len(results)} images")

if __name__ == "__main__":
    main()