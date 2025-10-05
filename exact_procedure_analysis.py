#!/usr/bin/env python3
"""
🎯 Exact Procedure Implementation: IoU, Anchor & Boolean Scores
Following the exact procedure from the specification document
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

class ExactProcedureAnalyzer:
    def __init__(self, model_path="models/best_fast_runway_unet.h5"):
        """Initialize analyzer following exact procedure"""
        print("🎯 EXACT PROCEDURE: IoU, ANCHOR & BOOLEAN SCORE ANALYZER")
        print("="*60)
        
        # Load model with custom objects
        print(f"🤖 Loading model from: {model_path}")
        
        def dice_loss(y_true, y_pred):
            smooth = 1e-6
            y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            return 1 - dice

        def combined_loss(y_true, y_pred):
            return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

        def dice_coefficient(y_true, y_pred):
            smooth = 1e-6
            y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        custom_objects = {
            'dice_loss': dice_loss,
            'combined_loss': combined_loss,
            'dice_coefficient': dice_coefficient
        }
        
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ Model loaded successfully!")
        
        # Create output directory
        self.output_dir = "exact_procedure_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def draw_text_with_background(self, image, text, position, font, font_scale, text_color, bg_color, thickness):
        """Draw text with background box for better visibility"""
        # Get text size to create background box
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Add padding around text
        padding = 10
        box_coords = ((position[0] - padding, position[1] + padding),
                      (position[0] + text_w + padding, position[1] - text_h - padding))
        
        # Draw background rectangle
        cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        
        # Draw border around rectangle  
        cv2.rectangle(image, box_coords[0], box_coords[1], (0, 0, 0), 2)
        
        # Draw text on top of background
        cv2.putText(image, text, position, font, font_scale, text_color, thickness)
    
    def load_and_preprocess_image(self, image_path, target_size=(128, 128)):
        """Load and preprocess image for model prediction"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = (image.shape[1], image.shape[0])
        
        # Resize for model input
        resized = cv2.resize(image_rgb, target_size) / 255.0
        return image_rgb, resized, original_size
    
    def load_ground_truth(self, gt_path, target_size=(128, 128)):
        """Load ground truth mask"""
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise ValueError(f"Could not load ground truth: {gt_path}")
        
        # Resize and binarize
        gt_resized = cv2.resize(gt_mask, target_size)
        threshold = gt_mask.max() * 0.3
        gt_binary = (gt_resized > threshold).astype(np.uint8)
        
        return gt_mask, gt_binary
    
    def extract_edge_points(self, segmentation_mask, confidence_threshold=0.5):
        """ULTIMATE: Extract LEDG, REDG, and CTL points with robust edge detection
        Handles severely fragmented detections and multiple disconnected regions"""
        
        print(f"🔍 Analyzing segmentation with shape: {segmentation_mask.shape}")
        print(f"📊 Segmentation stats: min={segmentation_mask.min():.3f}, max={segmentation_mask.max():.3f}, mean={segmentation_mask.mean():.3f}")
        
        # ULTIMATE: Adaptive threshold selection based on image content
        binary_mask = None
        thresholds = [confidence_threshold, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
        
        for thresh in thresholds:
            test_mask = (segmentation_mask > thresh).astype(np.uint8)
            pixel_count = np.sum(test_mask)
            print(f"🎯 Threshold {thresh:.2f}: {pixel_count} pixels")
            
            if pixel_count > 50:  # Minimum viable pixels
                binary_mask = test_mask
                print(f"✅ Selected threshold: {thresh:.2f}")
                break
        
        if binary_mask is None or np.sum(binary_mask) == 0:
            print("❌ No viable mask found at any threshold")
            return None, None, None
        
        # ULTIMATE: Aggressive region connection for severely fragmented cases
        print("🔧 Applying morphological operations...")
        
        # Start with original mask
        working_mask = binary_mask.copy()
        
        # Create different kernels for various connection strategies
        kernels = {
            'small': np.ones((3,3), np.uint8),
            'medium': np.ones((5,5), np.uint8), 
            'large': np.ones((7,7), np.uint8),
            'horizontal': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 5)),  # Wider horizontal
            'vertical': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15)),    # Taller vertical
        }
        
        # Progressive morphological operations
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernels['horizontal'])
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernels['vertical'])
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernels['large'])
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_DILATE, kernels['medium'])
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernels['large'])
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, kernels['small'])
        
        print(f"📈 After morphology: {np.sum(working_mask)} pixels")
        binary_mask = working_mask
        
        # ENHANCED: Find and process contours with better region selection
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        # ENHANCED: Smart contour selection - prefer larger, more central regions
        h, w = binary_mask.shape
        center_x, center_y = w // 2, h // 2
        
        scored_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Skip tiny contours
                continue
                
            # Calculate contour center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Score based on size and proximity to image center
                distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                # Normalize distance (0 to 1, where 0 is center)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                normalized_distance = distance_from_center / max_distance
                
                # Combined score: larger area and closer to center is better
                score = area * (1 - normalized_distance * 0.5)  # Distance penalty is moderate
                scored_contours.append((score, contour, area))
        
        if not scored_contours:
            return None, None, None
        
        # Select the best contour (highest score)
        scored_contours.sort(reverse=True, key=lambda x: x[0])
        best_contour = scored_contours[0][1]
        
        # ENHANCED: Create a combined mask from top contours if they're significant
        combined_mask = np.zeros_like(binary_mask)
        
        # Include contours that are at least 30% the size of the largest one
        largest_area = scored_contours[0][2]
        for score, contour, area in scored_contours:
            if area >= largest_area * 0.3:  # Include significant regions
                cv2.fillPoly(combined_mask, [contour], 255)
        
        # Use the combined mask for edge detection
        binary_mask = combined_mask
        
        # ENHANCED: Robust edge detection using multiple methods
        h, w = binary_mask.shape
        
        # ENHANCED: Multi-method edge detection approach
        left_edge_points = []
        right_edge_points = []
        
        # Method 1: Scan-line based edge detection with smoothing
        valid_rows = 0
        for y in range(h):
            row = binary_mask[y, :]
            if np.sum(row) > 0:  # If there are white pixels in this row
                white_pixels = np.where(row > 0)[0]
                if len(white_pixels) > 0:
                    left_x = white_pixels[0]
                    right_x = white_pixels[-1]
                    
                    # ENHANCED: Only add if there's sufficient runway width
                    width = right_x - left_x
                    if width >= 2:  # Minimum runway width (reduced for fragmented cases)
                        left_edge_points.append((left_x, y))
                        right_edge_points.append((right_x, y))
                        valid_rows += 1
        
        # ENHANCED: If scan-line method fails, try contour-based method
        if len(left_edge_points) < 3 or valid_rows < 5:
            print("🔄 Scan-line method insufficient, trying contour-based approach...")
            
            # Find the best contour again for contour-based edge detection
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best_contour = max(contours, key=cv2.contourArea)
                
                # Extract points from contour
                contour_points = best_contour.reshape(-1, 2)
                
                # Separate left and right points based on x-coordinate
                center_x = np.mean(contour_points[:, 0])
                
                left_points = contour_points[contour_points[:, 0] <= center_x]
                right_points = contour_points[contour_points[:, 0] > center_x]
                
                # Sort by y-coordinate
                if len(left_points) > 0:
                    left_points = left_points[np.argsort(left_points[:, 1])]
                if len(right_points) > 0:
                    right_points = right_points[np.argsort(right_points[:, 1])]
                
                # Convert to the expected format
                if len(left_points) >= 2 and len(right_points) >= 2:
                    left_edge_points = [(int(p[0]), int(p[1])) for p in left_points]
                    right_edge_points = [(int(p[0]), int(p[1])) for p in right_points]
        
        # Final validation
        if len(left_edge_points) < 2 or len(right_edge_points) < 2:
            print("⚠️ Insufficient edge points detected")
            return None, None, None
        
        # Convert to arrays and sort by y-coordinate
        left_edge_points = np.array(left_edge_points)
        right_edge_points = np.array(right_edge_points)
        
        left_edge_points = left_edge_points[np.argsort(left_edge_points[:, 1])]
        right_edge_points = right_edge_points[np.argsort(right_edge_points[:, 1])]
        
        # ENHANCED: Smoothing for noisy edge points
        if len(left_edge_points) > 4:
            # Apply simple moving average to x-coordinates
            window_size = min(3, len(left_edge_points) // 2)
            left_x_smooth = np.convolve(left_edge_points[:, 0], np.ones(window_size)/window_size, mode='same')
            left_edge_points[:, 0] = left_x_smooth.astype(int)
        
        if len(right_edge_points) > 4:
            window_size = min(3, len(right_edge_points) // 2)
            right_x_smooth = np.convolve(right_edge_points[:, 0], np.ones(window_size)/window_size, mode='same')
            right_edge_points[:, 0] = right_x_smooth.astype(int)
        
        # Get endpoints for each edge (first and last points)
        ledg_endpoints = {
            'x0': int(left_edge_points[0][0]), 'y0': int(left_edge_points[0][1]),
            'x1': int(left_edge_points[-1][0]), 'y1': int(left_edge_points[-1][1])
        }
        
        redg_endpoints = {
            'x0': int(right_edge_points[0][0]), 'y0': int(right_edge_points[0][1]),
            'x1': int(right_edge_points[-1][0]), 'y1': int(right_edge_points[-1][1])
        }
        
        # IMPROVED: Calculate CTL as precise geometric midpoint between LEDG and REDG
        if ledg_endpoints and redg_endpoints:
            # Use floating point calculation for better precision, then round
            ctl_x0 = (ledg_endpoints['x0'] + redg_endpoints['x0']) / 2.0
            ctl_y0 = (ledg_endpoints['y0'] + redg_endpoints['y0']) / 2.0
            ctl_x1 = (ledg_endpoints['x1'] + redg_endpoints['x1']) / 2.0
            ctl_y1 = (ledg_endpoints['y1'] + redg_endpoints['y1']) / 2.0
            
            ctl_endpoints = {
                'x0': int(round(ctl_x0)),
                'y0': int(round(ctl_y0)),
                'x1': int(round(ctl_x1)),
                'y1': int(round(ctl_y1))
            }
        
        return ledg_endpoints, redg_endpoints, ctl_endpoints
    
    def create_polygon_from_edges(self, ledg_points, redg_points, ctl_points):
        """Create polygon by joining points from LEDG, REDG and CTL as specified"""
        
        if not all([ledg_points, redg_points, ctl_points]):
            return None
        
        # Create polygon vertices in proper order to form a valid quadrilateral
        # Order: LEDG_start -> REDG_start -> REDG_end -> LEDG_end -> back to LEDG_start
        polygon_vertices = [
            (ledg_points['x0'], ledg_points['y0']),  # LEDG start
            (redg_points['x0'], redg_points['y0']),  # REDG start  
            (redg_points['x1'], redg_points['y1']),  # REDG end
            (ledg_points['x1'], ledg_points['y1'])   # LEDG end
        ]
        
        # Create polygon
        try:
            polygon = Polygon(polygon_vertices)
            if not polygon.is_valid:
                # Try to fix invalid polygon
                polygon = polygon.buffer(0)
            return polygon
        except:
            return None
    
    def calculate_anchor_score(self, segmentation_mask, polygon, confidence_threshold=0.5):
        """Calculate Anchor Score: Check number of pixels segmented as runway that fall within polygon"""
        
        if polygon is None:
            return 0.0
        
        # Get segmented runway pixels
        binary_mask = (segmentation_mask > confidence_threshold).astype(np.uint8)
        runway_pixels = np.where(binary_mask > 0)
        
        if len(runway_pixels[0]) == 0:
            return 0.0
        
        # Count pixels inside polygon
        pixels_in_polygon = 0
        total_runway_pixels = len(runway_pixels[0])
        
        for i in range(total_runway_pixels):
            pixel_point = Point(runway_pixels[1][i], runway_pixels[0][i])  # (x, y)
            if polygon.contains(pixel_point):
                pixels_in_polygon += 1
        
        # Anchor score as ratio (0.0 to 1.0)
        anchor_score = pixels_in_polygon / total_runway_pixels if total_runway_pixels > 0 else 0.0
        
        return {
            'anchor_score': float(anchor_score),
            'pixels_in_polygon': int(pixels_in_polygon),
            'total_runway_pixels': int(total_runway_pixels)
        }
    
    def calculate_boolean_score(self, ledg_points, redg_points, ctl_points, polygon):
        """Calculate Boolean Score: Check if CTL points are within polygon and between LEDG and REDG"""
        
        if not all([ledg_points, redg_points, ctl_points, polygon]):
            return False
        
        # Check CTL endpoints
        ctl_point1 = Point(ctl_points['x0'], ctl_points['y0'])
        ctl_point2 = Point(ctl_points['x1'], ctl_points['y1'])
        
        # Check if CTL points are within polygon (with small buffer for numerical precision)
        buffered_polygon = polygon.buffer(0.5)  # Small buffer for edge cases
        ctl1_in_polygon = buffered_polygon.contains(ctl_point1) or polygon.contains(ctl_point1)
        ctl2_in_polygon = buffered_polygon.contains(ctl_point2) or polygon.contains(ctl_point2)
        
        # Check if CTL points are between LEDG and REDG
        # Simple check: CTL x-coordinates should be between LEDG and REDG x-coordinates
        ledg_x_range = [ledg_points['x0'], ledg_points['x1']]
        redg_x_range = [redg_points['x0'], redg_points['x1']]
        
        min_ledg_x, max_ledg_x = min(ledg_x_range), max(ledg_x_range)
        min_redg_x, max_redg_x = min(redg_x_range), max(redg_x_range)
        
        # CTL should be between left and right edges
        overall_min_x = min(min_ledg_x, min_redg_x)
        overall_max_x = max(max_ledg_x, max_redg_x)
        
        ctl1_between = overall_min_x <= ctl_points['x0'] <= overall_max_x
        ctl2_between = overall_min_x <= ctl_points['x1'] <= overall_max_x
        
        # Boolean score: True if CTL points are both in polygon AND between edges
        boolean_score = (ctl1_in_polygon and ctl1_between) and (ctl2_in_polygon and ctl2_between)
        
        return {
            'boolean_score': boolean_score,
            'ctl1_in_polygon': ctl1_in_polygon,
            'ctl2_in_polygon': ctl2_in_polygon,
            'ctl1_between_edges': ctl1_between,
            'ctl2_between_edges': ctl2_between
        }
    
    def calculate_iou_score(self, pred_mask, gt_mask, threshold=0.5):
        """Calculate standard IoU score for runway segmentation"""
        
        # Binarize predictions
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        gt_binary = gt_mask.astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # IoU score
        iou_score = intersection / union if union > 0 else 0.0
        
        return {
            'iou_score': float(iou_score),
            'intersection': int(intersection),
            'union': int(union),
            'pred_pixels': int(pred_binary.sum()),
            'gt_pixels': int(gt_binary.sum())
        }
    
    def create_procedure_visualization(self, original_image, pred_mask, gt_mask, 
                                     ledg_points, redg_points, ctl_points, polygon,
                                     anchor_result, boolean_result, iou_result, image_name):
        """Create visualization following the exact procedure"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Exact Procedure Analysis: {image_name}', fontsize=16, fontweight='bold')
        
        # 1. Original Image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Input RGB Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Segmented Image
        axes[0, 1].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Segmented Image', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Ground Truth
        axes[0, 2].imshow(gt_mask, cmap='gray')
        axes[0, 2].set_title('Ground Truth Segment Mask', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # 4. Left Edge, Right Edge, and Control Line (Separate Lines) - HIGH RESOLUTION
        edge_viz_full = original_image.copy()  # Use full resolution original image
        h_model, w_model = pred_mask.shape
        h_orig, w_orig = original_image.shape[:2]
        
        # Calculate scaling factors to map model coordinates to original image coordinates
        scale_x = w_orig / w_model
        scale_y = h_orig / h_model
        
        if all([ledg_points, redg_points, ctl_points]):
            # Scale coordinates to original image resolution
            ledg_x0_scaled = int(ledg_points['x0'] * scale_x)
            ledg_y0_scaled = int(ledg_points['y0'] * scale_y)
            ledg_x1_scaled = int(ledg_points['x1'] * scale_x)
            ledg_y1_scaled = int(ledg_points['y1'] * scale_y)
            
            redg_x0_scaled = int(redg_points['x0'] * scale_x)
            redg_y0_scaled = int(redg_points['y0'] * scale_y)
            redg_x1_scaled = int(redg_points['x1'] * scale_x)
            redg_y1_scaled = int(redg_points['y1'] * scale_y)
            
            ctl_x0_scaled = int(ctl_points['x0'] * scale_x)
            ctl_y0_scaled = int(ctl_points['y0'] * scale_y)
            ctl_x1_scaled = int(ctl_points['x1'] * scale_x)
            ctl_y1_scaled = int(ctl_points['y1'] * scale_y)
            
            # Calculate line thickness based on image size for better visibility
            line_thickness = max(3, int(min(w_orig, h_orig) / 200))
            circle_radius = max(4, int(min(w_orig, h_orig) / 150))
            
            # COMPLETELY SEPARATE LINES - NO MEETING POINTS AT ALL
            
            # Create larger gap to ensure complete separation
            gap_offset = 5  # Increased gap for better separation
            y_offset = 3    # Additional Y offset to prevent meeting points
            
            # 1. Draw Left Edge (LEDG) in ORANGE - completely isolated
            ledg_color = (0, 165, 255)  # Orange in BGR
            # Move left edge inward and adjust Y coordinates
            ledg_x0_adj = ledg_x0_scaled + gap_offset
            ledg_y0_adj = ledg_y0_scaled + y_offset
            ledg_x1_adj = ledg_x1_scaled + gap_offset
            ledg_y1_adj = ledg_y1_scaled - y_offset
            cv2.line(edge_viz_full, (ledg_x0_adj, ledg_y0_adj), 
                    (ledg_x1_adj, ledg_y1_adj), ledg_color, line_thickness)
            
            # 2. Draw Right Edge (REDG) in GREEN - completely isolated with better alignment
            redg_color = (0, 255, 0)  # Green in BGR
            # Move right edge inward and adjust Y coordinates
            redg_x0_adj = redg_x0_scaled - gap_offset
            redg_y0_adj = redg_y0_scaled + y_offset
            redg_x1_adj = redg_x1_scaled - gap_offset  
            redg_y1_adj = redg_y1_scaled - y_offset
            cv2.line(edge_viz_full, (redg_x0_adj, redg_y0_adj), 
                    (redg_x1_adj, redg_y1_adj), redg_color, line_thickness)
            
            # 3. Draw Control Line in BRIGHT RED - perfectly centered with proper alignment
            ctl_color = (0, 0, 255)  # Bright Red in BGR
            control_line_thickness = max(line_thickness + 1, 5)  # Slightly thicker
            
            # Calculate control line as perfect geometric center between ADJUSTED edges
            perfect_ctl_x0 = int((ledg_x0_adj + redg_x0_adj) / 2)
            perfect_ctl_y0 = int((ledg_y0_adj + redg_y0_adj) / 2)
            perfect_ctl_x1 = int((ledg_x1_adj + redg_x1_adj) / 2)
            perfect_ctl_y1 = int((ledg_y1_adj + redg_y1_adj) / 2)
            
            # Ensure control line has its own Y-space to avoid meeting points
            perfect_ctl_y0 = perfect_ctl_y0 - 1  # Slight offset
            perfect_ctl_y1 = perfect_ctl_y1 + 1  # Slight offset
            
            cv2.line(edge_viz_full, (perfect_ctl_x0, perfect_ctl_y0), 
                    (perfect_ctl_x1, perfect_ctl_y1), ctl_color, control_line_thickness)
            
            # Add small endpoint markers with adjusted coordinates (completely separate)
            marker_size = 2  # Smaller markers to reduce visual impact
            cv2.circle(edge_viz_full, (ledg_x0_adj, ledg_y0_adj), marker_size, ledg_color, -1)
            cv2.circle(edge_viz_full, (ledg_x1_adj, ledg_y1_adj), marker_size, ledg_color, -1)
            cv2.circle(edge_viz_full, (redg_x0_adj, redg_y0_adj), marker_size, redg_color, -1)
            cv2.circle(edge_viz_full, (redg_x1_adj, redg_y1_adj), marker_size, redg_color, -1)
            cv2.circle(edge_viz_full, (perfect_ctl_x0, perfect_ctl_y0), marker_size + 1, ctl_color, -1)
            cv2.circle(edge_viz_full, (perfect_ctl_x1, perfect_ctl_y1), marker_size + 1, ctl_color, -1)
            
            # ADD TEXT LABELS WITH BACKGROUND BOXES (like reference image)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.8, int(min(w_orig, h_orig) / 800))
            font_thickness = max(2, int(font_scale * 2))
            
            # Calculate label positions (offset from lines to avoid overlap)
            label_offset_x = int(w_orig * 0.05)
            label_offset_y = int(h_orig * 0.03)
            
            # Left Edge Label (use adjusted coordinates)
            left_label_x = max(10, ledg_x0_adj - label_offset_x)
            left_label_y = max(40, ledg_y0_adj - label_offset_y)
            self.draw_text_with_background(edge_viz_full, "Left Edge", (left_label_x, left_label_y), 
                                         font, font_scale, ledg_color, (255, 255, 255), font_thickness)
            
            # Right Edge Label (use adjusted coordinates)
            right_label_x = min(w_orig - 150, redg_x0_adj + label_offset_x)
            right_label_y = max(40, redg_y0_adj - label_offset_y)
            self.draw_text_with_background(edge_viz_full, "Right Edge", (right_label_x, right_label_y), 
                                         font, font_scale, redg_color, (255, 255, 255), font_thickness)
            
            # Control Line Label (use perfectly centered coordinates)
            ctl_label_x = max(10, min(w_orig - 180, perfect_ctl_x1 + label_offset_x))
            ctl_label_y = min(h_orig - 20, perfect_ctl_y1 + label_offset_y)
            self.draw_text_with_background(edge_viz_full, "Control Line", (ctl_label_x, ctl_label_y), 
                                         font, font_scale, ctl_color, (255, 255, 255), font_thickness)
            
            # Draw polygon outline on full resolution (optional)
            if polygon:
                poly_coords_scaled = []
                for coord in polygon.exterior.coords:
                    x_scaled = int(coord[0] * scale_x)
                    y_scaled = int(coord[1] * scale_y)
                    poly_coords_scaled.append([x_scaled, y_scaled])
                poly_coords_scaled = np.array(poly_coords_scaled, dtype=np.int32)
                cv2.polylines(edge_viz_full, [poly_coords_scaled], True, (255, 0, 255), 2)
        
        axes[1, 0].imshow(edge_viz_full)
        axes[1, 0].set_title('Left Edge (Orange), Right Edge (Green), Control Line (Red) - With Labels', 
                            fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 5. Scores Display
        scores_text = f"""
EXACT PROCEDURE SCORES:

IoU Score:
• Value: {iou_result['iou_score']:.4f}
• Intersection: {iou_result['intersection']:,} pixels
• Union: {iou_result['union']:,} pixels

Anchor Score:
• Value: {anchor_result['anchor_score']:.4f}
• Pixels in Polygon: {anchor_result['pixels_in_polygon']:,}
• Total Runway Pixels: {anchor_result['total_runway_pixels']:,}

Boolean Score:
• Value: {boolean_result['boolean_score']}
• CTL in Polygon: {boolean_result['ctl1_in_polygon']} & {boolean_result['ctl2_in_polygon']}
• CTL Between Edges: {boolean_result['ctl1_between_edges']} & {boolean_result['ctl2_between_edges']}
        """
        
        axes[1, 1].text(0.05, 0.95, scores_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        # 6. Edge Points Details
        points_text = "EDGE POINTS:\n\n"
        
        if ledg_points:
            points_text += f"LEDG Points:\n• (x0,y0): ({ledg_points['x0']}, {ledg_points['y0']})\n• (x1,y1): ({ledg_points['x1']}, {ledg_points['y1']})\n\n"
        
        if redg_points:
            points_text += f"REDG Points:\n• (x0,y0): ({redg_points['x0']}, {redg_points['y0']})\n• (x1,y1): ({redg_points['x1']}, {redg_points['y1']})\n\n"
        
        if ctl_points:
            points_text += f"CTL Points:\n• (x0,y0): ({ctl_points['x0']}, {ctl_points['y0']})\n• (x1,y1): ({ctl_points['x1']}, {ctl_points['y1']})\n"
        
        axes[1, 2].text(0.05, 0.95, points_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{image_name}_exact_procedure.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Exact procedure visualization saved: {output_path}")
        
        return output_path
    
    def analyze_exact_procedure(self, image_path, gt_path):
        """Perform analysis following exact procedure specification"""
        print(f"\n🔍 EXACT PROCEDURE ANALYSIS: {os.path.basename(image_path)}")
        print("="*60)
        
        try:
            # Load and preprocess
            original_image, processed_image, original_size = self.load_and_preprocess_image(image_path)
            gt_original, gt_processed = self.load_ground_truth(gt_path)
            
            # Get model prediction
            pred_input = np.expand_dims(processed_image, axis=0)
            prediction = self.model.predict(pred_input, verbose=0)[0, :, :, 0]
            
            print("📊 Step 1: Extracting Left Edge, Right Edge, and Control Line points...")
            # Extract edge points as specified in procedure
            ledg_points, redg_points, ctl_points = self.extract_edge_points(prediction)
            
            print("🔲 Step 2: Creating polygon from edge points...")
            # Create polygon from edge points
            polygon = self.create_polygon_from_edges(ledg_points, redg_points, ctl_points)
            
            print("🎯 Step 3: Calculating Anchor Score...")
            # Calculate Anchor Score
            anchor_result = self.calculate_anchor_score(prediction, polygon)
            
            print("✅ Step 4: Calculating Boolean Score...")
            # Calculate Boolean Score
            boolean_result = self.calculate_boolean_score(ledg_points, redg_points, ctl_points, polygon)
            
            print("📈 Step 5: Calculating IoU Score...")
            # Calculate IoU Score
            iou_result = self.calculate_iou_score(prediction, gt_processed)
            
            # Compile results
            results = {
                'image_path': image_path,
                'ground_truth_path': gt_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'edge_points': {
                    'ledg_points': ledg_points,
                    'redg_points': redg_points,
                    'ctl_points': ctl_points
                },
                'polygon_created': polygon is not None,
                'anchor_score': anchor_result,
                'boolean_score': boolean_result,
                'iou_score': iou_result
            }
            
            # Create visualization
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            print("🎨 Step 6: Creating visualization...")
            viz_path = self.create_procedure_visualization(
                original_image, prediction, gt_processed,
                ledg_points, redg_points, ctl_points, polygon,
                anchor_result, boolean_result, iou_result, image_name
            )
            
            # Save JSON results
            json_path = os.path.join(self.output_dir, f"{image_name}_exact_procedure.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved: {json_path}")
            
            # Print summary
            self.print_exact_summary(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in exact procedure analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_exact_summary(self, results):
        """Print exact procedure summary"""
        print("\n" + "="*60)
        print("🎯 EXACT PROCEDURE ANALYSIS SUMMARY")
        print("="*60)
        
        # IoU Score
        iou = results['iou_score']
        print(f"""
📈 IoU SCORE:
   • IoU Score: {iou['iou_score']:.4f}
   • Intersection: {iou['intersection']:,} pixels
   • Union: {iou['union']:,} pixels
        """)
        
        # Anchor Score
        anchor = results['anchor_score']
        print(f"""
🎯 ANCHOR SCORE:
   • Anchor Score: {anchor['anchor_score']:.4f}
   • Pixels in Polygon: {anchor['pixels_in_polygon']:,}
   • Total Runway Pixels: {anchor['total_runway_pixels']:,}
        """)
        
        # Boolean Score
        boolean = results['boolean_score']
        print(f"""
✅ BOOLEAN SCORE:
   • Boolean Score: {boolean['boolean_score']}
   • Control Line Points in Polygon: {boolean['ctl1_in_polygon']} & {boolean['ctl2_in_polygon']}
   • Control Line Points Between Edges: {boolean['ctl1_between_edges']} & {boolean['ctl2_between_edges']}
        """)
        
        # Edge Points
        edges = results['edge_points']
        if edges['ledg_points']:
            ledg = edges['ledg_points']
            print(f"""
📍 LEDG POINTS:
   • (x0,y0): ({ledg['x0']}, {ledg['y0']})
   • (x1,y1): ({ledg['x1']}, {ledg['y1']})
            """)
        
        if edges['redg_points']:
            redg = edges['redg_points']
            print(f"""
📍 REDG POINTS:
   • (x0,y0): ({redg['x0']}, {redg['y0']})
   • (x1,y1): ({redg['x1']}, {redg['y1']})
            """)
        
        if edges['ctl_points']:
            ctl = edges['ctl_points']
            print(f"""
📍 CONTROL LINE POINTS:
   • (x0,y0): ({ctl['x0']}, {ctl['y0']})
   • (x1,y1): ({ctl['x1']}, {ctl['y1']})
            """)
        
        print("\n🎉 EXACT PROCEDURE ANALYSIS COMPLETE!")
        print("="*60)


def main():
    """Main execution following exact procedure"""
    # Initialize analyzer
    analyzer = ExactProcedureAnalyzer()

    # Target image and ground truth for KJAC01_1_7FNLImage2
    image_path = "D://as of 26-09/1920x1080/1920x1080/test/YBAS12_1_3FNLImage3.png"
    gt_path = "D://as of 26-09/labels/labels/areas/test_labels_1920x1080/YBAS12_1_3FNLImage3.png"

    # Perform exact procedure analysis
    results = analyzer.analyze_exact_procedure(image_path, gt_path)
    
    if results:
        print(f"\n✅ Exact procedure analysis complete! Check '{analyzer.output_dir}' for results.")
    else:
        print("\n❌ Exact procedure analysis failed!")


if __name__ == "__main__":
    main()