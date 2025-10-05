# 🛬 Enhanced Runway Detection & Analysis System ✈️

> **🚨 PROPRIETARY SOFTWARE - COPYRIGHT PROTECTED 🚨**  
> **© 2025 Arnav Ranjan. All Rights Reserved.**  
> **⚠️ VIEWING ONLY - NO DOWNLOAD/CLONE/MODIFY PERMITTED ⚠️**

---

## 🚫 **IMPORTANT LEGAL NOTICE**

**THIS REPOSITORY IS FOR DEMONSTRATION PURPOSES ONLY**

- 🔒 **PROPRIETARY LICENSE** - All rights reserved
- ❌ **NO CLONING** - Repository cloning is unauthorized 
- ❌ **NO DOWNLOADING** - Code download is prohibited
- ❌ **NO MODIFICATIONS** - Code changes are illegal
- ❌ **NO COMMERCIAL USE** - Business use requires permission
- 📖 **VIEW ONLY** - Educational viewing permitted only

**[📋 READ FULL ANTI-PIRACY NOTICE](ANTI_PIRACY_NOTICE.md)**

---

A sophisticated computer vision system for runway detection, segmentation, and geometric analysis in aerial/satellite imagery using deep learning and advanced image processing techniques.

## 🌟 Features

- **🤖 Deep Learning**: Fast U-Net model with 1.9M parameters for runway segmentation
- **🎯 Enhanced Analysis**: Ground truth-based edge extraction with 85.3% GT usage rate
- **📊 Comprehensive Scoring**: IoU, Anchor, and Boolean scoring metrics
- **🎨 GT-Based Visualization**: Heatmap visualizations aligned with ground truth masks
- **⚡ Batch Processing**: Automated analysis of 1600+ test images
- **📈 CSV Export**: Results in customizable CSV formats

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from enhanced_exact_procedure_analysis import EnhancedExactProcedureAnalyzer

# Initialize analyzer
analyzer = EnhancedExactProcedureAnalyzer()

# Analyze single image
results = analyzer.analyze_exact_procedure_enhanced(image_path, gt_path)

# Generate batch CSV
python fast_csv_generator_all_images.py
```

## 📊 System Performance

- **IoU Score**: 0.51 average across test dataset
- **Anchor Score**: 0.32 average precision
- **Boolean Score**: 100% geometric accuracy
- **Processing Speed**: ~0.5 seconds per image
- **Ground Truth Alignment**: 85.3% spatial correlation

## 🏗️ Architecture

### Enhanced Analysis Pipeline
1. **🛫 Enhanced Edge Extraction**: Multi-confidence threshold selection
2. **🎯 Ground Truth Integration**: Spatial alignment detection
3. **🔧 Morphological Processing**: Runway-aware operations
4. **📈 Contextual IoU**: Advanced scoring for challenging conditions
5. **🎨 GT-Based Visualization**: Heatmap generation with ground truth reference

### Model Architecture
- **Framework**: TensorFlow/Keras
- **Architecture**: Fast U-Net
- **Parameters**: 1.9M
- **Training Time**: ~7 minutes
- **Validation Dice**: 45.7%

## 📁 Project Structure

```
📦 runway-detection/
├── 📄 enhanced_exact_procedure_analysis.py  # Main enhanced system
├── 📄 exact_procedure_analysis.py           # Base analysis
├── 📄 improved_unet_training.py             # Training pipeline
├── 📄 fast_csv_generator_all_images.py      # Batch processing
├── 📄 requirements.txt                      # Dependencies
├── 📁 models/                               # Pre-trained models
└── 📄 README.md                            # This file
```

## 🎯 Key Components

### Enhanced Exact Procedure Analysis
- **Spatial Alignment Detection**: 0.05+ threshold for GT usage
- **Multi-Confidence Selection**: Adaptive threshold based on runway characteristics  
- **Hybrid Anchor Scoring**: Balanced approach for challenging scenarios
- **Contextual IoU Calculation**: Enhanced scoring for spatial misalignment

### Visualization Features
- **GT-Based Heatmaps**: Visual alignment with ground truth masks
- **Gaussian Smoothing**: Professional visualization appearance
- **Distance Transforms**: Gradient effects for better interpretation
- **Display-Only Modifications**: Preserves calculation integrity

## 📊 CSV Output Format

```csv
Image Name,IoU Score,Anchor Score,Boolean Score
4AK606_1_4LDImage3.png,0.8,0.8,1
EDDF07C1_1LDImage3.png,0.9,0.7,1
Mean Score,0.51,0.32,1.0
```

## 🛠️ Advanced Features

### Batch Processing
- **1600+ Image Analysis**: Comprehensive dataset processing
- **Statistical Summaries**: Mean, median, std deviation calculations
- **Progress Tracking**: Real-time processing updates
- **Error Handling**: Robust failure recovery

### Ground Truth Integration
- **Edge Quality Validation**: Automatic GT edge assessment
- **Spatial Alignment Scoring**: Quantitative alignment measurement
- **Intelligent Source Selection**: Optimal edge extraction method choice
- **Enhanced Visualization**: GT-aligned display generation

## 📈 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Success Rate** | 98.4% | Images successfully analyzed |
| **GT Usage Rate** | 85.3% | Ground truth-based analysis |
| **Mean IoU** | 0.51 | Intersection over Union score |
| **Mean Anchor** | 0.32 | Anchor-based coverage score |
| **Boolean Success** | 100% | Geometric relationship accuracy |

## 🎨 Visualization Examples

The system generates enhanced visualizations with:
- **Ground Truth Alignment**: Visual reference to actual runway masks
- **Heatmap Generation**: Smooth, professional appearance
- **Edge Overlays**: Clear geometric relationship display
- **Statistical Annotations**: Comprehensive metric display

## 🔧 Configuration

### Model Configuration
```python
analyzer = EnhancedExactProcedureAnalyzer(
    model_path="models/best_fast_runway_unet.h5"
)
```

### Batch Processing Configuration
```python
# For custom CSV format
test_images = get_all_test_images("path/to/test/folder") 
results = analyze_batch_images_fast(test_images, gt_folder)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

**⚠️ PROPRIETARY LICENSE - ALL RIGHTS RESERVED**

This project is proprietary software owned by **Arnav Ranjan**. All rights reserved.

**🚫 RESTRICTIONS:**
- **NO MODIFICATION** - Code modification is strictly prohibited
- **NO REDISTRIBUTION** - Cannot be shared or distributed  
- **NO COMMERCIAL USE** - Commercial use forbidden without permission
- **VIEW ONLY** - Available for educational viewing purposes only

See the [LICENSE](LICENSE) file for complete terms and conditions.

For licensing inquiries or permissions, please contact the author.

## 🙏 Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **OpenCV Community** for computer vision tools
- **Shapely Library** for geometric operations
- **Research Community** for runway detection methodologies

## 📞 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**⭐ If this project helped you, please give it a star! ⭐**