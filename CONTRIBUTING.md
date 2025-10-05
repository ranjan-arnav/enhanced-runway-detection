# Contributing to Enhanced Runway Detection System

⚠️ **NOTICE: This project is under PROPRIETARY LICENSE**

**🚫 CONTRIBUTIONS NOT ACCEPTED**

This repository is proprietary software owned by Arnav Ranjan. Due to the restrictive license:

- **NO CODE MODIFICATIONS** are permitted
- **NO PULL REQUESTS** will be accepted  
- **NO COLLABORATIVE DEVELOPMENT** is allowed
- **NO DERIVATIVE WORKS** can be created

## 📖 View-Only Repository

This repository is shared for **educational and portfolio purposes only**. You may:
- ✅ View and study the code for learning
- ✅ Reference the techniques in academic discussions
- ✅ Use as inspiration for your own separate projects

## 📞 Contact for Licensing

If you're interested in:
- Commercial licensing
- Code modifications
- Collaboration opportunities
- Academic partnerships

Please contact **Arnav Ranjan** directly for licensing inquiries.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.7+
- Git

### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/runway-detection.git
cd runway-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Include detailed description and steps to reproduce
- Add relevant labels (bug, enhancement, documentation)
- Include system information and error logs

### Submitting Changes

1. **Fork the Repository**
   ```bash
   git fork https://github.com/yourusername/runway-detection.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Changes**
   ```bash
   python -m pytest tests/
   python enhanced_exact_procedure_analysis.py  # Run basic test
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 📝 Coding Standards

### Python Style Guide
- Follow PEP 8
- Use descriptive variable names
- Add docstrings to all functions and classes
- Maximum line length: 100 characters

### Example Function Documentation
```python
def analyze_exact_procedure_enhanced(self, image_path, gt_path):
    """
    Enhanced exact procedure analysis with ground truth integration.
    
    Args:
        image_path (str): Path to input image
        gt_path (str): Path to ground truth mask
        
    Returns:
        dict: Analysis results with IoU, Anchor, Boolean scores
        
    Raises:
        FileNotFoundError: If image or ground truth files don't exist
    """
```

### Code Organization
- Keep functions focused and small
- Use meaningful class and function names  
- Group related functionality in classes
- Add type hints where helpful

## 🧪 Testing Guidelines

### Test Structure
```python
def test_enhanced_analysis():
    """Test enhanced analysis functionality"""
    analyzer = EnhancedExactProcedureAnalyzer()
    
    # Test with sample data
    results = analyzer.analyze_exact_procedure_enhanced(
        "test_data/sample_image.png",
        "test_data/sample_gt.png"
    )
    
    # Assertions
    assert results is not None
    assert 0 <= results['iou_score'] <= 1
    assert 0 <= results['anchor_score'] <= 1
    assert results['boolean_score'] in [True, False]
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_enhanced_analysis.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## 📊 Performance Guidelines

### Optimization Principles
- Minimize memory usage for large batch processing
- Use vectorized operations where possible
- Profile code for bottlenecks
- Cache expensive computations

### Benchmarking
```python
import time

start_time = time.time()
# Your code here
processing_time = time.time() - start_time
print(f"Processing time: {processing_time:.2f} seconds")
```

## 📁 Project Structure

```
📦 runway-detection/
├── 📄 enhanced_exact_procedure_analysis.py  # Main enhanced system
├── 📄 exact_procedure_analysis.py           # Base analysis
├── 📄 improved_unet_training.py             # Training pipeline
├── 📁 tests/                                # Test files
├── 📁 docs/                                 # Documentation
├── 📁 examples/                             # Usage examples
└── 📁 models/                               # Pre-trained models
```

## 🎨 Documentation Guidelines

### Adding New Features
- Update README.md with new functionality
- Add docstrings to all new functions/classes
- Include usage examples
- Update CHANGELOG.md

### Documentation Style
- Use clear, concise language
- Include code examples
- Add performance notes where relevant
- Use emojis for better readability

## 🐛 Bug Reports

### Good Bug Report Template
```
**Bug Description**
Clear description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Expected vs actual behavior

**Environment**
- OS: Windows 10 / Ubuntu 20.04 / macOS
- Python: 3.8.5
- TensorFlow: 2.10.0
- OpenCV: 4.7.0

**Error Logs**
```
Paste error logs here
```

**Additional Context**
Any other relevant information
```

## 🚀 Feature Requests

### Enhancement Template
```
**Feature Description**
What functionality would you like added?

**Use Case** 
Why is this feature needed?

**Proposed Implementation**
How could this be implemented?

**Alternatives**
Other solutions you've considered
```

## 📋 Review Process

### PR Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass and cover new functionality  
- [ ] Documentation updated
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

### Merge Criteria
- All tests pass
- Code review approved
- Documentation complete
- No merge conflicts

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- Release notes for major features

## 📞 Getting Help

- **Documentation**: Check README.md and docstrings
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

Thank you for contributing to the Enhanced Runway Detection System! 🛬✈️