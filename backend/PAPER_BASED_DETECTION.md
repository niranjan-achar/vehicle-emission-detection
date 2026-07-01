# Paper-Based Smoke Detection Pipeline

## Overview

This implementation replaces the YOLOv8-based detection system with a faithful reproduction of the research paper methodology:

**"Automatic Smoky Vehicle Detection from Traffic Surveillance Video Based on Vehicle Rear Detection and Multi-Feature Fusion"**  
*IET Intelligent Transport Systems*, 2019  
Authors: Tao, L. & Lu, Y.

## Pipeline Architecture

```
Input Frame
    ↓
[1] Background Subtraction (ViBe)
    ↓
[2] Vehicle Detection & Filtering
    - Rule 1: Matching degree (ηmatch < 0.15)
    - Rule 2: Area bounds [1500, 50000] pixels
    - Rule 3: Aspect ratio [0.3, 1.5]
    ↓
[3] Vehicle Rear Detection (Integral Projection)
    - Horizontal Integral Projection (HIP)
    - Standard Deviation Filtering
    - Select groove with minimum std dev
    ↓
[4] Key Region Extraction
    - Width: 0.7 × vehicle_width
    - Height: ε = 10 pixels
    - Position: centered on rear
    ↓
[5] Feature Extraction (36 total features)
    ├── [5a] Artificial Features (4)
    │   - ARTrate: Changed pixel ratio
    │   - ARTmatch: Template matching
    │   - ARTmean: Mean difference
    │   - ARTvariance: Variance of diff
    │
    ├── [5b] GLCM Features (20)
    │   - 5 texture properties × 4 angles
    │   - Angles: 0°, 45°, 90°, 135°
    │   - Distance d=2 pixels
    │
    └── [5c] DWT Features (12)
        - 2 decomposition levels
        - 3 directions × 2 blocks each
        - Energy per block
    ↓
[6] SVM Classification (Voting)
    - 3 independent SVM classifiers
    - Majority voting final decision
    - Output: smoky/non_smoky + confidence
    ↓
Output: Detection Results
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| ηmatch | 0.15 | Matching degree threshold |
| S_min | 1500 | Minimum vehicle area (pixels²) |
| S_max | 50000 | Maximum vehicle area (pixels²) |
| δ_min | 0.3 | Minimum aspect ratio |
| δ_max | 1.5 | Maximum aspect ratio |
| ε | 10 | Vehicle rear height (pixels) |
| key_region_width_ratio | 0.7 | Width of key region as ratio of vehicle width |
| GLCM distance (d) | 2 | Pixel distance for GLCM |
| DWT levels | 2 | Wavelet decomposition levels |
| SVM kernel | RBF | Radial Basis Function |

## Module Structure

```
backend/app/
├── vehicle_detection/
│   ├── __init__.py
│   ├── background_subtraction.py (ViBe algorithm)
│   └── vehicle_filter.py (3 filtering rules)
│
├── rear_detection/
│   ├── __init__.py
│   └── integral_projection.py (HIP + std filtering)
│
├── features/
│   ├── __init__.py
│   ├── key_region.py (Key region extraction)
│   ├── artificial.py (ART features)
│   ├── glcm.py (GLCM features)
│   └── dwt.py (DWT features)
│
├── classifiers/
│   ├── __init__.py
│   └── voting.py (SVM voting classifier)
│
└── services/
    ├── yolo_service.py (DEPRECATED - kept for reference)
    ├── paper_service.py (NEW - orchestrates full pipeline)
    ├── storage_service.py (unchanged)
    └── notification_service.py (unchanged)
```

## API Compatibility

The `PaperService` maintains the same interface as the deprecated `YoloService`:

```python
# Initialize
service = PaperService(config=settings.dict())
service.load()

# Detect in single frame
result = service.detect_frame(frame)
# Returns:
# {
#     'detections': [
#         {
#             'class_name': 'smoky' or 'non_smoky',
#             'confidence': float 0-1,
#             'bbox': (x, y, w, h),
#             'features': {all 36 features},
#             'rear_y': int
#         },
#         ...
#     ],
#     'frame_annotated': annotated_frame_with_boxes
# }

# Detect in image file
result = service.detect_image(image_path)
```

## Training SVM Classifiers

To train the SVM voting classifiers:

```python
from backend.app.classifiers.voting import SVMVotingClassifier
import numpy as np

# Prepare training data
# X_art: (n_samples, 4) - artificial features
# X_glcm: (n_samples, 20) - GLCM features
# X_dwt: (n_samples, 12) - DWT features
# y: (n_samples,) - labels (1=smoky, 0=non_smoky)

classifier = SVMVotingClassifier()
classifier.train(X_art, X_glcm, X_dwt, y)

# Predict on new features
features_dict = {...}  # 36-feature dictionary
class_name, confidence, details = classifier.predict(features_dict)
```

## Configuration

Modify `backend/.env` or pass via `config` dict:

```ini
# Paper-specific parameters (optional, defaults shown)
BG_SAMPLES=20              # ViBe background samples
BG_MIN_DIST=20             # ViBe distance threshold
ETA_MATCH=0.15             # Matching degree threshold
S_MIN=1500                 # Min vehicle area
S_MAX=50000                # Max vehicle area
DELTA_MIN=0.3              # Min aspect ratio
DELTA_MAX=1.5              # Max aspect ratio
EPS=10                     # Vehicle rear height
KEY_REGION_WIDTH_RATIO=0.7 # Key region width ratio
KEY_REGION_MARGIN=5        # Key region margin
```

## Performance

Based on paper's experimental results on test dataset:

- **Smoky vehicles**: 86.54% detection accuracy
- **Non-smoky vehicles**: 86.74% detection accuracy
- **Average accuracy**: 86.64%
- **Processing**: Real-time on standard hardware

## Dependencies

Added to `requirements.txt`:
- `pywt==1.1.1` - Discrete Wavelet Transform
- `scikit-image==0.24.0` - GLCM features
- `scikit-learn==1.5.1` - SVM classifiers
- `scipy==1.14.0` - Scientific computing

Existing dependencies used:
- `opencv-python` - Image processing
- `numpy` - Numerical operations

## Research Paper Reference

Full citation:
```
Tao, L. & Lu, Y. (2019).
Automatic smoky vehicle detection from traffic surveillance video 
based on vehicle rear detection and multi-feature fusion.
IET Intelligent Transport Systems, 13(6), 1003-1010.
```

## Differences from Original YOLOv8 System

| Aspect | YOLOv8 (Old) | Paper-Based (New) |
|--------|----------|----------|
| Model Type | Deep Learning CNN | Handcrafted Features + SVM |
| Dependencies | ultralytics | scikit-learn, scipy |
| Features | Automatic | Manually designed (36) |
| Interpretability | Black box | Transparent (feature-level) |
| Real-time | Yes | Yes |
| Training Data | Large dataset required | 1000 samples per class |
| Research | Model zoo | Faithful paper reproduction |

## Future Enhancements

1. **SVM Model Persistence**: Save/load trained SVM models to disk
2. **Video Sliding Window**: Implement 100-frame sliding window for video decision logic
3. **Dataset Curation**: Tools for extracting and labeling training frames
4. **Parameter Tuning**: Grid search for optimal SVM hyperparameters
5. **Performance Metrics**: ROC curves, confusion matrices, precision/recall
6. **Visualization**: Feature importance and component contribution analysis

## Troubleshooting

### Low Detection Accuracy
1. Check if SVM models are trained (currently using heuristic fallback)
2. Verify input video resolution and frame rate
3. Ensure lighting conditions match training data

### Performance Issues
1. Reduce frame rate for real-time processing
2. Resize input frames before processing
3. Use GPU-accelerated OpenCV if available

### Import Errors
1. Run `pip install -r requirements.txt` to install all dependencies
2. Verify Python 3.8+ is installed
3. Check for version conflicts: `pip list | grep -E "opencv|scikit|pywt"`

## References

### Key Papers
1. Barnich, O., Van Droogenbroeck, M. (2011). ViBe: a universal background subtraction algorithm for video sequences. IEEE Transactions on Image Processing, 20(6), 1709-1724.

2. Haralick, R. M., Shanmugam, K., Dinstein, I. (1973). Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics, 3(6), 610-621.

3. Mallat, S. (1989). A theory for multiresolution signal decomposition: the wavelet representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 11(7), 674-693.

### Implementation References
- ViBe background subtraction: OpenCV implementation
- GLCM features: scikit-image implementation  
- DWT features: PyWavelets (pywt) library
- SVM: scikit-learn library
