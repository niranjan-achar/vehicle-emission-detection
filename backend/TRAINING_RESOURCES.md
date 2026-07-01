# Training the Paper-Based SVM Detector

This project loads the paper-based SVM model from:

```text
backend/weights/paper_svm_models.pkl
```

The trainer now writes to that path by default.

## Dataset Format

Use cropped smoke-relevant rear/key-region images, organized by class:

```text
datasets/paper_svm/
  smoky/
    sample_001.jpg
    sample_002.jpg
  non_smoky/
    sample_001.jpg
    sample_002.jpg
```

Accepted class folder aliases:

- Smoky: `smoky`, `smoke`, `polluting`, `positive`, `1`
- Non-smoky: `non_smoky`, `non-smoke`, `non_smoke`, `clean`, `negative`, `0`

Accepted image extensions:

- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## Train Command

Run from the `backend` directory:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.classifiers.train --dataset-root ..\datasets\paper_svm
```

That creates:

```text
backend/weights/paper_svm_models.pkl
backend/weights/paper_svm_models.metrics.json
```

Optional flags:

```powershell
.\.venv\Scripts\python.exe -m app.classifiers.train `
  --dataset-root ..\datasets\paper_svm `
  --output .\weights\paper_svm_models.pkl `
  --test-size 0.2 `
  --max-per-class 1000 `
  --cv-folds 5
```

Useful flags:

- `--no-balance`: keep original class imbalance instead of downsampling.
- `--max-per-class N`: cap each class for quick experiments.
- `--no-cv`: skip cross-validation and only run holdout evaluation.
- `--min-samples-per-class N`: lower this only for smoke tests, not real training.

## Recommended Dataset Sources

### 1. Roboflow: smoke vehicles

URL: https://universe.roboflow.com/alpha-ai-nwrrb/smoke-vehicles

Why it helps:

- Open Roboflow Universe dataset.
- CC BY 4.0 license.
- About 1,034 images.
- Object-detection annotations for smoke-vehicle classes.

How to use it here:

1. Export/download the dataset from Roboflow.
2. Use annotated smoke/vehicle-rear areas to crop `smoky` samples.
3. Add clean rear/exhaust crops as `non_smoky` samples.
4. Train with the command above.

### 2. Roboflow: Car Smoke Detection

URL: https://universe.roboflow.com/national-university-of-computer-and-emerging-sciences-lahore/car-smoke-detection

Why it helps:

- Open Roboflow Universe dataset.
- About 357 images.
- Includes emission classes and a `No Emission` class, which is useful for negatives.

How to use it here:

1. Export/download the dataset.
2. Map emission classes to `smoky`.
3. Map `No Emission` examples to `non_smoky`.
4. Prefer rear/exhaust crops over full-frame images.

### 3. CoDeS Smoky Vehicle Dataset / Paper

Paper: https://arxiv.org/abs/2207.03708

Code/data placeholder: https://github.com/pengxj/smokyvehicle

Why it helps:

- The paper reports a large smoky vehicle dataset with 75,000 annotated images and 163 videos.
- It is highly relevant to traffic surveillance smoky vehicle detection.

Current caution:

- The GitHub repository is currently minimal, so check for future releases or contact the authors if you need the full dataset.

### 4. DB-Net Vehicle Smoke Dataset Paper

Paper: https://www.mdpi.com/2076-3417/13/8/4941

Why it helps:

- The paper describes a vehicle-smoke dataset with 3,962 polygon-annotated smoke images.
- Good reference if you want segmentation-quality smoke localization.

Current caution:

- Treat it as a research resource unless the authors provide direct downloadable data.

## Practical Dataset Advice

For this project, balanced class folders matter more than raw dataset size.

Start with:

- At least 200 cropped `smoky` samples.
- At least 200 cropped `non_smoky` samples.
- Similar camera angle and distance to your expected input videos.
- A mix of daytime, cloudy, shadow, and wet-road cases.

Better target:

- 1,000+ cropped samples per class, matching the paper-style setup.

Avoid training only on obvious smoke. Include hard negatives:

- Road shadow behind vehicles.
- Dust.
- Fog.
- Wet road reflections.
- Exhaust area with no visible smoke.
- Vehicles partly occluded.

## Preparing Your Own Data

If you have videos:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.datasets.extract_frames --input ..\raw_videos --output ..\datasets\frames --every-n 5
```

Then crop rear/key-region samples manually or using an annotation tool, place them in `smoky` and `non_smoky`, and train.

If you already have a CSV manifest:

```csv
path,label
D:\data\crop001.jpg,smoky
D:\data\crop002.jpg,non_smoky
```

Use:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.datasets.label_dataset --manifest ..\labels.csv --output ..\datasets\paper_svm
```

Then run the SVM trainer.
