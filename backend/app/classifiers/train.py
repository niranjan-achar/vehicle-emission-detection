"""Train and save the paper-based smoky vehicle SVM bundle.

Expected dataset layout:
    dataset_root/
        smoky/
            sample_001.jpg
            ...
        non_smoky/
            sample_002.jpg
            ...

The input images should be cropped smoke/key-region samples. If you only have
full vehicle frames, first crop the rear smoke-relevant region or use the
dataset helper scripts in app.datasets to prepare class folders.

Default output:
    backend/weights/paper_svm_models.pkl

Usage from the backend directory:
    python -m app.classifiers.train --dataset-root ..\\datasets\\paper_svm
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import sklearn
from app.classifiers.voting import SVMVotingClassifier
from app.features.artificial import ArtificialFeatureExtractor
from app.features.dwt import DWTFeatureExtractor
from app.features.glcm import GLCMFeatureExtractor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_ALIASES = {
    "smoky": 1,
    "smoke": 1,
    "polluting": 1,
    "positive": 1,
    "1": 1,
    "non_smoky": 0,
    "non-smoky": 0,
    "non_smoke": 0,
    "clean": 0,
    "negative": 0,
    "0": 0,
}

CLASS_NAMES = {
    0: "non_smoky",
    1: "smoky",
}

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "weights" / "paper_svm_models.pkl"


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output: str
    test_size: float
    random_state: int
    balance_classes: bool
    max_per_class: int | None
    min_samples_per_class: int
    cross_validate: bool
    cv_folds: int


def _iter_labeled_images(dataset_root: Path) -> Iterable[tuple[Path, int]]:
    """Yield image paths from class-named folders under dataset_root."""
    seen: set[Path] = set()

    for label_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        label_key = label_dir.name.strip().lower()
        if label_key not in LABEL_ALIASES:
            continue

        label = LABEL_ALIASES[label_key]
        for image_path in sorted(label_dir.rglob("*")):
            resolved = image_path.resolve()
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
                and resolved not in seen
            ):
                seen.add(resolved)
                yield image_path, label


def _extract_feature_dict(
    image: np.ndarray,
    art_extractor: ArtificialFeatureExtractor,
    glcm_extractor: GLCMFeatureExtractor,
    dwt_extractor: DWTFeatureExtractor,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(art_extractor.extract_features(image))
    features.update(glcm_extractor.extract_features(image))
    features.update(dwt_extractor.extract_features(image))
    return features


def _load_samples(dataset_root: Path) -> tuple[list[dict[str, float]], list[int], list[str]]:
    art_extractor = ArtificialFeatureExtractor()
    glcm_extractor = GLCMFeatureExtractor()
    dwt_extractor = DWTFeatureExtractor()

    feature_dicts: list[dict[str, float]] = []
    labels: list[int] = []
    source_paths: list[str] = []

    for image_path, label in _iter_labeled_images(dataset_root):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        feature_dicts.append(
            _extract_feature_dict(image, art_extractor, glcm_extractor, dwt_extractor)
        )
        labels.append(label)
        source_paths.append(str(image_path))

    return feature_dicts, labels, source_paths


def _limit_samples_per_class(
    feature_dicts: list[dict[str, float]],
    labels: list[int],
    source_paths: list[str],
    max_per_class: int | None,
    random_state: int,
) -> tuple[list[dict[str, float]], list[int], list[str]]:
    if max_per_class is None:
        return feature_dicts, labels, source_paths

    rng = np.random.default_rng(random_state)
    selected_indices: list[int] = []
    y = np.asarray(labels)

    for label in sorted(set(labels)):
        label_indices = np.where(y == label)[0]
        if len(label_indices) > max_per_class:
            label_indices = rng.choice(label_indices, size=max_per_class, replace=False)
        selected_indices.extend(int(index) for index in label_indices)

    selected_indices.sort()
    return (
        [feature_dicts[index] for index in selected_indices],
        [labels[index] for index in selected_indices],
        [source_paths[index] for index in selected_indices],
    )


def _balance_samples(
    feature_dicts: list[dict[str, float]],
    labels: list[int],
    source_paths: list[str],
    random_state: int,
) -> tuple[list[dict[str, float]], list[int], list[str]]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(labels)
    class_counts = {label: int(np.sum(y == label)) for label in sorted(set(labels))}
    target_count = min(class_counts.values())

    selected_indices: list[int] = []
    for label in sorted(class_counts):
        label_indices = np.where(y == label)[0]
        if len(label_indices) > target_count:
            label_indices = rng.choice(label_indices, size=target_count, replace=False)
        selected_indices.extend(int(index) for index in label_indices)

    rng.shuffle(selected_indices)
    return (
        [feature_dicts[index] for index in selected_indices],
        [labels[index] for index in selected_indices],
        [source_paths[index] for index in selected_indices],
    )


def _dicts_to_matrices(
    feature_dicts: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classifier = SVMVotingClassifier()

    art_rows = []
    glcm_rows = []
    dwt_rows = []

    for features in feature_dicts:
        art_rows.append(classifier._extract_art_features(features))
        glcm_rows.append(classifier._extract_glcm_features(features))
        dwt_rows.append(classifier._extract_dwt_features(features))

    return (
        np.asarray(art_rows, dtype=np.float32),
        np.asarray(glcm_rows, dtype=np.float32),
        np.asarray(dwt_rows, dtype=np.float32),
    )


def _validate_dataset(labels: list[int], min_samples_per_class: int) -> dict[str, int]:
    counts = {CLASS_NAMES[label]: labels.count(label) for label in sorted(set(labels))}
    missing = [name for name in CLASS_NAMES.values() if name not in counts]
    if missing:
        raise ValueError(f"Dataset is missing class folder(s): {', '.join(missing)}")

    too_small = [
        f"{class_name}={count}"
        for class_name, count in counts.items()
        if count < min_samples_per_class
    ]
    if too_small:
        raise ValueError(
            "Not enough samples per class. "
            f"Need at least {min_samples_per_class}; found {', '.join(too_small)}."
        )

    return counts


def _cross_validate_block(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> dict[str, float | list[float]]:
    if cv_folds < 2:
        return {}

    class_counts = np.bincount(y)
    min_class_count = int(class_counts[class_counts > 0].min())
    folds = min(cv_folds, min_class_count)
    if folds < 2:
        return {}

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {
        "folds": folds,
        "accuracy_scores": [float(score) for score in scores],
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
    }


def _evaluate_classifier(
    classifier: SVMVotingClassifier,
    feature_dicts: list[dict[str, float]],
    y_true: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    predictions = []
    confidences = []

    for index in test_idx:
        predicted_name, confidence, _ = classifier.predict(feature_dicts[int(index)])
        predictions.append(1 if predicted_name == "smoky" else 0)
        confidences.append(float(confidence))

    y_test = y_true[test_idx]
    target_names = [CLASS_NAMES[0], CLASS_NAMES[1]]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        ),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "test_samples": int(len(test_idx)),
    }


def _atomic_pickle_dump(bundle: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".tmp",
        prefix=output_path.name,
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        pickle.dump(bundle, handle)

    temp_path.replace(output_path)


def _write_metrics(metrics: dict, output_path: Path) -> Path:
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def train(
    dataset_root: Path,
    output_path: Path = DEFAULT_OUTPUT,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_classes: bool = True,
    max_per_class: int | None = None,
    min_samples_per_class: int = 5,
    cross_validate: bool = True,
    cv_folds: int = 5,
) -> Path:
    """Train the three SVMs and persist the fitted classifier bundle."""
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    feature_dicts, labels, source_paths = _load_samples(dataset_root)
    if not feature_dicts:
        raise ValueError(f"No labeled images found under {dataset_root}")

    feature_dicts, labels, source_paths = _limit_samples_per_class(
        feature_dicts,
        labels,
        source_paths,
        max_per_class=max_per_class,
        random_state=random_state,
    )

    if balance_classes:
        feature_dicts, labels, source_paths = _balance_samples(
            feature_dicts,
            labels,
            source_paths,
            random_state=random_state,
        )

    class_counts = _validate_dataset(labels, min_samples_per_class)
    X_art, X_glcm, X_dwt = _dicts_to_matrices(feature_dicts)
    y = np.asarray(labels, dtype=np.int32)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    classifier = SVMVotingClassifier()
    trained = classifier.train(
        X_art[train_idx],
        X_glcm[train_idx],
        X_dwt[train_idx],
        y[train_idx],
    )
    if not trained:
        raise RuntimeError("SVM training failed. Check the feature extraction logs.")

    holdout_metrics = _evaluate_classifier(classifier, feature_dicts, y, test_idx)

    cv_metrics = {}
    if cross_validate:
        cv_metrics = {
            "art": _cross_validate_block(X_art, y, cv_folds, random_state),
            "glcm": _cross_validate_block(X_glcm, y, cv_folds, random_state),
            "dwt": _cross_validate_block(X_dwt, y, cv_folds, random_state),
        }

    config = TrainConfig(
        dataset_root=str(dataset_root),
        output=str(output_path),
        test_size=test_size,
        random_state=random_state,
        balance_classes=balance_classes,
        max_per_class=max_per_class,
        min_samples_per_class=min_samples_per_class,
        cross_validate=cross_validate,
        cv_folds=cv_folds,
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(len(y)),
        "class_counts": class_counts,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "source_paths_sample": source_paths[:20],
        "sklearn_version": sklearn.__version__,
        "opencv_version": cv2.__version__,
        "config": asdict(config),
    }

    metrics = {
        "metadata": metadata,
        "holdout": holdout_metrics,
        "cross_validation": cv_metrics,
    }

    bundle = {
        "classifier": classifier,
        "metadata": metadata,
        "metrics": metrics,
    }

    _atomic_pickle_dump(bundle, output_path)
    metrics_path = _write_metrics(metrics, output_path)

    print(f"Saved trained SVM bundle: {output_path}")
    print(f"Saved metrics report:     {metrics_path}")
    print(f"Samples: {len(y)} | train: {len(train_idx)} | test: {len(test_idx)}")
    print(f"Class counts: {class_counts}")
    print(f"Holdout accuracy: {holdout_metrics['accuracy']:.4f}")

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the paper-based SVM voting classifier."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory containing smoky/non_smoky key-region images.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output .pkl path. Defaults to backend/weights/paper_svm_models.pkl.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and sampling.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable downsampling to equalize smoky/non_smoky class counts.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional maximum number of images to use from each class.",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=5,
        help="Minimum samples required in each class before training.",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation metrics.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Requested cross-validation folds. Automatically reduced for small datasets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(
        dataset_root=Path(args.dataset_root),
        output_path=Path(args.output),
        test_size=args.test_size,
        random_state=args.random_state,
        balance_classes=not args.no_balance,
        max_per_class=args.max_per_class,
        min_samples_per_class=args.min_samples_per_class,
        cross_validate=not args.no_cv,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
