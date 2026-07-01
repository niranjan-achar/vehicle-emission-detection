"""Extract frames from videos for dataset preparation.

Usage:
    python -m app.datasets.extract_frames --input <video-or-dir> --output <dir> --every-n 5

The tool preserves the input label directory when the source is organized as:
    input_root/
        smoky/
            video1.mp4
        non_smoky/
            video2.mp4

Frames are written to:
    output_root/<label>/<video_stem>/frame_000001.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_video_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            yield source
        return

    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def extract_frames(source: Path, output_root: Path, every_n: int = 5, max_frames: int | None = None) -> int:
    """Extract frames from a video or a directory of videos."""
    output_root.mkdir(parents=True, exist_ok=True)
    extracted = 0

    for video_path in _iter_video_files(source):
        label = video_path.parent.name if video_path.parent != source else "unlabeled"
        label_dir = output_root / label / video_path.stem
        label_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            continue

        frame_index = 0
        saved_index = 0
        try:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                if frame_index % max(1, every_n) == 0:
                    frame_name = f"frame_{saved_index:06d}.jpg"
                    output_path = label_dir / frame_name
                    cv2.imwrite(str(output_path), frame)
                    extracted += 1
                    saved_index += 1

                    if max_frames is not None and saved_index >= max_frames:
                        break

                frame_index += 1
        finally:
            capture.release()

    return extracted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frames from videos for dataset preparation")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output directory for extracted frames")
    parser.add_argument("--every-n", type=int, default=5, help="Save every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap per video")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Path(args.input)
    output_root = Path(args.output)
    count = extract_frames(source, output_root, every_n=args.every_n, max_frames=args.max_frames)
    print(f"Extracted {count} frames into {output_root}")


if __name__ == "__main__":
    main()
