"""
Demo readiness checker for Traffic_YOLOv8_LSTM_System.

Usage:
    python scripts/check_demo_ready.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
MODEL_PATH = DATA_DIR / "models" / "best_model.pth"
NORM_PATH = DATA_DIR / "models" / "norm_params.json"
CSV_PATH = DATA_DIR / "traffic_data.csv"


def print_header() -> None:
    print("=" * 70)
    print("Demo Readiness Check")
    print(f"Project root: {ROOT}")
    print("=" * 70)


def check_videos(errors: List[str], warnings: List[str]) -> None:
    if not VIDEO_DIR.exists():
        errors.append(f"Missing directory: {VIDEO_DIR}")
        return

    video_files = []
    for pattern in ("*.mp4", "*.avi", "*.mov"):
        video_files.extend(VIDEO_DIR.glob(pattern))

    if not video_files:
        errors.append(
            f"No demo video found in {VIDEO_DIR} (expected .mp4/.avi/.mov)"
        )
        return

    print(f"[OK] video files: {len(video_files)}")
    for p in video_files[:3]:
        print(f"  - {p.name}")
    if len(video_files) > 3:
        warnings.append("More than 3 videos found; keep one short demo clip for stability.")


def check_csv(errors: List[str]) -> None:
    required_cols = {"timestamp", "vehicle_type", "count", "direction", "confidence"}
    if not CSV_PATH.exists():
        errors.append(f"Missing traffic data CSV: {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        errors.append(f"Cannot read CSV {CSV_PATH}: {e}")
        return

    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"CSV missing required columns: {sorted(missing)}")
        return

    if df.empty:
        errors.append("CSV exists but is empty.")
        return

    print(f"[OK] traffic_data.csv rows: {len(df)}")


def check_model(errors: List[str], warnings: List[str]) -> None:
    if not MODEL_PATH.exists():
        errors.append(f"Missing model file: {MODEL_PATH}")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    except Exception as e:
        errors.append(f"Cannot load model checkpoint {MODEL_PATH}: {e}")
        return

    if "model_state_dict" not in checkpoint:
        errors.append("Model checkpoint missing key: model_state_dict")
    else:
        print("[OK] model checkpoint loaded (model_state_dict found)")

    if not NORM_PATH.exists():
        warnings.append(f"Normalization file not found: {NORM_PATH}")
    else:
        print("[OK] norm_params.json found")


def check_environment(warnings: List[str]) -> None:
    cuda_ok = torch.cuda.is_available()
    print(f"[INFO] torch: {torch.__version__}")
    print(f"[INFO] cuda_available: {cuda_ok}")
    if not cuda_ok:
        warnings.append("CUDA unavailable. Demo can run on CPU but may be slower.")


def print_next_steps(errors: List[str]) -> None:
    if not errors:
        return
    print("\nSuggested actions:")
    print("1) Generate traffic data:")
    print("   python main.py --video data/videos/<your_video>.mp4 --save-data")
    print("2) Train model:")
    print("   python train.py --data data/traffic_data.csv")
    print("3) Launch UI:")
    print("   streamlit run src/visualization/app.py")


def main() -> int:
    print_header()
    errors: List[str] = []
    warnings: List[str] = []

    check_environment(warnings)
    check_videos(errors, warnings)
    check_csv(errors)
    check_model(errors, warnings)

    print("\n" + "-" * 70)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Warnings: none")

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        print_next_steps(errors)
        print("\nResult: NOT READY")
        return 1

    print("Errors: none")
    print("\nResult: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
