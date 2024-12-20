import sys
import os
import argparse
from pathlib import Path
import h5py
import numpy as np
import subprocess

def ensure_directory_exists(path: Path) -> Path:
    path = path.resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise ValueError(f"The path {path} exists but is not a directory.")
    return path

def run_mussel_tessellate(slide_path: str, output_h5_path: str):
    cmd = [
        "mussel", "tessellate",
        f"--slide_path={slide_path}",
        f"--output_h5_path={output_h5_path}"
    ]
    subprocess.run(cmd, check=True)

def run_mussel_extract_features(patch_h5_path: str, slide_path: str, output_h5_path: str, foundation_model_path: str):
    cmd = [
        "mussel", "extract_features",
        f"--patch_h5_path={patch_h5_path}",
        f"--slide_path={slide_path}",
        f"--output_h5_path={output_h5_path}",
        "--model_type=GIGAPATH",
        "--batch_size=64",
        "--use_gpu",
        "--num_workers=8"
    ]
    if foundation_model_path:
        cmd.append(f"--model_path={foundation_model_path}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUSSEL wrapper to create embeddings for sub-patches.")
    parser.add_argument("--slide_path", type=str, required=True, help="Path to the input WSI (e.g. .svs)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to store embeddings")
    parser.add_argument("--foundation_model_path", type=str, required=True, help="Path to the Gigapath foundation model")
    args = parser.parse_args()

    out_dir = ensure_directory_exists(Path(args.out_dir))
    slide_id = Path(args.slide_path).stem
    slide_results_dir = ensure_directory_exists(out_dir / slide_id)

    patch_h5_path = slide_results_dir / "coords.h5"
    features_h5_path = slide_results_dir / "features.h5"

    print(f"Running tessellation for slide {slide_id}")
    run_mussel_tessellate(args.slide_path, str(patch_h5_path))

    print(f"Running feature extraction using Gigapath for slide {slide_id}")
    run_mussel_extract_features(str(patch_h5_path), args.slide_path, str(features_h5_path), args.foundation_model_path)

    print(f"Embeddings generated at: {features_h5_path}")
    print("MUSSEL wrapper completed successfully.")
