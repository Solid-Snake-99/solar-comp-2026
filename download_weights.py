#!/usr/bin/env python3
"""
Automated Model Weights Downloader & Integrity Verifier for Solar Filament Segmentation.

This script allows the competition evaluation committee to securely download the exact
pre-trained model checkpoints from private Google Drive links and verify their cryptographic
MD5 checksums.

DO NOT RENAME the downloaded files: the pipeline expects:
  - models/RF-DETR-SEG-LARGE-TILING-NMM.pth
  - models/final_refiner.pt
"""

import os
import sys
import hashlib
from pathlib import Path

# Always set working directory to this script's directory (essential for Windows double-click)
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# Ensure UTF-8 output encoding on all platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXPECTED_MD5 = {
    "RF-DETR-SEG-LARGE-TILING-NMM.pth": "2fcde57114b2483130ccab596a61bcd7",
    "final_refiner.pt": "74991b8d95f7922f0ca6fd75f8ab4f5f",
}

GOOGLE_DRIVE_FILES = {
    "RF-DETR-SEG-LARGE-TILING-NMM.pth": {
        "id": "1n2MYBKVQHJTfV5VEDwCm1IjpdPc3R4Jr",
        "url": "https://drive.google.com/file/d/1n2MYBKVQHJTfV5VEDwCm1IjpdPc3R4Jr/view?usp=sharing",
        "desc": "Stage 1 Detector: RF-DETR-Seg-Large Tiling (139.4 MB)",
    },
    "final_refiner.pt": {
        "id": "1bSkYbWXGHP2ydGrjcRe3Cug8aYRT5cDf",
        "url": "https://drive.google.com/file/d/1bSkYbWXGHP2ydGrjcRe3Cug8aYRT5cDf/view?usp=sharing",
        "desc": "Stage 2 Refiner: CropUNet ResNet-18 (54.4 MB)",
    },
}

def compute_md5(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Computes MD5 hash reading file in binary chunks of 8 MB."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_file(file_path: Path, filename: str) -> bool:
    """Verifies that the file exists and matches the expected MD5 hash."""
    if not file_path.is_file():
        return False
    actual_md5 = compute_md5(file_path)
    expected = EXPECTED_MD5[filename]
    if actual_md5.lower() == expected.lower():
        print(f"   [MD5 OK] {actual_md5} (Checksum verified successfully)")
        return True
    else:
        print(f"   [MD5 MISMATCH] Expected: {expected}, got: {actual_md5}")
        return False

def ensure_gdown():
    """Ensures gdown is installed for reliable Google Drive downloads."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("Installing 'gdown' for automated Google Drive download...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
        return gdown

def run_download():
    print("=" * 75)
    print("Solar Filament Segmentation - Model Checkpoints Downloader")
    print("Evaluation Committee Private Access via Google Drive")
    print(f"Working Directory: {SCRIPT_DIR}")
    print("=" * 75)

    gdown = ensure_gdown()

    target_dir = SCRIPT_DIR / "models"
    target_dir.mkdir(exist_ok=True)

    all_ok = True
    for filename, info in GOOGLE_DRIVE_FILES.items():
        dest = target_dir / filename
        print(f"\n--> Checking {filename} [{info['desc']}]...")
        
        if dest.is_file() and verify_file(dest, filename):
            print(f"    {filename} already present and verified in {dest}.")
            continue

        print(f"    Downloading {filename} from Google Drive...")
        print(f"    Drive ID: {info['id']}")
        print(f"    URL:      {info['url']}")
        
        success = False
        try:
            res = gdown.download(id=info["id"], output=str(dest), quiet=False)
            if res and dest.is_file():
                success = True
        except Exception as e:
            print(f"    gdown download note: {e}")

        if not success or not verify_file(dest, filename):
            try:
                print("    Retrying download with fuzzy URL...")
                res = gdown.download(url=info["url"], output=str(dest), quiet=False, fuzzy=True)
                if res and dest.is_file():
                    success = True
            except Exception as e2:
                print(f"    Fuzzy download note: {e2}")

        if verify_file(dest, filename):
            print(f"    Successfully downloaded and verified: {dest}")
        else:
            all_ok = False
            print(f"    Failed to verify {filename}.")
            print(f"    Manual Download Link: {info['url']}")
            print(f"    Target Location:      {dest}")

    print("\n" + "=" * 75)
    if all_ok:
        print("SUCCESS: All model checkpoints are ready in ./models/!")
        print("The pipeline notebook (pipeline_inference.ipynb) can now be run directly.")
    else:
        print("WARNING: Some checkpoints could not be verified automatically.")
        print("Please check the manual links above and ensure files are placed in ./models/.")
    print("=" * 75)

def main():
    try:
        run_download()
    except Exception as exc:
        print(f"\nAn error occurred: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        # Prevent window from closing immediately when run via double-click on Windows
        try:
            input("\nPress Enter to exit...")
        except Exception:
            pass

if __name__ == "__main__":
    main()
