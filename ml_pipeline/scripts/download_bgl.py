#!/usr/bin/env python3
"""
Download the BGL (Blue Gene/L) supercomputer log dataset from LogHub.

The BGL dataset contains 4,747,963 log messages from IBM's Blue Gene/L
supercomputer at Lawrence Livermore National Laboratory with binary alert labels.

Dataset repository: https://github.com/logpai/loghub/tree/master/BGL
"""

import os
import sys
import requests
import zipfile
import tarfile
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Full dataset (large) — Zenodo / GitHub release URLs change; try several.
BGL_FULL_URLS = [
    "https://zenodo.org/records/8275861/files/BGL.zip",
    "https://zenodo.org/record/8196385/files/BGL.tar.gz",
    "https://github.com/logpai/loghub/releases/download/v2.0/BGL.tar.gz",
]

# Small **real** subset from LogHub (always available) — good for honest metrics without GB download
BGL_SAMPLE_RAW = (
    "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log"
)


def download_bgl():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "BGL.log")
    sample_path = os.path.join(DATA_DIR, "BGL_2k.log")

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        logger.info(f"BGL.log already exists ({size_mb:.1f} MB). Skipping download.")
        return out_path

    # -- 1) Fast path: 2k real lines from LogHub GitHub (recommended minimum for dev) --
    try:
        logger.info(f"Downloading real sample {BGL_SAMPLE_RAW} …")
        r = requests.get(BGL_SAMPLE_RAW, timeout=120)
        r.raise_for_status()
        with open(sample_path, "wb") as f:
            f.write(r.content)
        logger.info(f"Saved real BGL subset to {sample_path} ({len(r.content)/1024:.1f} KB)")
        return sample_path
    except Exception as e:
        logger.warning(f"Sample download failed: {e}")

    downloaded = None
    for url in BGL_FULL_URLS:
        archive_name = os.path.basename(url.split("?")[0])
        archive_path = os.path.join(DATA_DIR, archive_name)
        logger.info(f"Attempting download from: {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total:
                            pct = downloaded_bytes / total * 100
                            print(f"\r  {pct:.1f}%  ({downloaded_bytes/1024/1024:.1f} MB)", end="")
            print()
            downloaded = archive_path
            logger.info(f"Download complete: {archive_path}")
            break
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    if downloaded is None:
        logger.error("All download URLs failed.")
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("1. Visit: https://github.com/logpai/loghub/tree/master/BGL")
        print("2. Download: BGL.tar.gz or BGL.log (compressed)")
        print(f"3. Place the extracted BGL.log in: {DATA_DIR}/")
        print("=" * 60)
        return None

    # Extract
    logger.info("Extracting archive ...")
    try:
        if downloaded.endswith(".tar.gz") or downloaded.endswith(".tgz"):
            with tarfile.open(downloaded, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("BGL.log"):
                        member.name = "BGL.log"
                        tar.extract(member, DATA_DIR)
                        break
        elif downloaded.endswith(".zip"):
            with zipfile.ZipFile(downloaded, "r") as z:
                for name in z.namelist():
                    if name.endswith("BGL.log"):
                        z.extract(name, DATA_DIR)
                        break
        os.remove(downloaded)
        logger.info(f"BGL.log extracted to {DATA_DIR}/")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        logger.info(f"Dataset ready: {out_path} ({size_mb:.1f} MB)")
        return out_path
    return None


if __name__ == "__main__":
    result = download_bgl()
    if result:
        print(f"\nBGL dataset ready at: {result}")
        print("Run training with: python scripts/train_pipeline.py --data data/BGL.log")
    else:
        print("\nYou can still train using a BGL-format proxy dataset:")
        print("  python scripts/train_pipeline.py")
        sys.exit(1)
