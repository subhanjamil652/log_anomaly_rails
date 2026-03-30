#!/usr/bin/env python3
"""
BGL Log Anomaly Detection — Main Training Script.

Usage:
  python scripts/train_pipeline.py                    # uses synthetic data
  python scripts/train_pipeline.py --data data/BGL.log  # uses real dataset
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

from src.trainer import AnomalyDetectionTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train BGL log anomaly detection models")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to BGL.log dataset file")
    parser.add_argument("--samples", type=int, default=80_000,
                        help="Number of synthetic samples if real data unavailable")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save model artifacts")
    args = parser.parse_args()

    trainer = AnomalyDetectionTrainer(save_dir=args.save_dir)
    metadata = trainer.run_full_pipeline(
        data_path=args.data,
        synthetic_samples=args.samples,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model : {metadata['best_model']}")
    print(f"Best F1    : {metadata['best_f1']:.4f}")
    print(f"Trained at : {metadata['trained_at']}")
    print(f"Artifacts  : {trainer.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
