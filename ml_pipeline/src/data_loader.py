"""
BGL Dataset Loader
Loads the Blue Gene/L supercomputer log dataset from LogHub.
Falls back to generating synthetic BGL-like data when the real dataset is unavailable.
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# -- BGL component / event taxonomy ------------------------------------------

BGL_COMPONENTS = [
    "kernel", "MMCS", "APP", "MPI-IO", "lustre", "BGLMASTER",
    "ciod", "ciodb", "jm", "bgl_clk_prog", "torus", "IO", "rts",
    "lwk", "comm", "serv", "pm", "mmcs", "syslog", "HARDWARE",
]

BGL_LEVELS = ["FATAL", "ERROR", "SEVERE", "WARNING", "INFO", "APPINFO", "UNKNOWN"]

NORMAL_TEMPLATES = [
    "instruction cache parity error corrected",
    "program loaded",
    "starting MPI task <*>",
    "total of <*> nodes in partition",
    "ciod: failed to read message prefix on control stream <*>",
    "job <*> completed successfully",
    "memory module <*> initialized",
    "link training complete on port <*>",
    "node card <*> powered on",
    "environment monitor reading <*> degrees",
    "checksum verified for block <*>",
    "torus network link <*> active",
    "lustre filesystem mounted at <*>",
    "heartbeat received from node <*>",
    "synchronization complete for rank <*>",
    "I/O node <*> registered",
    "barrier reached by all <*> tasks",
    "file <*> opened for read",
    "application checkpoint written to <*>",
    "system clock synchronized",
]

ANOMALY_TEMPLATES = [
    "data bus error on node <*>",
    "uncorrectable ECC memory error at address <*>",
    "hardware watchdog timeout on core <*>",
    "machine check exception on node <*>",
    "link failure detected on torus port <*>",
    "node <*> failed to boot after <*> retries",
    "FATAL: memory scrubbing failed on DIMM <*>",
    "rts: kernel panic - not syncing on node <*>",
    "I/O forwarding layer unresponsive on node <*>",
    "power supply fault detected on rack <*>",
    "temperature threshold exceeded on node <*> - <*> degrees",
    "network interface reset due to excessive errors",
    "job <*> terminated abnormally with signal <*>",
    "ciod: MPI task exited with status <*>",
    "inter-node communication timeout after <*> ms",
]

LEVEL_MAP = {
    "FATAL": 5, "SEVERE": 4, "ERROR": 4,
    "WARNING": 3, "INFO": 2, "APPINFO": 1, "UNKNOWN": 0,
}


class BGLDataLoader:
    """Load and preprocess the BGL supercomputer log dataset."""

    def __init__(self, drain_parser=None):
        self.drain_parser = drain_parser

    # -- Real data loading ----------------------------------------------------

    def load(self, filepath: str) -> pd.DataFrame:
        """Load real BGL.log file from LogHub."""
        logger.info(f"Loading BGL dataset from {filepath}")
        records = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in tqdm(f, desc="Parsing BGL logs"):
                line = line.strip()
                if not line:
                    continue
                if self.drain_parser:
                    record = self.drain_parser.parse_bgl_line(line)
                else:
                    record = self._parse_bgl_line_raw(line)
                records.append(record)
        df = pd.DataFrame(records)
        df = self._postprocess(df)
        logger.info(f"Loaded {len(df)} log entries, {df['is_anomaly'].sum()} anomalies "
                    f"({df['is_anomaly'].mean()*100:.1f}%)")
        return df

    def load_sample(self, filepath: str, n: int = 50_000) -> pd.DataFrame:
        """Load a random sample of the BGL dataset for faster iteration."""
        full = self.load(filepath)
        if len(full) > n:
            normal = full[full["is_anomaly"] == 0].sample(
                min(int(n * 0.92), len(full[full["is_anomaly"] == 0])), random_state=42)
            anomaly = full[full["is_anomaly"] == 1].sample(
                min(int(n * 0.08), len(full[full["is_anomaly"] == 1])), random_state=42)
            return pd.concat([normal, anomaly]).sort_values("timestamp").reset_index(drop=True)
        return full

    def _parse_bgl_line_raw(self, line: str) -> dict:
        pattern = re.compile(
            r'^(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+)$')
        m = pattern.match(line)
        if m:
            label, ts, date, time_, node, _, _, component, content = m.groups()
            return {
                "label": label, "timestamp": int(ts), "date": date, "time": time_,
                "node": node, "component": component, "content": content,
                "is_anomaly": 0 if label == "-" else 1,
                "template": content[:80],
                "cluster_id": hash(content[:40]) % 1000,
            }
        return {
            "label": "-", "timestamp": 0, "date": "", "time": "",
            "node": "", "component": "UNKNOWN", "content": line,
            "is_anomaly": 0, "template": line[:80], "cluster_id": 0,
        }

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df["severity_level"] = df["component"].apply(
            lambda c: "ERROR" if c in ["FATAL", "SEVERE", "ERROR"] else "INFO")
        df["severity_code"] = df["severity_level"].map(LEVEL_MAP).fillna(0).astype(int)
        df["component_clean"] = df["component"].apply(
            lambda c: c if c in BGL_COMPONENTS else "OTHER")
        return df

    # -- Synthetic data generation --------------------------------------------

    def generate_synthetic_bgl(self, n_samples: int = 100_000,
                                anomaly_rate: float = 0.079,
                                random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic BGL-style log data that closely mirrors the statistical
        properties of the real BGL dataset from Lawrence Livermore National Laboratory.

        Anomaly patterns include clustered FATAL/SEVERE events, repeated hardware
        failures, and abnormal inter-event timing - consistent with real fault signatures.
        """
        rng = np.random.default_rng(random_state)
        logger.info(f"Generating {n_samples:,} synthetic BGL log entries "
                    f"(anomaly rate ≈ {anomaly_rate*100:.1f}%)")

        n_anomaly = int(n_samples * anomaly_rate)
        n_normal = n_samples - n_anomaly

        records = []

        # -- Normal log entries --
        base_ts = 1_117_838_570
        ts = base_ts
        for i in range(n_normal):
            gap = int(rng.exponential(0.18))          # ~5.5 events/second
            ts += gap
            template = NORMAL_TEMPLATES[rng.integers(len(NORMAL_TEMPLATES))]
            component = BGL_COMPONENTS[rng.integers(len(BGL_COMPONENTS))]
            level = rng.choice(["INFO", "APPINFO", "INFO", "INFO", "WARNING"],
                               p=[0.5, 0.15, 0.2, 0.1, 0.05])
            node = f"R{rng.integers(8):02d}-M{rng.integers(4)}-N{rng.integers(16):02d}-J00"
            records.append({
                "label": "-", "timestamp": ts,
                "date": "2005.06.03", "time": "12:00:00.000",
                "node": node, "component": component,
                "content": template.replace("<*>", str(rng.integers(1000))),
                "is_anomaly": 0,
                "template": template,
                "cluster_id": NORMAL_TEMPLATES.index(template),
                "severity_level": level,
                "severity_code": LEVEL_MAP.get(level, 0),
            })

        # -- Anomalous entries (clustered to mimic real fault propagation) --
        n_bursts = max(1, n_anomaly // 15)
        anomaly_per_burst = n_anomaly // n_bursts

        burst_start_offsets = sorted(
            rng.integers(0, n_samples, size=n_bursts))

        for burst in range(n_bursts):
            burst_ts = base_ts + int(burst_start_offsets[burst] * 0.18)
            burst_component = BGL_COMPONENTS[rng.integers(len(BGL_COMPONENTS))]
            burst_node = f"R{rng.integers(8):02d}-M{rng.integers(4)}-N{rng.integers(16):02d}-J00"

            for j in range(anomaly_per_burst):
                gap = int(rng.exponential(0.05))     # anomalies cluster tightly
                burst_ts += gap
                template = ANOMALY_TEMPLATES[rng.integers(len(ANOMALY_TEMPLATES))]
                level = rng.choice(
                    ["FATAL", "SEVERE", "ERROR", "WARNING"],
                    p=[0.35, 0.30, 0.25, 0.10])
                alert_label = level
                records.append({
                    "label": alert_label, "timestamp": burst_ts,
                    "date": "2005.06.03", "time": "12:00:00.000",
                    "node": burst_node, "component": burst_component,
                    "content": template.replace("<*>", str(rng.integers(1000))),
                    "is_anomaly": 1,
                    "template": template,
                    "cluster_id": len(NORMAL_TEMPLATES) + ANOMALY_TEMPLATES.index(template),
                    "severity_level": level,
                    "severity_code": LEVEL_MAP.get(level, 5),
                })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["component_clean"] = df["component"].apply(
            lambda c: c if c in BGL_COMPONENTS else "OTHER")
        logger.info(f"Synthetic dataset: {len(df):,} entries, "
                    f"{df['is_anomaly'].sum():,} anomalies "
                    f"({df['is_anomaly'].mean()*100:.1f}%)")
        return df

    # -- Sliding window -------------------------------------------------------

    def create_windows(self, df: pd.DataFrame,
                       window_size: int = 20,
                       stride: int = 10) -> tuple:
        """
        Create sliding-window feature matrices for sequence modelling.
        A window is labelled anomalous if any entry within it is anomalous.
        Returns (windows_df, y) where windows_df has one row per window.
        """
        windows, labels = [], []
        n = len(df)
        for start in range(0, n - window_size + 1, stride):
            window = df.iloc[start: start + window_size]
            windows.append(window)
            labels.append(int(window["is_anomaly"].any()))
        logger.info(f"Created {len(windows):,} windows "
                    f"(size={window_size}, stride={stride}), "
                    f"{sum(labels):,} anomalous ({sum(labels)/len(labels)*100:.1f}%)")
        return windows, np.array(labels, dtype=int)
