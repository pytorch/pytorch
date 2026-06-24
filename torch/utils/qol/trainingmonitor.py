import torch
import psutil
import csv
import time
import os
from tqdm.auto import tqdm
from typing import Iterable, Dict, Any

class TrainingMonitor:
    def __init__(self, iterable: Iterable, desc: str = "Training", log_file: str = "train_log.csv"):
        self.iterable = iterable
        self.total = len(iterable) if hasattr(iterable, "__len__") else None
        self.desc = desc
        self.log_file = log_file
        
        # Metrics state
        self.metrics = {}
        self.metric_counts = {}  # Track number of updates per metric
        self.column_names = None
        self.file_handle = None
        self.writer = None
        self.step = 0  # Track global step counter

    def _init_logger(self, keys):
        """Initializes the CSV file with headers."""
        self.column_names = ["timestamp", "step"] + list(keys) + ["RAM_pct", "VRAM_gb"]
        file_exists = os.path.isfile(self.log_file)
        self.file_handle = open(self.log_file, mode='a', newline='')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.column_names)
        if not file_exists:
            self.writer.writeheader()

    def __iter__(self):
        self.pbar = tqdm(self.iterable, total=self.total, desc=self.desc, dynamic_ncols=True)
        try:
            for item in self.pbar:
                yield item
                self._refresh_display()
        finally:
            self.close()

    def log(self, metrics: Dict[str, float]):
        """Records metrics, updates running averages, and writes to disk."""
        # 1. Initialize CSV on first log call
        if self.writer is None:
            self._init_logger(metrics.keys())

        # 2. Update Running Averages (Keras style)
        for k, v in metrics.items():
            if k in self.metrics:
                count = self.metric_counts[k]
                self.metrics[k] = (self.metrics[k] * count + v) / (count + 1)
                self.metric_counts[k] = count + 1
            else:
                self.metrics[k] = v
                self.metric_counts[k] = 1

        # 3. Auto-write to CSV
        res = self._get_resources()
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": self.step,
            "RAM_pct": res["RAM"],
            "VRAM_gb": res["VRAM"],
            **self.metrics
        }
        self.writer.writerow(row)
        self.file_handle.flush()  # Ensure data is written even if crash occurs
        self.step += 1

    def _get_resources(self) -> Dict[str, Any]:
        res = {"RAM": psutil.virtual_memory().percent, "VRAM": 0.0}
        if torch.cuda.is_available():
            res["VRAM"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
        return res

    def _refresh_display(self):
        display = {k: f"{v:.4f}" for k, v in self.metrics.items()}
        res = self._get_resources()
        display.update({ "RAM": f"{res['RAM']}%", "VRAM": f"{res['VRAM']}GB" })
        self.pbar.set_postfix(display)

    def close(self):
        if self.file_handle:
            self.file_handle.close()
        if hasattr(self, 'pbar'):
            self.pbar.close()
