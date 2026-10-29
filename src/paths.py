"""Output paths shared by training and evaluation."""
import os
from pathlib import Path

result_dir = os.environ.get("COT_RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "results"))
