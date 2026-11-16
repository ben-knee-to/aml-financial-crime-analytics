from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data" / "raw"
out.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d",
                "berkanoztas/synthetic-transaction-monitoring-dataset-aml",
                "-p", str(out), "--unzip"], check=True)
print("Downloaded dataset to", out)