"""Tracked training (TP03): uv run python scripts/train_mlflow.py [--config configs/tp03.json]

Needs the MLflow tracking server running (see README).
"""

import sys

from churnlab.tracking import main

# MLflow prints emoji run links; a redirected Windows console (cp1252) cannot encode them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    main()
