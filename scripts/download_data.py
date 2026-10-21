import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

target = Path("data/raw/telco.csv")
metadata = Path("data/raw/telco.source.json")

if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        target.write_bytes(response.read())

sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
if sha256 != EXPECTED_SHA256:
    raise ValueError(f"Source hash mismatch: {sha256}")

record = {
    "source_url": URL,
    "source_repository": "https://github.com/IBM/telco-customer-churn-on-icp4d",
    "source_commit": "1fd6fd70906479be1712e64cfc0ea89f58466629",
    "local_path": target.as_posix(),
    "sha256": sha256,
    "bytes": target.stat().st_size,
    "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
