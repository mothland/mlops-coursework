"""Run the test suite and save its output as evidence, with project-relative paths.

pytest prints absolute paths (rootdir, warning locations); the project root is replaced by "."
so the saved log does not depend on where the repository is cloned.

    uv run python scripts/run_tests.py
"""

import subprocess
import sys
from pathlib import Path

LOG = Path("reports/tp02/pytest.log")

result = subprocess.run(
    [sys.executable, "-m", "pytest", "-rA"],  # -rA: list every passed test in the summary
    capture_output=True,
    text=True,
)
output = (result.stdout + result.stderr).replace(str(Path.cwd().resolve()), ".")

LOG.parent.mkdir(parents=True, exist_ok=True)
LOG.write_text(output, encoding="utf-8")
print(output)
sys.exit(result.returncode)
