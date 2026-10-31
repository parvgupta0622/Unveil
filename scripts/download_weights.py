"""Pre-download every model used by Unveil into the Hugging Face / rembg caches (~6 GB)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unveil.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--preload-only"]))
