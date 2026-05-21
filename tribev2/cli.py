from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def launch_reel2brain() -> int:
    try:
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "FastAPI runtime is not installed. Run `pip install -e \".[plotting,webapp]\"` first."
        ) from exc

    app_path = Path(__file__).with_name("production_api.py")
    if not app_path.exists():
        raise SystemExit("Missing production API entrypoint.")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "tribev2.production_api:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))
