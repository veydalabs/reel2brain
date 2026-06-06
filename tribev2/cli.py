from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


FRONTEND_SKIP_BUILD_ENV = "REEL2BRAIN_SKIP_FRONTEND_BUILD"
FRONTEND_FORCE_BUILD_ENV = "REEL2BRAIN_FORCE_FRONTEND_BUILD"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_frontend_root() -> Path:
    return get_project_root() / "frontend"


def get_frontend_dist() -> Path:
    return Path(__file__).with_name("web_dist")


def _iter_files(path: Path):
    if not path.exists():
        return
    if path.is_file():
        yield path
        return
    for child in path.rglob("*"):
        if child.is_file():
            yield child


def _latest_mtime(paths: list[Path]) -> float:
    latest = 0.0
    for path in paths:
        for file_path in _iter_files(path):
            latest = max(latest, file_path.stat().st_mtime)
    return latest


def frontend_build_required(
    frontend_root: Path | None = None,
    frontend_dist: Path | None = None,
    project_root: Path | None = None,
) -> bool:
    if os.environ.get(FRONTEND_SKIP_BUILD_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get(FRONTEND_FORCE_BUILD_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return True

    resolved_frontend_root = frontend_root or get_frontend_root()
    resolved_frontend_dist = frontend_dist or get_frontend_dist()
    resolved_project_root = project_root or get_project_root()

    dist_index = resolved_frontend_dist / "index.html"
    if not dist_index.exists():
        return True

    source_paths = [
        resolved_frontend_root / "src",
        resolved_frontend_root / "package.json",
        resolved_frontend_root / "vite.config.js",
        resolved_project_root / "public",
    ]
    source_mtime = _latest_mtime(source_paths)
    dist_mtime = _latest_mtime([resolved_frontend_dist])
    return source_mtime > dist_mtime


def ensure_frontend_bundle() -> None:
    frontend_root = get_frontend_root()
    frontend_dist = get_frontend_dist()
    project_root = get_project_root()

    if not frontend_build_required(frontend_root, frontend_dist, project_root):
        return

    npm_path = shutil.which("npm")
    if npm_path is None:
        raise SystemExit(
            "Frontend source is newer than the bundled app, but `npm` is not available. "
            "Install Node.js/npm and run `reel2brain` again, or set "
            f"`{FRONTEND_SKIP_BUILD_ENV}=1` to use the existing bundle."
        )

    print("Reel2Brain: rebuilding frontend bundle...", file=sys.stderr)
    try:
        subprocess.run(
            [npm_path, "run", "build"],
            cwd=frontend_root,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Frontend build failed while launching `reel2brain`. "
            "Run `cd frontend && npm run build` to inspect the error."
        ) from exc


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

    ensure_frontend_bundle()

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
