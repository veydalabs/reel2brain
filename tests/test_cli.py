from __future__ import annotations

import time
from pathlib import Path

from tribev2.cli import (
    FRONTEND_FORCE_BUILD_ENV,
    FRONTEND_SKIP_BUILD_ENV,
    frontend_build_required,
)


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_frontend_build_required_when_dist_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    frontend_root = project_root / "frontend"
    frontend_dist = project_root / "tribev2" / "web_dist"

    _write(frontend_root / "src" / "App.jsx")
    _write(frontend_root / "package.json", "{}")
    _write(frontend_root / "vite.config.js", "export default {}")
    _write(project_root / "public" / "branding" / "logo.txt")

    assert frontend_build_required(frontend_root, frontend_dist, project_root) is True


def test_frontend_build_required_when_source_is_newer(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    frontend_root = project_root / "frontend"
    frontend_dist = project_root / "tribev2" / "web_dist"

    _write(frontend_dist / "index.html", "<html></html>")
    _write(frontend_root / "src" / "App.jsx", "old")
    _write(frontend_root / "package.json", "{}")
    _write(frontend_root / "vite.config.js", "export default {}")
    _write(project_root / "public" / "branding" / "logo.txt")

    time.sleep(0.02)
    _write(frontend_root / "src" / "App.jsx", "new")

    assert frontend_build_required(frontend_root, frontend_dist, project_root) is True


def test_frontend_build_not_required_when_dist_is_current(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    frontend_root = project_root / "frontend"
    frontend_dist = project_root / "tribev2" / "web_dist"

    _write(frontend_root / "src" / "App.jsx")
    _write(frontend_root / "package.json", "{}")
    _write(frontend_root / "vite.config.js", "export default {}")
    _write(project_root / "public" / "branding" / "logo.txt")

    time.sleep(0.02)
    _write(frontend_dist / "index.html", "<html></html>")

    assert frontend_build_required(frontend_root, frontend_dist, project_root) is False


def test_frontend_build_skip_env_overrides_detection(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    frontend_root = project_root / "frontend"
    frontend_dist = project_root / "tribev2" / "web_dist"

    _write(frontend_root / "src" / "App.jsx")
    monkeypatch.setenv(FRONTEND_SKIP_BUILD_ENV, "1")

    assert frontend_build_required(frontend_root, frontend_dist, project_root) is False


def test_frontend_build_force_env_overrides_detection(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    frontend_root = project_root / "frontend"
    frontend_dist = project_root / "tribev2" / "web_dist"

    _write(frontend_dist / "index.html", "<html></html>")
    monkeypatch.setenv(FRONTEND_FORCE_BUILD_ENV, "1")

    assert frontend_build_required(frontend_root, frontend_dist, project_root) is True

