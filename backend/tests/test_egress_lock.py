"""Fail if runtime source reintroduces a known third-party network destination."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNTIME_PATHS = [
    "backend/app",
    "backend/scripts",
    "backend/run.py",
    "frontend/index.html",
    "frontend/src",
    ".env.example",
    "docker-compose.yml",
]
FORBIDDEN = [
    "zep_cloud",
    "getzep.com",
    "dashscope",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
]
# docker-compose.yml must build this checkout, not pull the upstream image.
FORBIDDEN_IN_COMPOSE = ["ghcr.io/666ghj/mirofish"]
TEXT_SUFFIXES = {".py", ".js", ".ts", ".vue", ".html", ".css", ".json", ".yml", ".yaml", ".example", ""}


def _files():
    for relative in RUNTIME_PATHS:
        path = ROOT / relative
        if path.is_file():
            yield path
            continue
        for file in path.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts and (
                file.suffix in TEXT_SUFFIXES
            ):
                yield file


def test_runtime_source_has_no_forbidden_destinations():
    files = list(_files())
    assert any(file.name == "llm_client.py" for file in files)
    hits = []
    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore").lower()
        forbidden = FORBIDDEN + (FORBIDDEN_IN_COMPOSE if file.name == "docker-compose.yml" else [])
        hits += [f"{file.relative_to(ROOT)}: {word}" for word in forbidden if word in text]
    assert hits == []


def test_compose_builds_from_local_source():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "build: ." in compose
    assert "image:" not in compose
