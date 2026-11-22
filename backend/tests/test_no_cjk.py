"""Lock test: the repository must not contain Chinese/CJK characters.

Walks every text file of the repository and fails, listing ``path:line``,
when a character in one of these ranges appears in a file's content or in a
file or directory name:

* U+3000-U+303F  CJK symbols and punctuation
* U+3400-U+4DBF  CJK unified ideographs, extension A
* U+4E00-U+9FFF  CJK unified ideographs
* U+FF00-U+FFEF  half-width and full-width forms

The pattern is built from escapes so this file itself stays clean. Code that
needs CJK characters to handle arbitrary input text must write them as
``\\uXXXX`` escapes.
"""

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CJK_PATTERN = re.compile(
    "["
    "\u3000-\u303f"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uff00-\uffef"
    "]"
)

SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "__pycache__",
    "dist",
    ".venv",
    "venv",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
}
# Runtime data, relative to the repository root.
SKIP_DIR_PATHS = {
    Path("backend") / "uploads",
    Path("backend") / "logs",
}
SKIP_FILE_NAMES = {"uv.lock", "package-lock.json"}


def _is_virtualenv(path: Path) -> bool:
    return (path / "pyvenv.cfg").is_file()


def _read_text(path: Path):
    """Return the file's text, or None for binary or undecodable files."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _find_hits():
    hits = []
    for root, dirs, files in os.walk(REPO_ROOT):
        root_path = Path(root)
        kept = []
        for name in sorted(dirs):
            dir_path = root_path / name
            if name in SKIP_DIR_NAMES:
                continue
            if dir_path.relative_to(REPO_ROOT) in SKIP_DIR_PATHS:
                continue
            if _is_virtualenv(dir_path):
                continue
            if CJK_PATTERN.search(name):
                hits.append(f"{dir_path.relative_to(REPO_ROOT)}: CJK in directory name")
            kept.append(name)
        dirs[:] = kept

        for name in sorted(files):
            if name in SKIP_FILE_NAMES:
                continue
            file_path = root_path / name
            rel = file_path.relative_to(REPO_ROOT)
            if CJK_PATTERN.search(name):
                hits.append(f"{rel}: CJK in file name")
            text = _read_text(file_path)
            if text is None:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if CJK_PATTERN.search(line):
                    hits.append(f"{rel}:{lineno}")
    return hits


def test_pattern_matches_each_range_and_ignores_other_scripts():
    for char in ("\u3002", "\u3400", "\u4e2d", "\u9fff", "\uff01", "\uffe5"):
        assert CJK_PATTERN.search(char), hex(ord(char))
    for text in ("English", "Espa\u00f1ol", "\u0420\u0443\u0441\u0441\u043a\u0438\u0439", "caf\u00e9 \u2014 \u201cquoted\u201d"):
        assert not CJK_PATTERN.search(text), text


def test_repository_has_no_cjk_characters():
    hits = _find_hits()
    assert not hits, (
        f"{len(hits)} line(s) or name(s) contain CJK characters "
        "(write them as \\uXXXX escapes if the code needs them):\n" + "\n".join(hits)
    )
