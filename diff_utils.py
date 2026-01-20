from __future__ import annotations

import difflib
import os
import subprocess
from pathlib import Path

from config import (
    MAX_SNAPSHOT_FILES,
    MAX_SNAPSHOT_FILE_BYTES,
    MAX_SNAPSHOT_TOTAL_BYTES,
)


def snapshot_files(repo: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    total_bytes = 0
    file_count = 0
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules"}

    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip_dirs]
        for name in files:
            if name.startswith("."):
                continue
            if file_count >= MAX_SNAPSHOT_FILES or total_bytes >= MAX_SNAPSHOT_TOTAL_BYTES:
                return snapshot
            path = Path(root) / name
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > MAX_SNAPSHOT_FILE_BYTES:
                continue
            if b"\x00" in data:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="ignore")
            rel_path = str(path.relative_to(repo))
            snapshot[rel_path] = text
            total_bytes += len(data)
            file_count += 1

    return snapshot


def build_snapshot_diff(before: dict[str, str], repo: Path) -> str:
    after = snapshot_files(repo)
    diffs: list[str] = []
    for path in sorted(set(before) | set(after)):
        before_text = before.get(path, "")
        after_text = after.get(path, "")
        if before_text == after_text:
            continue
        diff = difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        if diff_text:
            diffs.append(diff_text)
    return "\n\n".join(diffs)


def get_git_diff(repo: Path) -> str | None:
    try:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if status.returncode != 0:
        return None

    if not status.stdout.strip():
        return ""

    diffs: list[str] = []
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff"],
        capture_output=True,
        text=True,
    )
    if diff.returncode == 0 and diff.stdout.strip():
        diffs.append(diff.stdout.strip())

    cached = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached"],
        capture_output=True,
        text=True,
    )
    if cached.returncode == 0 and cached.stdout.strip():
        diffs.append(cached.stdout.strip())

    untracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    if untracked.returncode == 0:
        for path in [line.strip() for line in untracked.stdout.splitlines() if line.strip()]:
            file_diff = subprocess.run(
                ["git", "-C", str(repo), "diff", "--no-index", "--", "/dev/null", path],
                capture_output=True,
                text=True,
            )
            if file_diff.returncode in {0, 1} and file_diff.stdout.strip():
                diffs.append(file_diff.stdout.strip())

    return "\n\n".join(diffs)


def infer_language_from_diff(diff_text: str) -> str | None:
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            suffix = Path(path).suffix.lower()
            if suffix == ".py":
                return "python"
            if suffix == ".js":
                return "javascript"
            if suffix in {".ts", ".tsx"}:
                return "typescript"
            if suffix == ".json":
                return "json"
            if suffix in {".yml", ".yaml"}:
                return "yaml"
            if suffix in {".md", ".markdown"}:
                return "markdown"
            if suffix == ".sh":
                return "bash"
            if suffix == ".go":
                return "go"
            if suffix == ".rs":
                return "rust"
            if suffix in {".c", ".h"}:
                return "c"
            if suffix in {".cpp", ".hpp", ".cc", ".cxx"}:
                return "cpp"
            return None
    return None


def summarize_diff(diff_text: str) -> tuple[list[str], int, int]:
    files: list[str] = []
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                files.append(path)
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return sorted(set(files)), added, removed
