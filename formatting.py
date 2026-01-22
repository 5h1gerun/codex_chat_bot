from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Deque

import discord

from config import CODEX_LANGUAGE


def build_prompt(history: Deque[tuple[str, str]], user_message: str) -> str:
    lines = []
    if CODEX_LANGUAGE.lower() == "ja":
        lines.append("System: 出力はすべて日本語で返答してください。")
        lines.append("System: 実行環境にはローカルのファイルアクセスが可能です。必要に応じて参照してください。")
        lines.append("System: もし参照できない場合は理由を具体的に説明してください。")
        lines.append("System: まずREADMEとtreeで全体把握し、目的に関係するファイルだけ読み、不足があれば追加で読む方針で進めてください。")
    for user_text, assistant_text in history:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def split_message(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def chunk_code_block(code: str, limit: int, language: str | None = None) -> list[str]:
    lang_tag = language or ""
    fence_start = f"```{lang_tag}\n"
    fence_end = "\n```"
    max_len = limit - len(fence_start) - len(fence_end)
    if max_len <= 0:
        return []

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_lines, current_len
        if not current_lines:
            return
        chunk_body = "\n".join(current_lines)
        chunks.append(f"{fence_start}{chunk_body}{fence_end}")
        current_lines = []
        current_len = 0

    def append_line(line: str) -> None:
        nonlocal current_len
        if len(line) > max_len:
            ellipsis = "..."
            if max_len <= len(ellipsis):
                part_len = max_len
            else:
                part_len = max_len - len(ellipsis)
            remaining = line
            while len(remaining) > max_len:
                part = remaining[:part_len]
                if max_len > len(ellipsis):
                    part = f"{part}{ellipsis}"
                remaining = remaining[part_len:]
                if current_lines:
                    flush()
                current_lines.append(part)
                current_len = len(part)
                flush()
            if remaining:
                if current_lines:
                    flush()
                current_lines.append(remaining)
                current_len = len(remaining)
                flush()
            return
        extra = len(line) + (1 if current_lines else 0)
        if current_len + extra > max_len and current_lines:
            flush()
            extra = len(line)
        current_lines.append(line)
        current_len += extra

    for line in code.splitlines():
        append_line(line)

    if current_lines:
        flush()

    return chunks


def code_block_max_len(limit: int, language: str | None = None) -> int:
    lang_tag = language or ""
    fence_start = f"```{lang_tag}\n"
    fence_end = "\n```"
    return limit - len(fence_start) - len(fence_end)


def text_to_file(text: str, filename: str) -> discord.File:
    data = text.encode("utf-8")
    return discord.File(io.BytesIO(data), filename=filename)


def language_to_extension(language: str | None) -> str:
    if not language:
        return "txt"
    lang = language.lower()
    if lang in {"py", "python"}:
        return "py"
    if lang in {"js", "javascript"}:
        return "js"
    if lang in {"ts", "typescript"}:
        return "ts"
    if lang in {"tsx"}:
        return "tsx"
    if lang in {"json"}:
        return "json"
    if lang in {"yaml", "yml"}:
        return "yml"
    if lang in {"md", "markdown"}:
        return "md"
    if lang in {"bash", "sh", "shell"}:
        return "sh"
    if lang in {"go"}:
        return "go"
    if lang in {"rs", "rust"}:
        return "rs"
    if lang in {"c"}:
        return "c"
    if lang in {"cpp", "c++"}:
        return "cpp"
    if lang in {"diff", "patch"}:
        return "diff"
    return "txt"


def build_attachment_name(prefix: str, extension: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_prefix = prefix.replace(" ", "_")
    return f"{safe_prefix}_{stamp}.{extension}"


def extract_code_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    out_lines: list[str] = []
    in_block = False
    lang = ""
    code_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                lang = line.strip()[3:].strip()
                code_lines = []
            else:
                in_block = False
                blocks.append((lang or "text", "\n".join(code_lines)))
                lang = ""
                code_lines = []
            continue

        if in_block:
            code_lines.append(line)
        else:
            out_lines.append(line)

    if in_block and code_lines:
        blocks.append((lang or "text", "\n".join(code_lines)))

    return "\n".join(out_lines).strip(), blocks


def summarize_output(
    plain_text: str,
    code_blocks: list[tuple[str, str]],
    diff_text: str | None,
) -> str:
    lines = [line.strip() for line in plain_text.splitlines() if line.strip()]
    if lines:
        summary = " / ".join(lines[:2])
    else:
        summary = "コード/差分のみ"
    tags: list[str] = []
    if code_blocks:
        tags.append("コードあり")
    if diff_text:
        tags.append("差分あり")
    if tags:
        summary = f"{summary} ({', '.join(tags)})"
    return truncate_text(single_line(summary), 200)


def single_line(text: str) -> str:
    return " ".join(text.strip().split())


def truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."
