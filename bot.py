import asyncio
import difflib
import io
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque

import discord
from dotenv import load_dotenv

from codex_runner import run_codex


@dataclass
class ChannelState:
    repo: Path | None
    history: Deque[tuple[str, str]]
    enabled_until: datetime | None


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT")
ENABLE_DEFAULT_MINUTES = int(os.getenv("ENABLE_DEFAULT_MINUTES", "15"))
ALLOWED_ROLE_ID = os.getenv("ALLOWED_ROLE_ID")
TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
MAX_OUTPUT_CHARS = int(os.getenv("MAX_OUTPUT_CHARS", "1900"))
CODEX_LANGUAGE = os.getenv("CODEX_LANGUAGE", "ja")
CODEX_DEFAULT_LANGUAGE = os.getenv("CODEX_DEFAULT_LANGUAGE", "text")
MAX_SNAPSHOT_FILES = int(os.getenv("MAX_SNAPSHOT_FILES", "200"))
MAX_SNAPSHOT_FILE_BYTES = int(os.getenv("MAX_SNAPSHOT_FILE_BYTES", "200000"))
MAX_SNAPSHOT_TOTAL_BYTES = int(os.getenv("MAX_SNAPSHOT_TOTAL_BYTES", "1000000"))

if not DISCORD_TOKEN:
    raise SystemExit("DISCORD_TOKEN is required")
if not WORKSPACE_ROOT:
    raise SystemExit("WORKSPACE_ROOT is required")

WORKSPACE_ROOT_PATH = Path(WORKSPACE_ROOT).expanduser().resolve()
if not WORKSPACE_ROOT_PATH.exists():
    raise SystemExit(f"WORKSPACE_ROOT not found: {WORKSPACE_ROOT_PATH}")

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

bot = discord.Bot(intents=intents)
channel_state: dict[int, ChannelState] = {}


def _get_state(channel_id: int) -> ChannelState:
    state = channel_state.get(channel_id)
    if state is None:
        state = ChannelState(repo=None, history=deque(maxlen=MAX_HISTORY_TURNS), enabled_until=None)
        channel_state[channel_id] = state
    return state


def _is_allowed_role(member: discord.Member | None) -> bool:
    if not ALLOWED_ROLE_ID:
        return True
    if member is None:
        return False
    try:
        allowed_id = int(ALLOWED_ROLE_ID)
    except ValueError:
        return False
    return any(role.id == allowed_id for role in member.roles)


def _is_allowed_channel(channel_id: int | None) -> bool:
    if not TARGET_CHANNEL_ID:
        return True
    if channel_id is None:
        return False
    try:
        target_id = int(TARGET_CHANNEL_ID)
    except ValueError:
        return False
    return channel_id == target_id


def _is_enabled(channel_id: int) -> bool:
    state = _get_state(channel_id)
    if state.enabled_until is None:
        return False
    return datetime.now(timezone.utc) < state.enabled_until


def _available_repos() -> list[Path]:
    repos: list[Path] = []
    for entry in WORKSPACE_ROOT_PATH.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            repos.append(entry)
    return sorted(repos)


def _filter_repos(repos: list[Path], prefix: str | None) -> list[Path]:
    if not prefix:
        return repos
    trimmed = prefix.strip()
    if not trimmed:
        return repos
    return [repo for repo in repos if repo.name.startswith(trimmed)]


def _safe_repo_path(selected_name: str) -> Path | None:
    candidate = (WORKSPACE_ROOT_PATH / selected_name).resolve()
    if candidate.is_dir() and candidate.is_relative_to(WORKSPACE_ROOT_PATH):
        return candidate
    return None


def _build_prompt(history: Deque[tuple[str, str]], user_message: str) -> str:
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


def _split_message(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def _chunk_code_block(code: str, limit: int, language: str | None = None) -> list[str]:
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


def _snapshot_files(repo: Path) -> dict[str, str]:
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


def _build_snapshot_diff(before: dict[str, str], repo: Path) -> str:
    after = _snapshot_files(repo)
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


def _code_block_max_len(limit: int, language: str | None = None) -> int:
    lang_tag = language or ""
    fence_start = f"```{lang_tag}\n"
    fence_end = "\n```"
    return limit - len(fence_start) - len(fence_end)


def _text_to_file(text: str, filename: str) -> discord.File:
    data = text.encode("utf-8")
    return discord.File(io.BytesIO(data), filename=filename)


def _language_to_extension(language: str | None) -> str:
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


def _build_attachment_name(prefix: str, extension: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_prefix = prefix.replace(" ", "_")
    return f"{safe_prefix}_{stamp}.{extension}"


def _get_git_diff(repo: Path) -> str | None:
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


def _infer_language_from_diff(diff_text: str) -> str | None:
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


def _extract_code_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
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


@bot.event
async def on_ready() -> None:
    if GUILD_ID:
        try:
            await bot.sync_commands(guild_ids=[int(GUILD_ID)])
        except ValueError:
            await bot.sync_commands()
    else:
        await bot.sync_commands()
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if not bot.user:
        return
    if bot.user not in message.mentions:
        return
    if not _is_allowed_channel(message.channel.id):
        return
    member = message.author if isinstance(message.author, discord.Member) else None
    if not _is_allowed_role(member):
        return
    if not _is_enabled(message.channel.id):
        return

    content = message.content
    content = content.replace(f"<@{bot.user.id}>", "")
    content = content.replace(f"<@!{bot.user.id}>", "")
    prompt = content.strip()
    if not prompt:
        return

    state = _get_state(message.channel.id)
    if state.repo is None:
        await message.channel.send("repoが未選択です。/repo list で選択してください。")
        return

    snapshot_before = _snapshot_files(state.repo)
    history_prompt = _build_prompt(state.history, prompt)
    async with message.channel.typing():
        output = await asyncio.to_thread(run_codex, history_prompt, state.repo)

    state.history.append((prompt, output))

    plain_text, code_blocks = _extract_code_blocks(output)
    diff_text = _get_git_diff(state.repo)
    if not diff_text:
        diff_text = _build_snapshot_diff(snapshot_before, state.repo)

    if plain_text:
        if len(plain_text) > MAX_OUTPUT_CHARS:
            filename = _build_attachment_name("codex_output", "txt")
            await message.channel.send(
                "長文のためファイルで送信します。",
                file=_text_to_file(plain_text, filename),
            )
        else:
            for chunk in _split_message(plain_text, MAX_OUTPUT_CHARS):
                await message.channel.send(chunk)

    for language, code in code_blocks:
        lang_tag = language if language else CODEX_DEFAULT_LANGUAGE
        chunks = _chunk_code_block(code, MAX_OUTPUT_CHARS, lang_tag)
        if not chunks:
            ext = _language_to_extension(lang_tag)
            filename = _build_attachment_name("codex_code", ext)
            await message.channel.send(
                "コードの送信に失敗したためファイルで送信します。",
                file=_text_to_file(code, filename),
            )
            continue
        for chunk in chunks:
            await message.channel.send(chunk)

    if diff_text:
        inferred = _infer_language_from_diff(diff_text) or CODEX_DEFAULT_LANGUAGE
        chunks = _chunk_code_block(diff_text, MAX_OUTPUT_CHARS, inferred)
        if not chunks:
            filename = _build_attachment_name("git_diff", "diff")
            await message.channel.send(
                "差分の送信に失敗したためファイルで送信します。",
                file=_text_to_file(diff_text, filename),
            )
        else:
            for chunk in chunks:
                await message.channel.send(chunk)


class RepoSelect(discord.ui.Select):
    def __init__(self, repos: list[Path]):
        options = [
            discord.SelectOption(label=repo.name, value=repo.name)
            for repo in repos[:25]
        ]
        super().__init__(placeholder="Select a repo", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_allowed_channel(interaction.channel_id):
            await interaction.response.send_message("このチャンネルでは使用できません。", ephemeral=True)
            return
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        if not _is_allowed_role(member):
            await interaction.response.send_message("必要なロールがありません。", ephemeral=True)
            return
        selected = self.values[0]
        repo = _safe_repo_path(selected)
        if repo is None:
            await interaction.response.send_message("repoの選択が無効です。", ephemeral=True)
            return
        if interaction.channel_id is None:
            await interaction.response.send_message("チャンネルが見つかりません。", ephemeral=True)
            return
        state = _get_state(interaction.channel_id)
        state.repo = repo
        state.history.clear()
        await interaction.response.send_message(f"`{repo.name}` に切り替え、履歴をリセットしました。")


class RepoView(discord.ui.View):
    def __init__(self, repos: list[Path]):
        super().__init__(timeout=60)
        self.add_item(RepoSelect(repos))

codex = discord.SlashCommandGroup("codex", "Codex controls")
repo = discord.SlashCommandGroup("repo", "Workspace selection")


@codex.command(name="enable", description="Enable Codex for a limited time")
@discord.option("minutes", int, description="有効化する分数", required=False)
async def enable(
    ctx: discord.ApplicationContext,
    minutes: int | None = None,
) -> None:
    if not _is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not _is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return

    duration = minutes if minutes and minutes > 0 else ENABLE_DEFAULT_MINUTES
    state = _get_state(ctx.channel.id)
    state.enabled_until = datetime.now(timezone.utc) + timedelta(minutes=duration)
    await ctx.respond(f"{duration}分間有効化しました。")


@codex.command(name="disable", description="Disable Codex immediately")
async def disable(ctx: discord.ApplicationContext) -> None:
    if not _is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not _is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return
    state = _get_state(ctx.channel.id)
    state.enabled_until = None
    await ctx.respond("無効化しました。")


@repo.command(name="current", description="Show the current repo")
async def current(ctx: discord.ApplicationContext) -> None:
    if not _is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not _is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return
    state = _get_state(ctx.channel.id)
    if state.repo is None:
        await ctx.respond("repoが未選択です。", ephemeral=True)
        return
    await ctx.respond(f"現在のrepo: `{state.repo.name}`")


@repo.command(name="list", description="Select a repo from workspace")
@discord.option("prefix", str, description="先頭一致フィルタ（例: app-）", required=False)
async def list_repos(ctx: discord.ApplicationContext, prefix: str | None = None) -> None:
    if not _is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not _is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return
    repos = _filter_repos(_available_repos(), prefix)
    if not repos:
        if prefix:
            await ctx.respond(f"`{prefix}` に一致するrepoが見つかりません。", ephemeral=True)
        else:
            await ctx.respond("workspaceにrepoが見つかりません。", ephemeral=True)
        return
    await ctx.respond("repoを選択してください。", view=RepoView(repos), ephemeral=True)


bot.add_application_command(codex)
bot.add_application_command(repo)

bot.run(DISCORD_TOKEN)
