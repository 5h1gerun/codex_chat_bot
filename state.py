from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

import discord

from config import (
    ALLOWED_ROLE_ID,
    AUTO_VERIFY_DEFAULT,
    MAX_HISTORY_TURNS,
    TARGET_CHANNEL_ID,
    WORKSPACE_ROOT_PATH,
)


@dataclass
class ChannelState:
    repo: Path | None
    history: Deque[tuple[str, str]]
    enabled_until: datetime | None
    auto_verify: bool


channel_state: dict[int, ChannelState] = {}


def get_state(channel_id: int) -> ChannelState:
    state = channel_state.get(channel_id)
    if state is None:
        state = ChannelState(
            repo=None,
            history=deque(maxlen=MAX_HISTORY_TURNS),
            enabled_until=None,
            auto_verify=AUTO_VERIFY_DEFAULT,
        )
        channel_state[channel_id] = state
    return state


def is_allowed_role(member: discord.Member | None) -> bool:
    if not ALLOWED_ROLE_ID:
        return True
    if member is None:
        return False
    try:
        allowed_id = int(ALLOWED_ROLE_ID)
    except ValueError:
        return False
    return any(role.id == allowed_id for role in member.roles)


def is_allowed_channel(channel_id: int | None) -> bool:
    if not TARGET_CHANNEL_ID:
        return True
    if channel_id is None:
        return False
    try:
        target_id = int(TARGET_CHANNEL_ID)
    except ValueError:
        return False
    return channel_id == target_id


def is_enabled(channel_id: int) -> bool:
    state = get_state(channel_id)
    if state.enabled_until is None:
        return False
    return datetime.now(timezone.utc) < state.enabled_until


def available_repos() -> list[Path]:
    repos: list[Path] = []
    for entry in WORKSPACE_ROOT_PATH.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            repos.append(entry)
    return sorted(repos)


def filter_repos(repos: list[Path], prefix: str | None) -> list[Path]:
    if not prefix:
        return repos
    trimmed = prefix.strip()
    if not trimmed:
        return repos
    return [repo for repo in repos if repo.name.startswith(trimmed)]


def safe_repo_path(selected_name: str) -> Path | None:
    candidate = (WORKSPACE_ROOT_PATH / selected_name).resolve()
    if candidate.is_dir() and candidate.is_relative_to(WORKSPACE_ROOT_PATH):
        return candidate
    return None
