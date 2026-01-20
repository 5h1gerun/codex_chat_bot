import os
from pathlib import Path

from dotenv import load_dotenv

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
AUTO_VERIFY_DEFAULT = os.getenv("AUTO_VERIFY_DEFAULT", "false").lower() in {"1", "true", "yes"}
AUTO_VERIFY_COMMAND = os.getenv("AUTO_VERIFY_COMMAND", "")
AUTO_VERIFY_TIMEOUT = int(os.getenv("AUTO_VERIFY_TIMEOUT", "120"))
WEBHOOK_ENABLED = os.getenv("WEBHOOK_ENABLED", "false").lower() in {"1", "true", "yes"}
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", "10"))

if not DISCORD_TOKEN:
    raise SystemExit("DISCORD_TOKEN is required")
if not WORKSPACE_ROOT:
    raise SystemExit("WORKSPACE_ROOT is required")

WORKSPACE_ROOT_PATH = Path(WORKSPACE_ROOT).expanduser().resolve()
if not WORKSPACE_ROOT_PATH.exists():
    raise SystemExit(f"WORKSPACE_ROOT not found: {WORKSPACE_ROOT_PATH}")
