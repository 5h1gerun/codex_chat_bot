import os
import shlex
import subprocess
import shutil
from pathlib import Path


def run_codex(prompt: str, cwd: Path) -> str:
    codex_path = os.getenv("CODEX_PATH", "codex")
    raw_args = os.getenv("CODEX_ARGS", "")
    timeout_s = int(os.getenv("CODEX_TIMEOUT", "120"))
    skip_git_check = os.getenv("CODEX_SKIP_GIT_CHECK", "true").lower() in {"1", "true", "yes"}

    if not cwd.exists() or not cwd.is_dir():
        return "作業ディレクトリが見つかりません。/repo list で選択し直してください。"

    if Path(codex_path).is_absolute():
        if not Path(codex_path).exists():
            return "CODEX_PATHが存在しません。パスを確認してください。"
    else:
        if shutil.which(codex_path) is None:
            return "CODEX_PATHが見つかりません。PATHまたはCODEX_PATHを確認してください。"

    args = shlex.split(raw_args)
    subcommands = {
        "exec",
        "review",
        "login",
        "logout",
        "mcp",
        "mcp-server",
        "app-server",
        "completion",
        "sandbox",
        "apply",
        "resume",
        "fork",
        "cloud",
        "features",
        "help",
    }
    cmd = [codex_path]
    if not args or args[0] not in subcommands:
        cmd.append("exec")
        if skip_git_check:
            cmd.append("--skip-git-repo-check")
    cmd += args
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "Codexの実行がタイムアウトしました。リクエストを小さくしてください。"
    except OSError:
        return "Codexの起動に失敗しました。CODEX_PATHと権限を確認してください。"

    if result.returncode != 0:
        return "Codexの実行に失敗しました。ログを確認してください。"

    output = result.stdout.strip()
    if not output:
        return "Codexからの出力がありませんでした。"

    return output
