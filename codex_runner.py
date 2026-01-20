import os
import shlex
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timezone


def _write_error_log(cwd: Path, label: str, content: str) -> str:
    logs_dir = cwd / "codex_error_logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"codex_error_{timestamp}.txt"
    try:
        path.write_text(f"{label}\n\n{content}".strip() + "\n", encoding="utf-8")
    except OSError:
        return ""
    return str(path)


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
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        cmd_text = shlex.join(cmd)
        log_text = (
            f"[timeout]\ncmd: {cmd_text}\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}\n"
        )
        log_path = _write_error_log(cwd, "Codex timeout", log_text)
        if log_path:
            return f"Codexの実行がタイムアウトしました。エラーログ: {log_path}"
        return "Codexの実行がタイムアウトしました。エラーログの出力に失敗しました。"
    except OSError as exc:
        cmd_text = shlex.join(cmd)
        log_text = f"[oserror]\ncmd: {cmd_text}\nerror: {exc}\n"
        log_path = _write_error_log(cwd, "Codex start failed", log_text)
        if log_path:
            return f"Codexの起動に失敗しました。エラーログ: {log_path}"
        return "Codexの起動に失敗しました。CODEX_PATHと権限を確認してください。"

    if result.returncode != 0:
        cmd_text = shlex.join(cmd)
        log_text = (
            f"[nonzero return]\ncmd: {cmd_text}\nreturncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n"
        )
        log_path = _write_error_log(cwd, "Codex execution failed", log_text)
        if log_path:
            return f"Codexの実行に失敗しました。エラーログ: {log_path}"
        return "Codexの実行に失敗しました。エラーログの出力に失敗しました。"

    output = result.stdout.strip()
    if not output:
        return "Codexからの出力がありませんでした。"

    return output
