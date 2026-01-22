import subprocess
from pathlib import Path

from config import AUTO_VERIFY_COMMAND, AUTO_VERIFY_TIMEOUT


def run_auto_verify(repo: Path) -> tuple[str, int] | None:
    if not AUTO_VERIFY_COMMAND:
        return ("検証コマンドが未設定です。AUTO_VERIFY_COMMAND を設定してください。", 2)
    try:
        result = subprocess.run(
            AUTO_VERIFY_COMMAND,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(repo),
            timeout=AUTO_VERIFY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ("検証がタイムアウトしました。", 124)
    except OSError:
        return ("検証コマンドの実行に失敗しました。", 125)

    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    if not output:
        output = "出力はありませんでした。"
    return (output, result.returncode)
