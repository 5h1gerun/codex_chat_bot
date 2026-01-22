import asyncio
import time
from datetime import datetime, timedelta, timezone

import discord

from codex_runner import run_codex
from config import (
    AUTO_VERIFY_COMMAND,
    CODEX_DEFAULT_LANGUAGE,
    DISCORD_TOKEN,
    ENABLE_DEFAULT_MINUTES,
    GUILD_ID,
    MAX_OUTPUT_CHARS,
    WEBHOOK_ENABLED,
    WEBHOOK_URL,
    WORKSPACE_ROOT_PATH,
)
from diff_utils import (
    build_snapshot_diff,
    get_git_diff,
    infer_language_from_diff,
    snapshot_files,
    summarize_diff,
)
from formatting import (
    build_attachment_name,
    build_prompt,
    chunk_code_block,
    extract_code_blocks,
    language_to_extension,
    single_line,
    split_message,
    summarize_output,
    text_to_file,
    truncate_text,
)
from state import (
    available_repos,
    filter_repos,
    get_state,
    is_allowed_channel,
    is_allowed_role,
    is_enabled,
)
from ui import RepoView
from verify import run_auto_verify
from webhook import send_webhook

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

bot = discord.Bot(intents=intents)


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
    if not is_allowed_channel(message.channel.id):
        return
    member = message.author if isinstance(message.author, discord.Member) else None
    if not is_allowed_role(member):
        return
    if not is_enabled(message.channel.id):
        return

    content = message.content
    content = content.replace(f"<@{bot.user.id}>", "")
    content = content.replace(f"<@!{bot.user.id}>", "")
    prompt = content.strip()
    if not prompt:
        return

    state = get_state(message.channel.id)
    if state.repo is None:
        await message.channel.send("repoが未選択です。/repo list で選択してください。")
        return

    state.pending_requests += 1
    if state.queue_lock.locked():
        await message.channel.send(
            f"現在他の依頼を処理中です。順番待ち({state.pending_requests}番目)に追加しました。"
        )

    async with state.queue_lock:
        state.pending_requests = max(0, state.pending_requests - 1)
        snapshot_before, external_before = snapshot_files(state.repo)
        history_prompt = build_prompt(state.history, prompt)
        start_time = time.monotonic()
        async with message.channel.typing():
            output = await asyncio.to_thread(run_codex, history_prompt, state.repo)
        elapsed_s = time.monotonic() - start_time

        state.history.append((prompt, output))

        plain_text, code_blocks = extract_code_blocks(output)
        git_diff = get_git_diff(state.repo)
        diff_source = "git" if git_diff is not None else "snapshot"
        diff_text = git_diff
        if not diff_text:
            diff_text = build_snapshot_diff(snapshot_before, external_before, state.repo)
        has_changes = bool(diff_text)
        files_changed: list[str] = []
        added_lines = 0
        removed_lines = 0
        if diff_text:
            files_changed, added_lines, removed_lines = summarize_diff(diff_text)

        if plain_text:
            if len(plain_text) > MAX_OUTPUT_CHARS:
                filename = build_attachment_name("codex_output", "txt")
                await message.channel.send(
                    "長文のためファイルで送信します。",
                    file=text_to_file(plain_text, filename),
                )
            else:
                for chunk in split_message(plain_text, MAX_OUTPUT_CHARS):
                    await message.channel.send(chunk)

        for language, code in code_blocks:
            lang_tag = language if language else CODEX_DEFAULT_LANGUAGE
            chunks = chunk_code_block(code, MAX_OUTPUT_CHARS, lang_tag)
            if not chunks:
                ext = language_to_extension(lang_tag)
                filename = build_attachment_name("codex_code", ext)
                await message.channel.send(
                    "コードの送信に失敗したためファイルで送信します。",
                    file=text_to_file(code, filename),
                )
                continue
            for chunk in chunks:
                await message.channel.send(chunk)

        if diff_text:
            inferred = infer_language_from_diff(diff_text) or CODEX_DEFAULT_LANGUAGE
            chunks = chunk_code_block(diff_text, MAX_OUTPUT_CHARS, inferred)
            if not chunks:
                filename = build_attachment_name("git_diff", "diff")
                await message.channel.send(
                    "差分の送信に失敗したためファイルで送信します。",
                    file=text_to_file(diff_text, filename),
                )
            else:
                for chunk in chunks:
                    await message.channel.send(chunk)

        verify_summary = "未実行"
        if state.auto_verify and has_changes:
            verify_result = await asyncio.to_thread(run_auto_verify, state.repo)
            if verify_result:
                verify_text, verify_code = verify_result
                verify_summary = f"exit={verify_code}"
                header = f"検証結果 (exit={verify_code})"
                message_body = f"{header}\n{verify_text}"
                if len(message_body) > MAX_OUTPUT_CHARS:
                    filename = build_attachment_name("verify_output", "txt")
                    await message.channel.send(
                        "検証結果が長いためファイルで送信します。",
                        file=text_to_file(message_body, filename),
                    )
                else:
                    for chunk in split_message(message_body, MAX_OUTPUT_CHARS):
                        await message.channel.send(chunk)
        elif state.auto_verify:
            verify_summary = "変更なしのため未実行"
        else:
            verify_summary = "無効"

        if WEBHOOK_ENABLED and WEBHOOK_URL:
            channel_name = getattr(message.channel, "name", "unknown")
            file_count = len(files_changed)
            if file_count:
                top_files = files_changed[:3]
                remainder = file_count - len(top_files)
                file_list = ", ".join(top_files)
                if remainder > 0:
                    file_list = f"{file_list} (+{remainder} files)"
            else:
                file_list = "なし"
            change_summary = f"{file_count} files, +{added_lines} -{removed_lines}"
            prompt_summary = truncate_text(single_line(prompt), 200) or "なし"
            response_summary = summarize_output(plain_text, code_blocks, diff_text)
            jst = timezone(timedelta(hours=9))
            timestamp = datetime.now(jst)
            embed = discord.Embed(
                title="Codex 実行レポート",
                description=f"依頼: {prompt_summary}\n変更: {change_summary}\n検証: {verify_summary}",
                color=0x5DADEC,
                timestamp=timestamp,
            )
            embed.set_author(name="Codex Runner")
            embed.set_thumbnail(url=message.author.display_avatar.url)
            embed.add_field(name="実行者", value=message.author.display_name, inline=True)
            embed.add_field(name="チャンネル", value=f"#{channel_name}", inline=True)
            embed.add_field(name="repo", value=state.repo.name, inline=True)
            embed.add_field(name="変更概要", value=change_summary, inline=True)
            embed.add_field(name="変更ファイル", value=file_list, inline=False)
            embed.add_field(name="検証", value=verify_summary, inline=True)
            embed.add_field(name="実行時間", value=f"{elapsed_s:.1f}s", inline=True)
            embed.add_field(name="差分取得", value=diff_source, inline=True)
            embed.add_field(name="応答サマリ", value=response_summary, inline=False)
            await send_webhook(embed)


codex = discord.SlashCommandGroup("codex", "Codex controls")
repo = discord.SlashCommandGroup("repo", "Workspace selection")


@codex.command(name="enable", description="Enable Codex for a limited time")
@discord.option("minutes", int, description="有効化する分数", required=False)
async def enable(
    ctx: discord.ApplicationContext,
    minutes: int | None = None,
) -> None:
    if not is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return

    duration = minutes if minutes and minutes > 0 else ENABLE_DEFAULT_MINUTES
    state = get_state(ctx.channel.id)
    state.enabled_until = datetime.now(timezone.utc) + timedelta(minutes=duration)
    await ctx.respond(f"{duration}分間有効化しました。")


@codex.command(name="disable", description="Disable Codex immediately")
async def disable(ctx: discord.ApplicationContext) -> None:
    if not is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return
    state = get_state(ctx.channel.id)
    state.enabled_until = None
    await ctx.respond("無効化しました。")


@codex.command(name="verify", description="Toggle auto verification after Codex runs")
@discord.option("enabled", bool, description="自動検証を有効化する", required=False)
async def verify(ctx: discord.ApplicationContext, enabled: bool | None = None) -> None:
    if not is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return
    state = get_state(ctx.channel.id)
    if enabled is None:
        state.auto_verify = not state.auto_verify
    else:
        state.auto_verify = enabled

    status = "有効" if state.auto_verify else "無効"
    if state.auto_verify and not AUTO_VERIFY_COMMAND:
        await ctx.respond(f"自動検証を{status}にしましたが、AUTO_VERIFY_COMMAND が未設定です。", ephemeral=True)
        return
    await ctx.respond(f"自動検証を{status}にしました。", ephemeral=True)


@repo.command(name="current", description="Show the current repo")
async def current(ctx: discord.ApplicationContext) -> None:
    if not is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return
    state = get_state(ctx.channel.id)
    if state.repo is None:
        await ctx.respond("repoが未選択です。", ephemeral=True)
        return
    await ctx.respond(f"現在のrepo: `{state.repo.name}`")


@repo.command(name="list", description="Select a repo from workspace")
@discord.option("prefix", str, description="先頭一致フィルタ（例: app-）", required=False)
async def list_repos(ctx: discord.ApplicationContext, prefix: str | None = None) -> None:
    if not is_allowed_channel(ctx.channel.id if ctx.channel else None):
        await ctx.respond("このチャンネルでは使用できません。", ephemeral=True)
        return
    member = ctx.user if isinstance(ctx.user, discord.Member) else None
    if not is_allowed_role(member):
        await ctx.respond("必要なロールがありません。", ephemeral=True)
        return
    if not ctx.channel:
        await ctx.respond("チャンネルが見つかりません。", ephemeral=True)
        return
    repos = filter_repos(available_repos(), prefix)
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
