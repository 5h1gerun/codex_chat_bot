from pathlib import Path

import discord

from state import get_state, is_allowed_channel, is_allowed_role, safe_repo_path


class RepoSelect(discord.ui.Select):
    def __init__(self, repos: list[Path]):
        options = [
            discord.SelectOption(label=repo.name, value=repo.name)
            for repo in repos[:25]
        ]
        super().__init__(placeholder="Select a repo", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if not is_allowed_channel(interaction.channel_id):
            await interaction.response.send_message("このチャンネルでは使用できません。", ephemeral=True)
            return
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        if not is_allowed_role(member):
            await interaction.response.send_message("必要なロールがありません。", ephemeral=True)
            return
        selected = self.values[0]
        repo = safe_repo_path(selected)
        if repo is None:
            await interaction.response.send_message("repoの選択が無効です。", ephemeral=True)
            return
        if interaction.channel_id is None:
            await interaction.response.send_message("チャンネルが見つかりません。", ephemeral=True)
            return
        state = get_state(interaction.channel_id)
        state.repo = repo
        state.history.clear()
        await interaction.response.send_message(f"`{repo.name}` に切り替え、履歴をリセットしました。")


class RepoView(discord.ui.View):
    def __init__(self, repos: list[Path]):
        super().__init__(timeout=60)
        self.add_item(RepoSelect(repos))
