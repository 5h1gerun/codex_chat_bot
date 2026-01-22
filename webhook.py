import aiohttp
import discord

from config import WEBHOOK_ENABLED, WEBHOOK_TIMEOUT, WEBHOOK_URL


async def send_webhook(embed: discord.Embed) -> None:
    if not WEBHOOK_ENABLED or not WEBHOOK_URL:
        return
    try:
        timeout = aiohttp.ClientTimeout(total=WEBHOOK_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            webhook = discord.Webhook.from_url(WEBHOOK_URL, session=session)
            await webhook.send(embed=embed)
    except aiohttp.ClientError:
        return
