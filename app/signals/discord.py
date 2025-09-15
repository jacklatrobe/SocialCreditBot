# signals/discord.py
from typing import Optional, List, Dict, Any
from .base import Signal


class DiscordMessage(Signal):
    """
    Discord-specific extension of Signal.
    
    Adds Discord-specific fields while maintaining compatibility with the base Signal interface.
    """
    guild_name: Optional[str] = None
    channel_name: Optional[str] = None
    attachments: List[Dict[str, Any]] = []
    embeds: List[Dict[str, Any]] = []
    mentions: List[Dict[str, Any]] = []
    
    @classmethod
    def from_discord_message(cls, discord_msg, source: str = "discord") -> "DiscordMessage":
        """
        Factory method to create DiscordMessage from discord.py Message object.
        
        Args:
            discord_msg: discord.py Message object
            source: Source identifier (default: "discord")
            
        Returns:
            DiscordMessage instance
        """
        # Build context dict with Discord-specific fields
        context = {
            "guild_id": str(discord_msg.guild.id) if discord_msg.guild else None,
            "channel_id": str(discord_msg.channel.id),
            "thread_id": str(discord_msg.channel.id) if hasattr(discord_msg.channel, 'parent') else None,
            "message_id": str(discord_msg.id),
            "reply_to_id": str(discord_msg.reference.message_id) if discord_msg.reference else None
        }
        
        # Build author dict
        author = {
            "user_id": str(discord_msg.author.id),
            "username": discord_msg.author.display_name or discord_msg.author.name
        }
        
        # Extract attachments info
        attachments = [
            {
                "id": str(att.id),
                "filename": att.filename,
                "url": att.url,
                "content_type": att.content_type,
                "size": att.size
            }
            for att in discord_msg.attachments
        ]
        
        # Extract embeds info (simplified)
        embeds = [
            {
                "type": embed.type,
                "title": embed.title,
                "description": embed.description,
                "url": embed.url
            }
            for embed in discord_msg.embeds
        ]
        
        # Extract mentions
        mentions = [
            {
                "user_id": str(mention.id),
                "username": mention.display_name or mention.name,
                "discriminator": mention.discriminator
            }
            for mention in discord_msg.mentions
        ]
        
        return cls(
            signal_id=f"src:discord:msg:{discord_msg.id}",
            source=source,
            created_at=discord_msg.created_at,
            author=author,
            context=context,
            content=discord_msg.content or "",  # Handle empty content (e.g., embeds only)
            guild_name=discord_msg.guild.name if discord_msg.guild else None,
            channel_name=discord_msg.channel.name if hasattr(discord_msg.channel, 'name') else None,
            attachments=attachments,
            embeds=embeds,
            mentions=mentions,
            metadata={
                "lang": "en",  # Default, could be detected later
                "message_type": "default"
            }
        )