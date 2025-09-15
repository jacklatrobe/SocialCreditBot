"""
Configuration management for Discord Observer/Orchestrator.
Handles environment variables and configuration validation.
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Discord Configuration (Required)
    discord_bot_token: str = Field(..., env="DISCORD_BOT_TOKEN")
    discord_app_id: Optional[str] = Field(None, env="DISCORD_APP_ID")
    discord_intents: str = Field("message_content,guilds,guild_messages", env="DISCORD_INTENTS")
    
    # LLM Configuration (Required)
    llm_provider: str = Field("openai", env="LLM_PROVIDER")
    llm_api_key: str = Field(..., env="LLM_API_KEY")
    llm_model: str = Field("gpt-4o-mini", env="LLM_MODEL")  # Using available model instead of GPT-5-nano
    
    # Database Configuration
    db_path: str = Field("./data/app.db", env="DB_PATH")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Admin API Configuration
    admin_api_host: str = Field("127.0.0.1", env="ADMIN_API_HOST")
    admin_api_port: int = Field(8000, env="ADMIN_API_PORT")
    
    # Rate Limiting Configuration
    max_messages_per_minute: int = Field(100, env="MAX_MESSAGES_PER_MINUTE")
    max_llm_calls_per_minute: int = Field(50, env="MAX_LLM_CALLS_PER_MINUTE")
    
    # Reply Configuration
    max_reply_length: int = Field(2000, env="MAX_REPLY_LENGTH")  # Discord message limit
    reply_timeout_seconds: int = Field(30, env="REPLY_TIMEOUT_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


def validate_discord_intents(intents_str: str) -> list[str]:
    """Validate and parse Discord intents string."""
    valid_intents = {
        "guilds", "members", "bans", "emojis", "integrations", "webhooks",
        "invites", "voice_states", "presences", "messages", "guild_messages",
        "dm_messages", "reactions", "guild_reactions", "dm_reactions",
        "typing", "guild_typing", "dm_typing", "message_content"
    }
    
    intents = [intent.strip() for intent in intents_str.split(",")]
    invalid_intents = [intent for intent in intents if intent not in valid_intents]
    
    if invalid_intents:
        raise ValueError(f"Invalid Discord intents: {invalid_intents}")
    
    if "message_content" not in intents:
        raise ValueError("message_content intent is required for this bot to function")
    
    return intents


def validate_configuration():
    """Validate all configuration settings on startup."""
    settings = get_settings()
    
    # Validate Discord intents
    validate_discord_intents(settings.discord_intents)
    
    # Validate LLM provider
    if settings.llm_provider not in ["openai"]:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    
    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if settings.log_level.upper() not in valid_log_levels:
        raise ValueError(f"Invalid log level: {settings.log_level}")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    
    return settings


# Global settings instance
settings = None

def init_settings() -> Settings:
    """Initialize and validate settings."""
    global settings
    if settings is None:
        settings = validate_configuration()
    return settings