# signals/action_log.py
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime


class ActionLog(BaseModel):
    """
    Log entry for all actions taken by the system.
    
    Provides audit trail for system decisions and actions.
    """
    action_id: str
    signal_id: str
    action: str  # e.g., "discord.reply", "discord.react", etc.
    request: Dict[str, Any]  # The request parameters sent
    response: Optional[Dict[str, Any]] = None  # The response received
    error: Optional[str] = None  # Error message if action failed
    ts: datetime
    retry_count: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }