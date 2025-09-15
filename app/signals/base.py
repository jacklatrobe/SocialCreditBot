# signals/base.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class Signal(BaseModel):
    """
    Unified envelope for any message-like event.
    
    This is the core abstraction that normalizes inputs from different platforms
    (Discord, potentially others in the future) into a consistent format.
    """
    signal_id: str
    source: str
    created_at: datetime
    author: Dict[str, str]  # user_id, username
    context: Dict[str, Optional[str]]  # guild_id, channel_id, thread_id, message_id, reply_to_id
    content: str
    nano_tags: Dict[str, Any] = Field(default_factory=dict)
    observer: Dict[str, Any] = Field(default_factory=dict)
    orchestrator: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:
        """Generate idempotency key from source and message_id."""
        return f"{self.source}:{self.context.get('message_id', self.signal_id)}"

    async def fast_classify(self, nano_llm) -> Dict[str, Any]:
        """
        Use nano LLM to classify message with tags: purpose, sentiment, urgency, toxicity.
        
        Args:
            nano_llm: The nano LLM classifier instance
            
        Returns:
            Dict with classification tags
        """
        if self.nano_tags:
            return self.nano_tags
            
        # Truncate content for classification (nano models have token limits)
        content_for_classification = self.content[:500] if len(self.content) > 500 else self.content
        
        classification = await nano_llm.classify(content_for_classification)
        self.nano_tags = classification
        return classification

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }