"""
Simple Training Data Collection

Just dumps classification results to JSONL. No fancy stuff.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import aiofiles

from app.config import Settings
from app.signals.base import Signal
from app.llm.classification import EnhancedClassificationResult

logger = logging.getLogger(__name__)


class SimpleTrainingDataCollector:
    """Dead simple JSONL collector. No rotation, no cleanup, no BS."""
    
    def __init__(self, config: Settings):
        self.enabled = config.training_data_enabled
        self.file_path = Path(config.training_data_path)
        
        # Create directory if needed
        if self.enabled:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def collect(self, signal: Signal, result: EnhancedClassificationResult):
        """Dump classification to JSONL file."""
        if not self.enabled:
            return
            
        try:
            # Build the data record
            record = {
                "timestamp": datetime.now().isoformat(),
                "input": {
                    "content": signal.content,
                    "author": signal.author,
                    "context": signal.context,
                },
                "output": {
                    "purpose": result.purpose,
                    "sentiment": result.sentiment,
                    "urgency": result.urgency,
                    "toxicity": result.toxicity,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "message_type": result.message_type.value if result.message_type else None,
                    "reasoning": result.reasoning,
                    "processing_time": result.processing_time,
                    "model_used": result.model_used,
                },
                "features": {
                    "rule_matches": result.rule_matches,
                    "content_analysis": result.content_analysis,
                    "context_factors": result.context_factors,
                }
            }
            
            # Append to JSONL file
            json_line = json.dumps(record, ensure_ascii=False) + "\n"
            
            async with aiofiles.open(self.file_path, mode='a', encoding='utf-8') as f:
                await f.write(json_line)
                
        except Exception as e:
            logger.error(f"Failed to write training data: {e}")


# Global instance
_collector: Optional[SimpleTrainingDataCollector] = None

def get_collector(config: Optional[Settings] = None) -> Optional[SimpleTrainingDataCollector]:
    global _collector
    if _collector is None and config:
        _collector = SimpleTrainingDataCollector(config)
    return _collector