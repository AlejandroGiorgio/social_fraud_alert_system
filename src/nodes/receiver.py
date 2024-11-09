from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TextInput:
    """Data class to store processed text input."""

    text: str
    source: str
    timestamp: datetime
    metadata: Optional[Dict] = None


class TextPreprocessor:
    """Class for preprocessing text inputs."""

    def __init__(self, min_length: int = 50, max_length: int = 5000):
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Clean raw text by removing special characters and extra whitespace."""
        # Remove URLs
        text = re.sub(r"http\S+|www.\S+", "", text)
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip()

    def is_valid_length(self, text: str) -> bool:
        """Check if text length is within acceptable range."""
        return self.min_length <= len(text) <= self.max_length

    def process_text(
        self, text: str, source: str, metadata: Optional[Dict] = None
    ) -> Optional[TextInput]:
        """Process raw text and return TextInput if valid."""
        cleaned_text = self.clean_text(text)

        if not self.is_valid_length(cleaned_text):
            return None

        return TextInput(
            text=cleaned_text,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata,
        )
