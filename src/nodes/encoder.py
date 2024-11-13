from typing import List, Union, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from src.nodes.receiver import TextInput

class TextEncoder:
    """Class for encoding text into embeddings using SBERT."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the encoder with specified SBERT model."""
        self.model = SentenceTransformer(model_name)
        
    def encode_text(self, text: Union[TextInput, str]) -> np.ndarray:
        """Generate embedding for a single text."""
        if isinstance(text, TextInput):
            text = text.text
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[Union[TextInput, str]]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        if texts and isinstance(texts[0], TextInput):
            texts = [t.text for t in texts]
        return self.model.encode(texts, convert_to_numpy=True)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))