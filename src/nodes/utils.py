from typing import List, Dict
import json
import numpy as np
from pathlib import Path

class EmbeddingStorage:
    """Utility class for storing and loading embeddings."""
    
    def __init__(self, storage_dir: str = 'data/embeddings'):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def save_embedding(self, embedding: np.ndarray, metadata: Dict, filename: str):
        """Save embedding and its metadata to disk."""
        embedding_path = self.storage_dir / f"{filename}.npy"
        metadata_path = self.storage_dir / f"{filename}.json"
        
        np.save(embedding_path, embedding)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
    def load_embedding(self, filename: str) -> tuple[np.ndarray, Dict]:
        """Load embedding and its metadata from disk."""
        embedding_path = self.storage_dir / f"{filename}.npy"
        metadata_path = self.storage_dir / f"{filename}.json"
        
        embedding = np.load(embedding_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return embedding, metadata

class FraudTypeRegistry:
    """Registry for managing known fraud types."""
    
    def __init__(self, storage_path: str = "data/fraud_types.json"):
        self.storage_path = storage_path
        self.known_types = self._load_types()
        
    def _load_types(self) -> List[str]:
        """Load known fraud types from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
            
    def save_types(self):
        """Save current fraud types to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.known_types, f)
            
    def add_type(self, fraud_type: str):
        """Add a new fraud type if not already known."""
        if fraud_type not in self.known_types:
            self.known_types.append(fraud_type)
            self.save_types()
            
    def get_types(self) -> List[str]:
        """Get list of all known fraud types."""
        return self.known_types.copy()