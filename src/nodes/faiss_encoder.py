import faiss
import numpy as np
from typing import List, Optional, Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from src.nodes.receiver import TextInput
from src.nodes.utils import EmbeddingStorage
import os


class FaissTextEncoder:
    """Clase para gestionar embeddings de texto y búsquedas con Faiss."""

    def __init__(self, 
                 model: Optional[OpenAIEmbeddings] = None,
                 faiss_index_path: str = "data/embeddings/faiss_index.bin"):
        """
        Inicializa el encoder con el modelo de embeddings y el índice Faiss.

        :param model: Instancia de OpenAIEmbeddings para generar embeddings.
        :param faiss_index_path: Ruta del archivo para guardar/cargar el índice Faiss.
        """
        self.model = model or OpenAIEmbeddings(model="text-embedding-3-large")
        self.faiss_index_path = faiss_index_path
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)

        # Intenta cargar el índice Faiss, o crea uno nuevo si no existe
        self.index = self._load_or_initialize_faiss_index()

    def _load_or_initialize_faiss_index(self):
        """Carga un índice Faiss desde disco o crea uno nuevo si no existe."""
        if os.path.exists(self.faiss_index_path):
            try:
                # Cargar índice Faiss desde archivo
                index = FAISS.load_local(self.faiss_index_path, self.model, allow_dangerous_deserialization=True)
                print(f"Índice Faiss cargado desde: {self.faiss_index_path}")
                return index
            except Exception as e:
                print(f"Error al cargar el índice: {e}")
                
        # Crear nuevo índice Faiss
        print("Creando nuevo índice Faiss...")
        dimension = len(self.model.embed_query("dummy"))
        faiss_index = faiss.IndexFlatL2(dimension)
        index = FAISS(
            embedding_function=self.model,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return index

    def save_faiss_index(self):
        """Guarda el índice Faiss en disco."""
        self.index.save_local(self.faiss_index_path)
        print(f"Índice Faiss guardado en: {self.faiss_index_path}")

    def encode_text(self, text: Union[TextInput, str]) -> np.ndarray:
        """Genera un embedding para un texto único."""
        if isinstance(text, TextInput):
            text = text.text
        return self.model.embed_query(text)

    def add_to_index(self, text: str, metadata: Optional[dict] = None):
        """
        Genera un embedding para un texto y lo añade al índice Faiss.

        :param text: Texto a procesar.
        :param metadata: Metadata asociada al texto (opcional).
        """
        self.index.add_texts([text], metadatas=[metadata])

    def search_similar_cases(self, text: str, k: int = 5):
        """
        Busca casos similares a un texto dado en el índice y devuelve sus metadatos.

        :param text: Texto a comparar.
        :param k: Número de casos similares a devolver.
        :return: Lista de metadatos de los casos similares.
        """
        results = self.index.similarity_search_with_score(text, k=k)
        return [{"metadata": result[0].metadata, "score": result[1]} for result in results]

    def encode_and_add(self, text: str, embedding_storage: EmbeddingStorage, metadata: Optional[dict] = None):
        """
        Genera el embedding para un texto y lo añade al índice Faiss, guardando el embedding en disco.

        :param text: Texto a procesar.
        :param embedding_storage: Instancia de almacenamiento para embeddings.
        :param metadata: Metadata asociada al texto (opcional).
        """
        self.add_to_index(text, metadata)

        # Guardar en disco
        if metadata:
            embedding = self.encode_text(text)
            embedding_storage.save_embedding(embedding, metadata)
