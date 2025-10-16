# Vector search implementation using SentenceTransformers
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import PodcastConstants


class VectorIndex:
    """Vector-based search index using SentenceTransformers embeddings"""

    def __init__(self, chunks: List[Dict[str, Any]], embedder: SentenceTransformer):
        """
        Initialize the vector index with document chunks and embedder.

        Args:
            chunks: List of document chunks to index
            embedder: SentenceTransformer model for generating embeddings
        """
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings = None
        self._build_index()

    def _build_index(self):
        """Build the vector index by generating embeddings for all chunks"""
        # Extract content from chunks for embedding
        texts = []
        for chunk in self.chunks:
            content = chunk.get("content", "")
            filename = chunk.get("filename", "")
            # Combine content and filename for better search
            combined_text = f"{filename}: {content}"
            texts.append(combined_text)

        # Generate embeddings
        self.embeddings = self.embedder.encode(texts)

    def search(
        self,
        query: str,
        num_results: int = PodcastConstants.DEFAULT_VECTOR_SEARCH_RESULTS.value,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity.

        Args:
            query: The search query
            num_results: Number of results to return

        Returns:
            List of relevant document chunks with similarity scores
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call _build_index() first.")

        # Generate query embedding
        query_embedding = self.embedder.encode([query])

        # Calculate similarities (cosine similarity)

        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:num_results]

        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["similarity_score"] = float(similarities[idx])
            results.append(chunk)

        return results


def create_vector_index(
    chunks: List[Dict[str, Any]], embedder: SentenceTransformer
) -> VectorIndex:
    """
    Create a vector-based search index from document chunks.

    Args:
        chunks: List of document chunks to index
        embedder: SentenceTransformer model for generating embeddings

    Returns:
        VectorIndex instance ready for searching
    """
    return VectorIndex(chunks, embedder)
