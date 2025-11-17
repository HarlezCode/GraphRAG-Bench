"""
BM25 Store Module

This module provides a BM25 based index implementation using rank_bm25.
BM25 is a traditional probabilistic retrieval function commonly used as a baseline.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Sequence
from llama_index.legacy.data_structs.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
from llama_index.legacy.schema import BaseNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage.docstore.types import RefDocInfo

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank_bm25: pip install rank-bm25")


class BM25Index(BaseIndex[IndexDict]):
    """
    BM25 based index implementation using rank_bm25.
    
    BM25 (Best Matching 25) is a probabilistic retrieval function
    that ranks documents based on query term frequency with
    saturation and length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Initialize the BM25 index.
        
        Args:
            k1: Controls term frequency saturation (typical: 1.2-2.0)
            b: Controls document length normalization (typical: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_corpus = None
        self.corpus_texts = []

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes into the BM25 index."""
        raise NotImplementedError("BM25 index does not support insertion yet.")

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node from the BM25 index."""
        raise NotImplementedError("BM25 index does not support deletion yet.")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Get a retriever instance for the BM25 index."""
        raise NotImplementedError("BM25 index does not support retrieval yet.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Get reference document information."""
        raise NotImplementedError("BM25 index does not support ref_doc_info.")

    def _build_index_from_nodes(self, nodes: List[BaseNode]) -> None:
        """Build index from a list of nodes."""
        raise NotImplementedError("BM25 index does not support node-based building yet.")

    def _build_index_from_list(self, docs_list: List[str]) -> None:
        """
        Build BM25 index from a list of documents.
        
        Args:
            docs_list: List of document strings to index
        """
        self.corpus_texts = docs_list
        # Simple tokenization (split by whitespace)
        self.tokenized_corpus = [doc.lower().split() for doc in docs_list]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def query(self, query_str: str, top_k: int = 10) -> List[int]:
        """
        Query the BM25 index for similar documents.
        
        Args:
            query_str: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            List of document indices sorted by BM25 score
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call _build_index_from_list first.")
        
        # Tokenize query
        tokenized_query = query_str.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return top_indices.tolist()

    def query_with_scores(self, query_str: str, top_k: int = 10) -> tuple[List[int], List[float]]:
        """
        Query the BM25 index and return both indices and scores.
        
        Args:
            query_str: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, scores)
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call _build_index_from_list first.")
        
        tokenized_query = query_str.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()
