"""
TF-IDF Store Module

This module provides a TF-IDF based index implementation using scikit-learn.
It offers a simple and efficient text-based retrieval method using
term frequency-inverse document frequency weighting.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama_index.legacy.data_structs.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
from llama_index.legacy.schema import BaseNode, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage.docstore.types import RefDocInfo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFIndex(BaseIndex[IndexDict]):
    """
    TF-IDF based index implementation using scikit-learn.
    
    This class provides a simple and efficient text-based retrieval method
    using term frequency-inverse document frequency weighting for document
    similarity calculations.
    """

    def __init__(self) -> None:
        """
        Initialize the TF-IDF index.
        
        Sets up the TF-IDF vectorizer for document processing and similarity search.
        """
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.95, ngram_range=(1,2), sublinear_tf=True)
        self.tfidf_matrix = None

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """
        Insert nodes into the TF-IDF index.
        
        Args:
            nodes: Sequence of nodes to insert
            **insert_kwargs: Additional insertion arguments
            
        Raises:
            NotImplementedError: TF-IDF index does not support insertion yet
        """
        raise NotImplementedError("TF-IDF index does not support insertion yet.")

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete a node from the TF-IDF index.
        
        Args:
            node_id: ID of the node to delete
            **delete_kwargs: Additional deletion arguments
            
        Raises:
            NotImplementedError: TF-IDF index does not support deletion yet
        """
        raise NotImplementedError("TF-IDF index does not support deletion yet.")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """
        Get a retriever instance for the TF-IDF index.
        
        Args:
            **kwargs: Additional retriever arguments
            
        Raises:
            NotImplementedError: TF-IDF index does not support retrieval yet
        """
        raise NotImplementedError("TF-IDF index does not support retrieval yet.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """
        Get reference document information.
        
        Raises:
            NotImplementedError: TF-IDF index does not support ref_doc_info
        """
        raise NotImplementedError("TF-IDF index does not support ref_doc_info.")

    def _build_index_from_nodes(self, nodes: List[BaseNode]) -> None:
        """
        Build index from a list of nodes.
        
        Args:
            nodes: List of nodes to build index from
            
        Raises:
            NotImplementedError: TF-IDF index does not support node-based building yet
        """
        raise NotImplementedError("TF-IDF index does not support node-based building yet.")

    def _build_index_from_list(self, docs_list: List[str]) -> None:
        """
        Build TF-IDF index from a list of documents.
        
        Args:
            docs_list: List of document strings to index
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(docs_list)

    def persist(self, persist_dir: str) -> None:
        """
        Persist the TF-IDF index to disk.
        
        Args:
            persist_dir: Directory to persist the index to
        """
        # Check if the destination directory exists and remove it
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        # Copy the index files to the destination
        shutil.copytree(
            Path(self.index_path) / self.index_name, 
            Path(persist_dir) / self.index_name
        )
        self._storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def load_from_disk(cls, persist_dir: str, index_name: str = "") -> 'TFIDFIndex':
        """
        Load TF-IDF index from disk.
        
        Args:
            persist_dir: Directory containing the persisted index
            index_name: Name of the index to load
            
        Raises:
            NotImplementedError: TF-IDF index does not support loading from disk yet
        """
        raise NotImplementedError("TF-IDF index does not support loading from disk yet.")

    def query(self, query_str: str, top_k: int = 10) -> List[int]:
        """
        Query the TF-IDF index for similar documents.
        
        Args:
            query_str: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            List of document indices sorted by similarity score
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call _build_index_from_list first.")
        
        # Transform the query using the fitted vectorizer
        query_emb = self.vectorizer.transform([query_str])
        
        # Calculate cosine similarity between query and all documents
        cosine_sim = cosine_similarity(query_emb, self.tfidf_matrix).flatten()

        # Get top-k most similar documents
        top_k = min(top_k, len(cosine_sim))
        idxs = cosine_sim.argsort()[::-1][:top_k]
        
        return idxs.tolist()
    
    def query_with_scores(self, query_str: str, top_k: int = 10) -> tuple[List[int], List[float]]:
        """
        Query the TF-IDF index and return both indices and scores.
        
        Args:
            query_str: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, scores)
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call _build_index_from_list first.")
        
        # Transform the query using the fitted vectorizer
        query_emb = self.vectorizer.transform([query_str])
        
        # Calculate cosine similarity between query and all documents
        cosine_sim = cosine_similarity(query_emb, self.tfidf_matrix).flatten()

        # Get top-k most similar documents
        top_k = min(top_k, len(cosine_sim))
        idxs = cosine_sim.argsort()[::-1][:top_k]
        scores = cosine_sim[idxs]
        
        return idxs.tolist(), scores.tolist()

    def query_batch(self, queries: List[str], top_k: int) -> List[List[int]]:
        """
        Query the TF-IDF index for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of top results to return per query
            
        Raises:
            NotImplementedError: TF-IDF index does not support batch querying yet
        """
        raise NotImplementedError("TF-IDF index does not support batch querying yet.")
        