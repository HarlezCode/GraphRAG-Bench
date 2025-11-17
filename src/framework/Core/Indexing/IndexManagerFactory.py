"""
Index manager factory for managing different index building strategies.
Provides a unified interface for various indexing approaches using factory and strategy patterns.
"""

import asyncio
import numpy as np

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from Core.Common.Logger import logger
from Core.Schema.VdbResult import VdbResult
from Core.Utils.Display import StatusDisplay


class IndexType(Enum):
    """Supported index types."""
    VECTOR = "vector"
    FAISS = "faiss"
    COLBERT = "colbert"
    TFIDF = "tfidf"


@dataclass
class IndexConfig:
    """Configuration for index building."""
    index_type: IndexType = IndexType.VECTOR
    persist_path: str = ""
    embed_model: Any = None
    dimension: int = 1536
    similarity_metric: str = "cosine"
    top_k: int = 10
    force_rebuild: bool = False


class IndexManager(ABC):
    """
    Abstract base class for index managers.
    
    Provides a unified interface for different index building strategies
    with common functionality for loading, saving, and searching.
    """
    
    def __init__(self, config: IndexConfig, context: Any):
        """Initialize index manager with configuration."""
        self.config = config
        self.context = context
        self.index = None
        self.embed_model = config.embed_model
    
    @abstractmethod
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute index building process."""
        pass
    
    @abstractmethod
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build the index from data and metadata."""
        pass
    
    @abstractmethod
    async def _load_index(self) -> bool:
        """Load existing index from storage."""
        pass
    
    @abstractmethod
    async def _save_index(self) -> None:
        """Save index to storage."""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index with a query."""
        pass
    
    def get_index(self) -> Any:
        """Get the current index instance."""
        return self.index
    
    def exists(self) -> bool:
        """Check if index exists in storage."""
        import os
        return os.path.exists(self.config.persist_path)
    
    def get_entity_index(self) -> Any:
        """Get entity index for graph augmentation."""
        # Return the index itself for now
        # In practice, this might need to return a specific entity index
        return self.index


class VectorIndexManager(IndexManager):
    """Index manager for vector-based indexing."""
    
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute vector index building."""
        StatusDisplay.show_processing_status("Index building", details="Vector index")
        
        # Check if rebuild is needed
        if not force_rebuild and self.exists():
            StatusDisplay.show_info("Loading existing index")
            success = await self._load_index()
            if success:
                return self.index
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        await self._save_index()
        
        StatusDisplay.show_success(f"Vector index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build vector index."""
        from llama_index.core import VectorStoreIndex, Document
        
        # Create document objects
        documents = []
        for i, (content, meta) in enumerate(zip(data, metadata)):
            if isinstance(content, str):
                doc = Document(text=content, metadata=meta)
            else:
                doc = Document(text=str(content), metadata=meta)
            documents.append(doc)
        
        # Create vector index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embed_model
        )
        
        return index
    
    async def _load_index(self) -> bool:
        """Load index from storage."""
        try:
            from llama_index.core import load_index_from_storage
            from llama_index.core import StorageContext
            
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self.index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    async def _save_index(self) -> None:
        """Save index to storage."""
        if self.index and self.config.persist_path:
            import os
            # Create directory if it doesn't exist
            os.makedirs(self.config.persist_path, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.config.persist_path)
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index."""
        if not self.index:
            return []
        
        top_k = top_k or self.config.top_k
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=query)
        
        results = await retriever.aretrieve(query_bundle)
        
        # Convert to VdbResult format
        vdb_results = []
        for result in results:
            vdb_result = VdbResult(
                content=result.text,
                metadata=result.metadata,
                score=result.score if hasattr(result, 'score') else 0.0
            )
            vdb_results.append(vdb_result)
        
        return vdb_results


class FaissIndexManager(IndexManager):
    """Index manager for FAISS-based indexing."""
    
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute FAISS index building."""
        StatusDisplay.show_processing_status("Index building", details="FAISS index")
        
        # Check if rebuild is needed
        if not force_rebuild and self.exists():
            StatusDisplay.show_info("Loading existing index")
            success = await self._load_index()
            if success:
                return self.index
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        await self._save_index()
        
        StatusDisplay.show_success(f"FAISS index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build FAISS index."""
        import faiss
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core.schema import Document, TextNode
        from llama_index.core import Settings
        
        # Create document objects
        documents = []
        for i, (content, meta) in enumerate(zip(data, metadata)):
            if isinstance(content, str):
                doc = Document(text=content, metadata=meta)
            else:
                doc = Document(text=str(content), metadata=meta)
            documents.append(doc)
        
        # Handle case where embedding model is None (embeddings disabled)
        if self.embed_model is None:
            # Create a simple mock embedding model
            class MockEmbedding:
                def __init__(self, embed_dim=1024):
                    self.embed_dim = embed_dim
                
                def get_text_embedding(self, text):
                    import numpy as np
                    # Generate a simple random embedding
                    return np.random.rand(self.embed_dim).tolist()
            
            embed_model = MockEmbedding(embed_dim=1024)
            Settings.embed_model = embed_model
        else:
            logger.info("Using provided embedding model for FAISS index")
            Settings.embed_model = self.embed_model
        
        # Generate embeddings
        texts = [doc.text for doc in documents]
        if self.embed_model is None:
            # Use our simple mock embedding
            class MockEmbedding:
                def __init__(self, embed_dim=1024):
                    self.embed_dim = embed_dim
                
                def get_text_embedding(self, text):
                    import numpy as np
                    # Generate a simple random embedding
                    return np.random.rand(self.embed_dim).tolist()
            
            embed_model = MockEmbedding(embed_dim=1024)
            text_embeddings = [embed_model.get_text_embedding(text) for text in texts]
        else:
            logger.info("Generating text embeddings using the provided embedding model")
            text_embeddings = self.embed_model._get_text_embeddings(texts)
        
        # Create FAISS vector store with HNSW index
        logger.info("Creating FAISS vector store with HNSW index")
        vector_store = FaissVectorStore(faiss_index=faiss.IndexHNSWFlat(1024, 32))

        logger.info("Adding documents to FAISS vector store")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create vector index
        logger.info("Creating VectorStoreIndex with FAISS vector store")
        index = VectorStoreIndex(
            [], 
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        # Insert nodes with pre-computed embeddings
        logger.info("Inserting nodes with pre-computed embeddings into VectorStoreIndex")
        nodes = []
        for doc, embedding in tqdm(zip(documents, text_embeddings), total=len(documents), desc="Inserting nodes"):
            node = TextNode(text=doc.text, embedding=embedding, metadata=doc.metadata)
            nodes.append(node)
        index.insert_nodes(nodes)

        logger.info("FAISS index building completed")
        
        return index
    
    async def _load_index(self) -> bool:
        """Load index from storage."""
        try:
            from llama_index.core import load_index_from_storage
            from llama_index.core import StorageContext
            
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self.index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    async def _save_index(self) -> None:
        """Save index to storage."""
        if self.index and self.config.persist_path:
            import os
            # Create directory if it doesn't exist
            os.makedirs(self.config.persist_path, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.config.persist_path)
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index."""
        if not self.index:
            return []
        
        top_k = top_k or self.config.top_k
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=query)
        
        results = await retriever.aretrieve(query_bundle)
        
        # Convert to VdbResult format
        vdb_results = []
        for result in results:
            vdb_result = VdbResult(
                content=result.text,
                metadata=result.metadata,
                score=result.score if hasattr(result, 'score') else 0.0
            )
            vdb_results.append(vdb_result)
        
        return vdb_results


class ColBertIndexManager(IndexManager):
    """Index manager for ColBERT-based indexing."""
    
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute ColBERT index building."""
        StatusDisplay.show_processing_status("Index building", details="ColBERT index")
        
        # Check if rebuild is needed
        if not force_rebuild and self.exists():
            StatusDisplay.show_info("Loading existing index")
            success = await self._load_index()
            if success:
                return self.index
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        await self._save_index()
        
        StatusDisplay.show_success(f"ColBERT index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build ColBERT index."""
        # Implement ColBERT index building logic here
        # Can reference the original ColBertIndex implementation
        from Core.Index.ColBertIndex import ColBertIndex
        
        colbert_index = ColBertIndex(self.config)
        await colbert_index.build_index(data, metadata, force=True)
        
        return colbert_index
    
    async def _load_index(self) -> bool:
        """Load index from storage."""
        try:
            from Core.Index.ColBertIndex import ColBertIndex
            
            colbert_index = ColBertIndex(self.config)
            success = await colbert_index._load_index()
            if success:
                self.index = colbert_index
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    async def _save_index(self) -> None:
        """Save index to storage."""
        if self.index:
            self.index._storage_index()
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index."""
        if not self.index:
            return []
        
        top_k = top_k or self.config.top_k
        results = await self.index.retrieval(query, top_k)
        
        # Convert to VdbResult format
        # ColBERT returns (node_idxs, ranks, scores) tuple
        vdb_results = []
        if results and len(results) >= 3:
            node_idxs, ranks, scores = results
            for i, node_idx in enumerate(node_idxs):
                # For ColBERT, we need to get the actual content from the index
                # This is a simplified conversion - in practice, you might need to
                # get the actual content from the ColBERT collection
                vdb_result = VdbResult(
                    content=f"node_{node_idx}",  # Placeholder content
                    metadata={"node_idx": node_idx, "rank": ranks[i] if i < len(ranks) else 0},
                    score=scores[i] if i < len(scores) else 0.0
                )
                vdb_results.append(vdb_result)
        
        return vdb_results


class TFIDFIndexManager(IndexManager):
    """Index manager for TF-IDF based indexing."""
    
    def __init__(self, config: IndexConfig, context: Any):
        """Initialize TF-IDF index manager."""
        super().__init__(config, context)
        self.documents = []  # Store original documents
        self.metadata_list = []  # Store metadata
        self.text_chunks_index = None  # Expose index for TraditionalRAGRetriever
        self.text_chunks = []  # Expose chunks for TraditionalRAGRetriever
        
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute TF-IDF index building."""
        StatusDisplay.show_processing_status("Index building", details="TF-IDF index")
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        
        StatusDisplay.show_success(f"TF-IDF index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build TF-IDF index."""
        from Core.Index.TFIDFStore import TFIDFIndex
        
        # Store documents and metadata for retrieval
        self.documents = data
        self.metadata_list = metadata
        # Note: text_chunks will be set separately to TextChunk objects
        # Here we only store the string data for indexing
        
        # Create TF-IDF index
        index = TFIDFIndex()
        docs_list = [str(content) for content in data]
        index._build_index_from_list(docs_list)
        
        self.text_chunks_index = index  # Expose for TraditionalRAGRetriever
        return index
    
    async def _load_index(self) -> bool:
        """Load TF-IDF index from storage (not implemented)."""
        logger.warning("TF-IDF index loading from disk not implemented yet")
        return False
    
    async def _save_index(self) -> None:
        """Save TF-IDF index to storage (not implemented)."""
        logger.info("TF-IDF index saving to disk not implemented yet")
        pass
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the TF-IDF index (alias for retrieval)."""
        return await self.retrieval(query, top_k)
    
    async def retrieval(self, query: str, top_k: Optional[int] = None) -> List[VdbResult]:
        """Retrieve relevant documents using TF-IDF."""
        if self.index is None:
            logger.warning("Index not built yet")
            return []
        
        top_k = top_k or self.config.top_k
        indices, scores = self.index.query_with_scores(query, top_k=top_k)
        
        # Convert to VdbResult format
        vdb_results = []
        for i, idx in enumerate(indices):
            if idx < len(self.documents):
                vdb_result = VdbResult(
                    content=str(self.documents[idx]),
                    metadata=self.metadata_list[idx] if idx < len(self.metadata_list) else {},
                    score=scores[i] if i < len(scores) else 0.0
                )
                vdb_results.append(vdb_result)
        
        return vdb_results


class BM25IndexManager(IndexManager):
    """Index manager for BM25 based indexing."""
    
    def __init__(self, config: IndexConfig, context: Any):
        """Initialize BM25 index manager."""
        super().__init__(config, context)
        self.documents = []  # Store original documents
        self.metadata_list = []  # Store metadata
        self.text_chunks_index = None  # Expose index for TraditionalRAGRetriever
        self.text_chunks = []  # Expose chunks for TraditionalRAGRetriever
        
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute BM25 index building."""
        StatusDisplay.show_processing_status("Index building", details="BM25 index")
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        
        StatusDisplay.show_success(f"BM25 index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build BM25 index."""
        from Core.Index.BM25Store import BM25Index
        
        # Store documents and metadata for retrieval
        self.documents = data
        self.metadata_list = metadata
        # Note: text_chunks will be set separately to TextChunk objects
        # Here we only store the string data for indexing
        
        # Get BM25 parameters from config (hardcoded in retriever config)
        k1 = 1.5
        b = 0.75
        # Try to get from retriever config first (new location)
        if hasattr(self.config, 'retriever'):
            k1 = getattr(self.config.retriever, 'k1', 1.5)
            b = getattr(self.config.retriever, 'b', 0.75)
        # Fallback to old bm25 config location
        elif hasattr(self.config, 'bm25'):
            k1 = getattr(self.config.bm25, 'k1', 1.5)
            b = getattr(self.config.bm25, 'b', 0.75)
        
        # Create BM25 index
        index = BM25Index(k1=k1, b=b)
        docs_list = [str(content) for content in data]
        index._build_index_from_list(docs_list)
        
        self.text_chunks_index = index  # Expose for TraditionalRAGRetriever
        return index
    
    async def _load_index(self) -> bool:
        """Load BM25 index from storage (not implemented)."""
        logger.warning("BM25 index loading from disk not implemented yet")
        return False
    
    async def _save_index(self) -> None:
        """Save BM25 index to storage (not implemented)."""
        logger.info("BM25 index saving to disk not implemented yet")
        pass
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the BM25 index (alias for retrieval)."""
        return await self.retrieval(query, top_k)
    
    async def retrieval(self, query: str, top_k: Optional[int] = None) -> List[VdbResult]:
        """Retrieve relevant documents using BM25."""
        if self.index is None:
            logger.warning("Index not built yet")
            return []
        
        top_k = top_k or self.config.top_k
        indices, scores = self.index.query_with_scores(query, top_k=top_k)
        
        # Convert to VdbResult format
        vdb_results = []
        for i, idx in enumerate(indices):
            if idx < len(self.documents):
                vdb_result = VdbResult(
                    content=str(self.documents[idx]),
                    metadata=self.metadata_list[idx] if idx < len(self.metadata_list) else {},
                    score=scores[i] if i < len(scores) else 0.0
                )
                vdb_results.append(vdb_result)
        
        return vdb_results


class IndexManagerFactory:
    """
    Factory class for creating index managers.
    
    Provides a centralized way to instantiate different index building
    strategies based on configuration parameters.
    """
    
    _managers = {
        IndexType.VECTOR: VectorIndexManager,
        IndexType.FAISS: FaissIndexManager,
        IndexType.COLBERT: ColBertIndexManager,
        IndexType.TFIDF: TFIDFIndexManager,
        "bm25": BM25IndexManager  # BM25 not in IndexType enum yet, add as string
    }
    
    @classmethod
    def create_manager(cls, config: Any, context: Any) -> IndexManager:
        """
        Create an index manager based on configuration.
        
        Args:
            config: Configuration object containing index parameters
            context: Context object for the manager
            
        Returns:
            IndexManager: Instance of the appropriate index manager
        """
        # Create embedding model using embedding factory
        embed_model = None
        try:
            from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
            embedding_factory = RAGEmbeddingFactory()
            embed_model = embedding_factory.get_rag_embedding(config=config)
        except Exception as e:
            logger.warning(f"Failed to create embedding model: {e}")
            # If embedding creation fails, embed_model will remain None
            # and the index manager will use mock embeddings
        
        # Create persist path based on working directory and index type
        persist_path = ""
        if hasattr(config, 'working_dir') and config.working_dir:
            import os
            persist_path = os.path.join(config.working_dir, "index", config.vdb_type)
        
        # Handle BM25 separately (not in IndexType enum)
        if config.vdb_type == "bm25":
            # Create a simple config object for BM25 (no embedding needed)
            index_config = IndexConfig(
                index_type=None,  # BM25 doesn't use IndexType
                persist_path=persist_path,
                embed_model=None,  # BM25 doesn't need embeddings
                top_k=config.retriever.top_k if hasattr(config, 'retriever') and hasattr(config.retriever, 'top_k') else 10,
                force_rebuild=config.force_rebuild if hasattr(config, 'force_rebuild') else False
            )
            # Pass BM25 config if available
            if hasattr(config, 'bm25'):
                index_config.bm25 = config.bm25
            return BM25IndexManager(index_config, context)
        
        # Extract index configuration from config
        index_config = IndexConfig(
            index_type=IndexType(config.vdb_type),
            persist_path=persist_path,
            embed_model=embed_model,
            dimension=config.embedding.dimensions if hasattr(config, 'embedding') and hasattr(config.embedding, 'dimensions') else 1536,
            similarity_metric=config.similarity_metric if hasattr(config, 'similarity_metric') else "cosine",
            top_k=config.top_k if hasattr(config, 'top_k') else 10,
            force_rebuild=config.force_rebuild if hasattr(config, 'force_rebuild') else False
        )
        
        manager_class = cls._managers.get(index_config.index_type, VectorIndexManager)
        return manager_class(index_config, context)
    
    @classmethod
    def register_manager(cls, index_type: IndexType, manager_class: type):
        """
        Register a new index manager.
        
        Args:
            index_type: Index type to register
            manager_class: Class implementing the index manager
        """
        cls._managers[index_type] = manager_class
        logger.info(f"Registered new index manager: {index_type.value}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available index types."""
        return [index_type.value for index_type in cls._managers.keys()] 