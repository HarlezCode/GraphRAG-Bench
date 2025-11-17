"""
Base retriever class providing common functionality for all retriever types.
Contains shared methods and utilities used across different retriever implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from Core.Common.Utils import truncate_list_by_token_size
from Core.Common.Logger import logger
from Core.Retriever.RetrieverFactory import get_retriever_operator


class BaseRetriever(ABC):
    """
    Abstract base class for all retriever implementations.
    
    Provides common functionality for retrieving relevant content from different
    data sources (entities, relationships, chunks, communities, etc.).
    """

    def __init__(self, config):
        """
        Initialize the base retriever with configuration.
        
        Args:
            config: Configuration object containing retriever settings
        """
        self.config = config

    @property
    @abstractmethod
    def mode_list(self) -> List[str]:
        """List of supported retrieval modes for this retriever type."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Type identifier for this retriever."""
        pass

    async def retrieve_relevant_content(self, **kwargs) -> Optional[Any]:
        """
        Find the relevant contexts for the given query.
        
        Args:
            **kwargs: Additional arguments including 'mode' for retrieval method
            
        Returns:
            Retrieved content or None if retrieval fails
        """
        mode = kwargs.pop("mode")
        if mode not in self.mode_list:
            logger.warning(f"Invalid mode: {mode} for retriever type: {self.type}")
            return None

        retrieve_fun = get_retriever_operator(self.type, mode)
        if retrieve_fun is None:
            logger.error(f"Retrieval method '{mode}' not found for retriever type: {self.type}")
            return None
            
        return await retrieve_fun(self, **kwargs)

    async def _construct_relationship_context(self, edge_datas: List[Dict]) -> List[Dict]:
        """
        Construct relationship context from edge data with ranking.
        
        Args:
            edge_datas: List of edge data dictionaries
            
        Returns:
            Processed and ranked edge data
        """
        if not all([n is not None for n in edge_datas]):
            logger.warning("Some edges are missing, storage may be damaged")
        
        # Process edge data to ensure they have required fields
        processed_edge_datas = []
        for edge_data in edge_datas:
            if edge_data is None:
                continue
                
            # Ensure edge data has src_id and tgt_id
            if "src_id" not in edge_data or "tgt_id" not in edge_data:
                # Create fallback edge data
                edge_data = {
                    "src_id": edge_data.get("src_id", "unknown"),
                    "tgt_id": edge_data.get("tgt_id", "unknown"),
                    "description": edge_data.get("description", edge_data.get("content", "Unknown edge")),
                    "content": edge_data.get("content", edge_data.get("description", "Unknown edge")),
                    "weight": edge_data.get("weight", 1.0),
                    "source_id": edge_data.get("source_id", "unknown"),
                    **edge_data
                }
            # Ensure source_id is always present
            if "source_id" not in edge_data or not edge_data["source_id"]:
                edge_data["source_id"] = edge_data.get("src_id", "unknown")
            processed_edge_datas.append(edge_data)
        
        # Calculate edge degrees
        edge_degree = await asyncio.gather(
            *[self.graph.edge_degree(r["src_id"], r["tgt_id"]) for r in processed_edge_datas],
            return_exceptions=True
        )
        
        # Process results, handling exceptions
        final_edge_datas = []
        for v, d in zip(processed_edge_datas, edge_degree):
            if isinstance(d, Exception):
                # If edge degree calculation failed, use default rank
                d = 1
            final_edge_datas.append({"src_id": v["src_id"], "tgt_id": v["tgt_id"], "rank": d, **v})
        
        # Sort by rank and weight
        final_edge_datas = sorted(
            final_edge_datas, key=lambda x: (x["rank"], x.get("weight", 1.0)), reverse=True
        )
        
        # Truncate by token size
        final_edge_datas = truncate_list_by_token_size(
            final_edge_datas,
            key=lambda x: x.get("description", ""),
            max_token_size=self.config.max_token_for_global_context,
        )
        return final_edge_datas

    async def _run_personalized_pagerank(self, query: str, query_entities: List[Dict]) -> np.ndarray:
        """
        Run Personalized PageRank algorithm for entity ranking.
        
        Args:
            query: Input query string
            query_entities: List of query entities
            
        Returns:
            Personalized PageRank scores for all nodes
        """
        reset_prob_matrix = np.zeros(self.graph.node_num)

        if self.config.use_entity_similarity_for_ppr:
            # Use entity similarity to compute the reset probability matrix
            # Implementation based on FastGraphRAG approach
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(
                query_entities, top_k=1, graph=self.graph
            )
            reset_prob_matrix += await self.entities_vdb.retrieval_nodes_with_score_matrix(
                query, top_k=self.config.top_k_entity_for_ppr, graph=self.graph
            )
        else:
            # Set weights based on document occurrence count
            # Implementation based on HippoRAG approach
            if not hasattr(self, "entity_chunk_count"):
                await self._initialize_entity_chunk_count()
                
            for entity in query_entities:
                entity_idx = await self.graph.get_node_index(entity["entity_name"])
                if self.config.node_specificity and self.entity_chunk_count is not None:
                    weight = 1.0 if self.entity_chunk_count[entity_idx] == 0 else 1.0 / float(self.entity_chunk_count[entity_idx])
                    reset_prob_matrix[entity_idx] = weight
                else:
                    reset_prob_matrix[entity_idx] = 1.0
                    
        return await self.graph.personalized_pagerank([reset_prob_matrix])

    async def _initialize_entity_chunk_count(self):
        """Initialize entity-chunk count matrix for PPR calculations."""
        if not hasattr(self, 'entities_to_relationships') or self.entities_to_relationships is None:
            logger.warning("entities_to_relationships not available, skipping entity-chunk count initialization")
            self.entity_chunk_count = None
            return
            
        if not hasattr(self, 'relationships_to_chunks') or self.relationships_to_chunks is None:
            logger.warning("relationships_to_chunks not available, skipping entity-chunk count initialization")
            self.entity_chunk_count = None
            return
            
        try:
            e2r = await self.entities_to_relationships.get()
            r2c = await self.relationships_to_chunks.get()
            
            if e2r is None or r2c is None:
                logger.warning("Entity-relationship matrices not available, skipping entity-chunk count initialization")
                self.entity_chunk_count = None
                return
                
            c2e = e2r.dot(r2c).T
            c2e[c2e.nonzero()] = 1
            self.entity_chunk_count = c2e.sum(0).T
        except Exception as e:
            logger.warning(f"Failed to initialize entity-chunk count: {e}")
            self.entity_chunk_count = None

    async def link_query_entities(self, query_entities: List[Dict]) -> List[Dict]:
        """
        Link query entities to graph entities using vector similarity.
        
        Args:
            query_entities: List of query entities to link
            
        Returns:
            List of linked entities
        """
        entities = []
        for query_entity in query_entities:
            node_datas = await self.entities_vdb.retrieval_nodes(
                query_entity, top_k=1, graph=self.graph
            )
            # For entity linking, only consider the top-ranked entity
            entities.append(node_datas[0])
        return entities

    async def _safe_async_gather(self, coroutines: List, error_msg: str = "Operation failed") -> List:
        """
        Safely execute multiple coroutines with error handling.
        
        Args:
            coroutines: List of coroutines to execute
            error_msg: Error message for logging
            
        Returns:
            List of results from coroutines
        """
        try:
            return await asyncio.gather(*coroutines)
        except Exception as e:
            logger.exception(f"{error_msg}: {e}")
            return []

    def _validate_input(self, data: Any, data_name: str) -> bool:
        """
        Validate input data and log warnings if invalid.
        
        Args:
            data: Data to validate
            data_name: Name of the data for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None:
            logger.warning(f"{data_name} is None")
            return False
        if isinstance(data, (list, tuple)) and len(data) == 0:
            logger.warning(f"{data_name} is empty")
            return False
        return True

class TraditionalRAGRetriever:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks
    
    async def retrieve(self, query: str, top_k: int = 5):
        indices = self.index.query(query, top_k=top_k)
        return [self.chunks[i] for i in indices]