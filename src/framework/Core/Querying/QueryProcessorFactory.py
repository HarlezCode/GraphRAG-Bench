"""
Query processor factory for managing different query processing strategies.
Provides a unified interface for various query processing approaches using factory and strategy patterns.
"""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from Core.Common.Logger import logger
from Core.Schema.RetrieverContext import RetrieverContext
from Core.Utils.Display import StatusDisplay


class QueryType(Enum):
    """Supported query types."""
    BASIC = "basic"
    PPR = "ppr"
    KGP = "kgp"
    TOG = "tog"
    GR = "gr"
    MED = "med"
    DALK = "dalk"


@dataclass
class QueryConfig:
    """Configuration for query processing."""
    query_type: QueryType = QueryType.BASIC
    query_mode: str = "qa"  # qa or summary
    enable_hybrid_query: bool = True
    enable_local_search: bool = False
    enable_global_search: bool = True
    use_keywords: bool = True
    max_token_for_context: int = 4000
    response_type: str = "Multiple Paragraphs"


class QueryProcessor(ABC):
    """
    Abstract base class for query processors.
    
    Provides a unified interface for different query processing strategies
    with common functionality for entity extraction and keyword processing.
    """
    
    def __init__(self, config: QueryConfig, context: Any):
        """Initialize query processor with configuration."""
        self.config = config
        self.context = context
        # Note: retriever_context will be set later when _build_retriever_context is called
        self.retriever_context = None
        self.llm = context.llm
    
    @abstractmethod
    async def process_query(self, query: str) -> str:
        """Process a query and return response."""
        pass
    
    @abstractmethod
    async def _retrieve_relevant_contexts(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant contexts for the query."""
        pass
    
    @abstractmethod
    async def _generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate response based on query and context."""
        pass
    
    async def execute(self, query: str, *args, **kwargs) -> str:
        """Execute query processing (alias for process_query)."""
        return await self.process_query(query)
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        from Core.Prompt import GraphPrompt
        from Core.Common.Utils import clean_string, prase_json_from_response
        
        try:
            ner_messages = GraphPrompt.NER.format(user_input=query)
            response_content = await self.llm.aask(ner_messages)
            entities = prase_json_from_response(response_content)
            
            # Check if entities is a dictionary
            if not isinstance(entities, dict):
                logger.warning(f'Entity extraction returned non-dict: {type(entities)}')
                return []
            
            if 'named_entities' not in entities:
                entities = []
            else:
                entities = entities['named_entities']
            
            # Ensure entities is a list
            if not isinstance(entities, list):
                logger.warning(f'Named entities is not a list: {type(entities)}')
                return []
            
            entities = [clean_string(p) for p in entities]
            return entities
        except Exception as e:
            logger.error(f'Entity extraction error: {e}')
            return []
    
    async def _extract_query_keywords(self, query: str, mode: str = "low") -> str:
        """Extract keywords from the query."""
        from Core.Prompt import QueryPrompt
        from Core.Common.Utils import prase_json_from_response
        
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)
        keywords = None
        
        try:
            keywords_data = prase_json_from_response(result)
            
            # Check if keywords_data is a dictionary
            if not isinstance(keywords_data, dict):
                logger.warning(f'Keyword extraction returned non-dict: {type(keywords_data)}')
                return ""
            
            if mode == "low":
                keywords = keywords_data.get("low_level_keywords", [])
                keywords = ", ".join(keywords) if isinstance(keywords, list) else ""
            elif mode == "high":
                keywords = keywords_data.get("high_level_keywords", [])
                keywords = ", ".join(keywords) if isinstance(keywords, list) else ""
            elif mode == "hybrid":
                low_level = keywords_data.get("low_level_keywords", [])
                high_level = keywords_data.get("high_level_keywords", [])
                keywords = [low_level, high_level]
        except Exception as e:
            logger.error(f'Keyword extraction error: {e}')
            keywords = ""
        
        return keywords

newline = "\n"
class BasicQueryProcessor(QueryProcessor):
    """Basic query processor for standard retrieval and generation."""
    
    async def process_query(self, query: str) -> str:
        """Process basic query."""
        StatusDisplay.show_processing_status("Query processing", details="Basic query")
        
        # Retrieve relevant contexts
        context = await self._retrieve_relevant_contexts(query)
        
        # Generate response
        response = await self._generate_response(query, context)
        
        StatusDisplay.show_success("Query processing completed")
        return response
    
    async def _retrieve_relevant_contexts(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant contexts using multiple retrieval strategies."""
        from Core.Common.Constants import Retriever
        
        context = {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "communities": []
        }
        
        # Extract query entities
        entities = await self._extract_query_entities(query)
        
        # Entity retrieval
        if entities:
            entity_results = await self._retrieve_entities(query, entities)
            context["entities"] = entity_results
        
        # Relationship retrieval
        if self.config.enable_hybrid_query:
            relation_results = await self._retrieve_relationships(query)
            context["relationships"] = relation_results
        
        # Text chunk retrieval
        chunk_results = await self._retrieve_chunks(query)
        context["chunks"] = chunk_results
        
        # Community retrieval (if enabled)
        if self.config.enable_global_search:
            community_results = await self._retrieve_communities(query)
            context["communities"] = community_results
        
        return context
    
    async def _retrieve_entities(self, query: str, entities: List[str]) -> List[Dict]:
        """Retrieve entities using vector database."""
        from Core.Common.Constants import Retriever
        
        entity_results = await self.retriever_context.retrieve_relevant_content(
            type=Retriever.ENTITY,
            mode="vdb",
            query=query,
            top_k=10
        )
        
        return entity_results or []
    
    async def _retrieve_relationships(self, query: str) -> List[Dict]:
        """Retrieve relationships using vector database."""
        from Core.Common.Constants import Retriever
        
        # Only retrieve relationships if relations_vdb is enabled
        if hasattr(self.retriever_context, 'retrievers') and 'relationship' in self.retriever_context.retrievers:
            relationship_retriever = self.retriever_context.retrievers['relationship']
            if hasattr(relationship_retriever, 'relations_vdb') and relationship_retriever.relations_vdb is not None:
                relation_results = await self.retriever_context.retrieve_relevant_content(
                    type=Retriever.RELATION,
                    mode="vdb",
                    query=query,
                    top_k=10
                )
                return relation_results or []
        
        return []
    
    async def _retrieve_chunks(self, query: str) -> List[Dict]:
        """Retrieve text chunks using vector database."""
        from Core.Common.Constants import Retriever
        
        chunk_results = await self.retriever_context.retrieve_relevant_content(
            type=Retriever.CHUNK,
            mode="vdb",
            query=query,
            top_k=10
        )
        
        return chunk_results or []
    
    async def _retrieve_communities(self, query: str) -> List[Dict]:
        """Retrieve communities using community retrieval."""
        from Core.Common.Constants import Retriever
        
        community_results = await self.retriever_context.retrieve_relevant_content(
            type=Retriever.COMMUNITY,
            mode="entity",
            query=query,
            top_k=5
        )
        
        return community_results or []
    
    async def _generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate response based on query and context."""
        from Core.Prompt import QueryPrompt
        
        # Build context string
        context_str = self._build_context_string(context)
        
        # Create the full prompt with query and context
        if self.config.query_mode == "qa":
            full_prompt = f"You are an AI assistant that helps people find information.\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        else:  # summary
            full_prompt = f"You are an AI assistant that helps summarize information.\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nSummary:"
        
        # Generate response
        response = await self.llm.aask(full_prompt)
        return response
    
    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string from retrieved information."""
        context_parts = []
        
        # Add entity information
        if context.get("entities"):
            entities = context["entities"]
            # Check if entities is a coroutine object
            if hasattr(entities, '__await__'):
                logger.warning("Entities is a coroutine object, skipping entity information")
            else:
                entities_str = []
                for entity in entities[:5]:
                    if isinstance(entity, dict):
                        entity_name = entity.get('entity_name', '')
                        description = entity.get('description', '')
                        entities_str.append(f"- {entity_name}: {description}")
                    else:
                        entities_str.append(f"- {str(entity)}")
                entities_text = '\n'.join(entities_str)
                context_parts.append(f"Entity information:\n{entities_text}")
        
        # Add relationship information
        if context.get("relationships"):
            relationships = context["relationships"]
            # Check if relationships is a coroutine object
            if hasattr(relationships, '__await__'):
                logger.warning("Relationships is a coroutine object, skipping relationship information")
            else:
                relations_str = []
                for rel in relationships[:5]:
                    if isinstance(rel, dict):
                        src_id = rel.get('src_id', '')
                        relation_name = rel.get('relation_name', '')
                        tgt_id = rel.get('tgt_id', '')
                        relations_str.append(f"- {src_id} {relation_name} {tgt_id}")
                    else:
                        relations_str.append(f"- {str(rel)}")
                relations_text = '\n'.join(relations_str)
                context_parts.append(f"Relationship information:\n{relations_text}")
        
        # Add text chunk information
        if context.get("chunks"):
            chunks = context["chunks"]
            # Check if chunks is a coroutine object
            if hasattr(chunks, '__await__'):
                logger.warning("Chunks is a coroutine object, skipping chunk information")
            else:
                chunks_str = []
                for chunk in chunks[:3]:
                    if isinstance(chunk, dict):
                        content = chunk.get('content', '')
                        chunks_str.append(f"- {content[:200]}...")
                    else:
                        chunks_str.append(f"- {str(chunk)[:200]}...")
                chunks_text = '\n'.join(chunks_str)
                context_parts.append(f"Text chunk information:\n{chunks_text}")
        
        # Add community information
        if context.get("communities"):
            communities = context["communities"]
            # Check if communities is a coroutine object
            if hasattr(communities, '__await__'):
                logger.warning("Communities is a coroutine object, skipping community information")
            else:
                communities_str = []
                for comm in communities[:2]:
                    if isinstance(comm, dict):
                        report_string = comm.get('report_string', '')
                        communities_str.append(f"- {report_string[:200]}...")
                    else:
                        communities_str.append(f"- {str(comm)[:200]}...")
                communities_text = '\n'.join(communities_str)
                context_parts.append(f"Community information:\n{communities_text}")
        
        return "\n\n".join(context_parts)


class PPRQueryProcessor(QueryProcessor):
    """PPR (Personalized PageRank) query processor."""
    
    async def process_query(self, query: str) -> str:
        """Process PPR query."""
        StatusDisplay.show_processing_status("Query processing", details="PPR query")
        
        # Retrieve relevant contexts
        context = await self._retrieve_relevant_contexts(query)
        
        # Generate response
        response = await self._generate_response(query, context)
        
        StatusDisplay.show_success("PPR query processing completed")
        return response
    
    async def _retrieve_relevant_contexts(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant contexts using PPR method."""
        from Core.Common.Constants import Retriever
        
        context = {
            "entities": [],
            "relationships": [],
            "chunks": []
        }
        
        # Extract query entities
        entities = await self._extract_query_entities(query)
        
        # Use PPR to retrieve entities
        if entities:
            # Convert string entities to dict format expected by PPR
            seed_entities = [{"entity_name": entity} for entity in entities]
            entity_results = await self.retriever_context.retrieve_relevant_content(
                type=Retriever.ENTITY,
                mode="ppr",
                query=query,
                seed_entities=seed_entities,
                top_k=10
            )
            context["entities"] = entity_results or []
        
        # Use PPR to retrieve relationships (only if relations_vdb is enabled)
        relation_results = []
        if hasattr(self.retriever_context, 'retrievers') and 'relationship' in self.retriever_context.retrievers:
            relationship_retriever = self.retriever_context.retrievers['relationship']
            if hasattr(relationship_retriever, 'relations_vdb') and relationship_retriever.relations_vdb is not None:
                relation_results = await self.retriever_context.retrieve_relevant_content(
                    type=Retriever.RELATION,
                    mode="vdb",
                    seed=query,
                    top_k=10
                )
        context["relationships"] = relation_results or []
        
        # Retrieve relevant text chunks
        chunk_results = await self.retriever_context.retrieve_relevant_content(
            type=Retriever.CHUNK,
            mode="entity_occurrence",
            node_datas=context["entities"]
        )
        context["chunks"] = chunk_results or []
        
        return context
    
    async def _generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate response using PPR-specific prompt."""
        from Core.Prompt import QueryPrompt
        
        # Build context string
        context_str = self._build_context_string(context)
        
        # Create the full prompt with query and context
        full_prompt = f"{QueryPrompt.COT_SYSTEM_DOC}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nThought:"
        
        # Generate response
        response = await self.llm.aask(full_prompt)
        return response
    
    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string with PPR-ranked information."""
        context_parts = []
        
        # Add PPR-ranked entity information
        if context.get("entities"):
            entities = context["entities"]
            # Check if entities is a coroutine object
            if hasattr(entities, '__await__'):
                logger.warning("Entities is a coroutine object, skipping entity information")
            else:
                entities_str = []
                for entity in entities[:5]:
                    if isinstance(entity, dict):
                        entity_name = entity.get('entity_name', '')
                        ppr_score = entity.get('ppr_score', 0)
                        entities_str.append(f"- {entity_name} (PPR score: {ppr_score:.3f})")
                    else:
                        entities_str.append(f"- {str(entity)}")
                entities_text = '\n'.join(entities_str)
                context_parts.append(f"Entity information:\n{entities_text}")
        
        # Add relationship information
        if context.get("relationships"):
            relationships = context["relationships"]
            # Check if relationships is a coroutine object
            if hasattr(relationships, '__await__'):
                logger.warning("Relationships is a coroutine object, skipping relationship information")
            else:
                relations_str = []
                for rel in relationships[:5]:
                    if isinstance(rel, dict):
                        src_id = rel.get('src_id', '')
                        relation_name = rel.get('relation_name', '')
                        tgt_id = rel.get('tgt_id', '')
                        relations_str.append(f"- {src_id} {relation_name} {tgt_id}")
                    else:
                        relations_str.append(f"- {str(rel)}")
                relations_text = '\n'.join(relations_str)
                context_parts.append(f"Relationship information:\n{relations_text}")
        
        # Add text chunk information
        if context.get("chunks"):
            chunks = context["chunks"]
            # Check if chunks is a coroutine object
            if hasattr(chunks, '__await__'):
                logger.warning("Chunks is a coroutine object, skipping chunk information")
            else:
                chunks_str = []
                for chunk in chunks[:3]:
                    if isinstance(chunk, dict):
                        content = chunk.get('content', '')
                        chunks_str.append(f"- {content[:200]}...")
                    else:
                        chunks_str.append(f"- {str(chunk)[:200]}...")
                chunks_text = '\n'.join(chunks_str)
                context_parts.append(f"Text chunk information:\n{chunks_text}")
        
        return "\n\n".join(context_parts)


class ToGQueryProcessor(QueryProcessor):
    """ToG (Tree of Thoughts on Graph) query processor."""
    
    async def process_query(self, query: str) -> str:
        """Process ToG query."""
        StatusDisplay.show_processing_status("Query processing", details="ToG query")
        
        # Retrieve relevant contexts
        context = await self._retrieve_relevant_contexts(query)
        
        # Generate response
        response = await self._generate_response(query, context)
        
        StatusDisplay.show_success("ToG query processing completed")
        return response
    
    async def _retrieve_relevant_contexts(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant contexts using ToG method."""
        from Core.Common.Constants import Retriever
        
        context = {
            "entities": [],
            "paths": [],
            "subgraphs": []
        }
        
        # Extract query entities
        entities = await self._extract_query_entities(query)
        
        # Use vector database to retrieve entities
        if entities:
            entity_results = await self.retriever_context.retrieve_relevant_content(
                type=Retriever.ENTITY,
                mode="vdb",
                seed=query,
                top_k=10
            )
            context["entities"] = entity_results or []
        
        # Use paths retrieval
        path_results = await self.retriever_context.retrieve_relevant_content(
            type=Retriever.SUBGRAPH,
            mode="paths_return_list",
            seed=entities if entities else [],
            cutoff=2
        )
        context["paths"] = path_results or []
        
        # Retrieve subgraphs
        subgraph_results = await self.retriever_context.retrieve_relevant_content(
            type=Retriever.SUBGRAPH,
            mode="k_hop_return_set",
            query=query,
            k=2
        )
        context["subgraphs"] = subgraph_results or []
        
        return context
    
    async def _generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate response using ToG-specific prompt."""
        from Core.Prompt import QueryPrompt
        
        # Build context string
        context_str = self._build_context_string(context)
        
        # Create the full prompt with query and context
        full_prompt = f"You are an AI assistant that helps people find information based on knowledge graph data.\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        response = await self.llm.aask(full_prompt)
        return response
    
    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string with ToG-specific information."""
        context_parts = []
        
        # Add entity information
        if context.get("entities"):
            entities = context["entities"]
            # Check if entities is a coroutine object
            if hasattr(entities, '__await__'):
                logger.warning("Entities is a coroutine object, skipping entity information")
            else:
                entities_str = "\n".join([f"- {entity.get('entity_name', '')}: {entity.get('description', '')}" 
                                        for entity in entities[:5]])
                context_parts.append(f"Relevant entities:\n{entities_str}")
        
        # Add path information
        if context.get("paths"):
            paths = context["paths"]
            # Check if paths is a coroutine object
            if hasattr(paths, '__await__'):
                logger.warning("Paths is a coroutine object, skipping path information")
            else:
                paths_str = []
                for path in paths[:3]:
                    try:
                        if isinstance(path, list):
                            path_str = " -> ".join([str(item) for item in path])
                        else:
                            path_str = str(path)
                        paths_str.append(f"- Path: {path_str}")
                    except Exception as e:
                        logger.warning(f"Error processing path {path}: {e}")
                        paths_str.append(f"- Path: {str(path)}")
                paths_text = '\n'.join(paths_str)
                context_parts.append(f"Relevant paths:\n{paths_text}")
        
        # Add subgraph information
        if context.get("subgraphs"):
            subgraphs = context["subgraphs"]
            # Check if subgraphs is a coroutine object
            if hasattr(subgraphs, '__await__'):
                logger.warning("Subgraphs is a coroutine object, skipping subgraph information")
            else:
                subgraphs_str = "\n".join([f"- Subgraph: {subgraph}" 
                                         for subgraph in subgraphs[:2]])
                context_parts.append(f"Relevant subgraphs:\n{subgraphs_str}")
        
        return "\n\n".join(context_parts)


class QueryProcessorFactory:
    """
    Factory class for creating query processors.
    
    Provides a centralized way to instantiate different query processing
    strategies based on configuration parameters.
    """
    
    _processors = {
        QueryType.BASIC: BasicQueryProcessor,
        QueryType.PPR: PPRQueryProcessor,
        QueryType.TOG: ToGQueryProcessor,
        QueryType.KGP: BasicQueryProcessor,  # Can create specialized KGP processor
        QueryType.GR: BasicQueryProcessor,   # Can create specialized GR processor
        QueryType.MED: BasicQueryProcessor,  # Can create specialized MED processor
        QueryType.DALK: BasicQueryProcessor  # Can create specialized DALK processor
    }
    
    @classmethod
    def create_processor(cls, config: Any, context: Any) -> QueryProcessor:
        """
        Create a query processor based on configuration.
        
        Args:
            config: Configuration object containing query parameters
            context: Context object for the processor
            
        Returns:
            QueryProcessor: Instance of the appropriate query processor
        """
        # Extract query configuration from config
        query_config = QueryConfig(
            query_type=QueryType(config.retriever.query_type),
            query_mode=config.query.query_type,
            enable_hybrid_query=config.query.enable_hybrid_query,
            enable_local_search=config.query.enable_local,
            enable_global_search=config.query.use_global_query,
            use_keywords=config.query.use_keywords,
            max_token_for_context=config.query.max_token_for_text_unit,
            response_type=config.query.response_type
        )
        
        processor_class = cls._processors.get(query_config.query_type, BasicQueryProcessor)
        return processor_class(query_config, context)
    
    @classmethod
    def register_processor(cls, query_type: QueryType, processor_class: type):
        """
        Register a new query processor.
        
        Args:
            query_type: Query type to register
            processor_class: Class implementing the query processor
        """
        cls._processors[query_type] = processor_class
        logger.info(f"Registered new query processor: {query_type.value}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available query types."""
        return [query_type.value for query_type in cls._processors.keys()] 
