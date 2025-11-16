"""
GraphRAG Engine - Refactored Main Controller
Using Strategy Pattern and Composition Pattern for flexible architecture
"""
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
import tiktoken
from pydantic import BaseModel, Field, model_validator

import networkx as nx

from Core.Common.Logger import logger
from Core.Common.TimeStatistic import TimeStatistic
from Core.Schema.RetrieverContext import RetrieverContext
from Core.Storage.NameSpace import Workspace

from Core.Utils.ComponentRegistry import ComponentRegistry


@dataclass
class ProcessingPipeline:
    """Processing pipeline configuration"""
    chunking_enabled: bool = True
    graph_building_enabled: bool = True
    indexing_enabled: bool = True
    augmentation_enabled: bool = False
    community_detection_enabled: bool = False


@dataclass
class ExecutionContext:
    """Execution context"""
    workspace: Workspace
    time_manager: TimeStatistic
    retriever_context: RetrieverContext
    encoder: Any
    llm: Any = None
    pipeline: ProcessingPipeline = field(default_factory=ProcessingPipeline)


class GraphRAGEngine(BaseModel):
    """
    Refactored GraphRAG Engine
    Using Strategy Pattern and Composition Pattern for flexible architecture
    """
    
    # Core components
    config: Any = Field(description="Configuration object")
    
    # Processing stages
    processing_stages: Dict[str, Any] = Field(default_factory=dict)
    
    # System components (will be initialized in validator)
    encoder: Optional[Any] = Field(default=None, description="Tiktoken encoder")
    workspace: Optional[Any] = Field(default=None, description="Workspace instance")
    time_manager: Optional[Any] = Field(default=None, description="Time manager")
    retriever_context: Optional[Any] = Field(default=None, description="Retriever context")
    registry: Optional[Any] = Field(default=None, description="Component registry")
    context: Optional[Any] = Field(default=None, description="Execution context")
    
    class Config:
        arbitrary_types_allowed = True
    
    @model_validator(mode="after")
    def initialize_system_components(self):
        """Initialize all system components after model validation."""
        # Initialize encoder
        self.encoder = tiktoken.encoding_for_model(self.config.token_model)
        
        # Initialize workspace
        self.workspace = Workspace(
            self.config.working_dir, 
            self.config.index_name
        )
        
        # Initialize time manager
        self.time_manager = TimeStatistic()
        
        # Initialize retriever context
        self.retriever_context = RetrieverContext()
        
        # Initialize component registry
        self.registry = ComponentRegistry()
        
        # Initialize LLM (lazy loading)
        self._llm = None
        
        # Create execution context
        self.context = ExecutionContext(
            workspace=self.workspace,
            time_manager=self.time_manager,
            retriever_context=self.retriever_context,
            encoder=self.encoder,
            llm=self.llm  # Pass the lazy-loaded LLM
        )
        
        # Register core components
        self._register_core_components()
        
        # Build processing pipeline
        self._build_processing_pipeline()
        
        return self
    
    @property
    def llm(self):
        """Get LLM instance with lazy initialization."""
        if self._llm is None:
            from Core.Provider.LLMProviderRegister import create_llm_instance
            self._llm = create_llm_instance(self.config.llm)
        return self._llm
    
    def _register_core_components(self):
        """Register core components"""
        # Register graph builder
        self.registry.register_component(
            "graph_builder", 
            self._create_graph_builder()
        )
        
        # Register document processor
        self.registry.register_component(
            "document_processor", 
            self._create_document_processor()
        )
        
        # Register index manager
        self.registry.register_component(
            "index_manager", 
            self._create_index_manager()
        )
        
        # Register query processor
        self.registry.register_component(
            "query_processor", 
            self._create_query_processor()
        )
        
        # Register storage manager
        self.registry.register_component(
            "storage_manager", 
            self._create_storage_manager()
        )
    
    def _create_graph_builder(self):
        """Create graph builder"""
        from Core.Graph.GraphBuilderFactory import GraphBuilderFactory
        return GraphBuilderFactory.create_builder(self.config, self.context)
    
    def _create_document_processor(self):
        """Create document processor"""
        from Core.Processing.DocumentProcessorFactory import DocumentProcessorFactory
        return DocumentProcessorFactory.create_processor(self.config, self.context)
    
    def _create_index_manager(self):
        """Create index manager"""
        from Core.Indexing.IndexManagerFactory import IndexManagerFactory
        return IndexManagerFactory.create_manager(self.config, self.context)
    
    def _create_query_processor(self):
        """Create query processor"""
        from Core.Querying.QueryProcessorFactory import QueryProcessorFactory
        return QueryProcessorFactory.create_processor(self.config, self.context)
    
    def _create_storage_manager(self):
        """Create storage manager"""
        from Core.Storage.StorageManagerFactory import StorageManagerFactory
        return StorageManagerFactory.create_manager(self.config, self.context)
    
    def _build_processing_pipeline(self):
        """Build processing pipeline"""
        self.processing_stages = {
            "document_processing": self.registry.get_component("document_processor"),
            "graph_construction": self.registry.get_component("graph_builder"),
            "index_building": self.registry.get_component("index_manager"),
            "query_processing": self.registry.get_component("query_processor")
        }
    
    async def process_documents(self, documents: Union[str, List[Any]], force_rebuild: bool = False):
        """
        Main entry point for document processing
        
        Args:
            documents: Documents to process
            force_rebuild: Whether to force rebuild
        """
        logger.info("🚀 Starting document processing workflow")
        
        try:
            # Stage 1: Document processing
            if self.context.pipeline.chunking_enabled:
                await self._execute_stage("document_processing", documents)
            
            # Stage 2: Graph building
            if self.context.pipeline.graph_building_enabled:
                chunks = await self.registry.get_component("document_processor").get_processed_chunks()
                await self._execute_stage("graph_construction", chunks, force_rebuild)
            
            # Stage 3: Index building
            if self.context.pipeline.indexing_enabled:
                # Get data from graph builder for indexing
                graph_builder = self.registry.get_component("graph_builder")
                graph = graph_builder.get_graph()
                
                
                # Extract data and metadata from graph for indexing
                data = []
                metadata = []
                
                # Get nodes data from networkx graph
                for node_id, node_data in graph.nodes(data=True):
                    # Use description or entity_name as content for indexing
                    content = node_data.get('description', '') or node_data.get('entity_name', '')
                    data.append(content)
                    metadata.append({
                        'node_id': node_id,
                        'entity_name': node_data.get('entity_name', ''),
                        'entity_type': node_data.get('entity_type', ''),
                        'source_id': node_data.get('source_id', '')
                    })
                
                await self._execute_stage("index_building", data, metadata, force_rebuild)
            
            # Stage 4: Graph augmentation (optional)
            if self.context.pipeline.augmentation_enabled:
                await self._execute_graph_augmentation()
            
            # Stage 5: Community detection (optional)
            if self.context.pipeline.community_detection_enabled:
                await self._execute_community_detection()
            
            # Build retrieval context
            await self._build_retriever_context()
            
            logger.info("✅ Document processing workflow completed")
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise
    
    async def _execute_stage(self, stage_name: str, *args, **kwargs):
        """Execute processing stage"""
        stage = self.processing_stages.get(stage_name)
        if stage:
            self.context.time_manager.start_stage()
            await stage.execute(*args, **kwargs)
            self.context.time_manager.end_stage(stage_name)
        else:
            logger.warning(f"Processing stage not found: {stage_name}")
    
    async def _execute_graph_augmentation(self):
        """Execute graph augmentation"""
        graph_builder = self.registry.get_component("graph_builder")
        index_manager = self.registry.get_component("index_manager")
        
        if hasattr(graph_builder, 'augment_graph'):
            await graph_builder.augment_graph(index_manager.get_entity_index())
    
    async def _execute_community_detection(self):
        """Execute community detection"""
        graph_builder = self.registry.get_component("graph_builder")
        if hasattr(graph_builder, 'detect_communities'):
            await graph_builder.detect_communities()
    
    async def _build_retriever_context(self):
        """Build retrieval context"""
        logger.info("🔧 Building retrieval context")
        
        # Create MixRetriever instance
        from Core.Retriever.MixRetriever import MixRetriever
        
        # Build context for retriever
        retriever_context = RetrieverContext()
        
        # Get the networkx graph and create a wrapper with required methods
        nx_graph = self.registry.get_component("graph_builder").get_graph()
        
        # Create a wrapper class to provide the required interface
        class GraphWrapper:
            def __init__(self, nx_graph):
                self.graph = nx_graph
            
            @property
            def node_num(self):
                return self.graph.number_of_nodes()
            
            @property
            def edge_num(self):
                return self.graph.number_of_edges()
            
            def get_node_num(self):
                return self.graph.number_of_nodes()
            
            def get_edge_num(self):
                return self.graph.number_of_edges()
            
            async def get_node(self, node_id):
                if node_id in self.graph.nodes:
                    node_data = self.graph.nodes[node_id]
                    # Return node data with node_id information
                    # Ensure source_id is always present
                    result = {
                        'entity_name': node_id,
                        **node_data
                    }
                    # If source_id is missing, use entity_name as fallback
                    if 'source_id' not in result or not result['source_id']:
                        result['source_id'] = node_id
                    return result
                return None
            
            def get_node_index(self, node_id):
                nodes = list(self.graph.nodes())
                return nodes.index(node_id) if node_id in nodes else -1
            
            async def get_node_index(self, node_id):
                nodes = list(self.graph.nodes())
                return nodes.index(node_id) if node_id in nodes else -1
            
            def get_node_by_index(self, index):
                nodes = list(self.graph.nodes())
                if 0 <= index < len(nodes):
                    node_id = nodes[index]
                    if node_id in self.graph.nodes:
                        node_data = self.graph.nodes[node_id]
                        # Ensure source_id is always present
                        result = {
                            'entity_name': node_id,
                            **node_data
                        }
                        # If source_id is missing, use entity_name as fallback
                        if 'source_id' not in result or not result['source_id']:
                            result['source_id'] = node_id
                        return result
                return None
            
            async def get_node_by_indices(self, indices):
                nodes = list(self.graph.nodes())
                node_ids = [nodes[i] for i in indices if 0 <= i < len(nodes)]
                # Return node data instead of just node IDs
                result = []
                for node_id in node_ids:
                    if node_id in self.graph.nodes:
                        node_data = self.graph.nodes[node_id]
                        # Ensure source_id is always present
                        node_result = {
                            'entity_name': node_id,
                            **node_data
                        }
                        # If source_id is missing, use entity_name as fallback
                        if 'source_id' not in node_result or not node_result['source_id']:
                            node_result['source_id'] = node_id
                        result.append(node_result)
                return result
            
            def get_edge(self, src_id, tgt_id):
                if self.graph.has_edge(src_id, tgt_id):
                    edge_data = self.graph.edges[src_id, tgt_id]
                    # Return edge data with source and target information
                    return {
                        'src_id': src_id,
                        'tgt_id': tgt_id,
                        **edge_data
                    }
                return None
            
            def has_edge(self, src_id, tgt_id):
                return self.graph.has_edge(src_id, tgt_id)
            
            def neighbors(self, node_id):
                return list(self.graph.neighbors(node_id))
            
            async def get_neighbors(self, node_id):
                return list(self.graph.neighbors(node_id))
            
            async def find_k_hop_neighbors_batch(self, start_nodes, k):
                """Find k-hop neighbors for a batch of nodes."""
                # This is a simplified implementation
                # In a real implementation, you would implement proper k-hop neighbor finding
                all_neighbors = set()
                for node in start_nodes:
                    neighbors = list(self.graph.neighbors(node))
                    all_neighbors.update(neighbors)
                return list(all_neighbors)
            
            async def get_paths_from_sources(self, start_nodes, cutoff=5):
                """Get paths from source nodes."""
                # This is a simplified implementation
                # In a real implementation, you would implement proper path finding
                paths = []
                for node in start_nodes:
                    neighbors = list(self.graph.neighbors(node))
                    for neighbor in neighbors:
                        paths.append([node, neighbor])
                return paths
            
            async def get_neighbors_from_sources(self, start_nodes):
                """Get neighbors from source nodes."""
                all_neighbors = []
                for node in start_nodes:
                    neighbors = list(self.graph.neighbors(node))
                    all_neighbors.extend(neighbors)
                return all_neighbors
            
            def get_induced_subgraph(self, nodes):
                """Get induced subgraph for given nodes."""
                # This is a simplified implementation
                # In a real implementation, you would create a proper subgraph
                return self.graph.subgraph(nodes)
            
            def nodes(self):
                return list(self.graph.nodes())
            
            def edges(self):
                return list(self.graph.edges())
            
            async def edges_data(self):
                """Get edge data."""
                edges = list(self.graph.edges(data=True))
                return edges
            
            async def node_degree(self, node_id):
                return self.graph.degree(node_id)
            
            def edge_degree(self, src_id, tgt_id):
                # For undirected graphs, edge degree is the same as node degree
                return self.graph.degree(src_id)
            
            async def get_node_edges(self, node_id):
                return [(node_id, neighbor) for neighbor in self.graph.neighbors(node_id)]
            
            def get_edge_weight(self, src_id, tgt_id):
                if self.graph.has_edge(src_id, tgt_id):
                    return self.graph.edges[src_id, tgt_id].get('weight', 1.0)
                return 0.0
            
            def personalized_pagerank(self, reset_prob_chunk, damping=0.1):
                # This is a simplified implementation
                # In a real implementation, you would use a proper PPR library
                import numpy as np
                node_count = self.graph.number_of_nodes()
                return np.ones(node_count) / node_count
            
            async def personalized_pagerank(self, reset_prob_chunk, damping=0.1):
                # This is a simplified implementation
                # In a real implementation, you would use a proper PPR library
                import numpy as np
                node_count = self.graph.number_of_nodes()
                return np.ones(node_count) / node_count
                
            def get_edge_relation_name_batch(self, edges):
                """Get relation names for a batch of edges."""
                relation_names = []
                for src, tgt in edges:
                    edge_data = self.get_edge(src, tgt)
                    if edge_data:
                        relation_name = edge_data.get('relation_name', 'unknown')
                        relation_names.append(relation_name)
                    else:
                        relation_names.append('unknown')
                return relation_names
            
            async def get_edge_relation_name_batch(self, edges):
                """Get relation names for a batch of edges (async version)."""
                relation_names = []
                for src, tgt in edges:
                    edge_data = self.get_edge(src, tgt)
                    if edge_data:
                        relation_name = edge_data.get('relation_name', 'unknown')
                        relation_names.append(relation_name)
                    else:
                        relation_names.append('unknown')
                return relation_names
                
            async def get_edge_by_indices(self, edge_idxs):
                """Get edges by indices."""
                edges = list(self.graph.edges())
                edge_tuples = [edges[i] if 0 <= i < len(edges) else None for i in edge_idxs]
                # Return edge data instead of just edge tuples
                return [self.graph.edges[edge] if edge and edge in self.graph.edges else None for edge in edge_tuples]
        
        graph_wrapper = GraphWrapper(nx_graph)
        
        # Initialize vector databases if enabled
        entities_vdb = None
        relations_vdb = None
        subgraphs_vdb = None
        
        if self.config.use_entities_vdb:
            from Core.Index import get_index, get_index_config
            try:
                # Create persist path for entities index
                entities_persist_path = os.path.join(self.workspace.working_dir, "entities_index")
                entities_vdb_config = get_index_config(self.config, entities_persist_path)
                entities_vdb = get_index(entities_vdb_config)
            except Exception as e:
                entities_vdb = None
            
        if self.config.use_relations_vdb:
            from Core.Index import get_index, get_index_config
            try:
                # Create persist path for relations index
                relations_persist_path = os.path.join(self.workspace.working_dir, "relations_index")
                relations_vdb_config = get_index_config(self.config, relations_persist_path)
                relations_vdb = get_index(relations_vdb_config)
            except Exception as e:
                relations_vdb = None
            
        if getattr(self.config, 'use_subgraphs_vdb', False):
            from Core.Index import get_index, get_index_config
            try:
                # Create persist path for subgraphs index
                subgraphs_persist_path = os.path.join(self.workspace.working_dir, "subgraphs_index")
                subgraphs_vdb_config = get_index_config(self.config, subgraphs_persist_path)
                subgraphs_vdb = get_index(subgraphs_vdb_config)
            except Exception as e:
                subgraphs_vdb = None
        
        context_components = {
            "config": self.config.retriever,
            "graph": graph_wrapper,
            "doc_chunk": self.registry.get_component("document_processor"),
            "llm": self.llm,  # Use lazy-loaded LLM
            "entities_vdb": entities_vdb,
            "relations_vdb": relations_vdb,
            "subgraphs_vdb": subgraphs_vdb,
            "community": getattr(self.config, 'use_community', False),
        }
        
        # Add entity-relationship mappings if enabled
        if self.config.use_entity_link_chunk:
            from Core.Storage.PickleBlobStorage import PickleBlobStorage
            e2r_namespace = self.workspace.make_for("map_e2r")
            r2c_namespace = self.workspace.make_for("map_r2c")
            context_components["entities_to_relationships"] = PickleBlobStorage(namespace=e2r_namespace)
            context_components["relationships_to_chunks"] = PickleBlobStorage(namespace=r2c_namespace)
        else:
            context_components["entities_to_relationships"] = None
            context_components["relationships_to_chunks"] = None
        
        for name, component in context_components.items():
            if component:
                retriever_context.register_context(name, component)
        
        # Create MixRetriever and set it as retriever_context
        mix_retriever = MixRetriever(retriever_context)
        self.context.retriever_context = mix_retriever
        
        # Update query processor's retriever_context
        query_processor = self.registry.get_component("query_processor")
        if query_processor:
            query_processor.retriever_context = mix_retriever
        
        # Create query processor
        self._querier = self.registry.get_component("query_processor")
    
    async def execute_query(self, query: str) -> str:
        """
        Execute query
        
        Args:
            query: Query string
            
        Returns:
            Query result
        """
        logger.info(f"🔍 Executing query: {query[:50]}...")
        
        try:
            query_processor = self.registry.get_component("query_processor")
            response = await query_processor.process_query(query)
            logger.info("✅ Query execution completed")
            return response
        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.context.time_manager.get_all_statistics()
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return self.registry.get_component_info() 