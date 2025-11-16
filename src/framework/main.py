"""
GraphRAG Main Entry File
Adopts new architecture design providing better modularity and extensibility
"""
import argparse
import asyncio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Core.Engine.GraphRAGEngine import GraphRAGEngine
from Option.merged_config import MergedConfig as Config
from Data.QueryDataset import RAGQueryDataset
from Core.Utils.Evaluation import Evaluator
from Core.Common.Logger import logger
from Core.Utils.Display import StatusDisplay, MetricsDisplay, ProgressDisplay
from Core.Utils.Display import TableDisplay


class GraphRAGApplication:
    """GraphRAG application main class"""
    
    def __init__(self, config_path: str, dataset_name: str, method_name: str = "hippo_rag"):
        self.config_path = config_path
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.config = None
        self.engine = None
        self.query_dataset = None
        
    def initialize(self):
        """Initialize application"""
        StatusDisplay.show_info(f"Initializing GraphRAG application with method: {self.method_name}")
        
        # Parse configuration
        self.config = Config.parse(Path(self.config_path), dataset_name=self.dataset_name, method_name=self.method_name)
        
        # Create GraphRAG engine
        self.engine = GraphRAGEngine(config=self.config)
        
        # Load query dataset
        self.query_dataset = RAGQueryDataset(
            data_dir=os.path.join(self.config.data_root, self.dataset_name)
        )
        
        StatusDisplay.show_success("Application initialization completed")
    
    async def process_documents(self, force_rebuild: bool = False):
        """Process documents"""
        StatusDisplay.show_info("Starting document processing workflow")
        
        try:
            # Get corpus
            corpus = self.query_dataset.get_corpus()
            
            # Process documents
            await self.engine.process_documents(corpus, force_rebuild=force_rebuild)
            
            StatusDisplay.show_success("Document processing completed")
            
        except Exception as e:
            StatusDisplay.show_error(f"Document processing failed: {e}")
            raise
    
    async def execute_queries(self, max_queries: int = None) -> List[Dict[str, Any]]:
        """Execute queries"""
        StatusDisplay.show_info("Starting query execution")
        
        results = []
        dataset_len = len(self.query_dataset)
        
        if max_queries:
            dataset_len = min(dataset_len, max_queries)
        
        for i in range(dataset_len):
            query = self.query_dataset[i]
            ProgressDisplay.show_progress(i + 1, dataset_len, f"Executing query: {query['question'][:50]}...")
            
            try:
                response = await self.engine.execute_query(query["question"])
                query["output"] = response
                results.append(query)
                
            except Exception as e:
                StatusDisplay.show_error(f"Query execution failed: {e}")
                query["output"] = f"Error: {e}"
                results.append(query)
        
        StatusDisplay.show_success(f"Query execution completed, processed {len(results)} queries")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """Save results"""
        StatusDisplay.show_info("Saving query results")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as JSON format
            results_df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, "results.json")
            results_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
            
            StatusDisplay.show_success(f"Results saved to: {output_path}")
            
        except Exception as e:
            StatusDisplay.show_error(f"Failed to save results: {e}")
            raise
    
    async def evaluate_results(self, results_path: str):
        """Evaluate results"""
        StatusDisplay.show_info("Starting result evaluation")
        
        try:
            evaluator = Evaluator(results_path, self.dataset_name)
            evaluation_results = await evaluator.evaluate()
            
            # Display evaluation results
            MetricsDisplay.show_performance_metrics(evaluation_results)
            
            # Save evaluation results
            eval_output_path = os.path.join(os.path.dirname(results_path), "evaluation_metrics.json")
            with open(eval_output_path, "w", encoding="utf-8") as f:
                import json
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            
            StatusDisplay.show_success(f"Evaluation results saved to: {eval_output_path}")
            
        except Exception as e:
            StatusDisplay.show_error(f"Result evaluation failed: {e}")
            raise
    
    def display_system_info(self):
        """Display system information"""
        StatusDisplay.show_info("System Information")
        
        try:
            # Display component information
            component_info = self.engine.get_component_info()
            MetricsDisplay.show_component_info(component_info)
            
            # Display performance metrics
            performance_metrics = self.engine.get_performance_metrics()
            MetricsDisplay.show_performance_metrics(performance_metrics)
            
            # Display configuration information
            config_table = [
                ["Configuration Item", "Value"],
                ["Dataset", self.dataset_name],
                ["Working Directory", getattr(self.config, 'working_dir', 'N/A')],
                ["Graph Type", getattr(self.config.graph, 'graph_type', 'N/A') if hasattr(self.config, 'graph') else 'N/A'],
                ["Query Type", getattr(self.config.retriever, 'query_type', 'N/A') if hasattr(self.config, 'retriever') else 'N/A'],
                ["LLM Model", getattr(self.config.llm, 'model', 'N/A') if hasattr(self.config, 'llm') else 'N/A'],
                ["Vector Database", getattr(self.config, 'vdb_type', 'N/A')]
            ]
            
            TableDisplay.show_table(
                headers=config_table[0],
                rows=config_table[1:],
                title="Configuration Information"
            )
        except Exception as e:
            StatusDisplay.show_warning(f"Could not display complete system info: {e}")


def create_directories(config: Config) -> str:
    """Create necessary directories"""
    # Create separate result directory for each query
    result_dir = os.path.join(config.working_dir, config.exp_name, "Results")
    
    # Directory to save currently used configuration
    config_dir = os.path.join(config.working_dir, config.exp_name, "Configs")
    
    # Directory to save overall experiment metrics
    metric_dir = os.path.join(config.working_dir, config.exp_name, "Metrics")
    
    # Create directories
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    
    # Copy configuration files
    import shutil
    # Note: config_path is not available in MergedConfig, so we skip copying config files
    # If needed, you can add config_path to MergedConfig or handle this differently
    
    return result_dir


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Refactored GraphRAG System")
    parser.add_argument("-opt", type=str, required=True, help="Configuration file path")
    parser.add_argument("-dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--method", type=str, default="hippo_rag", 
                       help="Algorithm method to use (hippo_rag, gr, tog, kgp, raptor, light_rag, lgraph_rag, ggraph_rag, dalk)")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild")
    parser.add_argument("--max_queries", type=int, help="Maximum number of queries")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--show_info", action="store_true", help="Show system information")
    
    args = parser.parse_args()
    
    try:
        # Create application
        app = GraphRAGApplication(args.opt, args.dataset_name, method_name=args.method)
        
        # Initialize
        app.initialize()
        
        # Display system information
        if args.show_info:
            app.display_system_info()
        
        # Process documents
        await app.process_documents(force_rebuild=args.force_rebuild)
        
        # Execute queries
        results = await app.execute_queries(max_queries=args.max_queries)
        
        # Create output directory
        output_dir = create_directories(app.config)
        
        # Save results
        app.save_results(results, output_dir)
        
        # Evaluate results (optional)
        if not args.skip_evaluation:
            results_path = os.path.join(output_dir, "results.json")
            await app.evaluate_results(results_path)
        
        StatusDisplay.show_success("🎉 GraphRAG application execution completed!")
        
    except Exception as e:
        StatusDisplay.show_error(f"Application execution failed: {e}")
        logger.exception("Application execution failed")
        sys.exit(1)


if __name__ == "__main__":
    # Set environment variables
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    # Run main function
    asyncio.run(main()) 
