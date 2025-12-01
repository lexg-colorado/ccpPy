#!/usr/bin/env python3
"""
Phase 3: Semantic Indexing - Generate embeddings and build vector index.

This script:
1. Loads parsed function data from Phase 1
2. Generates embeddings for all functions
3. Builds FAISS vector index for similarity search
4. Saves embeddings and index for later use
5. Validates with sample queries
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from indexing.embedder import EmbeddingGenerator
from indexing.vector_store import VectorStore
from utils.config import Config
from utils.logger import setup_logger


class SemanticIndexer:
    """Build semantic index for C functions."""
    
    def __init__(self, config: Config, logger):
        """Initialize semantic indexer with configuration."""
        self.config = config
        self.logger = logger
        
        # Get paths
        self.cache_dir = Path(config.get('output.ast_cache'))
        self.embeddings_dir = Path(config.get('output.embeddings'))
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Get embedding config
        self.model_name = config.get('embeddings.model_name', 'all-MiniLM-L6-v2')
        self.batch_size = config.get('embeddings.batch_size', 32)
        
        # Get vector store config
        self.metric = config.get('vector_store.similarity_metric', 'cosine')
        self.top_k = config.get('vector_store.top_k', 5)
        
        # Initialize components
        self.embedder = None
        self.vector_store = None
        self.functions = []
    
    def load_parsed_data(self) -> List[Dict[str, Any]]:
        """
        Load parsed function data from cache.
        
        Returns:
            List of function data dictionaries
        """
        self.logger.info("Loading parsed function data...")
        
        summary_path = self.cache_dir / 'parse_summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Parse summary not found at {summary_path}. "
                "Run scripts/01_parse_htop.py first!"
            )
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        functions = []
        files_processed = 0
        
        for result in summary['results']:
            if not result['success']:
                continue
            
            data = result['data']
            
            # Collect all functions from this file
            for func in data['functions']:
                functions.append(func)
            
            files_processed += 1
        
        self.logger.info(
            f"Loaded {len(functions)} functions from {files_processed} files"
        )
        
        return functions
    
    def generate_embeddings(
        self,
        functions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Generate embeddings for all functions.
        
        Args:
            functions: List of function data dictionaries
            
        Returns:
            NumPy array of embeddings
        """
        self.logger.info(
            f"Generating embeddings using model: {self.model_name}"
        )
        
        self.embedder = EmbeddingGenerator(
            model_name=self.model_name,
            batch_size=self.batch_size
        )
        
        embeddings = self.embedder.generate_embeddings(
            functions,
            show_progress=True
        )
        
        self.logger.info(
            f"Generated embeddings: shape {embeddings.shape}, "
            f"dtype {embeddings.dtype}"
        )
        
        return embeddings
    
    def build_vector_index(
        self,
        embeddings: np.ndarray,
        functions: List[Dict[str, Any]]
    ) -> VectorStore:
        """
        Build FAISS vector index.
        
        Args:
            embeddings: Function embeddings array
            functions: Function metadata
            
        Returns:
            VectorStore instance
        """
        self.logger.info("Building FAISS vector index...")
        
        dimension = embeddings.shape[1]
        
        self.vector_store = VectorStore(
            dimension=dimension,
            metric=self.metric,
            index_type='flat'
        )
        
        # Prepare metadata (minimal fields for efficiency)
        metadata = []
        for func in functions:
            metadata.append({
                'name': func['name'],
                'file': func['file_path'],
                'line': func['start_line'],
                'return_type': func['return_type'],
                'parameters': func['parameters'],
                'calls': func['calls'][:20],  # Limit calls for size
                'body': func['body'][:500]  # Truncate body
            })
        
        self.vector_store.add_embeddings(embeddings, metadata)
        
        self.logger.info("Vector index built successfully")
        
        return self.vector_store
    
    def save_artifacts(
        self,
        embeddings: np.ndarray,
        functions: List[Dict[str, Any]]
    ) -> None:
        """
        Save embeddings, index, and metadata.
        
        Args:
            embeddings: Function embeddings array
            functions: Function data
        """
        self.logger.info("Saving artifacts...")
        
        # Save vector store (includes index and metadata)
        self.vector_store.save(self.embeddings_dir)
        
        # Save raw embeddings
        embeddings_path = self.embeddings_dir / 'function_embeddings.npy'
        np.save(embeddings_path, embeddings)
        self.logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save indexing statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'num_functions': len(functions),
            'embedding_dimension': embeddings.shape[1],
            'model_name': self.model_name,
            'metric': self.metric,
            'total_size_mb': embeddings.nbytes / (1024 * 1024)
        }
        
        stats_path = self.embeddings_dir / 'index_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved statistics to {stats_path}")
    
    def validate_index(self) -> Dict[str, Any]:
        """
        Validate the index with sample queries.
        
        Returns:
            Validation results dictionary
        """
        self.logger.info("Validating index with sample queries...")
        
        validation_results = {
            'sample_queries': [],
            'avg_search_time_ms': 0
        }
        
        # Sample function names to test
        test_functions = [
            'Process_new',
            'Panel_add',
            'Vector_get',
            'xSnprintf',
            'Settings_read'
        ]
        
        import time
        search_times = []
        
        for func_name in test_functions:
            if func_name not in self.vector_store.function_name_to_id:
                continue
            
            start = time.time()
            similar = self.vector_store.find_similar_functions(
                func_name,
                top_k=self.top_k,
                exclude_self=True
            )
            search_time = (time.time() - start) * 1000  # Convert to ms
            search_times.append(search_time)
            
            result = {
                'query': func_name,
                'similar_functions': [
                    {'name': name, 'similarity': float(score)}
                    for name, score in similar
                ],
                'search_time_ms': search_time
            }
            
            validation_results['sample_queries'].append(result)
            
            self.logger.info(
                f"Query: {func_name} -> "
                f"Found {len(similar)} similar functions "
                f"({search_time:.2f}ms)"
            )
        
        if search_times:
            validation_results['avg_search_time_ms'] = np.mean(search_times)
        
        return validation_results
    
    def print_validation_results(self, results: Dict[str, Any]) -> None:
        """Print validation results in readable format."""
        print("\n" + "="*60)
        print("SEMANTIC INDEX VALIDATION")
        print("="*60)
        
        for query_result in results['sample_queries']:
            print(f"\nQuery: {query_result['query']}")
            print("  Similar functions:")
            for func in query_result['similar_functions']:
                print(f"    - {func['name']:30s} (similarity: {func['similarity']:.3f})")
        
        print(f"\nAverage search time: {results['avg_search_time_ms']:.2f}ms")
        print("\n" + "="*60)
    
    def print_statistics(self) -> None:
        """Print final statistics."""
        stats = self.vector_store.get_statistics()
        
        print("\n" + "="*60)
        print("SEMANTIC INDEXING COMPLETE")
        print("="*60)
        print(f"\nTotal functions indexed: {stats['total_vectors']}")
        print(f"Embedding dimension:     {stats['dimension']}")
        print(f"Similarity metric:       {stats['metric']}")
        print(f"Index type:              {stats['index_type']}")
        print(f"\nOutputs saved to: {self.embeddings_dir}")
        print("  - function_index.faiss")
        print("  - function_metadata.json")
        print("  - function_embeddings.npy")
        print("  - index_stats.json")
        print("\n" + "="*60)


def main():
    """Main entry point for semantic indexing."""
    # Load configuration
    config_path = project_root / "config.yaml"
    config = Config(str(config_path))
    
    # Setup logging
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', 'logs/translator.log')
    logger = setup_logger("semantic_indexer", level=log_level, log_file=log_file)
    
    logger.info("="*60)
    logger.info("Starting semantic indexing")
    logger.info("="*60)
    
    try:
        # Create indexer
        indexer = SemanticIndexer(config, logger)
        
        # Load parsed data
        functions = indexer.load_parsed_data()
        
        if not functions:
            logger.error("No functions found to index!")
            return 1
        
        # Generate embeddings
        embeddings = indexer.generate_embeddings(functions)
        
        # Build vector index
        indexer.build_vector_index(embeddings, functions)
        
        # Save everything
        indexer.save_artifacts(embeddings, functions)
        
        # Validate
        validation_results = indexer.validate_index()
        
        # Print results
        indexer.print_validation_results(validation_results)
        indexer.print_statistics()
        
        logger.info("Semantic indexing complete!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during semantic indexing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

