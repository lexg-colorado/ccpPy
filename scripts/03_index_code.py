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
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from indexing.embedder import EmbeddingGenerator
from indexing.vector_store import VectorStore
from utils.config import Config
from utils.logger import setup_logger
from utils.cli import (
    create_base_parser,
    load_config_from_args,
    handle_list_profiles,
    error_exit
)


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

    def has_existing_index(self) -> bool:
        """Check if an existing index exists."""
        index_path = self.embeddings_dir / 'function_index.faiss'
        metadata_path = self.embeddings_dir / 'function_metadata.json'
        return index_path.exists() and metadata_path.exists()

    def load_existing_index(self) -> bool:
        """
        Load an existing vector index.

        Returns:
            True if successful, False otherwise
        """
        if not self.has_existing_index():
            return False

        self.logger.info("Loading existing vector index...")

        # Get dimension from stats if available
        stats_path = self.embeddings_dir / 'index_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            dimension = stats.get('embedding_dimension', 384)
        else:
            dimension = 384  # Default for all-MiniLM-L6-v2

        self.vector_store = VectorStore(
            dimension=dimension,
            metric=self.metric,
            index_type='flat'
        )
        self.vector_store.load(self.embeddings_dir)

        self.logger.info(
            f"Loaded index with {len(self.vector_store.function_name_to_id)} functions"
        )
        return True

    def query_similar_functions(
        self,
        query: str,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Query for similar functions by name.

        Args:
            query: Function name to search for
            top_k: Number of results to return

        Returns:
            List of (function_name, similarity_score) tuples
        """
        if self.vector_store is None:
            raise RuntimeError("Index not loaded. Run indexing first.")

        if query not in self.vector_store.function_name_to_id:
            self.logger.warning(f"Function '{query}' not found in index")
            return []

        return self.vector_store.find_similar_functions(
            query,
            top_k=top_k,
            exclude_self=True
        )
    
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
                "Run scripts/01_parse_c_code.py first!"
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
        
        # Sample function names to test (pick from indexed functions)
        all_function_names = list(self.vector_store.function_name_to_id.keys())
        # Select up to 5 functions evenly distributed across the index
        if len(all_function_names) >= 5:
            step = len(all_function_names) // 5
            test_functions = [all_function_names[i * step] for i in range(5)]
        else:
            test_functions = all_function_names[:5]
        
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = create_base_parser(
        description="Phase 3: Semantic Indexing - Generate embeddings and build vector index."
    )

    # Add script-specific arguments
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild embeddings and index (ignore existing)'
    )
    parser.add_argument(
        '--query', '-q',
        metavar='FUNC',
        help='Query mode: find similar functions to FUNC'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        metavar='N',
        help='Number of similar functions to return (default: 5)'
    )

    return parser.parse_args()


def main():
    """Main entry point for semantic indexing."""
    args = parse_args()

    # Handle --list-profiles
    if handle_list_profiles(args, project_root):
        return 0

    # Load configuration with CLI args
    config, logger = load_config_from_args(args, project_root, "semantic_indexer")

    logger.info("=" * 60)
    logger.info("Starting semantic indexing")
    logger.info("=" * 60)

    try:
        # Create indexer
        indexer = SemanticIndexer(config, logger)

        # Query mode - just load index and query
        if args.query:
            if not indexer.load_existing_index():
                error_exit("No existing index found. Run indexing first!")

            logger.info(f"Querying for functions similar to: {args.query}")
            similar = indexer.query_similar_functions(args.query, top_k=args.top_k)

            if not similar:
                print(f"\nNo similar functions found for: {args.query}")
                print("(Function may not exist in the index)")
            else:
                print(f"\nFunctions similar to '{args.query}':")
                print("-" * 50)
                for name, score in similar:
                    print(f"  {name:40s} {score:.3f}")
                print("-" * 50)

            return 0

        # Check if we need to rebuild
        if indexer.has_existing_index() and not args.rebuild:
            logger.info("Existing index found. Use --rebuild to regenerate.")
            # Just validate existing index
            if indexer.load_existing_index():
                validation_results = indexer.validate_index()
                indexer.print_validation_results(validation_results)
                indexer.print_statistics()
                return 0

        # Load parsed data
        functions = indexer.load_parsed_data()

        if not functions:
            error_exit("No functions found to index!")

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

