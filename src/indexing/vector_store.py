"""
Vector Store for semantic search using FAISS.
Enables efficient similarity search over function embeddings.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import numpy as np
import faiss


class VectorStore:
    """FAISS-based vector store for function embeddings."""
    
    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "flat"
    ):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension
            metric: Similarity metric ("cosine" or "euclidean")
            index_type: Type of FAISS index ("flat" or "ivf")
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.index = None
        self.metadata = []  # List of function metadata dicts
        self.function_name_to_id = {}  # Map function names to index IDs
        
        self._create_index()
    
    def _create_index(self) -> None:
        """Create FAISS index based on metric and type."""
        if self.metric == "cosine":
            # For cosine similarity, normalize vectors and use inner product
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                # IVF index for larger datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:  # euclidean
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of shape (n_vectors, dimension)
            metadata: List of metadata dicts for each embedding
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) must match "
                f"number of metadata entries ({len(metadata)})"
            )
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            print(f"Training index on {embeddings.shape[0]} vectors...")
            self.index.train(embeddings)
        
        # Add to index
        start_id = len(self.metadata)
        self.index.add(embeddings)
        
        # Store metadata
        for i, meta in enumerate(metadata):
            self.metadata.append(meta)
            func_name = meta.get('name')
            if func_name:
                self.function_name_to_id[func_name] = start_id + i
        
        print(f"Added {len(metadata)} embeddings. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            top_k: Number of results to return
            
        Returns:
            List of (metadata, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            
            metadata = self.metadata[idx]
            
            # Convert distance to similarity score
            if self.metric == "cosine":
                # Inner product is already similarity for normalized vectors
                similarity = float(dist)
            else:
                # Convert L2 distance to similarity (inverse)
                similarity = 1.0 / (1.0 + float(dist))
            
            results.append((metadata, similarity))
        
        return results
    
    def find_similar_functions(
        self,
        func_name: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find functions similar to a given function.
        
        Args:
            func_name: Name of the function to find similar functions for
            top_k: Number of results to return
            exclude_self: Whether to exclude the query function from results
            
        Returns:
            List of (function_name, similarity_score) tuples
        """
        if func_name not in self.function_name_to_id:
            return []
        
        func_id = self.function_name_to_id[func_name]
        
        # Get embedding from index
        query_embedding = self.index.reconstruct(func_id)
        
        # Search (get extra if excluding self)
        k = top_k + 1 if exclude_self else top_k
        results = self.search(query_embedding, top_k=k)
        
        # Filter and format
        similar_funcs = []
        for metadata, score in results:
            name = metadata.get('name')
            if exclude_self and name == func_name:
                continue
            similar_funcs.append((name, score))
        
        return similar_funcs[:top_k]
    
    def get_context_for_translation(
        self,
        func_name: str,
        num_examples: int = 3
    ) -> Dict[str, Any]:
        """
        Get translation context for a function (for RAG).
        
        Args:
            func_name: Name of the function to get context for
            num_examples: Number of similar examples to include
            
        Returns:
            Dictionary with target function info and similar examples
        """
        if func_name not in self.function_name_to_id:
            return {'error': f'Function {func_name} not found'}
        
        func_id = self.function_name_to_id[func_name]
        target_func = self.metadata[func_id]
        
        # Find similar functions
        similar = self.find_similar_functions(
            func_name,
            top_k=num_examples,
            exclude_self=True
        )
        
        # Get full metadata for similar functions
        examples = []
        for similar_name, score in similar:
            if similar_name in self.function_name_to_id:
                similar_id = self.function_name_to_id[similar_name]
                examples.append({
                    'metadata': self.metadata[similar_id],
                    'similarity': score
                })
        
        return {
            'target_function': target_func,
            'similar_examples': examples,
            'num_examples': len(examples)
        }
    
    def save(self, save_dir: Path) -> None:
        """
        Save index and metadata to disk.
        
        Args:
            save_dir: Directory to save files to
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / 'function_index.faiss'
        faiss.write_index(self.index, str(index_path))
        print(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_path = save_dir / 'function_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self.metadata,
                'function_name_to_id': self.function_name_to_id,
                'dimension': self.dimension,
                'metric': self.metric,
                'index_type': self.index_type
            }, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, load_dir: Path) -> 'VectorStore':
        """
        Load index and metadata from disk.
        
        Args:
            load_dir: Directory containing saved files
            
        Returns:
            Loaded VectorStore instance
        """
        load_dir = Path(load_dir)
        
        # Load metadata
        metadata_path = load_dir / 'function_metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create instance
        store = cls(
            dimension=data['dimension'],
            metric=data['metric'],
            index_type=data['index_type']
        )
        
        # Load FAISS index
        index_path = load_dir / 'function_index.faiss'
        store.index = faiss.read_index(str(index_path))
        
        # Restore metadata
        store.metadata = data['metadata']
        store.function_name_to_id = data['function_name_to_id']
        
        print(f"Loaded index with {store.index.ntotal} vectors")
        
        return store
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'metric': self.metric,
            'index_type': self.index_type,
            'num_functions': len(self.function_name_to_id)
        }

