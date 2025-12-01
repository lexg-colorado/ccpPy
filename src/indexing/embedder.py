"""
Embedding Generator for C code functions.
Converts function data into vector embeddings for semantic search.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings for C functions using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
            batch_size: Number of texts to embed at once
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.embedding_dim = None
        
    def load_model(self) -> None:
        """Load the sentence-transformers model."""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def function_to_text(self, func_data: Dict[str, Any]) -> str:
        """
        Convert function data to text representation for embedding.
        
        Combines multiple aspects:
        - Function signature (name, return type, parameters)
        - Called functions (dependencies)
        - Function body (actual code)
        
        Args:
            func_data: Dictionary containing function information
            
        Returns:
            Text representation of the function
        """
        name = func_data.get('name', 'unknown')
        return_type = func_data.get('return_type', 'void')
        parameters = func_data.get('parameters', [])
        calls = func_data.get('calls', [])
        body = func_data.get('body', '')
        
        # Build parameter string
        param_str = ', '.join([
            f"{p.get('type', '')} {p.get('name', '')}"
            for p in parameters
        ])
        
        # Build calls string
        calls_str = ', '.join(calls[:10])  # Limit to first 10 calls
        if len(calls) > 10:
            calls_str += '...'
        
        # Truncate body if too long (keep first 500 chars)
        body_truncated = body[:500] if len(body) > 500 else body
        
        # Combine into semantic representation
        text = f"""Function: {name}
Returns: {return_type}
Parameters: {param_str}
Calls: {calls_str}
Code:
{body_truncated}"""
        
        return text
    
    def generate_embeddings(
        self,
        functions: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of functions.
        
        Args:
            functions: List of function data dictionaries
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (n_functions, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        # Convert functions to text
        texts = []
        for func in tqdm(functions, desc="Converting to text", disable=not show_progress):
            try:
                text = self.function_to_text(func)
                texts.append(text)
            except Exception as e:
                print(f"Error converting function {func.get('name', 'unknown')}: {e}")
                # Use minimal representation as fallback
                texts.append(f"Function: {func.get('name', 'unknown')}")
        
        # Generate embeddings in batches
        embeddings = []
        
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Generating embeddings")
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch {i//self.batch_size}: {e}")
                # Create zero embeddings as fallback
                zero_batch = np.zeros((len(batch_texts), self.embedding_dim))
                embeddings.append(zero_batch)
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        
        return all_embeddings
    
    def generate_single_embedding(self, func_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for a single function.
        
        Args:
            func_data: Dictionary containing function information
            
        Returns:
            NumPy array of shape (embedding_dim,)
        """
        if self.model is None:
            self.load_model()
        
        text = self.function_to_text(func_data)
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Generate embedding for a text query.
        
        Args:
            query_text: Query string
            
        Returns:
            NumPy array of shape (embedding_dim,)
        """
        if self.model is None:
            self.load_model()
        
        embedding = self.model.encode(query_text, convert_to_numpy=True)
        
        return embedding

