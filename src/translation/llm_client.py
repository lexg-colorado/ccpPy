"""
LLM Client for Ollama integration.
Handles API requests to local Ollama instance for code translation.
"""

from typing import Dict, Any, Optional, Iterator
import requests
import json
import time


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        temperature: float = 0.2,
        max_retries: int = 3
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = 120  # 2 minutes timeout
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about available models.
        
        Returns:
            Dictionary with model information or None if failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096
    ) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            Exception if generation fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                # Build request payload
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": max_tokens
                    }
                }
                
                if system_prompt:
                    payload["system"] = system_prompt
                
                # Make request
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
                else:
                    error_msg = f"API error (status {response.status_code})"
                    if attempt < self.max_retries - 1:
                        print(f"{error_msg}, retrying... ({attempt + 1}/{self.max_retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise Exception(error_msg)
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"Request timeout, retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                raise Exception("Request timed out after all retries")
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error: {e}, retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        raise Exception("Failed to generate after all retries")
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """
        Generate text with streaming response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Yields:
            Text chunks as they arrive
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        yield chunk['response']
                    if chunk.get('done', False):
                        break
                        
        except Exception as e:
            print(f"Streaming error: {e}")
            raise

