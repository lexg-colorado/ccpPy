"""
Function Translator - Core translation logic.
Orchestrates LLM-based C-to-Python translation with RAG and validation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re

from translation.llm_client import OllamaClient
from translation.prompt_builder import PromptBuilder
from validation.validator import CodeValidator, ValidationResult


@dataclass
class TranslationResult:
    """Result of translating a single function."""
    function_name: str
    success: bool
    python_code: str
    c_code: str
    validation: Optional[ValidationResult]
    attempts: int
    error_message: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class FunctionTranslator:
    """Translate C functions to Python using LLM with RAG."""
    
    def __init__(
        self,
        llm_client: OllamaClient,
        prompt_builder: PromptBuilder,
        validator: CodeValidator,
        vector_store,
        max_iterations: int = 3
    ):
        """
        Initialize function translator.
        
        Args:
            llm_client: Ollama LLM client
            prompt_builder: Prompt construction service
            validator: Code validation service
            vector_store: Vector store for RAG
            max_iterations: Maximum translation attempts
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.validator = validator
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        self.translation_memory = {}
    
    def translate_function(
        self,
        func_data: Dict[str, Any],
        include_rag: bool = True
    ) -> TranslationResult:
        """
        Translate a single C function to Python.
        
        Args:
            func_data: Function data from AST parser
            include_rag: Whether to include RAG context
            
        Returns:
            TranslationResult with translation and validation info
        """
        func_name = func_data.get('name', 'unknown')
        c_code = func_data.get('body', '')
        dependencies = func_data.get('calls', [])
        
        # Get RAG context if requested
        rag_context = None
        if include_rag:
            rag_context = self.vector_store.get_context_for_translation(
                func_name,
                num_examples=self.prompt_builder.num_examples
            )
        
        # Build system prompt
        system_prompt = self.prompt_builder.build_system_prompt()
        
        # Try translation with retries
        for attempt in range(1, self.max_iterations + 1):
            try:
                # Build prompt
                if attempt == 1:
                    prompt = self.prompt_builder.build_translation_prompt(
                        func_data,
                        rag_context,
                        self.translation_memory
                    )
                else:
                    # Retry with validation errors
                    prompt = self.prompt_builder.build_retry_prompt(
                        original_prompt,
                        previous_output,
                        validation_result.errors
                    )
                
                if attempt == 1:
                    original_prompt = prompt
                
                # Call LLM
                llm_response = self.llm_client.generate(
                    prompt,
                    system_prompt=system_prompt
                )
                
                # Parse and clean output
                python_code = self._parse_output(llm_response)
                
                if not python_code:
                    previous_output = llm_response
                    continue
                
                # Validate
                validation_result = self.validator.validate(python_code)
                
                if validation_result.is_valid:
                    # Success!
                    result = TranslationResult(
                        function_name=func_name,
                        success=True,
                        python_code=python_code,
                        c_code=c_code,
                        validation=validation_result,
                        attempts=attempt,
                        dependencies=dependencies
                    )
                    
                    # Add to translation memory
                    self._add_to_memory(func_data, python_code, validation_result)
                    
                    return result
                
                # Validation failed, prepare for retry
                previous_output = python_code
                
                if attempt == self.max_iterations:
                    # Final attempt failed
                    return TranslationResult(
                        function_name=func_name,
                        success=False,
                        python_code=python_code,
                        c_code=c_code,
                        validation=validation_result,
                        attempts=attempt,
                        error_message=f"Validation failed after {attempt} attempts",
                        dependencies=dependencies
                    )
                    
            except Exception as e:
                if attempt == self.max_iterations:
                    return TranslationResult(
                        function_name=func_name,
                        success=False,
                        python_code="",
                        c_code=c_code,
                        validation=None,
                        attempts=attempt,
                        error_message=str(e),
                        dependencies=dependencies
                    )
                previous_output = ""
                continue
        
        # Should not reach here
        return TranslationResult(
            function_name=func_name,
            success=False,
            python_code="",
            c_code=c_code,
            validation=None,
            attempts=self.max_iterations,
            error_message="Translation failed",
            dependencies=dependencies
        )
    
    def _parse_output(self, llm_response: str) -> str:
        """
        Parse and clean LLM output to extract Python code.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Cleaned Python code
        """
        # Remove markdown code fences if present
        code = llm_response.strip()
        
        # Remove ```python or ``` markers
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        
        # Find the function definition
        lines = code.split('\n')
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                start_idx = i
                break
        
        if start_idx is not None:
            code = '\n'.join(lines[start_idx:])
        
        # Clean up any explanatory text before the function
        code = code.strip()
        
        return code
    
    def _add_to_memory(
        self,
        func_data: Dict[str, Any],
        python_code: str,
        validation: ValidationResult
    ) -> None:
        """
        Add successful translation to memory.
        
        Args:
            func_data: Original function data
            python_code: Translated Python code
            validation: Validation result
        """
        func_name = func_data.get('name')
        if not func_name:
            return
        
        self.translation_memory[func_name] = {
            'c_code': func_data.get('body', ''),
            'python_code': python_code,
            'quality_score': validation.quality_score,
            'validation_passed': validation.is_valid,
            'dependencies': func_data.get('calls', []),
            'file_path': func_data.get('file_path', ''),
            'return_type': func_data.get('return_type', ''),
            'parameters': func_data.get('parameters', [])
        }
    
    def save_translation_memory(self, output_path: Path) -> None:
        """
        Save translation memory to disk.
        
        Args:
            output_path: Path to save translation memory JSON
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.translation_memory, f, indent=2, ensure_ascii=False)
    
    def load_translation_memory(self, input_path: Path) -> None:
        """
        Load translation memory from disk.
        
        Args:
            input_path: Path to translation memory JSON
        """
        if not input_path.exists():
            return
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.translation_memory = json.load(f)
    
    def get_translation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about translation memory.
        
        Returns:
            Dictionary with statistics
        """
        if not self.translation_memory:
            return {
                'total_translations': 0,
                'average_quality': 0.0,
                'validation_pass_rate': 0.0
            }
        
        total = len(self.translation_memory)
        quality_scores = [
            t['quality_score']
            for t in self.translation_memory.values()
            if 'quality_score' in t
        ]
        passed = sum(
            1 for t in self.translation_memory.values()
            if t.get('validation_passed', False)
        )
        
        return {
            'total_translations': total,
            'average_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'validation_pass_rate': (passed / total * 100) if total > 0 else 0.0
        }

