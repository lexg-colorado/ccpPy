"""
Prompt Builder for C-to-Python translation.
Constructs prompts with RAG context and few-shot examples.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path


class PromptBuilder:
    """Build structured prompts for LLM translation."""
    
    def __init__(self, include_examples: bool = True, num_examples: int = 3):
        """
        Initialize prompt builder.
        
        Args:
            include_examples: Whether to include few-shot examples
            num_examples: Number of similar examples to include
        """
        self.include_examples = include_examples
        self.num_examples = num_examples
    
    def build_system_prompt(self) -> str:
        """
        Build system prompt for the LLM.
        
        Returns:
            System prompt string
        """
        return """You are an expert C to Python translator. Your task is to translate C functions to idiomatic Python code.

TRANSLATION GUIDELINES:
1. Use Python type hints for all parameters and return types
2. Follow PEP 8 style guidelines
3. Add comprehensive docstrings (Google style)
4. Preserve the original functionality exactly
5. Handle errors appropriately with Python exceptions
6. Use Python idioms where appropriate (e.g., list comprehensions, context managers)
7. Convert C pointers to appropriate Python data structures
8. Replace manual memory management with Python's automatic memory management
9. Use descriptive variable names following Python conventions (snake_case)
10. Output ONLY the Python function code, no explanations or markdown

IMPORTANT: Your response must contain ONLY valid Python code, nothing else."""
    
    def build_translation_prompt(
        self,
        func_data: Dict[str, Any],
        rag_context: Optional[Dict[str, Any]] = None,
        translation_memory: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build complete translation prompt with context.
        
        Args:
            func_data: Function data from AST parser
            rag_context: RAG context with similar examples
            translation_memory: Previously translated functions
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # Add function context
        prompt_parts.append(self._format_function_context(func_data))
        
        # Add few-shot examples if available
        if self.include_examples and rag_context:
            examples_text = self._format_few_shot_examples(
                rag_context,
                translation_memory
            )
            if examples_text:
                prompt_parts.append(examples_text)
        
        # Add the function to translate
        prompt_parts.append(self._format_target_function(func_data))
        
        # Add requirements
        prompt_parts.append(self._format_requirements())
        
        return "\n\n".join(prompt_parts)
    
    def _format_function_context(self, func_data: Dict[str, Any]) -> str:
        """Format function metadata context."""
        name = func_data.get('name', 'unknown')
        file_path = func_data.get('file_path', 'unknown')
        return_type = func_data.get('return_type', 'void')
        params = func_data.get('parameters', [])
        calls = func_data.get('calls', [])
        
        # Format parameters
        param_list = []
        for p in params:
            ptype = p.get('type', 'unknown')
            pname = p.get('name', '')
            if pname:
                param_list.append(f"{pname}: {ptype}")
            else:
                param_list.append(ptype)
        
        params_str = ', '.join(param_list) if param_list else 'void'
        
        # Format dependencies
        calls_str = ', '.join(calls[:10]) if calls else 'none'
        if len(calls) > 10:
            calls_str += f', ... ({len(calls) - 10} more)'
        
        context = f"""CONTEXT:
- Function name: {name}
- Source file: {Path(file_path).name}
- Return type: {return_type}
- Parameters: {params_str}
- Calls these functions: {calls_str}"""
        
        return context
    
    def _format_few_shot_examples(
        self,
        rag_context: Dict[str, Any],
        translation_memory: Optional[Dict[str, Any]]
    ) -> str:
        """Format few-shot examples from RAG context and translation memory."""
        examples = []
        
        if not rag_context or 'similar_examples' not in rag_context:
            return ""
        
        similar_examples = rag_context['similar_examples'][:self.num_examples]
        
        for i, example in enumerate(similar_examples, 1):
            metadata = example.get('metadata', {})
            similarity = example.get('similarity', 0.0)
            func_name = metadata.get('name', 'unknown')
            
            # Check if we have translation for this example
            if translation_memory and func_name in translation_memory:
                trans = translation_memory[func_name]
                c_code = trans.get('c_code', metadata.get('body', ''))
                py_code = trans.get('python_code', '')
                
                if py_code:
                    example_text = f"""EXAMPLE {i} (similarity: {similarity:.2f}):
C Code:
```c
{c_code[:300]}  # Truncated for brevity
```

Python Translation:
```python
{py_code}
```"""
                    examples.append(example_text)
        
        if examples:
            header = "SIMILAR FUNCTION EXAMPLES (for reference):"
            return header + "\n\n" + "\n\n".join(examples)
        
        return ""
    
    def _format_target_function(self, func_data: Dict[str, Any]) -> str:
        """Format the target function to translate."""
        body = func_data.get('body', '')
        name = func_data.get('name', 'unknown')
        
        return f"""FUNCTION TO TRANSLATE:
```c
{body}
```

Translate the above C function '{name}' to Python."""
    
    def _format_requirements(self) -> str:
        """Format translation requirements."""
        return """REQUIREMENTS:
1. Include type hints for all parameters and return type
2. Add a comprehensive docstring (Google style)
3. Follow PEP 8 style guidelines
4. Preserve exact functionality
5. Use appropriate Python idioms
6. Handle potential errors with try/except where needed
7. Replace C-specific constructs with Python equivalents

OUTPUT FORMAT:
Provide ONLY the Python function code. Do not include explanations, markdown formatting, or code fences.
Start directly with 'def function_name...'"""
    
    def build_retry_prompt(
        self,
        original_prompt: str,
        previous_output: str,
        validation_errors: List[str]
    ) -> str:
        """
        Build prompt for retry after validation failure.
        
        Args:
            original_prompt: Original translation prompt
            previous_output: Previous LLM output that failed
            validation_errors: List of validation error messages
            
        Returns:
            Retry prompt string
        """
        errors_text = "\n".join(f"- {error}" for error in validation_errors)
        
        retry_prompt = f"""The previous translation had validation errors:

{errors_text}

PREVIOUS ATTEMPT:
```python
{previous_output}
```

Please fix these issues and provide a corrected translation.

{original_prompt}

Remember: Output ONLY valid Python code, starting with 'def'."""
        
        return retry_prompt

