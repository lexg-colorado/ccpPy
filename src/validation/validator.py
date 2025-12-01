"""
Code Validator for translated Python code.
Performs syntax, style, and quality checks.
"""

import ast
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    checks: Dict[str, bool]


class CodeValidator:
    """Validate translated Python code quality."""
    
    def __init__(
        self,
        require_type_hints: bool = True,
        require_docstring: bool = True,
        min_quality_score: float = 6.0
    ):
        """
        Initialize code validator.
        
        Args:
            require_type_hints: Whether type hints are required
            require_docstring: Whether docstrings are required
            min_quality_score: Minimum quality score (0-10)
        """
        self.require_type_hints = require_type_hints
        self.require_docstring = require_docstring
        self.min_quality_score = min_quality_score
    
    def validate(self, code: str) -> ValidationResult:
        """
        Perform complete validation on code.
        
        Args:
            code: Python code string to validate
            
        Returns:
            ValidationResult with detailed findings
        """
        errors = []
        warnings = []
        checks = {}
        
        # 1. Syntax validation
        syntax_valid, syntax_error = self.validate_syntax(code)
        checks['syntax'] = syntax_valid
        if not syntax_valid:
            errors.append(f"Syntax error: {syntax_error}")
            # Can't continue other checks if syntax is invalid
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                quality_score=0.0,
                checks=checks
            )
        
        # Parse AST for further checks
        try:
            tree = ast.parse(code)
        except:
            return ValidationResult(
                is_valid=False,
                errors=["Failed to parse AST"],
                warnings=warnings,
                quality_score=0.0,
                checks=checks
            )
        
        # 2. Check for function definition
        has_function, func_node = self._has_function_def(tree)
        checks['has_function'] = has_function
        if not has_function:
            errors.append("No function definition found")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                quality_score=0.0,
                checks=checks
            )
        
        # 3. Type hints validation
        has_type_hints = self.check_type_hints(func_node)
        checks['type_hints'] = has_type_hints
        if self.require_type_hints and not has_type_hints:
            errors.append("Missing or incomplete type hints")
        elif not has_type_hints:
            warnings.append("Type hints recommended")
        
        # 4. Docstring validation
        has_docstring = self.check_docstring(func_node)
        checks['docstring'] = has_docstring
        if self.require_docstring and not has_docstring:
            errors.append("Missing docstring")
        elif not has_docstring:
            warnings.append("Docstring recommended")
        
        # 5. Basic PEP 8 checks
        pep8_issues = self.check_basic_pep8(code)
        checks['pep8'] = len(pep8_issues) == 0
        if pep8_issues:
            warnings.extend(pep8_issues[:3])  # Limit to first 3
        
        # 6. Check for imports
        used_modules = self._extract_used_modules(tree)
        imported_modules = self._extract_imports(tree)
        missing_imports = used_modules - imported_modules
        checks['imports'] = len(missing_imports) == 0
        if missing_imports:
            warnings.append(f"Potentially missing imports: {', '.join(missing_imports)}")
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(code, checks, errors, warnings)
        
        # Determine if valid
        is_valid = (
            len(errors) == 0 and
            quality_score >= self.min_quality_score
        )
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            checks=checks
        )
    
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python syntax.
        
        Args:
            code: Python code string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Parse error: {e}"
    
    def check_type_hints(self, func_node: ast.FunctionDef) -> bool:
        """
        Check if function has type hints.
        
        Args:
            func_node: AST FunctionDef node
            
        Returns:
            True if has complete type hints
        """
        # Check return type annotation
        has_return_annotation = func_node.returns is not None
        
        # Check parameter annotations
        args = func_node.args
        total_args = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
        
        # Count args with annotations
        annotated_args = sum(1 for arg in args.args if arg.annotation is not None)
        annotated_args += sum(1 for arg in args.posonlyargs if arg.annotation is not None)
        annotated_args += sum(1 for arg in args.kwonlyargs if arg.annotation is not None)
        
        # Allow self/cls without annotation
        if args.args and args.args[0].arg in ('self', 'cls'):
            total_args -= 1
        
        has_param_annotations = (total_args == 0) or (annotated_args == total_args)
        
        return has_return_annotation and has_param_annotations
    
    def check_docstring(self, func_node: ast.FunctionDef) -> bool:
        """
        Check if function has a docstring.
        
        Args:
            func_node: AST FunctionDef node
            
        Returns:
            True if has docstring
        """
        return (
            len(func_node.body) > 0 and
            isinstance(func_node.body[0], ast.Expr) and
            isinstance(func_node.body[0].value, ast.Constant) and
            isinstance(func_node.body[0].value.value, str)
        )
    
    def check_basic_pep8(self, code: str) -> List[str]:
        """
        Perform basic PEP 8 style checks.
        
        Args:
            code: Python code string
            
        Returns:
            List of PEP 8 violations found
        """
        issues = []
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check line length (allow some flexibility)
            if len(line) > 120:
                issues.append(f"Line {i}: Line too long ({len(line)} > 120 chars)")
            
            # Check trailing whitespace
            if line.rstrip() != line:
                issues.append(f"Line {i}: Trailing whitespace")
            
            # Check multiple statements on one line
            if ';' in line and not line.strip().startswith('#'):
                issues.append(f"Line {i}: Multiple statements on one line")
        
        return issues
    
    def calculate_quality_score(
        self,
        code: str,
        checks: Dict[str, bool],
        errors: List[str],
        warnings: List[str]
    ) -> float:
        """
        Calculate overall quality score (0-10).
        
        Args:
            code: Python code string
            checks: Dictionary of check results
            errors: List of errors
            warnings: List of warnings
            
        Returns:
            Quality score from 0-10
        """
        score = 10.0
        
        # Deduct for errors
        score -= len(errors) * 2.0
        
        # Deduct for warnings
        score -= len(warnings) * 0.5
        
        # Deduct for missing checks
        if not checks.get('type_hints', False):
            score -= 1.0
        if not checks.get('docstring', False):
            score -= 1.0
        if not checks.get('pep8', False):
            score -= 0.5
        
        # Bonus for good practices
        if self._has_error_handling(code):
            score += 0.5
        
        # Clamp to 0-10 range
        return max(0.0, min(10.0, score))
    
    def get_validation_report(self, result: ValidationResult) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            result: ValidationResult object
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append(f"Validation: {'PASS' if result.is_valid else 'FAIL'}")
        lines.append(f"Quality Score: {result.quality_score:.1f}/10.0")
        
        if result.errors:
            lines.append("\nErrors:")
            for error in result.errors:
                lines.append(f"  ✗ {error}")
        
        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"  ⚠ {warning}")
        
        lines.append("\nChecks:")
        for check, passed in result.checks.items():
            symbol = "✓" if passed else "✗"
            lines.append(f"  {symbol} {check}")
        
        return "\n".join(lines)
    
    def _has_function_def(self, tree: ast.AST) -> Tuple[bool, Any]:
        """Check if AST has a function definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return True, node
        return False, None
    
    def _extract_used_modules(self, tree: ast.AST) -> set:
        """Extract potentially used modules from code."""
        used = set()
        
        # Common modules based on attribute access
        module_indicators = {
            'os': ['path', 'getcwd', 'environ'],
            'sys': ['argv', 'exit', 'stderr'],
            're': ['match', 'search', 'compile'],
            'json': ['loads', 'dumps'],
            'time': ['sleep', 'time'],
            'datetime': ['now', 'datetime'],
            'pathlib': ['Path'],
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr = node.attr
                for module, indicators in module_indicators.items():
                    if attr in indicators:
                        used.add(module)
        
        return used
    
    def _extract_imports(self, tree: ast.AST) -> set:
        """Extract imported modules from AST."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        return imports
    
    def _has_error_handling(self, code: str) -> bool:
        """Check if code has try/except blocks."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    return True
        except:
            pass
        return False

