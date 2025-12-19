"""
Parser factory for C and C++ code.

Provides a unified interface for parsing both C and C++ source files,
automatically selecting the appropriate parser based on file extension
or explicit language specification.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from tree_sitter import Language, Parser

if TYPE_CHECKING:
    from .ast_parser import CParser
    from .cpp_parser import CppParser


class ParserFactory:
    """
    Factory to create and manage C/C++ parsers.

    Automatically selects the appropriate parser based on file extension
    or allows explicit language specification.
    """

    # File extension to language mapping
    EXTENSION_MAP = {
        # C files
        '.c': 'c',
        # C++ files
        '.cpp': 'cpp',
        '.cxx': 'cpp',
        '.cc': 'cpp',
        '.C': 'cpp',  # Some systems use .C for C++
        # C++ headers
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        '.hh': 'cpp',
        '.H': 'cpp',
        # Ambiguous - header files (configurable)
        '.h': 'c',  # Default, can be overridden
    }

    def __init__(
        self,
        default_header_language: str = 'c',
        config: Optional[Any] = None
    ):
        """
        Initialize parser factory.

        Args:
            default_header_language: Language for .h files ('c' or 'cpp')
            config: Optional Config instance for settings
        """
        self.default_header_language = default_header_language
        self.config = config

        # Override from config if available
        if config:
            self.default_header_language = config.get(
                'parsing.header_language', default_header_language
            )

        # Lazy-loaded parsers
        self._c_parser: Optional['CParser'] = None
        self._cpp_parser: Optional['CppParser'] = None

        # Language instances (lazy-loaded)
        self._c_language: Optional[Language] = None
        self._cpp_language: Optional[Language] = None

    def get_parser(self, file_path: Path, language: Optional[str] = None) -> Any:
        """
        Get appropriate parser for a file.

        Args:
            file_path: Path to the source file
            language: Optional explicit language ('c' or 'cpp')

        Returns:
            Parser instance (CParser or CppParser)
        """
        if language is None:
            language = self.detect_language(file_path)

        if language == 'cpp':
            return self.get_cpp_parser()
        else:
            return self.get_c_parser()

    def get_c_parser(self) -> 'CParser':
        """Get or create C parser instance."""
        if self._c_parser is None:
            from .ast_parser import CParser
            self._c_parser = CParser()
        return self._c_parser

    def get_cpp_parser(self) -> 'CppParser':
        """Get or create C++ parser instance."""
        if self._cpp_parser is None:
            from .cpp_parser import CppParser
            self._cpp_parser = CppParser()
        return self._cpp_parser

    def detect_language(self, file_path: Path) -> str:
        """
        Detect language from file extension.

        Args:
            file_path: Path to source file

        Returns:
            'c' or 'cpp'
        """
        ext = file_path.suffix.lower()

        # Handle .h files specially
        if ext == '.h':
            return self.default_header_language

        return self.EXTENSION_MAP.get(ext, 'c')

    def parse_file(
        self,
        file_path: Path,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a source file using the appropriate parser.

        Args:
            file_path: Path to source file
            language: Optional explicit language

        Returns:
            Parsed data dictionary
        """
        parser = self.get_parser(file_path, language)
        return parser.parse_file(file_path)

    def get_language_instance(self, language: str) -> Language:
        """
        Get tree-sitter Language instance.

        Args:
            language: 'c' or 'cpp'

        Returns:
            Tree-sitter Language instance
        """
        if language == 'c':
            if self._c_language is None:
                import tree_sitter_c
                self._c_language = Language(tree_sitter_c.language(), 'c')
            return self._c_language
        else:
            if self._cpp_language is None:
                import tree_sitter_cpp
                self._cpp_language = Language(tree_sitter_cpp.language(), 'cpp')
            return self._cpp_language

    def get_extensions_for_language(self, language: str) -> List[str]:
        """
        Get file extensions for a language.

        Args:
            language: 'c' or 'cpp'

        Returns:
            List of file extensions
        """
        extensions = []
        for ext, lang in self.EXTENSION_MAP.items():
            if lang == language:
                extensions.append(ext)

        # Add .h based on configuration
        if language == self.default_header_language and '.h' not in extensions:
            extensions.append('.h')

        return extensions

    def get_all_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        return list(set(self.EXTENSION_MAP.keys()))

    @staticmethod
    def is_cpp_file(file_path: Path) -> bool:
        """Check if a file is a C++ file based on extension."""
        ext = file_path.suffix.lower()
        return ext in {'.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh', '.C', '.H'}

    @staticmethod
    def is_c_file(file_path: Path) -> bool:
        """Check if a file is definitely a C file."""
        ext = file_path.suffix.lower()
        return ext == '.c'

    @staticmethod
    def is_header_file(file_path: Path) -> bool:
        """Check if a file is a header file."""
        ext = file_path.suffix.lower()
        return ext in {'.h', '.hpp', '.hxx', '.hh', '.H'}


def create_parser_from_config(config: Any) -> ParserFactory:
    """
    Create a ParserFactory configured from a Config instance.

    Args:
        config: Config instance

    Returns:
        Configured ParserFactory
    """
    header_lang = config.get('parsing.header_language', 'c')
    return ParserFactory(default_header_language=header_lang, config=config)
