#!/usr/bin/env python3
"""
Setup script to create the htop-translator project structure.
Run this once to initialize all directories and placeholder files.
"""

import os
from pathlib import Path


def create_directory_structure(base_path: Path):
    """Create all necessary directories for the project."""
    
    directories = [
        # Source code structure
        "src/parser",
        "src/analysis",
        "src/indexing",
        "src/translation",
        "src/validation",
        "src/utils",
        
        # Data directories (will be gitignored)
        "data/ast_cache",
        "data/graphs",
        "data/embeddings",
        "data/translation_memory",
        
        # Output directory
        "output/htop_py",
        
        # Tests
        "tests",
        
        # Notebooks
        "notebooks",
        
        # Scripts
        "scripts",
        
        # Logs
        "logs",
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")
        
        # Create __init__.py for Python packages
        if directory.startswith("src/") or directory == "tests":
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Package initialization."""\n')


def create_placeholder_files(base_path: Path):
    """Create placeholder Python files with basic structure."""
    
    placeholders = {
        "src/parser/ast_parser.py": '''"""
AST parser using tree-sitter for C code.
"""

from pathlib import Path
from typing import Dict, List, Any


class CParser:
    """Parse C source files into AST representation."""
    
    def __init__(self):
        """Initialize the C parser with tree-sitter."""
        pass
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a single C file.
        
        Args:
            file_path: Path to C source file
            
        Returns:
            Dictionary containing parsed AST information
        """
        raise NotImplementedError("TODO: Implement C parsing")
    
    def extract_functions(self, ast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract function definitions from AST."""
        raise NotImplementedError("TODO: Implement function extraction")
    
    def extract_structs(self, ast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract struct definitions from AST."""
        raise NotImplementedError("TODO: Implement struct extraction")
''',
        
        "src/utils/config.py": '''"""
Configuration management.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Manage project configuration from YAML file."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
''',
        
        "src/utils/logger.py": '''"""
Logging configuration.
"""

import logging
from pathlib import Path


def setup_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
''',
        
        "scripts/01_parse_htop.py": '''#!/usr/bin/env python3
"""
Phase 1: Parse htop C codebase into AST representation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.logger import setup_logger


def main():
    """Parse htop C codebase."""
    config = Config()
    logger = setup_logger("parse_htop", level=config.get("logging.level", "INFO"))
    
    logger.info("Starting htop parsing...")
    logger.info(f"Source path: {config.get('source.htop_path')}")
    
    # TODO: Implement parsing logic
    logger.warning("Parsing not yet implemented!")


if __name__ == "__main__":
    main()
''',
    }
    
    for file_path, content in placeholders.items():
        full_path = base_path / file_path
        if not full_path.exists():
            full_path.write_text(content)
            print(f"✓ Created: {file_path}")


def main():
    """Main setup function."""
    # Assume we're running from the project root
    base_path = Path.cwd()
    
    print("Setting up htop-translator project structure...\n")
    
    # Create directories
    create_directory_structure(base_path)
    
    print()
    
    # Create placeholder files
    create_placeholder_files(base_path)
    
    print("\n✓ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Review and update config.yaml with your paths")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start implementing: src/parser/ast_parser.py")


if __name__ == "__main__":
    main()
