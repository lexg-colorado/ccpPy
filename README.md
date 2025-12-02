# C-to-Python Translator

A proof-of-concept RAG-based code translation system that automatically converts C codebases to Python using dependency-aware context retrieval and local LLMs.

## Overview

This project explores how commercial "unlimited context" code translation tools actually work by building a transparent implementation using:
- **AST parsing** for code structure extraction
- **Dependency graph analysis** for intelligent context retrieval
- **Semantic embeddings** for finding similar code patterns
- **Local LLM (qwen3:4b)** for translation via Ollama

## Project Structure

See `GAMEPLAN.md` for detailed architecture and implementation plan.

```
├── src/           # Core translation pipeline
├── scripts/       # CLI tools for each phase
├── data/          # Cached analysis (gitignored)
├── output/        # Translated Python code (gitignored)
└── tests/         # Unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
# Install tree-sitter with C language support
pip install -r requirements.txt

# Ensure Ollama is running with qwen3:4b
ollama pull qwen3:4b
ollama serve
```

### 2. Configure Paths

Edit `config.yaml` to point to your C source directory:
```yaml
source:
  source_path: "/path/to/your/c/project"
```

### 3. Run the Pipeline

```bash
# Phase 1: Parse C code
python scripts/01_parse_c_code.py

# Phase 2: Build dependency graph
python scripts/02_build_graph.py

# Phase 3: Create semantic index
python scripts/03_index_code.py

# Phase 4: Translate code
python scripts/04_translate.py --module Process
```

## Development Workflow

1. **Start small**: Translate a single module first (e.g., Process.c)
2. **Validate early**: Check Python syntax and basic functionality
3. **Iterate**: Refine prompts based on translation quality
4. **Scale up**: Gradually translate more complex modules

## Troubleshooting

### tree-sitter Language Initialization Error

If you encounter `TypeError: Language.__init__() missing 1 required positional argument: 'name'`:

This was a compatibility issue with modern tree-sitter Python bindings. The fix has been applied in `src/parser/ast_parser.py` - the language object from `tree_sitter_c.language()` is now assigned directly without wrapping in `Language()`.

## Current Status

- [x] Project structure created
- [x] **Phase 1 Complete: AST Parser Implementation**
  - CParser class extracts functions, structs, includes, and call relationships
  - BatchParser handles parallel file processing with caching
  - `01_parse_c_code.py` script fully functional
  - Generates comprehensive statistics and summaries
- [x] **Phase 2 Complete: Dependency Graph Builder**
  - Function call graph (who calls whom)
  - File dependency graph (include relationships)
  - Topological sort for optimal translation order
  - Cycle detection and strongly connected components
  - `02_build_graph.py` script fully functional
- [x] **Phase 3 Complete: Semantic Indexing**
  - Sentence-transformers embeddings (all-MiniLM-L6-v2)
  - FAISS vector index for fast similarity search
  - Function-to-function similarity matching
  - Context retrieval for RAG translation
  - `03_index_code.py` script fully functional
- [x] **Phase 4 Complete: RAG-Based Translation Engine**
  - Ollama LLM integration (qwen3:4b) with fixes for model quirks
  - RAG-aware prompt construction with few-shot examples
  - Retry mechanism with validation feedback (max 3 iterations)
  - Comprehensive Python code validation (syntax, type hints, PEP 8)
  - Translation memory for consistency
  - Quality scoring (0-10 scale)
  - Debug mode with failed translation logging
  - Verbose logging for troubleshooting
  - `04_translate.py` script fully functional
- [x] **Validation Suite** (integrated into Phase 4)
  - Syntax validation, type hints, docstrings, PEP 8 checks

## Goals

**MVP**: Successfully translate core C modules with syntactically valid, functional Python code.

**Full Success**: Translate 80%+ of C codebase functionality while maintaining semantic correctness and code quality.

## License

This is a research/learning project. Ensure you respect the license of any source code you translate.
