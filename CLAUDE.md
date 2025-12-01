# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG-based code translation system that automatically converts the htop C codebase (~37k total lines, ~29k C code) to Python using dependency-aware context retrieval and local LLMs via Ollama.

**Target Source**: htop repository at `~/Python/htop`
**Key Insight**: This project demonstrates how "unlimited context" translation tools work by using smart chunking, dependency-aware retrieval, iterative processing, and semantic indexing.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with qwen3:4b
ollama pull qwen3:4b
ollama serve
```

### Running the Translation Pipeline

The translation pipeline has 4 phases that must be run sequentially:

```bash
# Phase 1: Parse htop C code into AST representation ✅ WORKING
python scripts/01_parse_htop.py
# Output: Creates data/asts/*.json files and data/asts/parse_summary.json
# Features: Parallel processing, caching, progress bars, statistics

# Phase 2: Build dependency graph ✅ WORKING
python scripts/02_build_graph.py
# Output: Creates data/graphs/function_graph.gpickle, file_graph.gpickle,
#         translation_order.json, graph_analysis.json
# Features: Call graph, file dependencies, topological sort, cycle detection

# Phase 3: Create semantic index for code search ✅ WORKING
python scripts/03_index_code.py
# Output: Creates data/embeddings/function_index.faiss, function_embeddings.npy,
#         function_metadata.json, index_stats.json
# Features: Sentence-transformers embeddings, FAISS similarity search,
#           context retrieval for RAG

# Phase 4: Translate code with RAG context retrieval ✅ WORKING
python scripts/04_translate.py --limit 10  # Translate first 10 functions
python scripts/04_translate.py --dry-run   # Preview without translating
# Output: Creates output/python/*.py files, translation_stats.json
#         Creates memory/translations.json (translation memory)
# Features: Ollama LLM integration, RAG with few-shot examples,
#           retry with validation feedback, comprehensive code validation,
#           quality scoring, translation memory
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Type checking
mypy src/

# Linting (optional, configured in config.yaml)
pylint src/
```

## Architecture

### Pipeline Flow

```
C Source → AST Parser → Dependency Graph → Vector Store → RAG Translator → Python Output
```

1. **AST Parser** (`src/parser/`): Uses tree-sitter-c to extract functions, structs, includes, calls, globals, and macros from C files
2. **Dependency Graph** (`src/analysis/`): Builds call graph, type dependencies, include graph, and module coupling using networkx
3. **Code Indexing** (`src/indexing/`): Creates semantic embeddings at function-level granularity using sentence-transformers and FAISS
4. **Context Retrieval** (`src/translation/context_builder.py`): Gathers relevant context for translation (direct dependencies, similar patterns, module context)
5. **Translation Engine** (`src/translation/translator.py`): Orchestrates C→Python translation using Ollama (qwen3:4b), starting with leaf nodes and working up the dependency tree
6. **Translation Memory** (`src/translation/translation_map.py`): Tracks C→Python entity mappings for consistency
7. **Validation** (`src/validation/`): Validates Python syntax, imports, type hints, and runs linting

### Module Organization

- `src/parser/`: AST parsing using tree-sitter
- `src/analysis/`: Dependency graph construction and analysis
- `src/indexing/`: Semantic code indexing and vector store management
- `src/translation/`: Core translation logic with context retrieval
- `src/validation/`: Python code validation suite
- `src/utils/`: Configuration management and logging utilities
- `scripts/`: CLI tools for each pipeline phase
- `data/`: Cached ASTs, graphs, embeddings, and translation memory (gitignored)
- `output/`: Generated Python code (gitignored)
- `tests/`: Unit tests

### Configuration System

All settings are centralized in `config.yaml`:
- **Source paths**: htop source location and filters
- **Output paths**: Data cache and Python output directories
- **Parsing**: File extensions and size limits
- **Embeddings**: Model type (sentence-transformers/all-MiniLM-L6-v2), chunk size, batch size
- **Vector store**: FAISS with cosine similarity, top-k=5
- **LLM**: Ollama endpoint (localhost:11434), model (qwen3:4b), temperature (0.2), max tokens (4096), context window (8192)
- **Translation**: Start with leaf nodes, max iterations=3, include similar examples
- **Validation**: Syntax/type checking, optional linting
- **Performance**: Parallel parsing, AST caching

Access config values using `Config().get('section.key', default_value)` with dot-notation.

### Key Design Patterns

**Translation Strategy**:
1. Start with leaf nodes (no dependencies on untranslated code)
2. Work up the dependency tree topologically
3. Maintain translation memory for consistency
4. Use RAG to provide relevant context within token budget (~8k context, ~4k output)

**Context Prioritization**:
- Direct dependencies > Similar code patterns > General module context
- Token budget management with truncation strategy

**Library Mappings** (tracked in translation memory):
- ncurses → Python curses or rich/textual
- Manual memory management → Python automatic GC
- C structs → Python dataclasses/classes
- Pointer arithmetic → Python idiomatic alternatives

## Development Workflow

1. **Start small**: Translate a single module first (e.g., Process.c)
2. **Validate early**: Check Python syntax and basic functionality after each translation
3. **Iterate**: Refine prompts and context retrieval based on translation quality
4. **Scale up**: Gradually translate more complex modules
5. **Update translation memory**: Document decisions and mappings

## Current Implementation Status

- [x] Project structure created
- [x] Configuration system implemented
- [x] **Phase 1 Complete: AST Parser** (`src/parser/ast_parser.py`, `scripts/01_parse_htop.py`)
  - Fixed tree-sitter language initialization (2025-12-01)
  - `CParser` class: Extracts functions, structs, includes, and call relationships from C files
  - `BatchParser` class: Parallel processing, caching, and statistics generation
  - Completed features:
    - Parallel file parsing with ProcessPoolExecutor
    - Smart caching system (skips unchanged files)
    - Comprehensive statistics (function counts, call graphs, struct analysis)
    - Progress tracking with tqdm
    - Error handling and reporting
    - JSON export of parsed ASTs
- [x] **Phase 2 Complete: Dependency Graph Builder** (`scripts/02_build_graph.py`)
  - `DependencyGraphBuilder` class: Analyzes code dependencies using NetworkX
  - Function call graph (directed graph of function calls)
  - File dependency graph (include relationships between files)
  - Completed features:
    - Topological sort for optimal translation ordering
    - Leaf node identification (functions with no dependencies)
    - Strongly connected components detection (cycle detection)
    - Graph complexity analysis (density, connectivity metrics)
    - Most called/calling function identification
    - Export graphs as pickle (for processing) and JSON (for inspection)
    - Translation order generation for Phase 4
- [x] **Phase 3 Complete: Semantic Indexing** (`scripts/03_index_code.py`, `src/indexing/`)
  - `EmbeddingGenerator` class: Converts C functions to semantic vectors using sentence-transformers
  - `VectorStore` class: FAISS-based similarity search and context retrieval
  - `SemanticIndexer` class: Orchestrates embedding generation and index building
  - Completed features:
    - Sentence-transformers model (all-MiniLM-L6-v2, configurable)
    - Rich text representation (signature + dependencies + code body)
    - Batch processing for efficient embedding generation
    - FAISS vector index with cosine similarity
    - Function-to-function similarity search
    - Context retrieval for RAG translation (`get_context_for_translation()`)
    - Index validation with sample queries
    - Save/load functionality for embeddings and index
    - Comprehensive statistics and performance metrics
- [x] **Phase 4 Complete: RAG Translation Engine** (`scripts/04_translate.py`, `src/translation/`, `src/validation/`)
  - `OllamaClient` class: Integration with local Ollama LLM (qwen3:4b)
  - `PromptBuilder` class: RAG-aware prompt construction with few-shot examples
  - `FunctionTranslator` class: Core translation logic with retry mechanism
  - `CodeValidator` class: Comprehensive Python code validation
  - `TranslationOrchestrator` class: End-to-end translation pipeline
  - Completed features:
    - Ollama API integration with retry logic and exponential backoff
    - RAG context retrieval using Phase 3 embeddings
    - Few-shot learning from similar translated functions
    - Iterative refinement (max 3 attempts) with validation feedback
    - Translation memory for C→Python entity mappings
    - Comprehensive validation:
      * Syntax checking (AST parsing)
      * Type hints validation (parameters and return type)
      * Docstring presence (Google style)
      * Basic PEP 8 compliance
      * Import analysis
    - Quality scoring system (0-10 scale)
    - Statistics tracking (success rate, avg quality, attempts)
    - CLI with --limit, --dry-run, --no-leaves options
    - Grouped output by source file

**All phases complete!** Full RAG-based C-to-Python translation pipeline operational.

## Important Constraints

**C-Specific Challenges**:
- Pointer arithmetic and manual memory management
- Preprocessor macros (context-dependent)
- Platform-specific system calls
- Performance-critical sections

**Context Window Management**:
- Even with smart retrieval, deeply interconnected functions may need multi-pass translation
- Priority ranking ensures most critical context fits in token budget

**LLM Configuration**:
- Low temperature (0.2) for consistency, not creativity
- Local Ollama instance required (no API costs, privacy-preserving)
- Model: qwen3:4b balances quality and speed

## Success Metrics

**MVP**: Successfully translate 3-5 core htop modules with syntactically valid, functional Python code

**Full Success**: Translate 80%+ of htop functionality while maintaining semantic correctness and code quality
