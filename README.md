# C2Py - RAG-Based C/C++ to Python Translator

A local-first code translation system that converts C/C++ codebases to Python using dependency-aware RAG (Retrieval-Augmented Generation) and local LLMs via Ollama.

## Features

- **AST-Based Parsing**: Uses tree-sitter for accurate C and C++ code structure extraction
- **Dependency-Aware Translation**: Builds call graphs to translate in optimal order (leaf nodes first)
- **Semantic Code Search**: FAISS-powered vector search finds similar code patterns for context
- **Local LLM Integration**: Uses Ollama for privacy-preserving, cost-free translation
- **Validation Pipeline**: Automatic syntax, type hint, and PEP 8 validation
- **Translation Memory**: Maintains consistency across related code translations
- **Multi-Language Support**: Handles both C and C++ source files
- **Iterative Refinement**: Retries failed translations with validation feedback

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) with a supported model (default: qwen3:4b)
- 8GB RAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/lexg-colorado/ccpPy.git
cd ccpPy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull qwen3:4b
```

## Quick Start

### 1. Configure Your Project

Edit `config.yaml` to point to your C/C++ source:

```yaml
source:
  source_path: "/path/to/your/c/project"
  language: "c"  # or "cpp" or "auto"
```

### 2. Run the Pipeline

```bash
# Phase 1: Parse source code into AST
python scripts/01_parse_c_code.py

# Phase 2: Build dependency graph
python scripts/02_build_graph.py

# Phase 3: Create semantic index
python scripts/03_index_code.py

# Phase 4: Translate to Python
python scripts/04_translate.py --limit 10
```

## CLI Reference

### Parse C/C++ Code

```bash
python scripts/01_parse_c_code.py [OPTIONS]

Options:
  --lang LANG      Language to parse (c, cpp, auto)
  --force          Force re-parsing of all files
  --verbose        Enable debug logging
  --profile NAME   Use a project profile from profiles/
```

### Build Dependency Graph

```bash
python scripts/02_build_graph.py [OPTIONS]

Options:
  --lang LANG        Language to analyze
  --analyze-only     Only show analysis, don't save
  --output-format    Output format (pickle, json, both)
  --verbose          Enable debug logging
```

### Create Semantic Index

```bash
python scripts/03_index_code.py [OPTIONS]

Options:
  --lang LANG      Language to index
  --rebuild        Force rebuild of index
  --query TEXT     Test query against index
  --top-k N        Number of results for test query
  --verbose        Enable debug logging
```

### Translate Code

```bash
python scripts/04_translate.py [OPTIONS]

Options:
  --limit N          Translate only N functions
  --function NAME    Translate specific function
  --dry-run          Preview without translating
  --verbose          Enable debug logging
  --debug            Save failed translations for analysis
  --no-leaves        Skip leaf node priority ordering
  --continue         Resume from previous translation
  --output-dir DIR   Custom output directory
```

## Architecture

```
C/C++ Source -> AST Parser -> Dependency Graph -> Vector Store -> RAG Translator -> Python Output
                    |              |                   |                |
              tree-sitter      networkx           FAISS +          Ollama LLM
                                              sentence-transformers
```

### Pipeline Phases

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | AST Parser | Extracts functions, structs, includes, call relationships |
| 2 | Dependency Graph | Builds call graphs, determines translation order |
| 3 | Semantic Index | Creates embeddings for code similarity search |
| 4 | Translation | RAG-powered translation with iterative refinement |
| 5 | Validation | Ensures output quality and correctness |

## How It Works

This project demonstrates how commercial "unlimited context" translation tools work:

1. **Smart Chunking**: Code is parsed into function-level units using tree-sitter AST parsing
2. **Dependency Ordering**: Translation follows the call graph (leaves first) to ensure dependencies are translated before dependents
3. **Semantic Retrieval**: Similar code patterns provide few-shot examples via FAISS similarity search
4. **Iterative Refinement**: Failed translations receive validation feedback and retry (up to 3 attempts)
5. **Translation Memory**: Maintains C-to-Python entity mappings for consistency

## Configuration

All settings are centralized in `config.yaml`:

| Section | Key Settings |
|---------|--------------|
| `source` | Source path, language (c/cpp/auto), exclude patterns |
| `llm` | Backend (ollama), model (qwen3:4b), temperature, max tokens |
| `embeddings` | Model (all-MiniLM-L6-v2), chunk size, batch size |
| `translation` | Max iterations, example count, Python style preferences |
| `validation` | Syntax check, type check, linting options |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `C2PY_CONFIG` | Path to config file | `config.yaml` |

## Library Mappings

Built-in mappings for common C libraries:

| C Library | Python Equivalent |
|-----------|-------------------|
| `stdlib.h` | Python builtins, `os`, `sys` |
| `ncurses.h` | `curses`, `rich`, `textual` |
| `pthread.h` | `threading`, `concurrent.futures` |
| `string.h` | Python string methods |
| `math.h` | `math` module |

Custom mappings can be added in `mappings/` directory.

## Project Structure

```
c2py/
├── src/
│   ├── parser/        # tree-sitter AST parsing (C and C++)
│   ├── analysis/      # Dependency graph building
│   ├── indexing/      # Semantic embeddings & FAISS
│   ├── translation/   # RAG translation engine
│   ├── validation/    # Python code validation
│   └── utils/         # Config, logging, CLI utilities
├── scripts/           # CLI pipeline tools
├── profiles/          # Project-specific configurations
├── mappings/          # Custom library mappings
├── tests/             # Unit tests
├── data/              # Generated cache (gitignored)
└── output/            # Translated Python (gitignored)
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src tests/

# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Type checking
mypy src/
```

## Limitations

- Complex macro expansion may require manual adjustment
- Platform-specific code needs review
- Pointer arithmetic is translated to Python idioms (verify correctness)
- Performance-critical sections may need optimization
- Deeply interconnected code may exceed context window

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Author

Lex Gaines

## Acknowledgments

- [tree-sitter](https://tree-sitter.github.io/) for robust C/C++ parsing
- [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://faiss.ai/) for efficient similarity search
- [Ollama](https://ollama.ai/) for local LLM inference
- [NetworkX](https://networkx.org/) for dependency graph analysis
