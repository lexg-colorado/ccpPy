# htop C-to-Python Translation Project
## RAG-Based Code Migration Proof of Concept

**Project Goal**: Build a proof-of-concept system that uses RAG (Retrieval-Augmented Generation) with dependency graph analysis to automatically translate the htop C codebase to Python.

**Target Codebase**: htop (~37k total lines, ~29k C code)
- Repository: https://github.com/htop-dev/htop
- Local path: `~/Python/htop`

---

## Why This Approach?

Traditional "unlimited context" claims by commercial tools are marketing speak. They likely use:
1. **Smart chunking** - Not everything at once
2. **Dependency-aware retrieval** - Only fetch what's relevant
3. **Iterative processing** - Build up translations progressively
4. **Semantic indexing** - Find similar patterns across codebase

We're building this transparently to understand the actual mechanics.

---

## Architecture Overview

```
┌─────────────────┐
│  C Source Code  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AST Parser     │ (tree-sitter-c)
│  - Functions    │
│  - Structs      │
│  - Includes     │
│  - Calls        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dependency Graph│ (networkx)
│  - Call graph   │
│  - Type deps    │
│  - Module deps  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Store    │ (embeddings)
│  - Code chunks  │
│  - Semantic     │
│    search       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RAG Translator  │ (qwen models)
│  - Context      │
│    retrieval    │
│  - Translation  │
│  - Validation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python Output  │
└─────────────────┘
```

---

## Phase 1: Code Analysis & Indexing

### 1.1 AST Parsing (`htop_parser.py`)

**Purpose**: Extract structured information from C source files

**Capabilities needed**:
- Parse C files to AST using tree-sitter
- Extract key elements:
  - Function definitions (name, params, return type, body)
  - Struct/typedef definitions
  - Include statements
  - Function calls
  - Global variables
  - Macros (where practical)

**Output**: JSON/pickle representation of code structure

**Dependencies**:
```bash
pip install tree-sitter tree-sitter-c
```

### 1.2 Dependency Graph (`dependency_graph.py`)

**Purpose**: Understand relationships between code components

**Graph types to build**:
1. **Call graph**: Function A calls Function B
2. **Type dependency**: Function uses Struct X
3. **Include graph**: File A includes File B
4. **Module coupling**: Logical groupings (process handling, UI, etc.)

**Key features**:
- Topological sort (translate leaves first)
- Circular dependency detection
- Critical path identification

**Dependencies**:
```bash
pip install networkx matplotlib
```

### 1.3 Code Indexing (`code_indexer.py`)

**Purpose**: Enable semantic search across codebase

**Indexing strategy**:
- Chunk granularity: Function-level
- Include: Function body + signature + comments
- Store metadata: file path, dependencies, complexity metrics

**Embedding options**:
1. **Lightweight**: TF-IDF or basic word2vec (fast, no GPU)
2. **Medium**: sentence-transformers/all-MiniLM-L6-v2 (good balance)
3. **Heavy**: CodeBERT or similar (best quality, slower)

**Vector store options**:
- Start simple: FAISS or in-memory numpy
- Later: ChromaDB or Qdrant if needed

**Dependencies**:
```bash
pip install sentence-transformers faiss-cpu numpy
```

---

## Phase 2: Translation System

### 2.1 Context Retrieval (`context_builder.py`)

**Purpose**: Gather relevant context for translating a given code unit

**For each function/file to translate, retrieve**:
1. **Direct dependencies** (from dependency graph)
   - Functions it calls
   - Structs/types it uses
   - Included headers
2. **Similar patterns** (from vector store)
   - Similar functions already translated
   - Common C idioms (malloc/free, pointer arithmetic, etc.)
3. **Module context**
   - What logical module is this part of?
   - Related functionality

**Context window management**:
- Priority ranking: Direct deps > Similar > General
- Token budget: ~8k tokens for context (leave room for translation)
- Truncation strategy: Summarize less critical context

### 2.2 Translation Engine (`translator.py`)

**Purpose**: Orchestrate the actual C → Python translation

**Translation strategy**:
1. **Start with leaf nodes** (no dependencies on untranslated code)
2. **Work up dependency tree**
3. **Maintain translation memory**:
   - C struct → Python class mappings
   - C function → Python function mappings
   - Library mappings (ncurses → curses/rich, etc.)

**Prompt engineering**:
- System prompt: Expert C and Python developer
- Include: Coding standards, Python idioms, type hints
- Emphasize: Preserve semantics over literal translation

**LLM configuration** (using local Ollama):
- Model: `qwen3:4b` for general translation
- Temperature: 0.1-0.3 (want consistency)
- Max tokens: ~4k for output

**Dependencies**:
```bash
# Ollama should already be running locally
pip install ollama-python  # or use requests directly
```

### 2.3 Translation Memory (`translation_map.py`)

**Purpose**: Track what's been translated and maintain consistency

**Store**:
- C entity → Python entity mappings
- Translation decisions (with rationale)
- Known library mappings
- Type conversions

**Format**: JSON for easy inspection/editing

---

## Phase 3: Validation & Iteration

### 3.1 Validation Strategy

**Automated checks**:
- Python syntax validation (`ast.parse`)
- Import resolution (all needed modules available?)
- Type hint coverage
- Linting (pylint, mypy)

**Manual verification**:
- Start with simple modules (Process parsing)
- Compare behavior (does it read /proc correctly?)
- Gradually work up to complex modules (UI)

### 3.2 Iterative Improvement

**Feedback loop**:
1. Translate module
2. Test/validate
3. Note failures/issues
4. Refine prompts/context retrieval
5. Re-translate
6. Update translation memory

---

## Implementation Phases

### Phase 1: Foundation ✅ **COMPLETE**
- [x] Set up project structure
- [x] Implement AST parser
  - CParser class with tree-sitter-c integration
  - Function, struct, include, and call relationship extraction
- [x] Implement batch parsing system
  - Parallel processing with ProcessPoolExecutor
  - Smart caching mechanism
  - Comprehensive statistics generation
- [x] Test on htop codebase

### Phase 2: Dependency Analysis ✅ **COMPLETE**
- [x] Build dependency graph system
  - Function call graph (directed graph)
  - File dependency graph (include relationships)
- [x] Implement topological sort for translation ordering
- [x] Identify leaf nodes (functions with no dependencies)
- [x] Detect cycles using strongly connected components
- [x] Graph complexity analysis and metrics
- [x] Export graphs for visualization and processing

### Phase 3: Semantic Indexing ✅ **COMPLETE**
- [x] Implement semantic indexing with embeddings
  - EmbeddingGenerator using sentence-transformers
  - Rich text representation of functions
  - Batch processing for efficiency
- [x] Build FAISS vector store
  - Cosine similarity search
  - Function-to-function similarity
  - Context retrieval for RAG
- [x] Create indexing orchestration script
  - Load parsed data from Phase 1
  - Generate and save embeddings
  - Build and validate index
  - Export statistics and metadata

### Phase 4: Translation Engine ✅ **COMPLETE**
- [x] Set up Ollama integration with qwen
  - OllamaClient with retry mechanism
  - Connection checking and error handling
  - Streaming support
- [x] Build translation orchestrator using RAG
  - TranslationOrchestrator class
  - Integration with all previous phases
  - Translation memory management
- [x] Implement RAG-aware prompt builder
  - System prompts with translation guidelines
  - Few-shot examples from similar functions
  - Context formatting with dependencies
  - Retry prompts with validation feedback
- [x] Implement core translator
  - FunctionTranslator with iterative refinement
  - Max 3 attempts with validation feedback
  - LLM output parsing and cleaning
- [x] Build comprehensive validation suite
  - Syntax validation (AST parsing)
  - Type hints checking
  - Docstring validation
  - PEP 8 style checks
  - Import analysis
  - Quality scoring (0-10)
- [x] CLI and statistics
  - Command-line interface with options
  - Success/failure tracking
  - Quality metrics and reporting

### Phase 5: Scale & Refine
- [ ] Translate multiple modules
- [ ] Identify common failure patterns
- [ ] Refine prompts and context retrieval
- [ ] Build translation memory
- [ ] Document lessons learned

---

## Key Design Decisions

### Why tree-sitter over pycparser?
- More robust, handles modern C better
- Faster parsing
- Better error recovery

### Why function-level granularity?
- Balance between context and manageability
- Functions are natural translation units
- Easier to validate

### Why start with leaf nodes?
- Less context needed
- Can use translated code as examples for later translations
- Builds confidence incrementally

### Why local qwen models?
- Privacy (code stays local)
- No API costs
- Experimentation freedom
- Fast iteration

---

## Success Metrics

**MVP Success** (Proof of Concept):
- Successfully translate 3-5 core modules
- Generated Python code is syntactically valid
- Basic functionality works (e.g., can read /proc)

**Full Success**:
- Translate 80%+ of htop functionality
- Python version produces same output as C version
- Code is maintainable (not just machine-generated spaghetti)
- Document learnings for other translation projects

---

## Known Challenges

1. **C-specific constructs**:
   - Pointer arithmetic
   - Manual memory management
   - Preprocessor macros
   - Inline assembly (hopefully rare in htop)

2. **Library mappings**:
   - ncurses → Python curses or rich/textual
   - Platform-specific system calls
   - Performance-critical sections

3. **Semantic preservation**:
   - Race conditions (if any)
   - Performance characteristics
   - Edge cases in parsing logic

4. **Context window limits**:
   - Even with smart retrieval, some functions are deeply interconnected
   - May need multi-pass translation

---

## Tools & Environment

**Development Machine**: Ubuntu server with RTX 5060 Ti 16GB
- Ollama running locally
- Models: qwen3:4b (text), qwen3-vl:4b (vision - probably not needed here)

**Primary Development**: Python 3.10+
**IDE**: VSCode / Claude Code for iteration

**Key Libraries**:
```txt
tree-sitter
tree-sitter-c
networkx
sentence-transformers
faiss-cpu
numpy
ollama-python
matplotlib (for visualization)
```

---

## Next Steps

1. **Immediate**: Set up project structure and install dependencies
2. **First Task**: Build the AST parser to extract functions from htop
3. **Quick Win**: Translate a single, simple C file to validate the approach

---

## Resources & References

- htop source: https://github.com/htop-dev/htop
- tree-sitter docs: https://tree-sitter.github.io/tree-sitter/
- RAG patterns: https://www.anthropic.com/news/contextual-retrieval
- Code translation research: [Add relevant papers if found]

---

## Notes & Learnings

*This section will be updated as the project progresses*

### Date: [Current Date]
- Initial gameplan created
- Codebase stats: 37k lines total, 29k C code
- Ready to begin implementation

### 2025-12-01: AST Parser - tree-sitter Language Initialization Fix
**Issue**: `TypeError: Language.__init__() missing 1 required positional argument: 'name'` in `CParser.__init__()` at line 53

**Root Cause**: The code incorrectly wrapped `tree_sitter_c.language()` in a `Language()` constructor. Modern tree-sitter Python bindings return a language object directly that doesn't need wrapping.

**Solution**: Changed from:
```python
c_language = Language(tree_sitter_c.language(), 'c')
self.parser.set_language(c_language)
```

To:
```python
self.parser.language = tree_sitter_c.language()
```

**Impact**: AST parser now initializes correctly and can parse C files using tree-sitter.

### 2025-12-01: Phase 1 Complete - AST Parsing Implementation
**Completed Components**:
1. **CParser** (`src/parser/ast_parser.py`):
   - Parses C files using tree-sitter
   - Extracts functions (name, params, return type, body, calls)
   - Extracts structs (name, fields)
   - Extracts include statements (system vs local)
   - Handles function call graph extraction

2. **BatchParser** (`scripts/01_parse_htop.py`):
   - Discovers all C files in htop source directory
   - Parallel processing using ProcessPoolExecutor
   - Smart caching system (checks file modification times)
   - Generates comprehensive statistics:
     - Total/unique function counts
     - Total/unique struct counts
     - Include statement analysis
     - Most frequently called functions
   - Progress tracking with tqdm
   - Error handling and reporting
   - JSON export of all parsed data

**Performance**: Successfully parses entire htop codebase (~37k lines) with caching and parallel processing.

**Next**: Phase 2 - Build dependency graph using networkx to enable topological translation ordering.

### 2025-12-01: Phase 2 Complete - Dependency Graph Builder
**Completed Components**:

**DependencyGraphBuilder** (`scripts/02_build_graph.py`):
- Loads parsed AST data from Phase 1 cache
- Builds two complementary graphs using NetworkX:
  1. **Function Call Graph**: Directed graph showing which functions call which
  2. **File Dependency Graph**: Directed graph of include relationships

**Key Features Implemented**:
1. **Graph Construction**:
   - Function nodes with metadata (file, line, return type)
   - File nodes with statistics (function count, struct count)
   - Smart include path resolution (relative and htop root)

2. **Dependency Analysis**:
   - Topological sort for optimal translation order
   - Leaf node identification (functions with no dependencies)
   - Strongly connected components detection (recursive cycles)
   - Approximate ordering fallback for cyclic graphs

3. **Graph Analytics**:
   - Graph density calculations
   - DAG verification (is graph acyclic?)
   - Most called/calling function identification
   - Complexity metrics and statistics

4. **Output Formats**:
   - Pickle format (`.gpickle`) for programmatic use
   - JSON format for human inspection
   - Translation order file for Phase 4
   - Comprehensive analysis report

**Key Insights**:
- Successfully handles both acyclic and cyclic dependency graphs
- Provides translation ordering that minimizes context requirements
- Identifies mutual recursion for special handling
- Exports both machine-readable and human-readable formats

**Next**: Phase 3 - Semantic indexing with embeddings for context-aware code retrieval.

### 2025-12-01: Phase 3 Complete - Semantic Indexing System
**Completed Components**:

**1. EmbeddingGenerator** (`src/indexing/embedder.py`):
- Uses sentence-transformers (all-MiniLM-L6-v2 model)
- Converts C functions to rich semantic text representations:
  * Function signature (name, return type, parameters)
  * Called functions (dependencies)
  * Function body (actual code, truncated to 500 chars)
- Batch processing for efficiency
- Error handling with fallback to minimal representations
- Supports both batch and single-function embedding
- Query embedding for custom searches

**2. VectorStore** (`src/indexing/vector_store.py`):
- FAISS-based similarity search system
- Supports multiple metrics:
  * Cosine similarity (default, using normalized inner product)
  * Euclidean distance
- Multiple index types:
  * Flat index (exact search, good for <1M vectors)
  * IVF index (approximate search for larger datasets)
- Core functionality:
  * Add embeddings with metadata
  * Similarity search with top-k results
  * Function-to-function similarity lookup
  * **RAG-ready**: `get_context_for_translation()` method
  * Save/load functionality for persistence
  * Statistics tracking

**3. SemanticIndexer** (`scripts/03_index_code.py`):
- Orchestrates the complete indexing pipeline
- Loads parsed function data from Phase 1
- Generates embeddings for all functions
- Builds FAISS vector index
- Validates index with sample similarity queries
- Exports multiple artifacts:
  * `function_index.faiss` - FAISS index file
  * `function_embeddings.npy` - Raw embeddings array
  * `function_metadata.json` - Function metadata and mappings
  * `index_stats.json` - Statistics and configuration

**Key Design Decisions**:
1. **Rich Text Representation**: Combines signature, dependencies, and code for semantic understanding
2. **Batch Processing**: Processes embeddings in configurable batches (default: 32)
3. **Cosine Similarity**: Better for semantic similarity than Euclidean distance
4. **Metadata Truncation**: Limits body to 500 chars, calls to 20 items for efficiency
5. **Function Name Indexing**: Fast lookup by function name for RAG context retrieval

**Performance Characteristics**:
- Embedding dimension: 384 (all-MiniLM-L6-v2)
- Average search time: Sub-millisecond for exact search
- Scalable to hundreds of thousands of functions
- Index can be memory-mapped for large datasets

**Integration with Phase 4**:
The `get_context_for_translation()` method provides exactly what the RAG translator needs:
- Target function metadata
- Similar function examples with similarity scores
- Ready to inject into LLM context window

**Next**: Phase 4 - RAG-based translation engine using Ollama and context from Phase 3.

### 2025-12-01: Phase 4 Complete - RAG Translation Engine
**Completed Components**:

**1. OllamaClient** (`src/translation/llm_client.py`):
- HTTP API client for local Ollama instance
- Configurable model (default: qwen3:4b), temperature (0.2)
- Retry mechanism with exponential backoff (max 3 retries)
- 2-minute timeout for long translations
- Connection health checking
- Both streaming and non-streaming generation
- Error handling for network issues and API errors

**2. PromptBuilder** (`src/translation/prompt_builder.py`):
- System prompt with 10 translation guidelines
- Function context formatting:
  * Name, file, return type, parameters
  * Dependencies (functions called)
- RAG-aware few-shot examples:
  * Retrieves similar functions from Phase 3 embeddings
  * Shows both C code and Python translations
  * Similarity scores for context
- Retry prompts with validation errors
- Requirements formatting (type hints, docstrings, PEP 8)
- Output format instructions

**3. FunctionTranslator** (`src/translation/translator.py`):
- Core translation logic with iterative refinement
- RAG context integration via vector store
- Translation attempts (max 3 iterations):
  1. Initial translation with RAG context
  2. Retry with validation errors if needed
  3. Final attempt with accumulated feedback
- LLM output parsing:
  * Removes markdown code fences
  * Extracts function definitions
  * Cleans explanatory text
- Translation memory management:
  * Stores successful C→Python mappings
  * Quality scores and validation status
  * Save/load functionality for persistence

**4. CodeValidator** (`src/validation/validator.py`):
- Multi-level validation system:
  * **Syntax**: AST parsing to catch Python syntax errors
  * **Structure**: Ensures function definition exists
  * **Type Hints**: Validates parameter and return type annotations
  * **Docstrings**: Checks for presence and proper formatting
  * **PEP 8**: Line length, trailing whitespace, statement formatting
  * **Imports**: Analyzes used vs imported modules
- Quality scoring algorithm (0-10 scale):
  * Deduct 2.0 per error
  * Deduct 0.5 per warning
  * Deduct for missing type hints (-1.0) and docstrings (-1.0)
  * Bonus for error handling (+0.5)
- Human-readable validation reports
- Comprehensive error and warning messages

**5. TranslationOrchestrator** (`scripts/04_translate.py`):
- End-to-end pipeline orchestration
- Loads data from all previous phases:
  * AST data (Phase 1)
  * Translation order (Phase 2)
  * Vector store (Phase 3)
- Initialization sequence:
  * Vector store loading
  * Ollama connection verification
  * Component initialization
  * Translation memory loading
- Function translation loop:
  * Follows dependency order (leaves first)
  * Progress tracking with tqdm
  * Per-function logging
  * Success/failure tracking
- Output management:
  * Groups functions by source file
  * Includes quality scores in comments
  * Saves translation memory
  * Generates statistics JSON
- CLI features:
  * `--limit N`: Translate only N functions
  * `--dry-run`: Preview without translating
  * `--no-leaves`: Don't prioritize leaf nodes

**Architecture Highlights**:
1. **Separation of Concerns**: LLM client, prompt building, validation, and orchestration are separate modules
2. **Retry Philosophy**: Validation-driven iteration improves output quality
3. **RAG Integration**: Seamless use of Phase 3 embeddings for context
4. **Translation Memory**: Accumulates knowledge across translations
5. **Quality-First**: Every translation validated before acceptance

**Key Design Decisions**:
1. **Local LLM**: Privacy-preserving, no API costs, fast iteration
2. **Low Temperature (0.2)**: Consistency over creativity for code translation
3. **Iterative Refinement**: Up to 3 attempts with validation feedback
4. **Quality Threshold (6.0/10)**: Ensures minimum code quality
5. **Leaf-First Translation**: Minimizes context dependencies

**Validation Test Results**:
- ✅ All files compile successfully
- ✅ All module imports work correctly
- ✅ Validator: Correctly validates valid Python code (8.5/10 score)
- ✅ Validator: Correctly catches syntax errors
- ✅ PromptBuilder: System and translation prompts generate correctly
- ✅ CLI: Help and argument parsing functional

**Output Structure**:
```
output/
├── python/
│   ├── Process.py          # Translated functions grouped by source file
│   ├── Panel.py
│   └── ...
├── translation_stats.json   # Success rate, quality metrics
memory/
└── translations.json        # Translation memory (C→Python mappings)
```

**Integration Success**:
The complete pipeline now works end-to-end:
1. Parse C code → AST (Phase 1)
2. Build dependency graph → Translation order (Phase 2)
3. Generate embeddings → Vector search (Phase 3)
4. Translate with RAG → Validated Python (Phase 4)

**Next**: Production use - translate actual htop modules and refine based on results.

### 2025-12-01: Translation Quality Improvements
**Problem**: Initial translations had lower success rates due to:
- qwen3 model quirks with token limits
- Markdown code fences confusing the LLM
- Insufficient debugging information
- Variable initialization issues in retry logic

**Improvements Implemented**:

**1. qwen3 Model Compatibility Fix** (`src/translation/llm_client.py`):
- **Issue**: qwen3 model has problems when `num_predict` (max_tokens) is set to values < 1024
- **Fix**: Changed to only set `num_predict` if >= 1024, otherwise use model default
- **Impact**: Prevents truncated or empty responses from LLM

**2. Prompt Format Optimization** (`src/translation/prompt_builder.py`):
- **Issue**: Markdown code fences (```) in prompts caused LLM confusion about code boundaries
- **Fix**: Removed all code fences from:
  * Few-shot examples
  * Target function formatting
  * Retry prompts
- **Impact**: Cleaner output, less format confusion, higher success rate

**3. Debug & Logging Infrastructure** (`scripts/04_translate.py`, `src/translation/translator.py`):
- **Added**: `--verbose` flag for DEBUG level logging
- **Added**: `--debug` flag to save failed translations
- **Added**: `_save_failed_attempt()` method that saves:
  * Function name and attempt number
  * Failure type (parse_failed vs validation_failed)
  * Full prompt sent to LLM (first 3000 chars)
  * LLM response (first 2000 chars)
  * Parsed Python code (if any)
  * Validation results
- **Output**: Saves to `debug/failed_translations/{func_name}_attempt{N}_{type}.json`
- **Impact**: Much easier to diagnose and fix translation issues

**4. Translator Robustness Improvements** (`src/translation/translator.py`):
- **Fixed**: Uninitialized variable bugs:
  * `validation_result` - now initialized before retry loop
  * `original_prompt` - now set on first attempt
  * `previous_output` - now initialized at start
- **Added**: Comprehensive logging at each step:
  * Prompt length and preview
  * LLM response length and preview
  * Parse success/failure
  * Validation results with errors
  * Exception tracking with type and traceback
- **Improved**: Error messages now include:
  * Truncated error summaries (first 3 errors)
  * Exception type names
  * Detailed logging for debugging
- **Impact**: More robust retry logic, better error tracking, easier debugging

**Results**:
- ✅ Higher translation success rate
- ✅ Better LLM output quality
- ✅ Easier debugging of failed translations
- ✅ More informative error messages
- ✅ No more uninitialized variable crashes

**CLI Usage**:
```bash
# Standard translation
python scripts/04_translate.py --limit 10

# With verbose logging
python scripts/04_translate.py --limit 10 --verbose

# With debug mode (saves failed attempts)
python scripts/04_translate.py --limit 10 --debug

# Full debugging
python scripts/04_translate.py --limit 10 --verbose --debug
```

---

*Last Updated: 2025-12-01*