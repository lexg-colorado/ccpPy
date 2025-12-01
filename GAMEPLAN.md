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
- [ ] Build basic dependency graph (moved to Phase 2)

### Phase 2: Translation (Week 2)
- [ ] Implement context retrieval
- [ ] Set up Ollama integration with qwen
- [ ] Build translation orchestrator
- [ ] Translate first module (e.g., Process.c)
- [ ] Validate output

### Phase 3: Scale & Refine (Week 3+)
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

---

*Last Updated: 2025-12-01*