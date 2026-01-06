# Translator Generalization Plan

This document tracks the implementation of generalizing the RAG-based C-to-Python translator to support any C/C++ codebase.

## Current State

The translator is already largely generic after commit `93e4daa`. However, several areas need attention for full generalization.

---

## Phase 1: Clean Up Remaining htop References (Low Effort)

**Status**: [x] Complete

### Tasks
- [x] Fix `scripts/02_build_graph.py:63` - htop-specific error message
- [x] Fix `scripts/02_build_graph.py:179` - `htop_path` config reference
- [x] Fix `scripts/03_index_code.py:69` - htop-specific error message
- [x] Fix `scripts/03_index_code.py:227-233` - hardcoded htop test functions (now dynamically selects from indexed functions)
- [x] Fix `src/utils/config.py:31` - htop docstring example

---

## Phase 2: Add C++ Support (Medium Effort)

**Status**: [x] Complete

### Tasks
- [x] Create `src/parser/parser_factory.py` - unified parser factory for C/C++
- [x] Create `src/parser/cpp_parser.py` with C++ constructs (classes, namespaces, templates)
- [x] Add `tree-sitter-cpp` dependency to `requirements.txt`
- [x] Update `config.yaml` for C++ file extensions and header language setting (done in Phase 5)

### Implementation Summary

**`src/parser/parser_factory.py`**:
- `ParserFactory` class with automatic language detection
- Extension-to-language mapping (.c, .cpp, .cxx, .cc, .hpp, etc.)
- Configurable .h file handling (C or C++ mode)
- Lazy-loaded parser instances
- `create_parser_from_config()` helper function

**`src/parser/cpp_parser.py`** (~650 lines):
- `CppParser` class using tree-sitter-cpp
- New dataclasses:
  - `MethodInfo`: Class methods with access, virtual, static, const flags
  - `ClassInfo`: Classes with base classes, methods, members, nested classes
  - `NamespaceInfo`: Namespace definitions with contents
  - `TemplateInfo`: Template declarations with parameters
  - `CppFileInfo`: Complete C++ file parsed data
- Extracts: functions, classes, structs, namespaces, templates, includes, using declarations
- Handles: constructors, destructors, operator overloading, access specifiers

**`requirements.txt`** updated:
- Added `tree-sitter-cpp==0.21.0`

### Dataclasses Implemented

```python
@dataclass
class MethodInfo:
    name: str
    return_type: str
    parameters: List[Dict[str, str]]
    calls: List[str]
    access: str  # "public", "private", "protected"
    is_virtual: bool
    is_static: bool
    is_const: bool
    is_constructor: bool
    is_destructor: bool
    start_line: int
    end_line: int
    body: str

@dataclass
class ClassInfo:
    name: str
    base_classes: List[Dict[str, str]]  # {'name': name, 'access': access}
    methods: List[MethodInfo]
    members: List[Dict[str, str]]  # {'name', 'type', 'access'}
    nested_classes: List[str]
    is_struct: bool
    file_path: str
    start_line: int
    end_line: int

@dataclass
class NamespaceInfo:
    name: str
    functions: List[str]
    classes: List[str]
    namespaces: List[str]  # Nested namespaces
    file_path: str
    start_line: int
    end_line: int

@dataclass
class TemplateInfo:
    name: str
    template_params: List[Dict[str, str]]  # {'name', 'type'}
    kind: str  # "function", "class", "variable"
    file_path: str
    start_line: int
    end_line: int
```

---

## Phase 3: Extensible Configuration System (Medium Effort)

**Status**: [x] Complete

### Tasks
- [x] Create `src/utils/project_profile.py` - project profile management
- [x] Create `profiles/` directory structure
- [x] Create example profile: `profiles/example.yaml`
- [x] Integrate profile loading into main config system

### Implementation Summary

Created a complete project profile system:

**`src/utils/project_profile.py`**:
- `ProjectProfile` dataclass with all planned fields plus `cpp_settings` and `docstring_style`
- `ProfileManager` class for loading/saving/managing profiles
- `create_profile_from_config()` helper for migration from config-only setups
- Helper methods: `is_cpp_enabled()`, `get_file_extensions()`, `get_source_path()`

**`src/utils/config.py`** extended with:
- `load_profile(name)` - load and apply a profile
- `get_profile()` - get active profile
- `get_available_profiles()` - list all profiles
- `has_profile()` - check if profile is active
- `get_effective_value()` - get value preferring profile over config

**`profiles/example.yaml`**:
- Template profile with all configurable options documented

### ProjectProfile Dataclass (Implemented)
```python
@dataclass
class ProjectProfile:
    name: str
    source_path: str
    language: str = "c"  # "c", "cpp", or "mixed"
    include_dirs: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    library_mappings: Dict[str, str] = field(default_factory=dict)
    target_python_style: str = "modern"
    use_dataclasses: bool = True
    use_type_hints: bool = True
    docstring_style: str = "google"  # "google", "numpy", "sphinx"
    naming_conventions: Dict[str, str] = field(default_factory=dict)
    cpp_settings: Dict[str, Any] = field(default_factory=lambda: {...})
```

---

## Phase 4: Pluggable Library Mapping System (High Effort)

**Status**: [x] Complete

### Tasks
- [x] Create `src/translation/library_mapper.py` with:
  - [x] `LibraryMapping` and `TypeMapping` dataclasses
  - [x] `LibraryMappingProvider` abstract base class
  - [x] `StandardLibraryMapper` - C stdlib mappings (100+ functions, 30+ types)
  - [x] `NCursesMapper` - ncurses to curses mappings (60+ functions)
  - [x] `PthreadMapper` - pthread to threading mappings (25+ functions)
  - [x] `LibraryMapperRegistry` - provider discovery and lookup
  - [x] `CustomMappingProvider` - YAML-based custom mappings
- [x] Create `mappings/` directory for custom mapping files
- [x] Integrate library mapper into `prompt_builder.py`

### Implementation Summary

**`src/translation/library_mapper.py`** (~750 lines):
- `LibraryMapping` dataclass: c_library, c_function, python_module, python_function, signature_transform, notes
- `TypeMapping` dataclass: c_type, python_type, import_required, transform, notes
- `LibraryMappingProvider` ABC with get_function_mapping(), get_type_mapping()
- `StandardLibraryMapper`: Comprehensive C stdlib coverage (stdio, stdlib, string, math, ctype, time, assert)
- `NCursesMapper`: Full ncurses API mapping to Python curses
- `PthreadMapper`: POSIX threads to Python threading (threads, mutexes, conditions, semaphores, barriers)
- `CustomMappingProvider`: Load custom mappings from YAML files
- `LibraryMapperRegistry`: Config-aware provider management with format_hints_for_prompt()

**`mappings/example.yaml`**: Template for custom project mappings

**`src/translation/prompt_builder.py`** updated:
- Added `library_mapper` parameter to constructor
- Added `_format_library_hints()` method
- Hints automatically included in translation prompts

### Custom Mapping YAML Format
```yaml
name: my_project_mappings
functions:
  - c_library: mylib.h
    c_function: my_init
    python_module: myproject
    python_function: initialize
    notes: "Project-specific initialization"

types:
  - c_type: MyStruct*
    python_type: MyClass
    import_required: myproject.types
```

---

## Phase 5: Enhanced Configuration (Low-Medium Effort)

**Status**: [x] Complete

### Tasks
- [x] Extend `config.yaml` schema with:
  - [x] `project` section (name, profile reference)
  - [x] `cpp` section (parse_templates, parse_namespaces, include_std_library)
  - [x] `translation.python_style` (use_type_hints, use_dataclasses, docstring_style, line_length, python_version)
  - [x] `library_mappings` section (enabled, builtin_mappers, custom_mappings)
  - [x] `llm.backend` support for multiple providers (ollama, openai, anthropic)

### Implementation Summary

**`config.yaml`** extended with new sections:

```yaml
project:
  name: "c-translator"
  profile: null  # Optional profile to load

source:
  language: "c"  # "c", "cpp", or "auto"

parsing:
  extensions:
    c: [".c", ".h"]
    cpp: [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"]
  header_language: "c"

cpp:
  parse_templates: true
  parse_namespaces: true
  include_std_library: false

translation:
  python_style:
    use_type_hints: true
    use_dataclasses: true
    docstring_style: "google"
    line_length: 88
    python_version: "3.9"

library_mappings:
  enabled: true
  builtin_mappers: ["stdlib", "ncurses", "pthread"]
  custom_mappings: []

llm:
  backend: "ollama"  # "ollama", "openai", "anthropic"
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen3:4b"
  openai:
    model: "gpt-4"
  anthropic:
    model: "claude-3-sonnet-20240229"
```

**Code updates**:
- `scripts/01_parse_c_code.py`: Updated to handle nested `parsing.extensions` format
- `scripts/04_translate.py`: Updated to use backend-specific LLM config
- `src/utils/project_profile.py`: Updated to use new config paths

---

## Phase 6: CLI Improvements (Low Effort)

**Status**: [x] Complete

### Tasks
- [x] Create `scripts/init_project.py` - project initialization wizard
- [x] Add `--profile` flag to existing scripts
- [x] Add `--lang` flag for C/C++ selection
- [x] Create `src/utils/cli.py` - shared CLI utilities
- [x] Update `src/utils/config.py` - CLI override support
- [x] Create `scripts/run_pipeline.py` - full pipeline runner

### Implementation Summary

All 8 CLI tasks from CLI-IMPROVEMENT-PLAN.md completed:
1. Created `src/utils/cli.py` with shared CLI utilities
2. Added CLI override methods to `src/utils/config.py`
3. Updated `scripts/01_parse_c_code.py` with --profile, --lang, --force
4. Updated `scripts/02_build_graph.py` with --profile, --output-format, --analyze-only
5. Updated `scripts/03_index_code.py` with --profile, --rebuild, --query
6. Updated `scripts/04_translate.py` with --profile, --function, --output-dir, --continue
7. Created `scripts/init_project.py` - interactive project wizard
8. Created `scripts/run_pipeline.py` - full pipeline runner

---

## Implementation Priority

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|--------------|
| Phase 1 | High | Low (1-2 hours) | None |
| Phase 2 | High | Medium (1-2 days) | Phase 1 |
| Phase 3 | Medium | Medium (1 day) | Phase 1 |
| Phase 4 | High | High (2-3 days) | Phases 1, 3 |
| Phase 5 | Medium | Low (0.5 day) | Phase 3 |
| Phase 6 | Low | Low (0.5 day) | Phases 3, 5 |

**Recommended Order**: 1 → 2 → 3 → 5 → 4 → 6

---

## Testing Strategy

### Unit Tests to Add
1. **Parser Tests** - C file parsing, C++ file parsing, mixed codebase detection
2. **Library Mapper Tests** - standard library mappings, custom mapping loading, type mappings
3. **Profile Tests** - profile loading/saving, profile merging with global config
4. **Integration Tests** - end-to-end translation of sample C and C++ code

---

## Migration Path for Existing Users

1. Backup existing data: `cp -r data/ data_backup/`
2. Update `config.yaml`: Add new sections, keep `source.source_path`
3. Re-run Phase 1-3: Parse with new parser
4. Continue with Phase 4: Existing translations preserved
