# CLI Improvement Plan - Phase 6

This document outlines the implementation plan for Phase 6: CLI Improvements for the C/C++ to Python translator.

## Current State

**Current CLI Scripts:**
| Script | Current CLI Support |
|--------|---------------------|
| `01_parse_c_code.py` | No CLI arguments |
| `02_build_graph.py` | No CLI arguments |
| `03_index_code.py` | No CLI arguments |
| `04_translate.py` | `--limit`, `--no-leaves`, `--dry-run`, `--verbose`, `--debug` |

**Available Infrastructure:**
- `src/utils/config.py` - Config class with `load_profile(name)` support
- `src/utils/project_profile.py` - ProjectProfile dataclass and ProfileManager
- `src/parser/parser_factory.py` - ParserFactory with language detection
- `profiles/` directory with example.yaml template

---

## Implementation Tasks

### Task 1: Create `src/utils/cli.py` - Shared CLI Utilities

**Status**: [x] Complete

**Purpose:** Centralize common CLI argument parsing and configuration loading.

**Features:**
- Standard argument definitions (`--profile`, `--lang`, `--verbose`, `--config`)
- `add_common_arguments(parser)` - add standard args to any ArgumentParser
- `load_config_from_args(args, project_root)` - create configured Config from CLI args
- `list_available_profiles(project_root)` - list all profiles
- `print_profile_info(profile)` - formatted profile output

**Implementation:**

```python
"""
Shared CLI utilities for all scripts.
"""

import argparse
from pathlib import Path
from typing import Tuple
import logging

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments used by all pipeline scripts."""
    parser.add_argument(
        '--profile', '-p',
        help='Project profile to use (from profiles/*.yaml)'
    )
    parser.add_argument(
        '--lang', '-l',
        choices=['c', 'cpp', 'mixed'],
        help='Override source language setting'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to config.yaml (default: config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

def load_config_from_args(args, project_root: Path) -> Tuple:
    """
    Load configuration based on CLI arguments.

    Priority: CLI --lang > Profile > config.yaml
    """
    from .config import Config
    from .logger import setup_logger

    config_path = project_root / args.config

    # Load config, optionally with profile
    profile_name = getattr(args, 'profile', None)
    config = Config(str(config_path), profile_name=profile_name)

    # Override language if specified
    if getattr(args, 'lang', None):
        config.set('source.language', args.lang)

    # Setup logging
    log_level = 'DEBUG' if args.verbose else config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', 'logs/translator.log')
    logger = setup_logger("cli", level=log_level, log_file=log_file)

    return config, logger
```

---

### Task 2: Update `src/utils/config.py` - CLI Override Support

**Status**: [x] Complete

**Changes:**
- Add `apply_cli_overrides(overrides)` method
- Add `get_value_source(key)` method for debugging

**Add methods:**

```python
def apply_cli_overrides(self, overrides: Dict[str, Any]) -> None:
    """Apply CLI argument overrides (highest priority)."""
    if not hasattr(self, '_cli_overrides'):
        self._cli_overrides = {}
    for key, value in overrides.items():
        if value is not None:
            self.set(key, value)
            self._cli_overrides[key] = value

def get_value_source(self, key: str) -> str:
    """Get the source of a configuration value: 'cli', 'profile', or 'config'."""
    if hasattr(self, '_cli_overrides') and key in self._cli_overrides:
        return 'cli'
    if self._profile:
        return 'profile'
    return 'config'
```

---

### Task 3: Update `scripts/01_parse_c_code.py`

**Status**: [x] Complete

**New arguments:**
- `--profile`, `-p` - Project profile to use
- `--lang`, `-l` - Override language (c, cpp, mixed)
- `--force`, `-f` - Force re-parse (bypass cache)
- `--verbose`, `-v` - Debug logging
- `--list-profiles` - List available profiles and exit

**Key changes:**
- Import and use shared CLI utilities
- Use `ParserFactory` instead of direct `CParser`
- Support mixed C/C++ codebases

**Example usage:**
```bash
# Using default config
python scripts/01_parse_c_code.py

# Using a profile
python scripts/01_parse_c_code.py --profile myproject

# Override language
python scripts/01_parse_c_code.py --profile myproject --lang cpp

# Force re-parse with verbose output
python scripts/01_parse_c_code.py --profile myproject --force --verbose

# List available profiles
python scripts/01_parse_c_code.py --list-profiles
```

---

### Task 4: Update `scripts/02_build_graph.py`

**Status**: [x] Complete

**New arguments:**
- `--profile`, `-p` - Project profile to use
- `--lang`, `-l` - Override language
- `--verbose`, `-v` - Debug logging
- `--output-format` - Graph output format (json, pickle, graphml)
- `--analyze-only` - Only analyze, don't save graphs

**Example usage:**
```bash
python scripts/02_build_graph.py --profile myproject
python scripts/02_build_graph.py --profile myproject --analyze-only --verbose
python scripts/02_build_graph.py --output-format graphml
```

---

### Task 5: Update `scripts/03_index_code.py`

**Status**: [x] Complete

**New arguments:**
- `--profile`, `-p` - Project profile to use
- `--lang`, `-l` - Override language
- `--verbose`, `-v` - Debug logging
- `--rebuild` - Force rebuild embeddings and index
- `--query`, `-q` - Test query: find similar functions
- `--top-k` - Number of similar functions to return

**Example usage:**
```bash
python scripts/03_index_code.py --profile myproject
python scripts/03_index_code.py --profile myproject --rebuild
python scripts/03_index_code.py --query "main" --top-k 10
```

---

### Task 6: Update `scripts/04_translate.py`

**Status**: [x] Complete

**New arguments (in addition to existing):**
- `--profile`, `-p` - Project profile to use
- `--lang`, `-l` - Override language
- `--function`, `-f` - Translate specific function(s) by name
- `--output-dir`, `-o` - Override output directory
- `--continue` - Continue from last translation session

**Example usage:**
```bash
# Full translation with profile
python scripts/04_translate.py --profile myproject --limit 50

# Translate specific functions
python scripts/04_translate.py --profile myproject -f main -f init_screen

# Continue previous session
python scripts/04_translate.py --profile myproject --continue

# Verbose with debug mode
python scripts/04_translate.py --profile myproject --limit 10 --verbose --debug
```

---

### Task 7: Create `scripts/init_project.py` - Project Initialization Wizard

**Status**: [x] Complete

**Purpose:** Interactive wizard to create new project profiles.

**Features:**
- Interactive prompts for project configuration
- Non-interactive mode with CLI flags
- Auto-detection of language from file extensions
- Validation of source path
- Option to set as default in config.yaml

**Arguments:**
- `--name`, `-n` - Project name
- `--source`, `-s` - Path to C/C++ source code
- `--lang`, `-l` - Source language (c, cpp, mixed, auto)
- `--interactive`, `-i` - Force interactive mode
- `--set-default` - Set this profile as default in config.yaml

**Example usage:**
```bash
# Interactive mode
python scripts/init_project.py

# Non-interactive with flags
python scripts/init_project.py --name myproject --source /path/to/code --lang cpp

# With auto-detection and set as default
python scripts/init_project.py --name myproject --source /path/to/code --lang auto --set-default
```

---

### Task 8: Create `scripts/run_pipeline.py` - Full Pipeline Runner (Optional)

**Status**: [x] Complete

**Purpose:** Run all phases (parse, graph, index, translate) in sequence.

**Arguments:**
- `--profile`, `-p` (required) - Project profile to use
- `--lang`, `-l` - Override source language
- `--limit` - Limit functions to translate
- `--skip-parse` - Skip Phase 1
- `--skip-graph` - Skip Phase 2
- `--skip-index` - Skip Phase 3
- `--verbose`, `-v` - Verbose output
- `--dry-run` - Show what would be translated

**Example usage:**
```bash
# Run complete pipeline
python scripts/run_pipeline.py --profile myproject

# Skip parsing (use cached)
python scripts/run_pipeline.py --profile myproject --skip-parse

# Limit translation
python scripts/run_pipeline.py --profile myproject --limit 100 --verbose

# Dry run
python scripts/run_pipeline.py --profile myproject --dry-run
```

---

## Summary of Files

### New Files to Create
| File | Purpose |
|------|---------|
| `src/utils/cli.py` | Shared CLI utilities |
| `scripts/init_project.py` | Project initialization wizard |
| `scripts/run_pipeline.py` | Full pipeline runner (optional) |

### Files to Modify
| File | Changes |
|------|---------|
| `src/utils/config.py` | Add CLI override methods |
| `scripts/01_parse_c_code.py` | Add `--profile`, `--lang`, `--force`, use ParserFactory |
| `scripts/02_build_graph.py` | Add `--profile`, `--lang`, `--output-format` |
| `scripts/03_index_code.py` | Add `--profile`, `--lang`, `--rebuild`, `--query` |
| `scripts/04_translate.py` | Add `--profile`, `--lang`, `--function`, `--output-dir` |

---

## Implementation Order

1. **First:** `src/utils/cli.py` - shared utilities (other scripts depend on this)
2. **Second:** `src/utils/config.py` - add CLI override support
3. **Third:** `scripts/01_parse_c_code.py` - most complex (ParserFactory integration)
4. **Fourth:** `scripts/02_build_graph.py` - simpler changes
5. **Fifth:** `scripts/03_index_code.py` - add query mode
6. **Sixth:** `scripts/04_translate.py` - already has argparse, mostly additions
7. **Seventh:** `scripts/init_project.py` - standalone wizard
8. **Eighth (Optional):** `scripts/run_pipeline.py` - convenience script

---

## Configuration Priority

When CLI arguments, profiles, and config.yaml have conflicting values:

```
CLI flags (--lang, --profile)   [Highest priority]
        ↓
Profile settings (profiles/*.yaml)
        ↓
config.yaml defaults            [Lowest priority]
```

---

## Testing Checklist

After implementation, verify:

- [ ] `--profile` loads correct profile settings
- [ ] `--lang` overrides profile and config language
- [ ] `--verbose` enables DEBUG logging
- [ ] `--list-profiles` shows available profiles
- [ ] `--force` bypasses cache in parsing
- [ ] `--query` works in indexing script
- [ ] `--function` translates specific functions
- [ ] `init_project.py` creates valid profile YAML
- [ ] `run_pipeline.py` executes all phases in order
- [ ] Error messages are helpful when profile not found
