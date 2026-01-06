#!/usr/bin/env python3
"""
Phase 4: C-to-Python Translation using LLM with RAG.

This script:
1. Loads dependency graph and vector store
2. Connects to Ollama LLM
3. Translates functions in dependency order (leaves first)
4. Validates translations
5. Saves results and statistics
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from translation.llm_client import OllamaClient
from translation.prompt_builder import PromptBuilder
from translation.translator import FunctionTranslator, TranslationResult
from validation.validator import CodeValidator
from indexing.vector_store import VectorStore
from utils.config import Config
from utils.logger import setup_logger
from utils.cli import (
    create_base_parser,
    load_config_from_args,
    handle_list_profiles,
    add_common_arguments,
    add_list_profiles_argument,
    error_exit
)


class TranslationOrchestrator:
    """Orchestrate the translation of C codebase to Python."""
    
    def __init__(self, config: Config, logger, debug_mode: bool = False):
        """Initialize translation orchestrator with configuration."""
        self.config = config
        self.logger = logger
        self.debug_mode = debug_mode
        
        # Get paths
        self.cache_dir = Path(config.get('output.ast_cache'))
        self.graph_dir = Path(config.get('output.graphs'))
        self.embeddings_dir = Path(config.get('output.embeddings'))
        self.output_dir = Path(config.get('output.python_output'))
        self.memory_dir = Path(config.get('output.translation_memory'))
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.llm_client = None
        self.prompt_builder = None
        self.validator = None
        self.translator = None
        self.vector_store = None
        
        # Data
        self.functions = {}
        self.translation_order = []
        self.translated_functions = set()

    def set_output_dir(self, output_dir: Path) -> None:
        """Override the output directory."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing components...")
        
        try:
            # 1. Load vector store
            self.logger.info("Loading vector store...")
            self.vector_store = VectorStore.load(self.embeddings_dir)
            
            # 2. Initialize LLM client
            llm_config = self.config.get('llm')
            backend = llm_config.get('backend', 'ollama')
            self.logger.info(f"Connecting to LLM backend: {backend}...")

            # Get backend-specific settings
            if backend == 'ollama':
                backend_config = llm_config.get('ollama', {})
                base_url = backend_config.get('base_url', 'http://localhost:11434')
                model = backend_config.get('model', 'qwen3:4b')
            else:
                # For future backends, use top-level defaults
                base_url = 'http://localhost:11434'
                model = llm_config.get(backend, {}).get('model', 'qwen3:4b')

            self.llm_client = OllamaClient(
                base_url=base_url,
                model=model,
                temperature=llm_config.get('temperature', 0.2)
            )
            
            if not self.llm_client.check_connection():
                self.logger.error("Failed to connect to Ollama. Is it running?")
                return False
            
            self.logger.info(f"✓ Connected to Ollama (model: {self.llm_client.model})")
            
            # 3. Initialize prompt builder
            trans_config = self.config.get('translation')
            self.prompt_builder = PromptBuilder(
                include_examples=trans_config.get('include_examples', True),
                num_examples=trans_config.get('num_examples', 3)
            )
            
            # 4. Initialize validator
            val_config = self.config.get('validation')
            self.validator = CodeValidator(
                require_type_hints=val_config.get('check_types', True),
                require_docstring=True,
                min_quality_score=val_config.get('min_quality_score', 6.0)
            )
            
            # 5. Initialize translator
            self.translator = FunctionTranslator(
                llm_client=self.llm_client,
                prompt_builder=self.prompt_builder,
                validator=self.validator,
                vector_store=self.vector_store,
                max_iterations=trans_config.get('max_iterations', 3),
                debug_mode=self.debug_mode,
                logger=self.logger
            )
            
            # Load existing translation memory if available
            memory_path = self.memory_dir / 'translations.json'
            if memory_path.exists():
                self.translator.load_translation_memory(memory_path)
                self.logger.info(f"Loaded existing translation memory")
            
            self.logger.info("✓ All components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def load_data(self) -> bool:
        """
        Load parsed function data and translation order.
        
        Returns:
            True if data loaded successfully
        """
        self.logger.info("Loading parsed data and dependency information...")
        
        try:
            # Load parse summary
            summary_path = self.cache_dir / 'parse_summary.json'
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Build function index
            for result in summary['results']:
                if not result['success']:
                    continue
                
                for func in result['data']['functions']:
                    func_name = func['name']
                    self.functions[func_name] = func
            
            self.logger.info(f"Loaded {len(self.functions)} functions")
            
            # Load translation order
            order_path = self.graph_dir / 'translation_order.json'
            with open(order_path, 'r') as f:
                order_data = json.load(f)
                self.translation_order = order_data['order']
            
            self.logger.info(f"Loaded translation order ({len(self.translation_order)} functions)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}", exc_info=True)
            return False
    
    def translate_functions(
        self,
        limit: Optional[int] = None,
        start_with_leaves: bool = True,
        specific_functions: Optional[List[str]] = None,
        continue_session: bool = False
    ) -> List[TranslationResult]:
        """
        Translate functions in dependency order.

        Args:
            limit: Maximum number of functions to translate
            start_with_leaves: Start with leaf nodes (no dependencies)
            specific_functions: List of specific function names to translate
            continue_session: Continue from last session (skip already translated)

        Returns:
            List of TranslationResults
        """
        # If specific functions requested, use those
        if specific_functions:
            funcs_to_translate = []
            for func_name in specific_functions:
                if func_name in self.functions:
                    funcs_to_translate.append(func_name)
                else:
                    self.logger.warning(f"Function not found: {func_name}")
        else:
            # Determine which functions to translate
            if start_with_leaves:
                # Filter to leaf functions
                graph_path = self.graph_dir / 'graph_analysis.json'
                with open(graph_path, 'r') as f:
                    analysis = json.load(f)

                # Functions with no outgoing edges (leaves)
                # For simplicity, use first N functions from translation order
                funcs_to_translate = self.translation_order[:limit] if limit else self.translation_order
            else:
                funcs_to_translate = self.translation_order[:limit] if limit else self.translation_order

        # Filter out already translated (if continue mode or already done)
        if continue_session:
            # Load previously translated functions from translation memory
            memory_funcs = set(self.translator.translation_memory.keys()) if self.translator.translation_memory else set()
            funcs_to_translate = [
                f for f in funcs_to_translate
                if f in self.functions and f not in memory_funcs
            ]
            self.logger.info(f"Continuing session: {len(memory_funcs)} already translated")
        else:
            funcs_to_translate = [
                f for f in funcs_to_translate
                if f in self.functions and f not in self.translated_functions
            ]

        if limit and not specific_functions:
            funcs_to_translate = funcs_to_translate[:limit]
        
        self.logger.info(f"Translating {len(funcs_to_translate)} functions...")
        
        results = []
        
        # Translate with progress bar
        for func_name in tqdm(funcs_to_translate, desc="Translating functions"):
            func_data = self.functions[func_name]
            
            self.logger.info(f"Translating: {func_name}")
            
            try:
                result = self.translator.translate_function(func_data)
                results.append(result)
                
                if result.success:
                    self.translated_functions.add(func_name)
                    self.logger.info(
                        f"✓ {func_name} - Quality: {result.validation.quality_score:.1f}/10 "
                        f"(attempts: {result.attempts})"
                    )
                else:
                    self.logger.warning(
                        f"✗ {func_name} - Failed: {result.error_message}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error translating {func_name}: {e}", exc_info=True)
                continue
        
        return results
    
    def save_results(self, results: List[TranslationResult]) -> None:
        """
        Save translation results to disk.
        
        Args:
            results: List of translation results
        """
        self.logger.info("Saving translation results...")
        
        # Save individual translations
        for result in results:
            if result.success:
                # Determine output file based on source
                func_data = self.functions[result.function_name]
                source_file = Path(func_data['file_path']).stem
                
                # Group by source file
                output_file = self.output_dir / f"{source_file}.py"
                
                # Append translation
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n# Translated from {func_data['file_path']}\n")
                    f.write(f"# Quality score: {result.validation.quality_score:.1f}/10\n")
                    f.write(result.python_code)
                    f.write("\n")
        
        # Save translation memory
        memory_path = self.memory_dir / 'translations.json'
        self.translator.save_translation_memory(memory_path)
        self.logger.info(f"Saved translation memory to {memory_path}")
        
        # Save statistics
        self._save_statistics(results)
    
    def _save_statistics(self, results: List[TranslationResult]) -> None:
        """Save translation statistics."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_attempted': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'average_quality': (
                sum(r.validation.quality_score for r in successful) / len(successful)
                if successful else 0
            ),
            'average_attempts': (
                sum(r.attempts for r in results) / len(results)
                if results else 0
            ),
            'failed_functions': [r.function_name for r in failed]
        }
        
        stats_path = self.output_dir / 'translation_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved statistics to {stats_path}")
    
    def print_summary(self, results: List[TranslationResult]) -> None:
        """Print translation summary."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print("\n" + "="*60)
        print("TRANSLATION SUMMARY")
        print("="*60)
        print(f"\nTotal functions:  {len(results)}")
        print(f"Successful:       {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed:           {len(failed)}")
        
        if successful:
            avg_quality = sum(r.validation.quality_score for r in successful) / len(successful)
            avg_attempts = sum(r.attempts for r in successful) / len(successful)
            print(f"\nAverage quality:  {avg_quality:.1f}/10.0")
            print(f"Average attempts: {avg_attempts:.1f}")
        
        if failed:
            print(f"\nFailed functions:")
            for r in failed[:5]:
                print(f"  - {r.function_name}: {r.error_message}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
        
        print(f"\nOutput directory: {self.output_dir}")
        print("="*60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Translate C/C++ code to Python using LLM with RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add common CLI arguments (--profile, --lang, --config, --verbose)
    add_common_arguments(parser)
    add_list_profiles_argument(parser)

    # Translation-specific arguments
    parser.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='Limit number of functions to translate'
    )
    parser.add_argument(
        '--no-leaves',
        action='store_true',
        help="Don't prioritize leaf nodes"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be translated without translating'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save failed translations to debug/ folder'
    )
    parser.add_argument(
        '--function', '-f',
        action='append',
        metavar='NAME',
        help='Translate specific function(s) by name (can be used multiple times)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        metavar='PATH',
        help='Override output directory for translated Python files'
    )
    parser.add_argument(
        '--continue',
        dest='continue_session',
        action='store_true',
        help='Continue from last translation session (skip already translated)'
    )

    return parser.parse_args()


def main():
    """Main entry point for translation."""
    args = parse_args()

    # Handle --list-profiles
    if handle_list_profiles(args, project_root):
        return 0

    # Load configuration with CLI args
    config, logger = load_config_from_args(args, project_root, "translator")

    logger.info("=" * 60)
    logger.info("Starting C/C++-to-Python translation")
    logger.info("=" * 60)

    # Show configuration
    logger.info(f"Source: {config.get('source.source_path')}")
    if args.function:
        logger.info(f"Translating specific functions: {', '.join(args.function)}")
    if args.continue_session:
        logger.info("Continuing from last session")

    try:
        # Create orchestrator
        orchestrator = TranslationOrchestrator(config, logger, debug_mode=args.debug)

        # Override output directory if specified
        if args.output_dir:
            orchestrator.set_output_dir(Path(args.output_dir))
            logger.info(f"Output directory: {args.output_dir}")

        # Initialize
        if not orchestrator.initialize():
            error_exit("Initialization failed!")

        # Load data
        if not orchestrator.load_data():
            error_exit("Failed to load data!")

        if args.dry_run:
            if args.function:
                funcs = [f for f in args.function if f in orchestrator.functions]
                print(f"Would translate {len(funcs)} specific functions: {', '.join(funcs)}")
            else:
                print(f"Would translate {args.limit or 'all'} functions")
            return 0

        # Translate
        results = orchestrator.translate_functions(
            limit=args.limit,
            start_with_leaves=not args.no_leaves,
            specific_functions=args.function,
            continue_session=args.continue_session
        )

        if not results:
            logger.info("No functions to translate")
            return 0

        # Save results
        orchestrator.save_results(results)

        # Print summary
        orchestrator.print_summary(results)

        logger.info("Translation complete!")

        return 0

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

