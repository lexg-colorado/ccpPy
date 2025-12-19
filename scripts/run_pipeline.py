#!/usr/bin/env python3
"""
Full Pipeline Runner.

Run all phases of the C/C++ to Python translation pipeline in sequence:
  Phase 1: Parse C/C++ code
  Phase 2: Build dependency graph
  Phase 3: Create semantic index
  Phase 4: Translate to Python
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cli import (
    create_base_parser,
    handle_list_profiles,
    validate_profile_exists,
    print_available_profiles,
    error_exit
)


class PipelineRunner:
    """Run the full translation pipeline."""

    def __init__(
        self,
        profile: Optional[str] = None,
        lang: Optional[str] = None,
        config: str = "config.yaml",
        verbose: bool = False
    ):
        """
        Initialize pipeline runner.

        Args:
            profile: Profile name to use
            lang: Language override
            config: Config file path
            verbose: Enable verbose output
        """
        self.profile = profile
        self.lang = lang
        self.config = config
        self.verbose = verbose
        self.scripts_dir = project_root / "scripts"

    def _build_common_args(self) -> List[str]:
        """Build common arguments for all scripts."""
        args = []

        if self.profile:
            args.extend(['--profile', self.profile])

        if self.lang:
            args.extend(['--lang', self.lang])

        if self.config != "config.yaml":
            args.extend(['--config', self.config])

        if self.verbose:
            args.append('--verbose')

        return args

    def _run_script(
        self,
        script_name: str,
        extra_args: Optional[List[str]] = None,
        description: str = ""
    ) -> bool:
        """
        Run a pipeline script.

        Args:
            script_name: Name of script to run
            extra_args: Additional arguments for the script
            description: Description for logging

        Returns:
            True if successful, False otherwise
        """
        script_path = self.scripts_dir / script_name

        cmd = [sys.executable, str(script_path)]
        cmd.extend(self._build_common_args())

        if extra_args:
            cmd.extend(extra_args)

        print(f"\n{'=' * 60}")
        print(f"Running: {description or script_name}")
        print(f"Command: {' '.join(cmd)}")
        print('=' * 60 + '\n')

        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error running {script_name}: {e}")
            return False

    def run_parse(self, force: bool = False) -> bool:
        """Run Phase 1: Parse C/C++ code."""
        extra_args = ['--force'] if force else []
        return self._run_script(
            "01_parse_c_code.py",
            extra_args,
            "Phase 1: Parsing C/C++ code"
        )

    def run_graph(self, analyze_only: bool = False) -> bool:
        """Run Phase 2: Build dependency graph."""
        extra_args = ['--analyze-only'] if analyze_only else []
        return self._run_script(
            "02_build_graph.py",
            extra_args,
            "Phase 2: Building dependency graph"
        )

    def run_index(self, rebuild: bool = False) -> bool:
        """Run Phase 3: Create semantic index."""
        extra_args = ['--rebuild'] if rebuild else []
        return self._run_script(
            "03_index_code.py",
            extra_args,
            "Phase 3: Creating semantic index"
        )

    def run_translate(
        self,
        limit: Optional[int] = None,
        dry_run: bool = False,
        continue_session: bool = False
    ) -> bool:
        """Run Phase 4: Translate to Python."""
        extra_args = []

        if limit:
            extra_args.extend(['--limit', str(limit)])

        if dry_run:
            extra_args.append('--dry-run')

        if continue_session:
            extra_args.append('--continue')

        return self._run_script(
            "04_translate.py",
            extra_args,
            "Phase 4: Translating to Python"
        )

    def run_full_pipeline(
        self,
        skip_parse: bool = False,
        skip_graph: bool = False,
        skip_index: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False,
        force_reparse: bool = False,
        rebuild_index: bool = False
    ) -> bool:
        """
        Run the full translation pipeline.

        Args:
            skip_parse: Skip Phase 1
            skip_graph: Skip Phase 2
            skip_index: Skip Phase 3
            limit: Limit functions to translate
            dry_run: Show what would be translated
            force_reparse: Force re-parse all files
            rebuild_index: Force rebuild semantic index

        Returns:
            True if all phases succeeded
        """
        start_time = datetime.now()

        print("\n" + "=" * 60)
        print("C/C++ TO PYTHON TRANSLATION PIPELINE")
        print("=" * 60)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.profile:
            print(f"Profile: {self.profile}")
        print("=" * 60)

        phases_run = 0
        phases_success = 0

        # Phase 1: Parse
        if not skip_parse:
            phases_run += 1
            if self.run_parse(force=force_reparse):
                phases_success += 1
            else:
                print("\nPhase 1 failed. Stopping pipeline.")
                return False
        else:
            print("\nSkipping Phase 1: Parse (--skip-parse)")

        # Phase 2: Build Graph
        if not skip_graph:
            phases_run += 1
            if self.run_graph():
                phases_success += 1
            else:
                print("\nPhase 2 failed. Stopping pipeline.")
                return False
        else:
            print("\nSkipping Phase 2: Graph (--skip-graph)")

        # Phase 3: Index
        if not skip_index:
            phases_run += 1
            if self.run_index(rebuild=rebuild_index):
                phases_success += 1
            else:
                print("\nPhase 3 failed. Stopping pipeline.")
                return False
        else:
            print("\nSkipping Phase 3: Index (--skip-index)")

        # Phase 4: Translate
        phases_run += 1
        if self.run_translate(limit=limit, dry_run=dry_run):
            phases_success += 1
        else:
            print("\nPhase 4 failed.")
            return False

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Phases run: {phases_run}")
        print(f"Successful: {phases_success}")
        print(f"Duration: {duration}")
        print("=" * 60 + "\n")

        return phases_success == phases_run


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full C/C++ to Python translation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with a profile
  python scripts/run_pipeline.py --profile myproject

  # Skip parsing (use cached ASTs)
  python scripts/run_pipeline.py --profile myproject --skip-parse

  # Limit translation with verbose output
  python scripts/run_pipeline.py --profile myproject --limit 100 --verbose

  # Dry run to see what would be translated
  python scripts/run_pipeline.py --profile myproject --dry-run

  # Force re-parse and rebuild index
  python scripts/run_pipeline.py --profile myproject --force --rebuild-index
"""
    )

    # Profile is effectively required for meaningful runs
    parser.add_argument(
        '--profile', '-p',
        metavar='NAME',
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
        metavar='PATH',
        help='Path to config.yaml (default: config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List available profiles and exit'
    )

    # Phase skip options
    skip_group = parser.add_argument_group('Skip phases')
    skip_group.add_argument(
        '--skip-parse',
        action='store_true',
        help='Skip Phase 1: Parsing (use cached ASTs)'
    )
    skip_group.add_argument(
        '--skip-graph',
        action='store_true',
        help='Skip Phase 2: Dependency graph building'
    )
    skip_group.add_argument(
        '--skip-index',
        action='store_true',
        help='Skip Phase 3: Semantic indexing'
    )

    # Force rebuild options
    rebuild_group = parser.add_argument_group('Rebuild options')
    rebuild_group.add_argument(
        '--force',
        action='store_true',
        help='Force re-parse all files (bypass cache)'
    )
    rebuild_group.add_argument(
        '--rebuild-index',
        action='store_true',
        help='Force rebuild semantic index'
    )

    # Translation options
    trans_group = parser.add_argument_group('Translation options')
    trans_group.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='Limit number of functions to translate'
    )
    trans_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be translated without translating'
    )

    return parser.parse_args()


def main():
    """Main entry point for pipeline runner."""
    args = parse_args()

    # Handle --list-profiles
    if args.list_profiles:
        print_available_profiles(project_root)
        return 0

    # Validate profile if specified
    if args.profile and not validate_profile_exists(args.profile, project_root):
        print(f"Error: Profile not found: {args.profile}")
        print("\nAvailable profiles:")
        print_available_profiles(project_root)
        return 1

    # Create runner
    runner = PipelineRunner(
        profile=args.profile,
        lang=args.lang,
        config=args.config,
        verbose=args.verbose
    )

    # Run pipeline
    success = runner.run_full_pipeline(
        skip_parse=args.skip_parse,
        skip_graph=args.skip_graph,
        skip_index=args.skip_index,
        limit=args.limit,
        dry_run=args.dry_run,
        force_reparse=args.force,
        rebuild_index=args.rebuild_index
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
