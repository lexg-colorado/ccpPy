"""
Shared CLI utilities for all pipeline scripts.

Provides common argument parsing, configuration loading, and output formatting.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .project_profile import ProjectProfile


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common arguments used by all pipeline scripts.

    Adds:
        --profile, -p: Project profile to use
        --lang, -l: Override language setting (c, cpp, mixed)
        --config, -c: Path to config.yaml
        --verbose, -v: Enable verbose logging
    """
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


def add_list_profiles_argument(parser: argparse.ArgumentParser) -> None:
    """Add --list-profiles argument to a parser."""
    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List available profiles and exit'
    )


def load_config_from_args(
    args: argparse.Namespace,
    project_root: Path,
    logger_name: str = "cli"
) -> Tuple['Config', logging.Logger]:
    """
    Load configuration based on CLI arguments.

    Priority:
    1. CLI --lang overrides profile and config
    2. Profile settings override config.yaml
    3. Config.yaml provides defaults

    Args:
        args: Parsed command-line arguments
        project_root: Path to project root directory
        logger_name: Name for the logger

    Returns:
        Tuple of (Config, Logger)
    """
    from .config import Config
    from .logger import setup_logger

    config_path = project_root / args.config

    # Load config, optionally with profile
    profile_name = getattr(args, 'profile', None)
    if profile_name:
        config = Config(str(config_path), profile_name=profile_name)
    else:
        config = Config(str(config_path))

    # Override language if specified via CLI
    lang = getattr(args, 'lang', None)
    if lang:
        config.set('source.language', lang)

    # Setup logging
    verbose = getattr(args, 'verbose', False)
    log_level = 'DEBUG' if verbose else config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', 'logs/translator.log')
    logger = setup_logger(logger_name, level=log_level, log_file=log_file)

    # Log profile info if using one
    if config.has_profile():
        profile = config.get_profile()
        logger.info(f"Using profile: {profile.name}")
        logger.debug(f"  Source: {profile.source_path}")
        logger.debug(f"  Language: {profile.language}")

    return config, logger


def list_available_profiles(project_root: Path) -> List[str]:
    """
    List all available profile names.

    Args:
        project_root: Path to project root directory

    Returns:
        List of profile names (without .yaml extension)
    """
    profiles_dir = project_root / 'profiles'
    if not profiles_dir.exists():
        return []

    profiles = []
    for p in profiles_dir.glob('*.yaml'):
        # Skip example template
        if p.stem != 'example':
            profiles.append(p.stem)

    return sorted(profiles)


def print_available_profiles(project_root: Path) -> None:
    """Print available profiles in a formatted way."""
    from .project_profile import ProfileManager

    profiles_dir = project_root / 'profiles'
    manager = ProfileManager(profiles_dir)
    manager.load_profiles()

    profile_names = manager.get_profile_names()

    if not profile_names:
        print("No profiles found.")
        print(f"Create a profile in: {profiles_dir}/")
        print("Or run: python scripts/init_project.py")
        return

    print("Available profiles:")
    print("-" * 40)

    for name in sorted(profile_names):
        if name == 'example':
            continue
        profile = manager.get_profile(name)
        if profile:
            lang_str = f"[{profile.language}]"
            print(f"  {name:20s} {lang_str:8s} {profile.source_path}")

    print("-" * 40)
    print(f"Use with: --profile <name>")


def print_profile_info(profile: 'ProjectProfile') -> None:
    """
    Print detailed profile information.

    Args:
        profile: ProjectProfile instance to display
    """
    print(f"\nProfile: {profile.name}")
    print("-" * 40)
    print(f"  Source path:    {profile.source_path}")
    print(f"  Language:       {profile.language}")

    if profile.include_dirs:
        print(f"  Include dirs:   {', '.join(profile.include_dirs)}")

    if profile.exclude_patterns:
        print(f"  Exclude:        {len(profile.exclude_patterns)} patterns")

    if profile.library_mappings:
        print(f"  Custom mappings: {len(profile.library_mappings)}")

    print(f"  Type hints:     {profile.use_type_hints}")
    print(f"  Dataclasses:    {profile.use_dataclasses}")
    print(f"  Docstring:      {profile.docstring_style}")

    if profile.is_cpp_enabled():
        print(f"  C++ templates:  {profile.cpp_settings.get('parse_templates', True)}")
        print(f"  C++ namespaces: {profile.cpp_settings.get('parse_namespaces', True)}")


def handle_list_profiles(args: argparse.Namespace, project_root: Path) -> bool:
    """
    Handle --list-profiles flag if present.

    Args:
        args: Parsed arguments
        project_root: Project root path

    Returns:
        True if --list-profiles was handled (caller should exit),
        False otherwise
    """
    if getattr(args, 'list_profiles', False):
        print_available_profiles(project_root)
        return True
    return False


def validate_profile_exists(
    profile_name: str,
    project_root: Path
) -> bool:
    """
    Validate that a profile exists.

    Args:
        profile_name: Name of profile to check
        project_root: Project root path

    Returns:
        True if profile exists
    """
    profile_path = project_root / 'profiles' / f'{profile_name}.yaml'
    return profile_path.exists()


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments.

    Args:
        description: Description for the parser

    Returns:
        ArgumentParser with common arguments added
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    add_list_profiles_argument(parser)
    return parser


class CLIError(Exception):
    """Exception for CLI-related errors."""
    pass


def error_exit(message: str, code: int = 1) -> None:
    """
    Print error message and exit.

    Args:
        message: Error message to display
        code: Exit code (default: 1)
    """
    import sys
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)
