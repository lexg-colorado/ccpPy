#!/usr/bin/env python3
"""
Project Initialization Wizard.

This script helps users create new project profiles for the C/C++ to Python translator.
It supports both interactive and non-interactive modes.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.project_profile import ProjectProfile, ProfileManager
from utils.config import Config


def detect_language(source_path: Path) -> str:
    """
    Auto-detect the source language based on file extensions.

    Args:
        source_path: Path to source directory

    Returns:
        Detected language: 'c', 'cpp', or 'mixed'
    """
    c_extensions = {'.c', '.h'}
    cpp_extensions = {'.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh'}

    c_files = 0
    cpp_files = 0

    for ext in c_extensions:
        c_files += len(list(source_path.rglob(f'*{ext}')))

    for ext in cpp_extensions:
        cpp_files += len(list(source_path.rglob(f'*{ext}')))

    if c_files > 0 and cpp_files > 0:
        return 'mixed'
    elif cpp_files > 0:
        return 'cpp'
    else:
        return 'c'


def count_source_files(source_path: Path, language: str) -> Dict[str, int]:
    """
    Count source files by type.

    Args:
        source_path: Path to source directory
        language: Language setting

    Returns:
        Dict with file counts by extension
    """
    counts = {}

    if language in ('c', 'mixed'):
        for ext in ['.c', '.h']:
            count = len(list(source_path.rglob(f'*{ext}')))
            if count > 0:
                counts[ext] = count

    if language in ('cpp', 'mixed'):
        for ext in ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh']:
            count = len(list(source_path.rglob(f'*{ext}')))
            if count > 0:
                counts[ext] = count

    return counts


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    """
    Prompt user for input with optional default.

    Args:
        prompt: Prompt message
        default: Default value

    Returns:
        User input or default
    """
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    try:
        value = input(full_prompt).strip()
        return value if value else (default or "")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)


def prompt_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    Prompt user to select from choices.

    Args:
        prompt: Prompt message
        choices: List of valid choices
        default: Default choice

    Returns:
        Selected choice
    """
    choices_str = "/".join(choices)
    while True:
        value = prompt_input(f"{prompt} ({choices_str})", default)
        if value in choices:
            return value
        print(f"Invalid choice. Please select from: {choices_str}")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Prompt for yes/no answer.

    Args:
        prompt: Prompt message
        default: Default value

    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not value:
            return default
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False
        print("Please answer 'yes' or 'no'.")


def interactive_wizard() -> Optional[ProjectProfile]:
    """
    Run interactive wizard to create a profile.

    Returns:
        Created ProjectProfile or None if cancelled
    """
    print("\n" + "=" * 60)
    print("C/C++ to Python Translator - Project Setup Wizard")
    print("=" * 60 + "\n")

    # Get project name
    name = prompt_input("Project name (used for profile filename)")
    if not name:
        print("Project name is required.")
        return None

    # Validate name (no special characters)
    if not name.replace('_', '').replace('-', '').isalnum():
        print("Project name should only contain letters, numbers, underscores, and hyphens.")
        return None

    # Get source path
    source_str = prompt_input("Path to C/C++ source code")
    if not source_str:
        print("Source path is required.")
        return None

    source_path = Path(source_str).expanduser().resolve()
    if not source_path.exists():
        print(f"Path does not exist: {source_path}")
        return None

    if not source_path.is_dir():
        print(f"Path is not a directory: {source_path}")
        return None

    # Detect language
    print("\nAnalyzing source directory...")
    detected_lang = detect_language(source_path)
    file_counts = count_source_files(source_path, 'mixed')

    print(f"  Detected language: {detected_lang}")
    for ext, count in sorted(file_counts.items()):
        print(f"    {ext}: {count} files")

    # Confirm or override language
    language = prompt_choice(
        "\nSource language",
        ['c', 'cpp', 'mixed'],
        detected_lang
    )

    # Python style options
    print("\n--- Python Output Style ---")
    use_type_hints = prompt_yes_no("Use type hints?", True)
    use_dataclasses = prompt_yes_no("Use dataclasses for structs?", True)
    docstring_style = prompt_choice(
        "Docstring style",
        ['google', 'numpy', 'sphinx'],
        'google'
    )

    # Create profile
    profile = ProjectProfile(
        name=name,
        source_path=str(source_path),
        language=language,
        use_type_hints=use_type_hints,
        use_dataclasses=use_dataclasses,
        docstring_style=docstring_style
    )

    # Show summary
    print("\n" + "-" * 40)
    print("Profile Summary:")
    print("-" * 40)
    print(f"  Name:           {profile.name}")
    print(f"  Source path:    {profile.source_path}")
    print(f"  Language:       {profile.language}")
    print(f"  Type hints:     {profile.use_type_hints}")
    print(f"  Dataclasses:    {profile.use_dataclasses}")
    print(f"  Docstring:      {profile.docstring_style}")
    print("-" * 40)

    if not prompt_yes_no("\nCreate this profile?", True):
        print("Cancelled.")
        return None

    return profile


def create_profile_noninteractive(args: argparse.Namespace) -> Optional[ProjectProfile]:
    """
    Create profile from command-line arguments.

    Args:
        args: Parsed arguments

    Returns:
        Created ProjectProfile or None on error
    """
    # Validate required args
    if not args.name:
        print("Error: --name is required in non-interactive mode")
        return None

    if not args.source:
        print("Error: --source is required in non-interactive mode")
        return None

    # Validate source path
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        return None

    if not source_path.is_dir():
        print(f"Error: Source path is not a directory: {source_path}")
        return None

    # Detect or use specified language
    if args.lang == 'auto':
        language = detect_language(source_path)
        print(f"Auto-detected language: {language}")
    else:
        language = args.lang

    # Create profile
    profile = ProjectProfile(
        name=args.name,
        source_path=str(source_path),
        language=language,
        use_type_hints=True,
        use_dataclasses=True,
        docstring_style='google'
    )

    return profile


def save_profile(profile: ProjectProfile, project_root: Path) -> bool:
    """
    Save profile to profiles directory.

    Args:
        profile: Profile to save
        project_root: Project root path

    Returns:
        True if saved successfully
    """
    profiles_dir = project_root / 'profiles'
    profiles_dir.mkdir(parents=True, exist_ok=True)

    manager = ProfileManager(profiles_dir)
    manager.save_profile(profile)

    profile_path = profiles_dir / f'{profile.name}.yaml'
    print(f"\nProfile saved to: {profile_path}")

    return True


def set_default_profile(profile_name: str, project_root: Path) -> bool:
    """
    Set profile as default in config.yaml.

    Args:
        profile_name: Profile name to set as default
        project_root: Project root path

    Returns:
        True if updated successfully
    """
    config_path = project_root / 'config.yaml'

    try:
        config = Config(str(config_path))
        config.set('project.profile', profile_name)
        config.save()
        print(f"Set '{profile_name}' as default profile in config.yaml")
        return True
    except Exception as e:
        print(f"Warning: Could not update config.yaml: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize a new project profile for the C/C++ to Python translator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python scripts/init_project.py

  # Non-interactive with all options
  python scripts/init_project.py --name myproject --source /path/to/code --lang cpp

  # Auto-detect language and set as default
  python scripts/init_project.py -n myproject -s /path/to/code -l auto --set-default
"""
    )

    parser.add_argument(
        '--name', '-n',
        metavar='NAME',
        help='Project name (used for profile filename)'
    )
    parser.add_argument(
        '--source', '-s',
        metavar='PATH',
        help='Path to C/C++ source code directory'
    )
    parser.add_argument(
        '--lang', '-l',
        choices=['c', 'cpp', 'mixed', 'auto'],
        default='auto',
        help='Source language (default: auto-detect)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Force interactive mode even with other flags'
    )
    parser.add_argument(
        '--set-default',
        action='store_true',
        help='Set this profile as default in config.yaml'
    )

    return parser.parse_args()


def main():
    """Main entry point for project initialization."""
    args = parse_args()

    # Determine mode
    has_required_args = args.name and args.source
    use_interactive = args.interactive or not has_required_args

    if use_interactive:
        profile = interactive_wizard()
    else:
        profile = create_profile_noninteractive(args)

    if profile is None:
        return 1

    # Save profile
    if not save_profile(profile, project_root):
        return 1

    # Set as default if requested
    if args.set_default:
        set_default_profile(profile.name, project_root)

    print("\nProfile created successfully!")
    print(f"\nTo use this profile, run:")
    print(f"  python scripts/01_parse_c_code.py --profile {profile.name}")
    print(f"  python scripts/02_build_graph.py --profile {profile.name}")
    print(f"  python scripts/03_index_code.py --profile {profile.name}")
    print(f"  python scripts/04_translate.py --profile {profile.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
