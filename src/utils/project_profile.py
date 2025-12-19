"""
Project profile management for different C/C++ codebases.

Profiles allow per-project configuration including:
- Source language (C, C++, mixed)
- Include/exclude patterns
- Library mappings specific to the project
- Python output style preferences
- Naming conventions
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml


@dataclass
class ProjectProfile:
    """Configuration profile for a specific project."""

    # Required fields
    name: str
    source_path: str

    # Language settings
    language: str = "c"  # "c", "cpp", or "mixed"

    # Include/exclude patterns
    include_dirs: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

    # Library mappings specific to this project
    library_mappings: Dict[str, str] = field(default_factory=dict)

    # Translation preferences
    target_python_style: str = "modern"  # "modern", "legacy"
    use_dataclasses: bool = True
    use_type_hints: bool = True
    docstring_style: str = "google"  # "google", "numpy", "sphinx"

    # Custom patterns for this codebase
    naming_conventions: Dict[str, str] = field(default_factory=dict)

    # C++ specific settings (only used when language is "cpp" or "mixed")
    cpp_settings: Dict[str, Any] = field(default_factory=lambda: {
        "parse_templates": True,
        "parse_namespaces": True,
        "include_std_library": False,
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'source_path': self.source_path,
            'language': self.language,
            'include_dirs': self.include_dirs,
            'exclude_patterns': self.exclude_patterns,
            'library_mappings': self.library_mappings,
            'target_python_style': self.target_python_style,
            'use_dataclasses': self.use_dataclasses,
            'use_type_hints': self.use_type_hints,
            'docstring_style': self.docstring_style,
            'naming_conventions': self.naming_conventions,
            'cpp_settings': self.cpp_settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectProfile':
        """
        Create from dictionary.

        Args:
            data: Dictionary with profile fields

        Returns:
            ProjectProfile instance
        """
        # Handle optional fields with defaults
        return cls(
            name=data['name'],
            source_path=data['source_path'],
            language=data.get('language', 'c'),
            include_dirs=data.get('include_dirs', []),
            exclude_patterns=data.get('exclude_patterns', []),
            library_mappings=data.get('library_mappings', {}),
            target_python_style=data.get('target_python_style', 'modern'),
            use_dataclasses=data.get('use_dataclasses', True),
            use_type_hints=data.get('use_type_hints', True),
            docstring_style=data.get('docstring_style', 'google'),
            naming_conventions=data.get('naming_conventions', {}),
            cpp_settings=data.get('cpp_settings', {
                "parse_templates": True,
                "parse_namespaces": True,
                "include_std_library": False,
            }),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'ProjectProfile':
        """
        Load profile from YAML file.

        Args:
            yaml_path: Path to YAML profile file

        Returns:
            ProjectProfile instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Profile not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def save_yaml(self, yaml_path: Path) -> None:
        """
        Save profile to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_source_path(self) -> Path:
        """Get source path as Path object."""
        return Path(self.source_path)

    def is_cpp_enabled(self) -> bool:
        """Check if C++ parsing is enabled."""
        return self.language in ('cpp', 'mixed')

    def get_file_extensions(self) -> List[str]:
        """Get file extensions to parse based on language setting."""
        if self.language == 'c':
            return ['.c', '.h']
        elif self.language == 'cpp':
            return ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh', '.h']
        else:  # mixed
            return ['.c', '.h', '.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh']


class ProfileManager:
    """Manage multiple project profiles."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize profile manager.

        Args:
            profiles_dir: Directory containing profile YAML files.
                         Defaults to 'profiles/' in project root.
        """
        if profiles_dir is None:
            # Default to profiles/ directory relative to this file
            project_root = Path(__file__).parent.parent.parent
            profiles_dir = project_root / 'profiles'

        self.profiles_dir = Path(profiles_dir)
        self.profiles: Dict[str, ProjectProfile] = {}
        self._active_profile: Optional[str] = None

    def load_profiles(self) -> int:
        """
        Load all profiles from profiles directory.

        Returns:
            Number of profiles loaded
        """
        if not self.profiles_dir.exists():
            return 0

        count = 0
        for yaml_file in self.profiles_dir.glob("*.yaml"):
            try:
                profile = ProjectProfile.from_yaml(yaml_file)
                self.profiles[profile.name] = profile
                count += 1
            except Exception as e:
                # Log error but continue loading other profiles
                print(f"Warning: Failed to load profile {yaml_file}: {e}")

        return count

    def get_profile(self, name: str) -> Optional[ProjectProfile]:
        """
        Get profile by name.

        Args:
            name: Profile name

        Returns:
            ProjectProfile if found, None otherwise
        """
        return self.profiles.get(name)

    def get_profile_names(self) -> List[str]:
        """Get list of available profile names."""
        return list(self.profiles.keys())

    def create_profile(self, profile: ProjectProfile, save: bool = True) -> None:
        """
        Create and optionally save a new profile.

        Args:
            profile: ProjectProfile instance
            save: Whether to save to disk
        """
        self.profiles[profile.name] = profile

        if save:
            yaml_path = self.profiles_dir / f"{profile.name}.yaml"
            profile.save_yaml(yaml_path)

    def delete_profile(self, name: str, delete_file: bool = True) -> bool:
        """
        Delete a profile.

        Args:
            name: Profile name
            delete_file: Whether to delete the YAML file

        Returns:
            True if deleted, False if not found
        """
        if name not in self.profiles:
            return False

        del self.profiles[name]

        if delete_file:
            yaml_path = self.profiles_dir / f"{name}.yaml"
            if yaml_path.exists():
                yaml_path.unlink()

        return True

    def set_active_profile(self, name: str) -> bool:
        """
        Set the active profile for the session.

        Args:
            name: Profile name

        Returns:
            True if profile exists and was set, False otherwise
        """
        if name in self.profiles:
            self._active_profile = name
            return True
        return False

    def get_active_profile(self) -> Optional[ProjectProfile]:
        """
        Get the currently active profile.

        Returns:
            Active ProjectProfile or None
        """
        if self._active_profile:
            return self.profiles.get(self._active_profile)
        return None

    def ensure_profiles_dir(self) -> None:
        """Create profiles directory if it doesn't exist."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)


def create_profile_from_config(config: 'Config', name: str = "default") -> ProjectProfile:
    """
    Create a ProjectProfile from existing config.yaml settings.

    This is a convenience function for migrating from config-only
    setup to profile-based setup.

    Args:
        config: Config instance
        name: Name for the new profile

    Returns:
        ProjectProfile with settings from config
    """
    return ProjectProfile(
        name=name,
        source_path=config.get('source.source_path', ''),
        language=config.get('source.language', 'c'),
        include_dirs=config.get('source.include_dirs', []),
        exclude_patterns=config.get('source.exclude_patterns', []),
        library_mappings=config.get('library_mappings.custom_mappings', {}),
        target_python_style='modern',  # Always modern for new profiles
        use_dataclasses=config.get('translation.python_style.use_dataclasses', True),
        use_type_hints=config.get('translation.python_style.use_type_hints', True),
        docstring_style=config.get('translation.python_style.docstring_style', 'google'),
        naming_conventions={},
        cpp_settings={
            'parse_templates': config.get('cpp.parse_templates', True),
            'parse_namespaces': config.get('cpp.parse_namespaces', True),
            'include_std_library': config.get('cpp.include_std_library', False),
        },
    )
