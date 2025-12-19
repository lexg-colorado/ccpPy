"""
Configuration management with profile support.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .project_profile import ProjectProfile, ProfileManager


class Config:
    """Manage project configuration from YAML file with optional profile support."""

    def __init__(self, config_path: str = "config.yaml", profile_name: Optional[str] = None):
        """
        Load configuration from file, optionally with a profile.

        Args:
            config_path: Path to config.yaml file
            profile_name: Optional profile name to load and apply
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._profile: Optional['ProjectProfile'] = None
        self._profile_manager: Optional['ProfileManager'] = None

        # Load profile if specified
        if profile_name:
            self.load_profile(profile_name)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-separated key path (e.g., 'source.source_path')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration back to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    # Profile integration methods

    def _get_profile_manager(self) -> 'ProfileManager':
        """Get or create the profile manager."""
        if self._profile_manager is None:
            from .project_profile import ProfileManager
            project_root = self.config_path.parent
            self._profile_manager = ProfileManager(project_root / 'profiles')
            self._profile_manager.load_profiles()
        return self._profile_manager

    def load_profile(self, name: str) -> bool:
        """
        Load and apply a profile by name.

        Args:
            name: Profile name to load

        Returns:
            True if profile was loaded successfully, False otherwise
        """
        manager = self._get_profile_manager()
        profile = manager.get_profile(name)

        if profile is None:
            return False

        self._profile = profile
        self._apply_profile_to_config()
        return True

    def _apply_profile_to_config(self) -> None:
        """Apply profile settings to config (profile overrides config)."""
        if self._profile is None:
            return

        # Override source settings from profile
        self.set('source.source_path', self._profile.source_path)
        self.set('source.language', self._profile.language)

        if self._profile.include_dirs:
            self.set('source.include_dirs', self._profile.include_dirs)

        if self._profile.exclude_patterns:
            self.set('source.exclude_patterns', self._profile.exclude_patterns)

        # Apply Python style settings
        self.set('experimental.generate_type_hints', self._profile.use_type_hints)

        # Apply C++ settings if applicable
        if self._profile.is_cpp_enabled():
            for key, value in self._profile.cpp_settings.items():
                self.set(f'cpp.{key}', value)

    def get_profile(self) -> Optional['ProjectProfile']:
        """
        Get the currently active profile.

        Returns:
            Active ProjectProfile or None if no profile is loaded
        """
        return self._profile

    def get_available_profiles(self) -> list:
        """
        Get list of available profile names.

        Returns:
            List of profile names
        """
        manager = self._get_profile_manager()
        return manager.get_profile_names()

    def has_profile(self) -> bool:
        """Check if a profile is currently loaded."""
        return self._profile is not None

    def get_effective_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value, preferring profile values over config.

        This method checks the profile first for certain keys, then falls
        back to config. Use this for values that can be overridden by profiles.

        Args:
            key: Dot-separated key path
            default: Default value if not found

        Returns:
            Configuration value from profile or config
        """
        # Profile override mapping
        profile_mappings = {
            'source.source_path': lambda p: p.source_path,
            'source.language': lambda p: p.language,
            'source.include_dirs': lambda p: p.include_dirs,
            'source.exclude_patterns': lambda p: p.exclude_patterns,
        }

        if self._profile and key in profile_mappings:
            value = profile_mappings[key](self._profile)
            if value:  # Only use profile value if non-empty
                return value

        return self.get(key, default)

    # CLI override methods

    def apply_cli_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply CLI argument overrides (highest priority).

        CLI overrides take precedence over both profile and config values.

        Args:
            overrides: Dictionary of key-value pairs to override
        """
        if not hasattr(self, '_cli_overrides'):
            self._cli_overrides: Dict[str, Any] = {}

        for key, value in overrides.items():
            if value is not None:
                self.set(key, value)
                self._cli_overrides[key] = value

    def get_value_source(self, key: str) -> str:
        """
        Get the source of a configuration value for debugging.

        Args:
            key: Dot-separated key path

        Returns:
            Source of the value: 'cli', 'profile', or 'config'
        """
        if hasattr(self, '_cli_overrides') and key in self._cli_overrides:
            return 'cli'
        if self._profile:
            # Check if this key is overridden by profile
            profile_keys = {
                'source.source_path', 'source.language',
                'source.include_dirs', 'source.exclude_patterns'
            }
            if key in profile_keys:
                return 'profile'
        return 'config'