#! /usr/bin/env python3
"""
Phase 1: Parse htop C codebase into AST representation.

This script:
1. Finds all C files in the htop source directory
2. Parses each file using tree-sitter
3. Caches parsed AST data as JSON
4. Generates summary statistics
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from parser.ast_parser import CParser
from utils.config import Config
from utils.logger import setup_logger


class BatchParser:
    """Parse multiple C files and cache results."""
    
    def __init__(self, config: Config, logger):
        """Initialize batch parser with configuration."""
        self.config = config
        self.logger = logger
        
        # Get paths from config
        self.htop_path = Path(config.get('source.htop_path'))
        self.cache_dir = Path(config.get('output.ast_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get parsing settings
        self.extensions = config.get('parsing.extensions', ['.c', '.h'])
        self.max_file_size_mb = config.get('parsing.max_file_size_mb', 10)
        self.parallel = config.get('performance.parallel_parsing', True)
        self.num_workers = config.get('performance.num_workers', None)
        self.use_cache = config.get('performance.cache_asts', True)
    
    def find_c_files(self) -> List[Path]:
        """
        Find all C source files in the htop directory.
        
        Returns:
            List of Path objects for C files
        """
        self.logger.info(f"Searching for C files in {self.htop_path}")
        
        c_files = []
        
        # Get include directories or scan all
        include_dirs = self.config.get('source.include_dirs', [])
        exclude_patterns = self.config.get('source.exclude_patterns', [])
        
        if include_dirs:
            # Only search specified directories
            search_paths = [self.htop_path / d for d in include_dirs]
        else:
            # Search entire htop directory
            search_paths = [self.htop_path]
        
        for search_path in search_paths:
            if not search_path.exists():
                self.logger.warning(f"Path does not exist: {search_path}")
                continue
            
            for ext in self.extensions:
                for file_path in search_path.rglob(f"*{ext}"):
                    # Check file size
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > self.max_file_size_mb:
                        self.logger.warning(f"Skipping large file ({size_mb:.1f}MB): {file_path}")
                        continue
                    
                    # Check exclude patterns
                    if any(file_path.match(pattern) for pattern in exclude_patterns):
                        continue
                    
                    c_files.append(file_path)
        
        self.logger.info(f"Found {len(c_files)} C files to parse")
        return sorted(c_files)
    
    def get_cache_path(self, source_file: Path) -> Path:
        """
        Get the cache file path for a source file.
        
        Args:
            source_file: Path to source C file
            
        Returns:
            Path to cached JSON file
        """
        # Create relative path from htop root
        rel_path = source_file.relative_to(self.htop_path)
        # Replace extension with .json
        cache_file = self.cache_dir / rel_path.with_suffix('.json')
        return cache_file
    
    def is_cached(self, source_file: Path) -> bool:
        """
        Check if a file has been parsed and cached.
        
        Args:
            source_file: Path to source C file
            
        Returns:
            True if cached and up-to-date
        """
        if not self.use_cache:
            return False
        
        cache_file = self.get_cache_path(source_file)
        
        if not cache_file.exists():
            return False
        
        # Check if cache is newer than source
        source_mtime = source_file.stat().st_mtime
        cache_mtime = cache_file.stat().st_mtime
        
        return cache_mtime > source_mtime
    
    def parse_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a single C file (for use in parallel processing).
        
        Args:
            file_path: Path to C file
            
        Returns:
            Dict with parse results and metadata
        """
        try:
            # Check cache first
            if self.is_cached(file_path):
                cache_file = self.get_cache_path(file_path)
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                return {
                    'success': True,
                    'file': str(file_path),
                    'cached': True,
                    'data': cached_data
                }
            
            # Parse the file
            parser = CParser()  # Create new parser instance for parallel processing
            result = parser.parse_file(file_path)
            
            # Save to cache
            cache_file = self.get_cache_path(file_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save without source code (too large)
            cache_data = {k: v for k, v in result.items() if k != 'source'}
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'file': str(file_path),
                'cached': False,
                'data': cache_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'file': str(file_path),
                'error': str(e)
            }
    
    def parse_all(self, c_files: List[Path]) -> Dict[str, Any]:
        """
        Parse all C files, either in parallel or sequentially.
        
        Args:
            c_files: List of C files to parse
            
        Returns:
            Dict containing all parsed data and statistics
        """
        results = []
        errors = []
        
        self.logger.info(f"Parsing {len(c_files)} files...")
        
        if self.parallel and len(c_files) > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(self.parse_single_file, f): f for f in c_files}
                
                # Collect results with progress bar
                with tqdm(total=len(c_files), desc="Parsing files") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        
                        if result['success']:
                            results.append(result)
                            if result['cached']:
                                pbar.set_postfix({'status': 'cached'})
                            else:
                                pbar.set_postfix({'status': 'parsed'})
                        else:
                            errors.append(result)
                            pbar.set_postfix({'status': 'error'})
                        
                        pbar.update(1)
        else:
            # Sequential processing
            for file_path in tqdm(c_files, desc="Parsing files"):
                result = self.parse_single_file(file_path)
                
                if result['success']:
                    results.append(result)
                else:
                    errors.append(result)
        
        # Generate statistics
        stats = self._generate_statistics(results)
        
        if errors:
            self.logger.warning(f"Failed to parse {len(errors)} files:")
            for error in errors[:5]:  # Show first 5 errors
                self.logger.warning(f"  - {error['file']}: {error['error']}")
            if len(errors) > 5:
                self.logger.warning(f"  ... and {len(errors) - 5} more")
        
        return {
            'results': results,
            'errors': errors,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics from parsed results.
        
        Args:
            results: List of parse results
            
        Returns:
            Dict of statistics
        """
        total_functions = 0
        total_structs = 0
        total_includes = 0
        total_files = len(results)
        cached_files = sum(1 for r in results if r.get('cached', False))
        
        function_names = set()
        struct_names = set()
        all_calls = []
        
        for result in results:
            data = result['data']
            
            # Count items
            total_functions += len(data['functions'])
            total_structs += len(data['structs'])
            total_includes += len(data['includes'])
            
            # Collect names
            for func in data['functions']:
                function_names.add(func['name'])
                all_calls.extend(func['calls'])
            
            for struct in data['structs']:
                struct_names.add(struct['name'])
        
        # Find most called functions
        from collections import Counter
        call_counts = Counter(all_calls)
        most_called = call_counts.most_common(10)
        
        return {
            'total_files': total_files,
            'cached_files': cached_files,
            'parsed_files': total_files - cached_files,
            'total_functions': total_functions,
            'unique_functions': len(function_names),
            'total_structs': total_structs,
            'unique_structs': len(struct_names),
            'total_includes': total_includes,
            'most_called_functions': [
                {'name': name, 'count': count}
                for name, count in most_called
            ]
        }
    
    def save_summary(self, parse_results: Dict[str, Any]) -> None:
        """
        Save parsing summary to JSON file.
        
        Args:
            parse_results: Results from parse_all()
        """
        summary_path = self.cache_dir / 'parse_summary.json'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(parse_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved summary to {summary_path}")
    
    def print_statistics(self, stats: Dict[str, Any]) -> None:
        """Print statistics in a readable format."""
        print("\n" + "="*60)
        print("HTOP PARSING SUMMARY")
        print("="*60)
        print(f"\nFiles:")
        print(f"  Total files:    {stats['total_files']}")
        print(f"  Cached files:   {stats['cached_files']}")
        print(f"  Parsed files:   {stats['parsed_files']}")
        
        print(f"\nCode Elements:")
        print(f"  Total functions:  {stats['total_functions']}")
        print(f"  Unique functions: {stats['unique_functions']}")
        print(f"  Total structs:    {stats['total_structs']}")
        print(f"  Unique structs:   {stats['unique_structs']}")
        print(f"  Total includes:   {stats['total_includes']}")
        
        print(f"\nMost Called Functions:")
        for item in stats['most_called_functions']:
            print(f"  {item['name']:30s} ({item['count']} calls)")
        
        print("\n" + "="*60)


def main():
    """Main entry point for batch parsing."""
    # Load configuration
    config_path = project_root / "config.yaml"
    config = Config(str(config_path))
    
    # Setup logging
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', 'logs/translator.log')
    logger = setup_logger("batch_parser", level=log_level, log_file=log_file)
    
    logger.info("="*60)
    logger.info("Starting htop batch parsing")
    logger.info("="*60)
    
    # Create batch parser
    batch_parser = BatchParser(config, logger)
    
    # Find C files
    c_files = batch_parser.find_c_files()
    
    if not c_files:
        logger.error("No C files found!")
        return 1
    
    # Parse all files
    results = batch_parser.parse_all(c_files)
    
    # Save summary
    batch_parser.save_summary(results)
    
    # Print statistics
    batch_parser.print_statistics(results['statistics'])
    
    logger.info("Batch parsing complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())