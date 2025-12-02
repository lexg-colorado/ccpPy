#!/usr/bin/env python3
"""
Phase 2: Build dependency graph from parsed C codebase.

This script:
1. Loads parsed AST data from cache
2. Builds function call graph (who calls whom)
3. Builds file dependency graph (include relationships)
4. Identifies leaf nodes (functions with no dependencies)
5. Performs topological sort for translation order
6. Saves graph data for later use
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime

import networkx as nx
from networkx.algorithms import dag

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.config import Config
from utils.logger import setup_logger


class DependencyGraphBuilder:
    """Build and analyze dependency graphs from parsed C code."""
    
    def __init__(self, config: Config, logger):
        """Initialize graph builder with configuration."""
        self.config = config
        self.logger = logger
        
        # Get paths from config
        self.cache_dir = Path(config.get('output.ast_cache'))
        self.graph_dir = Path(config.get('output.graphs'))
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize graphs
        self.function_graph = nx.DiGraph()  # Function call graph
        self.file_graph = nx.DiGraph()      # File dependency graph
        
        # Data structures
        self.functions = {}  # function_name -> {file, line, etc.}
        self.files = {}      # file_path -> {functions, structs, includes}
        
    def load_parsed_data(self) -> None:
        """Load all parsed AST data from cache."""
        self.logger.info("Loading parsed data from cache...")
        
        # Load summary to get list of files
        summary_path = self.cache_dir / 'parse_summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Parse summary not found at {summary_path}. "
                "Run scripts/01_parse_htop.py first!"
            )
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Load each parsed file
        file_count = 0
        for result in summary['results']:
            if not result['success']:
                continue
            
            data = result['data']
            file_path = data['file_path']
            
            # Store file data
            self.files[file_path] = data
            file_count += 1
            
            # Index functions by name
            for func in data['functions']:
                func_name = func['name']
                
                # Handle duplicate function names (keep first occurrence)
                if func_name not in self.functions:
                    self.functions[func_name] = {
                        'file': file_path,
                        'line': func['start_line'],
                        'return_type': func['return_type'],
                        'parameters': func['parameters'],
                        'calls': func['calls']
                    }
        
        self.logger.info(f"Loaded {file_count} files with {len(self.functions)} unique functions")
    
    def build_function_call_graph(self) -> None:
        """Build directed graph of function calls."""
        self.logger.info("Building function call graph...")
        
        # Add all functions as nodes
        for func_name, func_data in self.functions.items():
            self.function_graph.add_node(
                func_name,
                file=func_data['file'],
                line=func_data['line'],
                return_type=func_data['return_type']
            )
        
        # Add edges for function calls
        edge_count = 0
        for func_name, func_data in self.functions.items():
            for called_func in func_data['calls']:
                # Only add edge if called function exists in our codebase
                if called_func in self.functions:
                    self.function_graph.add_edge(func_name, called_func)
                    edge_count += 1
        
        self.logger.info(
            f"Function graph: {self.function_graph.number_of_nodes()} nodes, "
            f"{self.function_graph.number_of_edges()} edges"
        )
    
    def build_file_dependency_graph(self) -> None:
        """Build directed graph of file dependencies (includes)."""
        self.logger.info("Building file dependency graph...")
        
        include_system = self.config.get('dependency_graph.include_system_headers', False)
        
        # Add all files as nodes
        for file_path, file_data in self.files.items():
            self.file_graph.add_node(
                file_path,
                num_functions=len(file_data['functions']),
                num_structs=len(file_data['structs'])
            )
        
        # Add edges for includes
        edge_count = 0
        for file_path, file_data in self.files.items():
            for include in file_data['includes']:
                # Skip system includes if configured
                if include['is_system'] and not include_system:
                    continue
                
                # Try to resolve include path
                include_path = include['path']
                resolved_path = self._resolve_include_path(file_path, include_path)
                
                if resolved_path and resolved_path in self.files:
                    self.file_graph.add_edge(file_path, resolved_path)
                    edge_count += 1
        
        self.logger.info(
            f"File graph: {self.file_graph.number_of_nodes()} nodes, "
            f"{self.file_graph.number_of_edges()} edges"
        )
    
    def _resolve_include_path(self, source_file: str, include_path: str) -> str:
        """
        Try to resolve an include path to an actual file.
        
        Args:
            source_file: File containing the include
            include_path: Path from #include statement
            
        Returns:
            Resolved absolute path or None
        """
        source_path = Path(source_file)
        
        # Try relative to source file's directory
        relative_path = source_path.parent / include_path
        if str(relative_path) in self.files:
            return str(relative_path)
        
        # Try in htop root
        htop_path = Path(self.config.get('source.htop_path'))
        root_path = htop_path / include_path
        if str(root_path) in self.files:
            return str(root_path)
        
        return None
    
    def identify_leaf_nodes(self) -> List[str]:
        """
        Identify leaf nodes (functions that don't call other functions in codebase).
        
        Returns:
            List of function names that are leaves
        """
        self.logger.info("Identifying leaf nodes...")
        
        leaves = []
        for node in self.function_graph.nodes():
            # A leaf has no outgoing edges
            if self.function_graph.out_degree(node) == 0:
                leaves.append(node)
        
        self.logger.info(f"Found {len(leaves)} leaf functions")
        return leaves
    
    def topological_sort_functions(self) -> List[str]:
        """
        Perform topological sort on function graph.
        
        Returns:
            List of function names in dependency order (leaves first)
        
        Raises:
            Exception if graph has cycles
        """
        self.logger.info("Performing topological sort...")
        
        try:
            # NetworkX topological_sort returns generator
            sorted_funcs = list(nx.topological_sort(self.function_graph))
            self.logger.info(f"Topological sort successful: {len(sorted_funcs)} functions")
            return sorted_funcs
        except (nx.NetworkXError, nx.NetworkXUnfeasible) as e:
            self.logger.warning(f"Graph has cycles, using approximate ordering: {e}")
            # If graph has cycles, use a heuristic ordering
            return self._approximate_topological_sort()
    
    def _approximate_topological_sort(self) -> List[str]:
        """
        Approximate topological sort for graphs with cycles.
        Uses in-degree as heuristic (fewer dependencies first).
        
        Returns:
            List of function names ordered by dependency count
        """
        nodes_by_indegree = sorted(
            self.function_graph.nodes(),
            key=lambda n: self.function_graph.in_degree(n)
        )
        return nodes_by_indegree
    
    def find_strongly_connected_components(self) -> List[Set[str]]:
        """
        Find strongly connected components (cycles) in function graph.
        
        Returns:
            List of sets, each containing mutually recursive functions
        """
        self.logger.info("Finding strongly connected components...")
        
        components = list(nx.strongly_connected_components(self.function_graph))
        
        # Filter to only components with size > 1 (actual cycles)
        cycles = [comp for comp in components if len(comp) > 1]
        
        self.logger.info(f"Found {len(cycles)} cycles involving {sum(len(c) for c in cycles)} functions")
        
        return cycles
    
    def analyze_complexity(self) -> Dict[str, Any]:
        """
        Analyze graph complexity and characteristics.
        
        Returns:
            Dict of analysis metrics
        """
        self.logger.info("Analyzing graph complexity...")
        
        analysis = {
            'function_graph': {
                'nodes': self.function_graph.number_of_nodes(),
                'edges': self.function_graph.number_of_edges(),
                'density': nx.density(self.function_graph),
                'is_dag': nx.is_directed_acyclic_graph(self.function_graph),
            },
            'file_graph': {
                'nodes': self.file_graph.number_of_nodes(),
                'edges': self.file_graph.number_of_edges(),
                'density': nx.density(self.file_graph),
                'is_dag': nx.is_directed_acyclic_graph(self.file_graph),
            }
        }
        
        # Find most connected functions
        in_degrees = dict(self.function_graph.in_degree())
        out_degrees = dict(self.function_graph.out_degree())
        
        most_called = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        most_calling = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis['most_called_functions'] = [
            {'name': name, 'call_count': count}
            for name, count in most_called
        ]
        
        analysis['most_calling_functions'] = [
            {'name': name, 'calls_made': count}
            for name, count in most_calling
        ]
        
        # Find cycles
        cycles = self.find_strongly_connected_components()
        analysis['cycles'] = [
            {'functions': list(cycle), 'size': len(cycle)}
            for cycle in cycles
        ]
        
        return analysis
    
    def save_graphs(self) -> None:
        """Save graph data to disk."""
        self.logger.info("Saving graphs...")
        
        # Save as pickle (preserves graph structure)
        function_graph_path = self.graph_dir / 'function_graph.gpickle'
        file_graph_path = self.graph_dir / 'file_graph.gpickle'
        
        with open(function_graph_path, 'wb') as f:
            pickle.dump(self.function_graph, f)
        with open(file_graph_path, 'wb') as f:
            pickle.dump(self.file_graph, f)
        
        self.logger.info(f"Saved function graph to {function_graph_path}")
        self.logger.info(f"Saved file graph to {file_graph_path}")
        
        # Save as JSON for easy inspection
        function_data = {
            'nodes': [
                {
                    'name': node,
                    **self.function_graph.nodes[node]
                }
                for node in self.function_graph.nodes()
            ],
            'edges': [
                {'from': u, 'to': v}
                for u, v in self.function_graph.edges()
            ]
        }
        
        function_json_path = self.graph_dir / 'function_graph.json'
        with open(function_json_path, 'w', encoding='utf-8') as f:
            json.dump(function_data, f, indent=2)
        
        self.logger.info(f"Saved function graph JSON to {function_json_path}")
    
    def save_translation_order(self, sorted_funcs: List[str]) -> None:
        """
        Save topological sort order for translation.
        
        Args:
            sorted_funcs: List of function names in dependency order
        """
        order_path = self.graph_dir / 'translation_order.json'
        
        order_data = {
            'timestamp': datetime.now().isoformat(),
            'total_functions': len(sorted_funcs),
            'order': sorted_funcs
        }
        
        with open(order_path, 'w', encoding='utf-8') as f:
            json.dump(order_data, f, indent=2)
        
        self.logger.info(f"Saved translation order to {order_path}")
    
    def save_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Save graph analysis to JSON.
        
        Args:
            analysis: Analysis results from analyze_complexity()
        """
        analysis_path = self.graph_dir / 'graph_analysis.json'
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Saved analysis to {analysis_path}")
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print analysis results in readable format."""
        print("\n" + "="*60)
        print("DEPENDENCY GRAPH ANALYSIS")
        print("="*60)
        
        print("\nFunction Call Graph:")
        fg = analysis['function_graph']
        print(f"  Nodes (functions): {fg['nodes']}")
        print(f"  Edges (calls):     {fg['edges']}")
        print(f"  Graph density:     {fg['density']:.4f}")
        print(f"  Is DAG:            {fg['is_dag']}")
        
        print("\nFile Dependency Graph:")
        filg = analysis['file_graph']
        print(f"  Nodes (files):     {filg['nodes']}")
        print(f"  Edges (includes):  {filg['edges']}")
        print(f"  Graph density:     {filg['density']:.4f}")
        print(f"  Is DAG:            {filg['is_dag']}")
        
        print("\nMost Called Functions:")
        for item in analysis['most_called_functions']:
            print(f"  {item['name']:30s} (called {item['call_count']} times)")
        
        print("\nMost Calling Functions:")
        for item in analysis['most_calling_functions']:
            print(f"  {item['name']:30s} (calls {item['calls_made']} functions)")
        
        if analysis['cycles']:
            print(f"\nCycles Detected: {len(analysis['cycles'])}")
            for i, cycle in enumerate(analysis['cycles'][:5], 1):
                print(f"  Cycle {i}: {cycle['size']} functions - {', '.join(cycle['functions'][:3])}...")
            if len(analysis['cycles']) > 5:
                print(f"  ... and {len(analysis['cycles']) - 5} more cycles")
        else:
            print("\nNo cycles detected - graph is a DAG!")
        
        print("\n" + "="*60)


def main():
    """Main entry point for dependency graph building."""
    # Load configuration
    config_path = project_root / "config.yaml"
    config = Config(str(config_path))
    
    # Setup logging
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', 'logs/translator.log')
    logger = setup_logger("graph_builder", level=log_level, log_file=log_file)
    
    logger.info("="*60)
    logger.info("Starting dependency graph building")
    logger.info("="*60)
    
    # Create graph builder
    builder = DependencyGraphBuilder(config, logger)
    
    # Load parsed data
    builder.load_parsed_data()
    
    # Build graphs
    builder.build_function_call_graph()
    builder.build_file_dependency_graph()
    
    # Analyze graphs
    analysis = builder.analyze_complexity()
    
    # Get translation order
    leaves = builder.identify_leaf_nodes()
    sorted_funcs = builder.topological_sort_functions()
    
    # Save everything
    builder.save_graphs()
    builder.save_translation_order(sorted_funcs)
    builder.save_analysis(analysis)
    
    # Print results
    builder.print_analysis(analysis)
    
    logger.info("Dependency graph building complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())