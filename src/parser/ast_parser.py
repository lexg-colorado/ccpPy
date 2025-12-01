"""
AST parser using tree-sitter for C code.
Extracts functions, structs, includes, and call relationships.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
from dataclasses import dataclass, asdict
from tree_sitter import Language, Parser, Node, Query
import tree_sitter_c


@dataclass
class FunctionInfo:
    """Information about a C function."""
    name: str
    return_type: str
    parameters: List[Dict[str, str]]
    calls: List[str]  # Functions this function calls
    file_path: str
    start_line: int
    end_line: int
    body: str  # Function body source code


@dataclass
class StructInfo:
    """Information about a C struct."""
    name: str
    fields: List[Dict[str, str]]  # {'name': field_name, 'type': field_type}
    file_path: str
    start_line: int
    end_line: int


@dataclass
class IncludeInfo:
    """Information about include statements."""
    path: str
    is_system: bool  # True for <...>, False for "..."
    file_path: str
    line: int


class CParser:
    """Parse C source files into structured AST representation."""
    
    def __init__(self):
        """Initialize the C parser with tree-sitter."""
        self.parser = Parser()
        # Create Language object from tree-sitter-c pointer
        c_language = Language(tree_sitter_c.language(), 'c')
        self.parser.set_language(c_language)
        
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a single C file and extract all relevant information.
        
        Args:
            file_path: Path to C source file
            
        Returns:
            Dictionary containing parsed information:
            {
                'file_path': str,
                'functions': List[FunctionInfo],
                'structs': List[StructInfo],
                'includes': List[IncludeInfo],
                'source': str (original source code)
            }
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read source code
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code = f.read()
        
        # Parse to AST
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node
        
        # Extract information
        result = {
            'file_path': str(file_path),
            'functions': [asdict(f) for f in self.extract_functions(root_node, source_code, file_path)],
            'structs': [asdict(s) for s in self.extract_structs(root_node, source_code, file_path)],
            'includes': [asdict(i) for i in self.extract_includes(root_node, source_code, file_path)],
            'source': source_code
        }
        
        return result
    
    def extract_functions(self, root_node: Node, source_code: str, file_path: Path) -> List[FunctionInfo]:
        """
        Extract all function definitions from the AST.
        
        Args:
            root_node: Root node of the AST
            source_code: Original source code
            file_path: Path to the source file
            
        Returns:
            List of FunctionInfo objects
        """
        functions = []
        
        def visit_node(node: Node):
            if node.type == 'function_definition':
                # Extract function information
                func_info = self._parse_function(node, source_code, file_path)
                if func_info:
                    functions.append(func_info)
            
            # Visit children
            for child in node.children:
                visit_node(child)
        
        visit_node(root_node)
        return functions
    
    def _parse_function(self, func_node: Node, source_code: str, file_path: Path) -> Optional[FunctionInfo]:
        """Parse a single function_definition node."""
        func_name = None
        return_type = "void"
        parameters = []
        body_node = None
        
        # Find the declarator (contains function name and params)
        for child in func_node.children:
            if child.type in ['type_identifier', 'primitive_type', 'sized_type_specifier', 'struct_specifier']:
                return_type = self._get_node_text(child, source_code)
            elif 'declarator' in child.type:
                # Navigate through declarators to find function name and params
                func_name, parameters = self._extract_function_declarator(child, source_code)
            elif child.type == 'compound_statement':
                body_node = child
        
        if not func_name:
            return None
        
        # Extract function calls from body
        calls = []
        if body_node:
            calls = self._extract_function_calls(body_node, source_code)
        
        return FunctionInfo(
            name=func_name,
            return_type=return_type.strip(),
            parameters=parameters,
            calls=sorted(list(set(calls))),
            file_path=str(file_path),
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            body=self._get_node_text(func_node, source_code)
        )
    
    def _extract_function_declarator(self, declarator_node: Node, source_code: str) -> tuple:
        """Extract function name and parameters from a declarator node."""
        func_name = None
        parameters = []
        
        def find_name_and_params(node: Node):
            nonlocal func_name, parameters
            
            if node.type == 'function_declarator':
                # Find identifier and parameter_list
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = self._get_node_text(child, source_code)
                    elif child.type == 'parameter_list':
                        parameters = self._extract_parameters(child, source_code)
                    elif 'declarator' in child.type:
                        find_name_and_params(child)
            elif node.type == 'pointer_declarator':
                for child in node.children:
                    if 'declarator' in child.type:
                        find_name_and_params(child)
            elif node.type == 'identifier':
                func_name = self._get_node_text(node, source_code)
        
        find_name_and_params(declarator_node)
        return func_name, parameters
    
    def extract_structs(self, root_node: Node, source_code: str, file_path: Path) -> List[StructInfo]:
        """
        Extract all struct definitions from the AST.
        
        Args:
            root_node: Root node of the AST
            source_code: Original source code
            file_path: Path to the source file
            
        Returns:
            List of StructInfo objects
        """
        structs = []
        
        def visit_node(node: Node):
            if node.type == 'struct_specifier':
                struct_info = self._parse_struct(node, source_code, file_path)
                if struct_info:
                    structs.append(struct_info)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(root_node)
        return structs
    
    def _parse_struct(self, struct_node: Node, source_code: str, file_path: Path) -> Optional[StructInfo]:
        """Parse a single struct_specifier node."""
        struct_name = None
        fields = []
        
        for child in struct_node.children:
            if child.type == 'type_identifier':
                struct_name = self._get_node_text(child, source_code)
            elif child.type == 'field_declaration_list':
                fields = self._extract_struct_fields(child, source_code)
        
        if not struct_name:
            return None
        
        return StructInfo(
            name=struct_name,
            fields=fields,
            file_path=str(file_path),
            start_line=struct_node.start_point[0] + 1,
            end_line=struct_node.end_point[0] + 1
        )
    
    def extract_includes(self, root_node: Node, source_code: str, file_path: Path) -> List[IncludeInfo]:
        """
        Extract all #include statements.
        
        Args:
            root_node: Root node of the AST
            source_code: Original source code
            file_path: Path to the source file
            
        Returns:
            List of IncludeInfo objects
        """
        includes = []
        
        def visit_node(node: Node):
            if node.type == 'preproc_include':
                include_info = self._parse_include(node, source_code, file_path)
                if include_info:
                    includes.append(include_info)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(root_node)
        return includes
    
    def _parse_include(self, include_node: Node, source_code: str, file_path: Path) -> Optional[IncludeInfo]:
        """Parse a single preproc_include node."""
        path_text = self._get_node_text(include_node, source_code)
        
        # Extract the path from #include "..." or #include <...>
        if '"' in path_text:
            path = path_text.split('"')[1]
            is_system = False
        elif '<' in path_text:
            path = path_text.split('<')[1].split('>')[0]
            is_system = True
        else:
            return None
        
        return IncludeInfo(
            path=path,
            is_system=is_system,
            file_path=str(file_path),
            line=include_node.start_point[0] + 1
        )
    
    def _extract_parameters(self, params_node: Node, source_code: str) -> List[Dict[str, str]]:
        """Extract function parameters from parameter_list node."""
        parameters = []
        
        for child in params_node.children:
            if child.type == 'parameter_declaration':
                param_text = self._get_node_text(child, source_code)
                # Try to split into type and name
                parts = param_text.strip().split()
                if len(parts) >= 2:
                    param_type = ' '.join(parts[:-1])
                    param_name = parts[-1].rstrip(',').rstrip(';')
                    # Remove pointer/array decorators from name
                    param_name = param_name.lstrip('*').split('[')[0]
                    parameters.append({
                        'name': param_name,
                        'type': param_type
                    })
                elif len(parts) == 1:
                    # Just a type, no name (e.g., in function pointer)
                    parameters.append({
                        'name': '',
                        'type': parts[0]
                    })
        
        return parameters
    
    def _extract_function_calls(self, body_node: Node, source_code: str) -> List[str]:
        """Extract all function calls from a function body."""
        calls = []
        
        def visit_node(node: Node):
            if node.type == 'call_expression':
                # Get the function being called
                func_node = node.child_by_field_name('function')
                if func_node:
                    func_name = self._get_node_text(func_node, source_code)
                    # Handle simple identifiers and field access (obj.method)
                    if '.' not in func_name and '->' not in func_name:
                        calls.append(func_name)
            
            # Recursively visit children
            for child in node.children:
                visit_node(child)
        
        visit_node(body_node)
        return calls
    
    def _extract_struct_fields(self, fields_node: Node, source_code: str) -> List[Dict[str, str]]:
        """Extract struct fields from field_declaration_list node."""
        fields = []
        
        for child in fields_node.children:
            if child.type == 'field_declaration':
                field_text = self._get_node_text(child, source_code).strip().rstrip(';')
                
                # Try to parse type and name
                parts = field_text.split()
                if len(parts) >= 2:
                    field_type = ' '.join(parts[:-1])
                    field_name = parts[-1].lstrip('*').split('[')[0]
                    fields.append({
                        'name': field_name,
                        'type': field_type
                    })
        
        return fields
    
    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Get the text content of a node."""
        return source_code[node.start_byte:node.end_byte]
    
    def save_to_json(self, parsed_data: Dict[str, Any], output_path: Path) -> None:
        """
        Save parsed data to JSON file.
        
        Args:
            parsed_data: Dictionary from parse_file()
            output_path: Where to save the JSON
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove source code from saved data (too large)
        save_data = {k: v for k, v in parsed_data.items() if k != 'source'}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, json_path: Path) -> Dict[str, Any]:
        """
        Load parsed data from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Parsed data dictionary
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Example usage of the C parser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ast_parser.py <c_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    parser = CParser()
    print(f"Parsing {file_path}...")
    
    result = parser.parse_file(file_path)
    
    print(f"\nFound {len(result['functions'])} functions:")
    for func in result['functions']:
        print(f"  - {func['name']}() at line {func['start_line']}")
        print(f"    Returns: {func['return_type']}")
        print(f"    Params: {func['parameters']}")
        print(f"    Calls: {func['calls']}")
    
    print(f"\nFound {len(result['structs'])} structs:")
    for struct in result['structs']:
        print(f"  - {struct['name']} at line {struct['start_line']}")
        print(f"    Fields: {len(struct['fields'])}")
    
    print(f"\nFound {len(result['includes'])} includes:")
    for inc in result['includes']:
        inc_type = "system" if inc['is_system'] else "local"
        print(f"  - {inc['path']} ({inc_type})")


if __name__ == "__main__":
    main()
