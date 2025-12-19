"""
C++ AST parser using tree-sitter.

Extends the C parser with support for C++ specific constructs:
- Classes and structs with methods
- Namespaces
- Templates
- Access specifiers (public, private, protected)
- Constructors and destructors
- Operator overloading
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
from dataclasses import dataclass, asdict, field
from tree_sitter import Language, Parser, Node

# Try to import tree-sitter-cpp, provide helpful error if not available
try:
    import tree_sitter_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

from .ast_parser import FunctionInfo, StructInfo, IncludeInfo


@dataclass
class MethodInfo:
    """Information about a C++ class method."""
    name: str
    return_type: str
    parameters: List[Dict[str, str]]
    calls: List[str]
    access: str  # "public", "private", "protected"
    is_virtual: bool
    is_static: bool
    is_const: bool
    is_constructor: bool
    is_destructor: bool
    start_line: int
    end_line: int
    body: str


@dataclass
class ClassInfo:
    """Information about a C++ class."""
    name: str
    base_classes: List[Dict[str, str]]  # {'name': class_name, 'access': public/private/protected}
    methods: List[MethodInfo]
    members: List[Dict[str, str]]  # {'name': name, 'type': type, 'access': access}
    nested_classes: List[str]  # Names of nested classes
    is_struct: bool  # True if declared with 'struct' keyword
    file_path: str
    start_line: int
    end_line: int


@dataclass
class NamespaceInfo:
    """Information about a C++ namespace."""
    name: str
    functions: List[str]  # Function names in this namespace
    classes: List[str]  # Class names in this namespace
    namespaces: List[str]  # Nested namespace names
    file_path: str
    start_line: int
    end_line: int


@dataclass
class TemplateInfo:
    """Information about a C++ template."""
    name: str
    template_params: List[Dict[str, str]]  # {'name': param_name, 'type': 'typename'/'class'/type}
    kind: str  # "function", "class", "variable"
    file_path: str
    start_line: int
    end_line: int


@dataclass
class CppFileInfo:
    """Complete parsed information for a C++ file."""
    file_path: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    structs: List[StructInfo]
    namespaces: List[NamespaceInfo]
    templates: List[TemplateInfo]
    includes: List[IncludeInfo]
    using_declarations: List[str]  # e.g., "using namespace std"
    source: str = ""


class CppParser:
    """Parse C++ source files into structured AST representation."""

    def __init__(self):
        """Initialize the C++ parser with tree-sitter."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "tree-sitter-cpp is not installed. "
                "Install it with: pip install tree-sitter-cpp"
            )

        self.parser = Parser()
        cpp_language = Language(tree_sitter_cpp.language(), 'cpp')
        self.parser.set_language(cpp_language)

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a C++ file and extract all relevant information.

        Args:
            file_path: Path to C++ source file

        Returns:
            Dictionary containing parsed information
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
            'language': 'cpp',
            'functions': [asdict(f) for f in self.extract_functions(root_node, source_code, file_path)],
            'classes': [asdict(c) for c in self.extract_classes(root_node, source_code, file_path)],
            'structs': [asdict(s) for s in self.extract_structs(root_node, source_code, file_path)],
            'namespaces': [asdict(n) for n in self.extract_namespaces(root_node, source_code, file_path)],
            'templates': [asdict(t) for t in self.extract_templates(root_node, source_code, file_path)],
            'includes': [asdict(i) for i in self.extract_includes(root_node, source_code, file_path)],
            'using_declarations': self.extract_using_declarations(root_node, source_code),
            'source': source_code
        }

        return result

    def extract_functions(self, root_node: Node, source_code: str, file_path: Path) -> List[FunctionInfo]:
        """Extract free-standing function definitions (not class methods)."""
        functions = []

        def visit_node(node: Node, in_class: bool = False):
            # Skip class/struct definitions - methods handled separately
            if node.type in ['class_specifier', 'struct_specifier']:
                return

            if node.type == 'function_definition' and not in_class:
                func_info = self._parse_function(node, source_code, file_path)
                if func_info:
                    functions.append(func_info)

            for child in node.children:
                visit_node(child, in_class)

        visit_node(root_node)
        return functions

    def _parse_function(self, func_node: Node, source_code: str, file_path: Path) -> Optional[FunctionInfo]:
        """Parse a function_definition node."""
        func_name = None
        return_type = "void"
        parameters = []
        body_node = None

        for child in func_node.children:
            if child.type in ['type_identifier', 'primitive_type', 'sized_type_specifier',
                              'template_type', 'qualified_identifier', 'auto']:
                return_type = self._get_node_text(child, source_code)
            elif 'declarator' in child.type:
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
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = self._get_node_text(child, source_code)
                    elif child.type == 'qualified_identifier':
                        # Handle Class::method names
                        func_name = self._get_node_text(child, source_code)
                    elif child.type == 'field_identifier':
                        func_name = self._get_node_text(child, source_code)
                    elif child.type == 'destructor_name':
                        func_name = self._get_node_text(child, source_code)
                    elif child.type == 'parameter_list':
                        parameters = self._extract_parameters(child, source_code)
                    elif 'declarator' in child.type:
                        find_name_and_params(child)
            elif node.type == 'pointer_declarator':
                for child in node.children:
                    if 'declarator' in child.type:
                        find_name_and_params(child)
            elif node.type == 'reference_declarator':
                for child in node.children:
                    if 'declarator' in child.type:
                        find_name_and_params(child)
            elif node.type == 'identifier':
                func_name = self._get_node_text(node, source_code)

        find_name_and_params(declarator_node)
        return func_name, parameters

    def extract_classes(self, root_node: Node, source_code: str, file_path: Path) -> List[ClassInfo]:
        """Extract class definitions."""
        classes = []

        def visit_node(node: Node):
            if node.type == 'class_specifier':
                class_info = self._parse_class(node, source_code, file_path, is_struct=False)
                if class_info:
                    classes.append(class_info)
            elif node.type == 'struct_specifier':
                # C++ structs can have methods
                has_methods = self._struct_has_methods(node)
                if has_methods:
                    class_info = self._parse_class(node, source_code, file_path, is_struct=True)
                    if class_info:
                        classes.append(class_info)

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return classes

    def _struct_has_methods(self, node: Node) -> bool:
        """Check if a struct has any method declarations."""
        for child in node.children:
            if child.type == 'field_declaration_list':
                for field in child.children:
                    if field.type in ['function_definition', 'declaration']:
                        # Check if it's a function declaration
                        text = field.text.decode('utf-8') if field.text else ''
                        if '(' in text and ')' in text:
                            return True
        return False

    def _parse_class(self, class_node: Node, source_code: str, file_path: Path, is_struct: bool) -> Optional[ClassInfo]:
        """Parse a class_specifier or struct_specifier node."""
        class_name = None
        base_classes = []
        methods = []
        members = []
        nested_classes = []

        current_access = "public" if is_struct else "private"

        for child in class_node.children:
            if child.type == 'type_identifier':
                class_name = self._get_node_text(child, source_code)
            elif child.type == 'base_class_clause':
                base_classes = self._extract_base_classes(child, source_code)
            elif child.type == 'field_declaration_list':
                methods, members, nested_classes, _ = self._extract_class_body(
                    child, source_code, file_path, current_access
                )

        if not class_name:
            return None

        # Convert MethodInfo to dict for serialization
        methods_as_dict = [asdict(m) for m in methods]

        return ClassInfo(
            name=class_name,
            base_classes=base_classes,
            methods=methods,
            members=members,
            nested_classes=nested_classes,
            is_struct=is_struct,
            file_path=str(file_path),
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1
        )

    def _extract_base_classes(self, base_clause: Node, source_code: str) -> List[Dict[str, str]]:
        """Extract base class information."""
        bases = []

        for child in base_clause.children:
            if child.type == 'base_class_specifier':
                access = "public"  # Default
                name = None

                for sub in child.children:
                    if sub.type in ['public', 'private', 'protected']:
                        access = self._get_node_text(sub, source_code)
                    elif sub.type == 'type_identifier':
                        name = self._get_node_text(sub, source_code)
                    elif sub.type == 'qualified_identifier':
                        name = self._get_node_text(sub, source_code)

                if name:
                    bases.append({'name': name, 'access': access})

        return bases

    def _extract_class_body(
        self,
        body_node: Node,
        source_code: str,
        file_path: Path,
        default_access: str
    ) -> tuple:
        """Extract methods, members, and nested classes from class body."""
        methods = []
        members = []
        nested_classes = []
        current_access = default_access

        for child in body_node.children:
            if child.type == 'access_specifier':
                # Update current access level
                text = self._get_node_text(child, source_code).rstrip(':').strip()
                if text in ['public', 'private', 'protected']:
                    current_access = text

            elif child.type == 'function_definition':
                method = self._parse_method(child, source_code, file_path, current_access)
                if method:
                    methods.append(method)

            elif child.type == 'declaration':
                # Could be a method declaration or member variable
                decl_text = self._get_node_text(child, source_code)
                if '(' in decl_text and ')' in decl_text:
                    # Method declaration (no body)
                    pass  # Skip declarations without definitions for now
                else:
                    # Member variable
                    member = self._parse_member(child, source_code, current_access)
                    if member:
                        members.append(member)

            elif child.type == 'field_declaration':
                member = self._parse_member(child, source_code, current_access)
                if member:
                    members.append(member)

            elif child.type in ['class_specifier', 'struct_specifier']:
                # Nested class
                for sub in child.children:
                    if sub.type == 'type_identifier':
                        nested_classes.append(self._get_node_text(sub, source_code))

        return methods, members, nested_classes, current_access

    def _parse_method(
        self,
        method_node: Node,
        source_code: str,
        file_path: Path,
        access: str
    ) -> Optional[MethodInfo]:
        """Parse a class method definition."""
        method_name = None
        return_type = "void"
        parameters = []
        body_node = None

        is_virtual = False
        is_static = False
        is_const = False
        is_constructor = False
        is_destructor = False

        # Check for modifiers
        method_text = self._get_node_text(method_node, source_code)
        if 'virtual' in method_text.split('(')[0]:
            is_virtual = True
        if 'static' in method_text.split('(')[0]:
            is_static = True

        for child in method_node.children:
            if child.type == 'virtual':
                is_virtual = True
            elif child.type == 'static':
                is_static = True
            elif child.type in ['type_identifier', 'primitive_type', 'sized_type_specifier',
                               'template_type', 'qualified_identifier', 'auto']:
                return_type = self._get_node_text(child, source_code)
            elif 'declarator' in child.type:
                method_name, parameters = self._extract_function_declarator(child, source_code)
                # Check for const method
                decl_text = self._get_node_text(child, source_code)
                if ') const' in decl_text:
                    is_const = True
            elif child.type == 'compound_statement':
                body_node = child

        if not method_name:
            return None

        # Detect constructor/destructor
        if method_name.startswith('~'):
            is_destructor = True
            return_type = ""
        elif return_type == "" or return_type == method_name:
            is_constructor = True
            return_type = ""

        # Extract function calls from body
        calls = []
        if body_node:
            calls = self._extract_function_calls(body_node, source_code)

        return MethodInfo(
            name=method_name,
            return_type=return_type.strip(),
            parameters=parameters,
            calls=sorted(list(set(calls))),
            access=access,
            is_virtual=is_virtual,
            is_static=is_static,
            is_const=is_const,
            is_constructor=is_constructor,
            is_destructor=is_destructor,
            start_line=method_node.start_point[0] + 1,
            end_line=method_node.end_point[0] + 1,
            body=self._get_node_text(method_node, source_code)
        )

    def _parse_member(self, decl_node: Node, source_code: str, access: str) -> Optional[Dict[str, str]]:
        """Parse a member variable declaration."""
        decl_text = self._get_node_text(decl_node, source_code).strip().rstrip(';')

        # Skip function declarations
        if '(' in decl_text and ')' in decl_text:
            return None

        parts = decl_text.split()
        if len(parts) >= 2:
            # Last part is typically the name
            member_name = parts[-1].lstrip('*').split('[')[0].split('=')[0].strip()
            member_type = ' '.join(parts[:-1])
            return {
                'name': member_name,
                'type': member_type,
                'access': access
            }
        return None

    def extract_structs(self, root_node: Node, source_code: str, file_path: Path) -> List[StructInfo]:
        """Extract C-style struct definitions (without methods)."""
        structs = []

        def visit_node(node: Node):
            if node.type == 'struct_specifier':
                # Only include if it doesn't have methods (pure C-style struct)
                if not self._struct_has_methods(node):
                    struct_info = self._parse_struct(node, source_code, file_path)
                    if struct_info:
                        structs.append(struct_info)

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return structs

    def _parse_struct(self, struct_node: Node, source_code: str, file_path: Path) -> Optional[StructInfo]:
        """Parse a C-style struct_specifier node."""
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

    def _extract_struct_fields(self, fields_node: Node, source_code: str) -> List[Dict[str, str]]:
        """Extract struct fields from field_declaration_list node."""
        fields = []

        for child in fields_node.children:
            if child.type == 'field_declaration':
                field_text = self._get_node_text(child, source_code).strip().rstrip(';')

                # Skip methods
                if '(' in field_text and ')' in field_text:
                    continue

                parts = field_text.split()
                if len(parts) >= 2:
                    field_type = ' '.join(parts[:-1])
                    field_name = parts[-1].lstrip('*').split('[')[0]
                    fields.append({
                        'name': field_name,
                        'type': field_type
                    })

        return fields

    def extract_namespaces(self, root_node: Node, source_code: str, file_path: Path) -> List[NamespaceInfo]:
        """Extract namespace definitions."""
        namespaces = []

        def visit_node(node: Node):
            if node.type == 'namespace_definition':
                ns_info = self._parse_namespace(node, source_code, file_path)
                if ns_info:
                    namespaces.append(ns_info)

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return namespaces

    def _parse_namespace(self, ns_node: Node, source_code: str, file_path: Path) -> Optional[NamespaceInfo]:
        """Parse a namespace_definition node."""
        ns_name = None
        functions = []
        classes = []
        nested_namespaces = []

        for child in ns_node.children:
            if child.type == 'identifier':
                ns_name = self._get_node_text(child, source_code)
            elif child.type == 'declaration_list':
                # Extract contents
                for item in child.children:
                    if item.type == 'function_definition':
                        func = self._parse_function(item, source_code, file_path)
                        if func:
                            functions.append(func.name)
                    elif item.type == 'class_specifier':
                        for sub in item.children:
                            if sub.type == 'type_identifier':
                                classes.append(self._get_node_text(sub, source_code))
                    elif item.type == 'namespace_definition':
                        for sub in item.children:
                            if sub.type == 'identifier':
                                nested_namespaces.append(self._get_node_text(sub, source_code))

        # Handle anonymous namespace
        if not ns_name:
            ns_name = "<anonymous>"

        return NamespaceInfo(
            name=ns_name,
            functions=functions,
            classes=classes,
            namespaces=nested_namespaces,
            file_path=str(file_path),
            start_line=ns_node.start_point[0] + 1,
            end_line=ns_node.end_point[0] + 1
        )

    def extract_templates(self, root_node: Node, source_code: str, file_path: Path) -> List[TemplateInfo]:
        """Extract template definitions."""
        templates = []

        def visit_node(node: Node):
            if node.type == 'template_declaration':
                tmpl_info = self._parse_template(node, source_code, file_path)
                if tmpl_info:
                    templates.append(tmpl_info)

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return templates

    def _parse_template(self, tmpl_node: Node, source_code: str, file_path: Path) -> Optional[TemplateInfo]:
        """Parse a template_declaration node."""
        template_params = []
        name = None
        kind = "unknown"

        for child in tmpl_node.children:
            if child.type == 'template_parameter_list':
                template_params = self._extract_template_params(child, source_code)
            elif child.type == 'function_definition':
                kind = "function"
                func_info = self._parse_function(child, source_code, file_path)
                if func_info:
                    name = func_info.name
            elif child.type == 'class_specifier':
                kind = "class"
                for sub in child.children:
                    if sub.type == 'type_identifier':
                        name = self._get_node_text(sub, source_code)
            elif child.type == 'declaration':
                # Could be a variable or function declaration
                kind = "variable"
                decl_text = self._get_node_text(child, source_code)
                # Try to extract name
                if '(' in decl_text:
                    kind = "function"

        if not name:
            return None

        return TemplateInfo(
            name=name,
            template_params=template_params,
            kind=kind,
            file_path=str(file_path),
            start_line=tmpl_node.start_point[0] + 1,
            end_line=tmpl_node.end_point[0] + 1
        )

    def _extract_template_params(self, params_node: Node, source_code: str) -> List[Dict[str, str]]:
        """Extract template parameters."""
        params = []

        for child in params_node.children:
            if child.type in ['type_parameter_declaration', 'parameter_declaration']:
                param_text = self._get_node_text(child, source_code)

                # Parse "typename T" or "class T" or "int N"
                parts = param_text.split()
                if len(parts) >= 2:
                    params.append({
                        'type': parts[0],
                        'name': parts[-1]
                    })
                elif len(parts) == 1:
                    params.append({
                        'type': 'typename',
                        'name': parts[0]
                    })

        return params

    def extract_includes(self, root_node: Node, source_code: str, file_path: Path) -> List[IncludeInfo]:
        """Extract #include statements."""
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
        """Parse a preproc_include node."""
        path_text = self._get_node_text(include_node, source_code)

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

    def extract_using_declarations(self, root_node: Node, source_code: str) -> List[str]:
        """Extract using declarations and directives."""
        using_decls = []

        def visit_node(node: Node):
            if node.type == 'using_declaration':
                using_decls.append(self._get_node_text(node, source_code).rstrip(';'))
            elif node.type == 'alias_declaration':
                using_decls.append(self._get_node_text(node, source_code).rstrip(';'))

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return using_decls

    def _extract_parameters(self, params_node: Node, source_code: str) -> List[Dict[str, str]]:
        """Extract function parameters from parameter_list node."""
        parameters = []

        for child in params_node.children:
            if child.type == 'parameter_declaration':
                param_text = self._get_node_text(child, source_code)
                parts = param_text.strip().split()

                if len(parts) >= 2:
                    param_type = ' '.join(parts[:-1])
                    param_name = parts[-1].rstrip(',').rstrip(';')
                    param_name = param_name.lstrip('*').lstrip('&').split('[')[0].split('=')[0].strip()
                    parameters.append({
                        'name': param_name,
                        'type': param_type
                    })
                elif len(parts) == 1:
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
                func_node = node.child_by_field_name('function')
                if func_node:
                    func_name = self._get_node_text(func_node, source_code)
                    # Handle simple identifiers, exclude complex expressions
                    if func_name.isidentifier() or '::' in func_name:
                        calls.append(func_name)

            for child in node.children:
                visit_node(child)

        visit_node(body_node)
        return calls

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Get the text content of a node."""
        return source_code[node.start_byte:node.end_byte]

    def save_to_json(self, parsed_data: Dict[str, Any], output_path: Path) -> None:
        """Save parsed data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove source code from saved data
        save_data = {k: v for k, v in parsed_data.items() if k != 'source'}

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, json_path: Path) -> Dict[str, Any]:
        """Load parsed data from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Example usage of the C++ parser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cpp_parser.py <cpp_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    parser = CppParser()
    print(f"Parsing {file_path}...")

    result = parser.parse_file(file_path)

    print(f"\nFound {len(result['functions'])} functions:")
    for func in result['functions']:
        print(f"  - {func['name']}() at line {func['start_line']}")

    print(f"\nFound {len(result['classes'])} classes:")
    for cls in result['classes']:
        print(f"  - {cls['name']} at line {cls['start_line']}")
        print(f"    Methods: {len(cls['methods'])}")
        print(f"    Members: {len(cls['members'])}")

    print(f"\nFound {len(result['namespaces'])} namespaces:")
    for ns in result['namespaces']:
        print(f"  - {ns['name']} at line {ns['start_line']}")

    print(f"\nFound {len(result['templates'])} templates:")
    for tmpl in result['templates']:
        print(f"  - {tmpl['name']} ({tmpl['kind']}) at line {tmpl['start_line']}")

    print(f"\nFound {len(result['includes'])} includes")
    print(f"Found {len(result['using_declarations'])} using declarations")


if __name__ == "__main__":
    main()
