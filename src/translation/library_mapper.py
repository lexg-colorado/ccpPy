"""
Pluggable library mapping system for C/C++ to Python translation.

This module provides a framework for mapping C/C++ library functions and types
to their Python equivalents. It supports:
- Built-in mappers for common libraries (stdlib, ncurses, pthread)
- Custom YAML-based mappings for project-specific libraries
- Type mappings for C types to Python types
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


@dataclass
class LibraryMapping:
    """A single library/function mapping from C to Python."""
    c_library: str          # e.g., "stdio.h"
    c_function: str         # e.g., "printf"
    python_module: str      # e.g., "builtins"
    python_function: str    # e.g., "print"
    signature_transform: Optional[str] = None  # Optional transform rules
    notes: str = ""         # Usage notes for the LLM


@dataclass
class TypeMapping:
    """C/C++ type to Python type mapping."""
    c_type: str             # e.g., "int", "char*", "FILE*"
    python_type: str        # e.g., "int", "str", "IO"
    import_required: Optional[str] = None  # e.g., "typing" for IO
    transform: Optional[str] = None  # Optional transform notes
    notes: str = ""         # Usage notes for the LLM


class LibraryMappingProvider(ABC):
    """Abstract base class for library mapping providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @abstractmethod
    def get_function_mapping(
        self,
        c_function: str,
        c_header: Optional[str] = None
    ) -> Optional[LibraryMapping]:
        """
        Get Python equivalent for a C function.

        Args:
            c_function: Name of the C function
            c_header: Optional header file for disambiguation

        Returns:
            LibraryMapping if found, None otherwise
        """
        pass

    @abstractmethod
    def get_type_mapping(self, c_type: str) -> Optional[TypeMapping]:
        """
        Get Python type for a C type.

        Args:
            c_type: C type string (e.g., "int", "char*")

        Returns:
            TypeMapping if found, None otherwise
        """
        pass

    def get_all_function_mappings(self) -> List[LibraryMapping]:
        """Get all function mappings from this provider."""
        return []

    def get_all_type_mappings(self) -> List[TypeMapping]:
        """Get all type mappings from this provider."""
        return []


class StandardLibraryMapper(LibraryMappingProvider):
    """Mappings for C standard library functions."""

    @property
    def name(self) -> str:
        return "stdlib"

    FUNCTION_MAPPINGS: Dict[str, LibraryMapping] = {
        # stdio.h - Input/Output
        'printf': LibraryMapping('stdio.h', 'printf', 'builtins', 'print',
                                 notes='Use print() with f-strings for formatting'),
        'fprintf': LibraryMapping('stdio.h', 'fprintf', 'builtins', 'print',
                                  notes='Use print(file=f) for file output'),
        'sprintf': LibraryMapping('stdio.h', 'sprintf', 'builtins', 'format',
                                  notes='Use f-strings or str.format()'),
        'snprintf': LibraryMapping('stdio.h', 'snprintf', 'builtins', 'format',
                                   notes='Use f-strings, Python handles length automatically'),
        'fopen': LibraryMapping('stdio.h', 'fopen', 'builtins', 'open',
                                notes='Use open() with context manager (with statement)'),
        'fclose': LibraryMapping('stdio.h', 'fclose', 'builtins', 'close',
                                 notes='Not needed with context manager'),
        'fread': LibraryMapping('stdio.h', 'fread', 'io', 'read',
                                notes='Use file.read() method'),
        'fwrite': LibraryMapping('stdio.h', 'fwrite', 'io', 'write',
                                 notes='Use file.write() method'),
        'fgets': LibraryMapping('stdio.h', 'fgets', 'io', 'readline',
                                notes='Use file.readline() method'),
        'fputs': LibraryMapping('stdio.h', 'fputs', 'io', 'write',
                                notes='Use file.write() method'),
        'fseek': LibraryMapping('stdio.h', 'fseek', 'io', 'seek',
                                notes='Use file.seek() method'),
        'ftell': LibraryMapping('stdio.h', 'ftell', 'io', 'tell',
                                notes='Use file.tell() method'),
        'fflush': LibraryMapping('stdio.h', 'fflush', 'io', 'flush',
                                 notes='Use file.flush() method'),
        'feof': LibraryMapping('stdio.h', 'feof', 'builtins', 'None',
                               notes='Check for empty read result instead'),
        'ferror': LibraryMapping('stdio.h', 'ferror', 'builtins', 'None',
                                 notes='Use try/except for error handling'),
        'getchar': LibraryMapping('stdio.h', 'getchar', 'sys', 'stdin.read',
                                  notes='Use sys.stdin.read(1)'),
        'putchar': LibraryMapping('stdio.h', 'putchar', 'sys', 'stdout.write',
                                  notes='Use sys.stdout.write()'),
        'puts': LibraryMapping('stdio.h', 'puts', 'builtins', 'print',
                               notes='Use print()'),
        'gets': LibraryMapping('stdio.h', 'gets', 'builtins', 'input',
                               notes='Use input() - gets() is unsafe in C too'),
        'scanf': LibraryMapping('stdio.h', 'scanf', 'builtins', 'input',
                                notes='Use input() with type conversion'),
        'sscanf': LibraryMapping('stdio.h', 'sscanf', 're', 'match',
                                 notes='Use regex or string parsing'),
        'perror': LibraryMapping('stdio.h', 'perror', 'sys', 'stderr.write',
                                 notes='Use print(..., file=sys.stderr)'),

        # stdlib.h - General utilities
        'malloc': LibraryMapping('stdlib.h', 'malloc', 'builtins', 'None',
                                 notes='Python handles memory automatically'),
        'calloc': LibraryMapping('stdlib.h', 'calloc', 'builtins', 'None',
                                 notes='Python handles memory automatically'),
        'realloc': LibraryMapping('stdlib.h', 'realloc', 'builtins', 'None',
                                  notes='Python handles memory automatically'),
        'free': LibraryMapping('stdlib.h', 'free', 'builtins', 'del',
                               notes='Use del or let garbage collector handle it'),
        'exit': LibraryMapping('stdlib.h', 'exit', 'sys', 'exit',
                               notes='Use sys.exit()'),
        'abort': LibraryMapping('stdlib.h', 'abort', 'os', '_exit',
                                notes='Use os._exit() for immediate termination'),
        'atexit': LibraryMapping('stdlib.h', 'atexit', 'atexit', 'register',
                                 notes='Use atexit.register()'),
        'atoi': LibraryMapping('stdlib.h', 'atoi', 'builtins', 'int',
                               notes='Use int()'),
        'atol': LibraryMapping('stdlib.h', 'atol', 'builtins', 'int',
                               notes='Use int()'),
        'atof': LibraryMapping('stdlib.h', 'atof', 'builtins', 'float',
                               notes='Use float()'),
        'strtol': LibraryMapping('stdlib.h', 'strtol', 'builtins', 'int',
                                 notes='Use int(s, base)'),
        'strtod': LibraryMapping('stdlib.h', 'strtod', 'builtins', 'float',
                                 notes='Use float()'),
        'rand': LibraryMapping('stdlib.h', 'rand', 'random', 'randint',
                               notes='Use random.randint()'),
        'srand': LibraryMapping('stdlib.h', 'srand', 'random', 'seed',
                                notes='Use random.seed()'),
        'abs': LibraryMapping('stdlib.h', 'abs', 'builtins', 'abs',
                              notes='Use abs()'),
        'qsort': LibraryMapping('stdlib.h', 'qsort', 'builtins', 'sorted',
                                notes='Use sorted() or list.sort()'),
        'bsearch': LibraryMapping('stdlib.h', 'bsearch', 'bisect', 'bisect',
                                  notes='Use bisect module'),
        'getenv': LibraryMapping('stdlib.h', 'getenv', 'os', 'environ.get',
                                 notes='Use os.environ.get()'),
        'setenv': LibraryMapping('stdlib.h', 'setenv', 'os', 'environ',
                                 notes='Use os.environ[key] = value'),
        'system': LibraryMapping('stdlib.h', 'system', 'subprocess', 'run',
                                 notes='Use subprocess.run()'),

        # string.h - String handling
        'strlen': LibraryMapping('string.h', 'strlen', 'builtins', 'len',
                                 notes='Use len()'),
        'strcpy': LibraryMapping('string.h', 'strcpy', 'builtins', 'str',
                                 notes='Use assignment or str()'),
        'strncpy': LibraryMapping('string.h', 'strncpy', 'builtins', 'str',
                                  notes='Use slicing: s[:n]'),
        'strcat': LibraryMapping('string.h', 'strcat', 'builtins', '+',
                                 notes='Use string concatenation (+)'),
        'strncat': LibraryMapping('string.h', 'strncat', 'builtins', '+',
                                  notes='Use s1 + s2[:n]'),
        'strcmp': LibraryMapping('string.h', 'strcmp', 'builtins', '==',
                                 notes='Use == for equality, < > for comparison'),
        'strncmp': LibraryMapping('string.h', 'strncmp', 'builtins', '==',
                                  notes='Use s1[:n] == s2[:n]'),
        'strchr': LibraryMapping('string.h', 'strchr', 'str', 'find',
                                 notes='Use str.find() or str.index()'),
        'strrchr': LibraryMapping('string.h', 'strrchr', 'str', 'rfind',
                                  notes='Use str.rfind()'),
        'strstr': LibraryMapping('string.h', 'strstr', 'str', 'find',
                                 notes='Use str.find() or "in" operator'),
        'strtok': LibraryMapping('string.h', 'strtok', 'str', 'split',
                                 notes='Use str.split()'),
        'memcpy': LibraryMapping('string.h', 'memcpy', 'builtins', 'slice',
                                 notes='Use slicing or copy()'),
        'memmove': LibraryMapping('string.h', 'memmove', 'builtins', 'slice',
                                  notes='Use slicing or copy()'),
        'memset': LibraryMapping('string.h', 'memset', 'builtins', '*',
                                 notes='Use [value] * n or bytearray'),
        'memcmp': LibraryMapping('string.h', 'memcmp', 'builtins', '==',
                                 notes='Use == for comparison'),

        # math.h - Mathematics
        'sin': LibraryMapping('math.h', 'sin', 'math', 'sin'),
        'cos': LibraryMapping('math.h', 'cos', 'math', 'cos'),
        'tan': LibraryMapping('math.h', 'tan', 'math', 'tan'),
        'asin': LibraryMapping('math.h', 'asin', 'math', 'asin'),
        'acos': LibraryMapping('math.h', 'acos', 'math', 'acos'),
        'atan': LibraryMapping('math.h', 'atan', 'math', 'atan'),
        'atan2': LibraryMapping('math.h', 'atan2', 'math', 'atan2'),
        'sinh': LibraryMapping('math.h', 'sinh', 'math', 'sinh'),
        'cosh': LibraryMapping('math.h', 'cosh', 'math', 'cosh'),
        'tanh': LibraryMapping('math.h', 'tanh', 'math', 'tanh'),
        'exp': LibraryMapping('math.h', 'exp', 'math', 'exp'),
        'log': LibraryMapping('math.h', 'log', 'math', 'log'),
        'log10': LibraryMapping('math.h', 'log10', 'math', 'log10'),
        'log2': LibraryMapping('math.h', 'log2', 'math', 'log2'),
        'pow': LibraryMapping('math.h', 'pow', 'math', 'pow',
                              notes='Can also use ** operator'),
        'sqrt': LibraryMapping('math.h', 'sqrt', 'math', 'sqrt'),
        'ceil': LibraryMapping('math.h', 'ceil', 'math', 'ceil'),
        'floor': LibraryMapping('math.h', 'floor', 'math', 'floor'),
        'fabs': LibraryMapping('math.h', 'fabs', 'builtins', 'abs',
                               notes='Use abs() for floats too'),
        'fmod': LibraryMapping('math.h', 'fmod', 'math', 'fmod',
                               notes='Can also use % operator'),
        'round': LibraryMapping('math.h', 'round', 'builtins', 'round'),

        # ctype.h - Character handling
        'isalpha': LibraryMapping('ctype.h', 'isalpha', 'str', 'isalpha'),
        'isdigit': LibraryMapping('ctype.h', 'isdigit', 'str', 'isdigit'),
        'isalnum': LibraryMapping('ctype.h', 'isalnum', 'str', 'isalnum'),
        'isspace': LibraryMapping('ctype.h', 'isspace', 'str', 'isspace'),
        'isupper': LibraryMapping('ctype.h', 'isupper', 'str', 'isupper'),
        'islower': LibraryMapping('ctype.h', 'islower', 'str', 'islower'),
        'toupper': LibraryMapping('ctype.h', 'toupper', 'str', 'upper'),
        'tolower': LibraryMapping('ctype.h', 'tolower', 'str', 'lower'),

        # time.h - Date and time
        'time': LibraryMapping('time.h', 'time', 'time', 'time'),
        'clock': LibraryMapping('time.h', 'clock', 'time', 'perf_counter',
                                notes='Use time.perf_counter() for timing'),
        'difftime': LibraryMapping('time.h', 'difftime', 'builtins', '-',
                                   notes='Use subtraction'),
        'mktime': LibraryMapping('time.h', 'mktime', 'time', 'mktime'),
        'strftime': LibraryMapping('time.h', 'strftime', 'time', 'strftime'),
        'localtime': LibraryMapping('time.h', 'localtime', 'time', 'localtime'),
        'gmtime': LibraryMapping('time.h', 'gmtime', 'time', 'gmtime'),
        'sleep': LibraryMapping('unistd.h', 'sleep', 'time', 'sleep'),
        'usleep': LibraryMapping('unistd.h', 'usleep', 'time', 'sleep',
                                 notes='Use time.sleep(microseconds/1000000)'),

        # assert.h
        'assert': LibraryMapping('assert.h', 'assert', 'builtins', 'assert'),

        # errno.h
        'errno': LibraryMapping('errno.h', 'errno', 'builtins', 'Exception',
                                notes='Use try/except for error handling'),
    }

    TYPE_MAPPINGS: Dict[str, TypeMapping] = {
        # Basic types
        'int': TypeMapping('int', 'int'),
        'short': TypeMapping('short', 'int'),
        'long': TypeMapping('long', 'int'),
        'long long': TypeMapping('long long', 'int'),
        'unsigned int': TypeMapping('unsigned int', 'int'),
        'unsigned short': TypeMapping('unsigned short', 'int'),
        'unsigned long': TypeMapping('unsigned long', 'int'),
        'unsigned char': TypeMapping('unsigned char', 'int'),
        'signed char': TypeMapping('signed char', 'int'),
        'float': TypeMapping('float', 'float'),
        'double': TypeMapping('double', 'float'),
        'long double': TypeMapping('long double', 'float'),
        'char': TypeMapping('char', 'str',
                           notes='Single character'),
        'char*': TypeMapping('char*', 'str'),
        'const char*': TypeMapping('const char*', 'str'),
        'void': TypeMapping('void', 'None'),
        'void*': TypeMapping('void*', 'Any', 'typing'),
        'bool': TypeMapping('bool', 'bool'),
        '_Bool': TypeMapping('_Bool', 'bool'),

        # Size types
        'size_t': TypeMapping('size_t', 'int'),
        'ssize_t': TypeMapping('ssize_t', 'int'),
        'ptrdiff_t': TypeMapping('ptrdiff_t', 'int'),
        'intptr_t': TypeMapping('intptr_t', 'int'),
        'uintptr_t': TypeMapping('uintptr_t', 'int'),

        # Fixed-width integers
        'int8_t': TypeMapping('int8_t', 'int'),
        'int16_t': TypeMapping('int16_t', 'int'),
        'int32_t': TypeMapping('int32_t', 'int'),
        'int64_t': TypeMapping('int64_t', 'int'),
        'uint8_t': TypeMapping('uint8_t', 'int'),
        'uint16_t': TypeMapping('uint16_t', 'int'),
        'uint32_t': TypeMapping('uint32_t', 'int'),
        'uint64_t': TypeMapping('uint64_t', 'int'),

        # File types
        'FILE': TypeMapping('FILE', 'IO', 'typing'),
        'FILE*': TypeMapping('FILE*', 'IO', 'typing'),

        # Time types
        'time_t': TypeMapping('time_t', 'float',
                             notes='Unix timestamp'),
        'clock_t': TypeMapping('clock_t', 'float'),
        'struct tm': TypeMapping('struct tm', 'time.struct_time', 'time'),
    }

    def get_function_mapping(
        self,
        c_function: str,
        c_header: Optional[str] = None
    ) -> Optional[LibraryMapping]:
        return self.FUNCTION_MAPPINGS.get(c_function)

    def get_type_mapping(self, c_type: str) -> Optional[TypeMapping]:
        # Clean up type string
        clean_type = c_type.replace('const ', '').strip()
        return self.TYPE_MAPPINGS.get(clean_type)

    def get_all_function_mappings(self) -> List[LibraryMapping]:
        return list(self.FUNCTION_MAPPINGS.values())

    def get_all_type_mappings(self) -> List[TypeMapping]:
        return list(self.TYPE_MAPPINGS.values())


class NCursesMapper(LibraryMappingProvider):
    """Mappings for ncurses library to Python curses."""

    @property
    def name(self) -> str:
        return "ncurses"

    FUNCTION_MAPPINGS: Dict[str, LibraryMapping] = {
        # Initialization
        'initscr': LibraryMapping('ncurses.h', 'initscr', 'curses', 'initscr'),
        'endwin': LibraryMapping('ncurses.h', 'endwin', 'curses', 'endwin'),
        'isendwin': LibraryMapping('ncurses.h', 'isendwin', 'curses', 'isendwin'),
        'newterm': LibraryMapping('ncurses.h', 'newterm', 'curses', 'initscr',
                                  notes='Python curses uses initscr()'),

        # Window management
        'newwin': LibraryMapping('ncurses.h', 'newwin', 'curses', 'newwin'),
        'delwin': LibraryMapping('ncurses.h', 'delwin', 'curses', 'delwin'),
        'subwin': LibraryMapping('ncurses.h', 'subwin', 'curses', 'subwin'),
        'derwin': LibraryMapping('ncurses.h', 'derwin', 'curses', 'derwin'),
        'mvwin': LibraryMapping('ncurses.h', 'mvwin', 'curses', 'mvwin'),
        'dupwin': LibraryMapping('ncurses.h', 'dupwin', 'curses', 'dupwin'),

        # Output
        'addch': LibraryMapping('ncurses.h', 'addch', 'curses', 'addch'),
        'mvaddch': LibraryMapping('ncurses.h', 'mvaddch', 'curses', 'addch',
                                  notes='Use window.addch(y, x, ch)'),
        'addstr': LibraryMapping('ncurses.h', 'addstr', 'curses', 'addstr'),
        'mvaddstr': LibraryMapping('ncurses.h', 'mvaddstr', 'curses', 'addstr',
                                   notes='Use window.addstr(y, x, str)'),
        'addnstr': LibraryMapping('ncurses.h', 'addnstr', 'curses', 'addnstr'),
        'printw': LibraryMapping('ncurses.h', 'printw', 'curses', 'addstr',
                                 notes='Use addstr with f-string formatting'),
        'mvprintw': LibraryMapping('ncurses.h', 'mvprintw', 'curses', 'addstr',
                                   notes='Use window.addstr(y, x, f"...")'),
        'wprintw': LibraryMapping('ncurses.h', 'wprintw', 'curses', 'addstr'),

        # Input
        'getch': LibraryMapping('ncurses.h', 'getch', 'curses', 'getch'),
        'wgetch': LibraryMapping('ncurses.h', 'wgetch', 'curses', 'getch'),
        'mvgetch': LibraryMapping('ncurses.h', 'mvgetch', 'curses', 'getch'),
        'getstr': LibraryMapping('ncurses.h', 'getstr', 'curses', 'getstr'),
        'wgetstr': LibraryMapping('ncurses.h', 'wgetstr', 'curses', 'getstr'),
        'ungetch': LibraryMapping('ncurses.h', 'ungetch', 'curses', 'ungetch'),

        # Refresh
        'refresh': LibraryMapping('ncurses.h', 'refresh', 'curses', 'refresh'),
        'wrefresh': LibraryMapping('ncurses.h', 'wrefresh', 'curses', 'refresh'),
        'wnoutrefresh': LibraryMapping('ncurses.h', 'wnoutrefresh', 'curses', 'noutrefresh'),
        'doupdate': LibraryMapping('ncurses.h', 'doupdate', 'curses', 'doupdate'),

        # Clear
        'clear': LibraryMapping('ncurses.h', 'clear', 'curses', 'clear'),
        'wclear': LibraryMapping('ncurses.h', 'wclear', 'curses', 'clear'),
        'erase': LibraryMapping('ncurses.h', 'erase', 'curses', 'erase'),
        'werase': LibraryMapping('ncurses.h', 'werase', 'curses', 'erase'),
        'clrtobot': LibraryMapping('ncurses.h', 'clrtobot', 'curses', 'clrtobot'),
        'clrtoeol': LibraryMapping('ncurses.h', 'clrtoeol', 'curses', 'clrtoeol'),

        # Cursor
        'move': LibraryMapping('ncurses.h', 'move', 'curses', 'move'),
        'wmove': LibraryMapping('ncurses.h', 'wmove', 'curses', 'move'),
        'getyx': LibraryMapping('ncurses.h', 'getyx', 'curses', 'getyx'),
        'getmaxyx': LibraryMapping('ncurses.h', 'getmaxyx', 'curses', 'getmaxyx'),
        'getbegyx': LibraryMapping('ncurses.h', 'getbegyx', 'curses', 'getbegyx'),

        # Attributes
        'attron': LibraryMapping('ncurses.h', 'attron', 'curses', 'attron'),
        'attroff': LibraryMapping('ncurses.h', 'attroff', 'curses', 'attroff'),
        'attrset': LibraryMapping('ncurses.h', 'attrset', 'curses', 'attrset'),
        'wattron': LibraryMapping('ncurses.h', 'wattron', 'curses', 'attron'),
        'wattroff': LibraryMapping('ncurses.h', 'wattroff', 'curses', 'attroff'),
        'wattrset': LibraryMapping('ncurses.h', 'wattrset', 'curses', 'attrset'),

        # Color
        'start_color': LibraryMapping('ncurses.h', 'start_color', 'curses', 'start_color'),
        'init_pair': LibraryMapping('ncurses.h', 'init_pair', 'curses', 'init_pair'),
        'init_color': LibraryMapping('ncurses.h', 'init_color', 'curses', 'init_color'),
        'color_pair': LibraryMapping('ncurses.h', 'color_pair', 'curses', 'color_pair'),
        'has_colors': LibraryMapping('ncurses.h', 'has_colors', 'curses', 'has_colors'),
        'can_change_color': LibraryMapping('ncurses.h', 'can_change_color', 'curses', 'can_change_color'),

        # Input modes
        'cbreak': LibraryMapping('ncurses.h', 'cbreak', 'curses', 'cbreak'),
        'nocbreak': LibraryMapping('ncurses.h', 'nocbreak', 'curses', 'nocbreak'),
        'echo': LibraryMapping('ncurses.h', 'echo', 'curses', 'echo'),
        'noecho': LibraryMapping('ncurses.h', 'noecho', 'curses', 'noecho'),
        'raw': LibraryMapping('ncurses.h', 'raw', 'curses', 'raw'),
        'noraw': LibraryMapping('ncurses.h', 'noraw', 'curses', 'noraw'),
        'keypad': LibraryMapping('ncurses.h', 'keypad', 'curses', 'keypad'),
        'nodelay': LibraryMapping('ncurses.h', 'nodelay', 'curses', 'nodelay'),
        'halfdelay': LibraryMapping('ncurses.h', 'halfdelay', 'curses', 'halfdelay'),
        'timeout': LibraryMapping('ncurses.h', 'timeout', 'curses', 'timeout'),
        'wtimeout': LibraryMapping('ncurses.h', 'wtimeout', 'curses', 'timeout'),

        # Misc
        'curs_set': LibraryMapping('ncurses.h', 'curs_set', 'curses', 'curs_set'),
        'beep': LibraryMapping('ncurses.h', 'beep', 'curses', 'beep'),
        'flash': LibraryMapping('ncurses.h', 'flash', 'curses', 'flash'),
        'napms': LibraryMapping('ncurses.h', 'napms', 'curses', 'napms'),
        'box': LibraryMapping('ncurses.h', 'box', 'curses', 'box'),
        'border': LibraryMapping('ncurses.h', 'border', 'curses', 'border'),
        'hline': LibraryMapping('ncurses.h', 'hline', 'curses', 'hline'),
        'vline': LibraryMapping('ncurses.h', 'vline', 'curses', 'vline'),
    }

    TYPE_MAPPINGS: Dict[str, TypeMapping] = {
        'WINDOW': TypeMapping('WINDOW', 'curses.window', 'curses'),
        'WINDOW*': TypeMapping('WINDOW*', 'curses.window', 'curses'),
        'SCREEN': TypeMapping('SCREEN', 'curses.window', 'curses'),
        'SCREEN*': TypeMapping('SCREEN*', 'curses.window', 'curses'),
        'chtype': TypeMapping('chtype', 'int'),
        'attr_t': TypeMapping('attr_t', 'int'),
        'MEVENT': TypeMapping('MEVENT', 'tuple',
                             notes='Mouse event as (id, x, y, z, bstate)'),
    }

    def get_function_mapping(
        self,
        c_function: str,
        c_header: Optional[str] = None
    ) -> Optional[LibraryMapping]:
        return self.FUNCTION_MAPPINGS.get(c_function)

    def get_type_mapping(self, c_type: str) -> Optional[TypeMapping]:
        return self.TYPE_MAPPINGS.get(c_type)

    def get_all_function_mappings(self) -> List[LibraryMapping]:
        return list(self.FUNCTION_MAPPINGS.values())

    def get_all_type_mappings(self) -> List[TypeMapping]:
        return list(self.TYPE_MAPPINGS.values())


class PthreadMapper(LibraryMappingProvider):
    """Mappings for POSIX threads library to Python threading."""

    @property
    def name(self) -> str:
        return "pthread"

    FUNCTION_MAPPINGS: Dict[str, LibraryMapping] = {
        # Thread management
        'pthread_create': LibraryMapping('pthread.h', 'pthread_create',
                                         'threading', 'Thread',
                                         notes='Use threading.Thread(target=func, args=())'),
        'pthread_join': LibraryMapping('pthread.h', 'pthread_join',
                                       'threading', 'Thread.join'),
        'pthread_exit': LibraryMapping('pthread.h', 'pthread_exit',
                                       'builtins', 'return',
                                       notes='Just return from thread function'),
        'pthread_self': LibraryMapping('pthread.h', 'pthread_self',
                                       'threading', 'current_thread'),
        'pthread_equal': LibraryMapping('pthread.h', 'pthread_equal',
                                        'builtins', '==',
                                        notes='Use == to compare threads'),
        'pthread_detach': LibraryMapping('pthread.h', 'pthread_detach',
                                         'threading', 'Thread.daemon',
                                         notes='Set thread.daemon = True before start'),
        'pthread_cancel': LibraryMapping('pthread.h', 'pthread_cancel',
                                         'threading', 'Event',
                                         notes='Use Event for cancellation signaling'),

        # Mutex
        'pthread_mutex_init': LibraryMapping('pthread.h', 'pthread_mutex_init',
                                             'threading', 'Lock',
                                             notes='Use threading.Lock()'),
        'pthread_mutex_destroy': LibraryMapping('pthread.h', 'pthread_mutex_destroy',
                                                'builtins', 'None',
                                                notes='Python handles cleanup'),
        'pthread_mutex_lock': LibraryMapping('pthread.h', 'pthread_mutex_lock',
                                             'threading', 'Lock.acquire'),
        'pthread_mutex_trylock': LibraryMapping('pthread.h', 'pthread_mutex_trylock',
                                                'threading', 'Lock.acquire',
                                                notes='Use lock.acquire(blocking=False)'),
        'pthread_mutex_unlock': LibraryMapping('pthread.h', 'pthread_mutex_unlock',
                                               'threading', 'Lock.release'),

        # Read-write locks
        'pthread_rwlock_init': LibraryMapping('pthread.h', 'pthread_rwlock_init',
                                              'threading', 'RLock',
                                              notes='Use RLock or custom implementation'),
        'pthread_rwlock_rdlock': LibraryMapping('pthread.h', 'pthread_rwlock_rdlock',
                                                'threading', 'RLock.acquire'),
        'pthread_rwlock_wrlock': LibraryMapping('pthread.h', 'pthread_rwlock_wrlock',
                                                'threading', 'RLock.acquire'),
        'pthread_rwlock_unlock': LibraryMapping('pthread.h', 'pthread_rwlock_unlock',
                                                'threading', 'RLock.release'),

        # Condition variables
        'pthread_cond_init': LibraryMapping('pthread.h', 'pthread_cond_init',
                                            'threading', 'Condition',
                                            notes='Use threading.Condition()'),
        'pthread_cond_destroy': LibraryMapping('pthread.h', 'pthread_cond_destroy',
                                               'builtins', 'None'),
        'pthread_cond_wait': LibraryMapping('pthread.h', 'pthread_cond_wait',
                                            'threading', 'Condition.wait'),
        'pthread_cond_timedwait': LibraryMapping('pthread.h', 'pthread_cond_timedwait',
                                                 'threading', 'Condition.wait',
                                                 notes='Use cond.wait(timeout=seconds)'),
        'pthread_cond_signal': LibraryMapping('pthread.h', 'pthread_cond_signal',
                                              'threading', 'Condition.notify'),
        'pthread_cond_broadcast': LibraryMapping('pthread.h', 'pthread_cond_broadcast',
                                                 'threading', 'Condition.notify_all'),

        # Semaphores
        'sem_init': LibraryMapping('semaphore.h', 'sem_init',
                                   'threading', 'Semaphore'),
        'sem_destroy': LibraryMapping('semaphore.h', 'sem_destroy',
                                      'builtins', 'None'),
        'sem_wait': LibraryMapping('semaphore.h', 'sem_wait',
                                   'threading', 'Semaphore.acquire'),
        'sem_trywait': LibraryMapping('semaphore.h', 'sem_trywait',
                                      'threading', 'Semaphore.acquire',
                                      notes='Use sem.acquire(blocking=False)'),
        'sem_post': LibraryMapping('semaphore.h', 'sem_post',
                                   'threading', 'Semaphore.release'),

        # Thread-local storage
        'pthread_key_create': LibraryMapping('pthread.h', 'pthread_key_create',
                                             'threading', 'local',
                                             notes='Use threading.local()'),
        'pthread_setspecific': LibraryMapping('pthread.h', 'pthread_setspecific',
                                              'threading', 'local',
                                              notes='Use local_data.attr = value'),
        'pthread_getspecific': LibraryMapping('pthread.h', 'pthread_getspecific',
                                              'threading', 'local',
                                              notes='Use local_data.attr'),

        # Barriers
        'pthread_barrier_init': LibraryMapping('pthread.h', 'pthread_barrier_init',
                                               'threading', 'Barrier'),
        'pthread_barrier_wait': LibraryMapping('pthread.h', 'pthread_barrier_wait',
                                               'threading', 'Barrier.wait'),
        'pthread_barrier_destroy': LibraryMapping('pthread.h', 'pthread_barrier_destroy',
                                                  'builtins', 'None'),
    }

    TYPE_MAPPINGS: Dict[str, TypeMapping] = {
        'pthread_t': TypeMapping('pthread_t', 'Thread', 'threading'),
        'pthread_mutex_t': TypeMapping('pthread_mutex_t', 'Lock', 'threading'),
        'pthread_rwlock_t': TypeMapping('pthread_rwlock_t', 'RLock', 'threading'),
        'pthread_cond_t': TypeMapping('pthread_cond_t', 'Condition', 'threading'),
        'pthread_key_t': TypeMapping('pthread_key_t', 'local', 'threading'),
        'pthread_barrier_t': TypeMapping('pthread_barrier_t', 'Barrier', 'threading'),
        'pthread_attr_t': TypeMapping('pthread_attr_t', 'dict',
                                     notes='Thread attributes as dict'),
        'sem_t': TypeMapping('sem_t', 'Semaphore', 'threading'),
    }

    def get_function_mapping(
        self,
        c_function: str,
        c_header: Optional[str] = None
    ) -> Optional[LibraryMapping]:
        return self.FUNCTION_MAPPINGS.get(c_function)

    def get_type_mapping(self, c_type: str) -> Optional[TypeMapping]:
        return self.TYPE_MAPPINGS.get(c_type)

    def get_all_function_mappings(self) -> List[LibraryMapping]:
        return list(self.FUNCTION_MAPPINGS.values())

    def get_all_type_mappings(self) -> List[TypeMapping]:
        return list(self.TYPE_MAPPINGS.values())


class CustomMappingProvider(LibraryMappingProvider):
    """Provider for custom user-defined mappings from YAML files."""

    def __init__(self, mapping_data: Dict[str, Any]):
        """
        Initialize with mapping data.

        Args:
            mapping_data: Dictionary loaded from YAML with 'name', 'functions', 'types'
        """
        self._name = mapping_data.get('name', 'custom')
        self._functions: Dict[str, LibraryMapping] = {}
        self._types: Dict[str, TypeMapping] = {}

        # Load function mappings
        for func_map in mapping_data.get('functions', []):
            key = func_map['c_function']
            self._functions[key] = LibraryMapping(
                c_library=func_map.get('c_library', ''),
                c_function=func_map['c_function'],
                python_module=func_map['python_module'],
                python_function=func_map['python_function'],
                signature_transform=func_map.get('signature_transform'),
                notes=func_map.get('notes', '')
            )

        # Load type mappings
        for type_map in mapping_data.get('types', []):
            key = type_map['c_type']
            self._types[key] = TypeMapping(
                c_type=type_map['c_type'],
                python_type=type_map['python_type'],
                import_required=type_map.get('import_required'),
                transform=type_map.get('transform')
            )

    @property
    def name(self) -> str:
        return self._name

    def get_function_mapping(
        self,
        c_function: str,
        c_header: Optional[str] = None
    ) -> Optional[LibraryMapping]:
        return self._functions.get(c_function)

    def get_type_mapping(self, c_type: str) -> Optional[TypeMapping]:
        return self._types.get(c_type)

    def get_all_function_mappings(self) -> List[LibraryMapping]:
        return list(self._functions.values())

    def get_all_type_mappings(self) -> List[TypeMapping]:
        return list(self._types.values())

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'CustomMappingProvider':
        """
        Load custom mappings from a YAML file.

        Args:
            yaml_path: Path to YAML mapping file

        Returns:
            CustomMappingProvider instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(data)


class LibraryMapperRegistry:
    """Registry for library mapping providers with lookup functionality."""

    # Built-in provider classes
    BUILTIN_PROVIDERS = {
        'stdlib': StandardLibraryMapper,
        'ncurses': NCursesMapper,
        'pthread': PthreadMapper,
    }

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize registry, optionally with config.

        Args:
            config: Optional Config instance to load settings from
        """
        self._providers: Dict[str, LibraryMappingProvider] = {}
        self._config = config

        # Load providers based on config or use defaults
        if config:
            self._load_from_config(config)
        else:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load all built-in providers."""
        for name, provider_class in self.BUILTIN_PROVIDERS.items():
            self._providers[name] = provider_class()

    def _load_from_config(self, config: Any) -> None:
        """Load providers based on config settings."""
        # Check if library mappings are enabled
        if not config.get('library_mappings.enabled', True):
            return

        # Load specified built-in mappers
        builtin_mappers = config.get('library_mappings.builtin_mappers', ['stdlib'])
        for name in builtin_mappers:
            if name in self.BUILTIN_PROVIDERS:
                self._providers[name] = self.BUILTIN_PROVIDERS[name]()

        # Load custom mapping files
        custom_mappings = config.get('library_mappings.custom_mappings', [])
        project_root = config.config_path.parent if hasattr(config, 'config_path') else Path.cwd()

        for mapping_path in custom_mappings:
            full_path = project_root / mapping_path
            if full_path.exists():
                try:
                    provider = CustomMappingProvider.from_yaml(full_path)
                    self._providers[provider.name] = provider
                except Exception as e:
                    print(f"Warning: Failed to load custom mapping {mapping_path}: {e}")

    def register(self, provider: LibraryMappingProvider) -> None:
        """
        Register a mapping provider.

        Args:
            provider: Provider instance to register
        """
        self._providers[provider.name] = provider

    def unregister(self, name: str) -> bool:
        """
        Unregister a mapping provider.

        Args:
            name: Provider name

        Returns:
            True if provider was removed, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            return True
        return False

    def get_provider(self, name: str) -> Optional[LibraryMappingProvider]:
        """Get a specific provider by name."""
        return self._providers.get(name)

    def get_provider_names(self) -> List[str]:
        """Get list of registered provider names."""
        return list(self._providers.keys())

    def get_function_mapping(
        self,
        c_function: str,
        c_header: Optional[str] = None
    ) -> Optional[LibraryMapping]:
        """
        Search all providers for a function mapping.

        Args:
            c_function: C function name
            c_header: Optional header for disambiguation

        Returns:
            First matching LibraryMapping or None
        """
        for provider in self._providers.values():
            mapping = provider.get_function_mapping(c_function, c_header)
            if mapping:
                return mapping
        return None

    def get_type_mapping(self, c_type: str) -> Optional[TypeMapping]:
        """
        Search all providers for a type mapping.

        Args:
            c_type: C type string

        Returns:
            First matching TypeMapping or None
        """
        for provider in self._providers.values():
            mapping = provider.get_type_mapping(c_type)
            if mapping:
                return mapping
        return None

    def get_mappings_for_functions(
        self,
        c_functions: List[str]
    ) -> Dict[str, LibraryMapping]:
        """
        Get mappings for a list of function names.

        Args:
            c_functions: List of C function names

        Returns:
            Dict mapping function names to their LibraryMappings
        """
        result = {}
        for func in c_functions:
            mapping = self.get_function_mapping(func)
            if mapping:
                result[func] = mapping
        return result

    def get_mappings_for_types(
        self,
        c_types: List[str]
    ) -> Dict[str, TypeMapping]:
        """
        Get mappings for a list of type names.

        Args:
            c_types: List of C type strings

        Returns:
            Dict mapping type names to their TypeMappings
        """
        result = {}
        for ctype in c_types:
            mapping = self.get_type_mapping(ctype)
            if mapping:
                result[ctype] = mapping
        return result

    def load_custom_mappings(self, yaml_path: Path) -> None:
        """
        Load and register custom mappings from a YAML file.

        Args:
            yaml_path: Path to YAML mapping file
        """
        provider = CustomMappingProvider.from_yaml(yaml_path)
        self.register(provider)

    def format_hints_for_prompt(
        self,
        c_functions: List[str],
        c_types: Optional[List[str]] = None,
        max_hints: int = 15
    ) -> str:
        """
        Format library mapping hints for inclusion in LLM prompt.

        Args:
            c_functions: List of C functions used
            c_types: Optional list of C types used
            max_hints: Maximum number of hints to include

        Returns:
            Formatted string with mapping hints
        """
        hints = []

        # Get function mappings
        func_mappings = self.get_mappings_for_functions(c_functions)
        for func_name, mapping in list(func_mappings.items())[:max_hints]:
            hint = f"- {mapping.c_function}() -> "
            if mapping.python_module == 'builtins':
                hint += f"{mapping.python_function}"
            else:
                hint += f"{mapping.python_module}.{mapping.python_function}"
            if mapping.notes:
                hint += f" ({mapping.notes})"
            hints.append(hint)

        # Get type mappings if provided
        if c_types:
            type_mappings = self.get_mappings_for_types(c_types)
            remaining = max_hints - len(hints)
            for type_name, mapping in list(type_mappings.items())[:remaining]:
                hint = f"- {mapping.c_type} -> {mapping.python_type}"
                if mapping.import_required:
                    hint += f" (from {mapping.import_required})"
                hints.append(hint)

        if not hints:
            return ""

        return "LIBRARY MAPPINGS:\n" + "\n".join(hints)
