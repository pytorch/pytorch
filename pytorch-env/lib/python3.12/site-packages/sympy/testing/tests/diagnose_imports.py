#!/usr/bin/env python

"""
Import diagnostics. Run bin/diagnose_imports.py --help for details.
"""

from __future__ import annotations

if __name__ == "__main__":

    import sys
    import inspect
    import builtins

    import optparse

    from os.path import abspath, dirname, join, normpath
    this_file = abspath(__file__)
    sympy_dir = join(dirname(this_file), '..', '..', '..')
    sympy_dir = normpath(sympy_dir)
    sys.path.insert(0, sympy_dir)

    option_parser = optparse.OptionParser(
        usage=
            "Usage: %prog option [options]\n"
            "\n"
            "Import analysis for imports between SymPy modules.")
    option_group = optparse.OptionGroup(
        option_parser,
        'Analysis options',
        'Options that define what to do. Exactly one of these must be given.')
    option_group.add_option(
        '--problems',
        help=
            'Print all import problems, that is: '
            'If an import pulls in a package instead of a module '
            '(e.g. sympy.core instead of sympy.core.add); ' # see ##PACKAGE##
            'if it imports a symbol that is already present; ' # see ##DUPLICATE##
            'if it imports a symbol '
            'from somewhere other than the defining module.', # see ##ORIGIN##
        action='count')
    option_group.add_option(
        '--origins',
        help=
            'For each imported symbol in each module, '
            'print the module that defined it. '
            '(This is useful for import refactoring.)',
        action='count')
    option_parser.add_option_group(option_group)
    option_group = optparse.OptionGroup(
        option_parser,
        'Sort options',
        'These options define the sort order for output lines. '
        'At most one of these options is allowed. '
        'Unsorted output will reflect the order in which imports happened.')
    option_group.add_option(
        '--by-importer',
        help='Sort output lines by name of importing module.',
        action='count')
    option_group.add_option(
        '--by-origin',
        help='Sort output lines by name of imported module.',
        action='count')
    option_parser.add_option_group(option_group)
    (options, args) = option_parser.parse_args()
    if args:
        option_parser.error(
            'Unexpected arguments %s (try %s --help)' % (args, sys.argv[0]))
    if options.problems > 1:
        option_parser.error('--problems must not be given more than once.')
    if options.origins > 1:
        option_parser.error('--origins must not be given more than once.')
    if options.by_importer > 1:
        option_parser.error('--by-importer must not be given more than once.')
    if options.by_origin > 1:
        option_parser.error('--by-origin must not be given more than once.')
    options.problems = options.problems == 1
    options.origins = options.origins == 1
    options.by_importer = options.by_importer == 1
    options.by_origin = options.by_origin == 1
    if not options.problems and not options.origins:
        option_parser.error(
            'At least one of --problems and --origins is required')
    if options.problems and options.origins:
        option_parser.error(
            'At most one of --problems and --origins is allowed')
    if options.by_importer and options.by_origin:
        option_parser.error(
            'At most one of --by-importer and --by-origin is allowed')
    options.by_process = not options.by_importer and not options.by_origin

    builtin_import = builtins.__import__

    class Definition:
        """Information about a symbol's definition."""
        def __init__(self, name, value, definer):
            self.name = name
            self.value = value
            self.definer = definer
        def __hash__(self):
            return hash(self.name)
        def __eq__(self, other):
            return self.name == other.name and self.value == other.value
        def __ne__(self, other):
            return not (self == other)
        def __repr__(self):
            return 'Definition(%s, ..., %s)' % (
                repr(self.name), repr(self.definer))

    # Maps each function/variable to name of module to define it
    symbol_definers: dict[Definition, str] = {}

    def in_module(a, b):
        """Is a the same module as or a submodule of b?"""
        return a == b or a != None and b != None and a.startswith(b + '.')

    def relevant(module):
        """Is module relevant for import checking?

        Only imports between relevant modules will be checked."""
        return in_module(module, 'sympy')

    sorted_messages = []

    def msg(msg, *args):
        global options, sorted_messages
        if options.by_process:
            print(msg % args)
        else:
            sorted_messages.append(msg % args)

    def tracking_import(module, globals=globals(), locals=[], fromlist=None, level=-1):
        """__import__ wrapper - does not change imports at all, but tracks them.

        Default order is implemented by doing output directly.
        All other orders are implemented by collecting output information into
        a sorted list that will be emitted after all imports are processed.

        Indirect imports can only occur after the requested symbol has been
        imported directly (because the indirect import would not have a module
        to pick the symbol up from).
        So this code detects indirect imports by checking whether the symbol in
        question was already imported.

        Keeps the semantics of __import__ unchanged."""
        global options, symbol_definers
        caller_frame = inspect.getframeinfo(sys._getframe(1))
        importer_filename = caller_frame.filename
        importer_module = globals['__name__']
        if importer_filename == caller_frame.filename:
            importer_reference = '%s line %s' % (
                importer_filename, str(caller_frame.lineno))
        else:
            importer_reference = importer_filename
        result = builtin_import(module, globals, locals, fromlist, level)
        importee_module = result.__name__
        # We're only interested if importer and importee are in SymPy
        if relevant(importer_module) and relevant(importee_module):
            for symbol in result.__dict__.iterkeys():
                definition = Definition(
                    symbol, result.__dict__[symbol], importer_module)
                if definition not in symbol_definers:
                    symbol_definers[definition] = importee_module
            if hasattr(result, '__path__'):
                ##PACKAGE##
                # The existence of __path__ is documented in the tutorial on modules.
                # Python 3.3 documents this in http://docs.python.org/3.3/reference/import.html
                if options.by_origin:
                    msg('Error: %s (a package) is imported by %s',
                        module, importer_reference)
                else:
                    msg('Error: %s contains package import %s',
                        importer_reference, module)
            if fromlist != None:
                symbol_list = fromlist
                if '*' in symbol_list:
                    if (importer_filename.endswith(("__init__.py", "__init__.pyc", "__init__.pyo"))):
                        # We do not check starred imports inside __init__
                        # That's the normal "please copy over its imports to my namespace"
                        symbol_list = []
                    else:
                        symbol_list = result.__dict__.iterkeys()
                for symbol in symbol_list:
                    if symbol not in result.__dict__:
                        if options.by_origin:
                            msg('Error: %s.%s is not defined (yet), but %s tries to import it',
                                importee_module, symbol, importer_reference)
                        else:
                            msg('Error: %s tries to import %s.%s, which did not define it (yet)',
                                importer_reference, importee_module, symbol)
                    else:
                        definition = Definition(
                            symbol, result.__dict__[symbol], importer_module)
                        symbol_definer = symbol_definers[definition]
                        if symbol_definer == importee_module:
                            ##DUPLICATE##
                            if options.by_origin:
                                msg('Error: %s.%s is imported again into %s',
                                    importee_module, symbol, importer_reference)
                            else:
                                msg('Error: %s imports %s.%s again',
                                      importer_reference, importee_module, symbol)
                        else:
                            ##ORIGIN##
                            if options.by_origin:
                                msg('Error: %s.%s is imported by %s, which should import %s.%s instead',
                                      importee_module, symbol, importer_reference, symbol_definer, symbol)
                            else:
                                msg('Error: %s imports %s.%s but should import %s.%s instead',
                                      importer_reference, importee_module, symbol, symbol_definer, symbol)
        return result

    builtins.__import__ = tracking_import
    __import__('sympy')

    sorted_messages.sort()
    for message in sorted_messages:
        print(message)
