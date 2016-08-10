import os
import yaml
from string import Template
from copy import deepcopy
from .plugins import ArgcountChecker, OptionalArguments, ArgumentReferences, BeforeCall, ConstantArguments, ReturnArguments


class cwrap(object):
    RETURN_WRAPPERS = {
        'void': Template('$call;\n      Py_RETURN_NONE;'),
        'long': Template('return PyLong_FromLong($call);'),
        'bool': Template('return PyBool_FromLong($call);'),
    }

    TYPE_CHECK = {
        'void*':            Template('PyLong_Check($arg)'),
        'bool':             Template('PyLong_Check($arg)'),
        'float':            Template('PyFloat_Check($arg)'),
        'double':           Template('PyFloat_Check($arg)'),
        # TODO: this will only work for python3
        'int':              Template('PyLong_Check($arg)'),
        'long':             Template('PyLong_Check($arg)'),
    }

    TYPE_UNPACK = {
        'void*':            Template('PyLong_AsVoidPtr($arg)'),
        'bool':             Template('PyLong_AsLong($arg)'),
        'float':            Template('(float)PyFloat_AsDouble($arg)'),
        'double':           Template('PyFloat_AsDouble($arg)'),
        # TODO: this will only work for python3
        'int':              Template('PyLong_AsLong($arg)'),
        'long':             Template('PyLong_AsLong($arg)'),
    }

    OPTION_TEMPLATE = Template("""
    ${els}if ($arg_check) {

      $call
    """)

    CALL_TEMPLATE = Template("$cname($arg_unpack)")

    DEFAULT_PLUGIN_CLASSES = [ArgcountChecker, ConstantArguments, OptionalArguments, ArgumentReferences, BeforeCall, ReturnArguments]

    def __init__(self, source, destination=None, plugins=[], default_plugins=True):
        if destination is None:
            destination = source.replace('.cwrap', '.cpp')

        self.plugins = plugins
        if default_plugins:
            defaults = [cls() for cls in self.DEFAULT_PLUGIN_CLASSES]
            self.plugins = defaults + self.plugins

        for plugin in self.plugins:
            plugin.initialize(self)

        with open(source, 'r') as f:
            declarations = f.read()

        wrapper = self.wrap_declarations(declarations)
        for plugin in self.plugins:
            wrapper = plugin.process_full_file(wrapper)

        with open(destination, 'w') as f:
            f.write(wrapper)

    def wrap_declarations(self, declarations):
        lines = declarations.split('\n')
        declaration_lines = []
        output = []
        in_declaration = False

        for line in lines:
            if line == '[[':
                declaration_lines = []
                in_declaration = True
            elif line == ']]':
                in_declaration = False
                declaration = yaml.load('\n'.join(declaration_lines))
                self.set_declaration_defaults(declaration)

                # Pass declaration in a list - maybe some plugins want to add
                # multiple wrappers
                declarations = [declaration]
                for plugin in self.plugins:
                    declarations = plugin.process_declarations(declarations)
                # Generate wrappers for all declarations and append them to
                # the output
                for declaration in declarations:
                    wrapper = self.generate_wrapper(declaration)
                    for plugin in self.plugins:
                        wrapper = plugin.process_wrapper(wrapper, declaration)
                    output.append(wrapper)
            elif in_declaration:
                declaration_lines.append(line)
            else:
                output.append(line)

        return '\n'.join(output)

    def set_declaration_defaults(self, declaration):
        declaration.setdefault('arguments', [])
        declaration.setdefault('return', 'void')
        if not 'cname' in declaration:
            declaration['cname'] = declaration['name']
        # Simulate multiple dispatch, even if it's not necessary
        if not 'options' in declaration:
            declaration['options'] = [{'arguments': declaration['arguments']}]
            del declaration['arguments']
        # Parse arguments (some of them can be strings)
        for option in declaration['options']:
            option['arguments'] = self.parse_arguments(option['arguments'])
        # Propagate defaults from declaration to options
        for option in declaration['options']:
            for k, v in declaration.items():
                if k != 'name' and k != 'options':
                    option.setdefault(k, v)

    def parse_arguments(self, args):
        new_args = []
        for arg in args:
            # Simple arg declaration of form "<type> <name>"
            if isinstance(arg, str):
                t, _, name = arg.partition(' ')
                new_args.append({'type': t, 'name': name})
            elif isinstance(arg, dict):
                if 'arg' in arg:
                    arg['type'], _, arg['name'] = arg['arg'].partition(' ')
                    del arg['arg']
                new_args.append(arg)
            else:
                assert False
        return new_args

    def search_plugins(self, fnname, args, fallback):
        for plugin in self.plugins:
            wrapper = getattr(plugin, fnname)(*args)
            if wrapper is not None:
                return wrapper
        return fallback(*args)

    def get_type_check(self, arg, option):
        return self.search_plugins('get_type_check', (arg, option), lambda arg,_: self.TYPE_CHECK[arg['type']])

    def get_type_unpack(self, arg, option):
        return self.search_plugins('get_type_unpack', (arg, option), lambda arg,_: self.TYPE_UNPACK[arg['type']])

    def get_return_wrapper(self, option):
        return self.search_plugins('get_return_wrapper', (option,), lambda t: self.RETURN_WRAPPERS[option['return']])

    def get_wrapper_template(self, declaration):
        return self.search_plugins('get_wrapper_template', (declaration,), lambda _: None)

    def get_arg_accessor(self, arg, option):
        return self.search_plugins('get_arg_accessor', (arg, option), lambda arg,_: 'PyTuple_GET_ITEM(args, {})'.format(arg['idx']))

    def generate_wrapper(self, declaration):
        wrapper = ''
        for i, option in enumerate(declaration['options']):
            option_wrapper = self.generate_option(option, is_first=(i == 0))
            for plugin in self.plugins:
                option_wrapper = plugin.process_option_code(option_wrapper, option)
            wrapper += option_wrapper
        return self.get_wrapper_template(declaration).substitute(name=declaration['name'], options=wrapper)

    def map_selected_arguments(self, base_fn_name, plugin_fn_name, option, arguments):
        result = []
        for arg in arguments:
            accessor = self.get_arg_accessor(arg, option)
            res = getattr(self, base_fn_name)(arg, option).substitute(arg=accessor)
            for plugin in self.plugins:
                res = getattr(plugin, plugin_fn_name)(res, arg, accessor)
            result.append(res)
        return result

    def generate_option(self, option, is_first):
        checked_args = list(filter(
            lambda arg: not 'ignore_check' in arg or not arg['ignore_check'],
            option['arguments']))
        option['num_checked_args'] = len(checked_args)
        for i, arg in enumerate(checked_args):
            arg['idx'] = i

        # Generate checks
        arg_checks = self.map_selected_arguments('get_type_check',
                'process_single_check', option, checked_args)
        arg_checks = ' &&\n          '.join(arg_checks)
        for plugin in self.plugins:
            arg_checks = plugin.process_all_checks(arg_checks, option)

        # Generate unpacks
        arg_unpack = self.map_selected_arguments('get_type_unpack',
                'process_single_unpack', option, option['arguments'])
        arg_unpack = ', '.join(arg_unpack)
        for plugin in self.plugins:
            arg_unpack = plugin.process_all_unpacks(arg_unpack, option)

        # Generate call
        raw_call = self.CALL_TEMPLATE.substitute(cname=option['cname'], arg_unpack=arg_unpack)
        call = self.get_return_wrapper(option).substitute(call=raw_call)
        for plugin in self.plugins:
            call = plugin.process_call(call, option)
        call = '\n      '.join(map(lambda s: s.strip(), call.split('\n')))

        # Put everything together
        return self.OPTION_TEMPLATE.substitute(
            els=('} else ' if not is_first else ''),
            arg_check=arg_checks,
            call=call
        )

