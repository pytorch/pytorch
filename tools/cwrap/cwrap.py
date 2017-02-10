import os
import yaml
from string import Template
from copy import deepcopy
from .plugins import ArgcountChecker, OptionalArguments, ArgumentReferences, \
    BeforeAfterCall, ConstantArguments, ReturnArguments, GILRelease


class cwrap(object):
    RETURN_WRAPPERS = {
        'void': Template('Py_RETURN_NONE;'),
        'long': Template('return PyLong_FromLong($result);'),
        'bool': Template('return PyBool_FromLong($result);'),
        'void*': Template('return PyLong_FromVoidPtr($result);'),
    }

    OPTION_TEMPLATE = Template("""
    ${els}if ($arg_check) {
      $code
    """)

    OPTION_CODE_TEMPLATE = [
        '$call',
        '$return_result',
    ]

    FUNCTION_CALL_TEMPLATE = Template("$capture_result$cname($arg_unpack);")

    DEFAULT_PLUGIN_CLASSES = [ArgcountChecker, ConstantArguments, OptionalArguments,
                              ArgumentReferences, BeforeAfterCall, ReturnArguments, GILRelease]

    def __init__(self, source, destination=None, plugins=[], default_plugins=True):
        if destination is None:
            destination = source.replace('.cwrap', '.cpp')

        self.plugins = plugins
        if default_plugins:
            defaults = [cls() for cls in self.DEFAULT_PLUGIN_CLASSES]
            self.plugins = defaults + self.plugins

        for plugin in self.plugins:
            plugin.initialize(self)

        self.base_path = os.path.dirname(os.path.abspath(source))
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
        i = 0

        while i < len(lines):
            line = lines[i]
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
            elif '!!inc ' == line[:6]:
                fname = os.path.join(self.base_path, line[6:].strip())
                with open(fname, 'r') as f:
                    included = f.read().split('\n')
                # insert it into lines at position i+1
                lines[i + 1:i + 1] = included
            else:
                output.append(line)
            i += 1

        return '\n'.join(output)

    def set_declaration_defaults(self, declaration):
        declaration.setdefault('arguments', [])
        declaration.setdefault('return', 'void')
        if 'cname' not in declaration:
            declaration['cname'] = declaration['name']
        # Simulate multiple dispatch, even if it's not necessary
        if 'options' not in declaration:
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
        return self.search_plugins('get_type_check', (arg, option), lambda arg, _: None)

    def get_type_unpack(self, arg, option):
        return self.search_plugins('get_type_unpack', (arg, option), lambda arg, _: None)

    def get_return_wrapper(self, option):
        return self.search_plugins('get_return_wrapper', (option,), lambda _: self.RETURN_WRAPPERS[option['return']])

    def get_wrapper_template(self, declaration):
        return self.search_plugins('get_wrapper_template', (declaration,), lambda _: None)

    def get_arg_accessor(self, arg, option):
        def wrap_accessor(arg, _):
            if arg.get('idx') is None:
                raise RuntimeError("Missing accessor for '{} {}'".format(
                                   arg['type'], arg['name']))
            return 'PyTuple_GET_ITEM(args, {})'.format(arg['idx'])

        return self.search_plugins('get_arg_accessor', (arg, option), wrap_accessor)

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
            tmpl = getattr(self, base_fn_name)(arg, option)
            if tmpl is None:
                fn = 'check' if base_fn_name == 'get_type_check' else 'unpack'
                raise RuntimeError("Missing type {} for '{} {}'".format(
                                   fn, arg['type'], arg['name']))
            res = tmpl.substitute(arg=accessor, idx=arg.get('idx'))
            for plugin in self.plugins:
                res = getattr(plugin, plugin_fn_name)(res, arg, accessor)
            result.append(res)
        return result

    def generate_option(self, option, is_first):
        checked_args = list(filter(
            lambda arg: 'ignore_check' not in arg or not arg['ignore_check'],
            option['arguments']))
        option['num_checked_args'] = len(checked_args)
        idx_args = list(filter(
            lambda arg: not arg.get('ignore_check') and not arg.get('no_idx'),
            option['arguments']))
        for i, arg in enumerate(idx_args):
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
        try:
            return_result = self.get_return_wrapper(option).substitute()
            call = self.FUNCTION_CALL_TEMPLATE.substitute(capture_result='',
                                                          cname=option['cname'], arg_unpack=arg_unpack)
        except KeyError:
            return_result = self.get_return_wrapper(option).substitute(result='__result')
            call = self.FUNCTION_CALL_TEMPLATE.substitute(capture_result=(option['return'] + ' __result = '),
                                                          cname=option['cname'], arg_unpack=arg_unpack)

        code_template = deepcopy(self.OPTION_CODE_TEMPLATE)
        for plugin in self.plugins:
            code_template = plugin.process_option_code_template(code_template,
                                                                option)
        code_template = Template('\n'.join(code_template))
        code = code_template.substitute(call=call, return_result=return_result)
        code_lines = map(lambda s: s.strip(), code.split('\n'))
        code = '\n'
        depth = 6
        for line in code_lines:
            depth -= line.count('}') * 2
            code += ' ' * depth + line + '\n'
            depth += line.count('{') * 2
            depth += line.count('(') * 4
            depth -= line.count(')') * 4

        # Put everything together
        return self.OPTION_TEMPLATE.substitute(
            els=('} else ' if not is_first else ''),
            arg_check=arg_checks,
            code=code,
        )
