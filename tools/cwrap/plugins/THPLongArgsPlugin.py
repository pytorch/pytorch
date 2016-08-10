from string import Template
from . import CWrapPlugin

class THPLongArgsPlugin(CWrapPlugin):
    PARSE_LONG_ARGS = Template("""\
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, $num_checked);
      THLongStorage* __long_args = __long_args_guard.get();
""")

    def get_arg_accessor(self, arg, option):
        if 'long_args' in option and option['long_args'] and arg['name'] == 'long_args':
            return '__long_args'

    def get_type_unpack(self, arg, option):
        if option.get('long_args', False) and arg['name'] == 'long_args':
            return Template('$arg')

    def process_declarations(self, declarations):
        for declaration in declarations:
            for option in declaration['options']:
                if not 'long_args' in option or not option['long_args']:
                    continue
                for arg in option['arguments']:
                    if arg['name'] == 'long_args':
                        arg['ignore_check'] = True
        return declarations

    def process_all_checks(self, code, option):
        if 'long_args' in option and option['long_args']:
            code = code.replace('__argcount ==', '__argcount >')
        return code

    def process_option_code(self, code, option):
        if 'long_args' in option and option['long_args']:
            lines = code.split('\n')
            end_checks = 0
            for i, line in enumerate(lines):
                if ') {' in line:
                    end_checks = i
                    break
            lines = lines[:end_checks+1] + [self.PARSE_LONG_ARGS.substitute(num_checked=option['num_checked_args'])] + lines[end_checks+1:]
            code = '\n'.join(lines)
        return code

