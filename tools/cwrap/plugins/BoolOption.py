from . import CWrapPlugin
from string import Template


class BoolOption(CWrapPlugin):

    UNPACK_TEMPLATE = Template('$arg == Py_True ? $if_true : $if_false')

    def is_bool_option(self, arg):
        return arg['type'] == 'bool' and 'if_true' in arg and 'if_false' in arg

    def process_declarations(self, declarations):
        for declaration in declarations:
            for option in declaration['options']:
                for arg in option['arguments']:
                    if self.is_bool_option(arg):
                        arg['is_bool_option'] = True
                        arg['type'] = 'const char*'
        return declarations

    def get_type_check(self, arg, option):
        if arg.get('is_bool_option', False):
            return Template('PyBool_Check($arg)')

    def get_type_unpack(self, arg, option):
        if arg.get('is_bool_option', False):
            return Template(self.UNPACK_TEMPLATE.safe_substitute(
                if_true=arg['if_true'], if_false=arg['if_false']))
