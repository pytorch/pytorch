from . import CWrapPlugin
from string import Template


class BoolOption(CWrapPlugin):

    UNPACK_TEMPLATE = Template('$arg == Py_True ? $if_true : $if_false')

    def is_bool_option(self, arg):
        return arg['type'] == 'bool' and 'if_true' in arg and 'if_false' in arg

    def get_type_check(self, arg, option):
        if self.is_bool_option(arg):
            return Template('PyBool_Check($arg)')

    def get_type_unpack(self, arg, option):
        if self.is_bool_option(arg):
            return Template(self.UNPACK_TEMPLATE.safe_substitute(
                if_true=arg['if_true'], if_false=arg['if_false']))
