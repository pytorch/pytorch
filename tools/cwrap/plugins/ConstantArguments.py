from . import CWrapPlugin
from string import Template


class ConstantArguments(CWrapPlugin):

    def process_declarations(self, declarations):
        for declaration in declarations:
            for option in declaration['options']:
                for arg in option['arguments']:
                    if arg['type'] == 'CONSTANT':
                        arg['ignore_check'] = True
        return declarations

    def get_type_unpack(self, arg, option):
        if arg['type'] == 'CONSTANT':
            return Template('$arg')

    def get_arg_accessor(self, arg, option):
        if arg['type'] == 'CONSTANT':
            return arg['name']
