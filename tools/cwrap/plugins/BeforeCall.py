from . import CWrapPlugin
from string import Template

class BeforeCall(CWrapPlugin):

    def initialize(self, cwrap):
        self.cwrap = cwrap

    def process_call(self, code, option):
        if option.get('before_call', False):
            if '$' in option['before_call']:
                template = Template(option['before_call'])
                args = {'arg' + str(i): self.cwrap.get_arg_accessor(arg, option) for i, arg
                            in enumerate(option['arguments'])}
                return template.substitute(args) + code
            else:
                return option['before_call'] + code
        return code
