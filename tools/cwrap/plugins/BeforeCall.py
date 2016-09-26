from . import CWrapPlugin
from string import Template

class BeforeCall(CWrapPlugin):

    def initialize(self, cwrap):
        self.cwrap = cwrap

    def process_option_code_template(self, template, option):
        if option.get('before_call', False):
            call_idx = template.index('$call')
            prepend_str = option['before_call']
            if '$' in prepend_str:
                before_call_template = Template(option['before_call'])
                args = {'arg' + str(i): self.cwrap.get_arg_accessor(arg, option) for i, arg
                            in enumerate(option['arguments'])}
                prepend_str = before_call_template.substitute(args)
            template.insert(call_idx, prepend_str)
        return template
