from . import CWrapPlugin
from string import Template


class BeforeAfterCall(CWrapPlugin):

    def initialize(self, cwrap):
        self.cwrap = cwrap

    def insert_snippet(self, template, option, offset, name):
        prepend_str = option.get(name)
        if prepend_str is None:
            return
        if '$' in prepend_str:
            before_call_template = Template(option[name])
            args = {'arg' + str(i): self.cwrap.get_arg_accessor(arg, option) for i, arg
                    in enumerate(option['arguments'])}
            prepend_str = before_call_template.substitute(args)
        template.insert(offset, prepend_str)

    def process_option_code_template(self, template, option):
        if option.get('before_call') or option.get('after_call'):
            call_idx = template.index('$call')
            self.insert_snippet(template, option, call_idx, 'before_call')
            # call position might have changed
            call_idx = template.index('$call')
            self.insert_snippet(template, option, call_idx + 1, 'after_call')
        return template
