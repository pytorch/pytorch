from . import CWrapPlugin
from string import Template


class BeforeAfterCall(CWrapPlugin):

    def initialize(self, cwrap):
        self.cwrap = cwrap

    def insert_snippet(self, template, option, offset, name):
        prepend_str = option.get(name)
        if isinstance(prepend_str, dict):
            backend = option['backends'][0]
            prepend_str = prepend_str.get(backend, None)

        if prepend_str is None:
            return
        if '$' in prepend_str:
            before_call_template = Template(prepend_str)
            args = {'arg' + str(i): self.cwrap.get_arg_accessor(arg, option) for i, arg
                    in enumerate(option['arguments'])}
            prepend_str = before_call_template.substitute(args)
        template.insert(offset, prepend_str)

    def process_pre_arg_assign(self, template, option):
        if option.get('before_arg_assign'):
            self.insert_snippet(template, option, 0, 'before_arg_assign')
        return template

    def process_option_code_template(self, template, option):
        if option.get('before_call') or option.get('after_call'):
            call_idx = template.index('$call')
            self.insert_snippet(template, option, call_idx, 'before_call')
            # call position might have changed
            call_idx = template.index('$call')
            self.insert_snippet(template, option, call_idx + 1, 'after_call')
        return template
