from . import CWrapPlugin
from string import Template


class ArgumentReferences(CWrapPlugin):

    def initialize(self, cwrap):
        self.cwrap = cwrap

    def process_declarations(self, declarations):
        for declaration in declarations:
            for option in declaration['options']:
                for arg in option['arguments']:
                    if arg['type'] == 'argument':
                        arg['ignore_check'] = True
                        arg['is_reference'] = True
                        # Copy type from referenced argument
                        idx = int(arg['name'])
                        arg['type'] = option['arguments'][idx]['type']
        return declarations

    def _get_true_idx(self, idx, option):
        return sum(not arg.get('ignore_check', False) for arg in option['arguments'][:idx])

    def get_arg_accessor(self, arg, option):
        if arg.get('is_reference', False):
            idx = int(arg['name'])
            referenced = option['arguments'][idx]
            return self.cwrap.get_arg_accessor(referenced, option)
