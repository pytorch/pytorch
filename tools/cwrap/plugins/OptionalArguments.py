from copy import deepcopy
from . import CWrapPlugin
from itertools import product

class OptionalArguments(CWrapPlugin):

    def process_declarations(self, declarations):
        new_options = []
        for declaration in declarations:
            for option in declaration['options']:
                optional_args = []
                for i, arg in enumerate(option['arguments']):
                    if 'default' in arg:
                        optional_args.append(i)
                for permutation in product((True, False), repeat=len(optional_args)):
                    option_copy = deepcopy(option)
                    for i, bit in zip(optional_args, permutation):
                        arg = option_copy['arguments'][i]
                        if not bit:
                            arg['type'] = 'CONSTANT'
                            arg['ignore_check'] = True
                            # PyYAML interprets NULL as None...
                            arg['name'] = 'NULL' if arg['default'] is None else arg['default']
                    new_options.append(option_copy)
            declaration['options'] = self.filter_unique_options(declaration['options'] + new_options)
        return declarations

    def filter_unique_options(self, options):
        def signature(option):
            return '#'.join(arg['type'] for arg in option['arguments'] if not 'ignore_check' in arg or not arg['ignore_check'])
        seen_signatures = set()
        unique = []
        for option in options:
            sig = signature(option)
            if sig not in seen_signatures:
                unique.append(option)
                seen_signatures.add(sig)
        return unique

