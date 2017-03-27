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
            declaration['options'] = self.filter_unique_options(new_options)
        return declarations

    def filter_unique_options(self, options):
        def signature(option, kwarg_only_count):
            if kwarg_only_count == 0:
                kwarg_only_count = None
            else:
                kwarg_only_count = -kwarg_only_count
            arg_signature = '#'.join(
                arg['type']
                for arg in option['arguments'][:kwarg_only_count]
                if not arg.get('ignore_check'))
            if kwarg_only_count is None:
                return arg_signature
            kwarg_only_signature = '#'.join(
                arg['name'] + '#' + arg['type']
                for arg in option['arguments'][kwarg_only_count:]
                if not arg.get('ignore_check'))
            return arg_signature + "#-#" + kwarg_only_signature
        seen_signatures = set()
        unique = []
        for option in options:
            for num_kwarg_only in range(0, len(option['arguments']) + 1):
                sig = signature(option, num_kwarg_only)
                if sig not in seen_signatures:
                    if num_kwarg_only > 0:
                        for arg in option['arguments'][-num_kwarg_only:]:
                            arg['kwarg_only'] = True
                    unique.append(option)
                    seen_signatures.add(sig)
                    break
        return unique
