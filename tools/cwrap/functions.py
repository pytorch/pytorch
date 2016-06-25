import math
from copy import deepcopy
from itertools import product

from .utils import argfilter
from .options import make_option
from .config import *

def make_function(lines, stateless):
    if not stateless:
        return Function(lines)
    else:
        return StatelessFunction(lines)


class Function(object):
    DEFINITION_START = Template("""
static PyObject * THPTensor_(${name})(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  Py_ssize_t _argcount = args ? PyTuple_Size(args) : 0;
    """)

    DEFINITION_END = Template("""
  THPUtils_invalidArguments(args, ${expected_args});
  return NULL;
  END_HANDLE_TH_ERRORS
}
    """)

    def __init__(self, lines):
        self._parse_lines(lines)

    def generate(self):
        if not self.options:
            return ''  # Ignore function

        definition = self.DEFINITION_START.substitute({'name': self.name})

        # Declare variables
        variables = set((arg.type, arg.name) for option in self.options
                            for arg in option.arguments)
        is_already_provided = argfilter()
        for variable in variables:
            if not is_already_provided(Argument(*variable)):
                definition += '  {} {};\n'.format(*variable)

        # Generate function body
        definition += self._generate_body()

        # Prepare quick docs and end the declaration
        accepted_args_str = self._describe_options()
        definition += self.DEFINITION_END.substitute(expected_args=accepted_args_str)
        return definition

    def _generate_body(self):
        """Generates code implementing all argument options

        Options are sorted according to their argument count. Ones with equal
        counts are wrapped in the same if block, that checks how many
        arguments have been provided. This allows to ignore checking some
        argument configurations, and save a couple of cycles (PyArg_ParseTuple
        calls add some overhead).
        """
        impl = ''
        prev_arg_count = -1
        for option in sorted(self.options, key=lambda o: o.num_required_args()):
            num_args = option.num_required_args()
            if num_args > prev_arg_count:
                # Nothing to close if it's the first option
                if prev_arg_count != -1 and prev_arg_count < math.inf:
                    impl += '  }\n'
                if num_args < math.inf:
                    impl += Template('  if (_argcount == $numargs) {') \
                                .substitute({'numargs': num_args})
                prev_arg_count = num_args
            else:
                impl += '    PyErr_Clear();'
            impl += '\n    {'
            impl += option.generate()
            impl += '    }\n'
        # Close last argcount block
        if prev_arg_count < math.inf:
            impl += '  }\n'
        return impl

    def _describe_options(self):
        """Generates a string describing accepted argument configurations.
        """
        def describe_arg(arg):
            return TYPE_DESCRIPTIONS.get(arg.type, arg.type) + ' ' + arg.name
        result = '"'
        for option in self.options:
            is_provided = argfilter()
            args = list(filter(lambda arg: not is_provided(arg), option.arguments))
            if args:
                result += '('
                result += ', '.join(map(describe_arg, args))
                result += ')'
            else:
                result += 'no arguments'
            result += ' or '
        return result[:-4] + '"'

    def _resolve_optional_args(self):
        resolved_options = []
        for option, optional_args in zip(self.options, self.optional_args):
            if not optional_args:
                resolved_options.append(option)
            # Generate options with all possible configurations of optional args
            for enabled_bits in product((True, False), repeat=len(optional_args)):
                new_option = option.copy()
                # Replace disabled args with their defaults
                for enabled, default in zip(enabled_bits, optional_args):
                    if enabled:
                        continue
                    new_option.arguments[default[0]] = Argument('CONSTANT', default[1])
                resolved_options.append(new_option)
        self.options = resolved_options

    def _should_ignore(self):
        return 'STATELESS_ONLY' in self.flags

    def _parse_lines(self, lines):
        """Parses cwrap declaration.

        Accepts an iterable of lines and a boolean indicating if the function
        should be stateless.
        Returns a tuple of function name and possible options.
        If option list is empty, the function should be ignored.
        """
        assert len(lines) > 1
        self.options = []
        self.optional_args = []
        self.name, self.flags = FUNCTION_NAME_REGEX.match(lines[0]).group(1, 2)
        if self._should_ignore():
            return

        for line in lines[1:]:
            match = OPTION_REGEX.match(line)
            if match:
                thname, rettype, flags = match.group(1, 2, 3)
                self.options.append(make_option(thname, rettype, flags))
                self.optional_args.append([])
            else:
                assert line.startswith(ARGUMENT_PREFIX)
                arg = line.replace(ARGUMENT_PREFIX, '').strip()
                option = self.options[-1]

                # Check for default values
                default_value = OPTIONAL_ARGUMENT_REGEX.match(arg)
                if default_value:
                    arg_nr = len(option.arguments)
                    self.optional_args[-1].append((arg_nr, default_value.group(1)))
                    arg = arg[:arg.find(' OPTIONAL')]

                # Parse argument
                t, name = arg.split() if arg != 'self' else ('THTensor', 'self')
                t = TYPE_TRANSFORMS.get(t, t)
                option.add_argument(Argument(t, name))
        self._resolve_optional_args()
        self._parse_options()

    def _parse_options(self):
        pass


class StatelessFunction(Function):
    DEFINITION_START = Template("""
static PyObject * THPTensor_stateless_(${name})(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  Py_ssize_t _argcount = args ? PyTuple_Size(args) : 0;
    """)

    def _should_ignore(self):
        return 'STATEFUL_ONLY' in self.flags

    def _parse_options(self):
        self.options = self._make_stateless()
        self._filter_options()

    def _filter_options(self):
        """Filters out options that will never be reached.
        """
        signatures = set()
        def uniq_signatures(option):
            h = option.signature_hash()
            if h in signatures:
                return False
            signatures.add(h)
            return True
        return list(filter(uniq_signatures, self.options))


    def _make_stateless(self):
        """Converts stateful options to stateless options.

        There are two ways of performing this conversion:
        1. If self is only an output argument (it's optional) it can be allocated
        2. The user can also provide self, irrespective of its purpose
        """
        stateless_options = []
        def self_to_(new_name):
            def self_to_new(arg):
                if arg.name != 'self':
                    return arg
                return Argument(arg.type, new_name)
            return self_to_new

        # First pass - try to allocate self, wherever possible
        # This has to go first, because it will be favored during unique
        for option in self.options:
            # If self is optional, it can be allocated
            if option.is_self_optional():
                assert option.return_type == 'self'
                new = option.copy()
                new.map_arguments(self_to_('_res_new'))
                new.return_type = 'STATELESS NEW self'
                stateless_options.append(new)

        # Second pass - if self is actually needed, it can be provided
        for option in self.options:
            provided = option.copy()
            provided.map_arguments(self_to_('_res'))
            if provided.return_type == 'self':
                provided.return_type = 'STATELESS PROV self'
            # This is where it gets tricky. There are two cases:
            # 1. User only provides an output tensor
            # 2. User provides both an output tensor, as well as an index tensor
            if provided.return_type == 'new SelfIndexPair':
                # Case 1.
                provided.insert_argument(0, Argument('THPTensor*', '_res'))
                provided.return_type = 'STATELESS PROV new SelfIndexPair'
                stateless_options.append(provided.copy())
                # Reuse option from case 1. to make 2.
                provided.insert_argument(1, Argument('THPLongTensor*', '_res_ind'))
                provided.return_type = 'STATELESS PROV2 new SelfIndexPair'
            stateless_options.append(provided)

        return stateless_options
