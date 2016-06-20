from string import Template
from pprint import pprint
from itertools import chain, product
from copy import deepcopy
import re

ARGUMENT_PREFIX = '  -'
OPTION_REGEX = re.compile('^\s*([a-zA-z0-9]+) -> (new [a-zA-Z]+|[a-zA-Z]+)(.*)')
FUNCTION_NAME_REGEX = re.compile('^\s*([a-zA-Z0-9]+)(.*)')
OPTIONAL_ARGUMENT_REGEX = re.compile('.* OPTIONAL (.*)$')

def cwrap(filename):
    """Parses and generates code for a .cwrap file

       Assumes that filename ends with .cwrap.cpp and saves the result to
       .cpp file with the same prefix.
    """
    assert filename.endswith('.cwrap.cpp')
    with open(filename, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    new_content = ''
    in_declaration = False
    for line in lines:
        if line == '[[':
            in_declaration = True
            func_lines = []
        elif line == ']]':
            in_declaration = False
            new_content += generate_function(func_lines, True)
            new_content += generate_function(func_lines, False)
        elif in_declaration:
            func_lines.append(line)
        else:
            new_content += line + '\n'
    with open(filename.replace('.cwrap', ''), 'w') as f:
        f.write(new_content)

class Argument(object):
    def __init__(self, type, name):
        self.type = type
        self.name = name

    def __hash__(self):
        return (self.type + '#' + self.name).__hash__()

    def copy(self):
        return deepcopy(self)

class Option(object):
    def __init__(self, funcname, return_type, flags):
        self.funcname = funcname
        self.flags = flags
        self.return_type = return_type
        self.arguments = []

    def add_argument(self, arg):
        self.arguments.append(arg)

    def insert_argument(self, idx, arg):
        self.arguments.insert(idx, arg)

    def map_arguments(self, fn):
        self.arguments = list(map(fn, self.arguments))

    def is_self_optional(self):
        return 'OPTIONAL_SELF' in self.flags

    def wrap(self, args):
        if 'PLAIN_CALL' in self.flags:
            return self._wrap_plain(args)
        else:
            return self._wrap_th(args)

    def _wrap_th(self, args):
        th_call = 'THTensor_({})({})'.format(self.funcname, args)
        if self.return_type == 'CUSTOM':
            return self.flags.format(expr=th_call)
        else:
            return RETURN_WRAPPER[self.return_type].substitute({'expr': th_call})

    def _wrap_plain(self, args):
        call = '{}({})'.format(self.funcname, args)
        return RETURN_WRAPPER[self.return_type].substitute({'expr': call})

    def copy(self):
        return deepcopy(self)

    def signature_hash(self):
        is_already_provided = argfilter()
        s = '#'.join(arg.type for arg in self.arguments if not is_already_provided(arg))
        return s.__hash__()

# Basic templates for declarations
DEFINITION_START = Template("""
static PyObject * THPTensor_(${name})(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  Py_ssize_t _argcount = args ? PyTuple_Size(args) : 0;
""")
STATELESS_DEFINITION_START = Template("""
static PyObject * THPTensor_stateless_(${name})(PyObject *_unused, PyObject *args)
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

# TODO: accreal
# Transforms applied to argument types declared in the definition
# these are mostly, so that the * can be omitted for convenience and clarity
TYPE_TRANSFORMS = {
    'THTensor': 'THPTensor*',
    'THByteTensor': 'THPByteTensor*',
    'THLongTensor': 'THPLongTensor*',
    'THFloatTensor': 'THPFloatTensor*',
    'THDoubleTensor': 'THPDoubleTensor*',
    'THLongStorage': 'THPLongStorage*',
    'THGenerator': 'THPGenerator*',
    # TODO
    'accreal': 'double',
}

# Code that will be used to generate each of argument options
OPTION_CODE = Template("""
    if (PyArg_ParseTuple(args, $format$parse_args)) {
      $expr;
    }
""")

# Used to build format string for PyArg_ParseTuple
FORMAT_STR_MAP = {
    'THPTensor*': 'O!',
    'THPLongTensor*': 'O!',
    'THPByteTensor*': 'O!',
    'THPFloatTensor*': 'O!',
    'THPDoubleTensor*': 'O!',
    'THPLongStorage*': 'O!',
    'THPGenerator*': 'O!',
    'real': 'O&',
    'long': 'l',
    'double': 'd',
    'bool': 'p',
}

# If O! is specified for any type in FORMAT_STR_MAP you should specify it's
# type here
# TODO: change to THP*Class or use a parser function
ARGPARSE_TYPE_CHECK = {
    'THPTensor*': 'THPTensorType',
    'THPLongTensor*': 'THPLongTensorType',
    'THPByteTensor*': 'THPByteTensorType',
    'THPFloatTensor*': 'THPFloatTensorType',
    'THPDoubleTensor*': 'THPDoubleTensorType',
    'THPLongStorage*': 'THPLongStorageType',
    'THPGenerator*': 'THPGeneratorType',
    'real': 'THPUtils_(parseReal)',
}

# Code used to convert return values to Python objects
RETURN_WRAPPER = {
    'THTensor':             Template('return THPTensor_(newObject)($expr)'),
    'THStorage':            Template('return THPStorage_(newObject)($expr)'),
    'THLongStorage':        Template('return THPLongStorage_newObject($expr)'),
    'bool':                 Template('return PyBool_FromLong($expr)'),
    'long':                 Template('return PyInt_FromLong($expr)'),
    'double':               Template('return PyFloat_FromDouble($expr)'),
    'self':                 Template('$expr; Py_INCREF(self); return (PyObject*)self'),
    # TODO
    'accreal':              Template('return PyFloat_FromDouble($expr)'),
    'real':                 Template('return THPUtils_(newReal)($expr)'),
    'new THByteTensor':     Template("""
        THByteTensorPtr _t = THByteTensor_new();
        THPByteTensorPtr _ret = (THPByteTensor*)THPByteTensor_newObject(_t);
        _t.release();
        $expr;
        return (PyObject*)_ret.release()"""),
    'new ValueIndexPair':   Template("""
        THTensorPtr _value = THTensor_(new)();
        THLongTensorPtr _indices = THLongTensor_new();
        THPTensorPtr _v = (THPTensor*)THPTensor_(newObject)(_value);
        THPLongTensorPtr _i = (THPLongTensor*)THPLongTensor_newObject(_indices);
        _value.release();
        _indices.release();
        $expr;
        PyObject *ret = Py_BuildValue("NN", (PyObject*)_v.get(), (PyObject*)_i.get());
        _v.release(); _i.release();
        return ret;"""),
    'new SelfIndexPair':    Template("""
        THLongTensorPtr _indices = THLongTensor_new();
        THPLongTensorPtr _i = (THPLongTensor*)THPLongTensor_newObject(_indices);
        _indices.release();
        $expr;
        PyObject *ret = Py_BuildValue("ON", (PyObject*)self, (PyObject*)_i.get());
        _i.release();
        return ret"""),
    'new THTensor':         Template("""
        THTensorPtr _value = THTensor_(new)();
        THPTensorPtr _ret = (THPTensor*)THPTensor_(newObject)(_value);
        _value.release();
        $expr;
        return (PyObject*)_ret.release()"""),
    'new THLongTensor':     Template("""
        THLongTensorPtr _i = THLongTensor_new();
        THPLongTensorPtr _ret = (THPLongTensor*)THPLongTensor_newObject(_i);
        _i.release();
        $expr;
        return (PyObject*)_ret.release()"""),

    # Stateless mode
    'STATELESS PROV new SelfIndexPair': Template("""
        THLongTensorPtr _indices = THLongTensor_new();
        THPLongTensorPtr _i = (THPLongTensor*)THPLongTensor_newObject(_indices);
        _indices.release();
        $expr;
        PyObject *ret = Py_BuildValue("ON", (PyObject*)_res, (PyObject*)_i.get());
        _i.release();
        return ret;"""),
    'STATELESS PROV2 new SelfIndexPair': Template("""
        $expr;
        return Py_BuildValue("OO", (PyObject*)_res, (PyObject*)_res_ind)"""),

    'STATELESS PROV self':   Template('$expr; Py_INCREF(_res); return (PyObject*)_res'),
    'STATELESS NEW self':        Template("""
        THTensorPtr _t = THTensor_(new)();
        THPTensorPtr _res_new = (THPTensor*)THPTensor_(newObject)(_t);
        _t.release();
        $expr;
        return (PyObject*)_res_new.release()"""),
}

# Additional args that are added to TH call
# tuples  are prepended
# dicts use integer keys to specify where to insert arguments
ADDITIONAL_ARGS = {
    'new THByteTensor': (Argument('THPByteTensor*', '_ret'),),
    'new THLongTensor': (Argument('THPLongTensor*', '_ret'),),
    'new THTensor':     (Argument('THPTensor*', '_ret'),),
    'new ValueIndexPair': (Argument('THPTensor*', '_v'), Argument('THPLongTensor*', '_i')),
    'new SelfIndexPair': (Argument('THPTensor*', 'self'), Argument('THPLongTensor*', '_i')),
    'STATELESS PROV new SelfIndexPair': {1: Argument('THPTensor*', '_i')},
}

# Types for which it's necessary to extract cdata
CDATA_TYPES = set((
    'THPTensor*',
    'THPByteTensor*',
    'THPLongTensor*',
    'THPFloatTensor*',
    'THPDoubleTensor*',
    'THPStorage*',
    'THPLongStorage*',
    'THPGenerator*',
))

TYPE_DESCRIPTIONS = {
    'THPTensor*': '" THPTensorStr "',
    'THPByteTensor*': 'ByteTensor',
    'THPLongTensor*': 'LongTensor',
    'THPFloatTensor*': 'FloatTensor',
    'THPDoubleTensor*': 'DoubleTensor',
    'THPStorage*': '" THPStorageStr "',
    'THPLongStorage*': 'LongStorage',
    'THPGenerator*': 'Generator',
    'real': '" RealStr "',
    'accreal': '" RealStr "',
}

def generate_function(lines, stateless):
    assert len(lines) > 1
    lines = remove_indentation(lines)
    function_name, arg_options = parse_lines(lines, stateless)
    if not arg_options:
        return ''  # Ignore function

    variables = set((arg.type, arg.name) for option in arg_options
                        for arg in option.arguments)
    start = DEFINITION_START if not stateless else STATELESS_DEFINITION_START
    definition = start.substitute({'name': function_name})

    # Declare variables
    is_already_provided = argfilter()
    for variable in variables:
        if not is_already_provided(Argument(*variable)):
            definition += '  {} {};\n'.format(*variable)

    # Generate function body
    definition += generate_all_options(arg_options)

    accepted_args_str = describe_options(arg_options, stateless)
    definition += DEFINITION_END.substitute(expected_args=accepted_args_str)
    return definition

def describe_options(options, is_stateless):
    def describe_arg(arg):
        return TYPE_DESCRIPTIONS.get(arg.type, arg.type) + ' ' + arg.name
    result = '"'
    for option in options:
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

def remove_indentation(lines):
    """Removes 2 spaces from the left from each line.

       If anyone wants to use another indentation depth, please update
       this function first.
    """
    return [line[2:] for line in lines]

def parse_lines(lines, stateless):
    """Parses cwrap declaration.

       Accepts an iterable of lines.
       Returns a tuple
    """
    arg_options = []
    function_name, options = FUNCTION_NAME_REGEX.match(lines[0]).group(1, 2)
    if stateless and 'STATEFUL_ONLY' in options:
        return function_name, []  # Ignore function
    if not stateless and 'STATELESS_ONLY' in options:
        return function_name, []  # Ignore function

    def resolve_optional_args():
        if not arg_options or not optional_args:
            return
        option = arg_options.pop()
        for enabled_bits in product('01', repeat=len(optional_args)):
            new_option = option.copy()
            for enabled, default in zip(enabled_bits, optional_args):
                if enabled == '1':
                    continue
                new_option.arguments[default[0]] = Argument('CONSTANT', default[1])
            arg_options.append(new_option)

    for line in lines[1:]:
        match = OPTION_REGEX.match(line)
        if match:
            thname, rettype, flags = match.group(1, 2, 3)
            resolve_optional_args()
            arg_options.append(Option(thname, rettype, flags.strip()))
            optional_args = []
        else:
            assert line.startswith(ARGUMENT_PREFIX)
            arg = line.replace(ARGUMENT_PREFIX, '').strip()
            option = arg_options[-1]

            # Check for default values
            default_value = OPTIONAL_ARGUMENT_REGEX.match(arg)
            if default_value:
                arg_nr = len(option.arguments)
                optional_args.append((arg_nr, default_value.group(1)))
                arg = arg[:arg.find(' OPTIONAL')]

            # Save argument
            if arg == 'self':
                t, name = 'THTensor', 'self'
            else:
                t, name = arg.split(' ')
            t = TYPE_TRANSFORMS.get(t, t)
            option.add_argument(Argument(t, name))
    resolve_optional_args()

    if stateless:
        arg_options = make_stateless(arg_options)

    # Function should be ignored if there are no options left
    return function_name, unique_options(arg_options)

def unique_options(arg_options):
    """Filters out options that will never be reached.
    """
    signatures = set()
    def uniq_signatures(option):
        h = option.signature_hash()
        if h in signatures:
            return False
        signatures.add(h)
        return True
    return list(filter(uniq_signatures, arg_options))

def make_stateless(arg_options):
    """Converts stateful options to stateless options.
    """
    stateless_options = []
    def self_to_(new_name):
        def self_to_new(arg):
            if arg.name != 'self':
                return arg
            return Argument(arg.type, new_name)
        return self_to_new

    # This has to go first, because it will be favored during unique
    for option in arg_options:
        if option.is_self_optional():
            new = option.copy()
            new.map_arguments(self_to_('_res_new'))
            if new.return_type == 'self':
                new.return_type = 'STATELESS NEW self'
            stateless_options.append(new)

    for option in arg_options:
        provided = option.copy()
        provided.map_arguments(self_to_('_res'))
        if provided.return_type == 'self':
            provided.return_type = 'STATELESS PROV self'
        if provided.return_type == 'new SelfIndexPair':
            provided.insert_argument(0, Argument('THPTensor*', '_res'))
            provided.return_type = 'STATELESS PROV new SelfIndexPair'
            stateless_options.append(provided.copy())
            provided.insert_argument(1, Argument('THPLongTensor*', '_res_ind'))
            provided.return_type = 'STATELESS PROV2 new SelfIndexPair'
        stateless_options.append(provided)

    return stateless_options


def generate_all_options(options):
    """Generates code implementing all argument options

       Options are sorted according to their argument count. Ones with equal
       counts are wrapped in the same if block, that checks how many
       arguments have been provided. This allows to ignore checking some
       argument configurations, and save a couple of cycles (PyArg_ParseTuple
       calls add some overhead).
    """
    impl = ''
    prev_arg_count = -1
    for option in sorted(options, key=arg_count):
        num_args = arg_count(option)
        if num_args > prev_arg_count:
            # Nothing to close if it's the first option
            if prev_arg_count != -1:
                impl += '  }\n'
            impl += Template('  if (_argcount == $numargs) {') \
                        .substitute({'numargs': num_args})
            prev_arg_count = num_args
        else:
            impl += '    PyErr_Clear();'
        impl += generate_option(option)
    # Close last argcount block
    impl += '  }\n'
    return impl


def generate_option(option):
    """Generates code implementing one call option
    """
    format_str = make_format_str(option.arguments)
    argparse_args = argparse_arguments(option.arguments)
    expression = build_expression(option)
    # This is not only an optimization, but also prevents PyArg_ParseTuple from
    # segfaulting - it doesn't handle args == NULL case.
    is_already_provided = argfilter()
    if sum(1 for arg in option.arguments if not is_already_provided(arg)) == 0:
        return expression + ';'
    return OPTION_CODE.substitute({
        'format': format_str,
        'parse_args': argparse_args,
        'expr': expression,
    })


def arg_count(option):
    """Counts how many arguments should be provided for a given option.
    """
    is_already_provided = argfilter()
    return sum(1 for arg in option.arguments if not is_already_provided(arg))


def make_format_str(args):
    """Returns a format string for PyArg_ParseTuple.
    """
    is_already_provided = argfilter()
    s = ''.join(FORMAT_STR_MAP[arg.type] for arg in args \
                     if not is_already_provided(arg))
    return '"' + s + '"'


def argfilter():
    """Returns a function, that allows to filter out already known arguments.

       self is used only in stateful mode and is always provided.
       _res_new is allocated automatically before call, so it is known.
       CONSTANT arguments are literals.
       Repeated arguments do not need to be specified twice.
    """
    # use class rather than nonlocal to maintain 2.7 compat
    # see http://stackoverflow.com/questions/3190706/nonlocal-keyword-in-python-2-x
    # TODO: check this works
    class context:
        provided = set()
    def is_already_provided(arg):
        ret = False
        ret |= arg.name == 'self'
        ret |= arg.name == '_res_new'
        ret |= arg.type == 'CONSTANT'
        ret |= arg.type == 'EXPRESSION'
        ret |= arg.name in context.provided
        context.provided.add(arg.name)
        return ret
    return is_already_provided


def argparse_arguments(args):
    """Builds a list of variables (and type pointers for type checking) to
       be used with PyArg_ParseTuple.
    """
    is_already_provided = argfilter()
    s = ', '
    for arg in args:
        if is_already_provided(arg):
            continue
        parsed_type = ARGPARSE_TYPE_CHECK.get(arg.type)
        if parsed_type:
            s += '&' + parsed_type + ', '
        s += '&' + arg.name + ', '
    return s.rstrip()[:-1] # Remove whitespace and trailing comma


def build_expression(option):
    """Creates an expression that executes given option.

       Every such expression is basically a TH library call, wrapped in a
       function that wraps it's return value in a Python object.
    """
    def make_arg(arg):
        if arg.type == 'EXPRESSION':
            return arg.name.format(*tuple(a.name for a in all_args))
        return arg.name + ('->cdata' if arg.type in CDATA_TYPES else '')
    additional_args = ADDITIONAL_ARGS.get(option.return_type, ())
    if isinstance(additional_args, dict):
        arg_iter = deepcopy(option.arguments)
        for k,v in additional_args.items():
            arg_iter.insert(k, v)
    else:
        arg_iter = chain(additional_args, option.arguments)
    all_args = list(arg_iter)

    args = ', '.join(make_arg(arg) for arg in all_args)

    return option.wrap(args)
