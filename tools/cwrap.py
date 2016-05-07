from string import Template
from pprint import pprint

OPTION_SEPARATOR = ' -> '
ARGUMENT_PREFIX = '  -'
CONSTANT = 'CONSTANT'

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
            new_content += generate_function(func_lines)
        elif in_declaration:
            func_lines.append(line)
        else:
            new_content += line + '\n'
    with open(filename.replace('.cwrap', ''), 'w') as f:
        f.write(new_content)


# Basic templates for declarations
DEFINITION_START = Template("""
static PyObject * THPTensor_(${name})(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  Py_ssize_t _argcount = PyTuple_Size(args);
""")
# TODO: Better error handling when args are bad
DEFINITION_END = """
  return NULL;
  END_HANDLE_TH_ERRORS
}
"""

# Transforms applied to argument types declared in the definition
# these are mostly, so that the * can be omitted for convenience and clarity
TYPE_TRANSFORMS = {
    'THTensor': 'THPTensor*',
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
    'real': 'O&',
    'long': 'l',
}

# If O! is specified for any type in FORMAT_STR_MAP you should specify it's
# type here
ARGPARSE_TYPE_CHECK = {
    'THPTensor*': 'THPTensorType'
}

# Code used to convert return values to Python objects
RETURN_WRAPPER = {
    'THTensor':       Template('return THPTensor_(newObject)($expr)'),
    'THStorage':      Template('return THPStorage_(newObject)($expr)'),
    'THLongStorage':  Template('return THPLongStorage_newObject($expr)'),
    'bool':           Template('return PyBool_FromLong($expr)'),
    'long':           Template('return PyLong_FromLong($expr)'),
    'double':         Template('return PyFloat_FromDouble($expr)'),
    'self':           Template('$expr; Py_INCREF(self); return (PyObject*)self'),
}

# Types for which it's necessary to extract cdata
CDATA_TYPES = set(('THPTensor*', 'THPStorage*'))


def generate_function(lines):
    assert len(lines) > 1
    lines = remove_indentation(lines)
    function_name = lines[0]
    arg_options, variables = parse_lines(lines)
    definition = DEFINITION_START.substitute({'name': function_name})

    # Declare variables
    for variable in variables:
        if not is_already_provided({'type': variable[0], 'name': variable[1]}):
            definition += '  {} {};\n'.format(*variable)

    # Generate function body
    definition += generate_all_options(arg_options)

    definition += DEFINITION_END
    return definition

def remove_indentation(lines):
    """Removes 2 spaces from the left from each line.

       If anyone wants to use another indentation depth, please update
       this function first.
    """
    return [line[2:] for line in lines]

def parse_lines(lines):
    """Parses cwrap declaration.

       Accepts an iterable of lines.
       Returns a pair of argument options and variables.
    """
    arg_options = []
    variables = set()

    for line in lines[1:]:
        if is_option_declaration(line):
            thname, _, rettype = line.partition(OPTION_SEPARATOR)
            arg_options.append({
                'thname': thname,
                'return_type': TYPE_TRANSFORMS.get(rettype, rettype),
                'arguments': []
            })
        else:
            assert line.startswith(ARGUMENT_PREFIX)
            arg = line.replace(ARGUMENT_PREFIX, '').strip()
            if arg == 'self':
                t, name = 'THTensor', 'self'
            else:
                t, name = arg.split(' ')
            t = TYPE_TRANSFORMS.get(t, t)
            variables.add((t, name))
            arg_options[-1]['arguments'].append({'type': t, 'name': name})
    return arg_options, variables


def is_option_declaration(line):
    return OPTION_SEPARATOR in line


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
    format_str = make_format_str(option['arguments'])
    argparse_args = argparse_arguments(option['arguments'])
    expression = build_expression(option)
    return OPTION_CODE.substitute({
        'format': format_str,
        'parse_args': argparse_args,
        'expr': expression,
        'numargs': arg_count(option)
    })


def arg_count(option):
    """Counts how many arguments should be provided for a given option.
    """
    return sum(1 for arg in option['arguments'] if not is_already_provided(arg))


def make_format_str(args):
    """Returns a format string for PyArg_ParseTuple.
    """
    s = ''.join(FORMAT_STR_MAP[arg['type']] for arg in args \
                     if not is_already_provided(arg))
    return '"' + s + '"'


def is_already_provided(arg):
    """Returns True, if arg's value is already known.

       self and constant arguments don't need to be provided to function call.
    """
    return arg['name'] == 'self' or arg['type'] == CONSTANT


def argparse_arguments(args):
    """Builds a list of variables (and type pointers for type checking) to
       be used with PyArg_ParseTuple.
    """
    s = ', '
    for arg in args:
        if is_already_provided(arg):
            continue
        parsed_type = ARGPARSE_TYPE_CHECK.get(arg['type'])
        if parsed_type:
            s += '&' + parsed_type + ', '
        elif arg['type'] == 'real':
            s += 'THPUtils_(parseReal), '
        s += '&' + arg['name'] + ', '
    return s.rstrip()[:-1] # Remove whitespace and trailing comma


def build_expression(option):
    """Creates an expression that executes given option.

       Every such expression is basically a TH library call, wrapped in a
       function that wraps it's return value in a Python object.
    """
    def make_arg(name, type, **kwargs):
        return name + ('->cdata' if type in CDATA_TYPES else '')

    th_call = 'THTensor_({})('.format(option['thname'])
    th_call += ', '.join(make_arg(**arg) for arg in option['arguments'])
    th_call += ')'

    return RETURN_WRAPPER[option['return_type']].substitute({'expr': th_call})
