from string import Template
from copy import deepcopy
from . import CWrapPlugin
from itertools import product, chain
from collections import OrderedDict

class THPPlugin(CWrapPlugin):

    TYPE_UNPACK = {
        'THFloatTensor*':   Template('((THPFloatTensor*)$arg)->cdata'),
        'THDoubleTensor*':  Template('((THPDoubleTensor*)$arg)->cdata'),
        'THLongTensor*':    Template('((THPLongTensor*)$arg)->cdata'),
        'THIntTensor*':     Template('((THPIntTensor*)$arg)->cdata'),
        'THTensor*':        Template('((THPTensor*)$arg)->cdata'),
        'THBoolTensor*':    Template('((THPBoolTensor*)$arg)->cdata'),
        'THIndexTensor*':   Template('((THPIndexTensor*)$arg)->cdata'),
        'THLongStorage*':   Template('((THPLongStorage*)$arg)->cdata'),
        'THStorage*':       Template('((THPStorage*)$arg)->cdata'),
        'THGenerator*':     Template('((THPGenerator*)$arg)->cdata'),
        'THSize*':          Template('__size.get()'),
        'THStride*':        Template('__stride.get()'),
        'void*':            Template('THPUtils_unpackLong($arg)'),
        'long':             Template('THPUtils_unpackLong($arg)'),
        'int':              Template('THPUtils_unpackLong($arg)'),
        'bool':             Template('($arg == Py_True ? true : false)'),
        'float':            Template('THPFloatUtils_unpackReal($arg)'),
        'double':           Template('THPDoubleUtils_unpackReal($arg)'),
        'real':             Template('THPUtils_(unpackReal)($arg)'),
        'accreal':          Template('THPUtils_(unpackAccreal)($arg)'),
    }

    TYPE_CHECK = {
        'THDoubleTensor*':  Template('(PyObject*)Py_TYPE($arg) == THPDoubleTensorClass'),
        'THFloatTensor*':   Template('(PyObject*)Py_TYPE($arg) == THPFloatTensorClass'),
        'THLongTensor*':    Template('(PyObject*)Py_TYPE($arg) == THPLongTensorClass'),
        'THIntTensor*':     Template('(PyObject*)Py_TYPE($arg) == THPIntTensorClass'),
        'THCudaTensor*':    Template('(PyObject*)Py_TYPE($arg) == THCPFloatTensorClass'),
        'THTensor*':        Template('(PyObject*)Py_TYPE($arg) == THPTensorClass'),
        'THBoolTensor*':    Template('(PyObject*)Py_TYPE($arg) == THPBoolTensorClass'),
        'THIndexTensor*':   Template('(PyObject*)Py_TYPE($arg) == THPIndexTensorClass'),
        'THLongStorage*':   Template('(PyObject*)Py_TYPE($arg) == THPLongStorageClass'),
        'THStorage*':       Template('(PyObject*)Py_TYPE($arg) == THPStorageClass'),
        'THGenerator*':     Template('(PyObject*)Py_TYPE($arg) == THPGeneratorClass'),
        'THSize*':          Template('THPUtils_tryUnpackLongs($arg, __size)'),
        'THStride*':        Template('THPUtils_tryUnpackLongs($arg, __stride)'),
        'void*':            Template('THPUtils_checkLong($arg)'),
        'long':             Template('THPUtils_checkLong($arg)'),
        'int':              Template('THPUtils_checkLong($arg)'),
        'bool':             Template('(($arg == Py_True) || ($arg == Py_False))'),
        'float':            Template('THPFloatUtils_checkReal($arg)'),
        'double':           Template('THPDoubleUtils_checkReal($arg)'),
        'real':             Template('THPUtils_(checkReal)($arg)'),
        'accreal':          Template('THPUtils_(checkAccreal)($arg)'),
    }

    SIZE_VARARG_CHECK = Template('THPUtils_tryUnpackLongVarArgs(args, $idx, __size)')

    RETURN_WRAPPER = {
        'THTensor*':        Template('return THPTensor_(New)($result);'),
        'THLongStorage*':   Template('return THPLongStorage_New($result);'),
        # TODO: make it smarter - it should return python long if result doesn't fit into an int
        'long':             Template('return PyInt_FromLong($result);'),
        'accreal':          Template('return THPUtils_(newAccreal)($result);'),
        'self':             Template('Py_INCREF(self);\nreturn (PyObject*)self;'),
        'real':             Template('return THPUtils_(newReal)($result);'),
    }

    TENSOR_METHODS_DECLARATION = Template("""
static PyMethodDef THPTensor_$stateless(methods)[] = {
$methods
  {NULL}
};
""")

    WRAPPER_TEMPLATE = Template("""\
PyObject * $name(PyObject *self, PyObject *args, PyObject *kwargs)
{
    HANDLE_TH_ERRORS
    int __tuplecount = args ? PyTuple_Size(args) : 0;
    int __dictcount = kwargs ? PyDict_Size(kwargs) : 0;
    int __argcount = __tuplecount + __dictcount;
    $variables

    $options
    }

    THPUtils_invalidArguments(args, "$readable_name", $num_options, $expected_args);
    return NULL;
    END_HANDLE_TH_ERRORS
}
""")

    ALLOCATE_TMPL = Template("""\
THP${type}TensorPtr _${name}_guard = (THP${type}Tensor*) THP${type}Tensor_NewEmpty();
if (!_${name}_guard.get()) return NULL;
THP${type}Tensor* $name = _${name}_guard.get();
""")

    ALLOCATE_CUDA = Template("""\
#if IS_CUDA
${cuda}
#else
${cpu}
#endif
""")

    def _allocate(typename, tmpl, cuda_tmpl=None):
        code = tmpl.safe_substitute(type=typename)
        if typename == '':
            code = code.replace('NewEmpty', '(NewEmpty)')
        if cuda_tmpl:
            cuda_code = code.replace('THP', 'THCP')
            code = cuda_tmpl.substitute(cuda=cuda_code, cpu=code)
        return Template(code)

    ALLOCATE_TYPE = {
        'THTensor*':        _allocate('', ALLOCATE_TMPL),
        'THLongTensor*':    _allocate('Long', ALLOCATE_TMPL),
        'THIntTensor*':     _allocate('Int', ALLOCATE_TMPL),
        'THBoolTensor*':    _allocate('Byte', ALLOCATE_TMPL, ALLOCATE_CUDA),
        'THIndexTensor*':   _allocate('Long', ALLOCATE_TMPL, ALLOCATE_CUDA),
    }

    TYPE_NAMES = {
        'THTensor*': '" THPTensorStr "',
        'THStorage*': '" THPStorageStr "',
        'THGenerator*': 'torch.Generator',
        'THLongStorage*': '" THPModuleStr "LongStorage',
        'THLongTensor*': '" THPModuleStr "LongTensor',
        'THIntTensor*': '" THPModuleStr "IntTensor',
        'THBoolTensor*': '" THPModuleStr "ByteTensor',
        'THIndexTensor*': '" THPModuleStr "LongTensor',
        'THFloatTensor*': '" THPModuleStr "FloatTensor',
        'THDoubleTensor*': '" THPModuleStr "DoubleTensor',
        'THSize*': 'torch.Size',
        'THStride*': 'tuple',
        'long': 'int',
        'real': '" RealStr "',
        'double': 'float',
        'accreal': '" RealStr "',
        'bool': 'bool',
    }

    def __init__(self):
        self.declarations = []
        self.stateless_declarations = []
        self.docstrings = []

    def get_type_unpack(self, arg, option):
        return self.TYPE_UNPACK.get(arg['type'], None)

    def get_type_check(self, arg, option):
        if arg['type'] == 'THSize*' and arg.get('long_args', False):
            return self.SIZE_VARARG_CHECK
        return self.TYPE_CHECK.get(arg['type'], None)

    # TODO: argument descriptions shouldn't be part of THP, but rather a general cwrap thing
    def get_wrapper_template(self, declaration):
        arg_desc = OrderedDict()

        def format_arg(arg, var_args=False):
            if var_args and arg.get('long_args', False):
                return 'int ... ' + arg['name']
            else:
                return self.TYPE_NAMES[arg['type']] + ' ' + arg['name']

        def format_args(args, var_args=False):
            option_desc = [format_arg(arg, var_args)
                           for arg in args
                           if not arg.get('ignore_check', False)]
            if option_desc:
                return '({})'.format(', '.join(option_desc))
            else:
                return 'no arguments'

        for option in declaration['options']:
            arg_desc[format_args(option['arguments'], False)] = True
            arg_desc[format_args(option['arguments'], True)] = True

        arg_desc = sorted(list(arg_desc.keys()), key=len)
        arg_desc = ['"' + desc + '"' for desc in arg_desc]
        arg_str = ', '.join(arg_desc)
        variables_str = '\n'.join(declaration.get('variables', []))
        if 'stateless' in declaration['name']:
            readable_name = 'torch.' + declaration['python_name']
        else:
            readable_name = declaration['python_name']
        return Template(self.WRAPPER_TEMPLATE.safe_substitute(
            readable_name=readable_name, num_options=len(arg_desc),
            expected_args=arg_str, variables=variables_str))

    def get_return_wrapper(self, option):
        return self.RETURN_WRAPPER.get(option['return'], None)

    def get_arg_accessor(self, arg, option):
        if arg['name'] == 'self':
            return 'self'
        if 'allocate' in arg and arg['allocate']:
            return arg['name']

    def process_docstrings(self):
        for declaration in self.declarations:
            docstr = declaration.get('docstring_method')
            if docstr is None:
                continue
            declaration['docstring_content'] = docstr.replace('\n', '\\n')
            declaration['docstring_var'] = 'docstr_' + declaration['python_name']
        for declaration in self.stateless_declarations:
            docstr = declaration.get('docstring_stateless')
            if docstr is None:
                continue
            declaration['docstring_content'] = docstr.replace('\n', '\\n')
            declaration['docstring_var'] = 'stateless_docstr_' + declaration['python_name']

    def process_declarations(self, declarations):
        new_declarations = []
        register_only = [d for d in declarations if d.get('only_register', False)]
        declarations = [d for d in declarations if not d.get('only_register', False)]

        def has_arg_type(declaration, type_name):
            return any(arg['type'] == type_name
                       for option in declaration['options']
                       for arg in option['arguments'])

        def has_long_args(declaration):
            return any(arg.get('long_args', False)
                       for option in declaration['options']
                       for arg in option['arguments'])

        for declaration in declarations:
            if declaration.get('only_register', False):
                continue
            declaration.setdefault('python_name', declaration['name'])
            declaration.setdefault('variables', [])
            if has_arg_type(declaration, 'THSize*'):
                declaration['variables'] += ['THLongStoragePtr __size;']
            if has_arg_type(declaration, 'THStride*'):
                declaration['variables'] += ['THLongStoragePtr __stride;']
            if has_long_args(declaration):
                declaration['no_kwargs'] = True
            if declaration.get('with_stateless', False) or declaration.get('only_stateless', False):
                stateless_declaration = self.make_stateless(deepcopy(declaration))
                new_declarations.append(stateless_declaration)
                self.stateless_declarations.append(stateless_declaration)
            if declaration.get('only_stateless', False):
                continue

            self.declarations.append(declaration)
            declaration['name'] = 'THPTensor_({})'.format(declaration['name'])
            for option in declaration['options']:
                option['cname'] = 'THTensor_({})'.format(option['cname'])
                for arg in option['arguments']:
                    if arg['name'] == 'self':
                        arg['ignore_check'] = True
                    if 'allocate' in arg and arg['allocate']:
                        arg['ignore_check'] = True
            # TODO: we can probably allow duplicate signatures once we implement
            # keyword arguments
            declaration['options'] = self.filter_unique_options(declaration['options'])


        declarations = [d for d in declarations if not d.get('only_stateless', False)]
        self.declarations.extend(filter(lambda x: not x.get('only_stateless', False), register_only))
        self.stateless_declarations.extend(filter(lambda x: x.get('only_stateless', False), register_only))

        self.process_docstrings()

        all_declarations = declarations + new_declarations
        return all_declarations

    def make_stateless(self, declaration):
        declaration['name'] = 'THPTensor_stateless_({})'.format(declaration['name'])
        new_options = []
        for option in declaration['options']:
            option['cname'] = 'THTensor_({})'.format(option['cname'])
            allocated = []
            for i, arg in enumerate(option['arguments']):
                if 'allocate' in arg and arg['allocate']:
                    arg['ignore_check'] = True
                    allocated.append(i)
                if arg['name'] == 'self':
                    arg['name'] = 'source'
            for permutation in product((True, False), repeat=len(allocated)):
                option_copy = deepcopy(option)
                for i, bit in zip(allocated, permutation):
                    arg = option_copy['arguments'][i]
                    # By default everything is allocated, so we don't have to do anything
                    if not bit:
                        del arg['allocate']
                        del arg['ignore_check']
                new_options.append(option_copy)
        declaration['options'] = self.filter_unique_options(declaration['options'] + new_options)
        return declaration

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

    def declare_methods(self, stateless):
        tensor_methods = ''
        for declaration in (self.declarations if not stateless else self.stateless_declarations):
            flags = 'METH_VARARGS'
            flags += ' | ' + declaration.get('method_flags') if 'method_flags' in declaration else ''
            if not declaration.get('only_register'):
                flags += ' | METH_KEYWORDS'
            if declaration.get('override_method_flags'):
                flags = declaration['override_method_flags']
            entry = Template('  {"$python_name", (PyCFunction)$name, $flags, $docstring},\n').substitute(
                    python_name=declaration['python_name'], name=declaration['name'], flags=flags,
                    docstring=declaration.get('docstring_var', 'NULL')
                )
            if 'defined_if' in declaration:
                entry = self.preprocessor_guard(entry, declaration['defined_if'])
            tensor_methods += entry
        return self.TENSOR_METHODS_DECLARATION.substitute(methods=tensor_methods, stateless=('' if not stateless else 'stateless_'))

    def process_full_file(self, code):
        # We have to find a place before all undefs
        idx = code.find('// PUT DEFINITIONS IN HERE PLEASE')
        return code[:idx] + self.declare_methods(False) + self.declare_methods(True) + code[idx:]

    def preprocessor_guard(self, code, condition):
            return '#if ' + condition + '\n' + code + '#endif\n'

    def process_wrapper(self, code, declaration):
        if 'defined_if' in declaration:
            return self.preprocessor_guard(code, declaration['defined_if'])
        return code

    def process_all_unpacks(self, code, option):
        return 'LIBRARY_STATE ' + code

    def process_all_checks(self, code, option):
        if any(arg.get('long_args', False) for arg in option['arguments']):
            code = code.replace('__argcount ==', '__argcount >=')
        return code

    def process_option_code_template(self, template, option):
        new_args = []
        for arg in option['arguments']:
            if 'allocate' in arg and arg['allocate']:
                new_args.append(self.ALLOCATE_TYPE[arg['type']].substitute(name=arg['name']))
        template = new_args + template
        return template

    def generate_docstrings_cpp(self):
        template = Template('char* $name = "$content";')
        return '\n\n'.join(
                template.substitute(name=decl['docstring_var'], content=decl['docstring_content'])
                for decl in chain(self.declarations, self.stateless_declarations)
                if 'docstring_var' in decl)

    def generate_docstrings_h(self):
        template = Template('extern char* $name;')
        return '\n\n'.join(
                template.substitute(name=decl['docstring_var'])
                for decl in chain(self.declarations, self.stateless_declarations)
                if 'docstring_var' in decl)
