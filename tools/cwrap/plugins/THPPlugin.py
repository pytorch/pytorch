from string import Template
from copy import deepcopy
from . import CWrapPlugin
from itertools import product, chain
from collections import OrderedDict


class THPPlugin(CWrapPlugin):

    TYPE_UNPACK = {
        'THFloatTensor*': Template('((THPFloatTensor*)$arg)->cdata'),
        'THDoubleTensor*': Template('((THPDoubleTensor*)$arg)->cdata'),
        'THLongTensor*': Template('((THPLongTensor*)$arg)->cdata'),
        'THIntTensor*': Template('((THPIntTensor*)$arg)->cdata'),
        'THTensor*': Template('((THPTensor*)$arg)->cdata'),
        'THBoolTensor*': Template('((THPBoolTensor*)$arg)->cdata'),
        'THIndexTensor*': Template('((THPIndexTensor*)$arg)->cdata'),
        'THIntegerTensor*': Template('((THPIntegerTensor*)$arg)->cdata'),

        'THCudaTensor*': Template('((THCPFloatTensor*)$arg)->cdata'),
        'THCudaDoubleTensor*': Template('((THCPDoubleTensor*)$arg)->cdata'),
        'THCudaIntTensor*': Template('((THCPIntTensor*)$arg)->cdata'),
        'THCudaLongTensor*': Template('((THCPLongTensor*)$arg)->cdata'),

        'THSFloatTensor*': Template('((THSPFloatTensor*)$arg)->cdata'),
        'THSDoubleTensor*': Template('((THSPDoubleTensor*)$arg)->cdata'),
        'THSLongTensor*': Template('((THSPLongTensor*)$arg)->cdata'),
        'THSIntTensor*': Template('((THSPIntTensor*)$arg)->cdata'),
        'THSTensor*': Template('((THSPTensor*)$arg)->cdata'),
        'THSBoolTensor*': Template('((THSPBoolTensor*)$arg)->cdata'),
        'THSIndexTensor*': Template('((THSPIndexTensor*)$arg)->cdata'),

        'THLongStorage*': Template('((THPLongStorage*)$arg)->cdata'),
        'THStorage*': Template('((THPStorage*)$arg)->cdata'),
        'THGenerator*': Template('THPGenerator_TH_CData($arg)'),
        'THSize*': Template('__size.get()'),
        'THStride*': Template('__stride.get()'),
        'void*': Template('THPUtils_unpackLong($arg)'),
        'long': Template('THPUtils_unpackLong($arg)'),
        'int': Template('((int) THPUtils_unpackLong($arg))'),
        'int64_t': Template('THPUtils_unpackLong($arg)'),
        'bool': Template('($arg == Py_True ? true : false)'),
        'float': Template('THPFloatUtils_unpackReal($arg)'),
        'double': Template('THPDoubleUtils_unpackReal($arg)'),
        'real': Template('THPUtils_(unpackReal)($arg)'),
    }

    TYPE_CHECK = {
        'THDoubleTensor*': Template('(PyObject*)Py_TYPE($arg) == THPDoubleTensorClass'),
        'THFloatTensor*': Template('(PyObject*)Py_TYPE($arg) == THPFloatTensorClass'),
        'THLongTensor*': Template('(PyObject*)Py_TYPE($arg) == THPLongTensorClass'),
        'THIntTensor*': Template('(PyObject*)Py_TYPE($arg) == THPIntTensorClass'),
        'THTensor*': Template('(PyObject*)Py_TYPE($arg) == THPTensorClass'),
        'THBoolTensor*': Template('(PyObject*)Py_TYPE($arg) == THPBoolTensorClass'),
        'THIndexTensor*': Template('(PyObject*)Py_TYPE($arg) == THPIndexTensorClass'),
        'THIntegerTensor*': Template('(PyObject*)Py_TYPE($arg) == THPIntegerTensorClass'),

        'THCudaTensor*': Template('(PyObject*)Py_TYPE($arg) == THCPFloatTensorClass'),
        'THCudaDoubleTensor*': Template('(PyObject*)Py_TYPE($arg) == THCPDoubleTensorClass'),
        'THCudaIntTensor*': Template('(PyObject*)Py_TYPE($arg) == THCPIntTensorClass'),
        'THCudaLongTensor*': Template('(PyObject*)Py_TYPE($arg) == THCPLongTensorClass'),

        'THSDoubleTensor*': Template('(PyObject*)Py_TYPE($arg) == THSPDoubleTensorClass'),
        'THSFloatTensor*': Template('(PyObject*)Py_TYPE($arg) == THSPFloatTensorClass'),
        'THSLongTensor*': Template('(PyObject*)Py_TYPE($arg) == THSPLongTensorClass'),
        'THSIntTensor*': Template('(PyObject*)Py_TYPE($arg) == THSPIntTensorClass'),
        'THSTensor*': Template('(PyObject*)Py_TYPE($arg) == THSPTensorClass'),
        'THSBoolTensor*': Template('(PyObject*)Py_TYPE($arg) == THSPBoolTensorClass'),
        'THSIndexTensor*': Template('(PyObject*)Py_TYPE($arg) == THSPIndexTensorClass'),

        'THLongStorage*': Template('(PyObject*)Py_TYPE($arg) == THPLongStorageClass'),
        'THStorage*': Template('(PyObject*)Py_TYPE($arg) == THPStorageClass'),
        'THGenerator*': Template('(PyObject*)Py_TYPE($arg) == THPGeneratorClass'),
        'THSize*': Template('THPUtils_tryUnpackLongs($arg, __size)'),
        'THStride*': Template('THPUtils_tryUnpackLongs($arg, __stride)'),
        'void*': Template('THPUtils_checkLong($arg)'),
        'long': Template('THPUtils_checkLong($arg)'),
        'int64_t': Template('THPUtils_checkLong($arg)'),
        'int': Template('THPUtils_checkLong($arg)'),
        'bool': Template('PyBool_Check($arg)'),
        'float': Template('THPFloatUtils_checkReal($arg)'),
        'double': Template('THPDoubleUtils_checkReal($arg)'),
        'real': Template('THPUtils_(checkReal)($arg)'),
    }

    SIZE_VARARG_CHECK = Template('THPUtils_tryUnpackLongVarArgs(args, $idx, __size)')

    RETURN_WRAPPER = {
        'THTensor*': Template('return THPTensor_(New)($result);'),
        'THSTensor*': Template('return THSPTensor_(New)($result);'),
        'THIndexTensor*': Template('return THPIndexTensor_(New)($result);'),
        'THLongTensor*': Template('return THPLongTensor_New($result);'),
        'THLongStorage*': Template('return THPLongStorage_New($result);'),
        'THCudaIntTensor*': Template('return THCPIntTensor_New($result);'),
        'THCudaLongTensor*': Template('return THCPLongTensor_New($result);'),
        # TODO: make it smarter - it should return python long if result doesn't fit into an int
        'long': Template('return PyInt_FromLong($result);'),
        'int64_t': Template('return PyInt_FromLong($result);'),
        'int': Template('return PyLong_FromLong($result);'),
        'self': Template('Py_INCREF(self);\nreturn (PyObject*)self;'),
        'real': Template('return THPUtils_(newReal)($result);'),
    }

    TENSOR_METHODS_DECLARATION = Template("""
static PyMethodDef TH${sparse}PTensor_$stateless(methods)[] = {
    $methods
    {NULL}
};
""")

    WRAPPER_TEMPLATE = Template("""\
PyObject * $name(PyObject *self, PyObject *args, PyObject *kwargs)
{
    HANDLE_TH_ERRORS
    int __tuplecount = args ? (int) PyTuple_Size(args) : 0;
    int __dictcount = kwargs ? (int) PyDict_Size(kwargs) : 0;
    int __argcount = __tuplecount + __dictcount;
    $variables
    $init

    $options
    }

    THPUtils_invalidArguments(args, kwargs, "$readable_name", $num_options, $expected_args);
    return NULL;
    END_HANDLE_TH_ERRORS
}
    """)

    ALLOCATE_TMPL = Template("""\
THP${type}TensorPtr _${name}_guard((THP${type}Tensor*) THP${type}Tensor_NewEmpty());
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

    def _allocate(typename, tmpl, cuda_tmpl=None, sparse=False):
        code = tmpl.safe_substitute(type=typename)
        if typename == '':
            code = code.replace('NewEmpty', '(NewEmpty)')
        if cuda_tmpl:
            cuda_code = code.replace('THP', 'THCP')
            code = cuda_tmpl.substitute(cuda=cuda_code, cpu=code)
        if sparse:
            code = code.replace('THP', 'THSP')
            code = code.replace('THCP', 'THCSP')
        return Template(code)

    ALLOCATE_TYPE = {
        'THTensor*': _allocate('', ALLOCATE_TMPL),
        'THLongTensor*': _allocate('Long', ALLOCATE_TMPL),
        'THIntTensor*': _allocate('Int', ALLOCATE_TMPL),
        'THBoolTensor*': _allocate('Byte', ALLOCATE_TMPL, ALLOCATE_CUDA),
        'THIndexTensor*': _allocate('Long', ALLOCATE_TMPL, ALLOCATE_CUDA),
        'THIntegerTensor*': _allocate('Int', ALLOCATE_TMPL, ALLOCATE_CUDA),

        'THSTensor*': _allocate('', ALLOCATE_TMPL, sparse=True),
    }

    TYPE_NAMES = {
        'THTensor*': '" THPTensorStr "',
        'THSTensor*': '" THSPTensorStr "',
        'THStorage*': '" THPStorageStr "',
        'THGenerator*': 'torch.Generator',
        'THLongStorage*': '" THPModuleStr "LongStorage',
        'THLongTensor*': '" THPModuleStr "LongTensor',
        'THIntTensor*': '" THPModuleStr "IntTensor',
        'THBoolTensor*': '" THPModuleStr "ByteTensor',
        'THIndexTensor*': '" THPModuleStr "LongTensor',
        'THIntegerTensor*': '" THPModuleStr "IntTensor',
        'THFloatTensor*': '" THPModuleStr "FloatTensor',
        'THDoubleTensor*': '" THPModuleStr "DoubleTensor',
        'THCudaTensor*': 'torch.cuda.FloatTensor',
        'THCudaDoubleTensor*': 'torch.cuda.DoubleTensor',
        'THCudaIntTensor*': 'torch.cuda.IntTensor',
        'THCudaLongTensor*': 'torch.cuda.LongTensor',
        'THSize*': 'torch.Size',
        'THStride*': 'tuple',
        'long': 'int',
        'int64_t': 'int',
        'int': 'int',
        'real': '" RealStr "',
        'double': 'float',
        'accreal': '" RealStr "',
        'bool': 'bool',
        'const char*': 'bool',  # Can come only from bool option.
    }

    OUT_INIT = """
    ___out = kwargs ? PyDict_GetItemString(kwargs, "out") : NULL;
    if (___out == Py_None) { ___out = NULL; __dictcount--; __argcount--; }
    """

    def __init__(self):
        self.declarations = []
        self.stateless_declarations = []
        self.docstrings = []

    BACKEND_SUBSTITUTIONS = {
        'CPU': 'TH',
        'CUDA': 'THCuda',
    }

    def substitute_tensor_backend(self, arg, option):
        if 'Backend' in arg['type']:
            arg['type'] = arg['type'].replace('Backend',
                                              self.BACKEND_SUBSTITUTIONS.get(option['backends'][0]))
            # handle the fact that THCudaTensor isn't THCudaFloatTensor
            if option['backends'][0] == 'CUDA' and 'Float' in arg['type']:
                arg['type'] = arg['type'].replace('Float', '')

    def get_type_unpack(self, arg, option):
        self.substitute_tensor_backend(arg, option)
        return self.TYPE_UNPACK.get(arg['type'], None)

    def get_type_check(self, arg, option):
        if arg['type'] == 'THSize*' and arg.get('long_args', False):
            return self.SIZE_VARARG_CHECK
        self.substitute_tensor_backend(arg, option)
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
                           if not arg.get('ignore_check', False) and
                           not arg.get('output')]
            output_args = list(filter(lambda a: a.get('output'), args))
            if output_args:
                if len(output_args) > 1:
                    out_type = 'tuple['
                    out_type += ', '.join(
                        self.TYPE_NAMES[arg['type']] for arg in output_args)
                    out_type += ']'
                    option_desc += ['#' + out_type + ' out']
                else:
                    arg = output_args[0]
                    option_desc += ['#' + self.TYPE_NAMES[arg['type']] + ' out']

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
        init_str = '\n'.join(declaration.get('init', []))
        if 'stateless' in declaration['name']:
            readable_name = 'torch.' + declaration['python_name']
        else:
            readable_name = declaration['python_name']
        return Template(self.WRAPPER_TEMPLATE.safe_substitute(
            readable_name=readable_name, num_options=len(arg_desc),
            expected_args=arg_str, variables=variables_str, init=init_str))

    def get_return_wrapper(self, option):
        return self.RETURN_WRAPPER.get(option['return'], None)

    def get_arg_accessor(self, arg, option):
        if arg['name'] == 'self':
            return 'self'
        if arg.get('output'):
            if not option['output_provided']:
                return arg['name']
            if option['output_count'] == 1:
                return '___out'
            else:
                return 'PyTuple_GET_ITEM(___out, {})'.format(arg['output_idx'])

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

    def generate_out_options(self, declaration):
        new_options = []
        declaration.setdefault('init', [])
        declaration['init'] += [self.OUT_INIT]
        for option in declaration['options']:
            out_idx = []
            for i, arg in enumerate(option['arguments']):
                if arg.get('output'):
                    out_idx.append(i)
            if not out_idx:
                option['has_output'] = True
                option['output_provided'] = False
                new_options.append(option)
                continue
            for output_provided in (True, False):
                option_copy = deepcopy(option)
                option_copy['has_output'] = True
                option_copy['output_provided'] = output_provided
                option_copy['output_count'] = len(out_idx)
                for i, idx in enumerate(out_idx):
                    arg = option_copy['arguments'][idx]
                    arg['output_idx'] = i
                    if not output_provided:
                        arg['ignore_check'] = True
                    else:
                        option_copy['argcount_offset'] = -len(out_idx) + 1
                        arg['no_kwargs'] = True
                        arg['no_idx'] = True
                new_options.append(option_copy)
        declaration['options'] = new_options

    def process_declarations(self, declarations):
        new_declarations = []

        def has_arg_type(declaration, type_name):
            return any(arg['type'] == type_name
                       for option in declaration['options']
                       for arg in option['arguments'])

        def has_long_args(declaration):
            return any(arg.get('long_args', False)
                       for option in declaration['options']
                       for arg in option['arguments'])

        def has_output_args(declaration):
            return any(arg.get('output')
                       for option in declaration['options']
                       for arg in option['arguments'])

        def backends_types_to_defined_if_string(declaration):
            # A declaration has two fields: 'backend', which stores a list of
            # backends (currently 'cpu' and 'cuda') the declaration applies
            # to, and 'types', which stores a list of real types the
            # declaration applies to. In PyTorch, when a function is only
            # supported by a subset of types, we wrap it in macro definition
            # checks.
            #
            # Previously, we manually required the cwrap declaration to
            # specify for which backend/type combinations a function was
            # defined for. Now, we explicitly list the types and backends for
            # a declaration, if it should only be supported for a specific
            # subset of types, backends, or type-backend pairs.

            types = declaration.get('types', [])
            backends = declaration['backends']
            all_backends = ['CPU', 'CUDA']

            def get_defined_string(backend, real):
                if backend == 'CUDA':
                    if real == 'all':
                        return "IS_CUDA"
                    else:
                        return 'CUDA_{0}'.format(real.upper())
                else:
                    if real == 'all':
                        return "!IS_CUDA"
                    else:
                        return 'defined(TH_REAL_IS_{0})'.format(real.upper())

            def expand_composite_type(p, t):
                if t == 'floating_point':
                    result = ['double', 'float']
                    if p == 'CUDA':
                        result.append('half')
                elif t == 'integral':
                    result = ['byte', 'char', 'short', 'int', 'long']
                else:
                    result = [t]
                return result

            defineds = []

            # The logic below does not handle corner cases well. We allow the
            # declaration to have a field 'backend_type_pairs' that stores a
            # dictionary from type --> backend representing allowed
            # combinations. Let's use these first.
            for pair in declaration.get('backend_type_pairs', []):
                p, t = pair
                defineds.extend([get_defined_string(p, et) for et in
                                expand_composite_type(p, t)])

            # In the base case, types is empty and backends contains both
            # 'CPU' and 'CUDA' --> this means we support all types, and our
            # string should be empty, or simply the list of explict type
            # backend pairs
            if (len(types) == 0 and all([proc in backends for proc in
                                         all_backends])):
                return " || ".join(defineds)

            # Case 2: types is empty, but only one backend type is specified
            if len(types) == 0 and len(backends) == 1:
                defineds.append('IS_CUDA' if backends[0] == 'CUDA' else
                                "!IS_CUDA")
                return " || ".join(defineds)

            # Else, we loop overall all of the backend, type pairs and add
            # them
            for p in backends:
                for t in types:
                    defineds.extend([get_defined_string(p, et) for et in
                                    expand_composite_type(p, t)])

            return " || ".join(defineds)

        for declaration in declarations:
            # Disable all methods for THHalfTensor, unless cpu_half is True

            dfstr = backends_types_to_defined_if_string(declaration)
            if len(dfstr) > 0:
                # for now, need to check for distributed defined if as well
                if 'defined_if' in declaration:
                    declaration['defined_if'] += ' && (' + dfstr + ')'
                else:
                    declaration['defined_if'] = dfstr

            if not declaration.get('cpu_half', False):
                defined_if = '!defined(TH_REAL_IS_HALF)'
                if 'defined_if' in declaration:
                    defined_if += ' && (' + declaration['defined_if'] + ')'
                declaration['defined_if'] = defined_if

            if declaration.get('only_register', False):
                continue

            declaration.setdefault('python_name', declaration['name'])
            declaration.setdefault('variables', [])
            if has_arg_type(declaration, 'THSize*'):
                declaration['variables'] += ['THLongStoragePtr __size;']
            if has_arg_type(declaration, 'THStride*'):
                declaration['variables'] += ['THLongStoragePtr __stride;']
            if has_output_args(declaration):
                declaration['variables'] += ['PyObject *___out;']
                self.generate_out_options(declaration)
            if has_long_args(declaration):
                for option in declaration['options']:
                    for arg in option['arguments']:
                        if arg.get('long_args', False):
                            arg['no_kwargs'] = True
            for option in declaration['options']:
                option['cname'] = 'TH{}Tensor_({})'.format(
                    'S' if option.get('sparse', False) else '', option['cname'])
                if option.get('sparse', False):
                    defined_if = option.get('defined_if', '')
                    option['defined_if'] = '!IS_DISTRIBUTED' + (' && ' if defined_if else '') + defined_if

            variants = declaration.get('variants', ['method'])
            if 'function' in variants:
                stateless_declaration = self.make_stateless(declaration)
                new_declarations.append(stateless_declaration)
                self.stateless_declarations.append(stateless_declaration)
            if 'method' not in variants:
                continue

            self.declarations.append(declaration)
            declaration['name'] = 'TH{}PTensor_({})'.format(
                'S' if declaration.get('sparse', False) else '', declaration['name'])
            for option in declaration['options']:
                for arg in option['arguments']:
                    if arg['name'] == 'self':
                        arg['ignore_check'] = True

        register_only = [d for d in declarations if d.get('only_register', False)]
        declarations = [d for d in declarations
                        if (('method' in d.get('variants', ['method'])) and
                            (not d.get('only_register', False)))]
        self.declarations.extend(filter(lambda x: 'method' in x.get('variants',
                                 ['method']), register_only))
        self.stateless_declarations.extend(filter(lambda x: 'method' not in
                                           x.get('variants', ['method']),
                                           register_only))

        self.process_docstrings()

        all_declarations = declarations + new_declarations
        return all_declarations

    def make_stateless(self, declaration):
        declaration = deepcopy(declaration)
        declaration['name'] = 'TH{}PTensor_stateless_({})'.format(
            'S' if declaration.get('sparse', False) else '', declaration['name'])
        for option in declaration['options']:
            for arg in option['arguments']:
                if arg['name'] == 'self':
                    arg['assign_name'] = 'self'
                    arg['name'] = 'source'
        return declaration

    def declare_methods(self, stateless, sparse):
        tensor_methods = ''
        for declaration in (self.declarations if not stateless else self.stateless_declarations):
            if declaration.get('sparse', False) != sparse:
                continue
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
        generated = self.TENSOR_METHODS_DECLARATION.substitute(
            methods=tensor_methods,
            stateless=('' if not stateless else 'stateless_'),
            sparse=('' if not sparse else 'S'),
        )
        if sparse:
            generated = '#if !defined(TH_REAL_IS_HALF) && !IS_DISTRIBUTED\n' + generated + '\n#endif\n\n'
        return generated

    def process_full_file(self, code, template_path):
        # We have to find a place before all undefs
        idx = code.find('// PUT DEFINITIONS IN HERE PLEASE')
        return (code[:idx] +
                self.declare_methods(False, False) +
                self.declare_methods(True, False) +
                self.declare_methods(False, True) +
                self.declare_methods(True, True) +
                code[idx:]
                )

    def preprocessor_guard(self, code, condition):
        return '#if ' + condition + '\n' + code + '#endif\n'

    def process_wrapper(self, code, declaration):
        if 'defined_if' in declaration:
            return self.preprocessor_guard(code, declaration['defined_if'])
        return code

    def process_all_call_arg(self, code, option):
        return 'LIBRARY_STATE ' + code

    def process_all_checks(self, code, option):
        if option.get('has_output'):
            indent = " " * 10
            if option['output_provided']:
                checks = "___out != NULL &&\n" + indent
                if option['output_count'] > 1:
                    checks += "PyTuple_Check(___out) &&\n" + indent
                    length_check = "PyTuple_GET_SIZE(___out) == {} &&\n".format(
                        option['output_count'])
                    checks += length_check + indent
                code = checks + code
            else:
                code = "___out == NULL &&\n" + indent + code

        if any(arg.get('long_args', False) for arg in option['arguments']):
            code = code.replace('__argcount ==', '__argcount >=')
            expected = str(int(option.get('output_provided', False)) +
                           sum(not arg.get('no_kwargs', False) and not arg.get('ignore_check', False)
                               for arg in option['arguments']))
            code = '__dictcount == ' + expected + ' &&\n          ' + code

        return code

    def process_option_code(self, code, option):
        if option.get('defined_if', ''):
            defined_if = option['defined_if']
            placeholder = ''
            # This means that it's a first option, so we need a dummy if,
            # so the next option can be an else if.
            if 'else if' not in code:
                placeholder = '\n    #else\n    if (false) {'
            return '#if ' + defined_if + '\n          ' + code + placeholder + '\n    #endif\n'
        return code

    def process_pre_arg_assign(self, template, option):
        new_args = []
        for arg in option['arguments']:
            if not option.get('output_provided', True) and arg.get('output'):
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
