from string import Template
from copy import deepcopy
from . import CWrapPlugin
from itertools import product

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
        'void*':            Template('THPUtils_unpackLong($arg)'),
        'long':             Template('THPUtils_unpackLong($arg)'),
        'int':              Template('THPUtils_unpackLong($arg)'),
        'bool':             Template('THPUtils_unpackLong($arg)'),
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
        'THGenerator*':     Template('Py_TYPE($arg) == &THPGeneratorType'),
        'void*':            Template('THPUtils_checkLong($arg)'),
        'long':             Template('THPUtils_checkLong($arg)'),
        'int':              Template('THPUtils_checkLong($arg)'),
        'bool':             Template('THPUtils_checkLong($arg)'),
        'float':            Template('THPFloatUtils_checkReal($arg)'),
        'double':           Template('THPDoubleUtils_checkReal($arg)'),
        'real':             Template('THPUtils_(checkReal)($arg)'),
        # TODO
        'accreal':          Template('THPUtils_(checkReal)($arg)'),
    }

    RETURN_WRAPPER = {
        'THTensor*':        Template('return THPTensor_(newObject)($call);'),
        'THLongStorage*':   Template('return THPLongStorage_newObject($call);'),
        # TODO: make it smarter - it should return python long if result doesn't fit into an int
        'long':             Template('return PyInt_FromLong($call);'),
        # TODO
        'accreal':          Template('return PyFloat_FromDouble($call);'),
        'self':             Template('$call;\nPy_INCREF(self);\nreturn (PyObject*)self;'),
        'real':             Template('return THPUtils_(newReal)($call);'),
    }

    TENSOR_METHODS_DECLARATION = Template("""
static PyMethodDef THPTensor_$stateless(methods)[] = {
$methods
  {NULL}
};
""")

    WRAPPER_TEMPLATE = Template("""\
PyObject * $name(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    $options
    } else {
      THPUtils_invalidArguments(args, $expected_args);
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
""")

    ALLOCATE_TYPE = {
        'THTensor*':        Template("""\
      THTensorPtr _th_$name = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _${name}_guard = (THPTensor*)THPTensor_(newObject)(_th_$name.get());
      THPTensor* $name = _${name}_guard.get();
      _th_$name.release();
"""),
        'THLongTensor*':        Template("""\
      THLongTensorPtr _th_$name = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _${name}_guard = (THPLongTensor*)THPLongTensor_newObject(_th_$name.get());
      THPLongTensor* $name = _${name}_guard.get();
      _th_$name.release();
"""),
        'THBoolTensor*':    Template("""
#if IS_CUDA
      THCByteTensorPtr _t_$name = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _${name}_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_$name);
      THCPByteTensor *$name = _${name}_guard.get();
#else
      THByteTensorPtr _t_$name = THByteTensor_new();
      THPByteTensorPtr _${name}_guard = (THPByteTensor*)THPByteTensor_newObject(_t_$name);
      THPByteTensor *$name = _${name}_guard.get();
#endif
      _t_$name.release();
"""),
        'THIndexTensor*':    Template("""
#if IS_CUDA
      THCLongTensorPtr _t_$name = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _${name}_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_$name);
      THCPLongTensor *$name = _${name}_guard.get();
#else
      THLongTensorPtr _t_$name = THLongTensor_new();
      THPLongTensorPtr _${name}_guard = (THPLongTensor*)THPLongTensor_newObject(_t_$name);
      THPLongTensor *$name = _${name}_guard.get();
#endif
      _t_$name.release();
"""),
    }

    RELEASE_ARG = Template("_${name}_guard.release();")

    def __init__(self):
        self.declarations = []
        self.stateless_declarations = []

    def get_type_unpack(self, arg, option):
        return self.TYPE_UNPACK.get(arg['type'], None)

    def get_type_check(self, arg, option):
        return self.TYPE_CHECK.get(arg['type'], None)

    # TODO: argument descriptions shouldn't be part of THP, but rather a general cwrap thing
    def get_wrapper_template(self, declaration):
        arg_desc = []
        for option in declaration['options']:
            option_desc = [arg['type'] + ' ' + arg['name'] for arg in option['arguments'] if not arg.get('ignore_check', False)]
            if option_desc:
                arg_desc.append('({})'.format(', '.join(option_desc)))
            else:
                arg_desc.append('no arguments')
        arg_str = '"' + ' or '.join(arg_desc) + '"'
        return Template(self.WRAPPER_TEMPLATE.safe_substitute(expected_args=arg_str))

    def get_return_wrapper(self, option):
        return self.RETURN_WRAPPER.get(option['return'], None)

    def get_arg_accessor(self, arg, option):
        if arg['name'] == 'self':
            return 'self'
        if 'allocate' in arg and arg['allocate']:
            return arg['name']

    def process_declarations(self, declarations):
        new_declarations = []
        register_only = [d for d in declarations if d.get('only_register', False)]
        declarations = [d for d in declarations if not d.get('only_register', False)]
        for declaration in declarations:
            if declaration.get('only_register', False):
                continue
            declaration['python_name'] = declaration['name']
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
        declarations = [d for d in declarations if not d.get('only_stateless', False)]
        self.declarations.extend(filter(lambda x: not x.get('only_stateless', False), register_only))
        self.stateless_declarations.extend(filter(lambda x: x.get('only_stateless', False), register_only))
        return declarations + new_declarations

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
            entry = Template('  {"$python_name", (PyCFunction)$name, METH_VARARGS, NULL},\n').substitute(
                    python_name=declaration['python_name'], name=declaration['name']
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

    def process_call(self, code, option):
        new_args = []
        for arg in option['arguments']:
            if 'allocate' in arg and arg['allocate']:
                new_args.append(self.ALLOCATE_TYPE[arg['type']].substitute(name=arg['name']))
        return '\n      '.join(new_args) + '\n' + code
