from string import Template
from copy import deepcopy
from . import CWrapPlugin
from itertools import product


class CuDNNPlugin(CWrapPlugin):

    TYPE_UNPACK = {
        'THTensor*': Template('((THPVoidTensor*)$arg)->cdata'),
        'int': Template('THPUtils_unpackLong($arg)'),
        'std::vector<int>': Template('THPUtils_unpackIntTuple($arg)'),
        'cudnnDataType_t': Template('$arg'),
        'cudnnHandle_t': Template('$arg'),
        'Convolution*': Template('(Convolution*)THPWrapper_get($arg)'),
        'bool': Template('$arg == Py_True'),
        'double': Template('THPDoubleUtils_unpackReal($arg)'),
    }

    TYPE_CHECK = {
        'Convolution*': Template('THPWrapper_check($arg)'),
        'THTensor*': Template('(PyObject*)Py_TYPE($arg) == tensorClass'),
        'int': Template('THPUtils_checkLong($arg)'),
        'std::vector<int>': Template('THPUtils_checkIntTuple($arg)'),
        'bool': Template('PyBool_Check($arg)'),
        'double': Template('THPDoubleUtils_checkReal($arg)'),
    }

    RETURN_WRAPPER = {
        'Convolution*': Template('return THPWrapper_New($result, [](void* arg) { delete (Convolution*)arg; });'),
    }

    METHODS_DECLARATION = Template("""
static PyMethodDef _THCUDNN_methods[] = {
$methods
  {NULL}
};

PyMethodDef* THCUDNN_methods()
{
  return _THCUDNN_methods;
}
""")

    WRAPPER_TEMPLATE = Template("""\
static PyObject * $name(PyObject *self, PyObject *args, PyObject *kwargs)
{
    HANDLE_TH_ERRORS
    int __tuplecount = args ? PyTuple_Size(args) : 0;
    int __dictcount = kwargs ? PyDict_Size(kwargs) : 0;
    int __argcount = __tuplecount + __dictcount;
    PyObject* tensorClass = getTensorClass(args);
    THCPAutoGPU __autogpu_guard = THCPAutoGPU(args);

    $options
    }

    THPUtils_invalidArguments(args, kwargs, "$readable_name", $num_options, $expected_args);
    return NULL;
    END_HANDLE_TH_ERRORS
}
""")

    RELEASE_ARG = Template("_${name}_guard.release();")

    TYPE_NAMES = {
        'THTensor*': '" THPTensorStr "',
        'long': 'int',
        'bool': 'bool',
        'int': 'int',
    }

    def __init__(self):
        self.declarations = []

    def get_type_unpack(self, arg, option):
        return self.TYPE_UNPACK.get(arg['type'], None)

    def get_type_check(self, arg, option):
        return self.TYPE_CHECK.get(arg['type'], None)

    def get_wrapper_template(self, declaration):
        arg_desc = []
        for option in declaration['options']:
            option_desc = [self.TYPE_NAMES.get(arg['type'], arg['type']) + ' ' + arg['name']
                           for arg in option['arguments']
                           if not arg.get('ignore_check', False)]
            # TODO: this should probably go to THPLongArgsPlugin
            if option_desc:
                arg_desc.append('({})'.format(', '.join(option_desc)))
            else:
                arg_desc.append('no arguments')
        arg_desc.sort(key=len)
        arg_desc = ['"' + desc + '"' for desc in arg_desc]
        arg_str = ', '.join(arg_desc)
        readable_name = declaration['python_name']
        return Template(self.WRAPPER_TEMPLATE.safe_substitute(
            readable_name=readable_name, num_options=len(arg_desc),
            expected_args=arg_str))

    def get_return_wrapper(self, option):
        return self.RETURN_WRAPPER.get(option['return'], None)

    def get_arg_accessor(self, arg, option):
        name = arg['name']
        if name == 'self':
            return 'self'
        elif name == 'dataType':
            return 'getCudnnDataType(tensorClass)'
        elif name == 'handle':
            return 'getCudnnHandle()'

    def process_declarations(self, declarations):
        for declaration in declarations:
            declaration.setdefault('python_name', '_{}'.format(declaration['name']))
            declaration['name'] = 'THCUDNN_{}'.format(declaration['name'])
            self.declarations.append(declaration)
            for option in declaration['options']:
                for arg in option['arguments']:
                    if arg['name'] in ['self', 'state', 'dataType', 'handle']:
                        arg['ignore_check'] = True
            declaration['options'] = self.filter_unique_options(declaration['options'])
        return declarations

    def filter_unique_options(self, options):
        def signature(option):
            return '#'.join(arg['type'] for arg in option['arguments']
                            if 'ignore_check' not in arg or not arg['ignore_check'])
        seen_signatures = set()
        unique = []
        for option in options:
            sig = signature(option)
            if sig not in seen_signatures:
                unique.append(option)
                seen_signatures.add(sig)
        return unique

    def preprocessor_guard(self, code, condition):
        return '#if ' + condition + '\n' + code + '#endif\n'

    def process_wrapper(self, code, declaration):
        if 'defined_if' in declaration:
            return self.preprocessor_guard(code, declaration['defined_if'])
        return code

    def process_all_unpacks(self, code, option):
        return 'state, ' + code

    def declare_methods(self):
        methods = ''
        for declaration in self.declarations:
            extra_flags = ' | ' + declaration.get('method_flags') if 'method_flags' in declaration else ''
            if not declaration.get('only_register'):
                extra_flags += ' | METH_KEYWORDS'
            entry = Template('  {"$python_name", (PyCFunction)$name, METH_VARARGS$extra_flags, NULL},\n').substitute(
                python_name=declaration['python_name'], name=declaration['name'], extra_flags=extra_flags
            )
            if 'defined_if' in declaration:
                entry = self.preprocessor_guard(entry, declaration['defined_if'])
            methods += entry
        return self.METHODS_DECLARATION.substitute(methods=methods)

    def process_full_file(self, code):
        return code + self.declare_methods()
