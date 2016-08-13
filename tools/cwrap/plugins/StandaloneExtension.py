import os
from string import Template
from . import CWrapPlugin


with open(os.path.join(os.path.dirname(__file__), 'templates', 'module_head.cpp'), 'r') as f:
    MODULE_HEAD = Template(f.read())
with open(os.path.join(os.path.dirname(__file__), 'templates', 'module_tail.cpp'), 'r') as f:
    MODULE_TAIL = Template(f.read())

REGISTER_METHOD_TEMPLATE = Template('  {"$name", (PyCFunction)$name, METH_VARARGS, NULL},\n')

MODULE_METHODS_TEMPLATE = Template("""
static PyMethodDef module_methods[] = {
$METHODS
  {NULL, NULL, 0, NULL}
};
""")


class StandaloneExtension(CWrapPlugin):

    TYPE_UNPACK = {
        'THFloatTensor*':   Template('(THFloatTensor*)(((Tensor*)$arg)->cdata)'),
        'THDoubleTensor*':  Template('(THDoubleTensor*)(((Tensor*)$arg)->cdata)'),
        'THLongTensor*':    Template('(THLongTensor*)(((Tensor*)$arg)->cdata)'),
        'THIntTensor*':     Template('(THIntTensor*)(((Tensor*)$arg)->cdata)'),
        'THCudaTensor*':    Template('(THCudaTensor*)(((Tensor*)$arg)->cdata)'),
        'THCudaLongTensor*': Template('(THCudaLongTensor*)(((Tensor*)$arg)->cdata)'),
        'float':            Template('__getFloat($arg)'),
        'double':           Template('__getFloat($arg)'),
        'bool':             Template('__getLong($arg)'),
        'int':              Template('__getLong($arg)'),
        'long':             Template('__getLong($arg)'),
        'void*':            Template('(void*)__getLong($arg)'),
        # TODO: implement this
        'THGenerator*':     Template('NULL'),
    }

    TYPE_CHECK = {
        'THDoubleTensor*':  Template('(PyObject*)Py_TYPE($arg) == THPDoubleTensorClass'),
        'THFloatTensor*':   Template('(PyObject*)Py_TYPE($arg) == THPFloatTensorClass'),
        'THLongTensor*':    Template('(PyObject*)Py_TYPE($arg) == THPLongTensorClass'),
        'THIntTensor*':     Template('(PyObject*)Py_TYPE($arg) == THPIntTensorClass'),
        'THCudaTensor*':    Template('(PyObject*)Py_TYPE($arg) == THCPFloatTensorClass'),
        'THCudaLongTensor*': Template('(PyObject*)Py_TYPE($arg) == THCPLongTensorClass'),
        'float':            Template('__checkFloat($arg)'),
        'double':           Template('__checkFloat($arg)'),
        'bool':             Template('__checkLong($arg)'),
        'int':              Template('__checkLong($arg)'),
        'long':             Template('__checkLong($arg)'),
        'void*':            Template('__checkLong($arg)'),
        # TODO: implement this
        'THGenerator*':     Template('false'),
    }

    WRAPPER_TEMPLATE = Template("""
PyObject * $name(PyObject *_unused, PyObject *args)
{
  int __argcount = args ? PyTuple_Size(args) : 0;
  try {
      $options
    } else {
      __invalidArgs(args, "");
      return NULL;
    }
  } catch (std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return NULL;
  }
}
    """)

    def __init__(self, module_name, with_cuda=False):
        self.module_name = module_name
        self.with_cuda = with_cuda
        self.declarations = []

    def process_full_file(self, code):
        short_name = self.module_name.split('.')[-1]
        new_code = MODULE_HEAD.substitute(requres_cuda=('1' if self.with_cuda else '0'))
        new_code += code
        new_code += self.declare_module_methods()
        new_code += MODULE_TAIL.substitute(full_name=self.module_name, short_name=short_name)
        return new_code

    def process_wrapper(self, code, declaration):
        self.declarations.append(declaration)
        return code

    def declare_module_methods(self):
        module_methods = ''
        for declaration in self.declarations:
            module_methods += REGISTER_METHOD_TEMPLATE.substitute(name=declaration['name'])
        return MODULE_METHODS_TEMPLATE.substitute(METHODS=module_methods)

    def get_type_unpack(self, arg, option):
        return self.TYPE_UNPACK.get(arg['type'], None)

    def get_type_check(self, arg, option):
        return self.TYPE_CHECK.get(arg['type'], None)

    def get_wrapper_template(self, declaration):
        return self.WRAPPER_TEMPLATE
