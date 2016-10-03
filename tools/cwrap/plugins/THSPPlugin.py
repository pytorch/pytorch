from string import Template
from copy import deepcopy
from .THPPlugin import THPPlugin
from itertools import product, chain

class THSPPlugin(THPPlugin):

    TYPE_UNPACK = dict(chain(THPPlugin.TYPE_UNPACK.items(), {
        'THSFloatTensor*':   Template('((THSPFloatTensor*)$arg)->cdata'),
        'THSDoubleTensor*':  Template('((THSPDoubleTensor*)$arg)->cdata'),
        'THSLongTensor*':    Template('((THSPLongTensor*)$arg)->cdata'),
        'THSIntTensor*':     Template('((THSPIntTensor*)$arg)->cdata'),
        'THSTensor*':        Template('((THSPTensor*)$arg)->cdata'),
        'THSBoolTensor*':    Template('((THSPBoolTensor*)$arg)->cdata'),
        'THSIndexTensor*':   Template('((THSPIndexTensor*)$arg)->cdata'),
        'THSLongStorage*':   Template('((THSPLongStorage*)$arg)->cdata'),
    }.items()))

    TYPE_CHECK = dict(chain(THPPlugin.TYPE_CHECK.items(), {
        'THSDoubleTensor*':  Template('(PyObject*)Py_TYPE($arg) == THSPDoubleTensorClass'),
        'THSFloatTensor*':   Template('(PyObject*)Py_TYPE($arg) == THSPFloatTensorClass'),
        'THSLongTensor*':    Template('(PyObject*)Py_TYPE($arg) == THSPLongTensorClass'),
        'THSIntTensor*':     Template('(PyObject*)Py_TYPE($arg) == THSPIntTensorClass'),
        'THSTensor*':        Template('(PyObject*)Py_TYPE($arg) == THSPTensorClass'),
        'THSBoolTensor*':    Template('(PyObject*)Py_TYPE($arg) == THSPBoolTensorClass'),
        'THSIndexTensor*':   Template('(PyObject*)Py_TYPE($arg) == THSPIndexTensorClass'),
        'THSLongStorage*':   Template('(PyObject*)Py_TYPE($arg) == THSPLongStorageClass'),
    }.items()))

    RETURN_WRAPPER = dict(chain(THPPlugin.RETURN_WRAPPER.items(), {
        'THSTensor*':        Template('return THSPTensor_(New)($result);'),
        'THLongTensor*':        Template('return THPLongTensor_New($result);'),
    }.items()))

    TENSOR_METHODS_DECLARATION = Template("""
static PyMethodDef THSPTensor_$stateless(methods)[] = {
$methods
  {NULL}
};
""")

    ALLOCATE_TYPE = dict(chain(THPPlugin.ALLOCATE_TYPE.items(), {
        'THSTensor*':        Template("""\
      THSTensorPtr _th_$name = THSTensor_(new)(LIBRARY_STATE_NOARGS);
      THSPTensorPtr _${name}_guard = (THSPTensor*)THSPTensor_(New)(_th_$name.get());
      THSPTensor* $name = _${name}_guard.get();
      if (!$name)
        return NULL;
      _th_$name.release();
"""),
    }.items()))

    TYPE_NAMES = dict(chain(THPPlugin.TYPE_NAMES.items(), {
        'THSTensor*': '" THSPTensorStr "',
        'THSLongTensor*': 'SparseLongTensor',
        'THSBoolTensor*': 'SparseByteTensor',
        'THSIndexTensor*': 'SparseLongTensor',
        'THSFloatTensor*': 'SparseFloatTensor',
        'THSDoubleTensor*': 'SparseDoubleTensor',
    }.items()))

    # These functions have TH hardcoded in, so we just copy them
    def process_declarations(self, declarations):
        new_declarations = []
        register_only = [d for d in declarations if d.get('only_register', False)]
        declarations = [d for d in declarations if not d.get('only_register', False)]
        for declaration in declarations:
            if declaration.get('only_register', False):
                continue
            declaration.setdefault('python_name', declaration['name'])
            if declaration.get('with_stateless', False) or declaration.get('only_stateless', False):
                stateless_declaration = self.make_stateless(deepcopy(declaration))
                new_declarations.append(stateless_declaration)
                self.stateless_declarations.append(stateless_declaration)
            if declaration.get('only_stateless', False):
                continue

            self.declarations.append(declaration)
            declaration['name'] = 'THSPTensor_({})'.format(declaration['name'])
            for option in declaration['options']:
                option['cname'] = 'THSTensor_({})'.format(option['cname'])
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
        return declarations + new_declarations

    def make_stateless(self, declaration):
        declaration['name'] = 'THSPTensor_stateless_({})'.format(declaration['name'])
        new_options = []
        for option in declaration['options']:
            option['cname'] = 'THSTensor_({})'.format(option['cname'])
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

