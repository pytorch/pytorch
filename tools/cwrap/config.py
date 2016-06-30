from string import Template
from .argument import Argument
import re

ARGUMENT_PREFIX = '  -'
OPTION_REGEX = re.compile('^\s*([a-zA-z0-9]+) -> (new [a-zA-Z]+|[a-zA-Z]+)(.*)')
FUNCTION_NAME_REGEX = re.compile('^\s*([a-zA-Z0-9]+)(.*)')
OPTIONAL_ARGUMENT_REGEX = re.compile('.* OPTIONAL (.*)$')

# Transforms applied to argument types declared in the definition
# these are mostly, so that the * can be omitted for convenience and clarity
TYPE_TRANSFORMS = {
    'THTensor': 'THPTensor*',
    'THStorage': 'THPStorage*',
    'THByteTensor': 'THPByteTensor*',
    'THLongTensor': 'THPLongTensor*',
    'THFloatTensor': 'THPFloatTensor*',
    'THDoubleTensor': 'THPDoubleTensor*',
    'THLongStorage': 'THPLongStorage*',
    'THGenerator': 'THPGenerator*',
    # TODO
    'accreal': 'double',
}


# Used to build format string for PyArg_ParseTuple
FORMAT_STR_MAP = {
    'THPTensor*': 'O!',
    'THPLongTensor*': 'O!',
    'THPByteTensor*': 'O!',
    'THPFloatTensor*': 'O!',
    'THPDoubleTensor*': 'O!',
    'THPLongStorage*': 'O!',
    'THPStorage*': 'O!',
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
    'THPStorage*': 'THPStorageType',
    'THPGenerator*': 'THPGeneratorType',
    'real': 'THPUtils_(parseReal)',
}

TYPE_CHECK = {
    'THPTensor*':       lambda arg: 'THPTensor_(IsSubclass)((PyObject*){})'.format(arg),
    'THPLongTensor*':   lambda arg: 'THPLongTensor_IsSubclass((PyObject*){})'.format(arg),
    'THPGenerator*':    lambda arg: 'THPGenerator_Check({})'.format(arg),
    'THPStorage*':      lambda arg: 'THPStorage_(IsSubclass)((PyObject*){})'.format(arg),
    'real':             lambda arg: 'THPUtils_(checkReal)({})'.format(arg),
    'long':             lambda arg: 'THPUtils_checkLong({})'.format(arg),
    'double':           lambda arg: 'PyFloat_Check({})'.format(arg),
    'bool':             lambda arg: 'PyBool_Check({})'.format(arg),
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
        THTensorPtr _value = THTensor_(new)(LIBRARY_STATE_NOARGS);
        THPTensorPtr _v = (THPTensor*)THPTensor_(newObject)(_value);
        THLongTensorPtr _indices = THLongTensor_new();
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
    'new ValueValuePair':   Template("""
        THTensorPtr _value = THTensor_(new)(LIBRARY_STATE_NOARGS);
        THPTensorPtr _v = (THPTensor*)THPTensor_(newObject)(_value);
        THTensorPtr _indices = THTensor_(new)(LIBRARY_STATE_NOARGS);
        THPTensorPtr _i = (THPTensor*)THPTensor_(newObject)(_indices);
        _value.release();
        _indices.release();
        $expr;
        PyObject *ret = Py_BuildValue("NN", (PyObject*)_v.get(), (PyObject*)_i.get());
        _v.release(); _i.release();
        return ret;"""),
    'new SelfValuePair':    Template("""
        THTensorPtr _indices = THTensor_(new)(LIBRARY_STATE_NOARGS);
        THPTensorPtr _i = (THPTensor*)THPTensor_(newObject)(_indices);
        _indices.release();
        $expr;
        PyObject *ret = Py_BuildValue("ON", (PyObject*)self, (PyObject*)_i.get());
        _i.release();
        return ret"""),
    'new THTensor':         Template("""
        THTensorPtr _value = THTensor_(new)(LIBRARY_STATE_NOARGS);
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
    'STATELESS PROV new SelfValuePair': Template("""
        THTensorPtr _indices = THTensor_(new)(LIBRARY_STATE_NOARGS);
        THPTensorPtr _i = (THPTensor*)THPTensor_(newObject)(_indices);
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
        THTensorPtr _t = THTensor_(new)(LIBRARY_STATE_NOARGS);
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
    'new ValueValuePair': (Argument('THPTensor*', '_v'), Argument('THPTensor*', '_i')),
    'new SelfValuePair': (Argument('THPTensor*', 'self'), Argument('THPTensor*', '_i')),
    'STATELESS PROV new SelfIndexPair': {1: Argument('THPTensor*', '_i')},
    'STATELESS PROV new SelfValuePair': {1: Argument('THPTensor*', '_i')},
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
