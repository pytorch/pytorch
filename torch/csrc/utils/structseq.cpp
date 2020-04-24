/* Copyright Python Software Foundation
 *
 * This file is copy-pasted from CPython source code with modifications:
 * https://github.com/python/cpython/blob/master/Objects/structseq.c
 * https://github.com/python/cpython/blob/2.7/Objects/structseq.c
 *
 * The purpose of this file is to overwrite the default behavior
 * of repr of structseq to provide better printting for returned
 * structseq objects from operators, aka torch.return_types.*
 *
 * For more information on copyright of CPython, see:
 * https://github.com/python/cpython#copyright-and-license-information
 */

#include <torch/csrc/utils/structseq.h>
#include <torch/csrc/utils/six.h>
#include <structmember.h>
#include <sstream>

namespace torch {
namespace utils {

#if PY_MAJOR_VERSION == 2
PyObject *structseq_slice(PyStructSequence *obj, Py_ssize_t low, Py_ssize_t high)
{
    PyTupleObject *np;
    Py_ssize_t i;

    if (low < 0) {
        low = 0;
    }
    if (high > Py_SIZE(obj)) {
        high = Py_SIZE(obj);
    }
    if (high < low) {
        high = low;
    }
    np = (PyTupleObject *)PyTuple_New(high-low);
    if (np == nullptr) {
        return nullptr;
    }
    for(i = low; i < high; ++i) {
        PyObject *v = obj->ob_item[i];
        Py_INCREF(v);
        PyTuple_SET_ITEM(np, i-low, v);
    }
    return (PyObject *) np;
}

#define PyUnicode_AsUTF8 PyString_AsString
#endif

PyObject *returned_structseq_repr(PyStructSequence *obj) {
    PyTypeObject *typ = Py_TYPE(obj);
    THPObjectPtr tup = six::maybeAsTuple(obj);
    if (tup == nullptr) {
        return nullptr;
    }

    std::stringstream ss;
    ss << typ->tp_name << "(\n";
    Py_ssize_t num_elements = Py_SIZE(obj);

    for (Py_ssize_t i = 0; i < num_elements; i++) {
        const char *cname = typ->tp_members[i].name;
        if (cname == nullptr) {
            PyErr_Format(PyExc_SystemError, "In structseq_repr(), member %zd name is nullptr"
                         " for type %.500s", i, typ->tp_name);
            return nullptr;
        }

        PyObject* val = PyTuple_GetItem(tup.get(), i);
        if (val == nullptr) {
            return nullptr;
        }

        auto repr = THPObjectPtr(PyObject_Repr(val));
        if (repr == nullptr) {
            return nullptr;
        }

        const char* crepr = PyUnicode_AsUTF8(repr);
        if (crepr == nullptr) {
            return nullptr;
        }

        ss << cname << '=' << crepr;
        if (i < num_elements - 1) {
            ss << ",\n";
        }
    }
    ss << ")";

    return PyUnicode_FromString(ss.str().c_str());
}

}
}
