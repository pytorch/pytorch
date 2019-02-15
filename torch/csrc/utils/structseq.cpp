#include "torch/csrc/utils/structseq.h"
#include "structmember.h"
#include <sstream>

namespace torch {
namespace utils {

#if PY_MAJOR_VERSION == 2
static PyObject*
structseq_slice(PyStructSequence *obj, Py_ssize_t low, Py_ssize_t high)
{
    PyTupleObject *np;
    Py_ssize_t i;

    if (low < 0)
        low = 0;
    if (high > Py_SIZE(obj))
        high = Py_SIZE(obj);
    if (high < low)
        high = low;
    np = (PyTupleObject *)PyTuple_New(high-low);
    if (np == NULL)
        return NULL;
    for(i = low; i < high; ++i) {
        PyObject *v = obj->ob_item[i];
        Py_INCREF(v);
        PyTuple_SET_ITEM(np, i-low, v);
    }
    return (PyObject *) np;
}

static PyObject *make_tuple(PyStructSequence *obj) {
    return structseq_slice(obj, 0, Py_SIZE(obj));
}
#endif

PyObject *returned_structseq_repr(PyStructSequence *obj) {
    PyTypeObject *typ = Py_TYPE(obj);
    std::stringstream ss;
    ss << typ->tp_name << "(\n";
    size_t num_elements = Py_SIZE(obj);
#if PY_MAJOR_VERSION == 2
    PyObject *tup;
    if ((tup = make_tuple(obj)) == NULL) {
        return NULL;
    }
#endif
    for (int i=0; i < num_elements; i++) {
        PyObject *val, *repr;
        const char *cname, *crepr;

        cname = typ->tp_members[i].name;
        if (cname == NULL) {
            PyErr_Format(PyExc_SystemError, "In structseq_repr(), member %d name is NULL"
                         " for type %.500s", i, typ->tp_name);
            return NULL;
        }
#if PY_MAJOR_VERSION == 2
        val = PyTuple_GetItem(tup, i);
        if (val == NULL) {
            return NULL;
        }
#else
        val = PyStructSequence_GET_ITEM(obj, i);
#endif
        repr = PyObject_Repr(val);
        if (repr == NULL)
            return NULL;
#if PY_MAJOR_VERSION == 2
        crepr = PyString_AsString(repr);
#else
        crepr = PyUnicode_AsUTF8(repr);
#endif
        if (crepr == NULL) {
#if PY_MAJOR_VERSION == 2
            Py_DECREF(tup);
#endif
            Py_DECREF(repr);
            return NULL;
        }

        ss << cname << '=' << crepr;
        if (i < num_elements - 1)
            ss << ",\n";
    }
    ss << ")";

    return PyUnicode_FromString(ss.str().c_str());
}

}
}
