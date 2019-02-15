#include "torch/csrc/utils/structseq.h"
#include "structmember.h"
#include <sstream>

namespace torch {
namespace utils {

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
