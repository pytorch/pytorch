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
    for (int i=0; i < num_elements; i++) {
        PyObject *val, *repr;
        const char *cname, *crepr;

        cname = typ->tp_members[i].name;
        if (cname == NULL) {
            PyErr_Format(PyExc_SystemError, "In structseq_repr(), member %d name is NULL"
                         " for type %.500s", i, typ->tp_name);
            return NULL;
        }
        val = PyStructSequence_GET_ITEM(obj, i);
        repr = PyObject_Repr(val);
        if (repr == NULL)
            return NULL;
        crepr = PyUnicode_AsUTF8(repr);
        if (crepr == NULL) {
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
