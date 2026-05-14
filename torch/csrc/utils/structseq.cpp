/* Copyright Python Software Foundation
 *
 * This file is copy-pasted from CPython source code with modifications:
 * https://github.com/python/cpython/blob/master/Objects/structseq.c
 * https://github.com/python/cpython/blob/2.7/Objects/structseq.c
 *
 * The purpose of this file is to overwrite the default behavior
 * of repr of structseq to provide better printing for returned
 * structseq objects from operators, aka torch.return_types.*
 *
 * For more information on copyright of CPython, see:
 * https://github.com/python/cpython#copyright-and-license-information
 */

#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/structseq.h>
#include <sstream>

#include <structmember.h>

namespace torch::utils {

PyObject* returned_structseq_repr(PyStructSequence* obj) {
  PyTypeObject* typ = Py_TYPE(obj);
  Py_ssize_t num_elements = PyTuple_GET_SIZE(obj);

  std::stringstream ss;
  ss << typ->tp_name << "(\n";

  for (Py_ssize_t i = 0; i < num_elements; i++) {
    const char* cname = typ->tp_members[i].name;
    if (cname == nullptr) {
      PyErr_Format(
          PyExc_SystemError,
          "In structseq_repr(), member %zd name is nullptr"
          " for type %.500s",
          i,
          typ->tp_name);
      return nullptr;
    }

    PyObject* val = PyTuple_GET_ITEM(obj, i);
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
  ss << ')';

  return PyUnicode_FromString(ss.str().c_str());
}

} // namespace torch::utils
