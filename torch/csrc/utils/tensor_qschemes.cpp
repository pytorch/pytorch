#include <torch/csrc/utils/tensor_qschemes.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/QScheme.h>
#include <c10/core/QScheme.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>


namespace torch {
namespace utils {

static PyObject* thp_qscheme_array[at::COMPILE_TIME_NUM_QSCHEMES];
#define _ADD_QSCHEME(qscheme)                                             \
  {                                                                       \
    PyObject* qscheme_obj = THPQScheme_New(qscheme, toString(qscheme));   \
    thp_qscheme_array[static_cast<int>(qscheme)] = qscheme_obj;           \
    Py_INCREF(qscheme_obj);                                               \
    if (PyModule_AddObject(                                               \
            torch_module, toString(qscheme).c_str(), qscheme_obj) != 0) { \
      throw python_error();                                               \
    }                                                                     \
  }

void initializeQSchemes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  _ADD_QSCHEME(at::kPerTensorAffine);
  _ADD_QSCHEME(at::kPerChannelAffine);
  _ADD_QSCHEME(at::kPerTensorSymmetric);
  _ADD_QSCHEME(at::kPerChannelSymmetric);
}

PyObject* getTHPQScheme(at::QScheme qscheme) {
  auto qscheme_ = thp_qscheme_array[static_cast<int>(qscheme)];
  if (!qscheme_) {
    throw std::invalid_argument("unsupported QScheme");
  }
  return qscheme_;
}

} // namespace utils
} // namespace torch
