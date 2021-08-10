#include <torch/csrc/utils/tensor_qschemes.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/QScheme.h>
#include <c10/core/QScheme.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>


namespace torch {
namespace utils {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyObject* thp_qscheme_array[at::COMPILE_TIME_NUM_QSCHEMES];

void initializeQSchemes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  for (int i = 0; i < at::COMPILE_TIME_NUM_QSCHEMES; ++i) {
    auto qscheme = static_cast<at::QScheme>(i);
    PyObject* qscheme_obj = THPQScheme_New(qscheme, toString(qscheme));
    thp_qscheme_array[static_cast<int>(qscheme)] = qscheme_obj;
    Py_INCREF(qscheme_obj);
    if (PyModule_AddObject(
            torch_module, toString(qscheme).c_str(), qscheme_obj) != 0) {
      throw python_error();
    }
  }
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
