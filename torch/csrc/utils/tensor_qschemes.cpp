#include <torch/csrc/utils/tensor_qschemes.h>

#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/QScheme.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::utils {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::array<PyObject*, at::COMPILE_TIME_NUM_QSCHEMES> thp_qscheme_array;

void initializeQSchemes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  for (const auto i : c10::irange(at::COMPILE_TIME_NUM_QSCHEMES)) {
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
} // namespace torch::utils
