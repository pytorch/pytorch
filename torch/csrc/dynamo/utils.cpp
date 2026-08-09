#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/utils.h>

namespace torch::dynamo {

static std::array<PyMethodDef, 1> _methods = {{
    {nullptr,
     nullptr,
     0,
     nullptr} // Sentinel value indicating the end of the array
}};

static bool is_instancemethod(const py::object& obj) {
  return PyInstanceMethod_Check(obj.ptr());
}

static bool has_active_python_hooks(PyObject* dict) {
  bool has_hooks = false;
  Py_BEGIN_CRITICAL_SECTION(dict);
  has_hooks = PyDict_GET_SIZE(dict) != 0;
  Py_END_CRITICAL_SECTION();
  return has_hooks;
}

static bool has_active_node_hooks(torch::autograd::Node& node) {
  for (const auto& hook : node.pre_hooks()) {
    if (auto pyhook =
            dynamic_cast<torch::autograd::PyFunctionPreHook*>(hook.get())) {
      if (has_active_python_hooks(pyhook->dict)) {
        return true;
      }
    } else {
      return true;
    }
  }
  for (const auto& hook : node.post_hooks()) {
    if (auto pyhook =
            dynamic_cast<torch::autograd::PyFunctionPostHook*>(hook.get())) {
      if (has_active_python_hooks(pyhook->dict)) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

static bool has_grad_accumulator_node_hooks(const at::Tensor& tensor) {
  auto grad_accumulator =
      torch::autograd::impl::try_get_grad_accumulator(tensor);
  return grad_accumulator && has_active_node_hooks(*grad_accumulator);
}

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.utils",
    "Module containing C utils",
    -1,
    _methods.data()};

PyObject* torch_c_dynamo_utils_init() {
  auto m = PyModule_Create(&_module);
  if (m == nullptr)
    return nullptr;

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

  auto py_m = py::handle(m).cast<py::module>();
  py_m.def("is_instancemethod", is_instancemethod);
  py_m.def("has_grad_accumulator_node_hooks", has_grad_accumulator_node_hooks);
  return m;
}

} // namespace torch::dynamo
