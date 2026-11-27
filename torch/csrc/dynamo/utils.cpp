#include <ATen/DeviceAccelerator.h>
#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>

namespace torch::dynamo {

static std::array<PyMethodDef, 1> _methods = {{
    {nullptr,
     nullptr,
     0,
     nullptr} // Sentinel value indicating the end of the array
}};

static bool is_instancemethod(py::object obj) {
  return PyInstanceMethod_Check(obj.ptr());
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
  py_m.def("get_current_stream", [](const at::Device& device) {
    auto acc_type{at::accelerator::getAccelerator(true).value()};

    TORCH_CHECK_VALUE(
        acc_type == device.type(),
        device.type(),
        " doesn't match the current accelerator ",
        acc_type);

    torch::utils::maybe_initialize_device(acc_type);
    auto device_index{
        device.has_index() ? device.index()
                           : at::accelerator::getDeviceIndex()};
    return at::accelerator::getCurrentStream(device_index);
  });
  return m;
}

} // namespace torch::dynamo
