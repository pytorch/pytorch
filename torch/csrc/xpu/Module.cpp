#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/util/CallOnce.h>
#include <c10/xpu/XPUFunctions.h>
#include <torch/csrc/Module.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

using namespace torch;

// XPU management methods

PyObject* THXPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to set_device");

  int device = THPUtils_unpackInt(arg);
  c10::xpu::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_exchangeDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchange_device");

  int device = THPUtils_unpackInt(arg);
  if (device < 0) {
    return THPUtils_packInt32(-1);
  }
  int current_device = c10::xpu::exchange_device(device);

  return THPUtils_packInt32(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_maybeExchangeDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to maybe_exchange_device");

  int device = THPUtils_unpackInt(arg);
  if (device < 0) {
    return THPUtils_packInt32(-1);
  }
  int current_device = c10::xpu::maybe_exchange_device(device);

  return THPUtils_packInt32(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int32_t>(c10::xpu::current_device());

  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  return THPUtils_packUInt64(at::xpu::device_count());
  END_HANDLE_TH_ERRORS
}

// XPU module initialization

static void registerXpuDeviceProperties(PyObject* module) {
  // Add _xpuDevicePropertires class to torch._C
  using namespace c10::xpu;
  auto get_device_type = [](const DeviceProp& prop) {
    std::ostringstream stream;
    using namespace sycl::info;
    switch (prop.device_type) {
      case device_type::cpu:
        stream << "cpu";
        break;
      case device_type::gpu:
        stream << "gpu";
        break;
      case device_type::accelerator:
        stream << "accelerator";
        break;
      case device_type::host:
        stream << "host";
        break;
      default:
        stream << "unknown device type:"
               << static_cast<typename std::underlying_type<device_type>::type>(
                      prop.device_type);
        break;
    }
    return stream.str();
  };
  auto gpu_subslice_count = [](const DeviceProp& prop) {
    return (prop.gpu_eu_count / prop.gpu_eu_count_per_subslice);
  };
  auto m = py::handle(module).cast<py::module>();
  py::class_<DeviceProp>(m, "_XpuDeviceProperties")
      .def_readonly("name", &DeviceProp::name)
      .def_readonly("platform_name", &DeviceProp::platform_name)
      .def_readonly("total_memory", &DeviceProp::global_mem_size)
      .def_readonly("max_compute_units", &DeviceProp::max_compute_units)
      .def_readonly("gpu_eu_count", &DeviceProp::gpu_eu_count)
      .def_property_readonly("gpu_subslice_count", gpu_subslice_count)
      .def_readonly("max_work_group_size", &DeviceProp::max_work_group_size)
      .def_readonly("max_num_sub_groups", &DeviceProp::max_num_sub_groups)
      .def_readonly("sub_group_sizes", &DeviceProp::sub_group_sizes)
      .def_property_readonly("type", get_device_type)
      .def(
          "__repr__",
          [&get_device_type, &gpu_subslice_count](const DeviceProp& prop) {
            std::ostringstream stream;
            stream << "_XpuDeviceProperties(name='" << prop.name
                   << "', platform_name='" << prop.platform_name << "', type='"
                   << get_device_type(prop)
                   << ", total_memory=" << prop.global_mem_size / (1024 * 1024)
                   << "MB, max_compute_units=" << prop.max_compute_units
                   << ", gpu_eu_count=" << prop.gpu_eu_count
                   << ", gpu_subslice_count=" << gpu_subslice_count(prop)
                   << ", max_work_group_size=" << prop.max_work_group_size
                   << ", max_num_sub_groups=" << prop.max_num_sub_groups
                   << ", sub_group_sizes=[" << prop.sub_group_sizes << "])";
            return stream.str();
          });
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to torch.xpu
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](int device) -> c10::xpu::DeviceProp* {
        return at::xpu::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THXPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  auto m = THPObjectPtr(PyImport_ImportModule("torch.xpu"));
  if (!m)
    throw python_error();

  bindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMethodDef _THXPModule_methods[] = {
    {"_xpu_init", THXPModule_initExtension, METH_NOARGS, nullptr},
    {"_xpu_setDevice", THXPModule_setDevice_wrap, METH_O, nullptr},
    {"_xpu_exchangeDevice", THXPModule_exchangeDevice_wrap, METH_O, nullptr},
    {"_xpu_maybeExchangeDevice",
     THXPModule_maybeExchangeDevice_wrap,
     METH_O,
     nullptr},
    {"_xpu_getDevice", THXPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_xpu_getDeviceCount",
     THXPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {nullptr}};

PyMethodDef* THXPModule_methods() {
  return _THXPModule_methods;
}

namespace torch::xpu {

void initModule(PyObject* module) {
  registerXpuDeviceProperties(module);
}

} // namespace torch::xpu
