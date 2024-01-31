#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/util/CallOnce.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <torch/csrc/Module.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include <pthread.h>

using namespace torch;

static bool in_bad_fork = false; // True for children forked after xpu init

// Called in the forked child if xpu has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_device_init(at::kXPU, true);
}

// Should be called before the first xpu call. It is mainly called in lazy_init.
// Note: This is distinct from initExtension because a stub xpu implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
}

// XPU management methods

static PyObject* THXPModule_isInBadFork_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

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

  torch::utils::device_lazy_init(at::kXPU);
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

  torch::utils::device_lazy_init(at::kXPU);
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
  poison_fork();
  return THPUtils_packUInt64(at::xpu::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_getCurrentStream_wrap(
    PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to current_stream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = at::xpu::getCurrentXPUStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_getCurrentStream_raw(
    PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index),
      "invalid argument to getCurrentRawStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromVoidPtr(&at::xpu::getCurrentXPUStream(device).queue());
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = at::xpu::XPUStream::unpack3(
      stream_id, device_index, static_cast<c10::DeviceType>(device_type));

  auto device = c10::xpu::current_device();
  if (device != stream.device_index()) {
    c10::xpu::set_device(stream.device_index());
  }
  at::xpu::setCurrentXPUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_xpuSynchronize(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to synchronize");
  int device = THPUtils_unpackInt(arg);
  {
    pybind11::gil_scoped_release no_gil;
    // Only the SYCL queues we have reserved will be synchronized, see Note
    // [Synchronize Streams on Device].
    c10::xpu::syncStreamsOnDevice(static_cast<c10::DeviceIndex>(device));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_emptyCache(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::xpu::XPUCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
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
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();
  at::globalContext().lazyInitXPU();

  auto m = THPObjectPtr(PyImport_ImportModule("torch.xpu"));
  if (!m)
    throw python_error();

  auto set_module_attr = [&](const char* name, PyObject* v) {
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  auto num_gpus = c10::xpu::device_count();
  auto default_xpu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for (const auto i : c10::irange(num_gpus)) {
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(
        at::xpu::detail::getDefaultXPUGenerator(i));
    PyTuple_SetItem(default_xpu_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_xpu_generators);
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
    {"_xpu_isInBadFork", THXPModule_isInBadFork_wrap, METH_NOARGS, nullptr},
    {"_xpu_getCurrentStream",
     THXPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_xpu_getCurrentRawStream",
     THXPModule_getCurrentStream_raw,
     METH_O,
     nullptr},
    {"_xpu_setStream",
     castPyCFunctionWithKeywords(THXPModule_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_xpu_synchronize", THXPModule_xpuSynchronize, METH_O, nullptr},
    {"_xpu_emptyCache", THXPModule_emptyCache, METH_NOARGS, nullptr},
    {nullptr}};

PyMethodDef* THXPModule_methods() {
  return _THXPModule_methods;
}

namespace torch::xpu {

void initModule(PyObject* module) {
  registerXpuDeviceProperties(module);
}

} // namespace torch::xpu
