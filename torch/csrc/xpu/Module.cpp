#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/util/CallOnce.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <torch/csrc/Module.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/xpu/python_comm.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace torch;

static bool in_bad_fork = false; // True for children forked after xpu init

#ifndef WIN32
// Called in the forked child if xpu has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_device_init(at::kXPU, true);
}
#endif

// Should be called before the first xpu call. It is mainly called in lazy_init.
// Note: This is distinct from initExtension because a stub xpu implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

// XPU management methods

PyObject* THXPModule_getArchFlags(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
#ifdef XPU_ARCH_FLAGS
  static const char* flags = C10_STRINGIZE(XPU_ARCH_FLAGS);
  return THPUtils_packString(flags);
#else
  Py_RETURN_NONE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_isInBadFork_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to set_device");

  auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::xpu::set_device(device_index);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_exchangeDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchange_device");

  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kXPU);
  auto current_device = c10::xpu::exchange_device(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_maybeExchangeDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to maybe_exchange_device");

  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kXPU);
  auto current_device = c10::xpu::maybe_exchange_device(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  auto device_index = c10::xpu::current_device();

  return THPUtils_packDeviceIndex(device_index);
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
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = at::xpu::getCurrentXPUStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
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
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  return PyLong_FromVoidPtr(
      &at::xpu::getCurrentXPUStream(c10_device_index).queue());
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
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = at::xpu::XPUStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

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
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  {
    pybind11::gil_scoped_release no_gil;
    // Only the SYCL queues we have reserved will be synchronized, see Note
    // [Synchronize Streams on Device].
    c10::xpu::syncStreamsOnDevice(device_index);
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

PyObject* THXPModule_memoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);

  using c10::CachingDeviceAllocator::DeviceStats;
  using c10::CachingDeviceAllocator::Stat;
  using c10::CachingDeviceAllocator::StatArray;
  using c10::CachingDeviceAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::xpu::XPUCachingAllocator::getDeviceStats(device_index);

  py::dict result;
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THXPModule_resetPeakMemoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::xpu::XPUCachingAllocator::resetPeakStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THXPModule_resetAccumulatedMemoryStats(
    PyObject* self,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::xpu::XPUCachingAllocator::resetAccumulatedStats(device_index);
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
               << static_cast<typename std::underlying_type_t<device_type>>(
                      prop.device_type);
        break;
    }
    return stream.str();
  };
  auto gpu_subslice_count = [](const DeviceProp& prop) {
    return (prop.gpu_eu_count / prop.gpu_eu_count_per_subslice);
  };
#if SYCL_COMPILER_VERSION >= 20250000
  auto get_device_architecture = [](const DeviceProp& prop) {
    return static_cast<int64_t>(prop.architecture);
  };
#endif
  auto m = py::handle(module).cast<py::module>();

#define DEFINE_READONLY_MEMBER(member) \
  def_readonly(#member, &DeviceProp::member)

#define THXP_FORALL_DEVICE_PROPERTIES(_)                         \
  py::class_<DeviceProp>(m, "_XpuDeviceProperties")              \
      ._(name)                                                   \
      ._(platform_name)                                          \
      ._(vendor)                                                 \
      ._(driver_version)                                         \
      ._(version)                                                \
      ._(max_compute_units)                                      \
      ._(gpu_eu_count)                                           \
      ._(max_work_group_size)                                    \
      ._(max_num_sub_groups)                                     \
      ._(sub_group_sizes)                                        \
      ._(has_fp16)                                               \
      ._(has_fp64)                                               \
      ._(has_atomic64)                                           \
      ._(has_bfloat16_conversions)                               \
      ._(has_subgroup_matrix_multiply_accumulate)                \
      ._(has_subgroup_matrix_multiply_accumulate_tensor_float32) \
      ._(has_subgroup_2d_block_io)

  THXP_FORALL_DEVICE_PROPERTIES(DEFINE_READONLY_MEMBER)
      .def_readonly("total_memory", &DeviceProp::global_mem_size)
      .def_property_readonly("gpu_subslice_count", gpu_subslice_count)
#if SYCL_COMPILER_VERSION >= 20250000
      .def_property_readonly("architecture", get_device_architecture)
#endif
      .def_property_readonly("type", get_device_type)
      .def(
          "__repr__",
          [&get_device_type, &gpu_subslice_count](const DeviceProp& prop) {
            std::ostringstream stream;
            stream << "_XpuDeviceProperties(name='" << prop.name
                   << "', platform_name='" << prop.platform_name << "', type='"
                   << get_device_type(prop) << "', driver_version='"
                   << prop.driver_version << "', total_memory="
                   << prop.global_mem_size / (1024ull * 1024) << "MB"
                   << ", max_compute_units=" << prop.max_compute_units
                   << ", gpu_eu_count=" << prop.gpu_eu_count
                   << ", gpu_subslice_count=" << gpu_subslice_count(prop)
                   << ", max_work_group_size=" << prop.max_work_group_size
                   << ", max_num_sub_groups=" << prop.max_num_sub_groups
                   << ", sub_group_sizes=[" << prop.sub_group_sizes
                   << "], has_fp16=" << prop.has_fp16
                   << ", has_fp64=" << prop.has_fp64
                   << ", has_atomic64=" << prop.has_atomic64 << ")";
            return stream.str();
          });
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to torch.xpu
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](c10::DeviceIndex device) -> c10::xpu::DeviceProp* {
        return at::xpu::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

static void initXpuMethodBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_xpu_getMemoryInfo", [](c10::DeviceIndex device_index) {
#if SYCL_COMPILER_VERSION >= 20250000
    auto total = at::xpu::getDeviceProperties(device_index)->global_mem_size;
    auto free = c10::xpu::get_raw_device(device_index)
                    .get_info<sycl::ext::intel::info::device::free_memory>();
    return std::make_tuple(free, total);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "torch.xpu.mem_get_info requires PyTorch to be built with SYCL compiler version 2025.0.0 or newer.");
#endif
  });
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THXPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);

  auto m = THPObjectPtr(PyImport_ImportModule("torch.xpu"));
  if (!m)
    throw python_error();

  auto set_module_attr = [&](const char* name, PyObject* v) {
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  auto num_gpus = c10::xpu::device_count();
  THPObjectPtr default_xpu_generators(
      PyTuple_New(static_cast<Py_ssize_t>(num_gpus)));
  for (const auto i : c10::irange(num_gpus)) {
    const auto& gen = at::xpu::detail::getDefaultXPUGenerator(i);
    auto* cast_gen = THPGenerator_initDefaultGenerator(gen);
    PyTuple_SetItem(default_xpu_generators.get(), i, cast_gen);
  }
  set_module_attr("default_generators", default_xpu_generators.get());
  bindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
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
    {"_xpu_getArchFlags", THXPModule_getArchFlags, METH_NOARGS, nullptr},
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
    {"_xpu_memoryStats", THXPModule_memoryStats, METH_O, nullptr},
    {"_xpu_resetAccumulatedMemoryStats",
     THXPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_xpu_resetPeakMemoryStats",
     THXPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    {nullptr}};

PyMethodDef* THXPModule_methods() {
  return _THXPModule_methods;
}

namespace torch::xpu {

void initModule(PyObject* module) {
  python::initCommMethods(module);
  registerXpuDeviceProperties(module);
  initXpuMethodBindings(module);
}

} // namespace torch::xpu
