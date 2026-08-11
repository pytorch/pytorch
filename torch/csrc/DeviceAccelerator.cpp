#include <c10/core/AllocatorConfig.h>
#include <c10/core/Device.h>
#include <torch/csrc/DeviceAccelerator.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/device_lazy_init.h>

namespace {

c10::DeviceIndex checkedDeviceIndex(int64_t device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < c10::Device::MAX_NUM_DEVICES,
      "Device index ",
      device_index,
      " is out of range for DeviceIndex [",
      0,
      ", ",
      c10::Device::MAX_NUM_DEVICES - 1,
      "]");
  return static_cast<c10::DeviceIndex>(device_index);
}

} // namespace

namespace torch::accelerator {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_accelerator_getAccelerator", []() -> std::optional<c10::Device> {
    // If no accelerator was available at compile time, return None.
    auto acc = at::getAccelerator(false);
    if (acc.has_value()) {
      return acc.value();
    } else {
      return std::nullopt;
    }
  });

  m.def("_accelerator_setDeviceIndex", [](int64_t device_index) {
    // If device index is negative, no-op
    if (device_index < 0) {
      return;
    }
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    torch::utils::maybe_initialize_device(device_type);
    at::accelerator::setDeviceIndex(c10_device_index);
  });

  m.def("_accelerator_getDeviceIndex", []() {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::getDeviceIndex();
  });

  m.def("_accelerator_getDeviceCapability", [](int64_t device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    torch::utils::maybe_initialize_device(device_type);
    auto caps = at::accelerator::getDeviceCapability(c10_device_index);

    py::dict dict;

    py::set dtype_set;
    caps.forEachSupportedScalarType([&](c10::ScalarType dtype) {
      THPDtype* thp_dtype = torch::getTHPDtype(dtype);
      py::object dtype_obj =
          py::reinterpret_borrow<py::object>((PyObject*)thp_dtype);
      dtype_set.add(dtype_obj);
    });

    dict["supported_dtypes"] = dtype_set;
    return dict;
  });

  m.def("_accelerator_setStream", [](c10::Stream stream) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    // Set the current device to the device of stream
    if (at::accelerator::getDeviceIndex() != stream.device_index()) {
      at::accelerator::setDeviceIndex(stream.device_index());
    }
    at::accelerator::setCurrentStream(stream);
  });

  m.def("_accelerator_getStream", [](int64_t device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::getCurrentStream(c10_device_index);
  });

  m.def("_accelerator_synchronizeDevice", [](int64_t device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    if (torch::utils::is_device_lazy_init_supported(device_type) &&
        !torch::utils::is_device_initialized(device_type)) {
      return;
    }
    torch::utils::maybe_initialize_device(device_type);
    {
      py::gil_scoped_release no_gil;
      at::accelerator::synchronizeDevice(c10_device_index);
    }
  });

  m.def("_accelerator_exchangeDevice", [](int64_t device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::exchangeDevice(c10_device_index);
  });

  m.def("_accelerator_maybeExchangeDevice", [](int64_t device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::maybeExchangeDevice(c10_device_index);
  });

  m.def("_accelerator_isAllocatorInitialized", []() {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    return at::getDeviceAllocator(device_type)->initialized();
  });

  m.def("_accelerator_emptyCache", []() { at::accelerator::emptyCache(); });

  m.def("_accelerator_emptyHostCache", []() {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    if (torch::utils::is_device_lazy_init_supported(device_type) &&
        !torch::utils::is_device_initialized(device_type)) {
      return;
    }
    at::accelerator::emptyHostCache();
  });

  m.def("_accelerator_getDeviceStats", [](int64_t device_index) {
    using c10::CachingAllocator::Stat;
    using c10::CachingAllocator::StatArray;
    using c10::CachingAllocator::StatType;
    using c10::CachingDeviceAllocator::DeviceStats;

    const auto c10_device_index = checkedDeviceIndex(device_index);
    const auto stats = at::accelerator::getDeviceStats(c10_device_index);
    const auto stat_to_dict = [](const Stat& stat) -> py::dict {
      py::dict dict;
      dict["current"] = stat.current;
      dict["peak"] = stat.peak;
      dict["allocated"] = stat.allocated;
      dict["freed"] = stat.freed;
      return dict;
    };

    const auto stat_array_to_dict = [=](const StatArray& stats) -> py::dict {
      const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
          kStatTypeNames = {"all", "small_pool", "large_pool"};
      py::dict dict;
      for (const auto i : c10::irange(kStatTypeNames.size())) {
        dict[kStatTypeNames[i]] = stat_to_dict(stats[i]);
      }
      return dict;
    };

    py::dict result;
    result["num_alloc_retries"] = stats.num_alloc_retries;
    result["num_ooms"] = stats.num_ooms;
    result["max_split_size"] = stats.max_split_size;
    result["num_sync_all_streams"] = stats.num_sync_all_streams;
    result["num_device_alloc"] = stats.num_device_alloc;
    result["num_device_free"] = stats.num_device_free;
    result["allocated_bytes"] = stat_array_to_dict(stats.allocated_bytes);
    result["reserved_bytes"] = stat_array_to_dict(stats.reserved_bytes);
    result["active_bytes"] = stat_array_to_dict(stats.active_bytes);
    result["requested_bytes"] = stat_array_to_dict(stats.requested_bytes);
    result["allocation"] = stat_array_to_dict(stats.allocation);
    result["segment"] = stat_array_to_dict(stats.segment);
    result["active"] = stat_array_to_dict(stats.active);
    result["inactive_split"] = stat_array_to_dict(stats.inactive_split);
    result["inactive_split_bytes"] =
        stat_array_to_dict(stats.inactive_split_bytes);
    result["oversize_allocations"] = stat_to_dict(stats.oversize_allocations);
    result["oversize_segments"] = stat_to_dict(stats.oversize_segments);
    return result;
  });

  m.def("_accelerator_resetAccumulatedStats", [](int64_t device_index) {
    at::accelerator::resetAccumulatedStats(checkedDeviceIndex(device_index));
  });

  m.def("_accelerator_resetPeakStats", [](int64_t device_index) {
    at::accelerator::resetPeakStats(checkedDeviceIndex(device_index));
  });

  m.def("_accelerator_getMemoryInfo", [](int64_t device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    torch::utils::maybe_initialize_device(device_type);
    py::gil_scoped_release no_gil;
    return at::accelerator::getMemoryInfo(c10_device_index);
  });

  m.def("_accelerator_getAllocatorSettings", []() {
    return c10::CachingAllocator::getAllocatorSettings();
  });

  m.def("_accelerator_setAllocatorSettings", [](std::string env) {
    c10::CachingAllocator::setAllocatorSettings(env);
  });

  m.def("_accelerator_getDefaultGenerator", [](int64_t device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    const auto c10_device_index = checkedDeviceIndex(device_index);
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::getDefaultGenerator(c10_device_index);
  });

  // Accelerator Graph class binding
  py::class_<at::accelerator::Graph, std::shared_ptr<at::accelerator::Graph>>(
      m, "_acceleratorGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](at::accelerator::Graph& self,
             std::optional<c10::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            c10::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::MempoolId_t{0, 0};
            at::GraphCaptureMode capture_mode = at::GraphCaptureMode::Default;
            if (capture_error_mode == "default") {
              capture_mode = at::GraphCaptureMode::Default;
            } else if (capture_error_mode == "global") {
              capture_mode = at::GraphCaptureMode::Global;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = at::GraphCaptureMode::ThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = at::GraphCaptureMode::Relaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `default`, `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool") = std::nullopt,
          py::arg("capture_error_mode") = "default",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(
              &at::accelerator::Graph::capture_end))
      .def(
          "instantiate",
          torch::wrap_pybind_function_no_gil(
              &at::accelerator::Graph::instantiate))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::accelerator::Graph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::accelerator::Graph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::accelerator::Graph::pool))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::accelerator::Graph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::accelerator::Graph::debug_dump),
          py::arg("path"));
}

} // namespace torch::accelerator
