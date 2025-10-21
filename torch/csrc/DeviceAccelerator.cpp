#include <c10/core/AllocatorConfig.h>
#include <torch/csrc/DeviceAccelerator.h>
#include <torch/csrc/utils/device_lazy_init.h>

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

  m.def("_accelerator_setDeviceIndex", [](c10::DeviceIndex device_index) {
    // If device index is negative, no-op
    if (device_index < 0) {
      return;
    }
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    at::accelerator::setDeviceIndex(device_index);
  });

  m.def("_accelerator_getDeviceIndex", []() {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::getDeviceIndex();
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

  m.def("_accelerator_getStream", [](c10::DeviceIndex device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::getCurrentStream(device_index);
  });

  m.def("_accelerator_synchronizeDevice", [](c10::DeviceIndex device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    if (torch::utils::is_device_lazy_init_supported(device_type) &&
        !torch::utils::is_device_initialized(device_type)) {
      return;
    }
    torch::utils::maybe_initialize_device(device_type);
    {
      py::gil_scoped_release no_gil;
      at::accelerator::synchronizeDevice(device_index);
    }
  });

  m.def("_accelerator_exchangeDevice", [](c10::DeviceIndex device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::exchangeDevice(device_index);
  });

  m.def("_accelerator_maybeExchangeDevice", [](c10::DeviceIndex device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::maybeExchangeDevice(device_index);
  });

  m.def("_accelerator_isAllocatorInitialized", []() {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    return at::getDeviceAllocator(device_type)->initialized();
  });

  m.def("_accelerator_emptyCache", []() { at::accelerator::emptyCache(); });

  m.def("_accelerator_getDeviceStats", [](c10::DeviceIndex device_index) {
    using c10::CachingAllocator::Stat;
    using c10::CachingAllocator::StatArray;
    using c10::CachingAllocator::StatType;
    using c10::CachingDeviceAllocator::DeviceStats;

    const auto stats = at::accelerator::getDeviceStats(device_index);
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

  m.def(
      "_accelerator_resetAccumulatedStats", [](c10::DeviceIndex device_index) {
        at::accelerator::resetAccumulatedStats(device_index);
      });

  m.def("_accelerator_resetPeakStats", [](c10::DeviceIndex device_index) {
    at::accelerator::resetPeakStats(device_index);
  });

  m.def("_accelerator_getMemoryInfo", [](c10::DeviceIndex device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    py::gil_scoped_release no_gil;
    return at::accelerator::getMemoryInfo(device_index);
  });

  m.def("_accelerator_setAllocatorSettings", [](std::string env) {
    c10::CachingAllocator::setAllocatorSettings(env);
  });
}

} // namespace torch::accelerator
