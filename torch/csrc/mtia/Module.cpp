#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/mtia/Module.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::mtia {

namespace {

c10::DeviceIndex checked_mtia_device_index(int64_t device_index) {
  TORCH_CHECK(
      device_index >= -1 && device_index < c10::Device::MAX_NUM_DEVICES,
      "Device index ",
      device_index,
      " is out of range for DeviceIndex [",
      -1,
      ", ",
      c10::Device::MAX_NUM_DEVICES - 1,
      "]");
  return static_cast<c10::DeviceIndex>(device_index);
}

} // namespace

struct _MTIAGraph {
  // MTIA use accelerator hooks to connect pytorch and outside.
  // We need to provide the MTIAGraph class at Python layer, but the hooks only
  // support hooking functions, not classes. Thus we store all MTIAGraph C++
  // instances in a map, and use a handle to choose the right instance.
  int64_t handle_;

  _MTIAGraph(bool keep_graph = false)
      : handle_(at::detail::getMTIAHooks().mtiagraphCreate(keep_graph)) {}

  ~_MTIAGraph() {
    at::detail::getMTIAHooks().mtiagraphDestroy(handle_);
  }

  void capture_begin(at::MempoolId_t pool) {
    at::detail::getMTIAHooks().mtiagraphCaptureBegin(handle_, pool);
  }

  void capture_end() {
    at::detail::getMTIAHooks().mtiagraphCaptureEnd(handle_);
  }

  void instantiate() {
    at::detail::getMTIAHooks().mtiagraphInstantiate(handle_);
  }

  void replay() {
    at::detail::getMTIAHooks().mtiagraphReplay(handle_);
  }

  void reset() {
    at::detail::getMTIAHooks().mtiagraphReset(handle_);
  }

  at::MempoolId_t pool() {
    return at::detail::getMTIAHooks().mtiagraphPool(handle_);
  }
};

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_mtia_init", []() {
    TORCH_INTERNAL_ASSERT(!torch::utils::is_device_in_bad_fork(at::kMTIA));
    torch::utils::register_fork_handler_for_device_init(at::kMTIA);
    at::globalContext().lazyInitDevice(c10::DeviceType::MTIA);

    // Initialize default generators for each MTIA device
    auto mtia_module = py::module_::import("torch.mtia");

    auto num_devices = at::detail::getMTIAHooks().deviceCount();
    py::tuple default_mtia_generators(num_devices);
    for (const auto i : c10::irange(num_devices)) {
      auto cast_gen = THPGenerator_initDefaultGenerator(
          at::detail::getMTIAHooks().getDefaultGenerator(i));
      default_mtia_generators[i] = py::reinterpret_steal<py::object>(cast_gen);
    }
    mtia_module.attr("default_generators") = default_mtia_generators;
  });

  m.def("_mtia_isBuilt", []() {
    // Check if the MTIAHooks class has been registered with the registry.
    return at::detail::isMTIAHooksBuilt();
  });

  m.def("_mtia_isInBadFork", []() {
    return torch::utils::is_device_in_bad_fork(at::kMTIA);
  });

  m.def("_mtia_getCurrentStream", [](int64_t device_index) {
    const auto c10_device_index = checked_mtia_device_index(device_index);
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getCurrentStream(c10_device_index);
  });

  m.def("_mtia_getCurrentRawStream", [](int64_t device_index) {
    const auto c10_device_index = checked_mtia_device_index(device_index);
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getCurrentRawStream(c10_device_index);
  });

  m.def("_mtia_deviceSynchronize", []() {
    torch::utils::device_lazy_init(at::kMTIA);
    at::detail::getMTIAHooks().deviceSynchronize(
        at::detail::getMTIAHooks().getCurrentDevice());
  });

  m.def("_mtia_exchangeDevice", [](int64_t device_index) {
    if (device_index < 0) {
      return static_cast<c10::DeviceIndex>(-1);
    }
    return at::detail::getMTIAHooks().exchangeDevice(
        checked_mtia_device_index(device_index));
  });

  m.def("_mtia_maybeExchangeDevice", [](int64_t device_index) {
    if (device_index < 0) {
      return static_cast<c10::DeviceIndex>(-1);
    }
    return at::detail::getMTIAHooks().maybeExchangeDevice(
        checked_mtia_device_index(device_index));
  });

  m.def("_mtia_getDefaultStream", [](int64_t device_index) {
    const auto c10_device_index = checked_mtia_device_index(device_index);
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getDefaultStream(c10_device_index);
  });

  m.def(
      "_mtia_setStream",
      [](int64_t stream_id, int64_t device_index, int64_t device_type) {
        const auto c10_device_index = checked_mtia_device_index(device_index);
        torch::utils::device_lazy_init(at::kMTIA);
        at::detail::getMTIAHooks().setCurrentStream(c10::Stream::unpack3(
            stream_id,
            c10_device_index,
            static_cast<c10::DeviceType>(device_type)));
      });

  m.def("_mtia_setCurrentStream", [](const c10::Stream& stream) {
    torch::utils::device_lazy_init(at::kMTIA);
    auto device = at::detail::getMTIAHooks().getCurrentDevice();
    if (device != stream.device_index()) {
      at::detail::getMTIAHooks().setCurrentDevice(stream.device_index());
    }
    at::detail::getMTIAHooks().setCurrentStream(stream);
  });

  m.def("_mtia_memoryStats", [](int64_t device_index) {
    const auto c10_device_index = checked_mtia_device_index(device_index);
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().memoryStats(c10_device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_getDeviceCapability", [](int64_t device_index) {
    const auto c10_device_index = checked_mtia_device_index(device_index);
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().getDeviceCapability(c10_device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_getDeviceProperties", [](int64_t device_index) {
    const auto c10_device_index = checked_mtia_device_index(device_index);
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().getDeviceProperties(c10_device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_emptyCache", []() { at::detail::getMTIAHooks().emptyCache(); });

  m.def(
      "_mtia_recordMemoryHistory",
      [](const std::optional<std::string>& enabled,
         const std::string& stacks,
         size_t max_entries) {
        at::detail::getMTIAHooks().recordMemoryHistory(
            enabled, stacks, max_entries);
      });

  m.def("_mtia_memorySnapshot", []() {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().memorySnapshot(std::nullopt);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_attachOutOfMemoryObserver", [](const py::function& observer) {
    at::detail::getMTIAHooks().attachOutOfMemoryObserver(observer.ptr());
    return;
  });

  m.def("_mtia_getDeviceCount", []() {
    return at::detail::getMTIAHooks().deviceCount();
  });

  m.def("_mtia_getDevice", []() {
    return at::detail::getMTIAHooks().getCurrentDevice();
  });

  m.def("_mtia_setDevice", [](int64_t device_index) {
    at::detail::getMTIAHooks().setCurrentDevice(
        checked_mtia_device_index(device_index));
  });

  m.def("_mtia_resetPeakMemoryStats", [](int64_t device_index) {
    at::detail::getMTIAHooks().resetPeakMemoryStats(
        checked_mtia_device_index(device_index));
  });

  m.def("_mtia_graphPoolHandle", []() {
    return at::detail::getMTIAHooks().graphPoolHandle();
  });

  py::class_<_MTIAGraph>(m, "_MTIAGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def("capture_begin", &_MTIAGraph::capture_begin)
      .def("capture_end", &_MTIAGraph::capture_end)
      .def("instantiate", &_MTIAGraph::instantiate)
      .def("replay", &_MTIAGraph::replay)
      .def("reset", &_MTIAGraph::reset)
      .def("pool", &_MTIAGraph::pool);
}

} // namespace torch::mtia
