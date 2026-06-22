#include <torch/csrc/profiler/cupti/monitor_python.h>

#include <torch/csrc/profiler/cupti/monitor_native.h>
#include <torch/csrc/utils/pybind.h>

#include <c10/util/ApproximateClock.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace torch::profiler::impl {

namespace {
// CUPTI invokes this to stamp records with an approximate timestamp; its
// address is handed to CUPTI on the Python side via the binding below.
uint64_t cuptiApproximateTimeCallback() {
  return c10::getApproximateTime();
}
} // namespace

void initCuptiMonitorBindings(py::module& m) {
  // GIL-free CUPTI monitor bindings, grouped under the
  // torch._C._profiler._cupti_monitor submodule. The two callback addresses are
  // registered with CUPTI on the Python side via ctypes; everything else is
  // driven from the decode thread.
  auto cupti_monitor = m.def_submodule(
      "_cupti_monitor",
      "GIL-free CUPTI monitor buffer pool + v2 record-layout capture.");
  using torch::profiler::impl::CuptiMonitorBuffers;
  cupti_monitor.def("approximate_time_callback_address", []() {
    return reinterpret_cast<uintptr_t>(&cuptiApproximateTimeCallback);
  });
  cupti_monitor.def("configure_buffers", [](size_t buffer_size) {
    CuptiMonitorBuffers::get().configure(buffer_size);
  });
  // version selects the CUPTI Activity-API generation: 1 for
  // cuptiActivityRegisterCallbacks, 2 for the subscriber-scoped
  // cuptiActivityRegisterCallbacks_v2. Both feed the same native pool/queue.
  cupti_monitor.def(
      "buffer_request_callback_address",
      [](int version) -> uintptr_t {
        TORCH_CHECK(
            version == 1 || version == 2,
            "cupti monitor callback version must be 1 or 2, got ",
            version);
        if (version == 1) {
          return reinterpret_cast<uintptr_t>(
              &torch::profiler::impl::cuptiMonitorBufferRequested);
        }
        return reinterpret_cast<uintptr_t>(
            &torch::profiler::impl::cuptiMonitorBufferRequestedV2);
      },
      py::arg("version") = 1);
  cupti_monitor.def(
      "buffer_complete_callback_address",
      [](int version) -> uintptr_t {
        TORCH_CHECK(
            version == 1 || version == 2,
            "cupti monitor callback version must be 1 or 2, got ",
            version);
        if (version == 1) {
          return reinterpret_cast<uintptr_t>(
              &torch::profiler::impl::cuptiMonitorBufferCompleted);
        }
        return reinterpret_cast<uintptr_t>(
            &torch::profiler::impl::cuptiMonitorBufferCompletedV2);
      },
      py::arg("version") = 1);
  // Opens a new layout epoch (called by the reconfigure path after flushing the
  // old config and before enabling the new one) and returns its id.
  cupti_monitor.def("next_layout_epoch", []() {
    return CuptiMonitorBuffers::get().next_layout_epoch();
  });
  // Returns the v2 user-defined record layout captured for the given epoch (the
  // layout_epoch field of a completed buffer), as a list of
  // (kind, record_size, [(field_id, offset, size)]). Empty if that epoch has no
  // captured layout.
  cupti_monitor.def("record_layouts", [](uint64_t epoch) {
    py::list result;
    for (const auto& layout :
         CuptiMonitorBuffers::get().record_layouts(epoch)) {
      py::list fields;
      for (const auto& field : layout.fields) {
        fields.append(py::make_tuple(field.field_id, field.offset, field.size));
      }
      result.append(
          py::make_tuple(layout.kind, layout.record_size, std::move(fields)));
    }
    return result;
  });
  cupti_monitor.def("get_completed", []() -> py::object {
    std::optional<torch::profiler::impl::CompletedCuptiBuffer> buf;
    {
      py::gil_scoped_release release;
      buf = CuptiMonitorBuffers::get().get_completed();
    }
    if (!buf.has_value()) {
      return py::none();
    }
    return py::make_tuple(
        reinterpret_cast<uintptr_t>(buf->ptr),
        buf->valid_size,
        buf->ctx,
        buf->stream,
        buf->layout_epoch);
  });
  cupti_monitor.def("return_buffer", [](uintptr_t ptr) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    CuptiMonitorBuffers::get().return_buffer(reinterpret_cast<uint8_t*>(ptr));
  });
  cupti_monitor.def("pending_buffers", []() {
    return CuptiMonitorBuffers::get().pending_count();
  });
  cupti_monitor.def("allocated_buffers", []() {
    return CuptiMonitorBuffers::get().allocated_count();
  });
  cupti_monitor.def(
      "shutdown_buffers", []() { CuptiMonitorBuffers::get().shutdown(); });
  cupti_monitor.def(
      "reset_buffers", []() { CuptiMonitorBuffers::get().reset(); });
}

} // namespace torch::profiler::impl
