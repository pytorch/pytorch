#pragma once
#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>

namespace at {

enum class GraphCaptureMode : int8_t {
  // Backend-defined default capture behavior.
  Default = 0,
  // Potentially unsafe API calls are prohibited. Errors may occur if capture in
  // the current thread affects other threads.
  Global,
  // Potentially unsafe API calls are prohibited. Errors occur only if capture
  // in the current thread affects itself.
  ThreadLocal,
  // The current thread is allowed to make potentially unsafe API calls, except
  // for calls that inherently conflict with stream capture.
  Relaxed,
};

// Arguments used to construct a GraphImplInterface instance.
//
// This struct is intentionally lightweight and extensible so that new options
// can be added in a backward-compatible way without breaking existing or
// out-of-tree backends.
struct TORCH_API GraphImplArgs {
  // Whether to keep the underlying raw graph after capture is complete.
  bool keep_graph = false;
};

// A lightweight, backend-agnostic interface that provides a unified API for
// graph capture and replay.
//
// Each backend (e.g. CUDA, XPU, etc.) implements this interface and registers
// its implementation via GraphImplRegistry. Implementations are required to
// provide a constructor that accepts `GraphImplArgs`.
// While the concrete semantics and detailed behavior of capture and replay may
// vary across backends, the API contract exposed here is consistent.
struct TORCH_API GraphImplInterface {
  virtual ~GraphImplInterface() = default;
  // Begin graph capture on the current device and stream.
  // `pool` specifies the memory pool to be used during capture.
  // `capture_mode` controls how capture interacts with other concurrent work.
  // Its exact semantics are backend-specific. If a backend does not support the
  // requested capture mode, it may choose to emit a warning, raise an error, or
  // fall back to the default mode.
  virtual void capture_begin(
      MempoolId_t pool = {0, 0},
      GraphCaptureMode capture_mode = GraphCaptureMode::Default) = 0;

  // End graph capture and finalize the captured graph.
  virtual void capture_end() = 0;

  // Instantiate the captured graph for execution.
  virtual void instantiate() = 0;

  // Replay the previously captured graph.
  virtual void replay() = 0;

  // Reset internal state and release any backend-specific resources.
  // After reset(), the instance may be reused for a new capture.
  virtual void reset() = 0;

  // Return the memory pool associated with the captured graph.
  virtual MempoolId_t pool() const = 0;

  // Enable backend-specific debug behavior for graph capture/replay.
  // Implementations may enable extra validation and/or logging to help diagnose
  // issues. Backends that do not support debug mode could implement this as a
  // no-op.
  virtual void enable_debug_mode() = 0;

  // Dump the captured graph to a file for debugging purposes. The file format
  // and content are backend-specific.
  virtual void debug_dump(const std::string& path) = 0;
};

TORCH_DECLARE_REGISTRY(GraphImplRegistry, GraphImplInterface, GraphImplArgs);

// Registry mapping DeviceType -> GraphImplInterface implementation.
// The key is the string returned by c10::DeviceTypeName(device_type, false).
#define REGISTER_GRAPH_IMPL(key, impl) \
C10_REGISTER_CLASS(GraphImplRegistry, key, impl)

// Check whether a graph implementation is registered for the given device type.
inline bool has_graph_impl(const c10::DeviceType device_type) {
  auto key = c10::DeviceTypeName(device_type, /*lowercase=*/false);
  return GraphImplRegistry()->Has(key);
}

// Factory function to create a graph implementation for the given device.
// Returns nullptr if no implementation is registered for the device.
inline std::unique_ptr<GraphImplInterface> create_graph_impl(
    const c10::DeviceType device_type,
    const GraphImplArgs& args = {}) {
  auto key = c10::DeviceTypeName(device_type, /*lowercase=*/false);
  return GraphImplRegistry()->Create(key, args);
}

} // namespace at
