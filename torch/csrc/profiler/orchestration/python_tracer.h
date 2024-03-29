#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <c10/util/ApproximateClock.h>
#include <c10/util/strong_type.h>

#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

class RecordQueue;
struct Result;
namespace python_tracer {

using TraceKey = strong::type<
    uint64_t,
    struct TraceKey_,
    strong::regular,
    strong::hashable,
    strong::ostreamable>;

struct CompressedEvent {
  TraceKey key_;
  uint64_t system_tid_{};
  kineto::DeviceAndResource kineto_info_{};
  c10::time_t enter_t_{};
};

/*
Libtorch does not depend on Python (e.g. cannot #include <Python.h>); however
when we call the profiler from libtorch_python we need the profiler to be able
to ingest the data that we collect from the Python tracer. (`PyEval_SetProfile`)

In order to solve this dependency issue we define a virtual base and a function
to register a getter. The python tracer then implements these functions and
exposes itself by calling `registerTracer` from `torch/csrc/autograd/init.cpp`.
This pattern of registration for faux python dependencies in libtorch is common
in the PyTorch codebase.
*/
struct TORCH_API PythonTracerBase {
  static std::unique_ptr<PythonTracerBase> make(RecordQueue* queue);
  virtual ~PythonTracerBase() = default;

  virtual void stop() = 0;
  virtual std::vector<std::shared_ptr<Result>> getEvents(
      std::function<c10::time_t(c10::approx_time_t)> time_converter,
      std::vector<CompressedEvent>& enters,
      c10::time_t end_time_ns) = 0;
};

using MakeFn = std::unique_ptr<PythonTracerBase> (*)(RecordQueue*);
TORCH_API void registerTracer(MakeFn make_tracer);
} // namespace python_tracer
} // namespace impl
} // namespace profiler
} // namespace torch
