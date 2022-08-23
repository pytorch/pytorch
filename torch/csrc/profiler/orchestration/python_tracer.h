#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <c10/util/strong_type.h>

#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

class RecordQueue;
class Result;
namespace python_tracer {
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

using TraceKey = strong::type<
    uint64_t,
    struct TraceKey_,
    strong::regular,
    strong::hashable,
    strong::ostreamable>;

struct CompressedEvent {
  TraceKey key_;
  uint64_t system_tid_;
  kineto::DeviceAndResource kineto_info_;
  time_t enter_t_;
};

struct TORCH_API PythonTracerBase {
  static PythonTracerBase& get();
  virtual ~PythonTracerBase() = default;

  virtual void start(RecordQueue* queue) = 0;
  virtual void stop() = 0;
  virtual std::vector<std::shared_ptr<Result>> getEvents(
      std::function<time_t(approx_time_t)> time_converter,
      std::vector<CompressedEvent>& enters,
      time_t end_time_ns) = 0;
  virtual void clear() = 0;
};

using GetFn = PythonTracerBase& (*)();
TORCH_API void registerTracer(GetFn get_tracer);
} // namespace python_tracer
} // namespace impl
} // namespace profiler
} // namespace torch
