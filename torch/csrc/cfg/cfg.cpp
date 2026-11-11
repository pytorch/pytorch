#include <torch/csrc/cfg/cfg.h>

#include <ATen/core/dispatch/DispatchCapture.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <vector>

namespace {

struct Record {
  c10::OperatorHandle op;
  c10::DispatchKeySet ks;
  bool is_redispatch;
  int32_t depth;
};

struct Trace {
  std::vector<Record> records;
  int32_t depth = 0;
};

class Sink final : public c10::DispatchCaptureSink {
 public:
  void onKernelEnter(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet ks,
      bool isRedispatch,
      c10::ArrayRef<const c10::IValue> /*inputs*/) override {
    auto& t = *active.back();
    t.records.push_back(Record{op, ks, isRedispatch, t.depth});
    t.depth++;
  }

  void onKernelExit(
      const c10::OperatorHandle& /*op*/,
      c10::ArrayRef<const c10::IValue> /*outputs*/,
      bool /*threw*/) override {
    active.back()->depth--;
  }

  // Captures nest; each records into its own trace.
  std::vector<std::unique_ptr<Trace>> active;
  // The sink that was installed when the outermost capture began.
  c10::DispatchCaptureSink* previous = nullptr;
};

thread_local Sink tls_sink;

void captureBegin() {
  tls_sink.active.push_back(std::make_unique<Trace>());
  if (tls_sink.active.size() == 1) {
    tls_sink.previous = c10::setDispatchCaptureSink(&tls_sink);
  }
}

std::unique_ptr<Trace> captureEnd() {
  auto trace = std::move(tls_sink.active.back());
  tls_sink.active.pop_back();
  if (tls_sink.active.empty()) {
    c10::setDispatchCaptureSink(tls_sink.previous);
  }
  return trace;
}

} // namespace

namespace torch::cfg {

void initCfgBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Runs fn under a dispatch capture and returns (result, records) where
  // records lists (op, dispatch key, depth, is_redispatch) for every kernel
  // the dispatcher invoked.
  m.def("_cfg_dispatch_trace", [](const py::object& fn) {
    captureBegin();
    py::object result;
    try {
      result = fn();
    } catch (...) {
      captureEnd();
      throw;
    }
    const auto trace = captureEnd();
    py::list records;
    for (const auto& r : trace->records) {
      records.append(py::make_tuple(
          toString(r.op.operator_name()),
          c10::toString(r.ks.highestPriorityTypeId()),
          r.depth,
          r.is_redispatch));
    }
    return py::make_tuple(result, records);
  });
}

} // namespace torch::cfg
