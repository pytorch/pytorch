#include <torch/csrc/cfg/cfg.h>

#include <ATen/core/dispatch/DispatchCapture.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <vector>

namespace {

struct Record {
  c10::OperatorHandle op;
  c10::DispatchKeySet ks;
  // The thread-local key sets the kernel ran under. Kernels adjust them
  // before they redispatch (the Autograd kernel excludes the autograd keys so
  // that the calls the backend kernel makes skip autograd as well), so they
  // are part of what a replay of the kernel has to reproduce.
  uint64_t tls_included;
  uint64_t tls_excluded;
  bool is_redispatch;
  bool threw;
  int32_t depth;
  int32_t parent;
  // Only kept for top-level calls: the values pin the tensors they name, so
  // their addresses stay meaningful until the trace is analyzed.
  std::vector<c10::IValue> inputs;
  std::vector<c10::IValue> outputs;
};

struct TcfgTrace {
  std::vector<Record> records;
  std::vector<int32_t> open;
  int32_t top_level = 0;
  // Top-level calls recorded by captures nested inside this one.
  int32_t nested_top_level = 0;
};

class Sink final : public c10::DispatchCaptureSink {
 public:
  void onKernelEnter(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet ks,
      bool isRedispatch,
      c10::ArrayRef<const c10::IValue> inputs) override {
    auto& t = *active.back();
    const bool top = t.open.empty();
    const auto tls = c10::impl::tls_local_dispatch_key_set();
    Record r{
        op,
        ks,
        tls.included_.raw_repr(),
        tls.excluded_.raw_repr(),
        isRedispatch,
        /*threw=*/false,
        static_cast<int32_t>(t.open.size()),
        top ? -1 : t.open.back(),
        {},
        {}};
    if (top) {
      r.inputs.assign(inputs.begin(), inputs.end());
      t.top_level++;
    }
    t.open.push_back(static_cast<int32_t>(t.records.size()));
    t.records.push_back(std::move(r));
  }

  void onKernelExit(
      const c10::OperatorHandle& /*op*/,
      c10::ArrayRef<const c10::IValue> outputs,
      bool threw) override {
    auto& t = *active.back();
    auto& r = t.records[t.open.back()];
    t.open.pop_back();
    r.threw = threw;
    if (t.open.empty() && !threw) {
      r.outputs.assign(outputs.begin(), outputs.end());
    }
  }

  std::vector<std::unique_ptr<TcfgTrace>> active;
  // The sink that was installed when the outermost capture began.
  c10::DispatchCaptureSink* previous = nullptr;
};

thread_local Sink tls_sink;

void captureBegin() {
  auto trace = std::make_unique<TcfgTrace>();
  tls_sink.active.push_back(std::move(trace));
  if (tls_sink.active.size() == 1) {
    tls_sink.previous = c10::setDispatchCaptureSink(&tls_sink);
  }
}

TcfgTrace* captureEnd() {
  if (tls_sink.active.empty()) {
    return nullptr;
  }
  auto trace = std::move(tls_sink.active.back());
  tls_sink.active.pop_back();
  if (tls_sink.active.empty()) {
    c10::setDispatchCaptureSink(tls_sink.previous);
  } else {
    tls_sink.active.back()->nested_top_level +=
        trace->top_level + trace->nested_top_level;
  }
  if (trace->records.empty() && trace->nested_top_level == 0) {
    return nullptr;
  }
  return trace.release();
}

void traceFree(TcfgTrace* trace) {
  delete trace;
}

PyObject* traceRecords(const TcfgTrace* trace) {
  HANDLE_TH_ERRORS
  py::list out;
  for (const auto& r : trace->records) {
    out.append(py::make_tuple(
        toString(r.op.operator_name()),
        c10::toString(r.ks.highestPriorityTypeId()),
        r.depth,
        r.is_redispatch));
  }
  return out.release().ptr();
  END_HANDLE_TH_ERRORS
}

// Everything a capture keeps about its records, for tests.
py::list traceDetail(const TcfgTrace& trace) {
  const auto keys = [](uint64_t raw) {
    return c10::toString(c10::DispatchKeySet::from_raw_repr(raw));
  };
  const auto values = [](const Record& r, const std::vector<c10::IValue>& v) {
    if (r.depth != 0) {
      return py::object(py::none());
    }
    py::tuple out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
      out[i] = torch::jit::toPyObject(v[i]);
    }
    return py::object(std::move(out));
  };
  py::list out;
  for (const auto& r : trace.records) {
    py::dict d;
    d["op"] = toString(r.op.operator_name());
    d["key"] = c10::toString(r.ks.highestPriorityTypeId());
    d["depth"] = r.depth;
    d["is_redispatch"] = r.is_redispatch;
    d["parent"] = r.parent;
    d["threw"] = r.threw;
    d["tls_included"] = keys(r.tls_included);
    d["tls_excluded"] = keys(r.tls_excluded);
    d["inputs"] = values(r, r.inputs);
    d["outputs"] = values(r, r.outputs);
    out.append(std::move(d));
  }
  return out;
}

} // namespace

namespace torch::cfg {

void initCfgBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Runs fn under a dispatch capture and returns (result, records) where
  // records lists (op, dispatch key, depth, is_redispatch) for every kernel
  // the dispatcher invoked.
  // With detail, a record is a dict that also has the rest of what the capture
  // kept about it.
  m.def(
      "_cfg_dispatch_trace",
      [](const py::object& fn, bool detail) -> py::tuple {
        captureBegin();
        py::object result;
        try {
          result = fn();
        } catch (...) {
          traceFree(captureEnd());
          throw;
        }
        std::unique_ptr<TcfgTrace> trace(captureEnd());
        if (detail) {
          return py::make_tuple(
              result, trace ? traceDetail(*trace) : py::list());
        }
        py::object records = trace
            ? py::reinterpret_steal<py::object>(traceRecords(trace.get()))
            : py::object(py::list());
        if (!records) {
          throw py::error_already_set();
        }
        return py::make_tuple(result, records);
      },
      py::arg("fn"),
      py::arg("detail") = false);
}

} // namespace torch::cfg
