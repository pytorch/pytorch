#include <torch/csrc/cfg/cfg.h>

#include <ATen/core/dispatch/DispatchCapture.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
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

struct ArgTemplate {
  enum Kind : uint8_t { Unbound, Const, Operand } kind = Unbound;
  int32_t operand = -1;
  c10::IValue constant;
};

struct TcfgSite {
  explicit TcfgSite(c10::OperatorHandle op_) : op(std::move(op_)) {}

  c10::OperatorHandle op;
  std::vector<ArgTemplate> args;
};

struct Binder {
  Binder(const TcfgPyOp& pyop_, const Record& call_)
      : pyop(pyop_), call(call_), args(call_.inputs.size()) {}

  bool fail(std::string msg) {
    error = std::move(msg);
    return false;
  }

  // Binds operand k to schema argument a if the value the kernel received for
  // a is the one the argument parser derives from that operand.
  bool bind(int k, size_t a) {
    PyObject* o = pyop.operands[k];
    const c10::IValue& v = call.inputs[a];
    if (THPVariable_CheckExact(o)) {
      const auto& t = THPVariable_Unpack(o);
      if (!v.isTensor() ||
          v.toTensor().unsafeGetTensorImpl() != t.unsafeGetTensorImpl()) {
        return fail("tensor operand is not the kernel argument");
      }
      args[a].kind = ArgTemplate::Operand;
      args[a].operand = k;
      return true;
    }
    return fail(std::string("unsupported operand type ") + Py_TYPE(o)->tp_name);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const TcfgPyOp& pyop;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const Record& call;
  std::vector<ArgTemplate> args;
  std::string error;
};

bool bindOperands(Binder& b, const c10::FunctionSchema& schema) {
  const TcfgPyOp& pyop = b.pyop;
  switch (pyop.kind) {
    case TCFG_BINARY: {
      if (pyop.n_operands != 2 || schema.arguments().size() < 2) {
        return b.fail("not a binary operator");
      }
      if (b.bind(0, 0) && b.bind(1, 1)) {
        return true;
      }
      // Reflected operators (2 - x calls rsub(x, 2)).
      Binder reflected(b.pyop, b.call);
      if (reflected.bind(1, 0) && reflected.bind(0, 1)) {
        b.args = std::move(reflected.args);
        b.error.clear();
        return true;
      }
      return false;
    }
    case TCFG_UNARY: {
      if (pyop.n_operands != 1 || schema.arguments().empty()) {
        return b.fail("not a unary operator");
      }
      return b.bind(0, 0);
    }
    default:
      return b.fail("unknown operation kind");
  }
}

bool matchResult(const c10::IValue& out, PyObject* result) {
  if (out.isTensor()) {
    if (!THPVariable_CheckExact(result) ||
        THPVariable_Unpack(result).unsafeGetTensorImpl() !=
            out.toTensor().unsafeGetTensorImpl()) {
      return false;
    }
    return true;
  }
  return false;
}

TcfgSite* siteFromTraceImpl(
    const TcfgTrace& trace,
    const TcfgPyOp& pyop,
    std::string& reason) {
  if (trace.top_level != 1 || trace.nested_top_level != 0) {
    reason = "not a single dispatcher call";
    return nullptr;
  }
  size_t call_index = 0;
  while (trace.records[call_index].depth != 0) {
    call_index++;
  }
  const Record& call = trace.records[call_index];
  if (call.threw) {
    reason = "the kernel raised";
    return nullptr;
  }
  const auto& schema = call.op.schema();
  if (schema.is_vararg() || schema.is_varret() ||
      schema.returns().size() != 1 || call.outputs.size() != 1 ||
      call.inputs.size() != schema.arguments().size()) {
    reason = "operator signature is not supported";
    return nullptr;
  }

  Binder binder(pyop, call);
  if (!bindOperands(binder, schema)) {
    reason = binder.error;
    return nullptr;
  }
  for (size_t a = 0; a < binder.args.size(); a++) {
    auto& arg = binder.args[a];
    if (arg.kind != ArgTemplate::Unbound) {
      continue;
    }
    // Arguments no operand supplies are defaults filled in by the argument
    // parser; they are the same for every call of this shape.
    if (call.inputs[a].isTensor() || call.inputs[a].isTensorList()) {
      reason = "a tensor argument does not come from an operand";
      return nullptr;
    }
    arg.kind = ArgTemplate::Const;
    arg.constant = call.inputs[a];
  }

  auto site = std::make_unique<TcfgSite>(call.op);
  if (!matchResult(call.outputs[0], pyop.result)) {
    reason = "the result is not the kernel output";
    return nullptr;
  }

  site->args = std::move(binder.args);
  return site.release();
}

PyObject* siteInfo(const TcfgSite* site) {
  HANDLE_TH_ERRORS
  const auto& schema = site->op.schema();
  py::dict info;
  info["op"] = toString(site->op.operator_name());
  py::list args;
  for (size_t a = 0; a < site->args.size(); a++) {
    const ArgTemplate& arg = site->args[a];
    std::ostringstream ss;
    ss << schema.arguments()[a].name() << "=";
    switch (arg.kind) {
      case ArgTemplate::Operand:
        ss << "$" << arg.operand;
        break;
      default:
        ss << arg.constant;
        break;
    }
    args.append(ss.str());
  }
  info["args"] = args;
  return info.release().ptr();
  END_HANDLE_TH_ERRORS
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

  py::class_<TcfgSite> site_class(m, "_CfgSite");
  site_class.def("info", [](const TcfgSite& site) {
    auto info = py::reinterpret_steal<py::object>(siteInfo(&site));
    if (!info) {
      throw py::error_already_set();
    }
    return info;
  });
  // Runs `run`, which performs one Python-level operation on `operands`, under
  // a dispatch capture and returns (result, site, reason): the call site of
  // the operation, or None and the reason it has none.
  m.def(
      "_cfg_make_site",
      [](const std::string& kind,
         const py::object& run,
         const py::tuple& operands) {
        static const std::unordered_map<std::string, int32_t> kinds = {
            {"binary", TCFG_BINARY},
            {"unary", TCFG_UNARY},
        };
        captureBegin();
        py::object result;
        try {
          result = run();
        } catch (...) {
          traceFree(captureEnd());
          throw;
        }
        std::unique_ptr<TcfgTrace> trace(captureEnd());
        const auto found = kinds.find(kind);
        TcfgPyOp pyop{};
        pyop.kind = found == kinds.end() ? -1 : found->second;
        pyop.n_operands = static_cast<int32_t>(operands.size());
        pyop.operands = PySequence_Fast_ITEMS(operands.ptr());
        pyop.result = result.ptr();
        std::string reason;
        std::unique_ptr<TcfgSite> site;
        if (trace) {
          site.reset(siteFromTraceImpl(*trace, pyop, reason));
        } else {
          reason = "no dispatcher call";
        }
        return py::make_tuple(result, std::move(site), reason);
      },
      py::arg("kind"),
      py::arg("run"),
      py::arg("operands"));
}

} // namespace torch::cfg
