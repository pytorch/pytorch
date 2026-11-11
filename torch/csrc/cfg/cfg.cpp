#include <torch/csrc/cfg/cfg.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/dispatch/DispatchCapture.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#ifdef _WIN32
#include <c10/util/win32-headers.h>
#else
#include <dlfcn.h>
#endif

#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// The dispatcher state a kernel resolution depends on besides its arguments.
struct World {
  uint64_t tls_included;
  uint64_t tls_excluded;
  bool grad_enabled;
  bool torch_function_mode;
};

World snapshotWorld() {
  const auto tls = c10::impl::tls_local_dispatch_key_set();
  return World{
      tls.included_.raw_repr(),
      tls.excluded_.raw_repr(),
      c10::GradMode::is_enabled(),
      at::impl::torch_function_mode_enabled()};
}

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

} // namespace

struct TcfgTrace {
  std::vector<Record> records;
  std::vector<int32_t> open;
  World world{};
  int32_t top_level = 0;
  // Top-level calls recorded by captures nested inside this one.
  int32_t nested_top_level = 0;
};

namespace {

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
  trace->world = snapshotWorld();
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

// How an operand of the Python-level operation becomes a kernel argument.
enum class Conv : uint8_t {
  Tensor,
  Bool,
  Int,
  IntAsDouble,
  IntAsList,
  Double,
  // A Python number passed where the kernel takes a Tensor: the argument
  // parser wraps it into a 0-dim "wrapped number" tensor.
  WrappedNumber,
  IntSeq,
};

struct ArgTemplate {
  enum Kind : uint8_t { Unbound, Const, Operand, PackedInts } kind = Unbound;
  Conv conv = Conv::Tensor;
  int32_t operand = -1;
  c10::IValue constant;
  std::vector<int32_t> packed;
};

struct OperandGuard {
  enum Kind : uint8_t {
    // Constant of the compiled function: nothing to check.
    None,
    Tensor,
    ExactType,
    IsNone,
    Identity,
  } kind = None;
  PyTypeObject* type = nullptr;
  unsigned int type_version = 0;
  uint64_t key_set = 0;
  PyObject* identity = nullptr;
};

enum class Ret : uint8_t { Tensor, Bool, Int, Double };

// A call of the entry kernel through its C++ signature. Tensors are passed by
// reference straight out of their Python objects, so unlike the boxed call
// this neither builds an IValue stack nor touches a reference count.
struct TypedCall {
  virtual ~TypedCall() = default;
  virtual PyObject* call(
      const TcfgSite& site,
      PyObject* const* operands,
      int32_t* status) const = 0;
};

} // namespace

struct TcfgSite {
  explicit TcfgSite(c10::OperatorHandle op_) : op(std::move(op_)) {}
  TcfgSite(const TcfgSite&) = delete;
  TcfgSite& operator=(const TcfgSite&) = delete;
  TcfgSite(TcfgSite&&) = delete;
  TcfgSite& operator=(TcfgSite&&) = delete;
  ~TcfgSite() {
    for (auto& g : guards) {
      Py_XDECREF(g.identity);
    }
  }

  c10::OperatorHandle op;
  // The DispatchKeySet the entry kernel is looked up with: the one recorded
  // for chain[entry_level].
  c10::DispatchKeySet entry_ks;
  // The thread-local key sets the entry kernel was recorded under.
  c10::impl::LocalDispatchKeySet entry_tls{c10::impl::PODLocalDispatchKeySet{}};
  std::vector<c10::DispatchKey> chain;
  size_t entry_level = 0;
  // The call tracked no gradient when it was captured, and the site relies on
  // that: it is guarded on it, and the kernel runs with autograd out of the
  // way.
  bool skips_autograd = false;
  World world{};
  std::vector<OperandGuard> guards;
  std::vector<ArgTemplate> args;
  Ret ret = Ret::Tensor;
  // Null when the operator's signature is not one of the typed ones; the
  // call then goes through the boxed calling convention.
  std::unique_ptr<TypedCall> typed;
};

namespace {

// Whether fn is a function of the library this file is part of, that is one
// of torch's own Python bindings.
bool sameLibraryAsTorch(void* fn) {
  auto* self = reinterpret_cast<void*>(&torch::cfg::initCfgBindings);
#ifdef _WIN32
  const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
      GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  HMODULE ours = nullptr;
  HMODULE theirs = nullptr;
  return GetModuleHandleExA(flags, static_cast<LPCSTR>(self), &ours) != 0 &&
      GetModuleHandleExA(flags, static_cast<LPCSTR>(fn), &theirs) != 0 &&
      ours == theirs;
#else
  Dl_info ours;
  Dl_info theirs;
  if (dladdr(self, &ours) == 0 || dladdr(fn, &theirs) == 0) {
    return false;
  }
  return ours.dli_fbase == theirs.dli_fbase;
#endif
}

// The same value as far as a kernel can tell: NaN matches NaN, and 0.0 does
// not match -0.0.
bool sameDouble(double a, double b) {
  return (a == b && std::signbit(a) == std::signbit(b)) ||
      (std::isnan(a) && std::isnan(b));
}

// Types whose identity pins their value, so that the argument the parser
// derived from one of them can be reused while the same object is passed.
bool isImmutableToken(PyObject* o) {
  return PyUnicode_CheckExact(o) || THPDtype_Check(o) || THPDevice_Check(o) ||
      THPLayout_Check(o) || THPMemoryFormat_Check(o);
}

bool isWrappedNumber(const c10::IValue& v, at::ScalarType dtype) {
  if (!v.isTensor()) {
    return false;
  }
  const auto& t = v.toTensor();
  return t.defined() && t.dim() == 0 && t.scalar_type() == dtype &&
      t.unsafeGetTensorImpl()->is_wrapped_number() && t.device().is_cpu();
}

struct Binder {
  Binder(const TcfgPyOp& pyop_, const Record& call_)
      : pyop(pyop_),
        call(call_),
        guards(pyop_.n_operands),
        args(call_.inputs.size()),
        operand_bound(pyop_.n_operands, false) {}

  bool fail(std::string msg) {
    error = std::move(msg);
    return false;
  }

  bool isConst(int k) const {
    return pyop.is_const != nullptr && pyop.is_const[k] != 0;
  }

  void setOperand(size_t a, int k, Conv conv, OperandGuard::Kind guard) {
    if (isConst(k) && conv != Conv::Tensor) {
      args[a].kind = ArgTemplate::Const;
      args[a].constant = call.inputs[a];
    } else {
      args[a].kind = ArgTemplate::Operand;
      args[a].operand = k;
      args[a].conv = conv;
      guards[k].kind = guard;
      guards[k].type = Py_TYPE(pyop.operands[k]);
    }
    operand_bound[k] = true;
  }

  // Binds operand k to schema argument a if the value the kernel received for
  // a is the one the argument parser derives from that operand.
  bool bind(int k, size_t a) {
    if (a >= args.size() || args[a].kind != ArgTemplate::Unbound) {
      return fail("operand does not map to a free schema argument");
    }
    PyObject* o = pyop.operands[k];
    const c10::IValue& v = call.inputs[a];
    if (THPVariable_CheckExact(o)) {
      const auto& t = THPVariable_Unpack(o);
      if (!v.isTensor() ||
          v.toTensor().unsafeGetTensorImpl() != t.unsafeGetTensorImpl()) {
        return fail("tensor operand is not the kernel argument");
      }
      setOperand(a, k, Conv::Tensor, OperandGuard::Tensor);
      guards[k].key_set = t.key_set().raw_repr();
      guards[k].type_version = Py_TYPE(o)->tp_version_tag;
      return true;
    }
    if (o == Py_None) {
      if (!v.isNone()) {
        return fail("None operand is not a None argument");
      }
      args[a].kind = ArgTemplate::Const;
      args[a].constant = v;
      guards[k].kind = isConst(k) ? OperandGuard::None : OperandGuard::IsNone;
      operand_bound[k] = true;
      return true;
    }
    if (PyBool_Check(o)) {
      const bool x = o == Py_True;
      if (v.isBool() && v.toBool() == x) {
        setOperand(a, k, Conv::Bool, OperandGuard::ExactType);
      } else if (
          isWrappedNumber(v, at::kBool) &&
          *v.toTensor().const_data_ptr<bool>() == x) {
        setOperand(a, k, Conv::WrappedNumber, OperandGuard::ExactType);
      } else {
        return fail("bool operand is not the kernel argument");
      }
      return true;
    }
    if (PyLong_CheckExact(o)) {
      int overflow = 0;
      const long long x = PyLong_AsLongLongAndOverflow(o, &overflow);
      if (overflow != 0) {
        return fail("int operand does not fit in int64");
      }
      if (v.isInt() && v.toInt() == x) {
        setOperand(a, k, Conv::Int, OperandGuard::ExactType);
      } else if (v.isDouble() && v.toDouble() == static_cast<double>(x)) {
        setOperand(a, k, Conv::IntAsDouble, OperandGuard::ExactType);
      } else if (
          v.isIntList() && v.toIntList().size() == 1 &&
          v.toIntList().get(0) == x) {
        setOperand(a, k, Conv::IntAsList, OperandGuard::ExactType);
      } else if (
          isWrappedNumber(v, at::kLong) &&
          *v.toTensor().const_data_ptr<int64_t>() == x) {
        setOperand(a, k, Conv::WrappedNumber, OperandGuard::ExactType);
      } else {
        return fail("int operand is not the kernel argument");
      }
      return true;
    }
    if (PyFloat_CheckExact(o)) {
      const double x = PyFloat_AS_DOUBLE(o);
      if (v.isDouble() && sameDouble(v.toDouble(), x)) {
        setOperand(a, k, Conv::Double, OperandGuard::ExactType);
      } else if (
          isWrappedNumber(v, at::kDouble) &&
          sameDouble(*v.toTensor().const_data_ptr<double>(), x)) {
        setOperand(a, k, Conv::WrappedNumber, OperandGuard::ExactType);
      } else {
        return fail("float operand is not the kernel argument");
      }
      return true;
    }
    if ((PyTuple_CheckExact(o) || PyList_CheckExact(o)) && v.isIntList()) {
      const auto list = v.toIntList();
      const Py_ssize_t n = PySequence_Fast_GET_SIZE(o);
      if (static_cast<size_t>(n) != list.size()) {
        return fail("int sequence operand is not the kernel argument");
      }
      for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(o, i);
        int overflow = 0;
        if (!PyLong_CheckExact(item) ||
            PyLong_AsLongLongAndOverflow(item, &overflow) != list.get(i) ||
            overflow != 0) {
          return fail("int sequence operand is not the kernel argument");
        }
      }
      setOperand(a, k, Conv::IntSeq, OperandGuard::ExactType);
      return true;
    }
    // Anything else is only usable as a constant: either a constant of the
    // compiled function, or an immutable object guarded by identity.
    if (v.isTensor()) {
      return fail("operand turns into a tensor the compiler cannot rebuild");
    }
    if (isConst(k)) {
      guards[k].kind = OperandGuard::None;
    } else if (isImmutableToken(o)) {
      guards[k].kind = OperandGuard::Identity;
      guards[k].identity = Py_NewRef(o);
    } else {
      return fail(
          std::string("unsupported operand type ") + Py_TYPE(o)->tp_name);
    }
    args[a].kind = ArgTemplate::Const;
    args[a].constant = v;
    operand_bound[k] = true;
    return true;
  }

  // f(2, 3) for a kernel taking one int list: the argument parser packs the
  // positional ints.
  bool bindPacked(int first, int count, size_t a) {
    const c10::IValue& v = call.inputs[a];
    if (!v.isIntList() || v.toIntList().size() != static_cast<size_t>(count)) {
      return false;
    }
    const auto list = v.toIntList();
    for (int i = 0; i < count; i++) {
      PyObject* o = pyop.operands[first + i];
      int overflow = 0;
      if (!PyLong_CheckExact(o) ||
          PyLong_AsLongLongAndOverflow(o, &overflow) != list.get(i) ||
          overflow != 0) {
        return false;
      }
    }
    bool all_const = true;
    for (int i = 0; i < count; i++) {
      all_const = all_const && isConst(first + i);
    }
    if (all_const) {
      args[a].kind = ArgTemplate::Const;
      args[a].constant = v;
    } else {
      args[a].kind = ArgTemplate::PackedInts;
      for (int i = 0; i < count; i++) {
        args[a].packed.push_back(first + i);
        guards[first + i].kind = OperandGuard::ExactType;
        guards[first + i].type = &PyLong_Type;
      }
    }
    for (int i = 0; i < count; i++) {
      operand_bound[first + i] = true;
    }
    return true;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const TcfgPyOp& pyop;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const Record& call;
  std::vector<OperandGuard> guards;
  std::vector<ArgTemplate> args;
  std::vector<bool> operand_bound;
  std::string error;
};

int findArgument(const c10::FunctionSchema& schema, const char* name) {
  const auto& arguments = schema.arguments();
  for (size_t i = 0; i < arguments.size(); i++) {
    if (arguments[i].name() == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Positional operands [first, first + count) go to the schema arguments in
// order, skipping `skip` (the receiver of a method call).
bool bindPositional(
    Binder& b,
    const c10::FunctionSchema& schema,
    int first,
    int count,
    int skip) {
  std::vector<size_t> slots;
  for (size_t i = 0; i < schema.arguments().size(); i++) {
    if (static_cast<int>(i) != skip && !schema.arguments()[i].kwarg_only()) {
      slots.push_back(i);
    }
  }
  if (count > 1 && !slots.empty() && b.bindPacked(first, count, slots[0])) {
    return true;
  }
  if (static_cast<size_t>(count) > slots.size()) {
    return b.fail("more positional operands than schema arguments");
  }
  for (int i = 0; i < count; i++) {
    if (!b.bind(first + i, slots[i])) {
      return false;
    }
  }
  return true;
}

bool bindKeywords(
    Binder& b,
    const c10::FunctionSchema& schema,
    PyObject* kwnames,
    int first) {
  const Py_ssize_t n = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
  for (Py_ssize_t i = 0; i < n; i++) {
    const char* name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(kwnames, i));
    if (name == nullptr) {
      PyErr_Clear();
      return b.fail("keyword name is not a string");
    }
    int a = findArgument(schema, name);
    if (a < 0 && std::strcmp(name, "input") == 0) {
      a = findArgument(schema, "self");
    }
    if (a < 0) {
      return b.fail(
          std::string("keyword '") + name + "' is not a schema argument");
    }
    if (!b.bind(first + static_cast<int>(i), static_cast<size_t>(a))) {
      return false;
    }
  }
  return true;
}

bool bindOperands(Binder& b, const c10::FunctionSchema& schema) {
  const TcfgPyOp& pyop = b.pyop;
  const int n_kw = pyop.kwnames == nullptr
      ? 0
      : static_cast<int>(PyTuple_GET_SIZE(pyop.kwnames));
  const int n_pos = pyop.n_operands - 1 - n_kw;
  switch (pyop.kind) {
    case TCFG_CALL: {
      if (pyop.callee == nullptr || !PyCFunction_Check(pyop.callee) ||
          !sameLibraryAsTorch(
              reinterpret_cast<void*>(PyCFunction_GET_FUNCTION(pyop.callee)))) {
        return b.fail("callee is not a torch builtin");
      }
      // The callee itself is not an argument; it only has to stay the same.
      b.guards[0].kind =
          b.isConst(0) ? OperandGuard::None : OperandGuard::Identity;
      if (b.guards[0].kind == OperandGuard::Identity) {
        b.guards[0].identity = Py_NewRef(pyop.operands[0]);
      }
      b.operand_bound[0] = true;
      return bindPositional(b, schema, 1, n_pos, -1) &&
          bindKeywords(b, schema, pyop.kwnames, 1 + n_pos);
    }
    case TCFG_METHOD: {
      if (pyop.callee == nullptr || !PyCFunction_Check(pyop.callee) ||
          PyCFunction_GET_SELF(pyop.callee) != pyop.operands[0] ||
          !sameLibraryAsTorch(
              reinterpret_cast<void*>(PyCFunction_GET_FUNCTION(pyop.callee)))) {
        return b.fail("callee is not a torch builtin method of the receiver");
      }
      const int self = findArgument(schema, "self");
      if (self < 0) {
        return b.fail("schema has no self argument");
      }
      return b.bind(0, static_cast<size_t>(self)) &&
          bindPositional(b, schema, 1, n_pos, self) &&
          bindKeywords(b, schema, pyop.kwnames, 1 + n_pos);
    }
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
        b.guards = std::move(reflected.guards);
        b.args = std::move(reflected.args);
        b.operand_bound = std::move(reflected.operand_bound);
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

bool anyTensorNeedsAutograd(const std::vector<c10::IValue>& inputs) {
  const bool grad_enabled = c10::GradMode::is_enabled();
  const auto needs = [&](const at::Tensor& t) {
    return t.defined() &&
        ((grad_enabled && t.requires_grad()) ||
         torch::autograd::isFwGradDefined(t));
  };
  for (const auto& v : inputs) {
    if (v.isTensor()) {
      if (needs(v.toTensor())) {
        return true;
      }
    } else if (v.isTensorList()) {
      for (const at::Tensor& t : v.toTensorList()) {
        if (needs(t)) {
          return true;
        }
      }
    } else if (v.isList()) {
      // Optional tensor lists and the like: stay on the safe side.
      for (const auto& e : v.toListRef()) {
        if (e.isTensor() && needs(e.toTensor())) {
          return true;
        }
      }
    }
  }
  return false;
}

bool matchResult(const c10::IValue& out, PyObject* result, Ret* ret) {
  if (out.isTensor()) {
    if (!THPVariable_CheckExact(result) ||
        THPVariable_Unpack(result).unsafeGetTensorImpl() !=
            out.toTensor().unsafeGetTensorImpl()) {
      return false;
    }
    *ret = Ret::Tensor;
    return true;
  }
  if (out.isBool()) {
    *ret = Ret::Bool;
    return PyBool_Check(result) && (result == Py_True) == out.toBool();
  }
  if (out.isInt()) {
    *ret = Ret::Int;
    int overflow = 0;
    return PyLong_CheckExact(result) &&
        PyLong_AsLongLongAndOverflow(result, &overflow) == out.toInt() &&
        overflow == 0;
  }
  if (out.isDouble()) {
    *ret = Ret::Double;
    return PyFloat_CheckExact(result) &&
        sameDouble(PyFloat_AS_DOUBLE(result), out.toDouble());
  }
  return false;
}

at::Tensor wrapNumber(const at::Scalar& s) {
  at::Tensor t = c10::scalar_to_tensor(s);
  t.unsafeGetTensorImpl()->set_wrapped_number(true);
  return t;
}

// How a kernel argument of C++ type A is produced from an ArgTemplate.
// accepts() is checked once when the site is built; prepare() runs per call
// and returns false if the operand cannot be converted after all.
template <class A>
struct Unbox;

template <>
struct Unbox<const at::Tensor&> {
  struct Holder {
    const at::Tensor* ptr = nullptr;
    at::Tensor temp;
  };
  static bool accepts(const ArgTemplate& a) {
    if (a.kind == ArgTemplate::Const) {
      return a.constant.isTensor();
    }
    return a.kind == ArgTemplate::Operand &&
        (a.conv == Conv::Tensor || a.conv == Conv::WrappedNumber);
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* operands) {
    if (a.kind == ArgTemplate::Const) {
      h.ptr = &a.constant.toTensor();
      return true;
    }
    PyObject* o = operands[a.operand];
    if (a.conv == Conv::Tensor) {
      h.ptr = &THPVariable_Unpack(o);
      return true;
    }
    if (PyBool_Check(o)) {
      h.temp = wrapNumber(at::Scalar(o == Py_True));
    } else if (PyFloat_CheckExact(o)) {
      h.temp = wrapNumber(at::Scalar(PyFloat_AS_DOUBLE(o)));
    } else {
      int overflow = 0;
      const long long x = PyLong_AsLongLongAndOverflow(o, &overflow);
      if (overflow != 0) {
        return false;
      }
      h.temp = wrapNumber(at::Scalar(static_cast<int64_t>(x)));
    }
    h.ptr = &h.temp;
    return true;
  }
  static const at::Tensor& get(Holder& h) {
    return *h.ptr;
  }
};

// The self argument of an in-place operator.
template <>
struct Unbox<at::Tensor&> {
  struct Holder {
    at::Tensor* ptr = nullptr;
  };
  static bool accepts(const ArgTemplate& a) {
    return a.kind == ArgTemplate::Operand && a.conv == Conv::Tensor;
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* operands) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    h.ptr = const_cast<at::Tensor*>(&THPVariable_Unpack(operands[a.operand]));
    return true;
  }
  static at::Tensor& get(Holder& h) {
    return *h.ptr;
  }
};

template <>
struct Unbox<const at::Scalar&> {
  struct Holder {
    at::Scalar value;
  };
  static bool accepts(const ArgTemplate& a) {
    if (a.kind == ArgTemplate::Const) {
      return a.constant.isScalar();
    }
    return a.kind == ArgTemplate::Operand &&
        (a.conv == Conv::Int || a.conv == Conv::Double ||
         a.conv == Conv::Bool || a.conv == Conv::IntAsDouble);
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* operands) {
    if (a.kind == ArgTemplate::Const) {
      h.value = a.constant.toScalar();
      return true;
    }
    PyObject* o = operands[a.operand];
    if (a.conv == Conv::Double) {
      h.value = at::Scalar(PyFloat_AS_DOUBLE(o));
    } else if (a.conv == Conv::Bool) {
      h.value = at::Scalar(o == Py_True);
    } else {
      int overflow = 0;
      const long long x = PyLong_AsLongLongAndOverflow(o, &overflow);
      if (overflow != 0) {
        return false;
      }
      h.value = a.conv == Conv::Int ? at::Scalar(static_cast<int64_t>(x))
                                    : at::Scalar(static_cast<double>(x));
    }
    return true;
  }
  static const at::Scalar& get(Holder& h) {
    return h.value;
  }
};

template <>
struct Unbox<int64_t> {
  struct Holder {
    int64_t value = 0;
  };
  static bool accepts(const ArgTemplate& a) {
    if (a.kind == ArgTemplate::Const) {
      return a.constant.isInt();
    }
    return a.kind == ArgTemplate::Operand && a.conv == Conv::Int;
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* operands) {
    if (a.kind == ArgTemplate::Const) {
      h.value = a.constant.toInt();
      return true;
    }
    int overflow = 0;
    h.value = PyLong_AsLongLongAndOverflow(operands[a.operand], &overflow);
    return overflow == 0;
  }
  static int64_t get(Holder& h) {
    return h.value;
  }
};

template <>
struct Unbox<bool> {
  struct Holder {
    bool value = false;
  };
  static bool accepts(const ArgTemplate& a) {
    if (a.kind == ArgTemplate::Const) {
      return a.constant.isBool();
    }
    return a.kind == ArgTemplate::Operand && a.conv == Conv::Bool;
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* operands) {
    h.value = a.kind == ArgTemplate::Const ? a.constant.toBool()
                                           : operands[a.operand] == Py_True;
    return true;
  }
  static bool get(Holder& h) {
    return h.value;
  }
};

template <>
struct Unbox<std::optional<at::ScalarType>> {
  struct Holder {
    std::optional<at::ScalarType> value;
  };
  static bool accepts(const ArgTemplate& a) {
    return a.kind == ArgTemplate::Const &&
        (a.constant.isNone() || a.constant.isInt());
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* /*operands*/) {
    if (a.constant.isInt()) {
      h.value = static_cast<at::ScalarType>(a.constant.toInt());
    }
    return true;
  }
  static std::optional<at::ScalarType> get(Holder& h) {
    return h.value;
  }
};

template <>
struct Unbox<at::OptionalIntArrayRef> {
  struct Holder {
    c10::SmallVector<int64_t, 4> values;
    bool present = false;
  };
  static bool accepts(const ArgTemplate& a) {
    switch (a.kind) {
      case ArgTemplate::Const:
        return a.constant.isNone() || a.constant.isIntList();
      case ArgTemplate::Operand:
        return a.conv == Conv::IntAsList || a.conv == Conv::IntSeq;
      case ArgTemplate::PackedInts:
        return true;
      default:
        return false;
    }
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* operands) {
    const auto push = [&](PyObject* o) {
      int overflow = 0;
      h.values.push_back(PyLong_AsLongLongAndOverflow(o, &overflow));
      return PyLong_CheckExact(o) && overflow == 0;
    };
    if (a.kind == ArgTemplate::Const) {
      if (a.constant.isNone()) {
        return true;
      }
      for (int64_t x : a.constant.toIntList()) {
        h.values.push_back(x);
      }
      h.present = true;
      return true;
    }
    h.present = true;
    if (a.kind == ArgTemplate::PackedInts) {
      for (int32_t k : a.packed) {
        if (!push(operands[k])) {
          return false;
        }
      }
      return true;
    }
    PyObject* o = operands[a.operand];
    if (a.conv == Conv::IntAsList) {
      return push(o);
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(o);
    for (Py_ssize_t i = 0; i < n; i++) {
      if (!push(PySequence_Fast_GET_ITEM(o, i))) {
        return false;
      }
    }
    return true;
  }
  static at::OptionalIntArrayRef get(Holder& h) {
    if (!h.present) {
      return std::nullopt;
    }
    return at::OptionalIntArrayRef(
        std::in_place, h.values.data(), h.values.size());
  }
};

PyObject* toPython(const TcfgSite& /*site*/, at::Tensor&& t) {
  return THPVariable_Wrap(std::move(t));
}
PyObject* toPython(const TcfgSite& /*site*/, at::Tensor& t) {
  return THPVariable_Wrap(t);
}
PyObject* toPython(const TcfgSite& /*site*/, bool b) {
  return Py_NewRef(b ? Py_True : Py_False);
}
PyObject* toPython(const TcfgSite& site, const at::Scalar& s) {
  switch (site.ret) {
    case Ret::Bool:
      return Py_NewRef(s.toBool() ? Py_True : Py_False);
    case Ret::Int:
      return PyLong_FromLongLong(s.toLong());
    default:
      return PyFloat_FromDouble(s.toDouble());
  }
}

template <class R>
bool returns(Ret ret) {
  if constexpr (std::is_same_v<R, bool>) {
    return ret == Ret::Bool;
  } else if constexpr (std::is_same_v<R, at::Scalar>) {
    return ret != Ret::Tensor;
  } else {
    return ret == Ret::Tensor;
  }
}

template <class R, class... A>
struct TypedCallImpl final : TypedCall {
  explicit TypedCallImpl(c10::TypedOperatorHandle<R(A...)> op)
      : op_(std::move(op)) {}

  PyObject* call(
      const TcfgSite& site,
      PyObject* const* operands,
      int32_t* status) const override {
    std::tuple<typename Unbox<A>::Holder...> holders;
    if (!prepare(site, operands, holders, std::index_sequence_for<A...>{})) {
      *status = TCFG_MISS;
      return nullptr;
    }
    return invoke(site, holders, std::index_sequence_for<A...>{});
  }

 private:
  template <size_t... I>
  static bool prepare(
      const TcfgSite& site,
      PyObject* const* operands,
      std::tuple<typename Unbox<A>::Holder...>& holders,
      std::index_sequence<I...> /*unused*/) {
    return (
        Unbox<A>::prepare(std::get<I>(holders), site.args[I], operands) && ...);
  }

  template <size_t... I>
  PyObject* invoke(
      const TcfgSite& site,
      std::tuple<typename Unbox<A>::Holder...>& holders,
      std::index_sequence<I...> /*unused*/) const {
    return toPython(
        site,
        op_.redispatch(site.entry_ks, Unbox<A>::get(std::get<I>(holders))...));
  }

  c10::TypedOperatorHandle<R(A...)> op_;
};

template <class R, class... A>
std::unique_ptr<TypedCall> tryTyped(const TcfgSite& site) {
  if (site.args.size() != sizeof...(A) || !returns<R>(site.ret)) {
    return nullptr;
  }
  size_t i = 0;
  if (!(Unbox<A>::accepts(site.args[i++]) && ...)) {
    return nullptr;
  }
  try {
    // typed() verifies the signature against the one the kernels were
    // registered with.
    return std::make_unique<TypedCallImpl<R, A...>>(site.op.typed<R(A...)>());
  } catch (const c10::Error&) {
    return nullptr;
  }
}

std::unique_ptr<TypedCall> makeTypedCall(const TcfgSite& site) {
  // Only operators with native kernels are known to have a C++ signature
  // registered for typed() to check against.
  if (site.op.operator_name().name.rfind("aten::", 0) != 0) {
    return nullptr;
  }
  using T = const at::Tensor&;
  using S = const at::Scalar&;
  using Self = at::Tensor&;
  using Dtype = std::optional<at::ScalarType>;
  using Dims = at::OptionalIntArrayRef;
#define TCFG_TRY(...)                             \
  if (auto typed = tryTyped<__VA_ARGS__>(site)) { \
    return typed;                                 \
  }
  TCFG_TRY(at::Tensor, T)
  TCFG_TRY(at::Tensor, T, T)
  TCFG_TRY(at::Tensor, T, T, S)
  TCFG_TRY(at::Tensor, T, S)
  TCFG_TRY(at::Tensor, T, S, S)
  TCFG_TRY(at::Tensor, T, T, T)
  TCFG_TRY(at::Tensor, T, T, T, S)
  TCFG_TRY(at::Tensor, T, T, T, S, S)
  TCFG_TRY(at::Tensor, T, Dtype)
  TCFG_TRY(at::Tensor, T, Dims, bool, Dtype)
  TCFG_TRY(at::Tensor, T, int64_t)
  TCFG_TRY(at::Tensor, T, int64_t, int64_t)
  TCFG_TRY(at::Tensor, T, int64_t, Dtype)
  TCFG_TRY(bool, T)
  TCFG_TRY(at::Scalar, T)
  TCFG_TRY(Self, Self)
  TCFG_TRY(Self, Self, T)
  TCFG_TRY(Self, Self, T, S)
  TCFG_TRY(Self, Self, S)
  TCFG_TRY(Self, Self, S, S)
#undef TCFG_TRY
  return nullptr;
}

TcfgSite* siteFromTraceImpl(
    const TcfgTrace& trace,
    const TcfgPyOp& pyop,
    std::string& reason) {
  if (trace.top_level != 1 || trace.nested_top_level != 0) {
    reason = "not a single dispatcher call";
    return nullptr;
  }
  if (trace.world.torch_function_mode) {
    reason = "a torch function mode is active";
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
  if (!call.op.hasSchema()) {
    reason = "operator has no schema";
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
  for (int k = 0; k < pyop.n_operands; k++) {
    if (!binder.operand_bound[k]) {
      reason = "an operand does not reach the kernel";
      return nullptr;
    }
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
  if (!matchResult(call.outputs[0], pyop.result, &site->ret)) {
    reason = "the result is not the kernel output";
    return nullptr;
  }

  // The resolution chain: the call, then every kernel that did nothing but
  // redispatch the same operator once.
  std::vector<size_t> chain{call_index};
  while (true) {
    const auto parent = static_cast<int32_t>(chain.back());
    int32_t only_child = -1;
    int32_t children = 0;
    for (size_t i = chain.back() + 1; i < trace.records.size(); i++) {
      if (trace.records[i].parent == parent) {
        children++;
        only_child = static_cast<int32_t>(i);
      }
    }
    if (children != 1) {
      break;
    }
    const Record& child = trace.records[only_child];
    if (!child.is_redispatch || child.op != call.op) {
      break;
    }
    chain.push_back(static_cast<size_t>(only_child));
  }
  for (size_t i : chain) {
    site->chain.push_back(trace.records[i].ks.highestPriorityTypeId());
  }
  if ((pyop.flags & TCFG_ALLOW_SKIP_AUTOGRAD) != 0 &&
      c10::isIncludedInAlias(site->chain[0], c10::DispatchKey::Autograd) &&
      !anyTensorNeedsAutograd(call.inputs)) {
    site->skips_autograd = true;
    // The Autograd kernel of a call that tracks no gradient only forwards to
    // the next kernel, so the site can enter there. An operator without such
    // a kernel is a composite: it stays the entry, and the operators it calls
    // skip autograd instead.
    if (chain.size() > 1) {
      site->entry_level = 1;
    }
  }
  const Record& entry = trace.records[chain[site->entry_level]];
  site->entry_ks = entry.ks;
  site->entry_tls.included_ =
      c10::DispatchKeySet::from_raw_repr(entry.tls_included);
  site->entry_tls.excluded_ =
      c10::DispatchKeySet::from_raw_repr(entry.tls_excluded);
  if (site->skips_autograd && site->entry_level == 0) {
    // What the Autograd kernels of the nested operators would do one by one.
    site->entry_tls.excluded_ =
        site->entry_tls.excluded_ | c10::autograd_dispatch_keyset;
  }
  site->world = trace.world;
  site->guards = std::move(binder.guards);
  site->args = std::move(binder.args);
  // Ownership of the identity references moved into the site.
  binder.guards.clear();
  site->typed = makeTypedCall(*site);
  return site.release();
}

TcfgSite* siteFromTrace(
    const TcfgTrace* trace,
    const TcfgPyOp* pyop,
    char* reason,
    size_t reason_len) {
  std::string why;
  TcfgSite* site = nullptr;
  try {
    site = siteFromTraceImpl(*trace, *pyop, why);
  } catch (const std::exception& e) {
    why = e.what();
  }
  if (PyErr_Occurred() != nullptr) {
    PyErr_Clear();
  }
  if (reason != nullptr && reason_len > 0) {
    std::strncpy(reason, why.c_str(), reason_len - 1);
    reason[reason_len - 1] = '\0';
  }
  return site;
}

void siteFree(TcfgSite* site) {
  delete site;
}

// Kernels can call back into Python and from there into another site, so each
// activation takes its own stack. The stacks are kept to reuse their storage.
struct StackLease {
  StackLease() {
    if (depth == pool.size()) {
      pool.emplace_back();
    }
    stack = &pool[depth++];
  }
  StackLease(const StackLease&) = delete;
  StackLease& operator=(const StackLease&) = delete;
  StackLease(StackLease&&) = delete;
  StackLease& operator=(StackLease&&) = delete;
  ~StackLease() {
    stack->clear();
    depth--;
  }
  torch::jit::Stack* stack;
  static thread_local std::vector<torch::jit::Stack> pool;
  static thread_local size_t depth;
};
thread_local std::vector<torch::jit::Stack> StackLease::pool;
thread_local size_t StackLease::depth = 0;

bool pushOperand(torch::jit::Stack& stack, PyObject* o, Conv conv) {
  switch (conv) {
    case Conv::Tensor:
      stack.emplace_back(THPVariable_Unpack(o));
      return true;
    case Conv::Bool:
      stack.emplace_back(o == Py_True);
      return true;
    case Conv::Double:
      stack.emplace_back(PyFloat_AS_DOUBLE(o));
      return true;
    case Conv::WrappedNumber:
      if (PyBool_Check(o)) {
        stack.emplace_back(wrapNumber(at::Scalar(o == Py_True)));
        return true;
      }
      if (PyFloat_CheckExact(o)) {
        stack.emplace_back(wrapNumber(at::Scalar(PyFloat_AS_DOUBLE(o))));
        return true;
      }
      break;
    case Conv::IntSeq: {
      const Py_ssize_t n = PySequence_Fast_GET_SIZE(o);
      c10::List<int64_t> list;
      list.reserve(n);
      for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(o, i);
        int overflow = 0;
        const long long x = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (!PyLong_CheckExact(item) || overflow != 0) {
          return false;
        }
        list.push_back(x);
      }
      stack.emplace_back(std::move(list));
      return true;
    }
    default:
      break;
  }
  int overflow = 0;
  const long long x = PyLong_AsLongLongAndOverflow(o, &overflow);
  if (overflow != 0) {
    return false;
  }
  switch (conv) {
    case Conv::Int:
      stack.emplace_back(static_cast<int64_t>(x));
      return true;
    case Conv::IntAsDouble:
      stack.emplace_back(static_cast<double>(x));
      return true;
    case Conv::IntAsList:
      stack.emplace_back(c10::List<int64_t>({static_cast<int64_t>(x)}));
      return true;
    case Conv::WrappedNumber:
      stack.emplace_back(wrapNumber(at::Scalar(static_cast<int64_t>(x))));
      return true;
    default:
      return false;
  }
}

PyObject* siteCall(TcfgSite* site, PyObject* const* operands, int32_t* status) {
  *status = TCFG_MISS;
  const auto tls = c10::impl::tls_local_dispatch_key_set();
  if (tls.included_.raw_repr() != site->world.tls_included ||
      tls.excluded_.raw_repr() != site->world.tls_excluded ||
      at::impl::torch_function_mode_enabled()) {
    return nullptr;
  }
  const bool skips_autograd = site->skips_autograd;
  const bool grad_enabled = skips_autograd && c10::GradMode::is_enabled();
  const size_t n_operands = site->guards.size();
  for (size_t k = 0; k < n_operands; k++) {
    const OperandGuard& g = site->guards[k];
    PyObject* o = operands[k];
    switch (g.kind) {
      case OperandGuard::None:
        break;
      case OperandGuard::Tensor: {
        if (Py_TYPE(o) != g.type ||
            Py_TYPE(o)->tp_version_tag != g.type_version) {
          return nullptr;
        }
        const auto& t = THPVariable_Unpack(o);
        if (t.key_set().raw_repr() != g.key_set) {
          return nullptr;
        }
        // A tensor without autograd metadata neither requires grad nor
        // carries a forward gradient.
        if (skips_autograd &&
            t.unsafeGetTensorImpl()->autograd_meta() != nullptr &&
            ((grad_enabled && t.requires_grad()) ||
             torch::autograd::isFwGradDefined(t))) {
          return nullptr;
        }
        break;
      }
      case OperandGuard::ExactType:
        if (Py_TYPE(o) != g.type) {
          return nullptr;
        }
        break;
      case OperandGuard::IsNone:
        if (o != Py_None) {
          return nullptr;
        }
        break;
      case OperandGuard::Identity:
        if (o != g.identity) {
          return nullptr;
        }
        break;
    }
  }

  *status = TCFG_ERROR;
  HANDLE_TH_ERRORS
  // Entering below the kernels that were skipped means entering in the
  // thread-local state they would have set up.
  std::optional<c10::impl::ForceDispatchKeyGuard> tls_guard;
  if (skips_autograd) {
    tls_guard.emplace(site->entry_tls);
  }
  if (site->typed) {
    PyObject* result = site->typed->call(*site, operands, status);
    if (result != nullptr) {
      *status = TCFG_OK;
    }
    return result;
  }
  StackLease lease;
  torch::jit::Stack& stack = *lease.stack;
  for (const ArgTemplate& arg : site->args) {
    bool ok = true;
    switch (arg.kind) {
      case ArgTemplate::Const:
        stack.push_back(arg.constant);
        break;
      case ArgTemplate::Operand:
        ok = pushOperand(stack, operands[arg.operand], arg.conv);
        break;
      case ArgTemplate::PackedInts: {
        c10::List<int64_t> list;
        list.reserve(arg.packed.size());
        for (int32_t k : arg.packed) {
          int overflow = 0;
          list.push_back(PyLong_AsLongLongAndOverflow(operands[k], &overflow));
          ok = ok && overflow == 0;
        }
        stack.emplace_back(std::move(list));
        break;
      }
      case ArgTemplate::Unbound:
        ok = false;
        break;
    }
    if (!ok) {
      // Let the generic operation report what is wrong with the operand.
      *status = TCFG_MISS;
      return nullptr;
    }
  }
  site->op.redispatchBoxed(site->entry_ks, &stack);
  PyObject* result = nullptr;
  c10::IValue& out = stack.back();
  switch (site->ret) {
    case Ret::Tensor:
      result = THPVariable_Wrap(std::move(out).toTensor());
      break;
    case Ret::Bool:
      result = Py_NewRef(out.toBool() ? Py_True : Py_False);
      break;
    case Ret::Int:
      result = PyLong_FromLongLong(out.toInt());
      break;
    case Ret::Double:
      result = PyFloat_FromDouble(out.toDouble());
      break;
  }
  if (result != nullptr) {
    *status = TCFG_OK;
  }
  return result;
  END_HANDLE_TH_ERRORS
}

PyObject* siteInfo(const TcfgSite* site) {
  HANDLE_TH_ERRORS
  const auto& schema = site->op.schema();
  py::dict info;
  info["op"] = toString(site->op.operator_name());
  info["key"] = c10::toString(site->entry_ks.highestPriorityTypeId());
  py::list chain;
  for (auto k : site->chain) {
    chain.append(c10::toString(k));
  }
  info["chain"] = chain;
  info["entry_level"] = site->entry_level;
  info["skips_autograd"] = site->skips_autograd;
  info["calling_convention"] = site->typed ? "unboxed" : "boxed";
  py::list args;
  for (size_t a = 0; a < site->args.size(); a++) {
    const ArgTemplate& arg = site->args[a];
    std::ostringstream ss;
    ss << schema.arguments()[a].name() << "=";
    switch (arg.kind) {
      case ArgTemplate::Operand:
        ss << "$" << arg.operand;
        break;
      case ArgTemplate::PackedInts: {
        ss << "[";
        for (size_t i = 0; i < arg.packed.size(); i++) {
          ss << (i == 0 ? "$" : ", $") << arg.packed[i];
        }
        ss << "]";
        break;
      }
      default:
        if (arg.constant.isTensor()) {
          ss << "<tensor>";
        } else {
          ss << arg.constant;
        }
        break;
    }
    args.append(ss.str());
  }
  info["args"] = args;
  py::list operand_args;
  for (size_t k = 0; k < site->guards.size(); k++) {
    int found = -1;
    for (size_t a = 0; a < site->args.size(); a++) {
      if (site->args[a].kind == ArgTemplate::Operand &&
          site->args[a].operand == static_cast<int32_t>(k)) {
        found = static_cast<int>(a);
      }
    }
    operand_args.append(found);
  }
  info["operand_args"] = operand_args;
  return info.release().ptr();
  END_HANDLE_TH_ERRORS
}

int32_t tensorInfo(PyObject* obj, TcfgTensorInfo* out) {
  if (!THPVariable_CheckExact(obj)) {
    return 0;
  }
  const auto& t = THPVariable_Unpack(obj);
  if (!t.defined()) {
    return 0;
  }
  const auto* impl = t.unsafeGetTensorImpl();
  if (impl->has_symbolic_sizes_strides()) {
    return 0;
  }
  const auto sizes = impl->sizes();
  out->ndim = static_cast<int32_t>(sizes.size());
  out->sizes = sizes.data();
  out->requires_grad = t.requires_grad() ? 1 : 0;
  out->key_set = t.key_set().raw_repr();
  return 1;
}

const TcfgApi api_table = {
    TCFG_API_VERSION,
    &captureBegin,
    &captureEnd,
    &traceFree,
    &traceRecords,
    &siteFromTrace,
    &siteFree,
    &siteCall,
    &siteInfo,
    &tensorInfo,
};

} // namespace

namespace torch::cfg {

void initCfgBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_cfg_c_api", []() {
    // PyCapsule_New takes a non-const pointer; the table is only read.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* api = const_cast<TcfgApi*>(&api_table);
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(api, "torch._C._cfg_c_api", nullptr));
  });

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
  // Returns NotImplemented when a guard of the site does not hold.
  site_class.def("call", [](TcfgSite& site, const py::args& operands) {
    TORCH_CHECK_VALUE(
        operands.size() == site.guards.size(), "wrong number of operands");
    int32_t status = TCFG_OK;
    auto result = py::reinterpret_steal<py::object>(
        siteCall(&site, PySequence_Fast_ITEMS(operands.ptr()), &status));
    if (status == TCFG_ERROR) {
      throw py::error_already_set();
    }
    return status == TCFG_MISS
        ? py::reinterpret_borrow<py::object>(Py_NotImplemented)
        : result;
  });

  // Runs `run`, which performs one Python-level operation on `operands`, under
  // a dispatch capture and returns (result, site, reason): the call site of
  // the operation, or None and the reason it has none.
  m.def(
      "_cfg_make_site",
      [](const std::string& kind,
         const py::object& run,
         const py::tuple& operands,
         const py::object& callee,
         const std::optional<py::tuple>& kwnames,
         const std::optional<std::vector<bool>>& constants,
         bool skip_autograd) {
        static const std::unordered_map<std::string, int32_t> kinds = {
            {"binary", TCFG_BINARY},
            {"unary", TCFG_UNARY},
            {"call", TCFG_CALL},
            {"method", TCFG_METHOD},
        };
        TORCH_CHECK_VALUE(!operands.empty(), "an operation has operands");
        TORCH_CHECK_VALUE(
            !constants || constants->size() == operands.size(),
            "constants needs one entry per operand");
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
        std::vector<uint8_t> is_const;
        if (constants) {
          is_const.assign(constants->begin(), constants->end());
          pyop.is_const = is_const.data();
        }
        pyop.kwnames = kwnames ? kwnames->ptr() : nullptr;
        pyop.callee = callee.is_none() ? nullptr : callee.ptr();
        pyop.result = result.ptr();
        pyop.flags = skip_autograd ? TCFG_ALLOW_SKIP_AUTOGRAD : 0;
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
      py::arg("operands"),
      py::kw_only(),
      py::arg("callee") = py::none(),
      py::arg("kwnames") = py::none(),
      py::arg("constants") = py::none(),
      py::arg("skip_autograd") = false);
}

} // namespace torch::cfg
