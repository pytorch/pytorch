#include <torch/csrc/cfg/cfg.h>
#include <torch/csrc/cfg/workspace.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/ScalarOps.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/dispatch/DispatchCapture.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/enum_tag.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_memoryformats.h>

#ifdef _WIN32
#include <c10/util/win32-headers.h>
#else
#include <dlfcn.h>
#endif

#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
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
  // Only kept for the top-level call of a capture that has one: the values
  // pin the tensors they name, so their addresses stay meaningful until the
  // trace is analyzed. A capture with a second top-level call is not made a
  // site of (siteFromTrace) and may be of anything, a whole model included,
  // whose tensors it must not keep alive.
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
      if (t.top_level == 0) {
        r.inputs.assign(inputs.begin(), inputs.end());
      } else if (t.top_level == 1) {
        // The first record is the first top-level call.
        t.records.front().inputs.clear();
        t.records.front().outputs.clear();
      }
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
    if (t.open.empty() && !threw && t.top_level == 1) {
      r.outputs.assign(outputs.begin(), outputs.end());
    }
  }

  std::vector<std::unique_ptr<TcfgTrace>> active;
  // The sink that was installed when the outermost capture began.
  c10::DispatchCaptureSink* previous = nullptr;
};

thread_local Sink tls_sink;

// Raw pointers and PyObject* results below: these functions become entries of
// the C function table that the Rust runtime calls.
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
      uint32_t flags,
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

// Unboxed tensors: TensorImpl* | 1, an owned reference (see cfg.h).
using TensorPtr = c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;

bool isUnboxed(PyObject* o) {
  return (reinterpret_cast<uintptr_t>(o) & 1) != 0;
}

c10::TensorImpl* unboxedImpl(PyObject* o) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<c10::TensorImpl*>(
      reinterpret_cast<uintptr_t>(o) & ~static_cast<uintptr_t>(1));
}

PyObject* tagUnboxed(c10::TensorImpl* impl) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(impl) | 1);
}

// An at::Tensor for an unboxed operand that does not own a reference, so
// that handing it to a kernel costs no reference counting.
struct BorrowedTensor {
  explicit BorrowedTensor(c10::TensorImpl* impl)
      : tensor(TensorPtr::reclaim(impl)) {}
  BorrowedTensor(const BorrowedTensor&) = delete;
  BorrowedTensor& operator=(const BorrowedTensor&) = delete;
  BorrowedTensor(BorrowedTensor&&) = delete;
  BorrowedTensor& operator=(BorrowedTensor&&) = delete;
  ~BorrowedTensor() {
    tensor.unsafeReleaseTensorImpl();
  }
  at::Tensor tensor;
};

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
  Binder(const Binder&) = delete;
  Binder& operator=(const Binder&) = delete;
  Binder(Binder&&) = delete;
  Binder& operator=(Binder&&) = delete;
  ~Binder() {
    for (auto& g : guards) {
      Py_XDECREF(g.identity);
    }
  }

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
      // Reflected operators (2 - x calls rsub(x, 2)). If neither order binds,
      // the reason reported is the one of the order tried first.
      Binder reflected(b.pyop, b.call);
      if (reflected.bind(1, 0) && reflected.bind(0, 1)) {
        // What b bound so far goes away with reflected.
        b.guards.swap(reflected.guards);
        b.args.swap(reflected.args);
        b.operand_bound.swap(reflected.operand_bound);
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
    std::optional<BorrowedTensor> borrowed;
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
      if (isUnboxed(o)) {
        h.borrowed.emplace(unboxedImpl(o));
        h.ptr = &h.borrowed->tensor;
      } else {
        h.ptr = &THPVariable_Unpack(o);
      }
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
    std::optional<BorrowedTensor> borrowed;
  };
  static bool accepts(const ArgTemplate& a) {
    return a.kind == ArgTemplate::Operand && a.conv == Conv::Tensor;
  }
  static bool prepare(
      Holder& h,
      const ArgTemplate& a,
      PyObject* const* operands) {
    PyObject* o = operands[a.operand];
    if (isUnboxed(o)) {
      h.borrowed.emplace(unboxedImpl(o));
      h.ptr = &h.borrowed->tensor;
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      h.ptr = const_cast<at::Tensor*>(&THPVariable_Unpack(o));
    }
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

// A tensor a kernel returned by value. Unless the caller wants it as a Python
// object right away it stays unboxed: if nothing but other kernels ever looks
// at it, it never gets one.
PyObject* tensorResult(at::Tensor&& t, uint32_t flags) {
  if ((flags & TCFG_UNBOXED_RESULT) != 0 && t.defined()) {
    return tagUnboxed(t.unsafeReleaseTensorImpl());
  }
  return THPVariable_Wrap(std::move(t));
}

PyObject* toPython(const TcfgSite& /*site*/, uint32_t flags, at::Tensor&& t) {
  return tensorResult(std::move(t), flags);
}
// The self an in-place operator returns. If it has a Python object, that is
// the result; taking another reference to the tensor instead would have to
// pin the Python object.
PyObject* toPython(const TcfgSite& /*site*/, uint32_t flags, at::Tensor& t) {
  auto* impl = t.unsafeGetTensorImpl();
  if ((flags & TCFG_UNBOXED_RESULT) != 0 && t.defined() &&
      impl->pyobj_slot()->load_pyobj() == nullptr) {
    c10::raw::intrusive_ptr::incref(impl);
    return tagUnboxed(impl);
  }
  return THPVariable_Wrap(t);
}
PyObject* toPython(const TcfgSite& /*site*/, uint32_t /*flags*/, bool b) {
  return Py_NewRef(b ? Py_True : Py_False);
}
PyObject* toPython(
    const TcfgSite& site,
    uint32_t /*flags*/,
    const at::Scalar& s) {
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
      uint32_t flags,
      int32_t* status) const override {
    std::tuple<typename Unbox<A>::Holder...> holders;
    if (!prepare(site, operands, holders, std::index_sequence_for<A...>{})) {
      *status = TCFG_MISS;
      return nullptr;
    }
    return invoke(site, flags, holders, std::index_sequence_for<A...>{});
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
      uint32_t flags,
      std::tuple<typename Unbox<A>::Holder...>& holders,
      std::index_sequence<I...> /*unused*/) const {
    return toPython(
        site,
        flags,
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

// Whether a kernel argument is a tensor or a list with a tensor in it, such as
// the Tensor?[] indices of aten::index.
bool holdsTensor(const c10::IValue& v) {
  if (v.isTensor()) {
    return true;
  }
  if (v.isList()) {
    for (const auto& e : v.toListRef()) {
      if (e.isTensor()) {
        return true;
      }
    }
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
    if (holdsTensor(call.inputs[a])) {
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
// activation takes its own stack. The stacks are kept to reuse their storage,
// in a deque so that a nested activation does not move the ones below it.
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
  static thread_local std::deque<torch::jit::Stack> pool;
  static thread_local size_t depth;
};
thread_local std::deque<torch::jit::Stack> StackLease::pool;
thread_local size_t StackLease::depth = 0;

bool pushOperand(torch::jit::Stack& stack, PyObject* o, Conv conv) {
  switch (conv) {
    case Conv::Tensor:
      if (isUnboxed(o)) {
        stack.emplace_back(at::Tensor(TensorPtr::reclaim_copy(unboxedImpl(o))));
      } else {
        stack.emplace_back(THPVariable_Unpack(o));
      }
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
        if (!PyLong_CheckExact(item)) {
          return false;
        }
        int overflow = 0;
        const long long x = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0) {
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

PyObject* siteCall(
    TcfgSite* site,
    PyObject* const* operands,
    uint32_t flags,
    int32_t* status) {
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
        // An unboxed tensor is one that would be a plain Tensor if it had a
        // Python object.
        PyTypeObject* type = isUnboxed(o)
            ? reinterpret_cast<PyTypeObject*>(THPVariableClass)
            : Py_TYPE(o);
        if (type != g.type || type->tp_version_tag != g.type_version) {
          return nullptr;
        }
        const c10::TensorImpl* impl = isUnboxed(o)
            ? unboxedImpl(o)
            : THPVariable_Unpack(o).unsafeGetTensorImpl();
        if (impl->key_set().raw_repr() != g.key_set) {
          return nullptr;
        }
        // A tensor without autograd metadata neither requires grad nor
        // carries a forward gradient.
        if (skips_autograd && impl->autograd_meta() != nullptr) {
          if (grad_enabled && impl->requires_grad()) {
            return nullptr;
          }
          const bool fw_grad = isUnboxed(o)
              ? torch::autograd::isFwGradDefined(
                    BorrowedTensor(unboxedImpl(o)).tensor)
              : torch::autograd::isFwGradDefined(THPVariable_Unpack(o));
          if (fw_grad) {
            return nullptr;
          }
        }
        break;
      }
      case OperandGuard::ExactType:
        if (isUnboxed(o) || Py_TYPE(o) != g.type) {
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
    PyObject* result = site->typed->call(*site, operands, flags, status);
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
      result = tensorResult(std::move(out).toTensor(), flags);
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

const char* retName(Ret ret) {
  switch (ret) {
    case Ret::Tensor:
      return "tensor";
    case Ret::Bool:
      return "bool";
    case Ret::Int:
      return "int";
    case Ret::Double:
      return "float";
  }
  return "tensor";
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
  info["returns"] = retName(site->ret);
  info["writes"] = schema.is_mutable();
  info["random"] = site->op.hasTag(at::Tag::nondeterministic_seeded);
  return info.release().ptr();
  END_HANDLE_TH_ERRORS
}

const char* convName(Conv conv) {
  switch (conv) {
    case Conv::Tensor:
      return "tensor";
    case Conv::Bool:
      return "bool";
    case Conv::Int:
      return "int";
    case Conv::IntAsDouble:
      return "int_as_double";
    case Conv::IntAsList:
      return "int_as_list";
    case Conv::Double:
      return "double";
    case Conv::WrappedNumber:
      return "wrapped_number";
    case Conv::IntSeq:
      return "int_seq";
  }
  return "tensor";
}

const char* guardName(OperandGuard::Kind kind) {
  switch (kind) {
    case OperandGuard::None:
      return "none";
    case OperandGuard::Tensor:
      return "tensor";
    case OperandGuard::ExactType:
      return "exact";
    case OperandGuard::IsNone:
      return "is_none";
    case OperandGuard::Identity:
      return "identity";
  }
  return "none";
}

// A constant of a site as the object an operator called from Python is given
// for it: the number a wrapped-number tensor was made of, and the dtype,
// layout or memory format that a kernel takes as an int.
py::object constantObject(const c10::Argument& argument, const c10::IValue& v) {
  if (v.isTensor()) {
    const auto& t = v.toTensor();
    if (t.defined() && t.dim() == 0 &&
        t.unsafeGetTensorImpl()->is_wrapped_number()) {
      return torch::jit::toPyObject(c10::IValue(t.item()));
    }
  }
  if (v.isInt()) {
    c10::TypePtr type = argument.real_type();
    if (const auto optional = type->cast<c10::OptionalType>()) {
      type = optional->getElementType();
    }
    const auto value = v.toInt();
    switch (type->kind()) {
      case c10::TypeKind::ScalarTypeType:
        return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(
            torch::getTHPDtype(static_cast<at::ScalarType>(value))));
      case c10::TypeKind::LayoutType:
        return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(
            torch::getTHPLayout(static_cast<at::Layout>(value))));
      case c10::TypeKind::MemoryFormatType:
        return py::reinterpret_borrow<py::object>(
            torch::utils::getTHPMemoryFormat(
                static_cast<at::MemoryFormat>(value)));
      default:
        break;
    }
  }
  return torch::jit::toPyObject(v);
}

PyObject* siteExport(const TcfgSite* site) {
  HANDLE_TH_ERRORS
  const auto& schema = site->op.schema();
  py::dict info;
  info["op"] = site->op.operator_name().name;
  info["overload"] = site->op.operator_name().overload_name;
  py::list args;
  for (size_t a = 0; a < site->args.size(); a++) {
    const ArgTemplate& arg = site->args[a];
    const auto& argument = schema.arguments()[a];
    py::object from;
    py::object value;
    switch (arg.kind) {
      case ArgTemplate::Operand:
        from = py::str("operand");
        value = py::make_tuple(arg.operand, convName(arg.conv));
        break;
      case ArgTemplate::PackedInts: {
        py::list packed;
        for (int32_t k : arg.packed) {
          packed.append(k);
        }
        from = py::str("packed");
        value = std::move(packed);
        break;
      }
      default:
        from = py::str("const");
        value = constantObject(argument, arg.constant);
        break;
    }
    args.append(
        py::make_tuple(argument.name(), argument.kwarg_only(), from, value));
  }
  info["args"] = args;
  py::list guards;
  for (const auto& g : site->guards) {
    guards.append(guardName(g.kind));
  }
  info["guards"] = guards;
  info["returns"] = retName(site->ret);
  info["writes"] = schema.is_mutable();
  info["random"] = site->op.hasTag(at::Tag::nondeterministic_seeded);
  return info.release().ptr();
  END_HANDLE_TH_ERRORS
}

int32_t traceCalls(const TcfgTrace* trace) {
  return trace->top_level + trace->nested_top_level;
}

void dispatchKeys(uint64_t* included, uint64_t* excluded) {
  const auto tls = c10::impl::tls_local_dispatch_key_set();
  *included = tls.included_.raw_repr();
  *excluded = tls.excluded_.raw_repr();
}

int32_t tensorInfo(PyObject* obj, TcfgTensorInfo* out) {
  const c10::TensorImpl* impl = nullptr;
  if (isUnboxed(obj)) {
    impl = unboxedImpl(obj);
  } else if (THPVariable_CheckExact(obj) && THPVariable_Unpack(obj).defined()) {
    impl = THPVariable_Unpack(obj).unsafeGetTensorImpl();
  } else {
    return 0;
  }
  if (impl->has_symbolic_sizes_strides()) {
    return 0;
  }
  const auto sizes = impl->sizes();
  out->ndim = static_cast<int32_t>(sizes.size());
  out->sizes = sizes.data();
  out->requires_grad = impl->requires_grad() ? 1 : 0;
  out->key_set = impl->key_set().raw_repr();
  // Not every kind of tensor has strides, and those that do not throw.
  try {
    out->strides = impl->strides().data();
    out->storage_offset = impl->storage_offset();
  } catch (...) {
    out->strides = nullptr;
    out->storage_offset = 0;
  }
  // Where the data is, not the data. Asked for its data, the storage of a
  // tensor that shares its memory with a lazy clone (every tensor a graph
  // has built in) first looks at the stream it would be read on, if it is
  // on a GPU: that is paid per tensor, on every call.
  out->data = nullptr;
  if (impl->has_storage()) {
    auto* storage = impl->unsafe_storage().unsafeGetStorageImpl();
    out->data = storage->_mutable_data_ptr_no_checks().get();
  }
  const auto& version = impl->version_counter();
  out->version =
      version.enabled() ? version.current_version() : TCFG_NO_VERSION;
  out->dtype = static_cast<int32_t>(impl->dtype().toScalarType());
  const auto device = impl->device();
  out->device = (static_cast<int32_t>(device.type()) << 16) |
      (static_cast<int32_t>(device.index()) + 1);
  return 1;
}

int32_t tensorForwardGrad(PyObject* obj) {
  if (isUnboxed(obj)) {
    auto* impl = unboxedImpl(obj);
    return impl->autograd_meta() != nullptr &&
            torch::autograd::isFwGradDefined(BorrowedTensor(impl).tensor)
        ? 1
        : 0;
  }
  if (!THPVariable_Check(obj)) {
    return 0;
  }
  const auto& t = THPVariable_Unpack(obj);
  return t.defined() && t.unsafeGetTensorImpl()->autograd_meta() != nullptr &&
          torch::autograd::isFwGradDefined(t)
      ? 1
      : 0;
}

PyObject* tensorBox(PyObject* tagged) {
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(at::Tensor(TensorPtr::reclaim(unboxedImpl(tagged))));
  END_HANDLE_TH_ERRORS
}

void tensorRetain(PyObject* tagged) {
  c10::raw::intrusive_ptr::incref(unboxedImpl(tagged));
}

void tensorRelease(PyObject* tagged) {
  c10::raw::intrusive_ptr::decref(unboxedImpl(tagged));
}

// Frames are for 3.12, and for 3.14 in its default build on a 64-bit
// platform: the variables of its frames are stack references, which the VM
// writes itself (`frame_ref` in vm.rs) and whose header MSVC does not compile.
#if IS_PYTHON_3_12_PLUS && !IS_PYTHON_3_13_PLUS
#define TCFG_HAS_FRAMES 1
#elif IS_PYTHON_3_14_PLUS && !IS_PYTHON_3_15_PLUS &&            \
    !defined(Py_GIL_DISABLED) && !defined(Py_STACKREF_DEBUG) && \
    !defined(_WIN32) && SIZEOF_VOID_P > 4
#define TCFG_HAS_FRAMES 1
#endif

#ifdef TCFG_HAS_FRAMES

extern "C" {
// Copies of CPython internals that torch/csrc/dynamo/cpython_defs.c keeps.
_PyInterpreterFrame* THP_PyThreadState_BumpFramePointerSlow(
    PyThreadState* tstate,
    size_t size);
#if !IS_PYTHON_3_13_PLUS
void THP_PyFrame_Clear(_PyInterpreterFrame* frame);
void THP_PyThreadState_PopFrame(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame);
#endif
}

// Where the thread has its current frame.
_PyInterpreterFrame*& currentFrame(PyThreadState* tstate) {
#if IS_PYTHON_3_14_PLUS
  return tstate->current_frame;
#else
  return tstate->cframe->current_frame;
#endif
}

// The function and the code of a frame.
PyFunctionObject* frameFunction(_PyInterpreterFrame* frame) {
#if IS_PYTHON_3_14_PLUS
  return reinterpret_cast<PyFunctionObject*>(
      PyStackRef_AsPyObjectBorrow(frame->f_funcobj));
#else
  return reinterpret_cast<PyFunctionObject*>(frame->f_funcobj);
#endif
}

PyCodeObject* frameCode(_PyInterpreterFrame* frame) {
#if IS_PYTHON_3_14_PLUS
  return reinterpret_cast<PyCodeObject*>(
      PyStackRef_AsPyObjectBorrow(frame->f_executable));
#else
  return frame->f_code;
#endif
}

// Disposes of a frame that is not, or no longer, the current one of the
// thread, as the interpreter does when a call returns.
void frameDispose(PyThreadState* tstate, _PyInterpreterFrame* frame) {
#if IS_PYTHON_3_13_PLUS
  // What the two copies below do, the interpreter exports from 3.13 on.
  _PyEval_FrameClearAndPop(tstate, frame);
#else
  THP_PyFrame_Clear(frame);
  THP_PyThreadState_PopFrame(tstate, frame);
#endif
}

// A function that the VM runs is a Python call: it counts against the
// recursion limit in the interpreter's own counter, as _Py_EnterRecursivePy
// does, so that the limit means the same depth whoever runs the functions.
bool enterPythonCall(PyThreadState* tstate) {
  if (tstate->py_recursion_remaining-- <= 0) {
    int depth = tstate->py_recursion_limit - tstate->py_recursion_remaining;
    int limit = Py_GetRecursionLimit();
    if (depth < limit) {
      // The limit was raised since the counter was set up.
      tstate->py_recursion_limit = limit;
      tstate->py_recursion_remaining = limit - depth;
    } else {
      ++tstate->py_recursion_remaining;
      PyErr_SetString(PyExc_RecursionError, "maximum recursion depth exceeded");
      return false;
    }
  }
  return true;
}

// Makes a frame that was initialized for a call the current one of the
// thread, with the cells of the function's closure in it.
void frameLink(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame,
    uint16_t*** instr,
    uint16_t** code_start,
    PyObject*** locals,
    PyObject*** object) {
  PyFunctionObject* function = frameFunction(frame);
  PyCodeObject* code = frameCode(frame);
  // What COPY_FREE_VARS does. The other variables are what the caller bound
  // (the arguments, for a frame the interpreter made) or start out unbound:
  // the VM fills them in as the function binds them.
  if (PyObject* closure = function->func_closure) {
    Py_ssize_t n = PyTuple_GET_SIZE(closure);
    Py_ssize_t first = code->co_nlocalsplus - n;
    for (Py_ssize_t i = 0; i < n; i++) {
      PyObject* cell = PyTuple_GET_ITEM(closure, i);
#if IS_PYTHON_3_14_PLUS
      frame->localsplus[first + i] = PyStackRef_FromPyObjectNew(cell);
#else
      frame->localsplus[first + i] = Py_NewRef(cell);
#endif
    }
  }
  // A frame that has not reached its RESUME counts as not started, and is
  // skipped by whoever walks the stack.
  PREV_INSTR(frame) = _PyCode_CODE(code) + code->_co_firsttraceable;
  frame->previous = currentFrame(tstate);
  currentFrame(tstate) = frame;
  *instr = reinterpret_cast<uint16_t**>(&PREV_INSTR(frame));
  *code_start = reinterpret_cast<uint16_t*>(_PyCode_CODE(code));
  *locals = reinterpret_cast<PyObject**>(frame->localsplus);
  *object = reinterpret_cast<PyObject**>(&frame->frame_obj);
}

void* framePush(
    PyObject* func,
    uint16_t*** instr,
    uint16_t** code_start,
    PyObject*** locals,
    PyObject*** object) {
  PyThreadState* tstate = PyThreadState_Get();
  if (!enterPythonCall(tstate)) {
    return nullptr;
  }
  auto* function = reinterpret_cast<PyFunctionObject*>(func);
  auto* code = reinterpret_cast<PyCodeObject*>(function->func_code);
  _PyInterpreterFrame* frame =
      THP_PyThreadState_BumpFramePointerSlow(tstate, code->co_framesize);
  if (frame == nullptr) {
    ++tstate->py_recursion_remaining;
    return nullptr;
  }
  Py_INCREF(func);
#if IS_PYTHON_3_14_PLUS
  _PyFrame_Initialize(
      tstate,
      frame,
      PyStackRef_FromPyObjectSteal(func),
      nullptr,
      code,
      0,
      nullptr);
#else
  _PyFrame_Initialize(frame, function, nullptr, code, 0);
#endif
  frameLink(tstate, frame, instr, code_start, locals, object);
  return frame;
}

int32_t frameAdopt(
    void* made,
    uint16_t*** instr,
    uint16_t** code_start,
    PyObject*** locals,
    PyObject*** object) {
  auto* frame = static_cast<_PyInterpreterFrame*>(made);
  PyThreadState* tstate = PyThreadState_Get();
  if (!enterPythonCall(tstate)) {
    // Whoever is handed a frame to run disposes of it.
    frameDispose(tstate, frame);
    return 0;
  }
  frameLink(tstate, frame, instr, code_start, locals, object);
  return 1;
}

int32_t frameHeld(void* held) {
  auto* frame = static_cast<_PyInterpreterFrame*>(held);
  return frame->frame_obj != nullptr &&
      Py_REFCNT(reinterpret_cast<PyObject*>(frame->frame_obj)) > 1;
}

void framePop(void* popped) {
  auto* frame = static_cast<_PyInterpreterFrame*>(popped);
  PyThreadState* tstate = PyThreadState_Get();
  // Unlinked before it is cleared, as in the interpreter: clearing can run
  // finalizers, which must not find a half-cleared frame on the stack.
  currentFrame(tstate) = frame->previous;
  frameDispose(tstate, frame);
  ++tstate->py_recursion_remaining;
}

void* frameSwap(void* frame) {
  PyThreadState* tstate = PyThreadState_Get();
  _PyInterpreterFrame* previous = currentFrame(tstate);
  currentFrame(tstate) = static_cast<_PyInterpreterFrame*>(frame);
  return previous;
}

#if IS_PYTHON_3_14_PLUS
// The tools of sys.monitoring that stand for a trace function: pdb is the
// first of them from 3.14 on, where it no longer calls sys.settrace.
constexpr uint8_t kTracingTools =
    1 << PY_MONITORING_DEBUGGER_ID | 1 << PY_MONITORING_COVERAGE_ID;
#endif

// Whether something is told about the frames the interpreter enters on this
// thread, and the lines it runs in them.
bool traced(PyThreadState* tstate) {
  if (tstate->c_tracefunc != nullptr) {
    return true;
  }
#if IS_PYTHON_3_14_PLUS
  const auto& starts = tstate->interp->monitors.tools;
  return (starts[PY_MONITORING_EVENT_PY_START] & kTracingTools) != 0;
#else
  return false;
#endif
}

int32_t frameTraced() {
  return traced(PyThreadState_Get());
}

int32_t frameReturns(void* returning, PyObject* result) {
  auto* frame = static_cast<_PyInterpreterFrame*>(returning);
  PyThreadState* tstate = PyThreadState_Get();
  PyFrameObject* object = frame->frame_obj;
  if (object == nullptr || object->f_trace == nullptr || tstate->tracing != 0) {
    return 0;
  }
  if (tstate->c_tracefunc != nullptr) {
    // What the interpreter does for a frame that has a trace function when
    // it returns; the function is not itself traced.
    Py_INCREF(object);
    tstate->tracing++;
    int failed =
        tstate->c_tracefunc(tstate->c_traceobj, object, PyTrace_RETURN, result);
    tstate->tracing--;
    Py_DECREF(object);
    return failed != 0 ? -1 : 0;
  }
#if IS_PYTHON_3_14_PLUS
  // The same for a debugger that listens through sys.monitoring: it asked
  // for the returns of the code of the frames it put its trace function on,
  // and is called as the interpreter calls a tool (call_one_instrument).
  PyCodeObject* code = frameCode(frame);
  uint8_t tools = code->_co_monitoring == nullptr
      ? 0
      : code->_co_monitoring->local_monitors
            .tools[PY_MONITORING_EVENT_PY_RETURN];
  tools &= kTracingTools;
  if (tools == 0) {
    return 0;
  }
  Py_ssize_t at =
      (frame->instr_ptr - _PyCode_CODE(code)) * sizeof(_Py_CODEUNIT);
  PyObject* offset = PyLong_FromSsize_t(at);
  if (offset == nullptr) {
    return -1;
  }
  PyObject* args[] = {reinterpret_cast<PyObject*>(code), offset, result};
  int failed = 0;
  for (int tool = 0; tool < PY_MONITORING_TOOL_IDS && failed == 0; tool++) {
    PyObject* callback =
        tstate->interp
            ->monitoring_callables[tool][PY_MONITORING_EVENT_PY_RETURN];
    if ((tools & (1 << tool)) == 0 || callback == nullptr) {
      continue;
    }
    int outer = tstate->what_event;
    tstate->what_event = PY_MONITORING_EVENT_PY_RETURN;
    tstate->tracing++;
    PyObject* told = PyObject_Vectorcall(callback, args, 3, nullptr);
    tstate->tracing--;
    tstate->what_event = outer;
    failed = told == nullptr ? -1 : 0;
    Py_XDECREF(told);
  }
  Py_DECREF(offset);
  return failed;
#else
  return 0;
#endif
}

// The frame evaluation function of the interpreter (PEP 523) while a
// rewritten function runs. The interpreter has one for all threads, and
// others set theirs too (Dynamo): this one is put in front of what is there
// when the first thread wants it, hands on to that what it does not take,
// and is taken out when the last thread is done, unless another one was put
// in front of it since; then it stays, and hands everything on.
TcfgTakeFrame hook_take = nullptr;
_PyFrameEvalFunction hook_below = nullptr;
bool hook_linked = false;
int32_t hook_wanted = 0;
// The frame this thread is handing on to what is below.
thread_local _PyInterpreterFrame* hook_handing = nullptr;
// How many frames the interpreter brought here on this thread.
thread_local uint64_t hook_frames = 0;
// Of those, the frames that a kernel enters by itself: the router of an
// operator whose kernel is chosen in Python (torch/_native/registry.py) with
// what the router calls, and what shows a warning that a kernel gave. And
// how many such frames are running on this thread.
thread_local uint64_t hook_kernel_frames = 0;
thread_local int32_t hook_in_kernel = 0;

// What the interpreter calls of the warnings module where a warning is given
// from C: what makes the message, and what shows it. Found when the hook is
// registered: looking for them runs Python code.
PyCodeObject* hook_warning_made = nullptr;
PyCodeObject* hook_warning_shown = nullptr;

// The code of a function of a module, found by the names that lead to it;
// null if there is none.
PyCodeObject* codeOf(
    const char* module,
    std::initializer_list<const char*> names) {
  PyObject* found = PyImport_ImportModule(module);
  for (const char* name : names) {
    if (found == nullptr) {
      break;
    }
    PyObject* next = PyObject_GetAttrString(found, name);
    Py_DECREF(found);
    found = next;
  }
  PyObject* code =
      found == nullptr ? nullptr : PyObject_GetAttrString(found, "__code__");
  Py_XDECREF(found);
  if (code == nullptr || !PyCode_Check(code)) {
    PyErr_Clear();
    Py_XDECREF(code);
    return nullptr;
  }
  // Kept for good: it is compared with, never run.
  return reinterpret_cast<PyCodeObject*>(code);
}

// Whether a frame is one that a kernel enters by itself. The dispatcher
// calls the router of an operator (eager_router of torch/_native/registry.py)
// as the operator's kernel, with the operator's arguments, and takes what it
// returns as the operator's result: what it runs is the kernel, and which
// implementation that is it decides at every call. A warning that a kernel
// gives is made and shown by the warnings module, at the call that gives it,
// whatever stands for the operator. Neither is Python code that the caller
// of the operator enters.
bool kernelFrame(PyCodeObject* code) {
  static PyCodeObject* router = nullptr;
  static PyObject* routes = PyUnicode_InternFromString("eager_router");
  if (code == router || code == hook_warning_made ||
      code == hook_warning_shown) {
    return true;
  }
  if (router != nullptr || code->co_name != routes) {
    return false;
  }
  const char* file = PyUnicode_AsUTF8(code->co_filename);
  if (file == nullptr) {
    PyErr_Clear();
    return false;
  }
  const std::string path(file);
  const std::string suffix = "torch/_native/registry.py";
  if (path.size() < suffix.size() ||
      path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return false;
  }
  Py_INCREF(code);
  router = code;
  return true;
}
// The interpreter the hook was registered in: the only one PyTorch runs in.
PyInterpreterState* hook_interp = nullptr;

PyObject* handOn(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame,
    int throwflag) {
  _PyInterpreterFrame* outer = hook_handing;
  hook_handing = frame;
  PyObject* result = hook_below(tstate, frame, throwflag);
  hook_handing = outer;
  return result;
}

PyObject* evalFrame(
    PyThreadState* tstate,
    _PyInterpreterFrame* frame,
    int throwflag) {
  // Evaluators that were put in front of one another out of order form a
  // ring: a frame that comes around is the interpreter's.
  if (frame == hook_handing) {
    return _PyEval_EvalFrameDefault(tstate, frame, throwflag);
  }
  hook_frames++;
  // The frames of a kernel are the interpreter's: the VM is offered none.
  const bool enters = kernelFrame(frameCode(frame));
  if (enters || hook_in_kernel > 0) {
    hook_kernel_frames++;
    hook_in_kernel += enters ? 1 : 0;
    PyObject* result = handOn(tstate, frame, throwflag);
    hook_in_kernel -= enters ? 1 : 0;
    return result;
  }
  // Offered: the frame of a call of a function, before its first
  // instruction. Not the body of a class or a module, which has its
  // variables in a dict; not a generator or a coroutine, whose frame outlives
  // the call; nothing while the thread is traced, when it is the interpreter
  // that tells of the frames it runs; and nothing that is another evaluator's
  // business: while one is in front of this one, or is running a frame that
  // this one handed on to it.
  constexpr int kinds =
      CO_OPTIMIZED | CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR;
  if (hook_wanted > 0 && throwflag == 0 &&
      frame->owner == FRAME_OWNED_BY_THREAD && frame->f_locals == nullptr &&
      (frameCode(frame)->co_flags & kinds) == CO_OPTIMIZED && !traced(tstate) &&
      PyFunction_Check(frameFunction(frame)) &&
      (hook_handing == nullptr || hook_below == &_PyEval_EvalFrameDefault) &&
      _PyInterpreterState_GetEvalFrameFunc(tstate->interp) == &evalFrame) {
    PyObject* result = nullptr;
    PyObject* function = reinterpret_cast<PyObject*>(frameFunction(frame));
    if (hook_take(function, frame, &result)) {
      return result;
    }
  }
  return handOn(tstate, frame, throwflag);
}

void frameHook(TcfgTakeFrame take) {
  hook_take = take;
  hook_interp = PyInterpreterState_Get();
  hook_warning_made = codeOf("warnings", {"WarningMessage", "__init__"});
  hook_warning_shown = codeOf("warnings", {"_showwarnmsg"});
}

void frameWant(int32_t delta) {
  hook_wanted += delta;
  // Nothing is done but for the first thread to want the hook and the last
  // to let go of it.
  if (hook_wanted > 1 || (hook_wanted == 1 && delta < 0) ||
      hook_take == nullptr) {
    return;
  }
  PyInterpreterState* interp = hook_interp;
  _PyFrameEvalFunction current = _PyInterpreterState_GetEvalFrameFunc(interp);
  if (hook_wanted > 0) {
    // Not linked in, or taken out by someone who put the interpreter's own
    // back.
    if (current != &evalFrame &&
        (!hook_linked || current == &_PyEval_EvalFrameDefault)) {
      hook_below = current;
      _PyInterpreterState_SetEvalFrameFunc(interp, &evalFrame);
      hook_linked = true;
    }
  } else if (current == &evalFrame) {
    _PyInterpreterState_SetEvalFrameFunc(interp, hook_below);
    hook_linked = false;
  }
}

PyObject* frameInterpret(void* frame) {
  return handOn(
      PyThreadState_Get(), static_cast<_PyInterpreterFrame*>(frame), 0);
}

uint64_t frameCount() {
  return hook_frames;
}

uint64_t frameCountKernel() {
  return hook_kernel_frames;
}
#endif

int32_t isTracing() {
  return torch::jit::tracer::isTracing();
}

uint32_t gradEnabled() {
  return c10::GradMode::is_enabled() ? 1 : 0;
}

uint64_t callModes() {
  uint64_t modes = c10::GradMode::is_enabled() ? 1 : 0;
  modes |= c10::InferenceMode::is_enabled() ? 2 : 0;
  modes |= at::autocast::is_autocast_enabled(at::kCUDA) ? 4 : 0;
  modes |= at::autocast::is_autocast_enabled(at::kCPU) ? 8 : 0;
  modes |= static_cast<uint64_t>(c10::get_default_dtype_as_scalartype()) << 8;
  modes |= static_cast<uint64_t>(at::impl::PythonTorchFunctionTLS::stack_len())
      << 16;
  return modes;
}

void* workspaceEnterRaw(void* workspace) {
  return torch::cfg::workspaceEnter(
      static_cast<torch::cfg::Workspace*>(workspace));
}

void workspaceLeaveRaw(void* workspace, void* previous) {
  torch::cfg::workspaceLeave(
      static_cast<torch::cfg::Workspace*>(workspace),
      static_cast<torch::cfg::Workspace*>(previous));
}

// ---------------------------------------------------------------------------
// The backward of a graph: what the Rust runtime takes the
// kernels of eager's autograd nodes down with, runs two straight-line graphs
// with, and gives the results of a graph one node with.

constexpr const char* kOperatorCapsule = "torch._C._cfg_operator";

// An operator, for the runtime: operators live as long as the dispatcher, and
// one handle per operator is kept for as long.
PyObject* operatorObject(const c10::OperatorHandle& op) {
  static std::mutex lock;
  static std::unordered_map<std::string, std::unique_ptr<c10::OperatorHandle>>
      known;
  std::lock_guard<std::mutex> guard(lock);
  auto& held = known[toString(op.operator_name())];
  if (!held) {
    held = std::make_unique<c10::OperatorHandle>(op);
  }
  return PyCapsule_New(held.get(), kOperatorCapsule, nullptr);
}

// What names an operator: what operatorObject gives, or an overload of
// torch.ops (the graphs a dispatch mode took down name operators so).
c10::OperatorHandle operatorOf(PyObject* obj) {
  if (PyCapsule_IsValid(obj, kOperatorCapsule) != 0) {
    return *static_cast<const c10::OperatorHandle*>(
        PyCapsule_GetPointer(obj, kOperatorCapsule));
  }
  const auto schema = py::getattr(py::handle(obj), "_schema");
  const auto name = py::cast<std::string>(schema.attr("name"));
  const auto overload = py::cast<std::string>(schema.attr("overload_name"));
  auto op = c10::Dispatcher::singleton().findSchema({name, overload});
  TORCH_CHECK(op.has_value(), "torch.vm: no operator ", name, ".", overload);
  return *op;
}

struct TapeRecord {
  c10::OperatorHandle op;
  std::vector<c10::IValue> inputs;
  std::vector<c10::IValue> outputs;
};

// Takes down, of every call of the dispatcher, the first kernel below the
// autograd keys. What that kernel calls is its own business.
class TapeSink final : public c10::DispatchCaptureSink {
 public:
  void onKernelEnter(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet ks,
      bool isRedispatch,
      c10::ArrayRef<const c10::IValue> inputs) override {
    if (previous != nullptr) {
      previous->onKernelEnter(op, ks, isRedispatch, inputs);
    }
    depth++;
    const auto key = static_cast<uint16_t>(ks.highestFunctionalityKey());
    if (taken_at < 0 &&
        key < static_cast<uint16_t>(c10::DispatchKey::ADInplaceOrView)) {
      taken_at = depth;
      TapeRecord record{op, {}, {}};
      record.inputs.assign(inputs.begin(), inputs.end());
      records.push_back(std::move(record));
    }
  }

  void onKernelExit(
      const c10::OperatorHandle& op,
      c10::ArrayRef<const c10::IValue> outputs,
      bool kernel_threw) override {
    if (depth == taken_at) {
      taken_at = -1;
      if (kernel_threw) {
        threw = true;
      } else {
        records.back().outputs.assign(outputs.begin(), outputs.end());
      }
    }
    depth--;
    if (previous != nullptr) {
      previous->onKernelExit(op, outputs, kernel_threw);
    }
  }

  std::vector<TapeRecord> records;
  int32_t depth = 0;
  int32_t taken_at = -1;
  bool threw = false;
  c10::DispatchCaptureSink* previous = nullptr;
};

thread_local std::vector<std::unique_ptr<TapeSink>> tls_tapes;

void tapeBegin() {
  auto tape = std::make_unique<TapeSink>();
  tape->previous = c10::setDispatchCaptureSink(tape.get());
  tls_tapes.push_back(std::move(tape));
}

PyObject* tapeEnd() {
  HANDLE_TH_ERRORS
  TORCH_CHECK(!tls_tapes.empty(), "torch.vm: no tape is being taken down");
  auto tape = std::move(tls_tapes.back());
  tls_tapes.pop_back();
  c10::setDispatchCaptureSink(tape->previous);
  TORCH_CHECK(
      !tape->threw, "torch.vm: a kernel raised while a tape was taken down");
  py::list out;
  for (auto& r : tape->records) {
    TORCH_CHECK(
        r.op.hasSchema(),
        "torch.vm: an operator without a schema: ",
        toString(r.op.operator_name()));
    const auto& arguments = r.op.schema().arguments();
    py::tuple args(r.inputs.size());
    py::list written;
    for (size_t i = 0; i < r.inputs.size(); ++i) {
      // As an operator called from Python is given it: a number for the
      // tensor a number was wrapped in, a dtype for the int of one.
      args[i] = i < arguments.size()
          ? constantObject(arguments[i], r.inputs[i])
          : torch::jit::toPyObject(std::move(r.inputs[i]));
      const auto* alias =
          i < arguments.size() ? arguments[i].alias_info() : nullptr;
      if (alias != nullptr && alias->isWrite()) {
        written.append(i);
      }
    }
    py::object result = py::none();
    if (r.outputs.size() == 1) {
      result = torch::jit::toPyObject(std::move(r.outputs[0]));
    } else if (!r.outputs.empty()) {
      py::tuple several(r.outputs.size());
      for (size_t i = 0; i < r.outputs.size(); ++i) {
        several[i] = torch::jit::toPyObject(std::move(r.outputs[i]));
      }
      result = std::move(several);
    }
    const auto& returns = r.op.schema().returns();
    const auto* aliases = returns.empty() ? nullptr : returns[0].alias_info();
    out.append(py::make_tuple(
        py::reinterpret_steal<py::object>(operatorObject(r.op)),
        toString(r.op.operator_name()),
        std::move(args),
        std::move(result),
        std::move(written),
        r.op.hasTag(at::Tag::nondeterministic_seeded),
        aliases != nullptr && !aliases->isWrite()));
  }
  return out.release().ptr();
  END_HANDLE_TH_ERRORS
}

// The node behind its Python object; null for one written in Python.
torch::autograd::Node* nodePtr(PyObject* obj) {
  if (torch::autograd::THPCppFunction_Check(obj)) {
    return reinterpret_cast<torch::autograd::THPCppFunction*>(obj)->cdata.get();
  }
  return nullptr;
}

PyObject* nodeOf(PyObject* tensor, uint32_t* input_nr) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(tensor), "torch.vm: node_of takes a tensor");
  const auto& t = THPVariable_Unpack(tensor);
  if (!t.defined() || !t.requires_grad()) {
    return nullptr;
  }
  const auto edge = torch::autograd::impl::gradient_edge(t);
  if (!edge.is_valid()) {
    return nullptr;
  }
  *input_nr = edge.input_nr;
  return torch::autograd::functionToPyObject(edge.function);
  END_HANDLE_TH_ERRORS
}

int32_t nodeInfo(PyObject* obj, TcfgNodeInfo* out) {
  static thread_local std::string name;
  if (THPFunction_Check(obj)) {
    name = "a node written in Python";
    *out = TcfgNodeInfo{name.c_str(), 0, 0, 0, TCFG_NODE_PYTHON};
    return 1;
  }
  auto* node = nodePtr(obj);
  if (node == nullptr) {
    return 0;
  }
  name = node->name();
  uint32_t flags = 0;
  if (dynamic_cast<torch::autograd::AccumulateGrad*>(node) != nullptr) {
    flags |= TCFG_NODE_LEAF;
  }
  *out = TcfgNodeInfo{
      name.c_str(),
      node->sequence_nr(),
      node->num_inputs(),
      static_cast<uint32_t>(node->next_edges().size()),
      flags};
  return 1;
}

PyObject* nodeNext(PyObject* obj, uint32_t edge, uint32_t* input_nr) {
  HANDLE_TH_ERRORS
  auto* node = nodePtr(obj);
  TORCH_CHECK(
      node != nullptr && edge < node->next_edges().size(),
      "torch.vm: node_next: no such edge");
  const auto& next = node->next_edges()[edge];
  if (!next.is_valid()) {
    return nullptr;
  }
  *input_nr = next.input_nr;
  return torch::autograd::functionToPyObject(next.function);
  END_HANDLE_TH_ERRORS
}

// A thread that calls eager's nodes for a thread on which a backward is
// running: one per such thread, kept, so that the nodes it makes (where a
// backward is differentiated again) are numbered on with those of the
// thread it works for. The autograd engine runs nodes by those numbers.
class NodeCaller {
 public:
  NodeCaller() : thread_([this] { loop(); }) {}

  ~NodeCaller() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
  }

  // Runs work on the thread and waits for it.
  void run(const std::function<void()>& work) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_ = &work;
    wake_.notify_all();
    wake_.wait(lock, [this] { return work_ == nullptr; });
  }

 private:
  void loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      wake_.wait(lock, [this] { return stop_ || work_ != nullptr; });
      if (stop_) {
        return;
      }
      (*work_)();
      work_ = nullptr;
      wake_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable wake_;
  const std::function<void()>* work_ = nullptr;
  bool stop_ = false;
  std::thread thread_;
};

thread_local std::unique_ptr<NodeCaller> tls_node_caller;

// Brings the numbering of the nodes of this thread up to a number.
void numberFrom(uint64_t number) {
  while (at::sequence_number::peek() < number) {
    at::sequence_number::get_and_increment();
  }
}

PyObject* nodeCall(PyObject* obj, PyObject* grads, uint32_t flags) {
  HANDLE_TH_ERRORS
  auto* raw = nodePtr(obj);
  TORCH_CHECK(
      raw != nullptr && PyTuple_Check(grads),
      "torch.vm: node_call takes a node of eager's and a tuple of gradients");
  const auto node = raw->getptr();
  const auto n = static_cast<size_t>(PyTuple_GET_SIZE(grads));
  TORCH_CHECK(
      n == node->num_inputs(),
      "torch.vm: node_call: one gradient per input of the node");
  torch::autograd::variable_list inputs;
  inputs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    PyObject* item = PyTuple_GET_ITEM(grads, i);
    if (item == Py_None) {
      inputs.emplace_back();
    } else {
      TORCH_CHECK(THPVariable_Check(item), "torch.vm: a gradient is a tensor");
      inputs.push_back(THPVariable_Unpack(item));
    }
  }
  torch::autograd::variable_list outputs;
  const bool on_tape = (flags & TCFG_RUN_ON_TAPE) != 0;
  const auto run = [&]() {
    at::AutoGradMode mode(on_tape);
    // As the autograd engine evaluates a node: on the stream its forward
    // ran on. A thread of its own starts on the default stream, which is
    // not the one the tensors of the caller are computed on.
    const c10::OptionalStreamGuard stream_guard{node->stream()};
    outputs = (*node)(std::move(inputs));
    torch::autograd::validate_outputs(
        node->next_edges(), outputs, [&](const std::string& msg) {
          return "Function " + node->name() + " returned an " + msg;
        });
  };
  const auto* wanted = torch::autograd::get_current_graph_task_exec_info();
  if (wanted == nullptr || wanted->empty()) {
    pybind11::gil_scoped_release no_gil;
    run();
  } else {
    // A backward for some inputs only is running on this thread. A node
    // asks the backward of its thread which of its gradients are wanted
    // (task_should_compute_output) and computes none that one does not
    // know of: the node is called on a thread where no backward runs, as
    // this thread is otherwise.
    const at::ThreadLocalState state;
    TapeSink* tape = tls_tapes.empty() ? nullptr : tls_tapes.back().get();
    std::exception_ptr failed;
    if (!tls_node_caller) {
      tls_node_caller = std::make_unique<NodeCaller>();
    }
    const uint64_t number = at::sequence_number::peek();
    uint64_t reached = number;
    {
      pybind11::gil_scoped_release no_gil;
      tls_node_caller->run([&]() {
        const at::ThreadLocalStateGuard guard(state);
        numberFrom(number);
        c10::DispatchCaptureSink* before =
            tape != nullptr ? c10::setDispatchCaptureSink(tape) : nullptr;
        try {
          run();
        } catch (...) {
          failed = std::current_exception();
        }
        if (tape != nullptr) {
          c10::setDispatchCaptureSink(before);
        }
        reached = at::sequence_number::peek();
      });
    }
    numberFrom(reached);
    if (failed) {
      std::rethrow_exception(failed);
    }
  }
  py::tuple out(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].defined()) {
      out[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(outputs[i]));
    } else {
      out[i] = py::none();
    }
  }
  return out.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* opFind(const char* name, const char* overload) {
  HANDLE_TH_ERRORS
  auto op = c10::Dispatcher::singleton().findSchema({name, overload});
  TORCH_CHECK(op.has_value(), "torch.vm: no operator ", name, ".", overload);
  return operatorObject(*op);
  END_HANDLE_TH_ERRORS
}

// An argument of an operator, of the object the tape gave for it: a number
// where a tensor is taken is the tensor the number was wrapped in.
c10::IValue argumentValue(
    const py::handle& obj,
    const c10::Argument& argument) {
  c10::TypePtr type = argument.type();
  if (const auto optional = type->cast<c10::OptionalType>()) {
    type = optional->getElementType();
  }
  if (type->kind() == c10::TypeKind::TensorType && !obj.is_none() &&
      !THPVariable_Check(obj.ptr())) {
    if (PyBool_Check(obj.ptr())) {
      return wrapNumber(at::Scalar(obj.ptr() == Py_True));
    }
    if (PyLong_Check(obj.ptr())) {
      return wrapNumber(at::Scalar(py::cast<int64_t>(obj)));
    }
    if (PyFloat_Check(obj.ptr())) {
      return wrapNumber(at::Scalar(py::cast<double>(obj)));
    }
    if (PyComplex_Check(obj.ptr())) {
      return wrapNumber(at::Scalar(c10::complex<double>(
          PyComplex_RealAsDouble(obj.ptr()),
          PyComplex_ImagAsDouble(obj.ptr()))));
    }
  }
  return torch::jit::toIValue(obj, argument.type(), argument.N());
}

// Whether the autograd kernel of an operator runs what is below it with the
// key that counts writes and makes views excluded too: it does for an
// operator that neither writes to an argument nor gives a view (no kernel
// for that key), so that what its kernel calls in turn leaves the version
// of a tensor alone. An operator that is made of other operators has no
// autograd kernel, and what it calls runs as it is called.
bool belowViews(const c10::OperatorHandle& op) {
  return op.schema().operator_name().name.rfind("aten::", 0) == 0 &&
      !op.hasKernelForDispatchKey(c10::DispatchKey::ADInplaceOrView) &&
      !op.hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd);
}

// How kernels run for the runtime: below the autograd keys (the grad mode
// stays what it is: a view made there is a view like one of eager's), on the
// tape with gradients enabled, or, without either flag, as the thread is.
struct RunGuards {
  explicit RunGuards(uint32_t flags) {
    if ((flags & TCFG_RUN_ON_TAPE) != 0) {
      grad.emplace(true);
    }
    if ((flags & TCFG_RUN_BELOW_AUTOGRAD) != 0) {
      below.emplace();
      // What the tape gave is what autocast made of an operation: it is
      // not cast again.
      uncast.emplace(c10::autocast_dispatch_keyset);
    }
  }
  std::optional<at::AutoGradMode> grad;
  std::optional<at::AutoDispatchBelowAutograd> below;
  std::optional<c10::impl::ExcludeDispatchKeyGuard> uncast;
};

PyObject* stackResult(torch::jit::Stack& stack) {
  if (stack.empty()) {
    Py_RETURN_NONE;
  }
  if (stack.size() == 1) {
    return torch::jit::toPyObject(std::move(stack[0])).release().ptr();
  }
  py::tuple out(stack.size());
  for (size_t i = 0; i < stack.size(); ++i) {
    out[i] = torch::jit::toPyObject(std::move(stack[i]));
  }
  return out.release().ptr();
}

PyObject* opCall(PyObject* obj, PyObject* args, uint32_t flags) {
  HANDLE_TH_ERRORS
  const auto op = operatorOf(obj);
  const auto& arguments = op.schema().arguments();
  TORCH_CHECK(
      PyTuple_Check(args) &&
          static_cast<size_t>(PyTuple_GET_SIZE(args)) == arguments.size(),
      "torch.vm: op_call takes the arguments of the schema, in order");
  torch::jit::Stack stack;
  stack.reserve(arguments.size());
  for (size_t i = 0; i < arguments.size(); ++i) {
    stack.push_back(argumentValue(PyTuple_GET_ITEM(args, i), arguments[i]));
  }
  {
    pybind11::gil_scoped_release no_gil;
    const RunGuards guards(flags);
    if ((flags & TCFG_RUN_BELOW_AUTOGRAD) != 0 && belowViews(op)) {
      const at::AutoDispatchBelowADInplaceOrView below;
      op.callBoxed(stack);
    } else {
      op.callBoxed(stack);
    }
  }
  return stackResult(stack);
  END_HANDLE_TH_ERRORS
}

// An argument of an operation of a program: a value of the program, a
// constant, or a list of tensors that are values of the program.
struct ProgramArg {
  int32_t value = -1;
  c10::IValue constant;
  bool several = false;
  bool optional_items = false;
  std::vector<int32_t> items;
};

struct ProgramStep {
  // The operator; none for a call of a Python function that makes a tensor
  // of what is no tensor (torch.tensor of numbers), which is called with
  // the arguments it was described with.
  std::optional<c10::OperatorHandle> op;
  py::object callable;
  py::object call_args;
  py::object call_kwargs;
  std::vector<ProgramArg> args;
  // Where what the operator returned goes: (place, value).
  std::vector<std::pair<uint32_t, uint32_t>> gives;
  bool single = false;
  // belowViews of the operator.
  bool below_views = false;
};

struct Program {
  std::vector<ProgramStep> steps;
  uint32_t n_values = 0;
};

// Whether an argument, as the joint graph describes one, names a value.
bool namesValue(const py::handle& described) {
  const auto pair = py::cast<py::tuple>(described);
  const auto kind = py::cast<std::string>(pair[0]);
  if (kind == "v") {
    return true;
  }
  if (kind == "c") {
    return false;
  }
  for (const auto& item : pair[1]) {
    if (namesValue(item)) {
      return true;
    }
  }
  return false;
}

// The Python object an argument without values describes.
py::object constantOf(const py::handle& described) {
  const auto pair = py::cast<py::tuple>(described);
  const auto kind = py::cast<std::string>(pair[0]);
  if (kind == "c") {
    return py::reinterpret_borrow<py::object>(pair[1]);
  }
  py::list items;
  for (const auto& item : pair[1]) {
    items.append(constantOf(item));
  }
  if (kind == "t") {
    return py::tuple(items);
  }
  return items;
}

ProgramArg programArg(
    const py::handle& described,
    const c10::Argument& argument,
    uint32_t n_values) {
  ProgramArg arg;
  if (!namesValue(described)) {
    arg.constant = argumentValue(constantOf(described), argument);
    return arg;
  }
  const auto pair = py::cast<py::tuple>(described);
  if (py::cast<std::string>(pair[0]) == "v") {
    arg.value = py::cast<int32_t>(pair[1]);
    TORCH_CHECK(
        arg.value >= 0 && static_cast<uint32_t>(arg.value) < n_values,
        "a value that the program does not have");
    return arg;
  }
  arg.several = true;
  arg.optional_items =
      argument.type()->isSubtypeOf(*c10::ListType::ofOptionalTensors()) &&
      !argument.type()->isSubtypeOf(*c10::ListType::ofTensors());
  for (const auto& item : pair[1]) {
    const auto inner = py::cast<py::tuple>(item);
    const auto kind = py::cast<std::string>(inner[0]);
    if (kind == "v") {
      arg.items.push_back(py::cast<int32_t>(inner[1]));
    } else {
      TORCH_CHECK(
          kind == "c" && inner[1].is_none() && arg.optional_items,
          "a list of tensors with what is no tensor in it");
      arg.items.push_back(-1);
    }
  }
  return arg;
}

void* programNew(
    PyObject* ops,
    uint32_t n_values,
    char* reason,
    size_t reason_len) {
  auto program = std::make_unique<Program>();
  program->n_values = n_values;
  try {
    for (const auto& described : py::handle(ops)) {
      const auto parts = py::cast<py::tuple>(described);
      ProgramStep step;
      const auto args = py::cast<py::sequence>(parts[1]);
      const auto kwargs = py::cast<py::dict>(parts[2]);
      for (const auto& given : parts[3]) {
        const auto pair = py::cast<py::tuple>(given);
        const auto value = py::cast<uint32_t>(pair[1]);
        TORCH_CHECK(value < n_values, "a value that the program does not have");
        step.gives.emplace_back(py::cast<uint32_t>(pair[0]), value);
      }
      step.single = py::cast<bool>(parts[4]);
      PyObject* named = parts[0].ptr();
      if (PyCapsule_IsValid(named, kOperatorCapsule) == 0 &&
          PyObject_HasAttrString(named, "_schema") == 0) {
        py::list positional;
        for (const auto& arg : args) {
          TORCH_CHECK(!namesValue(arg), "a call of a function with a tensor");
          positional.append(constantOf(arg));
        }
        py::dict by_name;
        for (const auto& item : kwargs) {
          TORCH_CHECK(
              !namesValue(item.second), "a call of a function with a tensor");
          by_name[item.first] = constantOf(item.second);
        }
        step.callable = py::reinterpret_borrow<py::object>(named);
        step.call_args = py::tuple(positional);
        step.call_kwargs = std::move(by_name);
        program->steps.push_back(std::move(step));
        continue;
      }
      step.op = operatorOf(named);
      step.below_views = belowViews(*step.op);
      const auto& arguments = step.op->schema().arguments();
      TORCH_CHECK(py::len(args) <= arguments.size(), "too many arguments");
      for (size_t i = 0; i < arguments.size(); ++i) {
        if (i < py::len(args)) {
          step.args.push_back(programArg(args[i], arguments[i], n_values));
        } else if (kwargs.contains(arguments[i].name())) {
          step.args.push_back(programArg(
              kwargs[arguments[i].name().c_str()], arguments[i], n_values));
        } else {
          TORCH_CHECK(
              arguments[i].default_value().has_value(),
              "no argument ",
              arguments[i].name());
          ProgramArg arg;
          arg.constant = *arguments[i].default_value();
          step.args.push_back(std::move(arg));
        }
      }
      program->steps.push_back(std::move(step));
    }
  } catch (const std::exception& error) {
    PyErr_Clear();
    std::snprintf(reason, reason_len, "%s", error.what());
    return nullptr;
  }
  return program.release();
}

void programFree(void* program) {
  delete static_cast<Program*>(program);
}

PyObject* programRun(
    void* raw,
    PyObject* const* inputs,
    const uint32_t* input_values,
    uint32_t n_inputs,
    const uint32_t* output_values,
    uint32_t n_outputs,
    uint32_t flags) {
  HANDLE_TH_ERRORS
  const auto& program = *static_cast<const Program*>(raw);
  std::vector<c10::IValue> values(program.n_values);
  for (uint32_t i = 0; i < n_inputs; ++i) {
    TORCH_CHECK(
        THPVariable_Check(inputs[i]) && input_values[i] < program.n_values,
        "torch.vm: program_run takes tensors for values of the program");
    values[input_values[i]] = THPVariable_Unpack(inputs[i]);
  }
  {
    pybind11::gil_scoped_release no_gil;
    const RunGuards guards(flags);
    torch::jit::Stack stack;
    for (const auto& step : program.steps) {
      if (!step.op.has_value()) {
        const pybind11::gil_scoped_acquire gil;
        const py::object given =
            step.callable(*step.call_args, **step.call_kwargs);
        for (const auto& [place, value] : step.gives) {
          const py::object item = step.single
              ? given
              : py::reinterpret_borrow<py::object>(given[py::int_(place)]);
          TORCH_CHECK(
              THPVariable_Check(item.ptr()),
              "torch.vm: what makes a tensor of numbers gave no tensor");
          values[value] = THPVariable_Unpack(item.ptr());
        }
        continue;
      }
      stack.clear();
      for (const auto& arg : step.args) {
        if (arg.value >= 0) {
          stack.push_back(values[arg.value]);
        } else if (!arg.several) {
          stack.push_back(arg.constant);
        } else if (arg.optional_items) {
          c10::List<std::optional<at::Tensor>> list;
          for (const auto item : arg.items) {
            list.push_back(
                item < 0 ? std::optional<at::Tensor>()
                         : std::optional<at::Tensor>(values[item].toTensor()));
          }
          stack.emplace_back(std::move(list));
        } else {
          c10::List<at::Tensor> list;
          for (const auto item : arg.items) {
            list.push_back(values[item].toTensor());
          }
          stack.emplace_back(std::move(list));
        }
      }
      if ((flags & TCFG_RUN_BELOW_AUTOGRAD) != 0 && step.below_views) {
        const at::AutoDispatchBelowADInplaceOrView below;
        step.op->callBoxed(stack);
      } else {
        step.op->callBoxed(stack);
      }
      for (const auto& [place, value] : step.gives) {
        if (stack.size() == 1 && stack[0].isList() && !step.single) {
          values[value] = stack[0].toListRef().at(place);
        } else {
          values[value] = stack.at(step.single ? 0 : place);
        }
      }
    }
  }
  py::tuple out(n_outputs);
  for (uint32_t i = 0; i < n_outputs; ++i) {
    TORCH_CHECK(
        output_values[i] < program.n_values,
        "torch.vm: program_run: a value that the program does not have");
    out[i] = torch::jit::toPyObject(values[output_values[i]]);
  }
  return out.release().ptr();
  END_HANDLE_TH_ERRORS
}

// The node of a graph: one node for all the results of a graph that need a
// gradient, with an edge per contribution to a gradient of an input. What it
// is handed goes to the runtime, which runs the backward graph.
class GraphBackward final : public torch::autograd::Node {
 public:
  GraphBackward(void* state, TcfgBackward backward, TcfgRelease release)
      : state_(state), backward_(backward), release_(release) {}

  // A node dies where the last tensor it gave dies, and a tensor is let go
  // without the GIL (THPVariable_clear): the runtime is called with it.
  ~GraphBackward() override {
    const pybind11::gil_scoped_acquire gil;
    release_(state_, 1);
  }

  std::string name() const override {
    return "GraphBackward";
  }

  void release_variables() override {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!freed_) {
      freed_ = true;
      const pybind11::gil_scoped_acquire gil;
      release_(state_, 0);
    }
  }

 protected:
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(!freed_, torch::autograd::ERR_BACKWARD_TWICE);
    const pybind11::gil_scoped_acquire gil;
    std::vector<py::object> held;
    std::vector<PyObject*> handed;
    for (const auto& grad : grads) {
      if (grad.defined()) {
        held.push_back(
            py::reinterpret_steal<py::object>(THPVariable_Wrap(grad)));
        handed.push_back(held.back().ptr());
      } else {
        handed.push_back(nullptr);
      }
    }
    std::vector<PyObject*> given(num_outputs(), nullptr);
    std::vector<uint8_t> needed(given.size());
    for (size_t i = 0; i < needed.size(); ++i) {
      needed[i] = task_should_compute_output(i) ? 1 : 0;
    }
    const uint32_t flags =
        c10::GradMode::is_enabled() ? uint32_t{TCFG_RUN_ON_TAPE} : 0U;
    const auto status = backward_(
        state_,
        handed.data(),
        static_cast<uint32_t>(handed.size()),
        given.data(),
        static_cast<uint32_t>(given.size()),
        needed.data(),
        flags);
    torch::autograd::variable_list outputs(given.size());
    for (size_t i = 0; i < given.size(); ++i) {
      if (given[i] != nullptr) {
        if (status == 0 && THPVariable_Check(given[i])) {
          outputs[i] = THPVariable_Unpack(given[i]);
        }
        Py_DECREF(given[i]);
      }
    }
    if (status != 0) {
      torch::throw_python_error();
    }
    return outputs;
  }

 private:
  void* state_;
  TcfgBackward backward_;
  TcfgRelease release_;
  bool freed_ = false;
};

// What a root of a graph has as its gradient function: it hands the gradient
// that reaches the root on to the node of the graph, to the place the root
// has among what that node takes. A hook on the root, and what keeps its
// gradient, are this node's, so they are called where a gradient reaches
// the root, and not where the node of the graph runs for another root: two
// results of eager's that are computed apart have nodes apart.
class GraphRoot final : public torch::autograd::Node {
 public:
  std::string name() const override {
    return "GraphBackward";
  }

 protected:
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    return std::move(grads);
  }
};

// state is the node's from the call on, also when the call fails: release is
// called for it when the node dies.
int32_t graphNode(
    PyObject* roots,
    PyObject* targets,
    void* state,
    TcfgBackward backward,
    TcfgRelease release) {
  HANDLE_TH_ERRORS
  const auto node =
      c10::make_intrusive<GraphBackward>(state, backward, release);
  TORCH_CHECK(
      PyTuple_Check(roots) && PyTuple_Check(targets),
      "torch.vm: graph_node takes a tuple of roots and a tuple of targets");
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(targets); ++i) {
    PyObject* item = PyTuple_GET_ITEM(targets, i);
    TORCH_CHECK(THPVariable_Check(item), "torch.vm: a target is a tensor");
    const auto& t = THPVariable_Unpack(item);
    node->add_next_edge(
        t.requires_grad() ? torch::autograd::impl::gradient_edge(t)
                          : torch::autograd::Edge());
  }
  const c10::intrusive_ptr<torch::autograd::Node> base = node;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(roots); ++i) {
    PyObject* item = PyTuple_GET_ITEM(roots, i);
    TORCH_CHECK(THPVariable_Check(item), "torch.vm: a root is a tensor");
    const auto& t = THPVariable_Unpack(item);
    TORCH_CHECK(
        !t.requires_grad() && !t.is_view(),
        "torch.vm: a root is a tensor that was made below the autograd keys "
        "and is no view");
    const c10::intrusive_ptr<torch::autograd::Node> front =
        c10::make_intrusive<GraphRoot>();
    front->add_next_edge(
        torch::autograd::Edge(base, node->add_input_metadata(t)));
    torch::autograd::set_history(t, front);
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

constexpr const char* kWorkspaceCapsule = "torch._C._cfg_workspace";

void workspaceCapsuleFree(PyObject* capsule) {
  torch::cfg::workspaceFree(static_cast<torch::cfg::Workspace*>(
      PyCapsule_GetPointer(capsule, kWorkspaceCapsule)));
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
    &tensorBox,
    &tensorRetain,
    &tensorRelease,
#ifdef TCFG_HAS_FRAMES
    &framePush,
    &framePop,
    &frameHeld,
    &frameSwap,
    &frameTraced,
    &frameHook,
    &frameWant,
    &frameAdopt,
    &frameInterpret,
    &frameReturns,
#else
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
#endif
    &isTracing,
    &workspaceEnterRaw,
    &workspaceLeaveRaw,
    &callModes,
    &siteExport,
    &traceCalls,
    &dispatchKeys,
#ifdef TCFG_HAS_FRAMES
    &frameCount,
#else
    nullptr,
#endif
    &tensorForwardGrad,
    &tapeBegin,
    &tapeEnd,
    &nodeOf,
    &nodeInfo,
    &nodeNext,
    &nodeCall,
    &opFind,
    &opCall,
    &programNew,
    &programFree,
    &programRun,
    &graphNode,
#ifdef TCFG_HAS_FRAMES
    &frameCountKernel,
#else
    nullptr,
#endif
    &gradEnabled,
};

} // namespace

namespace torch::cfg {

void initCfgBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  // The node of a graph has a Python class of its name, as eager's nodes do.
  static PyTypeObject graph_backward_type;
  torch::autograd::_initFunctionPyTypeObject(
      graph_backward_type, "GraphBackward", nullptr, nullptr);
  // A class that is no heap type is never let go: it holds itself.
  Py_INCREF(&graph_backward_type);
  torch::autograd::registerCppFunction(
      typeid(GraphBackward), &graph_backward_type);
  torch::autograd::registerCppFunction(typeid(GraphRoot), &graph_backward_type);

  m.def("_cfg_c_api", []() {
    // PyCapsule_New takes a non-const pointer; the table is only read.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* api = const_cast<TcfgApi*>(&api_table);
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(api, "torch._C._cfg_c_api", nullptr));
  });

  // A workspace for a compiled function (workspace.h), or None if the
  // process has another CPU allocator than PyTorch's default.
  m.def("_cfg_workspace", []() -> py::object {
    auto* workspace = workspaceNew();
    if (workspace == nullptr) {
      return py::none();
    }
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(workspace, kWorkspaceCapsule, &workspaceCapsuleFree));
  });
  m.def("_cfg_workspace_stats", [](const py::capsule& capsule) {
    const auto stats = workspaceStats(static_cast<Workspace*>(
        PyCapsule_GetPointer(capsule.ptr(), kWorkspaceCapsule)));
    py::dict result;
    result["calls"] = stats.calls;
    result["hits"] = stats.hits;
    result["misses"] = stats.misses;
    result["kept"] = stats.kept;
    result["live"] = stats.live;
    result["peak"] = stats.peak;
    return result;
  });

  // Runs fn under a dispatch capture and returns (result, records) where
  // records lists (op, dispatch key, depth, is_redispatch) for every kernel
  // the dispatcher invoked.
  // With detail, a record is a dict that also has the rest of what the capture
  // kept about it. A capture keeps the arguments and results of every
  // top-level call alive until it ends.
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
  site_class.def("export", [](const TcfgSite& site) {
    auto exported = py::reinterpret_steal<py::object>(siteExport(&site));
    if (!exported) {
      throw py::error_already_set();
    }
    return exported;
  });
  // Returns NotImplemented when a guard of the site does not hold.
  site_class.def("call", [](TcfgSite& site, const py::args& operands) {
    TORCH_CHECK_VALUE(
        operands.size() == site.guards.size(), "wrong number of operands");
    int32_t status = TCFG_OK;
    auto result = py::reinterpret_steal<py::object>(siteCall(
        &site, PySequence_Fast_ITEMS(operands.ptr()), /*flags=*/0, &status));
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
        pyop.flags = skip_autograd ? uint32_t{TCFG_ALLOW_SKIP_AUTOGRAD} : 0U;
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
