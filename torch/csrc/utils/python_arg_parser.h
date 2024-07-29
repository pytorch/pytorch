#pragma once

// Parse arguments to Python functions implemented in C++
// This is similar to PyArg_ParseTupleAndKeywords(), but specifically handles
// the types relevant to PyTorch and distinguishes between overloaded function
// signatures.
//
// Example:
//
//   static PythonArgParser parser({
//     "norm(Scalar p, int64_t dim, bool keepdim=False)",
//     "norm(Scalar p=2)",
//   });
//   ParsedArgs<3> parsed_args;
//   auto r = parser.parse(args, kwargs, parsed_args);
//   if (r.idx == 0) {
//     norm(r.scalar(0), r.int64(1), r.bool(0));
//   } else {
//     norm(r.scalar(0));
//   }
//
// We auto-generate most uses of PythonArgParser; the generated files
// are torch/csrc/autograd/generated/python_*.cpp
//
// Some gotchas that you should watch out for:
//
//    - Note [Order of overloads matters]
//      Order of overloads matters.  A set of input arguments may
//      bind to multiple argument specs; we will always pick the
//      first one in PythonArgParser.  However, when you are writing
//      overloads in, e.g., native_functions.yaml, you don't have to
//      worry about what order you write them, because the code
//      generation logic always gives the overloads a canonical
//      order, where Tensor overloads come first, before Scalar overloads.
//      This logic is in sort_declarations in
//      tools/autograd/gen_python_functions.py
//
//    - Zero-dim tensors (e.g., torch.tensor(2)) bind to both
//      Scalar and Tensor, UNLESS they require grad (in which case
//      they only bind to Tensor).

#include <pybind11/pytypes.h>
#include <torch/csrc/python_headers.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/python_dimname.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_symnode.h>
#include <torch/csrc/utils/six.h>

#include <ATen/DeviceAccelerator.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <c10/core/SymFloat.h>
#include <c10/core/SymNodeImpl.h>

#include <c10/core/DispatchKeySet.h>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

inline bool THPUtils_checkScalar(PyObject* obj) {
#ifdef USE_NUMPY
  if (torch::utils::is_numpy_scalar(obj)) {
    return true;
  }
#endif
  return PyFloat_Check(obj) || PyLong_Check(obj) || PyComplex_Check(obj) ||
      torch::is_symint(py::handle(obj)) ||
      torch::is_symfloat(py::handle(obj)) || torch::is_symbool(py::handle(obj));
}

namespace torch {

bool should_allow_numbers_as_tensors(const std::string& name);

enum class ParameterType {
  TENSOR,
  SCALAR,
  INT64,
  SYM_INT,
  DOUBLE,
  COMPLEX,
  TENSOR_LIST,
  INT_LIST,
  GENERATOR,
  BOOL,
  STORAGE,
  PYOBJECT,
  SCALARTYPE,
  LAYOUT,
  MEMORY_FORMAT,
  DEVICE,
  STREAM,
  STRING,
  DIMNAME,
  DIMNAME_LIST,
  QSCHEME,
  FLOAT_LIST,
  SCALAR_LIST,
  SYM_INT_LIST,
  DISPATCH_KEY_SET
};

struct FunctionParameter;
struct FunctionSignature;
struct PythonArgs;

// Contains bound Python arguments in declaration order
template <int N>
struct ParsedArgs {
  ParsedArgs() : args() {}
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  PyObject* args[N];
};

// A PythonArgParser contains a list of valid signatures. Instances are
// typically global variables and should be immutable.
struct PYBIND11_EXPORT PythonArgParser {
  explicit PythonArgParser(
      const std::vector<std::string>& fmts,
      bool traceable = false);

  // meant only for `torch` functions.
  template <int N>
  inline PythonArgs parse(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      ParsedArgs<N>& dst);

  template <int N>
  inline PythonArgs parse(PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst);

  inline PythonArgs parse(PyObject* self, ParsedArgs<0>& dst);

  // Formatted strings of non-hidden signatures
  std::vector<std::string> get_signatures() const;

 private:
  [[noreturn]] void print_error(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      PyObject* parsed_args[]);
  void check_deprecated(const FunctionSignature& signature);
  PythonArgs raw_parse(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      PyObject* parsed_args[]);

  std::vector<FunctionSignature> signatures_;
  std::string function_name;
  size_t max_args;
  bool traceable;
};

// FunctionSignature represents a single valid signature for a Python function.
// It is immutable once constructed. The contained data can be concurrently
// accessed by multiple calls.
struct FunctionSignature {
  explicit FunctionSignature(const std::string& fmt, int index);

  bool parse(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      PyObject* dst[],
      std::vector<PyObject*>& overloaded_args,
      bool raise_exception);

  std::string toString() const;

  std::string name;
  std::vector<FunctionParameter> params;
  size_t min_args;
  size_t max_args;
  size_t max_pos_args;
  int index;
  bool hidden;
  bool deprecated;
};

// PythonArgs contains bound Python arguments for an actual invocation
// along with references to the matched signature.
struct PythonArgs {
  PythonArgs(
      bool traceable,
      const FunctionSignature& signature,
      PyObject** args,
      std::vector<PyObject*> overloaded_args)
      : idx(signature.index),
        traceable(traceable),
        signature(signature),
        args(args),
        overloaded_args(std::move(overloaded_args)) {}

  int idx;
  bool traceable;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const FunctionSignature& signature;
  PyObject** args;
  std::vector<PyObject*> overloaded_args; // NOTE: borrowed references

  inline bool has_torch_function();
  inline std::string get_func_name();
  inline at::Tensor tensor(int i);
  inline std::optional<at::Tensor> optionalTensor(int i);
  inline at::Scalar scalar(int i);
  inline at::Scalar scalarWithDefault(int i, const at::Scalar& default_scalar);
  inline std::vector<at::Scalar> scalarlist(int i);
  inline std::vector<at::Tensor> tensorlist(int i);
  inline torch::List<std::optional<at::Tensor>> list_of_optional_tensors(int i);
  template <int N>
  inline std::array<at::Tensor, N> tensorlist_n(int i);
  inline std::vector<int64_t> intlist(int i);
  inline std::vector<c10::SymInt> symintlist(int i);
  inline c10::OptionalArray<int64_t> intlistOptional(int i);
  inline c10::OptionalArray<c10::SymInt> symintlistOptional(int i);
  inline std::vector<int64_t> intlistWithDefault(
      int i,
      std::vector<int64_t> default_intlist);
  inline std::optional<at::Generator> generator(int i);
  inline at::Storage storage(int i);
  inline at::Storage storage(
      int i,
      at::ScalarType& storage_scalar_type,
      bool& is_typed_storage);
  inline c10::Stream stream(int i);
  inline at::ScalarType scalartype(int i);
  inline at::ScalarType scalartypeWithDefault(
      int i,
      at::ScalarType default_scalartype);
  inline std::optional<at::ScalarType> scalartypeOptional(int i);
  inline std::optional<at::Scalar> scalarOptional(int i);
  inline std::optional<int64_t> toInt64Optional(int i);
  inline std::optional<c10::SymInt> toSymIntOptional(int i);
  inline std::optional<bool> toBoolOptional(int i);
  inline std::optional<double> toDoubleOptional(int i);
  inline c10::OptionalArray<double> doublelistOptional(int i);
  inline std::vector<double> doublelist(int i);
  inline std::vector<double> getDoublelist(int i);
  inline at::Layout layout(int i);
  inline at::Layout layoutWithDefault(int i, at::Layout default_layout);
  inline std::optional<at::Layout> layoutOptional(int i);
  inline at::Device device(int i);
  inline at::Device deviceWithDefault(int i, const at::Device& default_device);
  inline std::optional<at::Device> deviceOptional(int i);
  inline at::Dimname dimname(int i);
  inline std::vector<at::Dimname> dimnamelist(int i);
  inline std::optional<std::vector<at::Dimname>> toDimnameListOptional(int i);
  inline at::MemoryFormat memoryformat(int i);
  inline std::optional<at::MemoryFormat> memoryformatOptional(int i);
  inline at::QScheme toQScheme(int i);
  inline std::string string(int i);
  inline std::string stringWithDefault(int i, const std::string& default_str);
  inline std::optional<std::string> stringOptional(int i);
  inline c10::string_view stringView(int i);
  inline c10::string_view stringViewWithDefault(
      int i,
      const c10::string_view default_str);
  inline std::optional<c10::string_view> stringViewOptional(int i);
  inline PyObject* pyobject(int i);
  inline int64_t toInt64(int i);
  inline c10::SymInt toSymInt(int i);
  inline c10::SymBool toSymBool(int i);
  inline int64_t toInt64WithDefault(int i, int64_t default_int);
  inline double toDouble(int i);
  inline double toDoubleWithDefault(int i, double default_double);
  inline c10::complex<double> toComplex(int i);
  inline c10::complex<double> toComplexWithDefault(
      int i,
      c10::complex<double> default_complex);
  inline bool toBool(int i);
  inline bool toBoolWithDefault(int i, bool default_bool);
  inline bool isNone(int i);
  inline std::optional<c10::DispatchKeySet> toDispatchKeySetOptional(int i);

 private:
  at::Tensor tensor_slow(int i);
  at::Scalar scalar_slow(int i);
  at::Scalar scalar_slow(PyObject* arg);
};

// FunctionParameter is a single formal parameter of a Python function.
// It is immutable once constructed.
struct FunctionParameter {
  FunctionParameter(const std::string& fmt, bool keyword_only);

  bool check(
      PyObject* obj,
      std::vector<PyObject*>& overloaded_args,
      int argnum,
      int64_t* failed_idx = nullptr);

  void set_default_str(const std::string& str);
  std::string type_name() const;

  ParameterType type_;
  bool optional;
  bool allow_none;
  bool keyword_only;
  bool allow_numbers_as_tensors = false;
  int size;
  std::string name;
  // having this as a raw PyObject * will presumably leak it, but these are only
  // held by static objects anyway, and Py_Finalize can already be called when
  // this is destructed.
  PyObject* python_name;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  at::SmallVector<PyObject*, 5> numpy_python_names;
  at::Scalar default_scalar;
  std::vector<int64_t> default_intlist;
  std::string default_string;
  union {
    bool default_bool;
    int64_t default_int;
    double default_double;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    double default_complex[2]; // see Scalar
    at::ScalarType default_scalartype;
    at::Layout default_layout;
  };
  std::string default_value;
};

template <int N>
inline PythonArgs PythonArgParser::parse(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    ParsedArgs<N>& dst) {
  TORCH_CHECK_VALUE(
      N >= max_args,
      "PythonArgParser: dst ParsedArgs buffer does not have enough capacity, expected ",
      max_args,
      " (got ",
      N,
      ")");
  return raw_parse(self, args, kwargs, dst.args);
}

template <int N>
inline PythonArgs PythonArgParser::parse(
    PyObject* args,
    PyObject* kwargs,
    ParsedArgs<N>& dst) {
  return parse(nullptr, args, kwargs, dst);
}

inline PythonArgs PythonArgParser::parse(PyObject* self, ParsedArgs<0>& dst) {
  return parse(self, nullptr, nullptr, dst);
}

inline bool PythonArgs::has_torch_function() {
  return !overloaded_args.empty() || at::impl::torch_function_mode_enabled();
}

inline std::string PythonArgs::get_func_name() {
  return signature.name;
}

// TODO: this can return MaybeOwned
inline at::Tensor PythonArgs::tensor(int i) {
  if (args[i] && THPVariable_CheckExact(args[i])) {
    return THPVariable_Unpack(args[i]);
  }
  return tensor_slow(i);
}

inline std::optional<at::Tensor> PythonArgs::optionalTensor(int i) {
  at::Tensor t = tensor(i);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (t.defined()) {
    return t;
  } else {
    return std::nullopt;
  }
}

inline at::Scalar PythonArgs::scalar(int i) {
  if (!args[i])
    return signature.params[i].default_scalar;
  return scalar_slow(i);
}

inline std::vector<at::Scalar> PythonArgs::scalarlist(int i) {
  if (!args[i])
    return std::vector<at::Scalar>();
  auto tuple = six::isTuple(args[i]);
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  std::vector<at::Scalar> res(size);
  for (const auto idx : c10::irange(size)) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    res[idx] = scalar_slow(obj);
  }
  return res;
}

inline at::Scalar PythonArgs::scalarWithDefault(
    int i,
    const at::Scalar& default_scalar) {
  if (!args[i])
    return default_scalar;
  return scalar_slow(i);
}

inline std::optional<at::Scalar> PythonArgs::scalarOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return scalar_slow(i);
}

inline std::vector<at::Tensor> PythonArgs::tensorlist(int i) {
  if (!args[i])
    return std::vector<at::Tensor>();
  auto tuple = six::isTuple(args[i]);
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  std::vector<at::Tensor> res(size);
  for (const auto idx : c10::irange(size)) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    // This is checked by the argument parser so it's safe to cast without
    // checking if this is a tensor first
    res[idx] = THPVariable_Unpack(obj);
  }
  return res;
}

inline torch::List<std::optional<at::Tensor>> PythonArgs::
    list_of_optional_tensors(int i) {
  if (!args[i])
    return torch::List<std::optional<at::Tensor>>();
  auto tuple = six::isTuple(args[i]);
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  torch::List<std::optional<at::Tensor>> res;
  res.reserve(size);
  for (const auto idx : c10::irange(size)) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    // This is checked by the argument parser so it's safe to cast without
    // checking if this is a tensor first
    res.push_back(THPVariable_Unpack(obj));
  }
  return res;
}

template <int N>
inline std::array<at::Tensor, N> PythonArgs::tensorlist_n(int i) {
  auto res = std::array<at::Tensor, N>();
  if (!args[i])
    return res;
  auto tuple = six::isTuple(args[i]);
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  if (size != N) {
    throw TypeError("expected tuple of %d elements but got %d", N, (int)size);
  }
  for (const auto idx : c10::irange(size)) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    // This is checked by the argument parser so it's safe to cast without
    // checking if this is a tensor first
    res[idx] = THPVariable_Unpack(obj);
  }
  return res;
}

inline std::vector<int64_t> PythonArgs::intlist(int i) {
  return intlistWithDefault(i, signature.params[i].default_intlist);
}

inline PyObject* toPyObject(const c10::SymInt& symint) {
  if (symint.is_symbolic()) {
    auto r = py::cast(symint).release().ptr();
    TORCH_INTERNAL_ASSERT(r);
    return r;
  } else {
    auto m = symint.maybe_as_int();
    return THPUtils_packInt64(*m);
  }
}

inline void throw_intlist_exception(
    const torch::PythonArgs* args,
    size_t i,
    PyObject* obj,
    size_t idx,
    const std::exception& e = python_error()) {
  std::string error = strlen(e.what())
      ? e.what()
      : std::string("type must be ") + args->signature.params[i].type_name() +
          ",but got " + Py_TYPE(obj)->tp_name;
  throw TypeError(
      "%s(): argument '%s' failed to unpack the object at pos %zu with error \"%s\"",
      args->signature.name.c_str(),
      args->signature.params[i].name.c_str(),
      idx + 1,
      error.c_str());
}

inline std::vector<c10::SymInt> PythonArgs::symintlist(int i) {
  if (!args[i]) {
    return c10::fmap(signature.params[i].default_intlist, [](int64_t di) {
      return c10::SymInt(di);
    });
  }

  const auto size1 = signature.params[i].size;
  if (size1 > 0 && THPUtils_checkLong(args[i])) {
    return std::vector<c10::SymInt>(
        size1, c10::SymInt(THPUtils_unpackLong(args[i])));
  }

  if (size1 > 0 && torch::is_symint(py::handle(args[i]))) {
    auto si = py::handle(args[i]).cast<c10::SymInt>();
    return std::vector<c10::SymInt>(size1, si);
  }

  PyObject* arg = args[i];
  auto tuple = PyTuple_Check(arg);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size2 = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<c10::SymInt> res;
  res.reserve(size2);
  for (const auto idx : c10::irange(size2)) {
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);

    // Elements of torch.Size are tensors during tracing, and we need to
    // record extra information before they are turned into an IntArrayRef
    if (traceable && jit::tracer::isTracing() && THPVariable_Check(obj)) {
      auto& var = THPVariable_Unpack(obj);
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          signature.params[i].name, size2, idx, var);
      try {
        res.emplace_back(var.item<int64_t>());
        continue;
      } catch (std::exception& e) {
        throw_intlist_exception(this, i, obj, idx, e);
      }
      continue;
    } else {
      // convert tensor to scalar outside of try / catch,
      // so that Tensor subclass exceptions will not be caught.
      if (THPUtils_checkLongExact(obj)) {
        // Fast path for plain numbers
        try {
          res.emplace_back(THPUtils_unpackLong(obj));
        } catch (std::exception& e) {
          throw_intlist_exception(this, i, obj, idx, e);
        }
      } else if (THPVariable_Check(obj)) {
        auto& var = THPVariable_Unpack(obj);
        if (var.numel() != 1 ||
            !at::isIntegralType(
                var.dtype().toScalarType(), /*include_bool*/ true)) {
          throw_intlist_exception(this, i, obj, idx);
        }
        auto scalar = var.item();
        TORCH_CHECK(scalar.isIntegral(/*include bool*/ false));
        res.push_back(scalar.toSymInt());
      } else {
        try {
          if (is_symint(py::handle(obj))) {
            res.push_back(py::handle(obj).cast<c10::SymInt>());
          } else {
            res.emplace_back(THPUtils_unpackIndex(obj));
          }
        } catch (std::exception& e) {
          throw_intlist_exception(this, i, obj, idx, e);
        }
      }
    }
  }

  return res;
}

inline std::vector<int64_t> PythonArgs::intlistWithDefault(
    int i,
    std::vector<int64_t> default_intlist) {
  if (!args[i])
    return default_intlist;
  PyObject* arg = args[i];
  const auto size1 = signature.params[i].size;
  if (size1 > 0 && THPUtils_checkLong(arg)) {
    return std::vector<int64_t>(size1, THPUtils_unpackLong(arg));
  }
  if (size1 > 0 && torch::is_symint(py::handle(arg))) {
    return std::vector<int64_t>(
        size1,
        py::handle(arg).cast<c10::SymInt>().guard_int(__FILE__, __LINE__));
  }
  auto tuple = PyTuple_Check(arg);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size2 = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<int64_t> res(size2);
  for (const auto idx : c10::irange(size2)) {
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    // Elements of torch.Size are tensors during tracing, and we need to
    // record extra information before they are turned into an IntArrayRef
    if (traceable && jit::tracer::isTracing() && THPVariable_Check(obj)) {
      auto& var = THPVariable_Unpack(obj);
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          signature.params[i].name, size2, idx, var);
      try {
        res[idx] = var.item<int64_t>();
        continue;
      } catch (std::exception& e) {
        throw_intlist_exception(this, i, obj, idx, e);
      }
    } else {
      // convert tensor to scalar outside of try / catch,
      // so that Tensor subclass exceptions will not be caught.
      if (THPUtils_checkLongExact(obj)) {
        // Fast path for plain numbers
        try {
          res[idx] = THPUtils_unpackLong(obj);
        } catch (std::exception& e) {
          throw_intlist_exception(this, i, obj, idx, e);
        }
      } else if (torch::is_symint(py::handle(obj))) {
        res[idx] = py::cast<c10::SymInt>(py::handle(obj))
                       .guard_int(__FILE__, __LINE__);
      } else if (THPVariable_Check(obj)) {
        auto& var = THPVariable_Unpack(obj);
        if (var.numel() != 1 ||
            !at::isIntegralType(
                var.dtype().toScalarType(), /*include_bool*/ true)) {
          throw_intlist_exception(this, i, obj, idx);
        }
        res[idx] = var.item<int64_t>();
      } else {
        try {
          res[idx] = THPUtils_unpackIndex(obj);
        } catch (std::exception& e) {
          throw_intlist_exception(this, i, obj, idx, e);
        }
      }
    }
  }
  return res;
}

inline c10::OptionalArray<int64_t> PythonArgs::intlistOptional(int i) {
  if (!args[i]) {
    return {};
  }
  return intlist(i);
}

inline c10::OptionalArray<c10::SymInt> PythonArgs::symintlistOptional(int i) {
  if (!args[i]) {
    return {};
  }
  return symintlist(i);
}

inline std::vector<double> PythonArgs::getDoublelist(int i) {
  PyObject* arg = args[i];
  auto tuple = PyTuple_Check(arg);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<double> res(size);
  for (const auto idx : c10::irange(size)) {
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    try {
      res[idx] = THPUtils_unpackDouble(obj);
    } catch (const std::exception&) {
      throw TypeError(
          "%s(): argument '%s' must be %s, but found element of type %s at pos %zu",
          signature.name.c_str(),
          signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(),
          Py_TYPE(obj)->tp_name,
          idx + 1);
    }
  }
  return res;
}

inline c10::OptionalArray<double> PythonArgs::doublelistOptional(int i) {
  if (!args[i]) {
    return {};
  }
  return this->getDoublelist(i);
}

inline std::vector<double> PythonArgs::doublelist(int i) {
  if (!args[i]) {
    return {};
  }
  return this->getDoublelist(i);
}

inline std::optional<c10::DispatchKeySet> PythonArgs::toDispatchKeySetOptional(
    int i) {
  if (!args[i]) {
    return {};
  }
  return py::cast<c10::DispatchKeySet>(py::handle(args[i]));
}

inline at::ScalarType PythonArgs::scalartypeWithDefault(
    int i,
    at::ScalarType default_scalartype) {
  if (!args[i])
    return default_scalartype;
  return scalartype(i);
}

inline at::ScalarType toScalarType(PyObject* obj) {
  if (obj == (PyObject*)&PyFloat_Type) {
    return at::ScalarType::Double;
  }
  if (obj == (PyObject*)&PyBool_Type) {
    return at::ScalarType::Bool;
  }
  if (obj == (PyObject*)&PyLong_Type) {
    return at::ScalarType::Long;
  }
  if (obj == (PyObject*)&PyComplex_Type) {
    return at::ScalarType::ComplexDouble;
  }
  return reinterpret_cast<THPDtype*>(obj)->scalar_type;
}

inline at::ScalarType PythonArgs::scalartype(int i) {
  if (!args[i]) {
    auto scalartype = signature.params[i].default_scalartype;
    return (scalartype == at::ScalarType::Undefined)
        ? torch::tensors::get_default_scalar_type()
        : scalartype;
  }
  PyObject* obj = args[i];
  return toScalarType(obj);
}

inline std::optional<at::ScalarType> PythonArgs::scalartypeOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return scalartype(i);
}

inline at::Layout toLayout(PyObject* obj) {
  const auto layout = reinterpret_cast<THPLayout*>(obj);
  return layout->layout;
}

inline at::Layout PythonArgs::layout(int i) {
  if (!args[i])
    return signature.params[i].default_layout;
  return toLayout(args[i]);
}

inline at::Layout PythonArgs::layoutWithDefault(
    int i,
    at::Layout default_layout) {
  if (!args[i])
    return default_layout;
  return layout(i);
}

inline std::optional<at::Layout> PythonArgs::layoutOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return layout(i);
}

inline at::Device deviceFromLong(int64_t device_index) {
  TORCH_CHECK(device_index >= 0, "Device index must not be negative");
  return at::Device(
      at::getAccelerator(true).value(),
      static_cast<c10::DeviceIndex>(device_index));
}

inline at::Device toDevice(PyObject* obj) {
  if (THPDevice_Check(obj)) {
    const auto device = reinterpret_cast<THPDevice*>(obj);
    return device->device;
  }
  if (THPUtils_checkLong(obj)) {
    return deviceFromLong(THPUtils_unpackLong(obj));
  }
  if (torch::is_symint(py::handle(obj))) {
    auto device_index =
        py::cast<c10::SymInt>(py::handle(obj)).guard_int(__FILE__, __LINE__);
    return deviceFromLong(device_index);
  }
  const std::string& device_str = THPUtils_unpackString(obj);
  return at::Device(device_str);
}

inline at::Device PythonArgs::device(int i) {
  if (!args[i]) {
    return torch::tensors::get_default_device();
  }
  return toDevice(args[i]);
}

inline at::Device PythonArgs::deviceWithDefault(
    int i,
    const at::Device& default_device) {
  if (!args[i])
    return default_device;
  return device(i);
}

inline std::optional<at::Device> PythonArgs::deviceOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return device(i);
}

inline at::Dimname PythonArgs::dimname(int i) {
  TORCH_INTERNAL_ASSERT(args[i] != nullptr);
  return THPDimname_parse(args[i]);
}

inline std::vector<at::Dimname> parseDimnameList(PyObject* arg) {
  auto tuple = PyTuple_Check(arg);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<at::Dimname> res;
  res.reserve(size);
  for (const auto idx : c10::irange(size)) {
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    res.push_back(THPDimname_parse(obj));
  }
  return res;
}

inline std::optional<std::vector<at::Dimname>> PythonArgs::
    toDimnameListOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return parseDimnameList(args[i]);
}

inline std::vector<at::Dimname> PythonArgs::dimnamelist(int i) {
  TORCH_INTERNAL_ASSERT(args[i]);
  PyObject* arg = args[i];
  auto size = signature.params[i].size;
  TORCH_INTERNAL_ASSERT(size == 0 || size == 1);
  if (size == 1 && THPUtils_checkDimname(arg)) {
    return {THPDimname_parse(arg)};
  }
  return parseDimnameList(arg);
}

inline at::MemoryFormat PythonArgs::memoryformat(int i) {
  if (!args[i])
    return at::MemoryFormat::Contiguous;
  TORCH_CHECK(
      THPMemoryFormat_Check(args[i]),
      "memory_format arg must be an instance of the torch.memory_format");
  const auto memory_format = reinterpret_cast<THPMemoryFormat*>(args[i]);
  return memory_format->memory_format;
}

inline std::optional<at::MemoryFormat> PythonArgs::memoryformatOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return memoryformat(i);
}

inline at::QScheme PythonArgs::toQScheme(int i) {
  if (!args[i])
    return at::kPerTensorAffine;
  TORCH_CHECK(
      THPQScheme_Check(args[i]),
      "qscheme arg must be an instance of the torch.qscheme");
  const auto qscheme = reinterpret_cast<THPQScheme*>(args[i]);
  return qscheme->qscheme;
}

inline std::string PythonArgs::string(int i) {
  return stringWithDefault(i, signature.params[i].default_string);
}

inline std::string PythonArgs::stringWithDefault(
    int i,
    const std::string& default_str) {
  if (!args[i])
    return default_str;
  return THPUtils_unpackString(args[i]);
}

inline std::optional<std::string> PythonArgs::stringOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return THPUtils_unpackString(args[i]);
}

inline c10::string_view PythonArgs::stringView(int i) {
  return stringViewWithDefault(i, signature.params[i].default_string);
}

inline c10::string_view PythonArgs::stringViewWithDefault(
    int i,
    const c10::string_view default_str) {
  if (!args[i])
    return default_str;
  return THPUtils_unpackStringView(args[i]);
}

inline std::optional<c10::string_view> PythonArgs::stringViewOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return THPUtils_unpackStringView(args[i]);
}

inline int64_t PythonArgs::toInt64(int i) {
  if (!args[i])
    return signature.params[i].default_int;
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::IntType::get());
  }
  if (torch::is_symint(py::handle(args[i]))) {
    return py::cast<c10::SymInt>(py::handle(args[i]))
        .guard_int(__FILE__, __LINE__);
  }
  return THPUtils_unpackLong(args[i]);
}

inline c10::SymInt PythonArgs::toSymInt(int i) {
  if (!args[i]) {
    return c10::SymInt(signature.params[i].default_int);
  }

  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::IntType::get());
  }

  return py::cast<c10::SymInt>(py::handle(args[i]));
}

inline c10::SymBool PythonArgs::toSymBool(int i) {
  if (!args[i]) {
    return c10::SymBool(signature.params[i].default_bool);
  }
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::BoolType::get());
  }

  return py::cast<c10::SymBool>(py::handle(args[i]));
}

inline int64_t PythonArgs::toInt64WithDefault(int i, int64_t default_int) {
  if (!args[i])
    return default_int;
  return toInt64(i);
}

inline std::optional<int64_t> PythonArgs::toInt64Optional(int i) {
  if (!args[i])
    return std::nullopt;
  return toInt64(i);
}

inline std::optional<c10::SymInt> PythonArgs::toSymIntOptional(int i) {
  if (!args[i])
    return std::nullopt;
  return toSymInt(i);
}

inline std::optional<bool> PythonArgs::toBoolOptional(int i) {
  if (!args[i]) {
    return std::nullopt;
  }
  return toBool(i);
}

inline std::optional<double> PythonArgs::toDoubleOptional(int i) {
  if (!args[i]) {
    return std::nullopt;
  }
  return toDouble(i);
}

inline double PythonArgs::toDouble(int i) {
  if (!args[i])
    return signature.params[i].default_double;
  if (torch::is_symfloat(py::handle(args[i]))) {
    return py::cast<c10::SymFloat>(py::handle(args[i]))
        .guard_float(__FILE__, __LINE__);
  }
  if (torch::is_symint(py::handle(args[i]))) {
    return static_cast<double>(py::cast<c10::SymInt>(py::handle(args[i]))
                                   .guard_int(__FILE__, __LINE__));
  }
  return THPUtils_unpackDouble(args[i]);
}

inline bool PythonArgs::toBool(int i) {
  if (!args[i])
    return signature.params[i].default_bool;
  if (torch::is_symbool(py::handle(args[i]))) {
    return py::cast<c10::SymBool>(py::handle(args[i]))
        .guard_bool(__FILE__, __LINE__);
  }
  return args[i] == Py_True;
}

inline double PythonArgs::toDoubleWithDefault(int i, double default_double) {
  if (!args[i])
    return default_double;
  return toDouble(i);
}

inline c10::complex<double> PythonArgs::toComplex(int i) {
  if (!args[i])
    return *(reinterpret_cast<const c10::complex<double>*>(
        signature.params[i].default_complex));
  return THPUtils_unpackComplexDouble(args[i]);
}

inline c10::complex<double> PythonArgs::toComplexWithDefault(
    int i,
    c10::complex<double> default_value) {
  if (!args[i])
    return default_value;
  return toComplex(i);
}

inline bool PythonArgs::toBoolWithDefault(int i, bool default_bool) {
  if (!args[i])
    return default_bool;
  return toBool(i);
}

inline bool PythonArgs::isNone(int i) {
  return args[i] == nullptr;
}

inline std::optional<at::Generator> PythonArgs::generator(int i) {
  if (!args[i])
    return std::nullopt;
  return reinterpret_cast<THPGenerator*>(args[i])->cdata;
}

inline at::Storage PythonArgs::storage(int i) {
  if (!args[i])
    return at::Storage();
  return createStorage(args[i]);
}

inline at::Storage PythonArgs::storage(
    int i,
    at::ScalarType& storage_scalar_type,
    bool& is_typed_storage) {
  at::Storage storage;
  if (!args[i]) {
    storage = at::Storage();
    is_typed_storage = false;
    storage_scalar_type = at::ScalarType::Undefined;
  } else {
    std::tie(storage, storage_scalar_type, is_typed_storage) =
        createStorageGetType(args[i]);
  }
  return storage;
}

inline c10::Stream PythonArgs::stream(int i) {
  if (!args[i])
    return c10::Stream(
        c10::Stream::Default::DEFAULT, c10::Device(c10::DeviceType::CPU, -1));
  if (!THPStream_Check(args[i])) {
    throw TypeError(
        "expected Stream object. Got '%s'", Py_TYPE(args[i])->tp_name);
  }
  return c10::Stream::unpack3(
      ((THPStream*)args[i])->stream_id,
      static_cast<c10::DeviceIndex>(((THPStream*)args[i])->device_index),
      static_cast<c10::DeviceType>(((THPStream*)args[i])->device_type));
}

inline PyObject* PythonArgs::pyobject(int i) {
  if (!args[i])
    return Py_None;
  return args[i];
}

/*
 *
 * Handle __torch_function__ overrides if we know that there are overloaded
 * arguments.  All objects stored in r.overloaded_args must have a
 * __torch_function__ implementation and the arguments must be ordered in order
 * of precedence. Precedence goes from left to right in the order of the
 * signature of the function the overloaded arguments were passed to, except
 * subclasses are always considered before superclasses.
 *
 * If the result of calling __torch_function__ is NotImplemented, the
 * next implementation in the precedence order is called. If all
 * arguments return NotImplemented from their __torch_function__
 * implementation, a TypeError is raised in Python.
 *
 * Assumes overloaded_args has at least one entry. All entries must have
 * a __torch_function__ attribute that resolves to a callable that
 * accepts a torch API function, a tuple of arguments, and a dict of
 * keyword arguments for the torch API function.
 *
 * It is sufficient to call PythonArgs::has_torch_function before
 * calling this function to verify that there are valid arguments
 * present. If that is not done then special care must be taken to
 * ensure there are arguments that are overloaded with
 * __torch_function__.
 *
 * See torch._overrides.handle_torch_function for the equivalent
 * code in the pure-python implementation.
 *
 * 'r' is a parsed PythonArgs instance, returned from
 * PythonArgParser::parse.
 *
 * 'args' is a reference to the python tuple of arguments to the torch
 * API function.
 *
 * 'kwargs' is a reference to the python dict of keyword arguments to
 * the torch API function.
 *
 * 'torch_api' is a reference to a python torch API namespace.
 *
 * 'torch_api_function' is the reference to the original torch method, usually,
 * we can use torch_api and func_name to get torch_api_function. In some cases,
 * e.g., torch custom op, we create the function in C++, if we still use
 * torch_api and func_name to fetch original api, a cyclic call will happen.
 *
 * 'overloaded_args' is the args which have overloaded __torch_function__.
 *
 * 'func_name' is the named of the original torch method.
 *
 * TODO: we could use different names for the following 'handle_torch_function'
 * instead of overloading.
 *
 */
// Used for Tensor methods with arguments.
auto handle_torch_function(
    PythonArgs& r,
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* torch_api,
    const char* module_name,
    const char* func_name_override = nullptr) -> PyObject*;

// Used for functions which needs to parse python args.
auto handle_torch_function(
    PythonArgs& r,
    PyObject* args,
    PyObject* kwargs,
    PyObject* torch_api,
    const char* module_name,
    const char* func_name_override = nullptr) -> PyObject*;

// Used for functions that have no argument parsing.
auto handle_torch_function(
    PyObject* self,
    const std::string& func_name,
    PyObject* args = nullptr,
    PyObject* kwargs = nullptr,
    PyObject* torch_api = THPVariableClass,
    const std::string& module_name = "torch.Tensor") -> PyObject*;

// Used for functions created in C++, e.g., C++ custom op, which doesn't use
// PythonArgParser to get overloaded_args.
enum class TorchFunctionName { TorchFunction, TorchDispatch };

auto TORCH_PYTHON_API handle_torch_function_no_python_arg_parser(
    at::ArrayRef<PyObject*> overloaded_args,
    PyObject* args,
    PyObject* kwargs,
    const char* func_name,
    PyObject* torch_api_function,
    const char* module_name,
    TorchFunctionName torch_function_name = TorchFunctionName::TorchFunction)
    -> PyObject*;

// Used for getters of Tensor properties
auto handle_torch_function_getter(
    THPVariable* self,
    const std::string& property_name) -> PyObject*;

// Used for setters of Tensor properties.
auto handle_torch_function_setter(
    THPVariable* self,
    const std::string& property_name,
    PyObject* value) -> int;

// Used for __getitem__ and __setitem__
auto handle_torch_function_indexing(
    PyObject* self,
    PyObject* index,
    PyObject* val = nullptr) -> PyObject*;

/*
 * Check if the input obj is Tensor type, including its subclass, or overloaded
 * type. If the type defines __torch_function__, it also returns true.
 * Otherwise returns flase. If the class is not torch.Tensor, and it defines
 * __torch_function__, we append obj to overloaded_args.
 *
 * 'obj': the input argument to be checked
 * 'overloaded_args': the vector to append the overloaded args.
 */
bool is_tensor_and_append_overloaded(
    PyObject* obj,
    std::vector<PyObject*>* overloaded_args);

/*
 * Check if the input obj is Tensor List or Tensor Tuple type. First check
 * whether obj is Tuple or List type, if true, iterate over each element and
 * check whether it is Tensor type, including its subclass or overloaded type.
 * At the same time, the overloaded arg is appended to the overloaded_args.
 *
 * 'obj': the input argument to be checked
 * 'overloaded_args': the vector to append the overloaded args.
 * 'argnum': the number of total arguments of the function being checked.
 * 'throw_error': whether throw error if any element in the list or tuple is
 *                not tensor type or overloaded.
 */
bool is_tensor_list_and_append_overloaded(
    PyObject* obj,
    std::vector<PyObject*>* overloaded_args,
    int argnum,
    bool throw_error);

/* Given an argument that is definitely a tensor and is definitely overloaded,
 * append it to the overloaded arguments list.  Use this instead of
 * is_tensor_and_append_overloaded in situations where you have a PyObject
 * and you know it definitely is a Tensor and it is definitely overloaded.
 *
 * 'overloaded_args': the vector to append the overloaded args
 * 'obj': the input tensor that is overloaded
 */
void append_overloaded_tensor(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj);

/* Given an argument that is definitely a type and is definitely overloaded,
 * append it to the overloaded arguments list. Use this only with
 * __torch_dispatch__, where we operate on classes that have a
 * __torch_dispatch__ classmethod.
 *
 * 'overloaded_args': the vector to append the overloaded type
 * 'obj': the input class that has a __torch_dispatch__ classmethod.
 */
void append_overloaded_type(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj);

} // namespace torch
