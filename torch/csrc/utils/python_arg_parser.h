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


#include <torch/csrc/python_headers.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>
#include <ATen/core/EnableNamedTensor.h>
#ifdef BUILD_NAMEDTENSOR
#include <torch/csrc/python_dimname.h>
#endif
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/six.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <array>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace torch {

enum class ParameterType {
  TENSOR, SCALAR, INT64, DOUBLE, COMPLEX, TENSOR_LIST, INT_LIST, GENERATOR,
  BOOL, STORAGE, PYOBJECT, SCALARTYPE, LAYOUT, MEMORY_FORMAT, DEVICE, STRING,
  DIMNAME, DIMNAME_LIST, QSCHEME
};

struct FunctionParameter;
struct FunctionSignature;
struct PythonArgs;

// Contains bound Python arguments in declaration order
template<int N>
struct ParsedArgs {
  ParsedArgs() : args() { }
  PyObject* args[N];
};

struct PythonArgParser {
  explicit PythonArgParser(std::vector<std::string> fmts, bool traceable=false);

  template<int N>
  inline PythonArgs parse(PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst);

private:
  [[noreturn]]
  void print_error(PyObject* args, PyObject* kwargs, PyObject* parsed_args[]);
  PythonArgs raw_parse(PyObject* args, PyObject* kwargs, PyObject* parsed_args[]);

  std::vector<FunctionSignature> signatures_;
  std::string function_name;
  ssize_t max_args;
  bool traceable;
};

struct PythonArgs {
  PythonArgs(int idx, bool traceable, const FunctionSignature& signature, PyObject** args)
    : idx(idx)
    , traceable(traceable)
    , signature(signature)
    , args(args) {}

  int idx;
  bool traceable;
  const FunctionSignature& signature;
  PyObject** args;

  inline at::Tensor tensor(int i);
  inline at::Scalar scalar(int i);
  inline at::Scalar scalarWithDefault(int i, at::Scalar default_scalar);
  inline std::vector<at::Tensor> tensorlist(int i);
  template<int N>
  inline std::array<at::Tensor, N> tensorlist_n(int i);
  inline std::vector<int64_t> intlist(int i);
  inline std::vector<int64_t> intlistWithDefault(int i, std::vector<int64_t> default_intlist);
  inline at::Generator* generator(int i);
  inline at::Storage storage(int i);
  inline at::ScalarType scalartype(int i);
  inline at::ScalarType scalartypeWithDefault(int i, at::ScalarType default_scalartype);
  inline c10::optional<at::ScalarType> scalartypeOptional(int i);
  inline c10::optional<at::Scalar> scalarOptional(int i);
  inline c10::optional<int64_t> toInt64Optional(int i);
  inline c10::optional<bool> toBoolOptional(int i);
  inline c10::optional<double> toDoubleOptional(int i);
  inline const THPLayout& layout(int i);
  inline const THPLayout& layoutWithDefault(int i, const THPLayout& default_layout);
  inline at::Device device(int i);
  inline at::Device deviceWithDefault(int i, const at::Device& default_device);
  inline c10::optional<at::Device> deviceOptional(int i);
#ifdef BUILD_NAMEDTENSOR
  inline at::Dimname dimname(int i);
  inline std::vector<at::Dimname> dimnamelist(int i);
  inline c10::optional<std::vector<at::Dimname>> toDimnameListOptional(int i);
#endif
  inline at::MemoryFormat memoryformat(int i);
  inline c10::optional<at::MemoryFormat> memoryformatOptional(int i);
  inline at::QScheme toQScheme(int i);
  inline std::string string(int i);
  inline PyObject* pyobject(int i);
  inline int64_t toInt64(int i);
  inline int64_t toInt64WithDefault(int i, int64_t default_int);
  inline double toDouble(int i);
  inline double toDoubleWithDefault(int i, double default_double);
  inline std::complex<double> toComplex(int i);
  inline std::complex<double> toComplexWithDefault(int i, std::complex<double> default_complex);
  inline bool toBool(int i);
  inline bool toBoolWithDefault(int i, bool default_bool);
  inline bool isNone(int i);

private:
  at::Tensor tensor_slow(int i);
  at::Scalar scalar_slow(int i);
};

struct FunctionSignature {
  explicit FunctionSignature(const std::string& fmt);

  bool parse(PyObject* args, PyObject* kwargs, PyObject* dst[], bool raise_exception);
  std::string toString() const;

  std::string name;
  std::vector<FunctionParameter> params;
  ssize_t min_args;
  ssize_t max_args;
  ssize_t max_pos_args;
  bool hidden;
  bool deprecated;
};

struct FunctionParameter {
  FunctionParameter(const std::string& fmt, bool keyword_only);

  bool check(PyObject* obj);
  void set_default_str(const std::string& str);
  std::string type_name() const;

  ParameterType type_;
  bool optional;
  bool allow_none;
  bool keyword_only;
  bool allow_numbers_as_tensors = false;
  int size;
  std::string name;
  // having this as a raw PyObject * will presumably leak it, but these are only held by static objects
  // anyway, and Py_Finalize can already be called when this is destructed.
  PyObject *python_name;
  at::SmallVector<PyObject *, 5> numpy_python_names;
  at::Scalar default_scalar;
  std::vector<int64_t> default_intlist;
  union {
    bool default_bool;
    int64_t default_int;
    double default_double;
    double default_complex[2]; // see Scalar
    at::ScalarType default_scalartype;
    THPLayout* default_layout;
  };
};

template<int N>
inline PythonArgs PythonArgParser::parse(PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst) {
  if (N < max_args) {
    throw ValueError("PythonArgParser: dst ParsedArgs buffer does not have enough capacity, expected %d (got %d)",
        (int)max_args, N);
  }
  return raw_parse(args, kwargs, dst.args);
}

inline at::Tensor PythonArgs::tensor(int i) {
  if (args[i] && THPVariable_CheckExact(args[i])) {
    return reinterpret_cast<THPVariable*>(args[i])->cdata;
  }
  return tensor_slow(i);
}

inline at::Scalar PythonArgs::scalar(int i) {
  if (!args[i]) return signature.params[i].default_scalar;
  return scalar_slow(i);
}

inline at::Scalar PythonArgs::scalarWithDefault(int i, at::Scalar default_scalar) {
  if (!args[i]) return default_scalar;
  return scalar_slow(i);
}

inline c10::optional<at::Scalar> PythonArgs::scalarOptional(int i) {
  if (!args[i]) return c10::nullopt;
  return scalar_slow(i);
}

inline std::vector<at::Tensor> PythonArgs::tensorlist(int i) {
  if (!args[i]) return std::vector<at::Tensor>();
  auto tuple = six::isTuple(args[i]);
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  std::vector<at::Tensor> res(size);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx) : PyList_GET_ITEM(arg.get(), idx);
    if (!THPVariable_Check(obj)) {
      throw TypeError("expected Tensor as element %d in argument %d, but got %s",
                 idx, i, Py_TYPE(obj)->tp_name);
    }
    res[idx] = reinterpret_cast<THPVariable*>(obj)->cdata;
  }
  return res;
}

template<int N>
inline std::array<at::Tensor, N> PythonArgs::tensorlist_n(int i) {
  auto res = std::array<at::Tensor, N>();
  if (!args[i]) return res;
  auto tuple = six::isTuple(args[i]);
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  if (size != N) {
    throw TypeError("expected tuple of %d elements but got %d", N, (int)size);
  }
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx) : PyList_GET_ITEM(arg.get(), idx);
    if (!THPVariable_Check(obj)) {
      throw TypeError("expected Tensor as element %d in argument %d, but got %s",
                 idx, i, Py_TYPE(obj)->tp_name);
    }
    res[idx] = reinterpret_cast<THPVariable*>(obj)->cdata;
  }
  return res;
}

inline std::vector<int64_t> PythonArgs::intlist(int i) {
  return intlistWithDefault(i, signature.params[i].default_intlist);
}

inline std::vector<int64_t> PythonArgs::intlistWithDefault(int i, std::vector<int64_t> default_intlist) {
  if (!args[i]) return default_intlist;
  PyObject* arg = args[i];
  auto size = signature.params[i].size;
  if (size > 0 && THPUtils_checkLong(arg)) {
    return std::vector<int64_t>(size, THPUtils_unpackIndex(arg));
  }
  auto tuple = PyTuple_Check(arg);
  size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<int64_t> res(size);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    try {
      // Elements of torch.Size are tensors during tracing, and we need to record extra
      // information before they are turned into an IntArrayRef
      if (traceable && jit::tracer::isTracing() && THPVariable_Check(obj)) {
        auto & var = THPVariable_Unpack(obj);
        jit::tracer::ArgumentStash::stashIntArrayRefElem(
            signature.params[i].name, size, idx, var);
        res[idx] = var.item<int64_t>();
        continue;
      } else {
        res[idx] = THPUtils_unpackIndex(obj);
      }
    } catch (const std::exception &e) {
      throw TypeError("%s(): argument '%s' must be %s, but found element of type %s at pos %d",
          signature.name.c_str(), signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(), Py_TYPE(obj)->tp_name, idx + 1);
    }
  }
  return res;
}

inline at::ScalarType PythonArgs::scalartypeWithDefault(int i, at::ScalarType default_scalartype) {
  if (!args[i]) return default_scalartype;
  return scalartype(i);
}

inline at::ScalarType PythonArgs::scalartype(int i) {
  if (!args[i]) {
    auto scalartype = signature.params[i].default_scalartype;
    return (scalartype == at::ScalarType::Undefined) ?
            torch::tensors::get_default_scalar_type() : scalartype;
  }
  PyObject *obj = args[i];
  if (obj == (PyObject*)&PyFloat_Type) {
    return at::ScalarType::Double;
  }
  if (obj == (PyObject*)&PyBool_Type) {
    return at::ScalarType::Bool;
  }
  if (obj == (PyObject*)&PyLong_Type
#if PY_MAJOR_VERSION == 2
      || obj == (PyObject*)&PyInt_Type
#endif
  ) {
    return at::ScalarType::Long;
  }
  return reinterpret_cast<THPDtype*>(obj)->scalar_type;
}

inline c10::optional<at::ScalarType> PythonArgs::scalartypeOptional(int i) {
  if (!args[i])
    return c10::nullopt;
  return scalartype(i);
}

inline const THPLayout& PythonArgs::layout(int i) {
  if (!args[i]) return *signature.params[i].default_layout;
  return *reinterpret_cast<THPLayout*>(args[i]);
}

inline const THPLayout& PythonArgs::layoutWithDefault(int i, const THPLayout& default_layout) {
  if (!args[i]) return default_layout;
  return layout(i);
}

inline at::Device PythonArgs::device(int i) {
  if (!args[i]) {
    return at::Device(backendToDeviceType(tensorTypeIdToBackend(torch::tensors::get_default_tensor_type_id())));
  }
  if (THPDevice_Check(args[i])) {
    const auto device = reinterpret_cast<THPDevice*>(args[i]);
    return device->device;
  }
  if (THPUtils_checkLong(args[i])) {
    const auto device_index = THPUtils_unpackLong(args[i]);
    TORCH_CHECK(device_index >= 0, "Device index must not be negative");
    return at::Device(at::DeviceType::CUDA, device_index);
  }
  const std::string &device_str = THPUtils_unpackString(args[i]);
  return at::Device(device_str);
}

inline at::Device PythonArgs::deviceWithDefault(int i, const at::Device& default_device) {
  if (!args[i]) return default_device;
  return device(i);
}

inline c10::optional<at::Device> PythonArgs::deviceOptional(int i) {
  if (!args[i])
    return c10::nullopt;
  return device(i);
}

#ifdef BUILD_NAMEDTENSOR
inline at::Dimname PythonArgs::dimname(int i) {
  TORCH_INTERNAL_ASSERT(args[i] != nullptr);
  return THPDimname_parse(args[i]);
}

inline std::vector<at::Dimname> parseDimnameList(PyObject* arg) {
  auto tuple = PyTuple_Check(arg);
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<at::Dimname> res;
  res.reserve(size);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    res.push_back(THPDimname_parse(obj));
  }
  return res;
}

inline c10::optional<std::vector<at::Dimname>> PythonArgs::toDimnameListOptional(int i) {
  if (!args[i]) return c10::nullopt;
  return parseDimnameList(args[i]);
}

inline std::vector<at::Dimname> PythonArgs::dimnamelist(int i) {
  TORCH_INTERNAL_ASSERT(args[i]);
  PyObject* arg = args[i];
  auto size = signature.params[i].size;
  TORCH_INTERNAL_ASSERT(size == 0 || size == 1);
  if (size == 1 && THPUtils_checkDimname(arg)) {
    return { THPDimname_parse(arg) };
  }
  return parseDimnameList(arg);
}
#endif

inline at::MemoryFormat PythonArgs::memoryformat(int i) {
  if (!args[i]) return at::MemoryFormat::Contiguous;
  TORCH_CHECK(THPMemoryFormat_Check(args[i]), "memory_format arg must be an instance of the torch.memory_format");
  const auto memory_format = reinterpret_cast<THPMemoryFormat*>(args[i]);
  return memory_format->memory_format;
}

inline c10::optional<at::MemoryFormat> PythonArgs::memoryformatOptional(int i) {
  if (!args[i])
    return c10::nullopt;
  return memoryformat(i);
}

inline at::QScheme PythonArgs::toQScheme(int i) {
  if (!args[i]) return at::kPerTensorAffine;
  TORCH_CHECK(THPQScheme_Check(args[i]), "qscheme arg must be an instance of the torch.qscheme");
  const auto qscheme = reinterpret_cast<THPQScheme*>(args[i]);
  return qscheme->qscheme;
}

inline std::string PythonArgs::string(int i) {
  if (!args[i]) return "";
  return THPUtils_unpackString(args[i]);
}

inline int64_t PythonArgs::toInt64(int i) {
  if (!args[i]) return signature.params[i].default_int;
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto & var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, jit::IntType::get());
  }
  return THPUtils_unpackLong(args[i]);
}

inline int64_t PythonArgs::toInt64WithDefault(int i, int64_t default_int) {
  if (!args[i]) return default_int;
  return toInt64(i);
}

inline c10::optional<int64_t> PythonArgs::toInt64Optional(int i) {
  if (!args[i])
    return c10::nullopt;
  return toInt64(i);
}

inline c10::optional<bool> PythonArgs::toBoolOptional(int i) {
  if (!args[i]) {
    return c10::nullopt;
  }
  return toBool(i);
}

inline c10::optional<double> PythonArgs::toDoubleOptional(int i) {
  if (!args[i]) {
    return c10::nullopt;
  }
  return toDouble(i);
}

inline double PythonArgs::toDouble(int i) {
  if (!args[i]) return signature.params[i].default_double;
  return THPUtils_unpackDouble(args[i]);
}

inline double PythonArgs::toDoubleWithDefault(int i, double default_double) {
  if (!args[i]) return default_double;
  return toDouble(i);
}

inline std::complex<double> PythonArgs::toComplex(int i) {
  std::complex<double> default_value = *const_cast<std::complex<double> *>(
    reinterpret_cast<const std::complex<double> *>(signature.params[i].default_complex));
  if (!args[i]) return default_value;
  return THPUtils_unpackComplexDouble(args[i]);
}

inline std::complex<double> PythonArgs::toComplexWithDefault(int i, std::complex<double> default_value) {
  if (!args[i]) return default_value;
  return toDouble(i);
}

inline bool PythonArgs::toBool(int i) {
  if (!args[i]) return signature.params[i].default_bool;
  return args[i] == Py_True;
}

inline bool PythonArgs::toBoolWithDefault(int i, bool default_bool) {
  if (!args[i]) return default_bool;
  return toBool(i);
}

inline bool PythonArgs::isNone(int i) {
  return args[i] == nullptr;
}

inline at::Generator* PythonArgs::generator(int i) {
  if (!args[i]) return nullptr;
  return reinterpret_cast<THPGenerator*>(args[i])->cdata;
}

inline at::Storage PythonArgs::storage(int i) {
  if (!args[i]) return at::Storage();
  return createStorage(args[i]);
}

inline PyObject* PythonArgs::pyobject(int i) {
  if (!args[i]) return Py_None;
  return args[i];
}

} // namespace torch
