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
#include <torch/csrc/Layout.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/python_dimname.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/disable_torch_function.h>
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
  DIMNAME, DIMNAME_LIST, QSCHEME, FLOAT_LIST
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

  // meant only for `torch` functions.
  template<int N>
  inline PythonArgs parse(PyObject* self, PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst);

  template<int N>
  inline PythonArgs parse(PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst);

  inline PythonArgs parse(PyObject* self, ParsedArgs<0>& dst);

  // Formatted strings of non-hidden signatures
  std::vector<std::string> get_signatures() const;

private:
  [[noreturn]]
  void print_error(PyObject* self, PyObject* args, PyObject* kwargs, PyObject* parsed_args[]);
  void check_deprecated(const FunctionSignature & signature);
  PythonArgs raw_parse(PyObject* self, PyObject* args, PyObject* kwargs, PyObject* parsed_args[]);

  std::vector<FunctionSignature> signatures_;
  std::string function_name;
  ssize_t max_args;
  bool traceable;
};

struct PYBIND11_EXPORT FunctionSignature {
  explicit FunctionSignature(const std::string& fmt, int index);

  bool parse(PyObject* self, PyObject* args, PyObject* kwargs, PyObject* dst[], bool raise_exception);

  std::string toString() const;

  std::string name;
  std::vector<FunctionParameter> params;
  std::vector<py::handle> overloaded_args;
  ssize_t min_args;
  ssize_t max_args;
  ssize_t max_pos_args;
  int index;
  bool hidden;
  bool deprecated;
  bool disable_torch_function;
};

struct PythonArgs {
  PythonArgs(bool traceable, const FunctionSignature& signature, PyObject** args)
    : idx(signature.index)
    , traceable(traceable)
    , signature(signature)
    , args(args) {}

  int idx;
  bool traceable;
  const FunctionSignature& signature;
  PyObject** args;

  inline bool has_torch_function();
  inline std::string get_func_name();
  inline at::Tensor tensor(int i);
  inline c10::optional<at::Tensor> optionalTensor(int i);
  inline at::Scalar scalar(int i);
  inline at::Scalar scalarWithDefault(int i, at::Scalar default_scalar);
  inline std::vector<at::Tensor> tensorlist(int i);
  template<int N>
  inline std::array<at::Tensor, N> tensorlist_n(int i);
  inline std::vector<int64_t> intlist(int i);
  inline c10::OptionalArray<int64_t> intlistOptional(int i);
  inline std::vector<int64_t> intlistWithDefault(int i, std::vector<int64_t> default_intlist);
  inline c10::optional<at::Generator> generator(int i);
  inline at::Storage storage(int i);
  inline at::ScalarType scalartype(int i);
  inline at::ScalarType scalartypeWithDefault(int i, at::ScalarType default_scalartype);
  inline c10::optional<at::ScalarType> scalartypeOptional(int i);
  inline c10::optional<at::Scalar> scalarOptional(int i);
  inline c10::optional<int64_t> toInt64Optional(int i);
  inline c10::optional<bool> toBoolOptional(int i);
  inline c10::optional<double> toDoubleOptional(int i);
  inline c10::OptionalArray<double> doublelistOptional(int i);
  inline std::vector<double> doublelist(int i);
  inline std::vector<double> getDoublelist(int i);
  inline at::Layout layout(int i);
  inline at::Layout layoutWithDefault(int i, at::Layout default_layout);
  inline c10::optional<at::Layout> layoutOptional(int i);
  inline at::Device device(int i);
  inline at::Device deviceWithDefault(int i, const at::Device& default_device);
  inline c10::optional<at::Device> deviceOptional(int i);
  inline at::Dimname dimname(int i);
  inline std::vector<at::Dimname> dimnamelist(int i);
  inline c10::optional<std::vector<at::Dimname>> toDimnameListOptional(int i);
  inline at::MemoryFormat memoryformat(int i);
  inline c10::optional<at::MemoryFormat> memoryformatOptional(int i);
  inline at::QScheme toQScheme(int i);
  inline std::string string(int i);
  inline c10::optional<std::string> stringOptional(int i);
  inline PyObject* pyobject(int i);
  inline int64_t toInt64(int i);
  inline int64_t toInt64WithDefault(int i, int64_t default_int);
  inline double toDouble(int i);
  inline double toDoubleWithDefault(int i, double default_double);
  inline c10::complex<double> toComplex(int i);
  inline c10::complex<double> toComplexWithDefault(int i, c10::complex<double> default_complex);
  inline bool toBool(int i);
  inline bool toBoolWithDefault(int i, bool default_bool);
  inline bool isNone(int i);

private:
  at::Tensor tensor_slow(int i);
  at::Scalar scalar_slow(int i);
};

struct FunctionParameter {
  FunctionParameter(const std::string& fmt, bool keyword_only);

  bool check(PyObject* obj, std::vector<py::handle> &overloaded_args, int argnum);

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
    at::Layout default_layout;
  };
};

template<int N>
inline PythonArgs PythonArgParser::parse(PyObject* self, PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst) {
  if (N < max_args) {
    throw ValueError("PythonArgParser: dst ParsedArgs buffer does not have enough capacity, expected %d (got %d)",
        (int)max_args, N);
  }
  return raw_parse(self, args, kwargs, dst.args);
}

template<int N>
inline PythonArgs PythonArgParser::parse(PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst) {
  return parse(nullptr, args, kwargs, dst);
}

inline PythonArgs PythonArgParser::parse(PyObject* self, ParsedArgs<0>& dst) {
  return parse(self, nullptr, nullptr, dst);
}

inline bool PythonArgs::has_torch_function(){
  return !this->signature.overloaded_args.empty();
}

inline std::string PythonArgs::get_func_name(){
  return signature.name;
}

inline at::Tensor PythonArgs::tensor(int i) {
  if (args[i] && THPVariable_CheckExact(args[i])) {
    return reinterpret_cast<THPVariable*>(args[i])->cdata;
  }
  return tensor_slow(i);
}

inline c10::optional<at::Tensor> PythonArgs::optionalTensor(int i) {
  at::Tensor t = tensor(i);
  if (t.defined()) {
    return t;
  } else {
    return c10::nullopt;
  }
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
    // This is checked by the argument parser so it's safe to cast without checking
    // if this is a tensor first
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
    // This is checked by the argument parser so it's safe to cast without checking
    // if this is a tensor first
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

inline c10::OptionalArray<int64_t> PythonArgs::intlistOptional(int i) {
  if (!args[i]) {
    return {};
  }
  return intlist(i);
}

inline std::vector<double> PythonArgs::getDoublelist(int i) {
  PyObject* arg = args[i];
  auto tuple = PyTuple_Check(arg);
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<double> res(size);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    try {
      res[idx] = THPUtils_unpackDouble(obj);
    } catch (const std::exception &e) {
      throw TypeError("%s(): argument '%s' must be %s, but found element of type %s at pos %d",
          signature.name.c_str(), signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(), Py_TYPE(obj)->tp_name, idx + 1);
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
  if (obj == (PyObject*)&PyLong_Type) {
    return at::ScalarType::Long;
  }
  return reinterpret_cast<THPDtype*>(obj)->scalar_type;
}

inline c10::optional<at::ScalarType> PythonArgs::scalartypeOptional(int i) {
  if (!args[i])
    return c10::nullopt;
  return scalartype(i);
}

inline at::Layout PythonArgs::layout(int i) {
  if (!args[i]) return signature.params[i].default_layout;
  return reinterpret_cast<THPLayout*>(args[i])->layout;
}

inline at::Layout PythonArgs::layoutWithDefault(int i, at::Layout default_layout) {
  if (!args[i]) return default_layout;
  return layout(i);
}

inline c10::optional<at::Layout> PythonArgs::layoutOptional(int i) {
  if (!args[i]) return c10::nullopt;
  return layout(i);
}

inline at::Device PythonArgs::device(int i) {
  if (!args[i]) {
    return at::Device(backendToDeviceType(dispatchKeyToBackend(torch::tensors::get_default_dispatch_key())));
  }
  if (THPDevice_Check(args[i])) {
    const auto device = reinterpret_cast<THPDevice*>(args[i]);
    return device->device;
  }
  if (THPUtils_checkLong(args[i])) {
    const auto device_index = THPUtils_unpackLong(args[i]);
    TORCH_CHECK(device_index >= 0, "Device index must not be negative");
    return at::Device(DeviceType::CUDA, device_index);
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

inline c10::optional<std::string> PythonArgs::stringOptional(int i) {
  if (!args[i]) return c10::nullopt;
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

inline c10::complex<double> PythonArgs::toComplex(int i) {
  c10::complex<double> default_value = *const_cast<c10::complex<double> *>(
    reinterpret_cast<const c10::complex<double> *>(signature.params[i].default_complex));
  if (!args[i]) return default_value;
  return THPUtils_unpackComplexDouble(args[i]);
}

inline c10::complex<double> PythonArgs::toComplexWithDefault(int i, c10::complex<double> default_value) {
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

inline c10::optional<at::Generator> PythonArgs::generator(int i) {
  if (!args[i]) return c10::nullopt;
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

/*
 * Reference: https://github.com/numpy/numpy/blob/f4c497c768e0646df740b647782df463825bfd27/numpy/core/src/common/get_attr_string.h#L42
 *
 * Stripped down version of PyObject_GetAttrString,
 * avoids lookups for None, tuple, and List objects,
 * and doesn't create a PyErr since this code ignores it.
 *
 * This can be much faster then PyObject_GetAttrString where
 * exceptions are not used by caller.
 *
 * 'obj' is the object to search for attribute.
 *
 * 'name' is the attribute to search for.
 *
 * Returns a py::object wrapping the return value. If the attribute lookup failed
 * the value will be NULL.
 *
 */

static py::object PyObject_FastGetAttrString(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, name);
        if (res == NULL) {
          PyErr_Clear();
        }
    }
    /* Attribute referenced by (PyObject *)name */
    else if (tp->tp_getattro != NULL) {
        PyObject *w = THPUtils_internString(name);
        if (w == NULL) {
          return py::object();
        }
        res = (*tp->tp_getattro)(obj, w);
        Py_DECREF(w);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    return py::reinterpret_steal<py::object>(res);
}

// Makes sure that we don't check for __torch_function__ on basic Python types
static bool _is_basic_python_type(PyTypeObject *tp)
{
  return (
    /* Basic number types */
    tp == &PyBool_Type ||

    tp == &PyLong_Type ||
    tp == &PyFloat_Type ||
    tp == &PyComplex_Type ||

    /* Basic sequence types */
    tp == &PyList_Type ||
    tp == &PyTuple_Type ||
    tp == &PyDict_Type ||
    tp == &PySet_Type ||
    tp == &PyFrozenSet_Type ||
    tp == &PyUnicode_Type ||
    tp == &PyBytes_Type ||

    /* other builtins */
    tp == &PySlice_Type ||
    tp == Py_TYPE(Py_None) ||
    tp == Py_TYPE(Py_Ellipsis) ||
    tp == Py_TYPE(Py_NotImplemented) ||

    PyModule_Check(tp) ||
    /* sentinel to swallow trailing || */
    false
  );
}

/*
 * Lookup a special method, following the python approach of looking up
 * on the type object, rather than on the instance itself.
 *
 * Assumes that the special method is a torch-specific one, so does not
 * look at builtin types, nor does it look at a base Tensor.
 *
 * If no special method is found, return NULL, otherwise returns a new
 * reference to the function object
 *
 * In future, could be made more like _Py_LookupSpecial
 */

static py::object PyTorch_LookupSpecial(PyObject *obj, char* name)
{
  if (THPVariable_CheckExact(obj)) {
      return py::object();
  }
  PyTypeObject *tp = Py_TYPE(obj);
  if (_is_basic_python_type(tp)) {
    return py::object();
  }
  return PyObject_FastGetAttrString((PyObject *)tp, name);
}

/*
 * Checks if obj has a __torch_function__ implementation
 *
 * Returns true if an implementation is found and false otherwise
 *
 */
static auto check_has_torch_function(PyObject* obj) -> bool
{
  if (!torch_function_enabled()) {
    return false;
  }
  py::object method = PyTorch_LookupSpecial(obj, "__torch_function__");
  if(method.ptr() != nullptr && method.ptr() != disabled_torch_function_impl()){
    return true;
  }
  return false;
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
auto handle_torch_function(PythonArgs &r, PyObject* self, PyObject* args, PyObject* kwargs, PyObject* torch_api, const char* module_name) -> PyObject*;

// Used for functions which needs to parse python args.
auto handle_torch_function(PythonArgs &r, PyObject* args, PyObject* kwargs, PyObject* torch_api, const char* module_name) -> PyObject*;

// Used for functions that accept no keyword arguments and have no argument parsing
auto handle_torch_function(PyObject* self, const std::string& func_name, PyObject* args=nullptr, PyObject* torch_api=THPVariableClass, const std::string& module_name="torch.Tensor") -> PyObject*;

// Used for functions created in C++, e.g., C++ custom op, which doesn't use PythonArgParser to get overloaded_args.
auto handle_torch_function_no_python_arg_parser(const std::vector<py::handle> &overloaded_args, PyObject* args, PyObject* kwargs, const char* func_name, PyObject* torch_api_function, const char* module_name) -> PyObject*;

// Used for getters of Tensor properties
auto handle_torch_function_getter(THPVariable* self, const std::string& property_name) -> PyObject*;

// Used for setters of Tensor properties.
auto handle_torch_function_setter(THPVariable* self, const std::string& property_name, PyObject* value) -> int;

/*
 * Check if the input obj is Tensor type, including its subclass, or overloaded
 * type. If the type defines __torch_function__, it also returns true.
 * Otherwise returns flase. If the class is not torch.Tensor, and it defines
 * __torch_function__, we append obj to overloaded_args.
 *
 * 'obj': the input argument to be checked
 * 'overloaded_args': the vector to append the overloaded args.
 */
bool is_tensor_and_append_overloaded(PyObject* obj, std::vector<py::handle>* overloaded_args);

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
bool is_tensor_list_and_append_overloaded(PyObject* obj, std::vector<py::handle>* overloaded_args, int argnum, bool throw_error);

} // namespace torch
