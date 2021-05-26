// ${generated_comment}

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// https://github.com/python/cpython/blob/c60394c7fc9cc09b16e9675a3eeb5844b6d8523f/PC/pyconfig.h#L196
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_layouts.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/jit/frontend/tracer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/ATen.h>

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;
using at::ArrayRef;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject* THPVariableFunctionsModule = NULL;

inline Tensor dispatch_arange(const Scalar& end, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::arange_out(result, end);
}

inline Tensor dispatch_arange(const Scalar& end, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::arange(end, options);
}

inline Tensor dispatch_arange(const Scalar& start, const Scalar& end, const Scalar& step, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::arange_out(result, start, end, step);
}

inline Tensor dispatch_arange(const Scalar& start, const Scalar& end, const Scalar& step, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::arange(start, end, step, options);
}

static PyObject * THPVariable_arange(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arange(Scalar end, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "arange(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()) {
    return handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto end = r.scalar(0);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      c10::optional<ScalarType> scalarType = r.scalartypeOptional(2);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(r.device(4))
          .layout(r.layout(3))
          .requires_grad(r.toBool(6))
          .pinned_memory(r.toBool(5));
      return wrap(dispatch_arange(end, options));
    } else {
      TORCH_CHECK(!r.toBool(5), " `pin_memory` and `out` parameters are incompatible");
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2), r.layout(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_arange(r.scalar(0), r.tensor(1)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      auto start = r.scalar(0);
      auto end = r.scalar(1);
      auto step = r.scalar(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      c10::optional<ScalarType> scalarType = r.scalartypeOptional(4);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(r.device(6))
          .layout(r.layout(5))
          .requires_grad(r.toBool(8))
          .pinned_memory(r.toBool(7));
      return wrap(dispatch_arange(start, end, step, options));
    } else {
      TORCH_CHECK(!r.toBool(7), " `pin_memory` and `out` parameters are incompatible");
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4), r.layout(5),
                               r.device(6), r.isNone(6));
      return wrap(dispatch_arange(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_range(const Scalar& start, const Scalar& end, const Scalar& step, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(result));
  return at::range_out(result, start, end, step);
}

inline Tensor dispatch_range(const Scalar& start, const Scalar& end, const Scalar& step, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  DeviceGuard device_guard(options.device());
  return torch::range(start, end, step, options);
}

static PyObject * THPVariable_range(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "range(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto ret = PyErr_WarnEx(
        PyExc_UserWarning,
        "torch.range is deprecated and will be removed in a future release "
        "because its behavior is inconsistent with Python's range builtin. "
        "Instead, use torch.arange, which produces values in [start, end).",
        1);
    if (ret != 0) throw python_error();
    if (r.isNone(3)) {
      const auto options = TensorOptions()
          .dtype(r.scalartype(4))
          .device(r.device(6))
          .layout(r.layout(5))
          .requires_grad(r.toBool(7));
      return wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.device(6), r.isNone(6));
      return wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_full(
    IntArrayRef size,
    const Scalar& fill_val,
    const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return at::full(size, fill_val, options);
}

inline Tensor dispatch_full(
    IntArrayRef size,
    const Scalar& fill_val,
    c10::optional<DimnameList> names,
    const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return at::full(size, fill_val, names, options);
}

inline Tensor dispatch_full(
    IntArrayRef size,
    const Scalar& fill_val,
    Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::full_out(result, size, fill_val);
}

static PyObject * THPVariable_full(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS

  static PythonArgParser parser({
    "full(IntArrayRef size, Scalar fill_value, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "full(IntArrayRef size, Scalar fill_value, *, DimnameList names=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  // Acquires (common) arguments
  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()) {
    return handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  auto size = r.intlist(0);
  auto fill_val = r.scalar(1);
  const auto options = TensorOptions{}
      .dtype(r.scalartypeOptional(3))
      .layout(r.layout(4))
      .device(r.device(5))
      .pinned_memory(r.toBool(6));

  if (r.idx == 0) {
    // full
    if (r.isNone(2)) {
      return wrap(dispatch_full(size, fill_val, options).set_requires_grad(r.toBool(7)));
    }

    // full.out
    // Validates out tensor and other kwargs
    auto result = r.tensor(2);
    TORCH_CHECK(!r.toBool(6), " `pin_memory` and `out` parameters are incompatible");
    check_out_type_matches(result, r.scalartype(3), r.isNone(3), r.layout(4),
                          r.device(5), r.isNone(5));

    return wrap(dispatch_full(size, fill_val, result).set_requires_grad(r.toBool(7)));
  } else if (r.idx == 1) {
    // full.names
    if (r.isNone(2)) {
      return wrap(dispatch_full(size, fill_val, c10::nullopt, options).set_requires_grad(r.toBool(7)));
    }

    // Converts from c10::optional<std:vector...> to c10::optional<ArrayRef...>
    auto raw_names = r.toDimnameListOptional(2);
    c10::optional<DimnameList> names(*raw_names);
    return wrap(dispatch_full(size, fill_val, names, options).set_requires_grad(r.toBool(7)));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_randint(int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size, generator);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size, generator);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, options);
}

static PyObject * THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint(int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/false);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()) {
    return handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    if (r.isNone(3)) {
      auto high = r.toInt64(0);
      auto size = r.intlist(1);
      auto generator = r.generator(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(4, at::ScalarType::Long);
      auto device = r.device(6);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(5))
          .requires_grad(r.toBool(7));
      return wrap(dispatch_randint(high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.device(6), r.isNone(6));
      return wrap(dispatch_randint(r.toInt64(0), r.intlist(1), r.generator(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      auto low = r.toInt64(0);
      auto high = r.toInt64(1);
      auto size = r.intlist(2);
      auto generator = r.generator(3);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(5, at::ScalarType::Long);
      auto device = r.device(7);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(6))
          .requires_grad(r.toBool(8));
      return wrap(dispatch_randint(low, high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(4), r.scalartype(5), r.isNone(5),
                             r.layout(6), r.device(7), r.isNone(7));
      return wrap(dispatch_randint(r.toInt64(0), r.toInt64(1), r.intlist(2), r.generator(3), r.tensor(4)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow torch.as_tensor to be constructed with arbitrarily nested
// python objects - list, tuple, np array, scalar, etc.
static PyObject * THPVariable_as_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.as_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::as_tensor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// implemented on python object here because PyObject currently not natively declarable
// See: ATen/native/README.md for more context
static PyObject * THPVariable_from_numpy(PyObject* module, PyObject* arg)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.from_numpy", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_from_numpy(arg));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static Tensor dispatch_nonzero(const Tensor & self, Tensor out) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return at::nonzero_out(out, self);
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs);

static PyObject * THPVariable__sparse_csr_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch._sparse_csr_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::_sparse_csr_tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_sparse_coo_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__sparse_coo_tensor_unsafe(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch._sparse_coo_tensor_unsafe", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::_sparse_coo_tensor_unsafe_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow torch.tensor to be constructed with arbitrarily nested
// python objects - list, tuple, np array, scalar, etc.
static PyObject * THPVariable_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_get_device(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "get_device(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(r.tensor(0).get_device());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_numel(PyObject* self_, PyObject* args, PyObject* kwargs);

// generated forward declarations start here

${py_forwards}

// Wrapper converts a raised TypeError into returning NotImplemented
// Used to implement binary arithmetic operators
template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
static PyObject * TypeError_to_NotImplemented_(PyObject* self, PyObject* args, PyObject* kwargs) {
  PyObject* ret = Func(self, args, kwargs);
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  return ret;
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
static PyMethodDef torch_functions[] = {
  {"arange", castPyCFunctionWithKeywords(THPVariable_arange),
    METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_tensor", castPyCFunctionWithKeywords(THPVariable_as_tensor),
    METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dsmm", castPyCFunctionWithKeywords(THPVariable_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"from_numpy", THPVariable_from_numpy, METH_STATIC | METH_O, NULL},
  {"full", castPyCFunctionWithKeywords(THPVariable_full), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hsmm", castPyCFunctionWithKeywords(THPVariable_hspmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"nonzero", castPyCFunctionWithKeywords(THPVariable_nonzero), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randint", castPyCFunctionWithKeywords(THPVariable_randint), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"range", castPyCFunctionWithKeywords(THPVariable_range), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"saddmm", castPyCFunctionWithKeywords(THPVariable_sspaddmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sparse_coo_tensor", castPyCFunctionWithKeywords(THPVariable_sparse_coo_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_csr_tensor", castPyCFunctionWithKeywords(THPVariable__sparse_csr_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_coo_tensor_unsafe", castPyCFunctionWithKeywords(THPVariable__sparse_coo_tensor_unsafe), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_validate_sparse_coo_tensor_args", castPyCFunctionWithKeywords(THPVariable__validate_sparse_coo_tensor_args), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"spmm", castPyCFunctionWithKeywords(THPVariable_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tensor", castPyCFunctionWithKeywords(THPVariable_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"get_device", castPyCFunctionWithKeywords(THPVariable_get_device), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"numel", castPyCFunctionWithKeywords(THPVariable_numel), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  ${py_method_defs}
  {NULL}
};

static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableFunctionsClass",    /* tp_name */
  0,                                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  torch_functions,                       /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0                                      /* tp_new */
};

void initTorchFunctions(PyObject* module) {
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);

  // Steals
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(module, "_VariableFunctionsClass", reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule = PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero(Tensor input, *, bool as_tuple=False, Tensor out=None)",
  });
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  const auto as_tuple = r.toBool(1);
  const auto has_out = !r.isNone(2);

  if (as_tuple) {
    TORCH_CHECK(!has_out, "nonzero does not support the out kwarg when as_tuple is True");
    return wrap(dispatch_nonzero_numpy(r.tensor(0)));
  }

  if (has_out) {
    return wrap(dispatch_nonzero(r.tensor(0), r.tensor(2)));
  }

  return wrap(dispatch_nonzero(r.tensor(0)));

  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_numel(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "numel(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    return wrap(r.tensor(0).numel());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
}} // namespace torch::autograd
