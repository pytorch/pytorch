// ${generated_comment}

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

#include "python_torch_functions_dispatch.h"

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_layouts.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/structseq.h"

#include <ATen/ATen.h>

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

using at::Tensor;
using at::Device;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;

using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static void check_out_type_matches(Tensor result,
                                   ScalarType scalarType, bool scalarType_is_none,
                                   const THPLayout& layout, bool layout_is_none,
                                   const Device& device, bool device_is_none) {
  if (scalarType_is_none && layout_is_none && device_is_none) {  // common case
    return;
  }
  if (!scalarType_is_none && result.scalar_type() != scalarType) {
    AT_ERROR(
        "dtype ", scalarType,
        " does not match dtype of out parameter (", result.scalar_type(), ")");
  }
  auto scalarType_arg = scalarType_is_none ? result.scalar_type() : scalarType;
  auto layout_arg = layout_is_none ? result.layout() : layout.layout;
  auto device_type_arg = device_is_none ? result.device().type() : device.type();
  if (result.scalar_type() != scalarType_arg) {
    AT_ERROR(
        "scalar type ", scalarType_arg,
        " does not match scalar type of out parameter (", result.scalar_type(), ")");
  }
  if (result.layout() != layout_arg) {
    AT_ERROR(
        "layout ", layout_arg,
        " does not match layout of out parameter (", result.layout(), ")");
  }
  if (result.device().type() != device_type_arg) {
    AT_ERROR(
        "device type ", device_type_arg,
        " does not match device type of out parameter (", result.device().type(), ")");
  }
}

inline Tensor dispatch_arange(Scalar end, Tensor result) {
  AutoNoGIL no_gil;
  return at::arange_out(result, end);
}

inline Tensor dispatch_arange(Scalar end, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::arange(end, options);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, Tensor result) {
  AutoNoGIL no_gil;
  return at::arange_out(result, start, end, step);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::arange(start, end, step, options);
}

static inline bool allIntegral(std::initializer_list<std::reference_wrapper<Scalar>> l) {
  for (Scalar& s : l) {
    if (!s.isIntegral(true)) {
      return false;
    }
  }
  return true;
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

  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto end = r.scalar(0);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto scalarType = r.isNone(2) && allIntegral({end}) ? at::ScalarType::Long : r.scalartype(2);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(r.device(4))
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(6))
          .pinned_memory(r.toBool(5));
      return wrap(dispatch_arange(end, options));
    } else {
      TORCH_CHECK(!r.toBool(5), " `pin_memory` and `out` parameters are incompatible");
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2), r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_arange(r.scalar(0), r.tensor(1)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      auto start = r.scalar(0);
      auto end = r.scalar(1);
      auto step = r.scalar(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto scalarType = r.isNone(4) && allIntegral({start, end, step}) ? at::ScalarType::Long : r.scalartype(4);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(r.device(6))
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(8))
          .pinned_memory(r.toBool(7));
      return wrap(dispatch_arange(start, end, step, options));
    } else {
      TORCH_CHECK(!r.toBool(7), " `pin_memory` and `out` parameters are incompatible");
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4), r.layout(5), r.isNone(5),
                               r.device(6), r.isNone(6));
      return wrap(dispatch_arange(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_range(Scalar start, Scalar end, Scalar step, Tensor result) {
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(result));
  return at::range_out(result, start, end, step);
}

inline Tensor dispatch_range(Scalar start, Scalar end, Scalar step, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
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
    PyErr_WarnEx(PyExc_UserWarning, "torch.range is deprecated in favor of torch.arange "
        "and will be removed in 0.5. Note that arange generates values in [start; end), "
        "not [start; end].", 1);
    if (r.isNone(3)) {
      const auto options = TensorOptions()
          .dtype(r.scalartype(4))
          .device(r.device(6))
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(7));
      return wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.isNone(5),
                             r.device(6), r.isNone(6));
      return wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Generator * generator, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, high, size, generator);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, high, size);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(high, size, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, low, high, size, generator);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(low, high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, low, high, size);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(low, high, size, options);
}

static PyObject * THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint(int64_t high, IntArrayRef size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t high, IntArrayRef size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntArrayRef size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntArrayRef size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/false);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
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
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(7));
      return wrap(dispatch_randint(high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.isNone(5),
                             r.device(6), r.isNone(6));
      return wrap(dispatch_randint(r.toInt64(0), r.intlist(1), r.generator(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      auto high = r.toInt64(0);
      auto size = r.intlist(1);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(3, at::ScalarType::Long);
      auto device = r.device(5);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(4).layout)
          .requires_grad(r.toBool(6));
      return wrap(dispatch_randint(high, size, options));
    } else {
      check_out_type_matches(r.tensor(2), r.scalartype(3), r.isNone(3),
                             r.layout(4), r.isNone(4),
                             r.device(5), r.isNone(5));
      return wrap(dispatch_randint(r.toInt64(0), r.intlist(1), r.tensor(2)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 2) {
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
          .layout(r.layout(6).layout)
          .requires_grad(r.toBool(8));
      return wrap(dispatch_randint(low, high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(4), r.scalartype(5), r.isNone(5),
                             r.layout(6), r.isNone(6),
                             r.device(7), r.isNone(7));
      return wrap(dispatch_randint(r.toInt64(0), r.toInt64(1), r.intlist(2), r.generator(3), r.tensor(4)).set_requires_grad(r.toBool(8)));
    }
  } else if (r.idx == 3) {
    if (r.isNone(3)) {
      auto low = r.toInt64(0);
      auto high = r.toInt64(1);
      auto size = r.intlist(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(4, at::ScalarType::Long);
      auto device = r.device(6);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(7));
      return wrap(dispatch_randint(low, high, size, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.isNone(5),
                             r.device(6), r.isNone(6));
      return wrap(dispatch_randint(r.toInt64(0), r.toInt64(1), r.intlist(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_as_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.as_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::as_tensor(torch::tensors::get_default_tensor_type_id(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_from_numpy(PyObject* module, PyObject* arg)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.from_numpy", jit::tracer::WARN_CONSTRUCTOR);
  auto data = torch::utils::tensor_from_numpy(arg);
  return THPVariable_Wrap(make_variable(std::move(data), /*requires_grad=*/false));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__promote_types(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_promote_types(ScalarType type1, ScalarType type2)",
  });
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    ScalarType promoted = at::promoteTypes(r.scalartype(0), r.scalartype(1));
    return torch::autograd::utils::wrap(torch::getDtype(promoted));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor & self) {
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static Tensor dispatch_nonzero(const Tensor & self, Tensor out) {
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return at::nonzero_out(out, self);
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero(Tensor input, *, Tensor out=None)|deprecated",
    "nonzero(Tensor input, *, bool as_tuple)",
  });
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_nonzero(r.tensor(0)));
    } else {
      return wrap(dispatch_nonzero(r.tensor(0), r.tensor(1)));
    }
  } else {
    if (r.toBool(1)) {
      return wrap(dispatch_nonzero_numpy(r.tensor(0)));
    } else {
      return wrap(dispatch_nonzero(r.tensor(0)));
    }
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_sparse_coo_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(torch::tensors::get_default_tensor_type_id(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_ctor(torch::tensors::get_default_tensor_type_id(), torch::tensors::get_default_scalar_type(), args, kwargs));
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

// generated methods start here

${py_methods}

static PyMethodDef torch_functions[] = {
  {"arange", (PyCFunction)(void(*)(void))THPVariable_arange, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_tensor", (PyCFunction)(void(*)(void))THPVariable_as_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dsmm", (PyCFunction)(void(*)(void))THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"from_numpy", (PyCFunction)THPVariable_from_numpy, METH_STATIC | METH_O, NULL},
  {"hsmm", (PyCFunction)(void(*)(void))THPVariable_hspmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_promote_types", (PyCFunction)(void(*)(void))THPVariable__promote_types, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"nonzero", (PyCFunction)(void(*)(void))THPVariable_nonzero, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randint", (PyCFunction)(void(*)(void))THPVariable_randint, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"range", (PyCFunction)(void(*)(void))THPVariable_range, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"saddmm", (PyCFunction)(void(*)(void))THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sparse_coo_tensor", (PyCFunction)(void(*)(void))THPVariable_sparse_coo_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"spmm", (PyCFunction)(void(*)(void))THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tensor", (PyCFunction)(void(*)(void))THPVariable_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"get_device", (PyCFunction)(void(*)(void))THPVariable_get_device, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  ${py_method_defs}
  {NULL}
};

static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableFunctions",         /* tp_name */
  0,                                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
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
  if (PyModule_AddObject(module, "_VariableFunctions", (PyObject*)&THPVariableFunctions) < 0) {
    throw python_error();
  }
}

}} // namespace torch::autograd
