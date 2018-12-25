// @generated from tools/autograd/templates/python_torch_functions.cpp

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
  auto scalarType_arg = scalarType_is_none ? result.type().scalarType() : scalarType;
  auto layout_arg = layout_is_none ? *torch::getLayout(result.type().backend()) : layout;
  auto device_type_arg = device_is_none ? torch::getDeviceType(result.type()) : device.type();
  const auto& type = torch::getVariableType(scalarType_arg, layout_arg, device_type_arg);
  if (result.type() != type) {
    AT_ERROR(
        "type corresponding to ", type.toString(),
        " does not match type of out parameter (", result.type().toString(), ")");
  }
}

inline Tensor dispatch_arange(Scalar end, Tensor result) {
  AutoNoGIL no_gil;
  return at::arange_out(result, end);
}

inline Tensor dispatch_arange(Scalar end, const TensorOptions& options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::arange(end, options);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, Tensor result) {
  AutoNoGIL no_gil;
  return at::arange_out(result, start, end, step);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, const TensorOptions& options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::arange(start, end, step, options);
}

static inline bool allIntegral(std::initializer_list<std::reference_wrapper<Scalar>> l) {
  for (Scalar& s : l) {
    if (!s.isIntegral()) {
      return false;
    }
  }
  return true;
}

static PyObject * THPVariable_arange(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arange(Scalar end, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "arange(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
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
          .requires_grad(r.toBool(5));
      return wrap(dispatch_arange(end, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2), r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_arange(r.scalar(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
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
          .requires_grad(r.toBool(7));
      return wrap(dispatch_arange(start, end, step, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4), r.layout(5), r.isNone(5),
                               r.device(6), r.isNone(6));
      return wrap(dispatch_arange(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
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
  maybe_initialize_cuda(options);
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

inline Tensor dispatch_randint(int64_t high, IntList size, Generator * generator, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, high, size, generator);
}
inline Tensor dispatch_randint(int64_t high, IntList size, Generator * generator, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t high, IntList size, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, high, size);
}
inline Tensor dispatch_randint(int64_t high, IntList size, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(high, size, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntList size, Generator * generator, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, low, high, size, generator);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntList size, Generator * generator, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(low, high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntList size, Tensor result) {
  AutoNoGIL no_gil;
  return at::randint_out(result, low, high, size);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntList size, const TensorOptions & options) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::randint(low, high, size, options);
}

static PyObject * THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint(int64_t high, IntList size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t high, IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntList size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
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
  return THPVariable_Wrap(torch::utils::as_tensor(default_type(), args, kwargs));
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

static PyObject * THPVariable_sparse_coo_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(default_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_ctor(default_type(), args, kwargs));
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

static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__and__(Tensor input, Tensor other)",
    "__and__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___and__(r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch___and__(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___lshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__lshift__(Tensor input, Tensor other)",
    "__lshift__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___lshift__(r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch___lshift__(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___or__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__or__(Tensor input, Tensor other)",
    "__or__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___or__(r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch___or__(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___rshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__rshift__(Tensor input, Tensor other)",
    "__rshift__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___rshift__(r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch___rshift__(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___xor__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__xor__(Tensor input, Tensor other)",
    "__xor__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___xor__(r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch___xor__(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__argmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_argmax(Tensor input, int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__argmax(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__argmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_argmin(Tensor input, int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__argmin(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__baddbmm_mkl_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_baddbmm_mkl_(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__baddbmm_mkl_(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Byte(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Byte(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Byte(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Char(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Char(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Char(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Double(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Double(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Double(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Float(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Float(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Float(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Half(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Half(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Half(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Int(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Int(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Int(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Long(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Long(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Long(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cast_Short(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Short(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cast_Short(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution(Tensor input, Tensor weight, Tensor? bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<12> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toBool(6), r.intlist(7), r.toInt64(8), r.toBool(9), r.toBool(10), r.toBool(11)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__convolution_nogroup(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__convolution_nogroup(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toBool(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__copy_same_type_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_copy_same_type_(Tensor input, Tensor src)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    dispatch__copy_same_type_(r.tensor(0), r.tensor(1));
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_ctc_loss(Tensor log_probs, Tensor targets, IntList input_lengths, IntList target_lengths, int64_t blank=0)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__ctc_loss(r.tensor(0), r.tensor(1), r.intlist(2), r.intlist(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_ctc_loss(Tensor log_probs, Tensor targets, IntList input_lengths, IntList target_lengths, int64_t blank, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cudnn_ctc_loss(r.tensor(0), r.tensor(1), r.intlist(2), r.intlist(3), r.toInt64(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cudnn_init_dropout_state(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto dropout = r.toDouble(0);
    auto train = r.toBool(1);
    auto dropout_seed = r.toInt64(2);
    auto dtype = r.scalartype(3);
    auto device = r.device(5);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(4).layout)
        .requires_grad(r.toBool(6));
    return wrap(dispatch__cudnn_init_dropout_state(dropout, train, dropout_seed, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cudnn_rnn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn(Tensor input, TensorList weight, int64_t weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, Tensor? dropout_state)",
  }, /*traceable=*/true);

  ParsedArgs<15> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cudnn_rnn(r.tensor(0), r.tensorlist(1), r.toInt64(2), r.tensor(3), r.tensor(4), r.tensor(5), r.toInt64(6), r.toInt64(7), r.toInt64(8), r.toBool(9), r.toDouble(10), r.toBool(11), r.toBool(12), r.intlist(13), r.tensor(14)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cudnn_rnn_flatten_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cudnn_rnn_flatten_weight(r.tensorlist(0), r.toInt64(1), r.toInt64(2), r.toInt64(3), r.toInt64(4), r.toInt64(5), r.toBool(6), r.toBool(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cufft_clear_plan_cache(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_clear_plan_cache()",
  }, /*traceable=*/false);

  ParsedArgs<0> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    dispatch__cufft_clear_plan_cache();
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cufft_get_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_get_plan_cache_max_size()",
  }, /*traceable=*/false);

  ParsedArgs<0> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cufft_get_plan_cache_max_size());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cufft_get_plan_cache_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_get_plan_cache_size()",
  }, /*traceable=*/false);

  ParsedArgs<0> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__cufft_get_plan_cache_size());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__cufft_set_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_set_plan_cache_max_size(int64_t max_size)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    dispatch__cufft_set_plan_cache_max_size(r.toInt64(0));
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dim_arange(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dim_arange(Tensor like, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__dim_arange(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dirichlet_grad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dirichlet_grad(Tensor x, Tensor alpha, Tensor total, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch__dirichlet_grad(r.tensor(0), r.tensor(1), r.tensor(2)));
    } else {
      return wrap(dispatch__dirichlet_grad(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__embedding_bag(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toInt64(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__fft_with_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fft_with_size(Tensor input, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__fft_with_size(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3), r.toBool(4), r.intlist(5), r.toBool(6), r.toBool(7), r.intlist(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__fused_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_dropout(Tensor input, double p, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__fused_dropout(r.tensor(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_log_softmax(Tensor input, int64_t dim, bool half_to_float)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__log_softmax(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__log_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_log_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__log_softmax_backward_data(r.tensor(0), r.tensor(1), r.toInt64(2), r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__masked_scale(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_masked_scale(Tensor input, Tensor mask, double scale)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__masked_scale(r.tensor(0), r.tensor(1), r.toDouble(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__pack_padded_sequence(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__pack_padded_sequence(r.tensor(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__pad_packed_sequence(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__pad_packed_sequence(r.tensor(0), r.tensor(1), r.toBool(2), r.scalar(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__reshape_from_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_reshape_from_tensor(Tensor input, Tensor shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__reshape_from_tensor(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__s_copy_from(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_s_copy_from(Tensor input, Tensor dst, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__s_copy_from(r.tensor(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__s_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_s_where(Tensor condition, Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__s_where(r.tensor(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__shape_as_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_shape_as_tensor(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__shape_as_tensor(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_softmax(Tensor input, int64_t dim, bool half_to_float)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__softmax(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__softmax_backward_data(r.tensor(0), r.tensor(1), r.toInt64(2), r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sparse_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_addmm(Tensor input, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__sparse_addmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sparse_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_mm(Tensor sparse, Tensor dense)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__sparse_mm(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sparse_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_sum(Tensor input)",
    "_sparse_sum(Tensor input, *, ScalarType dtype)",
    "_sparse_sum(Tensor input, IntList[1] dim)",
    "_sparse_sum(Tensor input, IntList[1] dim, *, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__sparse_sum(r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch__sparse_sum(r.tensor(0), r.scalartype(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch__sparse_sum(r.tensor(0), r.intlist(1)));
  } else if (r.idx == 3) {
    return wrap(dispatch__sparse_sum(r.tensor(0), r.intlist(1), r.scalartype(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__standard_gamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma(Tensor input, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__standard_gamma(r.tensor(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__standard_gamma_grad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma_grad(Tensor input, Tensor output)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__standard_gamma_grad(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__trilinear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_trilinear(Tensor i1, Tensor i2, Tensor i3, IntList expand1, IntList expand2, IntList expand3, IntList sumdim, int64_t unroll_dim=1)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__trilinear(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.intlist(6), r.toInt64(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__unique(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unique(Tensor input, bool sorted=False, bool return_inverse=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__unique(r.tensor(0), r.toBool(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__unique_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unique_dim(Tensor input, int64_t dim, bool sorted=False, bool return_inverse=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__unique_dim(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__weight_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_norm(Tensor v, Tensor g, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__weight_norm(r.tensor(0), r.tensor(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__weight_norm_cuda_interface(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_norm_cuda_interface(Tensor v, Tensor g, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__weight_norm_cuda_interface(r.tensor(0), r.tensor(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_abs(r.tensor(0)));
    } else {
      return wrap(dispatch_abs(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_abs_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acos(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_acos(r.tensor(0)));
    } else {
      return wrap(dispatch_acos(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acos_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_acos_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool1d(Tensor input, IntList[1] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_adaptive_avg_pool1d(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool1d(Tensor input, IntList[1] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_adaptive_max_pool1d(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "add(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_add(r.tensor(0), r.scalar(1), r.tensor(2)));
    } else {
      return wrap(dispatch_add(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_add(r.tensor(0), r.tensor(1), r.scalar(2)));
    } else {
      return wrap(dispatch_add(r.tensor(0), r.tensor(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcdiv(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcdiv(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addcdiv(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcdiv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3)));
    } else {
      return wrap(dispatch_addcdiv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcmul(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcmul(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcmul(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addcmul(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addcmul(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3)));
    } else {
      return wrap(dispatch_addcmul(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Scalar beta, Tensor input, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addmv(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addmv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addmv(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv_(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor input, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addmv_(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmv_(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmv_(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addr(Scalar beta, Tensor input, Scalar alpha, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Scalar beta, Tensor input, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Tensor input, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_addr(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_addr(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_addr(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "affine_grid_generator(Tensor theta, IntList size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_affine_grid_generator(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "all(Tensor input)",
    "all(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_all(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_all(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch_all(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "allclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_allclose(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "alpha_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_alpha_dropout(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "alpha_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_alpha_dropout_(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "any(Tensor input)",
    "any(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_any(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_any(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch_any(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided(Tensor input, IntList size, IntList stride)",
    "as_strided(Tensor input, IntList size, IntList stride, int64_t storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_as_strided(r.tensor(0), r.intlist(1), r.intlist(2)));
  } else if (r.idx == 1) {
    auto self = r.tensor(0);
    auto size = r.intlist(1);
    auto stride = r.intlist(2);
    auto storage_offset = r.toInt64WithDefault(3, self.storage_offset());
    return wrap(dispatch_as_strided(self, size, stride, storage_offset));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided_(Tensor input, IntList size, IntList stride)",
    "as_strided_(Tensor input, IntList size, IntList stride, int64_t storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_as_strided_(r.tensor(0), r.intlist(1), r.intlist(2)));
  } else if (r.idx == 1) {
    auto self = r.tensor(0);
    auto size = r.intlist(1);
    auto stride = r.intlist(2);
    auto storage_offset = r.toInt64WithDefault(3, self.storage_offset());
    return wrap(dispatch_as_strided_(self, size, stride, storage_offset));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asin(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_asin(r.tensor(0)));
    } else {
      return wrap(dispatch_asin(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asin_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_asin_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_atan(r.tensor(0)));
    } else {
      return wrap(dispatch_atan(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_atan2(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_atan2(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_atan_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool1d(Tensor input, IntList[1] kernel_size, IntList[1] stride=None, IntList[1] padding=0, bool ceil_mode=False, bool count_include_pad=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_avg_pool1d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.toBool(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
    } else {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4), r.tensor(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    } else {
      return wrap(dispatch_baddbmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_baddbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_baddbmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bartlett_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bartlett_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "bartlett_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto window_length = r.toInt64(0);
    auto dtype = r.scalartype(1);
    auto device = r.device(3);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(2).layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_bartlett_window(window_length, options));
  } else if (r.idx == 1) {
    auto window_length = r.toInt64(0);
    auto periodic = r.toBool(1);
    auto dtype = r.scalartype(2);
    auto device = r.device(4);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(3).layout)
        .requires_grad(r.toBool(5));
    return wrap(dispatch_bartlett_window(window_length, periodic, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bernoulli(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli(Tensor input, *, Generator generator=None, Tensor out=None)",
    "bernoulli(Tensor input, double p, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_bernoulli(r.tensor(0), r.generator(1)));
    } else {
      return wrap(dispatch_bernoulli(r.tensor(0), r.generator(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    return wrap(dispatch_bernoulli(r.tensor(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bilinear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_bilinear(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_binary_cross_entropy_with_logits(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "binary_cross_entropy_with_logits(Tensor input, Tensor target, Tensor? weight, Tensor? pos_weight, int64_t reduction)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_binary_cross_entropy_with_logits(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bincount(Tensor input, Tensor? weights=None, int64_t minlength=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_bincount(r.tensor(0), r.tensor(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_blackman_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "blackman_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "blackman_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto window_length = r.toInt64(0);
    auto dtype = r.scalartype(1);
    auto device = r.device(3);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(2).layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_blackman_window(window_length, options));
  } else if (r.idx == 1) {
    auto window_length = r.toInt64(0);
    auto periodic = r.toBool(1);
    auto dtype = r.scalartype(2);
    auto device = r.device(4);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(3).layout)
        .requires_grad(r.toBool(5));
    return wrap(dispatch_blackman_window(window_length, periodic, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bmm(Tensor input, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_bmm(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_bmm(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_broadcast_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "broadcast_tensors(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_broadcast_tensors(r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrifact(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrifact(Tensor input, *, bool pivot=True, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_btrifact(r.tensor(0), r.toBool(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_btrifact(r.tensor(0), r.toBool(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrifact_with_info(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrifact_with_info(Tensor input, *, bool pivot=True, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_btrifact_with_info(r.tensor(0), r.toBool(1)));
    } else {
      auto results = r.tensorlist_n<3>(2);
      return wrap(dispatch_btrifact_with_info(r.tensor(0), r.toBool(1), results[0], results[1], results[2]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrisolve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrisolve(Tensor input, Tensor LU_data, Tensor LU_pivots, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_btrisolve(r.tensor(0), r.tensor(1), r.tensor(2)));
    } else {
      return wrap(dispatch_btrisolve(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cat(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_cat(r.tensorlist(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_cat(r.tensorlist(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_ceil(r.tensor(0)));
    } else {
      return wrap(dispatch_ceil(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ceil_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_celu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "celu(Tensor input, Scalar alpha=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_celu(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_celu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "celu_(Tensor input, Scalar alpha=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_celu_(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_chain_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chain_matmul(TensorList matrices)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_chain_matmul(r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky(Tensor input, bool upper=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_cholesky(r.tensor(0), r.toBool(1)));
    } else {
      return wrap(dispatch_cholesky(r.tensor(0), r.toBool(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chunk(Tensor input, int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_chunk(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Tensor input, Scalar? min=c10::nullopt, Scalar? max=c10::nullopt, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_clamp(r.tensor(0), r.scalarOptional(1), r.scalarOptional(2)));
    } else {
      return wrap(dispatch_clamp(r.tensor(0), r.scalarOptional(1), r.scalarOptional(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_(Tensor input, Scalar? min=c10::nullopt, Scalar? max=c10::nullopt)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_clamp_(r.tensor(0), r.scalarOptional(1), r.scalarOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_max(Tensor input, Scalar max, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_clamp_max(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_clamp_max(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_max_(Tensor input, Scalar max)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_clamp_max_(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_min(Tensor input, Scalar min, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_clamp_min(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_clamp_min(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_min_(Tensor input, Scalar min)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_clamp_min_(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clone(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_clone(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_constant_pad_nd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "constant_pad_nd(Tensor input, IntList pad, Scalar value=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_constant_pad_nd(r.tensor(0), r.intlist(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv1d(Tensor input, Tensor weight, Tensor? bias=None, IntList[1] stride=1, IntList[1] padding=0, IntList[1] dilation=1, int64_t groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv1d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, IntList[2] stride=1, IntList[2] padding=0, IntList[2] dilation=1, int64_t groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv3d(Tensor input, Tensor weight, Tensor? bias=None, IntList[3] stride=1, IntList[3] padding=0, IntList[3] dilation=1, int64_t groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv3d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_tbc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_tbc(Tensor input, Tensor weight, Tensor bias, int64_t pad=0)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_tbc(r.tensor(0), r.tensor(1), r.tensor(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_transpose1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, IntList[1] stride=1, IntList[1] padding=0, IntList[1] output_padding=0, int64_t groups=1, IntList[1] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_transpose1d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose2d(Tensor input, Tensor weight, Tensor? bias=None, IntList[2] stride=1, IntList[2] padding=0, IntList[2] output_padding=0, int64_t groups=1, IntList[2] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_transpose2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose3d(Tensor input, Tensor weight, Tensor? bias=None, IntList[3] stride=1, IntList[3] padding=0, IntList[3] output_padding=0, int64_t groups=1, IntList[3] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_transpose3d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "convolution(Tensor input, Tensor weight, Tensor? bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups)",
  }, /*traceable=*/false);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toBool(6), r.intlist(7), r.toInt64(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_cos(r.tensor(0)));
    } else {
      return wrap(dispatch_cos(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cos_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_cosh(r.tensor(0)));
    } else {
      return wrap(dispatch_cosh(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cosh_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosine_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, int64_t reduction=Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cosine_embedding_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toDouble(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosine_similarity(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosine_similarity(Tensor x1, Tensor x2, int64_t dim=1, double eps=1e-08)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cosine_similarity(r.tensor(0), r.tensor(1), r.toInt64(2), r.toDouble(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cross(Tensor input, Tensor other, int64_t dim=-1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_cross(r.tensor(0), r.tensor(1), r.toInt64(2)));
    } else {
      return wrap(dispatch_cross(r.tensor(0), r.tensor(1), r.toInt64(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ctc_loss(Tensor log_probs, Tensor targets, IntList input_lengths, IntList target_lengths, int64_t blank=0, int64_t reduction=Reduction::Mean)",
    "ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int64_t blank=0, int64_t reduction=Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ctc_loss(r.tensor(0), r.tensor(1), r.intlist(2), r.intlist(3), r.toInt64(4), r.toInt64(5)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ctc_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.toInt64(4), r.toInt64(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_affine_grid_generator(Tensor theta, int64_t N, int64_t C, int64_t H, int64_t W)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_affine_grid_generator(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toInt64(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double exponential_average_factor, double epsilon)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_convolution_transpose(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.intlist(6), r.toInt64(7), r.toBool(8), r.toBool(9)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_grid_sampler(Tensor input, Tensor grid)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_grid_sampler(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cudnn_is_acceptable(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_is_acceptable(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cudnn_is_acceptable(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumprod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumprod(Tensor input, int64_t dim, *, Tensor out=None)",
    "cumprod(Tensor input, int64_t dim, *, ScalarType dtype, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_cumprod(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_cumprod(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_cumprod(r.tensor(0), r.toInt64(1), r.scalartype(2)));
    } else {
      return wrap(dispatch_cumprod(r.tensor(0), r.toInt64(1), r.scalartype(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumsum(Tensor input, int64_t dim, *, Tensor out=None)",
    "cumsum(Tensor input, int64_t dim, *, ScalarType dtype, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_cumsum(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_cumsum(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_cumsum(r.tensor(0), r.toInt64(1), r.scalartype(2)));
    } else {
      return wrap(dispatch_cumsum(r.tensor(0), r.toInt64(1), r.scalartype(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_det(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "det(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_det(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "detach(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_detach(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "detach_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_detach_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_diag(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_diag(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag_embed(Tensor input, int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_diag_embed(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagflat(Tensor input, int64_t offset=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_diagflat(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagonal(Tensor input, int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_diagonal(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "digamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_digamma(r.tensor(0)));
    } else {
      return wrap(dispatch_digamma(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dist(Tensor input, Tensor other, Scalar p=2)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_dist(r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_div(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_div(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dot(Tensor input, Tensor tensor, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_dot(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_dot(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_dropout(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_dropout_(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eig(Tensor input, bool eigenvectors=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_eig(r.tensor(0), r.toBool(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_eig(r.tensor(0), r.toBool(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_einsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "einsum(std::string equation, TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_einsum(r.string(0), r.tensorlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_embedding(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding(Tensor weight, Tensor indices, int64_t padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_embedding(r.tensor(0), r.tensor(1), r.toInt64(2), r.toBool(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_embedding_bag(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toInt64(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_embedding_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_renorm_(Tensor input, Tensor indices, double max_norm, double norm_type)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_embedding_renorm_(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_empty(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto size = r.intlist(0);
      auto dtype = r.scalartype(2);
      auto device = r.device(4);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(5));
      return wrap(dispatch_empty(size, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                             r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_empty(r.intlist(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_empty_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "empty_like(Tensor input, *, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_empty_like(self, options));
  } else if (r.idx == 1) {
    return wrap(dispatch_empty_like(r.tensor(0)).set_requires_grad(r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_strided(IntList size, IntList stride, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto size = r.intlist(0);
    auto stride = r.intlist(1);
    auto dtype = r.scalartype(2);
    auto device = r.device(4);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(3).layout)
        .requires_grad(r.toBool(5));
    return wrap(dispatch_empty_strided(size, stride, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eq(Tensor input, Tensor other, *, Tensor out=None)",
    "eq(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_eq(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_eq(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_eq(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_eq(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "equal(Tensor input, Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_equal(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_erf(r.tensor(0)));
    } else {
      return wrap(dispatch_erf(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erf_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_erf_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_erfc(r.tensor(0)));
    } else {
      return wrap(dispatch_erfc(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfc_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_erfc_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfinv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_erfinv(r.tensor(0)));
    } else {
      return wrap(dispatch_erfinv(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_exp(r.tensor(0)));
    } else {
      return wrap(dispatch_exp(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_exp_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expm1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_expm1(r.tensor(0)));
    } else {
      return wrap(dispatch_expm1(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expm1_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_expm1_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eye(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eye(int64_t n, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "eye(int64_t n, int64_t m, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto n = r.toInt64(0);
      auto dtype = r.scalartype(2);
      auto device = r.device(4);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(5));
      return wrap(dispatch_eye(n, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                             r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_eye(r.toInt64(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      auto n = r.toInt64(0);
      auto m = r.toInt64(1);
      auto dtype = r.scalartype(3);
      auto device = r.device(5);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(4).layout)
          .requires_grad(r.toBool(6));
      return wrap(dispatch_eye(n, m, options));
    } else {
      check_out_type_matches(r.tensor(2), r.scalartype(3), r.isNone(3),
                             r.layout(4), r.isNone(4),
                             r.device(5), r.isNone(5));
      return wrap(dispatch_eye(r.toInt64(0), r.toInt64(1), r.tensor(2)).set_requires_grad(r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_feature_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_alpha_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_feature_alpha_dropout(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_feature_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_alpha_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_feature_alpha_dropout_(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_feature_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_feature_dropout(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_feature_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_feature_dropout_(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft(Tensor input, int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_fft(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fill_(Tensor input, Tensor value)",
    "fill_(Tensor input, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_fill_(r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_fill_(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flatten(Tensor input, int64_t start_dim=0, int64_t end_dim=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_flatten(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flip(Tensor input, IntList dims)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_flip(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_floor(r.tensor(0)));
    } else {
      return wrap(dispatch_floor(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_floor_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod(Tensor input, Tensor other, *, Tensor out=None)",
    "fmod(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_fmod(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_fmod(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_fmod(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_fmod(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frac(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_frac(r.tensor(0)));
    } else {
      return wrap(dispatch_frac(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_frobenius_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frobenius_norm(Tensor input)",
    "frobenius_norm(Tensor input, IntList[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_frobenius_norm(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_frobenius_norm(r.tensor(0), r.intlist(1), r.toBool(2)));
    } else {
      return wrap(dispatch_frobenius_norm(r.tensor(0), r.intlist(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_full(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full(IntList size, Scalar fill_value, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto size = r.intlist(0);
      auto fill_value = r.scalar(1);
      auto dtype = r.scalartype(3);
      auto device = r.device(5);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(4).layout)
          .requires_grad(r.toBool(6));
      return wrap(dispatch_full(size, fill_value, options));
    } else {
      check_out_type_matches(r.tensor(2), r.scalartype(3), r.isNone(3),
                             r.layout(4), r.isNone(4),
                             r.device(5), r.isNone(5));
      return wrap(dispatch_full(r.intlist(0), r.scalar(1), r.tensor(2)).set_requires_grad(r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_full_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full_like(Tensor input, Scalar fill_value, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "full_like(Tensor input, Scalar fill_value, *, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto fill_value = r.scalar(1);
    auto dtype = r.scalartypeWithDefault(2, self.type().scalarType());
    auto layout = r.layoutWithDefault(3, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(4, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(5));
    return wrap(dispatch_full_like(self, fill_value, options));
  } else if (r.idx == 1) {
    return wrap(dispatch_full_like(r.tensor(0), r.scalar(1)).set_requires_grad(r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gather(Tensor input, int64_t dim, Tensor index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_gather(r.tensor(0), r.toInt64(1), r.tensor(2)));
    } else {
      return wrap(dispatch_gather(r.tensor(0), r.toInt64(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ge(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ge(Tensor input, Tensor other, *, Tensor out=None)",
    "ge(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_ge(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_ge(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_ge(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_ge(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gels(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gels(Tensor input, Tensor A, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_gels(r.tensor(0), r.tensor(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_gels(r.tensor(0), r.tensor(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "geqrf(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_geqrf(r.tensor(0)));
    } else {
      auto results = r.tensorlist_n<2>(1);
      return wrap(dispatch_geqrf(r.tensor(0), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ger(Tensor input, Tensor vec2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_ger(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_ger(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gesv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gesv(Tensor input, Tensor A, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_gesv(r.tensor(0), r.tensor(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_gesv(r.tensor(0), r.tensor(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_grid_sampler(r.tensor(0), r.tensor(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_grid_sampler_2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler_2d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_grid_sampler_2d(r.tensor(0), r.tensor(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_grid_sampler_3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler_3d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_grid_sampler_3d(r.tensor(0), r.tensor(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_group_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "group_norm(Tensor input, int64_t num_groups, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enabled=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_group_norm(r.tensor(0), r.toInt64(1), r.tensor(2), r.tensor(3), r.toDouble(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gru(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gru(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "gru(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_gru(r.tensor(0), r.tensor(1), r.tensor(2), r.tensorlist(3), r.toBool(4), r.toInt64(5), r.toDouble(6), r.toBool(7), r.toBool(8)));
  } else if (r.idx == 1) {
    return wrap(dispatch_gru(r.tensor(0), r.tensor(1), r.tensorlist(2), r.toBool(3), r.toInt64(4), r.toDouble(5), r.toBool(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_gru_cell(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.tensor(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gt(Tensor input, Tensor other, *, Tensor out=None)",
    "gt(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_gt(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_gt(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_gt(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_gt(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hamming_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hamming_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, double alpha, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, double alpha, double beta, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto window_length = r.toInt64(0);
    auto dtype = r.scalartype(1);
    auto device = r.device(3);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(2).layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_hamming_window(window_length, options));
  } else if (r.idx == 1) {
    auto window_length = r.toInt64(0);
    auto periodic = r.toBool(1);
    auto dtype = r.scalartype(2);
    auto device = r.device(4);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(3).layout)
        .requires_grad(r.toBool(5));
    return wrap(dispatch_hamming_window(window_length, periodic, options));
  } else if (r.idx == 2) {
    auto window_length = r.toInt64(0);
    auto periodic = r.toBool(1);
    auto alpha = r.toDouble(2);
    auto dtype = r.scalartype(3);
    auto device = r.device(5);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(4).layout)
        .requires_grad(r.toBool(6));
    return wrap(dispatch_hamming_window(window_length, periodic, alpha, options));
  } else if (r.idx == 3) {
    auto window_length = r.toInt64(0);
    auto periodic = r.toBool(1);
    auto alpha = r.toDouble(2);
    auto beta = r.toDouble(3);
    auto dtype = r.scalartype(4);
    auto device = r.device(6);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(5).layout)
        .requires_grad(r.toBool(7));
    return wrap(dispatch_hamming_window(window_length, periodic, alpha, beta, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hann_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hann_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "hann_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto window_length = r.toInt64(0);
    auto dtype = r.scalartype(1);
    auto device = r.device(3);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(2).layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_hann_window(window_length, options));
  } else if (r.idx == 1) {
    auto window_length = r.toInt64(0);
    auto periodic = r.toBool(1);
    auto dtype = r.scalartype(2);
    auto device = r.device(4);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(3).layout)
        .requires_grad(r.toBool(5));
    return wrap(dispatch_hann_window(window_length, periodic, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardshrink(Tensor input, Scalar lambd=0.5)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_hardshrink(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hinge_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hinge_embedding_loss(Tensor input, Tensor target, double margin=1.0, int64_t reduction=Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_hinge_embedding_loss(r.tensor(0), r.tensor(1), r.toDouble(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "histc(Tensor input, int64_t bins=100, Scalar min=0, Scalar max=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_histc(r.tensor(0), r.toInt64(1), r.scalar(2), r.scalar(3)));
    } else {
      return wrap(dispatch_histc(r.tensor(0), r.toInt64(1), r.scalar(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hspmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hspmm(Tensor mat1, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_hspmm(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_hspmm(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ifft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ifft(Tensor input, int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ifft(r.tensor(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put(Tensor input, TensorList indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_put(r.tensor(0), r.tensorlist(1), r.tensor(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put_(Tensor input, TensorList indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_put_(r.tensor(0), r.tensorlist(1), r.tensor(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_select(Tensor input, int64_t dim, Tensor index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_index_select(r.tensor(0), r.toInt64(1), r.tensor(2)));
    } else {
      return wrap(dispatch_index_select(r.tensor(0), r.toInt64(1), r.tensor(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_instance_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_instance_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "inverse(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_inverse(r.tensor(0)));
    } else {
      return wrap(dispatch_inverse(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_irfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "irfft(Tensor input, int64_t signal_ndim, bool normalized=False, bool onesided=True, IntList signal_sizes=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_irfft(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3), r.intlist(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_complex(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_complex(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_distributed(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_distributed(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_floating_point(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_floating_point(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_nonzero(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_nonzero(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_same_size(Tensor input, Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_same_size(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_signed(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_signed(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_isclose(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_kl_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kl_div(Tensor input, Tensor target, int64_t reduction=Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_kl_div(r.tensor(0), r.tensor(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_kthvalue(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kthvalue(Tensor input, int64_t k, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_kthvalue(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3)));
    } else {
      auto results = r.tensorlist_n<2>(4);
      return wrap(dispatch_kthvalue(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "layer_norm(Tensor input, IntList normalized_shape, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enable=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_layer_norm(r.tensor(0), r.intlist(1), r.tensor(2), r.tensor(3), r.toDouble(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le(Tensor input, Tensor other, *, Tensor out=None)",
    "le(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_le(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_le(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_le(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_le(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lerp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lerp(Tensor input, Tensor end, Scalar weight, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_lerp(r.tensor(0), r.tensor(1), r.scalar(2)));
    } else {
      return wrap(dispatch_lerp(r.tensor(0), r.tensor(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lgamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_lgamma(r.tensor(0)));
    } else {
      return wrap(dispatch_lgamma(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_linspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linspace(Scalar start, Scalar end, int64_t steps=100, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      auto start = r.scalar(0);
      auto end = r.scalar(1);
      auto steps = r.toInt64(2);
      auto dtype = r.scalartype(4);
      auto device = r.device(6);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(7));
      return wrap(dispatch_linspace(start, end, steps, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.isNone(5),
                             r.device(6), r.isNone(6));
      return wrap(dispatch_linspace(r.scalar(0), r.scalar(1), r.toInt64(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log(r.tensor(0)));
    } else {
      return wrap(dispatch_log(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log10(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log10(r.tensor(0)));
    } else {
      return wrap(dispatch_log10(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log10_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_log10_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log1p(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log1p(r.tensor(0)));
    } else {
      return wrap(dispatch_log1p(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log1p_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_log1p_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log2(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log2(r.tensor(0)));
    } else {
      return wrap(dispatch_log2(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log2_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_log2_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_log_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_softmax(Tensor input, int64_t dim)",
    "log_softmax(Tensor input, int64_t dim, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_log_softmax(r.tensor(0), r.toInt64(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_log_softmax(r.tensor(0), r.toInt64(1), r.scalartype(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logdet(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_logdet(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logspace(Scalar start, Scalar end, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "logspace(Scalar start, Scalar end, int64_t steps, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto start = r.scalar(0);
      auto end = r.scalar(1);
      auto dtype = r.scalartype(3);
      auto device = r.device(5);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(4).layout)
          .requires_grad(r.toBool(6));
      return wrap(dispatch_logspace(start, end, options));
    } else {
      check_out_type_matches(r.tensor(2), r.scalartype(3), r.isNone(3),
                             r.layout(4), r.isNone(4),
                             r.device(5), r.isNone(5));
      return wrap(dispatch_logspace(r.scalar(0), r.scalar(1), r.tensor(2)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      auto start = r.scalar(0);
      auto end = r.scalar(1);
      auto steps = r.toInt64(2);
      auto dtype = r.scalartype(4);
      auto device = r.device(6);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(7));
      return wrap(dispatch_logspace(start, end, steps, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.isNone(5),
                             r.device(6), r.isNone(6));
      return wrap(dispatch_logspace(r.scalar(0), r.scalar(1), r.toInt64(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logsumexp(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_logsumexp(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch_logsumexp(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lstm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lstm(Tensor data, Tensor batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "lstm(Tensor input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_lstm(r.tensor(0), r.tensor(1), r.tensorlist(2), r.tensorlist(3), r.toBool(4), r.toInt64(5), r.toDouble(6), r.toBool(7), r.toBool(8)));
  } else if (r.idx == 1) {
    return wrap(dispatch_lstm(r.tensor(0), r.tensorlist(1), r.tensorlist(2), r.toBool(3), r.toInt64(4), r.toDouble(5), r.toBool(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lstm_cell(Tensor input, TensorList hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_lstm_cell(r.tensor(0), r.tensorlist(1), r.tensor(2), r.tensor(3), r.tensor(4), r.tensor(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lt(Tensor input, Tensor other, *, Tensor out=None)",
    "lt(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_lt(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_lt(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_lt(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_lt(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_margin_ranking_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, int64_t reduction=Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_margin_ranking_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toDouble(3), r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_select(Tensor input, Tensor mask, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_masked_select(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_masked_select(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matmul(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_matmul(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_matmul(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matrix_power(Tensor input, int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_matrix_power(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matrix_rank(Tensor input, bool symmetric=False)",
    "matrix_rank(Tensor input, double tol, bool symmetric=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_matrix_rank(r.tensor(0), r.toBool(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_matrix_rank(r.tensor(0), r.toDouble(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max(Tensor input)",
    "max(Tensor input, Tensor other, *, Tensor out=None)",
    "max(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_max(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_max(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_max(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_max(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_max(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max_pool1d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool1d_with_indices(Tensor input, IntList[1] kernel_size, IntList[1] stride=None, IntList[1] padding=0, IntList[1] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_max_pool1d_with_indices(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.intlist(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mean(Tensor input)",
    "mean(Tensor input, *, ScalarType dtype)",
    "mean(Tensor input, IntList[1] dim, *, ScalarType dtype, Tensor out=None)",
    "mean(Tensor input, IntList[1] dim, bool keepdim, *, ScalarType dtype, Tensor out=None)",
    "mean(Tensor input, IntList[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mean(r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_mean(r.tensor(0), r.scalartype(1)));
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_mean(r.tensor(0), r.intlist(1), r.scalartype(2)));
    } else {
      return wrap(dispatch_mean(r.tensor(0), r.intlist(1), r.scalartype(2), r.tensor(3)));
    }
  } else if (r.idx == 3) {
    if (r.isNone(4)) {
      return wrap(dispatch_mean(r.tensor(0), r.intlist(1), r.toBool(2), r.scalartype(3)));
    } else {
      return wrap(dispatch_mean(r.tensor(0), r.intlist(1), r.toBool(2), r.scalartype(3), r.tensor(4)));
    }
  } else if (r.idx == 4) {
    if (r.isNone(3)) {
      return wrap(dispatch_mean(r.tensor(0), r.intlist(1), r.toBool(2)));
    } else {
      return wrap(dispatch_mean(r.tensor(0), r.intlist(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_median(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "median(Tensor input)",
    "median(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_median(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_median(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_median(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_meshgrid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "meshgrid(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_meshgrid(r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "min(Tensor input)",
    "min(Tensor input, Tensor other, *, Tensor out=None)",
    "min(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_min(r.tensor(0)));
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_min(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_min(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_min(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_min(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_miopen_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double exponential_average_factor, double epsilon)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_miopen_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_miopen_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_miopen_convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_miopen_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_miopen_convolution_transpose(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.intlist(6), r.toInt64(7), r.toBool(8), r.toBool(9)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mkldnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_convolution(Tensor input, Tensor weight, Tensor? bias, IntList padding, IntList stride, IntList dilation, int64_t groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mkldnn_convolution(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mkldnn_convolution_backward_weights(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_convolution_backward_weights(IntList weight_size, Tensor grad_output, Tensor input, IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mkldnn_convolution_backward_weights(r.intlist(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6), r.toBool(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mm(Tensor input, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_mm(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_mm(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mode(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mode(Tensor input, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_mode(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_mode(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_mul(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_mul(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multinomial(Tensor input, int64_t num_samples, bool replacement=False, *, Generator generator=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_multinomial(r.tensor(0), r.toInt64(1), r.toBool(2), r.generator(3)));
    } else {
      return wrap(dispatch_multinomial(r.tensor(0), r.toInt64(1), r.toBool(2), r.generator(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mv(Tensor input, Tensor vec, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_mv(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_mv(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mvlgamma(Tensor input, int64_t p)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mvlgamma(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow(Tensor input, int64_t dim, int64_t start, int64_t length)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_narrow(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_native_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_native_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_native_clone(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_clone(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_native_clone(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_native_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_norm(Tensor input, Scalar p=2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_native_norm(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_native_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_pow(Tensor input, Scalar exponent, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_native_pow(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_native_pow(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_native_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_resize_as_(Tensor input, Tensor the_template)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_native_resize_as_(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_native_zero_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_zero_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_native_zero_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne(Tensor input, Tensor other, *, Tensor out=None)",
    "ne(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_ne(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_ne(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_ne(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_ne(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "neg(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_neg(r.tensor(0)));
    } else {
      return wrap(dispatch_neg(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_nonzero(r.tensor(0)));
    } else {
      return wrap(dispatch_nonzero(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm(Tensor input, Scalar p=2)",
    "norm(Tensor input, Scalar p=None, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_norm(r.tensor(0), r.scalar(1)));
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      auto self = r.tensor(0);
      auto p = r.scalarWithDefault(1, 2);
      auto dim = r.toInt64(2);
      auto keepdim = r.toBool(3);
      return wrap(dispatch_norm(self, p, dim, keepdim));
    } else {
      auto self = r.tensor(0);
      auto p = r.scalarWithDefault(1, 2);
      auto dim = r.toInt64(2);
      auto keepdim = r.toBool(3);
      return wrap(dispatch_norm(self, p, dim, keepdim, r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_norm_except_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm_except_dim(Tensor v, int64_t pow=2, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_norm_except_dim(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_normal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "normal(Tensor mean, Tensor std, *, Generator generator=None, Tensor out=None)",
    "normal(Tensor mean, double std=1, *, Generator generator=None, Tensor out=None)",
    "normal(double mean, Tensor std, *, Generator generator=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_normal(r.tensor(0), r.tensor(1), r.generator(2)));
    } else {
      return wrap(dispatch_normal(r.tensor(0), r.tensor(1), r.generator(2), r.tensor(3)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_normal(r.tensor(0), r.toDouble(1), r.generator(2)));
    } else {
      return wrap(dispatch_normal(r.tensor(0), r.toDouble(1), r.generator(2), r.tensor(3)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_normal(r.toDouble(0), r.tensor(1), r.generator(2)));
    } else {
      return wrap(dispatch_normal(r.toDouble(0), r.tensor(1), r.generator(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_nuclear_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nuclear_norm(Tensor input, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_nuclear_norm(r.tensor(0), r.toBool(1)));
    } else {
      return wrap(dispatch_nuclear_norm(r.tensor(0), r.toBool(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
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
  if (r.idx == 0) {
    return wrap(dispatch_numel(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ones(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto size = r.intlist(0);
      auto dtype = r.scalartype(2);
      auto device = r.device(4);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(5));
      return wrap(dispatch_ones(size, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                             r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_ones(r.intlist(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ones_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "ones_like(Tensor input, *, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_ones_like(self, options));
  } else if (r.idx == 1) {
    return wrap(dispatch_ones_like(r.tensor(0)).set_requires_grad(r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "orgqr(Tensor input, Tensor input2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_orgqr(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_orgqr(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ormqr(Tensor input, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_ormqr(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toBool(4)));
    } else {
      return wrap(dispatch_ormqr(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toBool(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pairwise_distance(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pairwise_distance(Tensor x1, Tensor x2, double p=2, double eps=1e-06, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pairwise_distance(r.tensor(0), r.tensor(1), r.toDouble(2), r.toDouble(3), r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pdist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pdist(Tensor input, double p=2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pdist(r.tensor(0), r.toDouble(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pin_memory(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pin_memory(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pin_memory(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pinverse(Tensor input, double rcond=1e-15)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pinverse(r.tensor(0), r.toDouble(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pixel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pixel_shuffle(Tensor input, int64_t upscale_factor)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pixel_shuffle(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_poisson(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "poisson(Tensor input, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_poisson(r.tensor(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma(int64_t n, Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_polygamma(r.toInt64(0), r.tensor(1)));
    } else {
      return wrap(dispatch_polygamma(r.toInt64(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potri(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potri(Tensor input, bool upper=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_potri(r.tensor(0), r.toBool(1)));
    } else {
      return wrap(dispatch_potri(r.tensor(0), r.toBool(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potrs(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potrs(Tensor input, Tensor input2, bool upper=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_potrs(r.tensor(0), r.tensor(1), r.toBool(2)));
    } else {
      return wrap(dispatch_potrs(r.tensor(0), r.tensor(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow(Tensor input, Tensor exponent, *, Tensor out=None)",
    "pow(Scalar self, Tensor exponent, *, Tensor out=None)",
    "pow(Tensor input, Scalar exponent, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_pow(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_pow(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_pow(r.scalar(0), r.tensor(1)));
    } else {
      return wrap(dispatch_pow(r.scalar(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 2) {
    if (r.isNone(2)) {
      return wrap(dispatch_pow(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_pow(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prelu(Tensor input, Tensor weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_prelu(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prod(Tensor input)",
    "prod(Tensor input, *, ScalarType dtype)",
    "prod(Tensor input, int64_t dim, *, ScalarType dtype, Tensor out=None)",
    "prod(Tensor input, int64_t dim, bool keepdim, *, ScalarType dtype, Tensor out=None)",
    "prod(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_prod(r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_prod(r.tensor(0), r.scalartype(1)));
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.scalartype(2)));
    } else {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.scalartype(2), r.tensor(3)));
    }
  } else if (r.idx == 3) {
    if (r.isNone(4)) {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.toBool(2), r.scalartype(3)));
    } else {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.toBool(2), r.scalartype(3), r.tensor(4)));
    }
  } else if (r.idx == 4) {
    if (r.isNone(3)) {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      return wrap(dispatch_prod(r.tensor(0), r.toInt64(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pstrf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pstrf(Tensor input, bool upper=True, Scalar tol=-1, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_pstrf(r.tensor(0), r.toBool(1), r.scalar(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_pstrf(r.tensor(0), r.toBool(1), r.scalar(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "qr(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_qr(r.tensor(0)));
    } else {
      auto results = r.tensorlist_n<2>(1);
      return wrap(dispatch_qr(r.tensor(0), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rand(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand(IntList size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "rand(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto size = r.intlist(0);
      auto generator = r.generator(1);
      auto dtype = r.scalartype(3);
      auto device = r.device(5);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(4).layout)
          .requires_grad(r.toBool(6));
      return wrap(dispatch_rand(size, generator, options));
    } else {
      check_out_type_matches(r.tensor(2), r.scalartype(3), r.isNone(3),
                             r.layout(4), r.isNone(4),
                             r.device(5), r.isNone(5));
      return wrap(dispatch_rand(r.intlist(0), r.generator(1), r.tensor(2)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(1)) {
      auto size = r.intlist(0);
      auto dtype = r.scalartype(2);
      auto device = r.device(4);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(5));
      return wrap(dispatch_rand(size, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                             r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_rand(r.intlist(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rand_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "rand_like(Tensor input, *, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_rand_like(self, options));
  } else if (r.idx == 1) {
    return wrap(dispatch_rand_like(r.tensor(0)).set_requires_grad(r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_randint_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint_like(Tensor input, int64_t high, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "randint_like(Tensor input, int64_t high, *, bool requires_grad=False)",
    "randint_like(Tensor input, int64_t low, int64_t high, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "randint_like(Tensor input, int64_t low, int64_t high, *, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto high = r.toInt64(1);
    auto dtype = r.scalartypeWithDefault(2, self.type().scalarType());
    auto layout = r.layoutWithDefault(3, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(4, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(5));
    return wrap(dispatch_randint_like(self, high, options));
  } else if (r.idx == 1) {
    return wrap(dispatch_randint_like(r.tensor(0), r.toInt64(1)).set_requires_grad(r.toBool(3)));
  } else if (r.idx == 2) {
    auto self = r.tensor(0);
    auto low = r.toInt64(1);
    auto high = r.toInt64(2);
    auto dtype = r.scalartypeWithDefault(3, self.type().scalarType());
    auto layout = r.layoutWithDefault(4, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(5, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(6));
    return wrap(dispatch_randint_like(self, low, high, options));
  } else if (r.idx == 3) {
    return wrap(dispatch_randint_like(r.tensor(0), r.toInt64(1), r.toInt64(2)).set_requires_grad(r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn(IntList size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randn(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto size = r.intlist(0);
      auto generator = r.generator(1);
      auto dtype = r.scalartype(3);
      auto device = r.device(5);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(4).layout)
          .requires_grad(r.toBool(6));
      return wrap(dispatch_randn(size, generator, options));
    } else {
      check_out_type_matches(r.tensor(2), r.scalartype(3), r.isNone(3),
                             r.layout(4), r.isNone(4),
                             r.device(5), r.isNone(5));
      return wrap(dispatch_randn(r.intlist(0), r.generator(1), r.tensor(2)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(1)) {
      auto size = r.intlist(0);
      auto dtype = r.scalartype(2);
      auto device = r.device(4);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(5));
      return wrap(dispatch_randn(size, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                             r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_randn(r.intlist(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_randn_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "randn_like(Tensor input, *, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_randn_like(self, options));
  } else if (r.idx == 1) {
    return wrap(dispatch_randn_like(r.tensor(0)).set_requires_grad(r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_randperm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randperm(int64_t n, *, Generator generator, Tensor out=None, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randperm(int64_t n, *, Tensor out=None, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      auto n = r.toInt64(0);
      auto generator = r.generator(1);
      auto dtype = r.scalartype(3);
      auto device = r.device(5);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(4).layout)
          .requires_grad(r.toBool(6));
      return wrap(dispatch_randperm(n, generator, options));
    } else {
      check_out_type_matches(r.tensor(2), r.scalartype(3), r.isNone(3),
                             r.layout(4), r.isNone(4),
                             r.device(5), r.isNone(5));
      return wrap(dispatch_randperm(r.toInt64(0), r.generator(1), r.tensor(2)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(1)) {
      auto n = r.toInt64(0);
      auto dtype = r.scalartype(2);
      auto device = r.device(4);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(5));
      return wrap(dispatch_randperm(n, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                             r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_randperm(r.toInt64(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reciprocal(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_reciprocal(r.tensor(0)));
    } else {
      return wrap(dispatch_reciprocal(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_relu(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_relu_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder(Tensor input, Tensor other, *, Tensor out=None)",
    "remainder(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_remainder(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_remainder(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(2)) {
      return wrap(dispatch_remainder(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_remainder(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm(Tensor input, Scalar p, int64_t dim, Scalar maxnorm, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_renorm(r.tensor(0), r.scalar(1), r.toInt64(2), r.scalar(3)));
    } else {
      return wrap(dispatch_renorm(r.tensor(0), r.scalar(1), r.toInt64(2), r.scalar(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reshape(Tensor input, IntList shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_reshape(r.tensor(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_as_(Tensor input, Tensor the_template)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_resize_as_(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rfft(Tensor input, int64_t signal_ndim, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rfft(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rnn_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_relu(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "rnn_relu(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rnn_relu(r.tensor(0), r.tensor(1), r.tensor(2), r.tensorlist(3), r.toBool(4), r.toInt64(5), r.toDouble(6), r.toBool(7), r.toBool(8)));
  } else if (r.idx == 1) {
    return wrap(dispatch_rnn_relu(r.tensor(0), r.tensor(1), r.tensorlist(2), r.toBool(3), r.toInt64(4), r.toDouble(5), r.toBool(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rnn_relu_cell(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.tensor(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rnn_tanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_tanh(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "rnn_tanh(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rnn_tanh(r.tensor(0), r.tensor(1), r.tensor(2), r.tensorlist(3), r.toBool(4), r.toInt64(5), r.toDouble(6), r.toBool(7), r.toBool(8)));
  } else if (r.idx == 1) {
    return wrap(dispatch_rnn_tanh(r.tensor(0), r.tensor(1), r.tensorlist(2), r.toBool(3), r.toInt64(4), r.toDouble(5), r.toBool(6), r.toBool(7), r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rnn_tanh_cell(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.tensor(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "roll(Tensor input, IntList[1] shifts, IntList[1] dims=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_roll(r.tensor(0), r.intlist(1), r.intlist(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rot90(Tensor input, int64_t k=1, IntList dims={0,1})",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rot90(r.tensor(0), r.toInt64(1), r.intlist(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "round(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_round(r.tensor(0)));
    } else {
      return wrap(dispatch_round(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "round_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_round_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rrelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu(Tensor input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rrelu(r.tensor(0), r.scalar(1), r.scalar(2), r.toBool(3), r.generator(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rrelu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_(Tensor input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rrelu_(r.tensor(0), r.scalar(1), r.scalar(2), r.toBool(3), r.generator(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsqrt(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_rsqrt(r.tensor(0)));
    } else {
      return wrap(dispatch_rsqrt(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsqrt_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rsqrt_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsub(Tensor input, Tensor other, *, Scalar alpha=1)",
    "rsub(Tensor input, Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rsub(r.tensor(0), r.tensor(1), r.scalar(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_rsub(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_s_copy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "s_copy_(Tensor input, Tensor src, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_s_copy_(r.tensor(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_s_native_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "s_native_addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_s_native_addmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_s_native_addmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_s_native_addmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "s_native_addmm_(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_s_native_addmm_(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_scalar_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scalar_tensor(Scalar s, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto s = r.scalar(0);
    auto dtype = r.scalartype(1);
    auto device = r.device(3);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(2).layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_scalar_tensor(s, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select(Tensor input, int64_t dim, int64_t index)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_select(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_selu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_selu(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_selu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_selu_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sigmoid(r.tensor(0)));
    } else {
      return wrap(dispatch_sigmoid(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sigmoid_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sigmoid_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sign(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sign(r.tensor(0)));
    } else {
      return wrap(dispatch_sign(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sin(r.tensor(0)));
    } else {
      return wrap(dispatch_sin(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sin_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sinh(r.tensor(0)));
    } else {
      return wrap(dispatch_sinh(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sinh_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slogdet(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_slogdet(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smm(Tensor input, Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_smm(r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softmax(Tensor input, int64_t dim)",
    "softmax(Tensor input, int64_t dim, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_softmax(r.tensor(0), r.toInt64(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_softmax(r.tensor(0), r.toInt64(1), r.scalartype(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sort(Tensor input, int64_t dim=-1, bool descending=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_sort(r.tensor(0), r.toInt64(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_sort(r.tensor(0), r.toInt64(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split(Tensor input, int64_t split_size, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_split(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split_with_sizes(Tensor input, IntList split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_split_with_sizes(r.tensor(0), r.intlist(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_sqrt(r.tensor(0)));
    } else {
      return wrap(dispatch_sqrt(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sqrt_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze(Tensor input)",
    "squeeze(Tensor input, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_squeeze(r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_squeeze(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sspaddmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sspaddmm(r.scalar(0), r.tensor(1), r.scalar(2), r.tensor(3), r.tensor(4)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sspaddmm(r.scalar(0), r.tensor(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 2) {
    if (r.isNone(5)) {
      return wrap(dispatch_sspaddmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4)));
    } else {
      return wrap(dispatch_sspaddmm(r.tensor(0), r.tensor(1), r.tensor(2), r.scalar(3), r.scalar(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_stack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stack(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_stack(r.tensorlist(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_stack(r.tensorlist(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_std(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "std(Tensor input, IntList[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "std(Tensor input, bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_std(r.tensor(0), r.intlist(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_std(r.tensor(0), r.intlist(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  } else if (r.idx == 1) {
    return wrap(dispatch_std(r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_stft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stft(Tensor input, int64_t n_fft, int64_t hop_length=None, int64_t win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto n_fft = r.toInt64(1);
    auto hop_length = r.toInt64WithDefault(2, n_fft >> 2);
    auto win_length = r.toInt64WithDefault(3, n_fft);
    auto window = r.tensor(4);
    auto normalized = r.toBool(5);
    auto onesided = r.toBool(6);
    return wrap(dispatch_stft(self, n_fft, hop_length, win_length, window, normalized, onesided));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "sub(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_sub(r.tensor(0), r.scalar(1), r.tensor(2)));
    } else {
      return wrap(dispatch_sub(r.tensor(0), r.scalar(1), r.tensor(2), r.tensor(3)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      return wrap(dispatch_sub(r.tensor(0), r.tensor(1), r.scalar(2)));
    } else {
      return wrap(dispatch_sub(r.tensor(0), r.tensor(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sum(Tensor input)",
    "sum(Tensor input, *, ScalarType dtype)",
    "sum(Tensor input, IntList[1] dim, *, ScalarType dtype, Tensor out=None)",
    "sum(Tensor input, IntList[1] dim, bool keepdim, *, ScalarType dtype, Tensor out=None)",
    "sum(Tensor input, IntList[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sum(r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sum(r.tensor(0), r.scalartype(1)));
  } else if (r.idx == 2) {
    if (r.isNone(3)) {
      return wrap(dispatch_sum(r.tensor(0), r.intlist(1), r.scalartype(2)));
    } else {
      return wrap(dispatch_sum(r.tensor(0), r.intlist(1), r.scalartype(2), r.tensor(3)));
    }
  } else if (r.idx == 3) {
    if (r.isNone(4)) {
      return wrap(dispatch_sum(r.tensor(0), r.intlist(1), r.toBool(2), r.scalartype(3)));
    } else {
      return wrap(dispatch_sum(r.tensor(0), r.intlist(1), r.toBool(2), r.scalartype(3), r.tensor(4)));
    }
  } else if (r.idx == 4) {
    if (r.isNone(3)) {
      return wrap(dispatch_sum(r.tensor(0), r.intlist(1), r.toBool(2)));
    } else {
      return wrap(dispatch_sum(r.tensor(0), r.intlist(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "svd(Tensor input, bool some=True, bool compute_uv=True, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_svd(r.tensor(0), r.toBool(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<3>(3);
      return wrap(dispatch_svd(r.tensor(0), r.toBool(1), r.toBool(2), results[0], results[1], results[2]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_symeig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "symeig(Tensor input, bool eigenvectors=False, bool upper=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_symeig(r.tensor(0), r.toBool(1), r.toBool(2)));
    } else {
      auto results = r.tensorlist_n<2>(3);
      return wrap(dispatch_symeig(r.tensor(0), r.toBool(1), r.toBool(2), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_t(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "t(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_t(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "take(Tensor input, Tensor index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_take(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_take(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tan(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_tan(r.tensor(0)));
    } else {
      return wrap(dispatch_tan(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tan_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_tan_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tanh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_tanh(r.tensor(0)));
    } else {
      return wrap(dispatch_tanh(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tanh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_tanh_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tensordot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tensordot(Tensor input, Tensor other, IntList dims_self, IntList dims_other)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_tensordot(r.tensor(0), r.tensor(1), r.intlist(2), r.intlist(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_threshold(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold(Tensor input, Scalar threshold, Scalar value, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_threshold(r.tensor(0), r.scalar(1), r.scalar(2)));
    } else {
      return wrap(dispatch_threshold(r.tensor(0), r.scalar(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_threshold_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold_(Tensor input, Scalar threshold, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_threshold_(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_topk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "topk(Tensor input, int64_t k, int64_t dim=-1, bool largest=True, bool sorted=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_topk(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3), r.toBool(4)));
    } else {
      auto results = r.tensorlist_n<2>(5);
      return wrap(dispatch_topk(r.tensor(0), r.toInt64(1), r.toInt64(2), r.toBool(3), r.toBool(4), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trace(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_trace(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose(Tensor input, int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_transpose(r.tensor(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_tril(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_tril(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tril_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril_indices(int64_t row, int64_t col, int64_t offset=0, *, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto row = r.toInt64(0);
    auto col = r.toInt64(1);
    auto offset = r.toInt64(2);
    auto dtype = r.scalartype(3);
    auto device = r.device(5);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(4).layout)
        .requires_grad(r.toBool(6));
    return wrap(dispatch_tril_indices(row, col, offset, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triplet_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin=1.0, double p=2, double eps=1e-06, bool swap=False, int64_t reduction=Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_triplet_margin_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toDouble(3), r.toDouble(4), r.toDouble(5), r.toBool(6), r.toInt64(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_triu(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_triu(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triu_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu_indices(int64_t row, int64_t col, int64_t offset=0, *, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto row = r.toInt64(0);
    auto col = r.toInt64(1);
    auto offset = r.toInt64(2);
    auto dtype = r.scalartype(3);
    auto device = r.device(5);
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layout(4).layout)
        .requires_grad(r.toBool(6));
    return wrap(dispatch_triu_indices(row, col, offset, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trtrs(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trtrs(Tensor input, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_trtrs(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.toBool(4)));
    } else {
      auto results = r.tensorlist_n<2>(5);
      return wrap(dispatch_trtrs(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.toBool(4), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_trunc(r.tensor(0)));
    } else {
      return wrap(dispatch_trunc(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_trunc_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unbind(Tensor input, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_unbind(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze(Tensor input, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_unsqueeze(r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var(Tensor input, bool unbiased=True)",
    "var(Tensor input, int64_t dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_var(r.tensor(0), r.toBool(1)));
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      return wrap(dispatch_var(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_var(r.tensor(0), r.toInt64(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "where(Tensor condition, Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_where(r.tensor(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zero_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_zero_(r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_zeros(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros(IntList size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto size = r.intlist(0);
      auto dtype = r.scalartype(2);
      auto device = r.device(4);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(5));
      return wrap(dispatch_zeros(size, options));
    } else {
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2),
                             r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_zeros(r.intlist(0), r.tensor(1)).set_requires_grad(r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_zeros_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros_like(Tensor input, *, ScalarType dtype=None, Layout layout=None, Device device=None, bool requires_grad=False)",
    "zeros_like(Tensor input, *, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto self = r.tensor(0);
    auto dtype = r.scalartypeWithDefault(1, self.type().scalarType());
    auto layout = r.layoutWithDefault(2, *torch::getLayout(self.type().backend()));
    auto device = r.deviceWithDefault(3, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(layout.layout)
        .requires_grad(r.toBool(4));
    return wrap(dispatch_zeros_like(self, options));
  } else if (r.idx == 1) {
    return wrap(dispatch_zeros_like(r.tensor(0)).set_requires_grad(r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef torch_functions[] = {
  {"arange", (PyCFunction)THPVariable_arange, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_tensor", (PyCFunction)THPVariable_as_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dsmm", (PyCFunction)THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"from_numpy", (PyCFunction)THPVariable_from_numpy, METH_STATIC | METH_O, NULL},
  {"hsmm", (PyCFunction)THPVariable_hspmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_promote_types", (PyCFunction)THPVariable__promote_types, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randint", (PyCFunction)THPVariable_randint, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"range", (PyCFunction)THPVariable_range, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"saddmm", (PyCFunction)THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sparse_coo_tensor", (PyCFunction)THPVariable_sparse_coo_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"spmm", (PyCFunction)THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tensor", (PyCFunction)THPVariable_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"get_device", (PyCFunction)THPVariable_get_device, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__and__", (PyCFunction)THPVariable___and__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__lshift__", (PyCFunction)THPVariable___lshift__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__or__", (PyCFunction)THPVariable___or__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__rshift__", (PyCFunction)THPVariable___rshift__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__xor__", (PyCFunction)THPVariable___xor__, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_argmax", (PyCFunction)THPVariable__argmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_argmin", (PyCFunction)THPVariable__argmin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_baddbmm_mkl_", (PyCFunction)THPVariable__baddbmm_mkl_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Byte", (PyCFunction)THPVariable__cast_Byte, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Char", (PyCFunction)THPVariable__cast_Char, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Double", (PyCFunction)THPVariable__cast_Double, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Float", (PyCFunction)THPVariable__cast_Float, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Half", (PyCFunction)THPVariable__cast_Half, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Int", (PyCFunction)THPVariable__cast_Int, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Long", (PyCFunction)THPVariable__cast_Long, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Short", (PyCFunction)THPVariable__cast_Short, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_convolution", (PyCFunction)THPVariable__convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_convolution_nogroup", (PyCFunction)THPVariable__convolution_nogroup, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_copy_same_type_", (PyCFunction)THPVariable__copy_same_type_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_ctc_loss", (PyCFunction)THPVariable__ctc_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_ctc_loss", (PyCFunction)THPVariable__cudnn_ctc_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_init_dropout_state", (PyCFunction)THPVariable__cudnn_init_dropout_state, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_rnn", (PyCFunction)THPVariable__cudnn_rnn, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_rnn_flatten_weight", (PyCFunction)THPVariable__cudnn_rnn_flatten_weight, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_clear_plan_cache", (PyCFunction)THPVariable__cufft_clear_plan_cache, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_get_plan_cache_max_size", (PyCFunction)THPVariable__cufft_get_plan_cache_max_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_get_plan_cache_size", (PyCFunction)THPVariable__cufft_get_plan_cache_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_set_plan_cache_max_size", (PyCFunction)THPVariable__cufft_set_plan_cache_max_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_dim_arange", (PyCFunction)THPVariable__dim_arange, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_dirichlet_grad", (PyCFunction)THPVariable__dirichlet_grad, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_embedding_bag", (PyCFunction)THPVariable__embedding_bag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_fft_with_size", (PyCFunction)THPVariable__fft_with_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_fused_dropout", (PyCFunction)THPVariable__fused_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_log_softmax", (PyCFunction)THPVariable__log_softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_log_softmax_backward_data", (PyCFunction)THPVariable__log_softmax_backward_data, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_masked_scale", (PyCFunction)THPVariable__masked_scale, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_pack_padded_sequence", (PyCFunction)THPVariable__pack_padded_sequence, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_pad_packed_sequence", (PyCFunction)THPVariable__pad_packed_sequence, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_reshape_from_tensor", (PyCFunction)THPVariable__reshape_from_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_s_copy_from", (PyCFunction)THPVariable__s_copy_from, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_s_where", (PyCFunction)THPVariable__s_where, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_shape_as_tensor", (PyCFunction)THPVariable__shape_as_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_softmax", (PyCFunction)THPVariable__softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_softmax_backward_data", (PyCFunction)THPVariable__softmax_backward_data, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_addmm", (PyCFunction)THPVariable__sparse_addmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_mm", (PyCFunction)THPVariable__sparse_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_sum", (PyCFunction)THPVariable__sparse_sum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_standard_gamma", (PyCFunction)THPVariable__standard_gamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_standard_gamma_grad", (PyCFunction)THPVariable__standard_gamma_grad, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_trilinear", (PyCFunction)THPVariable__trilinear, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_unique", (PyCFunction)THPVariable__unique, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_unique_dim", (PyCFunction)THPVariable__unique_dim, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_weight_norm", (PyCFunction)THPVariable__weight_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_weight_norm_cuda_interface", (PyCFunction)THPVariable__weight_norm_cuda_interface, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"abs", (PyCFunction)THPVariable_abs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"abs_", (PyCFunction)THPVariable_abs_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"acos", (PyCFunction)THPVariable_acos, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"acos_", (PyCFunction)THPVariable_acos_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"adaptive_avg_pool1d", (PyCFunction)THPVariable_adaptive_avg_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"adaptive_max_pool1d", (PyCFunction)THPVariable_adaptive_max_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"add", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addbmm", (PyCFunction)THPVariable_addbmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addcdiv", (PyCFunction)THPVariable_addcdiv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addcmul", (PyCFunction)THPVariable_addcmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmm", (PyCFunction)THPVariable_addmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmv", (PyCFunction)THPVariable_addmv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmv_", (PyCFunction)THPVariable_addmv_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addr", (PyCFunction)THPVariable_addr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"affine_grid_generator", (PyCFunction)THPVariable_affine_grid_generator, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"all", (PyCFunction)THPVariable_all, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"allclose", (PyCFunction)THPVariable_allclose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"alpha_dropout", (PyCFunction)THPVariable_alpha_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"alpha_dropout_", (PyCFunction)THPVariable_alpha_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"any", (PyCFunction)THPVariable_any, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_strided", (PyCFunction)THPVariable_as_strided, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_strided_", (PyCFunction)THPVariable_as_strided_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"asin", (PyCFunction)THPVariable_asin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"asin_", (PyCFunction)THPVariable_asin_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan", (PyCFunction)THPVariable_atan, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan2", (PyCFunction)THPVariable_atan2, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan_", (PyCFunction)THPVariable_atan_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"avg_pool1d", (PyCFunction)THPVariable_avg_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"baddbmm", (PyCFunction)THPVariable_baddbmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bartlett_window", (PyCFunction)THPVariable_bartlett_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm", (PyCFunction)THPVariable_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bernoulli", (PyCFunction)THPVariable_bernoulli, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bilinear", (PyCFunction)THPVariable_bilinear, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"binary_cross_entropy_with_logits", (PyCFunction)THPVariable_binary_cross_entropy_with_logits, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bincount", (PyCFunction)THPVariable_bincount, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"blackman_window", (PyCFunction)THPVariable_blackman_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bmm", (PyCFunction)THPVariable_bmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"broadcast_tensors", (PyCFunction)THPVariable_broadcast_tensors, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"btrifact", (PyCFunction)THPVariable_btrifact, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"btrifact_with_info", (PyCFunction)THPVariable_btrifact_with_info, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"btrisolve", (PyCFunction)THPVariable_btrisolve, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cat", (PyCFunction)THPVariable_cat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ceil", (PyCFunction)THPVariable_ceil, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ceil_", (PyCFunction)THPVariable_ceil_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"celu", (PyCFunction)THPVariable_celu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"celu_", (PyCFunction)THPVariable_celu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"chain_matmul", (PyCFunction)THPVariable_chain_matmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cholesky", (PyCFunction)THPVariable_cholesky, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"chunk", (PyCFunction)THPVariable_chunk, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp", (PyCFunction)THPVariable_clamp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_", (PyCFunction)THPVariable_clamp_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_max", (PyCFunction)THPVariable_clamp_max, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_max_", (PyCFunction)THPVariable_clamp_max_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_min", (PyCFunction)THPVariable_clamp_min, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_min_", (PyCFunction)THPVariable_clamp_min_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clone", (PyCFunction)THPVariable_clone, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"constant_pad_nd", (PyCFunction)THPVariable_constant_pad_nd, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv1d", (PyCFunction)THPVariable_conv1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv2d", (PyCFunction)THPVariable_conv2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv3d", (PyCFunction)THPVariable_conv3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_tbc", (PyCFunction)THPVariable_conv_tbc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose1d", (PyCFunction)THPVariable_conv_transpose1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose2d", (PyCFunction)THPVariable_conv_transpose2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose3d", (PyCFunction)THPVariable_conv_transpose3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"convolution", (PyCFunction)THPVariable_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cos", (PyCFunction)THPVariable_cos, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cos_", (PyCFunction)THPVariable_cos_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosh", (PyCFunction)THPVariable_cosh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosh_", (PyCFunction)THPVariable_cosh_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosine_embedding_loss", (PyCFunction)THPVariable_cosine_embedding_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosine_similarity", (PyCFunction)THPVariable_cosine_similarity, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cross", (PyCFunction)THPVariable_cross, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ctc_loss", (PyCFunction)THPVariable_ctc_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_affine_grid_generator", (PyCFunction)THPVariable_cudnn_affine_grid_generator, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_batch_norm", (PyCFunction)THPVariable_cudnn_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_convolution", (PyCFunction)THPVariable_cudnn_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_convolution_transpose", (PyCFunction)THPVariable_cudnn_convolution_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_grid_sampler", (PyCFunction)THPVariable_cudnn_grid_sampler, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_is_acceptable", (PyCFunction)THPVariable_cudnn_is_acceptable, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cumprod", (PyCFunction)THPVariable_cumprod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cumsum", (PyCFunction)THPVariable_cumsum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"det", (PyCFunction)THPVariable_det, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"detach", (PyCFunction)THPVariable_detach, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diag", (PyCFunction)THPVariable_diag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diag_embed", (PyCFunction)THPVariable_diag_embed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diagflat", (PyCFunction)THPVariable_diagflat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diagonal", (PyCFunction)THPVariable_diagonal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"digamma", (PyCFunction)THPVariable_digamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dist", (PyCFunction)THPVariable_dist, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"div", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dot", (PyCFunction)THPVariable_dot, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dropout", (PyCFunction)THPVariable_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dropout_", (PyCFunction)THPVariable_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eig", (PyCFunction)THPVariable_eig, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"einsum", (PyCFunction)THPVariable_einsum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding", (PyCFunction)THPVariable_embedding, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding_bag", (PyCFunction)THPVariable_embedding_bag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding_renorm_", (PyCFunction)THPVariable_embedding_renorm_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty", (PyCFunction)THPVariable_empty, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty_like", (PyCFunction)THPVariable_empty_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty_strided", (PyCFunction)THPVariable_empty_strided, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eq", (PyCFunction)THPVariable_eq, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"equal", (PyCFunction)THPVariable_equal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erf", (PyCFunction)THPVariable_erf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erf_", (PyCFunction)THPVariable_erf_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erfc", (PyCFunction)THPVariable_erfc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erfc_", (PyCFunction)THPVariable_erfc_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erfinv", (PyCFunction)THPVariable_erfinv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"exp", (PyCFunction)THPVariable_exp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"exp_", (PyCFunction)THPVariable_exp_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"expm1", (PyCFunction)THPVariable_expm1, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"expm1_", (PyCFunction)THPVariable_expm1_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eye", (PyCFunction)THPVariable_eye, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_alpha_dropout", (PyCFunction)THPVariable_feature_alpha_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_alpha_dropout_", (PyCFunction)THPVariable_feature_alpha_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_dropout", (PyCFunction)THPVariable_feature_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_dropout_", (PyCFunction)THPVariable_feature_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fft", (PyCFunction)THPVariable_fft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fill_", (PyCFunction)THPVariable_fill_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"flatten", (PyCFunction)THPVariable_flatten, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"flip", (PyCFunction)THPVariable_flip, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"floor", (PyCFunction)THPVariable_floor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"floor_", (PyCFunction)THPVariable_floor_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fmod", (PyCFunction)THPVariable_fmod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"frac", (PyCFunction)THPVariable_frac, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"frobenius_norm", (PyCFunction)THPVariable_frobenius_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"full", (PyCFunction)THPVariable_full, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"full_like", (PyCFunction)THPVariable_full_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gather", (PyCFunction)THPVariable_gather, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ge", (PyCFunction)THPVariable_ge, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gels", (PyCFunction)THPVariable_gels, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"geqrf", (PyCFunction)THPVariable_geqrf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ger", (PyCFunction)THPVariable_ger, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gesv", (PyCFunction)THPVariable_gesv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"grid_sampler", (PyCFunction)THPVariable_grid_sampler, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"grid_sampler_2d", (PyCFunction)THPVariable_grid_sampler_2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"grid_sampler_3d", (PyCFunction)THPVariable_grid_sampler_3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"group_norm", (PyCFunction)THPVariable_group_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gru", (PyCFunction)THPVariable_gru, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gru_cell", (PyCFunction)THPVariable_gru_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gt", (PyCFunction)THPVariable_gt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hamming_window", (PyCFunction)THPVariable_hamming_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hann_window", (PyCFunction)THPVariable_hann_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hardshrink", (PyCFunction)THPVariable_hardshrink, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hinge_embedding_loss", (PyCFunction)THPVariable_hinge_embedding_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"histc", (PyCFunction)THPVariable_histc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hspmm", (PyCFunction)THPVariable_hspmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ifft", (PyCFunction)THPVariable_ifft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_put", (PyCFunction)THPVariable_index_put, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_put_", (PyCFunction)THPVariable_index_put_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_select", (PyCFunction)THPVariable_index_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"instance_norm", (PyCFunction)THPVariable_instance_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"inverse", (PyCFunction)THPVariable_inverse, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"irfft", (PyCFunction)THPVariable_irfft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_complex", (PyCFunction)THPVariable_is_complex, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_distributed", (PyCFunction)THPVariable_is_distributed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_floating_point", (PyCFunction)THPVariable_is_floating_point, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_nonzero", (PyCFunction)THPVariable_is_nonzero, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_same_size", (PyCFunction)THPVariable_is_same_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_signed", (PyCFunction)THPVariable_is_signed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"isclose", (PyCFunction)THPVariable_isclose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"kl_div", (PyCFunction)THPVariable_kl_div, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"kthvalue", (PyCFunction)THPVariable_kthvalue, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"layer_norm", (PyCFunction)THPVariable_layer_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"le", (PyCFunction)THPVariable_le, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lerp", (PyCFunction)THPVariable_lerp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lgamma", (PyCFunction)THPVariable_lgamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"linspace", (PyCFunction)THPVariable_linspace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log", (PyCFunction)THPVariable_log, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log10", (PyCFunction)THPVariable_log10, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log10_", (PyCFunction)THPVariable_log10_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log1p", (PyCFunction)THPVariable_log1p, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log1p_", (PyCFunction)THPVariable_log1p_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log2", (PyCFunction)THPVariable_log2, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log2_", (PyCFunction)THPVariable_log2_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log_", (PyCFunction)THPVariable_log_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log_softmax", (PyCFunction)THPVariable_log_softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logdet", (PyCFunction)THPVariable_logdet, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logspace", (PyCFunction)THPVariable_logspace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logsumexp", (PyCFunction)THPVariable_logsumexp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lstm", (PyCFunction)THPVariable_lstm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lstm_cell", (PyCFunction)THPVariable_lstm_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lt", (PyCFunction)THPVariable_lt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"margin_ranking_loss", (PyCFunction)THPVariable_margin_ranking_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"masked_select", (PyCFunction)THPVariable_masked_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"matmul", (PyCFunction)THPVariable_matmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"matrix_power", (PyCFunction)THPVariable_matrix_power, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"matrix_rank", (PyCFunction)THPVariable_matrix_rank, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max", (PyCFunction)THPVariable_max, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max_pool1d_with_indices", (PyCFunction)THPVariable_max_pool1d_with_indices, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mean", (PyCFunction)THPVariable_mean, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"median", (PyCFunction)THPVariable_median, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"meshgrid", (PyCFunction)THPVariable_meshgrid, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"min", (PyCFunction)THPVariable_min, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_batch_norm", (PyCFunction)THPVariable_miopen_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_convolution", (PyCFunction)THPVariable_miopen_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_convolution_transpose", (PyCFunction)THPVariable_miopen_convolution_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_convolution", (PyCFunction)THPVariable_mkldnn_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_convolution_backward_weights", (PyCFunction)THPVariable_mkldnn_convolution_backward_weights, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mm", (PyCFunction)THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mode", (PyCFunction)THPVariable_mode, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mul", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"multinomial", (PyCFunction)THPVariable_multinomial, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mv", (PyCFunction)THPVariable_mv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mvlgamma", (PyCFunction)THPVariable_mvlgamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"narrow", (PyCFunction)THPVariable_narrow, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_batch_norm", (PyCFunction)THPVariable_native_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_clone", (PyCFunction)THPVariable_native_clone, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_norm", (PyCFunction)THPVariable_native_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_pow", (PyCFunction)THPVariable_native_pow, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_resize_as_", (PyCFunction)THPVariable_native_resize_as_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_zero_", (PyCFunction)THPVariable_native_zero_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ne", (PyCFunction)THPVariable_ne, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"neg", (PyCFunction)THPVariable_neg, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"nonzero", (PyCFunction)THPVariable_nonzero, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"norm", (PyCFunction)THPVariable_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"norm_except_dim", (PyCFunction)THPVariable_norm_except_dim, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"normal", (PyCFunction)THPVariable_normal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"nuclear_norm", (PyCFunction)THPVariable_nuclear_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"numel", (PyCFunction)THPVariable_numel, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ones", (PyCFunction)THPVariable_ones, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ones_like", (PyCFunction)THPVariable_ones_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"orgqr", (PyCFunction)THPVariable_orgqr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ormqr", (PyCFunction)THPVariable_ormqr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pairwise_distance", (PyCFunction)THPVariable_pairwise_distance, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pdist", (PyCFunction)THPVariable_pdist, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pin_memory", (PyCFunction)THPVariable_pin_memory, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pinverse", (PyCFunction)THPVariable_pinverse, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pixel_shuffle", (PyCFunction)THPVariable_pixel_shuffle, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"poisson", (PyCFunction)THPVariable_poisson, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"polygamma", (PyCFunction)THPVariable_polygamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"potri", (PyCFunction)THPVariable_potri, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"potrs", (PyCFunction)THPVariable_potrs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pow", (PyCFunction)THPVariable_pow, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"prelu", (PyCFunction)THPVariable_prelu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"prod", (PyCFunction)THPVariable_prod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pstrf", (PyCFunction)THPVariable_pstrf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"qr", (PyCFunction)THPVariable_qr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rand", (PyCFunction)THPVariable_rand, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rand_like", (PyCFunction)THPVariable_rand_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randint_like", (PyCFunction)THPVariable_randint_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randn", (PyCFunction)THPVariable_randn, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randn_like", (PyCFunction)THPVariable_randn_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randperm", (PyCFunction)THPVariable_randperm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"reciprocal", (PyCFunction)THPVariable_reciprocal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"relu", (PyCFunction)THPVariable_relu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"relu_", (PyCFunction)THPVariable_relu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"remainder", (PyCFunction)THPVariable_remainder, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"renorm", (PyCFunction)THPVariable_renorm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"reshape", (PyCFunction)THPVariable_reshape, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"resize_as_", (PyCFunction)THPVariable_resize_as_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rfft", (PyCFunction)THPVariable_rfft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_relu", (PyCFunction)THPVariable_rnn_relu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_relu_cell", (PyCFunction)THPVariable_rnn_relu_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_tanh", (PyCFunction)THPVariable_rnn_tanh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_tanh_cell", (PyCFunction)THPVariable_rnn_tanh_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"roll", (PyCFunction)THPVariable_roll, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rot90", (PyCFunction)THPVariable_rot90, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"round", (PyCFunction)THPVariable_round, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"round_", (PyCFunction)THPVariable_round_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rrelu", (PyCFunction)THPVariable_rrelu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rrelu_", (PyCFunction)THPVariable_rrelu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rsqrt", (PyCFunction)THPVariable_rsqrt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rsqrt_", (PyCFunction)THPVariable_rsqrt_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rsub", (PyCFunction)THPVariable_rsub, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"s_copy_", (PyCFunction)THPVariable_s_copy_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"s_native_addmm", (PyCFunction)THPVariable_s_native_addmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"s_native_addmm_", (PyCFunction)THPVariable_s_native_addmm_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"scalar_tensor", (PyCFunction)THPVariable_scalar_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"select", (PyCFunction)THPVariable_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"selu", (PyCFunction)THPVariable_selu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"selu_", (PyCFunction)THPVariable_selu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sigmoid", (PyCFunction)THPVariable_sigmoid, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sigmoid_", (PyCFunction)THPVariable_sigmoid_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sign", (PyCFunction)THPVariable_sign, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sin", (PyCFunction)THPVariable_sin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sin_", (PyCFunction)THPVariable_sin_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sinh", (PyCFunction)THPVariable_sinh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sinh_", (PyCFunction)THPVariable_sinh_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"slogdet", (PyCFunction)THPVariable_slogdet, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"smm", (PyCFunction)THPVariable_smm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"softmax", (PyCFunction)THPVariable_softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sort", (PyCFunction)THPVariable_sort, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"split", (PyCFunction)THPVariable_split, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"split_with_sizes", (PyCFunction)THPVariable_split_with_sizes, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sqrt", (PyCFunction)THPVariable_sqrt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sqrt_", (PyCFunction)THPVariable_sqrt_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"squeeze", (PyCFunction)THPVariable_squeeze, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sspaddmm", (PyCFunction)THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"stack", (PyCFunction)THPVariable_stack, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"std", (PyCFunction)THPVariable_std, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"stft", (PyCFunction)THPVariable_stft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sub", (PyCFunction)THPVariable_sub, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sum", (PyCFunction)THPVariable_sum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"svd", (PyCFunction)THPVariable_svd, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"symeig", (PyCFunction)THPVariable_symeig, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"t", (PyCFunction)THPVariable_t, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"take", (PyCFunction)THPVariable_take, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tan", (PyCFunction)THPVariable_tan, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tan_", (PyCFunction)THPVariable_tan_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tanh", (PyCFunction)THPVariable_tanh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tanh_", (PyCFunction)THPVariable_tanh_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tensordot", (PyCFunction)THPVariable_tensordot, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"threshold", (PyCFunction)THPVariable_threshold, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"threshold_", (PyCFunction)THPVariable_threshold_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"topk", (PyCFunction)THPVariable_topk, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trace", (PyCFunction)THPVariable_trace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"transpose", (PyCFunction)THPVariable_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tril", (PyCFunction)THPVariable_tril, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tril_indices", (PyCFunction)THPVariable_tril_indices, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triplet_margin_loss", (PyCFunction)THPVariable_triplet_margin_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triu", (PyCFunction)THPVariable_triu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triu_indices", (PyCFunction)THPVariable_triu_indices, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trtrs", (PyCFunction)THPVariable_trtrs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trunc", (PyCFunction)THPVariable_trunc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trunc_", (PyCFunction)THPVariable_trunc_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"unbind", (PyCFunction)THPVariable_unbind, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"unsqueeze", (PyCFunction)THPVariable_unsqueeze, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"var", (PyCFunction)THPVariable_var, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"where", (PyCFunction)THPVariable_where, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zero_", (PyCFunction)THPVariable_zero_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zeros", (PyCFunction)THPVariable_zeros, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zeros_like", (PyCFunction)THPVariable_zeros_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
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
