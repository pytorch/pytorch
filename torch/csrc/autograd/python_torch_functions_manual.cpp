#include <torch/csrc/autograd/python_torch_functions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/out_types.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/structseq.h>
#include <torch/csrc/utils/cuda_lazy_init.h>

#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>

#include <fmt/format.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <vector>

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* THPVariableFunctionsModule = nullptr;


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
  static PythonArgParser parser({
      "as_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  jit::tracer::warn("torch.as_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::as_tensor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
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

#define THPVARIABLE_SPARSE_COMPRESSED_CTOR(NAME, NARGS, SIGNATURES)      \
static PyObject * THPVariable_ ## NAME(PyObject* self, PyObject* args, PyObject* kwargs) \
{                                                                       \
  HANDLE_TH_ERRORS                                                      \
  static PythonArgParser parser SIGNATURES ;                          \
  ParsedArgs<NARGS> parsed_args;                                        \
  auto r = parser.parse(args, kwargs, parsed_args);                     \
  if (r.has_torch_function()) {                                         \
    return handle_torch_function(r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch"); \
  }                                                                     \
  jit::tracer::warn("torch."  # NAME, jit::tracer::WARN_CONSTRUCTOR);   \
  return THPVariable_Wrap(torch::utils::NAME ## _ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), r)); \
  END_HANDLE_TH_ERRORS                                                  \
}

THPVARIABLE_SPARSE_COMPRESSED_CTOR(sparse_compressed_tensor, 9,
    ({"sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
      "sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(sparse_csr_tensor, 9,
    ({"sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
      "sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(sparse_csc_tensor, 9,
                                   ({"sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
                                     "sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(sparse_bsr_tensor, 9,
                                   ({"sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
                                     "sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(sparse_bsc_tensor, 9,
                                   ({"sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
                                     "sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)"}))

THPVARIABLE_SPARSE_COMPRESSED_CTOR(_sparse_csc_tensor_unsafe, 7,
                                   ({"_sparse_csc_tensor_unsafe(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(_sparse_bsr_tensor_unsafe, 7,
                                   ({"_sparse_bsr_tensor_unsafe(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(_sparse_bsc_tensor_unsafe, 7,
                                   ({"_sparse_bsc_tensor_unsafe(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)"}))

THPVARIABLE_SPARSE_COMPRESSED_CTOR(_sparse_compressed_tensor_unsafe, 8,
    ({"_sparse_compressed_tensor_unsafe(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(_sparse_csr_tensor_unsafe, 7,
    ({"_sparse_csr_tensor_unsafe(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)"}))

static PyObject * THPVariable_sparse_coo_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
      "sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
      "sparse_coo_tensor(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  jit::tracer::warn("torch.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__sparse_coo_tensor_unsafe(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "_sparse_coo_tensor_unsafe(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  jit::tracer::warn("torch._sparse_coo_tensor_unsafe", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::_sparse_coo_tensor_unsafe_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow torch.tensor to be constructed with arbitrarily nested
// python objects - list, tuple, np array, scalar, etc.
static PyObject * THPVariable_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  constexpr int ctor_num_args = 6;
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  jit::tracer::warn("torch.tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
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

static PyObject * THPVariable_frombuffer(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frombuffer(PyObject* buffer, *, ScalarType dtype, int64_t count=-1, int64_t offset=0, bool requires_grad=False)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto buffer = r.pyobject(0);
    auto dtype = r.scalartype(1);
    auto count = r.toInt64(2);
    auto offset = r.toInt64(3);
    auto requires_grad = r.toBool(4);

    TORCH_CHECK_VALUE(
        PyObject_CheckBuffer(buffer) != 0,
        "object does not implement Python buffer protocol.");
    return wrap(torch::utils::tensor_frombuffer(
        buffer, dtype, count, offset, requires_grad));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_asarray(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asarray(PyObject* obj, *, ScalarType? dtype=None, Device? device=None, bool? copy=None, bool requires_grad=False)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto obj = r.pyobject(0);
    auto dtype = r.scalartypeOptional(1);
    auto device = r.deviceOptional(2);
    auto copy = r.toBoolOptional(3);
    auto requires_grad = r.toBool(4);
    return wrap(torch::utils::asarray(obj, dtype, device, copy, requires_grad));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_numel(PyObject* self_, PyObject* args, PyObject* kwargs);

// linspace
static PyObject * THPVariable_linspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linspace(Scalar start, Scalar end, int64_t steps, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
    // This leads to problem in the operator argument checks,
    // when either `start` or `end` is complex and dtype is None
    const auto options = TensorOptions()
        .dtype(_r.scalartypeOptional(4))
        .device(_r.device(6))
        .layout(_r.layoutOptional(5))
        .requires_grad(_r.toBool(8))
        .pinned_memory(_r.toBool(7));
    torch::utils::maybe_initialize_cuda(options);

    auto dispatch_linspace = [](Scalar start, Scalar end, int64_t steps, TensorOptions options) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return torch::linspace(start, end, steps, options);
    };
    return wrap(dispatch_linspace(_r.scalar(0), _r.scalar(1), _r.toInt64(2), options));
  } else {
    // aten::linspace.out(Scalar start, Scalar end, int? steps=None, *, Tensor(a!) out) -> Tensor(a!)
    check_out_type_matches(_r.tensor(3), _r.scalartype(4),
                           _r.isNone(4), _r.layoutOptional(5),
                           _r.device(6), _r.isNone(6));

    auto dispatch_linspace_out = [](Tensor out, Scalar start, Scalar end, int64_t steps) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linspace_out(out, start, end, steps);
    };
    return wrap(dispatch_linspace_out(_r.tensor(3), _r.scalar(0), _r.scalar(1), _r.toInt64(2)).set_requires_grad(_r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logspace
static PyObject * THPVariable_logspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logspace(Scalar start, Scalar end, int64_t steps, double base=10.0, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::logspace(Scalar start, Scalar end, int steps, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
    // This leads to problem in the operator argument checks,
    // when either `start` or `end` is complex and dtype is None
    const auto options = TensorOptions()
        .dtype(_r.scalartypeOptional(5))
        .device(_r.device(7))
        .layout(_r.layoutOptional(6))
        .requires_grad(_r.toBool(9))
        .pinned_memory(_r.toBool(8));
    torch::utils::maybe_initialize_cuda(options);

    auto dispatch_logspace = [](Scalar start, Scalar end, int64_t steps, double base, TensorOptions options) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return torch::logspace(start, end, steps, base, options);
    };
    return wrap(dispatch_logspace(_r.scalar(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3), options));
  } else {
    // aten::logspace.out(Scalar start, Scalar end, int steps, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
    check_out_type_matches(_r.tensor(4), _r.scalartype(5),
                           _r.isNone(5), _r.layoutOptional(6),
                           _r.device(7), _r.isNone(7));

    auto dispatch_logspace_out = [](Tensor out, Scalar start, Scalar end, int64_t steps, double base) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logspace_out(out, start, end, steps, base);
    };
    return wrap(dispatch_logspace_out(_r.tensor(4), _r.scalar(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3)).set_requires_grad(_r.toBool(9)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__to_functional_tensor(PyObject *self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"_to_functional_tensor(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto wrapped = at::functionalization::impl::to_functional_tensor(self_);
  return wrap(wrapped);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__from_functional_tensor(PyObject *self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"_from_functional_tensor(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto unwrapped = at::functionalization::impl::from_functional_tensor(self_);
  return wrap(unwrapped);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__is_functional_tensor(PyObject *self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"_is_functional_tensor(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (at::functionalization::impl::isFunctionalTensor(self_)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__sync(PyObject *self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"_sync(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  at::functionalization::impl::sync(self_);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__enable_functionalization(PyObject *self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"_enable_functionalization(*, bool reapply_views=False)"}, /*traceable=*/true);
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  const auto reapply_views = r.toBool(0);

  if (c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Functionalize)) {
    TORCH_INTERNAL_ASSERT(false, "multiple layers of mode-style functionalization nesting is not"
     " currently supported, outside of the functionalize() transform");
  }
  c10::impl::tls_set_dispatch_key_included(at::DispatchKey::Functionalize, true);
  if (reapply_views) {
      at::functionalization::impl::setFunctionalizationReapplyViewsTLS(true);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable__disable_functionalization(PyObject *self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  c10::impl::tls_set_dispatch_key_included(at::DispatchKey::Functionalize, false);
  at::functionalization::impl::setFunctionalizationReapplyViewsTLS(false);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef torch_functions_manual[] = {
  {"arange", castPyCFunctionWithKeywords(THPVariable_arange),
    METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"asarray", castPyCFunctionWithKeywords(THPVariable_asarray),
    METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"as_tensor", castPyCFunctionWithKeywords(THPVariable_as_tensor),
    METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"from_numpy", THPVariable_from_numpy, METH_STATIC | METH_O, nullptr},
  {"frombuffer", castPyCFunctionWithKeywords(THPVariable_frombuffer), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"full", castPyCFunctionWithKeywords(THPVariable_full), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"linspace", castPyCFunctionWithKeywords(THPVariable_linspace), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logspace", castPyCFunctionWithKeywords(THPVariable_logspace), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_is_functional_tensor", castPyCFunctionWithKeywords(THPVariable__is_functional_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_to_functional_tensor", castPyCFunctionWithKeywords(THPVariable__to_functional_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_from_functional_tensor", castPyCFunctionWithKeywords(THPVariable__from_functional_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sync", castPyCFunctionWithKeywords(THPVariable__sync), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_enable_functionalization", castPyCFunctionWithKeywords(THPVariable__enable_functionalization), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_disable_functionalization", castPyCFunctionWithKeywords(THPVariable__disable_functionalization), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nonzero", castPyCFunctionWithKeywords(THPVariable_nonzero), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"randint", castPyCFunctionWithKeywords(THPVariable_randint), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"range", castPyCFunctionWithKeywords(THPVariable_range), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sparse_coo_tensor", castPyCFunctionWithKeywords(THPVariable_sparse_coo_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_coo_tensor_unsafe", castPyCFunctionWithKeywords(THPVariable__sparse_coo_tensor_unsafe), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sparse_compressed_tensor", castPyCFunctionWithKeywords(THPVariable_sparse_compressed_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sparse_csr_tensor", castPyCFunctionWithKeywords(THPVariable_sparse_csr_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sparse_csc_tensor", castPyCFunctionWithKeywords(THPVariable_sparse_csc_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sparse_bsr_tensor", castPyCFunctionWithKeywords(THPVariable_sparse_bsr_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sparse_bsc_tensor", castPyCFunctionWithKeywords(THPVariable_sparse_bsc_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_compressed_tensor_unsafe", castPyCFunctionWithKeywords(THPVariable__sparse_compressed_tensor_unsafe), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_csr_tensor_unsafe", castPyCFunctionWithKeywords(THPVariable__sparse_csr_tensor_unsafe), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_csc_tensor_unsafe", castPyCFunctionWithKeywords(THPVariable__sparse_csc_tensor_unsafe), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_bsr_tensor_unsafe", castPyCFunctionWithKeywords(THPVariable__sparse_bsr_tensor_unsafe), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_bsc_tensor_unsafe", castPyCFunctionWithKeywords(THPVariable__sparse_bsc_tensor_unsafe), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tensor", castPyCFunctionWithKeywords(THPVariable_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"get_device", castPyCFunctionWithKeywords(THPVariable_get_device), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"numel", castPyCFunctionWithKeywords(THPVariable_numel), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
};

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

// Sharded function definitions
void gatherTorchFunctions_0(std::vector<PyMethodDef> &torch_functions);
void gatherTorchFunctions_1(std::vector<PyMethodDef> &torch_functions);
void gatherTorchFunctions_2(std::vector<PyMethodDef> &torch_functions);

void gatherTorchFunctions(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_manual) / sizeof(torch_functions_manual[0]);
  torch_functions.assign(torch_functions_manual,
                         torch_functions_manual + num_functions);
  // NOTE: Must be synced with num_shards in tools/autograd/gen_python_functions.py
  gatherTorchFunctions_0(torch_functions);
  gatherTorchFunctions_1(torch_functions);
  gatherTorchFunctions_2(torch_functions);

  static std::array<std::pair<const char *, const char *>, 4> aliases{{
    // Canonical function, alias name
    {"sspaddmm", "saddmm"},
    {"mm", "spmm"},
    {"mm", "dsmm"},
    {"hspmm", "hsmm"}
  }};

  for (const auto& alias : aliases) {
    auto it = std::find_if(torch_functions.begin(), torch_functions.end(),
                          [&](const PyMethodDef& def) {
                            return strcmp(def.ml_name, alias.first) == 0;
                          });
    TORCH_INTERNAL_ASSERT(
        it != torch_functions.end(),
        "Failed to create function alias from ", alias.first, " to ", alias.second);
    PyMethodDef alias_def = *it;
    alias_def.ml_name = alias.second;

    torch_functions.push_back(alias_def);
  }

  torch_functions.push_back({nullptr});
  torch_functions.shrink_to_fit();
}

static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._VariableFunctionsClass",    /* tp_name */
  0,                                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  nullptr,                               /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  nullptr,                               /* tp_getattr */
  nullptr,                               /* tp_setattr */
  nullptr,                               /* tp_reserved */
  nullptr,                               /* tp_repr */
  nullptr,                               /* tp_as_number */
  nullptr,                               /* tp_as_sequence */
  nullptr,                               /* tp_as_mapping */
  nullptr,                               /* tp_hash  */
  nullptr,                               /* tp_call */
  nullptr,                               /* tp_str */
  nullptr,                               /* tp_getattro */
  nullptr,                               /* tp_setattro */
  nullptr,                               /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  nullptr,                               /* tp_doc */
  nullptr,                               /* tp_traverse */
  nullptr,                               /* tp_clear */
  nullptr,                               /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  nullptr,                               /* tp_iter */
  nullptr,                               /* tp_iternext */
  nullptr,                               /* tp_methods */
  nullptr,                               /* tp_members */
  nullptr,                               /* tp_getset */
  nullptr,                               /* tp_base */
  nullptr,                               /* tp_dict */
  nullptr,                               /* tp_descr_get */
  nullptr,                               /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                               /* tp_init */
  nullptr,                               /* tp_alloc */
  nullptr                                /* tp_new */
};

void initTorchFunctions(PyObject *module) {
  static std::vector<PyMethodDef> torch_functions;
  gatherTorchFunctions(torch_functions);
  THPVariableFunctions.tp_methods = torch_functions.data();

  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);

  // Steals
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(module, "_VariableFunctionsClass",
                         reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule = PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}

}}  // namespace torch::autograd
