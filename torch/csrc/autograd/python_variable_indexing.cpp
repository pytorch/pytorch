#include <torch/csrc/autograd/python_variable_indexing.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TracerMode.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <vector>
#include <tuple>

using namespace at;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

Py_ssize_t THPVariable_length(PyObject* self) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    py::object ret = py::reinterpret_steal<py::object>(handle_torch_function(self, "__len__"));
    Py_ssize_t length = PyLong_AsSsize_t(ret.ptr());
    if (PyErr_Occurred()) {
      throw python_error();
    }
    return length;
  }
  const auto& self_ = THPVariable_Unpack(self);
  if (self_.dim() == 0) {
    return 0;
  }
  return (Py_ssize_t)self_.size(0);
  END_HANDLE_TH_ERRORS_RET(-1)
}

[[noreturn]]
static inline void invalid_index(PyObject* obj) {
  throw IndexError(
    "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
    "Variables are valid indices (got %s)", Py_TYPE(obj)->tp_name);
}

static inline Variable sequenceToVariable(c10::TensorOptions options, PyObject* seq) {
  return torch::utils::indexing_tensor_from_data(options, kLong, c10::nullopt, seq);
}

static inline Variable valueToTensor(c10::TensorOptions options, PyObject* value, const at::Device& device) {
  if (THPVariable_Check(value)) {
    return THPVariable_Unpack(value);
  }
  at::AutoDispatchBelowADInplaceOrView guard;  // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  if (THPUtils_checkLong(value) || PyBool_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackLong(value)), options, device);
  }
  if (PyFloat_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackDouble(value)), options, device);
  }
  if (PyComplex_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackComplexDouble(value)), options, device);
  }
  throw TypeError(
    "can't assign a %s to a %s",
    Py_TYPE(value)->tp_name,
    torch::utils::options_to_string(options).c_str());
}

static inline void checkUnpackSlice(PyObject* index, Py_ssize_t* start_ptr, Py_ssize_t* stop_ptr, Py_ssize_t* step_ptr) {
  if (!THPUtils_unpackSlice(index, start_ptr, stop_ptr, step_ptr)) {
    throw python_error();
  }
}

static inline void recordSliceTrace(PyObject* obj) {
  PySliceObject* sliceobj = (PySliceObject*)obj;
  if (THPVariable_Check(sliceobj->start)) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("start"), 1, THPVariable_Unpack(sliceobj->start), torch::jit::IntType::get());
  }
  if (THPVariable_Check(sliceobj->stop)) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("end"), 1, THPVariable_Unpack(sliceobj->stop), torch::jit::IntType::get());
  }
  if (THPVariable_Check(sliceobj->step)) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("step"), 1, THPVariable_Unpack(sliceobj->step), torch::jit::IntType::get());
  }
}

static inline void recordSelectTrace(const Tensor& index_tensor) {
  torch::jit::tracer::ArgumentStash::stashValue(std::string("index"), 1, index_tensor, torch::jit::IntType::get());
}

at::indexing::TensorIndex IndexFromSingleObject(
    PyObject *obj, const TensorOptions & options, bool is_tracing) {
  if (THPUtils_checkLong(obj)) {
    if (is_tracing && THPVariable_Check(obj)) {
      recordSelectTrace(THPVariable_Unpack(obj));
    }
    return at::indexing::TensorIndex(THPUtils_unpackLong(obj));
  } else if (PySlice_Check(obj)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, step;
    checkUnpackSlice(obj, &start, &stop, &step);
    if (is_tracing) {
      recordSliceTrace(obj);
    }
    return at::indexing::TensorIndex(at::indexing::Slice(start, stop, step));
  } else if (obj == Py_Ellipsis) {
    return at::indexing::TensorIndex(at::indexing::Ellipsis);
  } else if (obj == Py_None) {
    return at::indexing::TensorIndex(at::indexing::None);
  } else if (PyBool_Check(obj)) {
    return at::indexing::TensorIndex(obj == Py_True);
  } else if (THPVariable_Check(obj)) {
    Tensor tensor = THPVariable_Unpack(obj);
    if (is_tracing) {
      auto scalar_type = tensor.scalar_type();
      if (tensor.dim() == 0 && at::isIntegralType(scalar_type, /*includeBool=*/false) && scalar_type != at::kByte) {
        recordSelectTrace(tensor);
      }
    }
    return at::indexing::TensorIndex(std::move(tensor));
  } else if (PySequence_Check(obj)) {
    return at::indexing::TensorIndex(sequenceToVariable(options, obj));
  } else {
    auto idx = THPObjectPtr(PyNumber_Index(obj));
    if (!idx) {
      PyErr_Clear();
      invalid_index(obj);
    }
    if (is_tracing && THPVariable_Check(idx)) {
      recordSelectTrace(THPVariable_Unpack(idx));
    }
    return at::indexing::TensorIndex(THPUtils_unpackLong(idx));
  }
}

static bool IndicesFromTuple(
    std::vector<at::indexing::TensorIndex> &indices,
    PyObject *index, const TensorOptions & options, bool is_tracing) {
  int64_t size = PyTuple_GET_SIZE(index);
  indices.clear();
  indices.reserve(size);
  for(const auto i : c10::irange(size)) {
    PyObject* obj = PyTuple_GET_ITEM(index, i);
    if (!THPVariable_CheckExact(obj) && check_has_torch_function(obj)) {
      return true;
    }

    indices.emplace_back(IndexFromSingleObject(obj, options, is_tracing));
  }
  return false;
}

static inline bool treatSequenceAsTuple(PyObject* index) {
  if (PyTuple_Check(index)) {
    return true;
  }
  if (THPVariable_Check(index)) {
    return false;
  }
  if (!PySequence_Check(index)) {
    return false;
  }
  // This uses a heuristics from NumPy for determining whether to treat
  // non-tuple sequences as if they were a tuple. From the NumPy code comments:
  //
  // "At this point, we're left with a non-tuple, non-array, sequence:
  //  typically, a list. We use some somewhat-arbitrary heuristics from here
  //  onwards to decided whether to treat that list as a single index, or a
  //  list of indices. Backwards compatibility only takes effect for short
  //  sequences - otherwise we treat it like any other scalar."
  auto n = PySequence_Size(index);
  if (n < 0) {
    // Negative size indicates a Python error in the PySequence_Size call.
    PyErr_Clear();
    return false;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  if (n >= 32) {
    return false;
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    auto obj = THPObjectPtr{PySequence_GetItem(index, i)};
    if (!obj.get()) {
      PyErr_Clear();
      return false;
    }
    if (THPVariable_Check(obj.get()) || PySequence_Check(obj.get()) || PySlice_Check(obj.get())) {
      return true;
    }
    if (obj.get() == Py_Ellipsis || obj.get() == Py_None) {
      return true;
    }
  }
  return false;
}

static inline THPObjectPtr wrapTuple(PyObject* index) {
  THPObjectPtr res;
  if (treatSequenceAsTuple(index)) {
    res = PySequence_Tuple(index);
  } else {
    res = PyTuple_Pack(1, index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  }
  if (!res) throw python_error();
  return res;
}


static inline Tensor dispatch_get_item(
    const Tensor& self, const ArrayRef<at::indexing::TensorIndex>& indices,
    bool disable_slice_optimization = false) {
  pybind11::gil_scoped_release no_gil;
  return at::indexing::get_item(self, indices, disable_slice_optimization);
}

// NOTE: Here is the dispatch structure for `THPVariable_getitem`:
//
// 1. Python 1-D getter calls C++ `at::indexing::get_item` after
// converting Python index to C++ TensorIndex.
//
// 2. Python N-D getter calls C++ `at::indexing::handleDimInMultiDimIndexing`
// for each dim, after converting Python index to C++ TensorIndex. If advanced
// indexing is needed, it calls C++ `at::indexing::dispatch_index`.
PyObject* THPVariable_getitem(PyObject* self, PyObject* index) {
  HANDLE_TH_ERRORS
  if (!THPVariable_CheckExact(self) && check_has_torch_function(self)) {
    return handle_torch_function_indexing(self, index);
  }
  const auto& self_ = THPVariable_Unpack(self);
  OptionalDeviceGuard device_guard(device_of(self_));

  // handle simple types: none, ellipsis
  if (index == Py_None) {
    return THPVariable_Wrap(
      dispatch_get_item(self_, {at::indexing::TensorIndex(at::indexing::None)}));
  } else if (index == Py_Ellipsis) {
    return THPVariable_Wrap(
      dispatch_get_item(self_, {at::indexing::TensorIndex(at::indexing::Ellipsis)}));
  }

  bool is_tracing = torch::jit::tracer::isTracing();

  // handle simple types: integers, slices, bool
  if (THPUtils_checkLong(index)) {
    if (is_tracing && THPVariable_Check(index)) {
      recordSelectTrace(THPVariable_Unpack(index));
    }
    return THPVariable_Wrap(
      dispatch_get_item(self_, {at::indexing::TensorIndex(THPUtils_unpackLong(index))}));
  } else if (PySlice_Check(index)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, step;
    checkUnpackSlice(index, &start, &stop, &step);
    if (is_tracing) {
      recordSliceTrace(index);
    }
    return THPVariable_Wrap(
      dispatch_get_item(self_, {at::indexing::TensorIndex(at::indexing::Slice(start, stop, step))}));
  } else if (index == Py_False || index == Py_True) {
    return THPVariable_Wrap(
        dispatch_get_item(self_, {at::indexing::TensorIndex(index == Py_True)}));
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);
  std::vector<at::indexing::TensorIndex> indices;
  bool call_torch_function = IndicesFromTuple(indices, holder.get(), self_.options(), is_tracing);
  if (call_torch_function) {
    return handle_torch_function_indexing(self, index);
  }

  Tensor sliced;
  {
    pybind11::gil_scoped_release no_gil;
    variable_list variableIndices;
    sliced = at::indexing::impl::applySlicing(
      self_, indices, variableIndices, /*disable_slice_optimization=*/is_tracing,
      self_.device(), self_.sizes());

    // indexing by tensors ("advanced" indexing)
    if (!variableIndices.empty()) {
      sliced = at::indexing::dispatch_index(sliced, std::move(variableIndices));
    }

    // ensure we return a shallow copy for things like x[...]
    if (sliced.is_same(self_)) {
      sliced = at::alias(sliced);
    }
  }

  return THPVariable_Wrap(sliced);
  END_HANDLE_TH_ERRORS
}

TORCH_PYTHON_API bool CheckGil() {
  return PyGILState_Check();
}

void dispatch_set_item(const Tensor& self, ArrayRef<at::indexing::TensorIndex> indices,
                       const Tensor& value, bool disable_slice_optimization=false) {
  pybind11::gil_scoped_release no_gil;
  at::indexing::set_item(self, indices, value, disable_slice_optimization);
}

// NOTE: Here is the dispatch structure for `THPVariable_setitem`:
//
// 1. Python 1-D setter calls C++ `at::indexing::set_item` after
// converting Python index to C++ TensorIndex.
//
// 2. Python N-D setter calls C++ `at::indexing::handleDimInMultiDimIndexing`
// for each dim, after converting Python index to C++ TensorIndex. If advanced
// indexing is needed, it calls C++ `at::indexing::dispatch_index_put_`.
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  HANDLE_TH_ERRORS
  if (py_value == nullptr) {
    throw TypeError("Tensor does not support deleting items");
  }
  if ((!THPVariable_CheckExact(self) && check_has_torch_function(self)) ||
      (!THPVariable_CheckExact(py_value) && check_has_torch_function(py_value))) {
    py::object ret = py::reinterpret_steal<py::object>(
      handle_torch_function_indexing(self, index, py_value)
    );
    return 0;
  }

  const auto& self_ = THPVariable_Unpack(self);
  if (self_.is_sparse())
  {
    throw TypeError("Cannot assign to a sparse tensor");
  }
  OptionalDeviceGuard device_guard(device_of(self_));
  at::Device self_device = self_.device();
  Variable value;
  // TODO: This qint special case looks very suspicious...
  if (isQIntType(self_.scalar_type())) {
    value = valueToTensor(device(kCPU).dtype(kFloat), py_value, at::Device(kCPU));
  } else if (self_device.is_cuda()) {
    value = valueToTensor(self_.options(), py_value, at::Device(kCPU));
  } else {
    value = valueToTensor(self_.options(), py_value, self_device);
  }

  // handle simple types: ellipsis, none, bool
  if (index == Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    // do nothing for false (technically we should check the size, but we don't have
    // real 0-sized shapes.
    return 0;
  } else if (index == Py_Ellipsis) {
    dispatch_set_item(self_, {at::indexing::TensorIndex(at::indexing::Ellipsis)}, value);
    return 0;
  } else if (index == Py_None) {
    dispatch_set_item(self_, {at::indexing::TensorIndex(at::indexing::None)}, value);
    return 0;
  } else if (index == Py_True) {
    dispatch_set_item(self_, {at::indexing::TensorIndex(true)}, value);
    return 0;
  }

  bool is_tracing = torch::jit::tracer::isTracing();

  // handle simple types: integers, slices
  if (THPUtils_checkLong(index)) {
    if (is_tracing && THPVariable_Check(index)) {
      recordSelectTrace(THPVariable_Unpack(index));
    }
    dispatch_set_item(self_, {at::indexing::TensorIndex(THPUtils_unpackLong(index))}, value);
    return 0;
  } else if (PySlice_Check(index)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, step;
    checkUnpackSlice(index, &start, &stop, &step);
    if (is_tracing) {
      recordSliceTrace(index);
    }
    // See NOTE [ Setting `disable_slice_optimization` when calling C++ tensor indexing functions from Python ]
    dispatch_set_item(
      self_, {at::indexing::TensorIndex(at::indexing::Slice(start, stop, step))}, value, /*disable_slice_optimization=*/is_tracing);
    return 0;
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);
  std::vector<at::indexing::TensorIndex> indices;
  bool call_torch_function = IndicesFromTuple(indices, holder.get(), self_.options(), is_tracing);
  if (call_torch_function) {
    py::object val = py::reinterpret_steal<py::object>(
      handle_torch_function_indexing(self, index, py_value)
    );
    return 0;
  }

  variable_list variableIndices;
  {
    pybind11::gil_scoped_release no_gil;
    Variable sliced = at::indexing::impl::applySlicing(
        self_, indices, variableIndices, /*is_tracing=*/is_tracing,
        self_device, self_.sizes());
    if (variableIndices.empty()) {
      at::indexing::copy_to(sliced, value);
      return 0;
    }

    IntArrayRef valueSizes = value.sizes();
    IntArrayRef slicedValueSizes = at::indexing::slicePrefix1sSize(valueSizes);
    torch::autograd::Variable valuesSliced;
    if (!valueSizes.equals(slicedValueSizes)) {
      valuesSliced = value.view(slicedValueSizes);
    } else {
      valuesSliced = value;
    }
    at::indexing::dispatch_index_put_(sliced, std::move(variableIndices), valuesSliced);
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

}} // namespace torch::autograd
