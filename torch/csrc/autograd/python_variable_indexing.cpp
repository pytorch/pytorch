#include <torch/csrc/autograd/python_variable_indexing.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP_export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/TensorIndexing.h>

#include <vector>
#include <tuple>

using namespace at;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

Py_ssize_t THPVariable_length(PyObject* self) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.dim() == 0) {
    return 0;
  }
  return (Py_ssize_t)self_.size(0);
  END_HANDLE_TH_ERRORS_RET(-1)
}


// We allow indexing by integers, slices, ellipsis, None, Variables,
// and tuples of those types. We also handle bools as if they were a
// Variable[ByteTensor].

static inline int64_t count_specified_dimensions(PyObject* index) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  auto size = PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    if (THPVariable_Check(obj)) {
      auto& var = reinterpret_cast<THPVariable*>(obj)->cdata;
      if (var.scalar_type() == kByte || var.scalar_type() == kBool) {
        count += var.dim();
      } else {
        count++;
      }
    } else if (obj != Py_None && obj != Py_Ellipsis && obj != Py_True && obj != Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
      count++;
    }
  }
  return count;
}

[[noreturn]]
static inline void invalid_index(PyObject* obj) {
  throw IndexError(
    "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
    "Variables are valid indices (got %s)", Py_TYPE(obj)->tp_name);
}

static inline Variable sequenceToVariable(c10::DispatchKey dispatch_key, PyObject* seq) {
  return torch::utils::indexing_tensor_from_data(dispatch_key, kLong, c10::nullopt, seq);
}

static inline Variable valueToTensor(c10::TensorOptions options, PyObject* value) {
  if (THPVariable_Check(value)) {
    return reinterpret_cast<THPVariable*>(value)->cdata;
  }
  at::AutoNonVariableTypeMode guard;
  if (THPUtils_checkLong(value) || PyBool_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackLong(value)), options);
  }
  if (PyFloat_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackDouble(value)), options);
  }
  throw TypeError(
    "can't assign a %s to a %s",
    Py_TYPE(value)->tp_name,
    torch::utils::options_to_string(options).c_str());
}

static inline void extractTensorsFromSlice(
  PyObject* obj,
  Tensor& start_tensor,
  Tensor& stop_tensor,
  Tensor& step_tensor) {
  PySliceObject* sliceobj = (PySliceObject*)obj;
  if (THPVariable_Check(sliceobj->start)) {
    start_tensor = THPVariable_Unpack(sliceobj->start);
  }
  if (THPVariable_Check(sliceobj->stop)) {
    stop_tensor = THPVariable_Unpack(sliceobj->stop);
  }
  if (THPVariable_Check(sliceobj->step)) {
    step_tensor = THPVariable_Unpack(sliceobj->step);
  }
}

static inline void recordSliceTrace(const Tensor& start_tensor, const Tensor& stop_tensor, const Tensor& step_tensor) {
  if (start_tensor.defined()) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("start"), 1, start_tensor, torch::jit::IntType::get());
  }
  if (stop_tensor.defined()) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("end"), 1, stop_tensor, torch::jit::IntType::get());
  }
  if (step_tensor.defined()) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("step"), 1, step_tensor, torch::jit::IntType::get());
  }
}

static inline void recordSelectTrace(const Tensor& index_tensor) {
  torch::jit::tracer::ArgumentStash::stashValue(std::string("index"), 1, index_tensor, torch::jit::IntType::get());
}

static inline PyObject* convertToPythonInt(PyObject* obj) {
  auto idx = THPObjectPtr(PyNumber_Index(obj));
  if (!idx) {
    PyErr_Clear();
    invalid_index(obj);
  }
  return idx;
}

static inline Variable applySlicing(const Variable& self, PyObject* index, variable_list& outIndices, bool is_tracing) {
  int64_t size = PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  int64_t dim = 0;
  int64_t specified_dims = count_specified_dimensions(index);

  if (specified_dims > self.dim()) {
    throw IndexError("too many indices for tensor of dimension %d", (int)(self.dim()));
  }

  Variable result = self;
  for (int64_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)

    // Fast path for integer indexing
    if (THPUtils_checkLong(obj)) {
      if (is_tracing && THPVariable_Check(obj)) {
        recordSelectTrace(THPVariable_Unpack(obj));
      }
      result = at::indexing::applySelect(result, dim, THPUtils_unpackLong(obj), i);
    } else {
      auto tensor_index = ([&]() {
        if (THPUtils_checkLong(obj)) {
          if (is_tracing && THPVariable_Check(obj)) {
            recordSelectTrace(THPVariable_Unpack(obj));
          }
          return at::indexing::TensorIndex(THPUtils_unpackLong(obj));
        } else if (PySlice_Check(obj)) {
          Py_ssize_t start, stop, step;
          if (!THPUtils_unpackSlice(obj, &start, &stop, &step)) {
            throw python_error();
          }
          if (is_tracing) {
            Tensor start_tensor, stop_tensor, step_tensor;
            extractTensorsFromSlice(obj, start_tensor, stop_tensor, step_tensor);
            recordSliceTrace(start_tensor, stop_tensor, step_tensor);
          }
          return at::indexing::TensorIndex({start, stop, step});
        } else if (obj == Py_Ellipsis) {
          return at::indexing::TensorIndex(at::indexing::Ellipsis);
        } else if (obj == Py_None) {
          return at::indexing::TensorIndex(at::indexing::None);
        } else if (PyBool_Check(obj)) {
          return at::indexing::TensorIndex(obj == Py_True);
        } else if (THPVariable_Check(obj)) {
          if (is_tracing) {
            Tensor tensor = THPVariable_Unpack(obj);
            auto scalar_type = tensor.scalar_type();
            if (tensor.dim() == 0 && at::isIntegralType(scalar_type, /*includeBool=*/false) && scalar_type != at::kByte) {
              recordSelectTrace(tensor);
            }
          }
          return at::indexing::TensorIndex(THPVariable_Unpack(obj));
        } else if (PySequence_Check(obj)) {
          // TODO: Naughty naughty get out of jail free
          // (Fixing this means I have to fix the call chain though :/)
          return at::indexing::TensorIndex(sequenceToVariable(legacyExtractDispatchKey(self), obj));
        } else {
          auto idx = convertToPythonInt(obj);
          if (is_tracing && THPVariable_Check(idx)) {
            recordSelectTrace(THPVariable_Unpack(idx));
          }
          return at::indexing::TensorIndex(THPUtils_unpackLong(idx));
        }
      })();

      result = at::indexing::handleDimInMultiDimIndexing(
        /*prev_dim_result=*/result,
        /*original_tensor=*/self,
        /*index=*/tensor_index,
        /*dim_ptr=*/&dim,
        /*specified_dims_ptr=*/&specified_dims,
        /*real_dim=*/i,
        /*outIndices=*/outIndices,
        /*is_tracing=*/is_tracing);
    }
  }
  return result;
}

static inline bool treatSequenceAsTuple(PyObject* index) {
  if (PyTuple_Check(index)) {
    return true;
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

PyObject* THPVariable_getitem(PyObject* self, PyObject* index) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  bool is_tracing = torch::jit::tracer::isTracing();

  // handle simple types: integers, slices, ellipsis, none
  if (index == Py_None) {
    return THPVariable_Wrap(self_.unsqueeze(0));
  } else if (index == Py_Ellipsis) {
    return THPVariable_Wrap(at::alias(self_));
  } else if (THPUtils_checkLong(index)) {
    if (is_tracing && THPVariable_Check(index)) {
      recordSelectTrace(THPVariable_Unpack(index));
    }
    return THPVariable_Wrap(at::indexing::applySelect(self_, 0, THPUtils_unpackLong(index)));
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, step;
    if (!THPUtils_unpackSlice(index, &start, &stop, &step)) {
      throw python_error();
    }
    if (is_tracing) {
      Tensor start_tensor, stop_tensor, step_tensor;
      extractTensorsFromSlice(index, start_tensor, stop_tensor, step_tensor);
      recordSliceTrace(start_tensor, stop_tensor, step_tensor);
    }
    return THPVariable_Wrap(at::indexing::applySlice(
      self_,
      0,
      start,
      stop,
      step,
      /*ensure_view=*/true,
      /*is_tracing=*/is_tracing));
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  Variable sliced = applySlicing(self_, holder.get(), variableIndices, /*is_tracing=*/is_tracing);
  if (variableIndices.empty()) {
    if (sliced.is_same(self_)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = at::alias(sliced);
    }
    return THPVariable_Wrap(sliced);
  }

  // indexing by tensors ("advanced" indexing)
  return THPVariable_Wrap(([&]() {
    pybind11::gil_scoped_release no_gil;
    return at::indexing::dispatch_index(sliced, variableIndices);
  })());

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  HANDLE_TH_ERRORS
  if (py_value == nullptr) {
    throw TypeError("Tensor does not support deleting items");
  }
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  Variable value;
  // TODO: This qint special case looks very suspicious...
  if (isQIntType(self_.scalar_type())) {
    value = valueToTensor(device(kCPU).dtype(kFloat), py_value);
  } else {
    value = valueToTensor(self_.options(), py_value);
  }
  bool is_tracing = torch::jit::tracer::isTracing();

  // handle simple types: integers, slices, ellipsis, none, bool
  if (index == Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    // do nothing for false (technically we should check the size, but we don't have
    // real 0-sized shapes.
    return 0;
  } else if (index == Py_Ellipsis) {
    at::indexing::handleSimpleTypesInSingleDimIndexingSet(
      self_,
      at::indexing::TensorIndex(at::indexing::Ellipsis),
      value,
      /*is_tracing=*/is_tracing);
    return 0;
  } else if (index == Py_None) {
    at::indexing::handleSimpleTypesInSingleDimIndexingSet(
      self_,
      at::indexing::TensorIndex(at::indexing::None),
      value,
      /*is_tracing=*/is_tracing);
    return 0;
  } else if (index == Py_True) {
    at::indexing::handleSimpleTypesInSingleDimIndexingSet(
      self_,
      at::indexing::TensorIndex(true),
      value,
      /*is_tracing=*/is_tracing);
    return 0;
  } else if (THPUtils_checkLong(index)) {
    if (is_tracing && THPVariable_Check(index)) {
      recordSelectTrace(THPVariable_Unpack(index));
    }
    at::indexing::handleSimpleTypesInSingleDimIndexingSet(
      self_,
      at::indexing::TensorIndex(THPUtils_unpackLong(index)),
      value,
      /*is_tracing=*/is_tracing);
    return 0;
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, step;
    if (!THPUtils_unpackSlice(index, &start, &stop, &step)) {
      throw python_error();
    }
    if (is_tracing) {
      Tensor start_tensor, stop_tensor, step_tensor;
      extractTensorsFromSlice(index, start_tensor, stop_tensor, step_tensor);
      recordSliceTrace(start_tensor, stop_tensor, step_tensor);
    }
    at::indexing::handleSimpleTypesInSingleDimIndexingSet(
      self_,
      at::indexing::TensorIndex({start, stop, step}),
      value,
      /*is_tracing=*/is_tracing);
    return 0;
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  Variable sliced = applySlicing(self_, holder.get(), variableIndices, /*is_tracing=*/is_tracing);
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
  {
    pybind11::gil_scoped_release no_gil;
    at::indexing::dispatch_index_put_(sliced, variableIndices, valuesSliced);
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

}} // namespace torch::autograd
