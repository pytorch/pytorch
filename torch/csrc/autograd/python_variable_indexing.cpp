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
using namespace at::indexing;
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

[[noreturn]]
static void invalid_index(PyObject* obj) {
  throw IndexError(
    "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
    "Variables are valid indices (got %s)", Py_TYPE(obj)->tp_name);
}

static Variable sequenceToVariable(c10::TensorTypeId type_id, PyObject* seq) {
  return torch::utils::indexing_tensor_from_data(type_id, kLong, c10::nullopt, seq);
}

static bool treatSequenceAsTuple(PyObject* index) {
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

static THPObjectPtr wrapTuple(PyObject* index) {
  THPObjectPtr res;
  if (treatSequenceAsTuple(index)) {
    res = PySequence_Tuple(index);
  } else {
    res = PyTuple_Pack(1, index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  }
  if (!res) throw python_error();
  return res;
}

static std::vector<TensorIndex> indexToTensorIndexList(PyObject* index) {
  THPObjectPtr holder = wrapTuple(index);
  int64_t size = PyTuple_GET_SIZE(holder.get());

  std::vector<TensorIndex> tensor_index_list;
  tensor_index_list.reserve(size);

  for (int64_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i);
    if (THPUtils_checkLong(obj)) {
      if (THPVariable_Check(obj)) {
        tensor_index_list.push_back(TensorIndex(THPUtils_unpackLong(obj), THPVariable_Unpack(obj)));
      } else {
        tensor_index_list.push_back(TensorIndex(THPUtils_unpackLong(obj)));
      }
    } else if (PySlice_Check(obj)) {
      Py_ssize_t start, stop, step;
      if (!THPUtils_unpackSlice(obj, &start, &stop, &step)) {
        throw python_error();
      }

      PySliceObject* sliceobj = (PySliceObject*)obj;
      Tensor start_tensor, stop_tensor, step_tensor;
      if (THPVariable_Check(sliceobj->start)) {
        start_tensor = THPVariable_Unpack(sliceobj->start);
      }
      if (THPVariable_Check(sliceobj->stop)) {
        stop_tensor = THPVariable_Unpack(sliceobj->stop);
      }
      if (THPVariable_Check(sliceobj->step)) {
        step_tensor = THPVariable_Unpack(sliceobj->step);
      }

      tensor_index_list.push_back(TensorIndex({start, stop, step}, {start_tensor, stop_tensor, step_tensor}));
    } else if (obj == Py_Ellipsis) {
      tensor_index_list.push_back(TensorIndex(at::indexing::Ellipsis));
    } else if (obj == Py_None) {
      tensor_index_list.push_back(TensorIndex(at::indexing::None));
    } else if (PyBool_Check(obj)) {
      tensor_index_list.push_back(TensorIndex(obj == Py_True));
    } else if (THPVariable_Check(obj)) {
      tensor_index_list.push_back(TensorIndex(THPVariable_Unpack(obj)));
    } else if (PySequence_Check(obj)) {
      // TODO: Naughty naughty get out of jail free
      // (Fixing this means I have to fix the call chain though :/)
      tensor_index_list.push_back(TensorIndex(sequenceToVariable(legacyExtractTypeId(self), obj)));
    } else {
      auto index = THPObjectPtr(PyNumber_Index(obj));
      if (!index) {
        PyErr_Clear();
        invalid_index(obj);
      }
      if (THPVariable_Check(index)) {
        tensor_index_list.push_back(TensorIndex(THPUtils_unpackLong(index), THPVariable_Unpack(index)));
      } else {
        tensor_index_list.push_back(TensorIndex(THPUtils_unpackLong(index)));
      }
    }
  }
}

PyObject* THPVariable_getitem(PyObject* self, PyObject* index) {
  HANDLE_TH_ERRORS
  pybind11::gil_scoped_release no_gil;

  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  std::vector<TensorIndex> tensor_index_list = indexToTensorIndexList(index);
  return wrap(self_.index(tensor_index_list));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  HANDLE_TH_ERRORS
  if (py_value == nullptr) {
    throw TypeError("Tensor does not support deleting items");
  }
  pybind11::gil_scoped_release no_gil;

  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  std::vector<TensorIndex> tensor_index_list = indexToTensorIndexList(index);

  if (THPVariable_Check(py_value)) {
    self_.index_put_(tensor_index_list, reinterpret_cast<THPVariable*>(py_value)->cdata);
    return 0;
  } else if (THPUtils_checkLong(py_value) || PyBool_Check(py_value)) {
    Scalar v = Scalar(THPUtils_unpackLong(py_value));
    self_.index_put_(tensor_index_list, v);
    return 0;
  } else if (PyFloat_Check(py_value)) {
    Scalar v = Scalar(THPUtils_unpackDouble(py_value));
    self_.index_put_(tensor_index_list, v);
    return 0;
  } else {
    throw TypeError(
      "can't assign a %s to a %s",
      Py_TYPE(value)->tp_name,
      torch::utils::options_to_string(options).c_str());
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}

}} // namespace torch::autograd
