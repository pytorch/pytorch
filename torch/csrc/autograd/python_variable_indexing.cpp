#include "torch/csrc/autograd/python_variable_indexing.h"

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/THP_export.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/python_compat.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_conversion_dispatch.h"

#include <ATen/ExpandUtils.h>
#include <vector>

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

static int64_t count_specified_dimensions(PyObject* index) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  auto size = PyTuple_GET_SIZE(index);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i);
    if (THPVariable_Check(obj)) {
      auto& var = reinterpret_cast<THPVariable*>(obj)->cdata;
      if (var.type().scalarType() == kByte) {
        count += var.dim();
      } else {
        count++;
      }
    } else if (obj != Py_None && obj != Py_Ellipsis) {
      count++;
    }
  }
  return count;
}

[[noreturn]]
static void invalid_index(PyObject* obj) {
  throw IndexError(
    "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
    "Variables are valid indices (got %s)", Py_TYPE(obj)->tp_name);
}

static Variable applySlice(const Variable& self, int64_t dim, PyObject* slice, bool ensure_view=false) {
  Py_ssize_t start, stop, step, slicelength;
  auto length = self.size(dim);
  if (!THPUtils_parseSlice(slice, length, &start, &stop, &step, &slicelength)) {
    throw python_error();
  }
  if (step == 0) {
    throw ValueError("step cannot be zero");
  }
  if (step < 0) {
    // TODO: implement negative step
    throw ValueError("negative step not yet supported");
  }
  if (!ensure_view && start == 0 && stop == length && step == 1) {
    return self;
  }
  return self.slice(dim, start, stop, step);
}

static Variable applySelect(const Variable& self, int64_t dim, int64_t index) {
  if (index == 0 && dim == 0 && self.dim() == 0) {
    // Deprecated support for indexing 0-dim tensors as if they were 1-dim.
    PyErr_WarnEx(PyExc_UserWarning,
        "invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. "
        "Use tensor.item() to convert a 0-dim tensor to a Python number", 1);
    return at::alias(self);
  }
  int64_t size = self.size(dim);
  if (index < -size || index >= size) {
    throw IndexError("index %lld is out of bounds for dimension %lld with size %lld",
      index, dim, size);
  }
  if (index < 0) {
    index += size;
  }
  return self.select(dim, index);
}

static Variable sequenceToVariable(const Type& type, PyObject* seq) {
  auto& idx_type = type.toScalarType(kLong);
  return torch::utils::legacy_new_from_data(idx_type, at::nullopt, seq);
}

static Variable valueToTensor(const Type & type, PyObject* value) {
  if (THPVariable_Check(value)) {
    return reinterpret_cast<THPVariable*>(value)->cdata;
  }
  if (THPUtils_checkLong(value)) {
    return type.scalarTensor(Scalar(THPUtils_unpackLong(value)));
  }
  if (PyFloat_Check(value)) {
    return type.scalarTensor(Scalar(THPUtils_unpackDouble(value)));
  }
  throw TypeError("can't assign a %s to a %s", Py_TYPE(value)->tp_name, type.toString());
}

static Variable applySlicing(const Variable& self, PyObject* index, variable_list& outIndices) {
  int64_t size = PyTuple_GET_SIZE(index);
  int64_t dim = 0;
  int64_t specified_dims = count_specified_dimensions(index);

  auto handle_var = [&](const Variable& var) {
    // TODO: check scalarType
    outIndices.resize(dim + 1);
    outIndices[dim] = var;
    dim++;
  };

  if (specified_dims > self.dim()) {
    throw IndexError("too many indices for tensor of dimension %d", (int)self.dim());
  }

  Variable result = self;
  for (int64_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i);
    if (THPUtils_checkLong(obj)) {
      result = applySelect(result, dim, THPUtils_unpackLong(obj));
    } else if (PySlice_Check(obj)) {
      result = applySlice(result, dim, obj);
      dim++;
    } else if (obj == Py_Ellipsis) {
      dim += self.dim() - specified_dims;
    } else if (obj == Py_None) {
      result = result.unsqueeze(dim);
      dim++;
    } else if (THPVariable_Check(obj)) {
      auto& var = THPVariable_Unpack(obj);
      auto scalar_type = var.type().scalarType();
      if (var.dim() == 0 && at::isIntegralType(scalar_type) && scalar_type != at::kByte) {
        result = applySelect(result, dim, THPUtils_unpackLong(obj));
      } else {
        handle_var(var);
      }
    } else if (PySequence_Check(obj)) {
      handle_var(sequenceToVariable(self.type(), obj));
    } else {
      auto index = THPObjectPtr(PyNumber_Index(obj));
      if (!index) {
        PyErr_Clear();
        invalid_index(obj);
      }
      result = applySelect(result, dim, THPUtils_unpackLong(index));
    }
  }
  return result;
}

static std::vector<Tensor> typeConvertIndices(const Variable& self, const variable_list& indices) {
  std::vector<Tensor> converted_inds(indices.size());
  int64_t device = self.is_cuda() ? self.get_device() : -1;
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto &ind = indices[i];
    if (ind.defined()) {
      auto& new_type = ind.type().toBackend(self.type().backend());
      converted_inds[i] = torch::utils::dispatch_type_conversion(ind, new_type, device, false);
    } else {
      converted_inds[i] = indices[i];
    }
  }
  return converted_inds;
}

static Variable dispatch_index(const Variable& self, const variable_list& indices) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index(converted_indices);
}

static Variable dispatch_index_put_(Variable& self, const variable_list& indices, const Variable& value) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_put_(converted_indices, value);
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
    res = PyTuple_Pack(1, index);
  }
  if (!res) throw python_error();
  return res;
}

static bool isSingleBoolScalar(const variable_list& vars) {
  return vars.size() == 1 && vars[0].type().scalarType() == ScalarType::Byte && vars[0].dim() == 0;
}

static PyObject* applyBoolGetitem(const Variable& self, bool index) {
  if (index) {
    return wrap(self.type().copy(self.unsqueeze(0)));
  } else {
    return wrap(self.type().tensor({0}));
  }
}

PyObject* THPVariable_getitem(PyObject* self, PyObject* index) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);

  // handle simple types: integers, slices, ellipsis
  if (index == Py_None) {
    return wrap(self_.unsqueeze(0));
  } else if (index == Py_Ellipsis) {
    return wrap(at::alias(self_));
  } else if (THPUtils_checkLong(index)) {
    return wrap(applySelect(self_, 0, THPUtils_unpackLong(index)));
  } else if (PyBool_Check(index)) {
    return applyBoolGetitem(self_, index == Py_True);
  } else if (PySlice_Check(index)) {
    return wrap(applySlice(self_, 0, index, true));
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  Variable sliced = applySlicing(self_, holder.get(), variableIndices);
  if (variableIndices.empty()) {
    if (sliced.is_same(self_)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = at::alias(sliced);
    }
    return wrap(sliced);
  }
  if (isSingleBoolScalar(variableIndices)) {
    return applyBoolGetitem(self_, variableIndices[0].toCByte());
  }

  // indexing by tensors ("advanced" indexing)
  return wrap(dispatch_index(sliced, variableIndices));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static void copy_to(Variable dst, const Variable& src) {
  Tensor b_src;
  // To match numpy semantics:
  // As a special case for backwards compatibility,
  // strip away unit dimensions from the left of 'src'
  auto src_sizes = src.sizes();
  size_t first_nonzero_src = src_sizes.size();
  for (size_t i = 0; i < src_sizes.size(); ++i) {
    if (src_sizes[i] != 1) {
      first_nonzero_src = i;
      break;
    }
  }

  src_sizes = src_sizes.slice(first_nonzero_src);
  std::tie(b_src) = expand_inplace(dst, src.view(src_sizes), "setitem");
  dst.copy_(b_src);
}

int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  auto value = valueToTensor(self_.type(), py_value);

  // handle simple types: integers, slices, ellipsis, bool
  if (index == Py_False) {
    // do nothing for false (technically we should check the size, but we don't have
    // real 0-sized shapes.
    return 0;
  } else if (index == Py_Ellipsis) {
    copy_to(self_, value);
    return 0;
  } else if (index == Py_None || index == Py_True) {
    copy_to(self_.unsqueeze(0), value);
    return 0;
  } else if (THPUtils_checkLong(index)) {
    copy_to(applySelect(self_, 0, THPUtils_unpackLong(index)), value);
    return 0;
  } else if (PySlice_Check(index)) {
    copy_to(applySlice(self_, 0, index), value);
    return 0;
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  Variable sliced = applySlicing(self_, holder.get(), variableIndices);
  if (variableIndices.empty()) {
    copy_to(sliced, value);
    return 0;
  }
  if (isSingleBoolScalar(variableIndices)) {
    if (variableIndices[0].toCByte()) {
      copy_to(self_.unsqueeze(0), value);
    }
    return 0;
  }

  // indexing by tensors ("advanced" indexing)
  dispatch_index_put_(sliced, variableIndices, value);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

}} // namespace torch::autograd
