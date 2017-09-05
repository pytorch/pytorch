#pragma once

#include <Python.h>
#include <vector>
#include <utility>
#include <memory>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/object_ptr.h"

// (class, gpu id, sizes)
using output_info_type = std::tuple<PyObject *, int, std::vector<int64_t>>;

namespace torch { namespace autograd {

// A Function which is implemented by a Python object (i.e., a THPFunction).
// Calls to 'apply' are forwarded to the Python method implementation.
struct PyFunction : public Function {
  PyFunction(PyObject* obj) : obj(obj) {}

  virtual variable_list apply(const variable_list& inputs) override;
  variable_list legacy_apply(const variable_list& inputs);

  virtual void releaseVariables() override;
  virtual std::string name() override;

  // THPFunction this Function is wrapping.
  PyObject* obj;
};

}} // namespace torch::autograd

struct THPFunction {
    PyObject_HEAD

    PyObject *needs_input_grad;

    // Python tuple of tensors whose variables we should save.  Set
    // by Python with 'save_for_backward'.  If NULL, no tensors were
    // saved.
    PyObject *to_save;
    // Python pairs of distinct tensors which share storage.  Set by
    // Python with 'mark_shared_storage'.  If NULL, no tensors share
    // storage.
    PyObject *shared_pairs;
    // Python tuple of tensors which are not differentiable.  Set by
    // Python with 'mark_non_differentiable'.  If NULL, no tensors were
    // non-differentiable.
    PyObject *non_differentiable;
    // Python tuple of tensors which had inplace updates in the forward()
    // pass.  Set by Python with 'mark_dirty'.  If NULL, no tensors were
    // modified inplace.
    PyObject *dirty_tensors;

    std::vector<output_info_type> *output_info;
    std::vector<torch::autograd::SavedVariable> *saved_variables;
    // For each input, true if the input is a THPVariable
    std::vector<bool> *is_variable_input;
    char has_freed_buffers;

    // The C++ wrapper for this Python function.
    // See a comment in THPFunction_asFunction for details about this field.
    torch::autograd::PyFunction cdata;
};

bool THPFunction_initModule(PyObject *module);
extern PyTypeObject THPFunctionType;
extern PyObject *THPFunctionClass;
extern PyObject *THPStochasticFunctionClass;
extern PyObject *THPBatchNormBackwardBackwardFunction;  // Temporarily here until we move it to C++

// XXX: this function requires the GIL (it can have side effects).
std::shared_ptr<torch::autograd::PyFunction> THPFunction_asFunction(THPFunction* self);

inline bool THPFunction_Check(PyObject* obj) {
  return PyObject_IsInstance(obj, (PyObject*)&THPFunctionType);
}
