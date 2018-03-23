#pragma once

#include <Python.h>
#include <vector>
#include <utility>
#include <memory>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/utils/object_ptr.h"


namespace torch { namespace jit { struct Graph; }}
namespace torch { namespace autograd {

struct VariableInfo {
  explicit VariableInfo(const Variable& var);

  Variable zeros(AutoGPU& gpu_guard) const;

  at::Type* type;
  int device;
  std::vector<int64_t> size;
  bool requires_grad;
};

// A Function which is implemented by a Python object (i.e., a THPFunction).
// Calls to 'apply' are forwarded to the Python method implementation.
struct PyFunction : public Function {
  PyFunction(PyObject* obj) : obj(obj) {}

  virtual variable_list apply(const variable_list& inputs) override;
  variable_list legacy_apply(const variable_list& inputs);

  virtual void release_variables() override;
  virtual std::string name() override;
  virtual std::shared_ptr<Function> get_shared_ptr() override;
  virtual bool is_traceable() override;

  // THPFunction this Function is wrapping.
  PyObject* obj;
};

/**
 * Cast an object into a tuple, if it is not a tuple already. Returns true
 * if the original object was not a tuple.
 */
inline bool ensure_tuple(THPObjectPtr& obj) {
  if (PyTuple_Check(obj.get()))
    return false;

  PyObject *tuple = PyTuple_New(1);
  if (!tuple) throw python_error();
  PyTuple_SET_ITEM(tuple, 0, obj.release());
  obj = tuple;
  return true;
}

}} // namespace torch::autograd

struct THPFunction {
    PyObject_HEAD

    PyObject *needs_input_grad;

    // Python tuple of tensors whose variables we should save.  Set
    // by Python with 'save_for_backward'.  If nullptr, no tensors were
    // saved.
    PyObject *to_save;
    // Python tuple of tensors which are not differentiable.  Set by
    // Python with 'mark_non_differentiable'.  If nullptr, no tensors were
    // non-differentiable.
    PyObject *non_differentiable;
    // Python tuple of tensors which had inplace updates in the forward()
    // pass.  Set by Python with 'mark_dirty'.  If nullptr, no tensors were
    // modified inplace.
    PyObject *dirty_tensors;

    std::vector<torch::autograd::VariableInfo> output_info;
    std::vector<torch::autograd::VariableInfo> input_info;
    std::vector<torch::autograd::SavedVariable> saved_variables;
    // For each input, true if the input is a THPVariable
    std::vector<bool> is_variable_input;
    char has_freed_buffers;
    char is_traced;

    // The C++ wrapper for this Python function.
    // See a comment in THPFunction_asFunction for details about this field.
    torch::autograd::PyFunction cdata;
};

bool THPFunction_initModule(PyObject *module);
extern PyTypeObject THPFunctionType;
extern PyObject *THPFunctionClass;

// XXX: this function requires the GIL (it can have side effects).
std::shared_ptr<torch::autograd::PyFunction> THPFunction_asFunction(THPFunction* self);

inline bool THPFunction_Check(PyObject* obj) {
  return PyObject_IsInstance(obj, (PyObject*)&THPFunctionType);
}
