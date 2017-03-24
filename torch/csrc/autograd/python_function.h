#pragma once

#include <Python.h>
#include <vector>
#include <utility>
#include <memory>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/object_ptr.h"

// (class, gpu id, sizes)
using output_info_type = std::tuple<PyObject *, int, std::vector<long>>;

namespace torch { namespace autograd {

struct PyFunction : public Function {
  PyFunction(PyObject* obj) : obj(obj) {}

  virtual variable_list apply(const variable_list& inputs) override;
  variable_list legacy_apply(const variable_list& inputs);

  virtual void releaseVariables() override;
  virtual std::string name() override;

  PyObject* obj;
};

}} // namespace torch::autograd

struct THPFunction {
    PyObject_HEAD

    PyObject *needs_input_grad;

    PyObject *to_save;
    PyObject *shared_pairs;
    PyObject *non_differentiable;
    PyObject *dirty_tensors;

    std::vector<output_info_type> *output_info;
    std::vector<torch::autograd::SavedVariable> *saved_variables;
    int num_forward_inputs;
    char has_freed_buffers;

    torch::autograd::PyFunction cdata;
};

bool THPFunction_initModule(PyObject *module);
extern PyTypeObject THPFunctionType;
extern PyObject *THPFunctionClass;
extern PyObject *THPStochasticFunctionClass;

std::shared_ptr<torch::autograd::PyFunction> THPFunction_asFunction(THPFunction* self);

inline bool THPFunction_Check(PyObject* obj) {
  return PyObject_IsInstance(obj, (PyObject*)&THPFunctionType);
}
