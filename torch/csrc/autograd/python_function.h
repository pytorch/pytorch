#pragma once

#include <Python.h>
#include <vector>
#include <utility>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/object_ptr.h"

// (class, gpu id, sizes)
using output_info_type = std::tuple<PyObject *, int, std::vector<long>>;
// (tensor, version when saved, version counter)
// or
// (None, 0, nullptr)
using saved_var_info_type = std::tuple<THPObjectPtr, int, std::unique_ptr<torch::autograd::VariableVersion>>;

namespace torch { namespace autograd {

struct PyFunction : public Function {
  PyFunction(PyObject* obj) : obj(obj) {}

  virtual variable_list apply(const variable_list& inputs) override;
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
    std::vector<saved_var_info_type> *saved_variables;
    int num_inputs;
    char has_freed_buffers;

    torch::autograd::PyFunction cdata;
};

bool THPFunction_initModule(PyObject *module);
extern PyObject *THPFunctionClass;
extern PyObject *THPStochasticFunctionClass;

std::shared_ptr<torch::autograd::PyFunction> THPFunction_asFunction(THPFunction* self);

inline bool THPFunction_Check(PyObject* obj) {
  return PyObject_IsInstance(obj, THPFunctionClass);
}
