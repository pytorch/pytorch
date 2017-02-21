#include <Python.h>
#include "batch_normalization.h"
#include "torch/csrc/autograd/python_cpp_function.h"

using namespace torch::autograd;

static PyTypeObject BatchNormClass;
static PyTypeObject BatchNormBackwardClass;

struct BatchNormCtor {
  BatchNormForward* operator()(PyObject* args) {
    std::unique_ptr<thpp::Tensor> running_mean;
    std::unique_ptr<thpp::Tensor> running_var;
    char training;
    double momentum;
    double eps;

    if (!PyArg_ParseTuple(args, "O&O&Bdd:BatchNorm",
          TensorConverter, &running_mean,
          TensorConverter, &running_var,
          &training, &momentum, &eps)) {
      return NULL;
    }

    return new BatchNormForward(
        std::move(running_mean),
        std::move(running_var),
        (bool)training,
        momentum,
        eps);
  }
};

struct NoCtor {
  Function* operator()(PyObject* args) {
    throw std::runtime_error("Cannot construct");
  }
};

template<typename C, typename T>
static void addClass(PyObject* module, PyTypeObject& type, const char* name)
{
  createForwardFunctionPyTypeObject<T>(type, name);
  Py_INCREF(&type);
  PyModule_AddObject(module, name, (PyObject*)&type);
  registerCppFunction(typeid(C), &type);
}

bool THPAutograd_initFunctions(PyObject* _unused)
{
  THPObjectPtr module = PyImport_ImportModule("torch.nn._functions.thnn");
  addClass<BatchNormForward, BatchNormCtor>(module, BatchNormClass, "BatchNorm");
  addClass<BatchNormBackward, NoCtor>(module, BatchNormBackwardClass, "BatchNormBackward");
  return true;
}
