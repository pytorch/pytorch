#include <Python.h>
#include "batch_normalization.h"
#include "convolution.h"
#include "accumulate_grad.h"
#include "basic_ops.h"
#include "tensor.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/utils/tuple_parser.h"

using namespace torch::autograd;
using torch::TupleParser;

struct BatchNormCtor {
  BatchNormForward* operator()(PyObject* args) {
    BatchNormParams params;

    TupleParser parser(args, 6);
    parser.parse(params.running_mean);
    parser.parse(params.running_var);
    parser.parse(params.training);
    parser.parse(params.momentum);
    parser.parse(params.eps);
    parser.parse(params.cudnn_enabled);

    return new BatchNormForward(std::move(params));
  }
};

struct ConvCtor {
  ConvForward* operator()(PyObject* args) {
    ConvParams params;

    TupleParser parser(args, 8);
    parser.parse(params.stride);
    parser.parse(params.padding);
    parser.parse(params.dilation);
    parser.parse(params.transposed);
    parser.parse(params.output_padding);
    parser.parse(params.groups);
    parser.parse(params.benchmark);
    parser.parse(params.cudnn_enabled);

    return new ConvForward(std::move(params));
  }
};

struct DelayedErrorCtor {
  DelayedError* operator()(PyObject* args) {
    std::string msg;

    TupleParser parser(args, 1);
    parser.parse(msg);

    return new DelayedError(msg);
  }
};

struct NoCtor {
  Function* operator()(PyObject* args) {
    throw std::runtime_error("Cannot construct");
  }
};

template<typename C, typename T, int args, int required_args>
static void addClass(PyObject* module, PyTypeObject& type, const char* name)
{
  createForwardFunctionPyTypeObject<T, args, required_args>(type, name);
  Py_INCREF(&type);
  PyModule_AddObject(module, name, (PyObject*)&type);
  registerCppFunction(typeid(C), &type);
}

bool THPAutograd_initFunctions(PyObject* _unused)
{
  THPObjectPtr module = PyModule_New("torch._C._functions");
  if (!module) return false;

  static PyTypeObject BatchNormClass, BatchNormBackwardClass;
  addClass<BatchNormForward, BatchNormCtor, 3, 1>(module, BatchNormClass, "BatchNorm");
  addClass<BatchNormBackward, NoCtor, 1, 1>(module, BatchNormBackwardClass, "BatchNormBackward");

  static PyTypeObject ConvClass, ConvBackwardClass;
  addClass<ConvForward, ConvCtor, 3, 2>(module, ConvClass, "ConvNd");
  addClass<ConvBackward, NoCtor, 1, 1>(module, ConvBackwardClass, "ConvNdBackward");

  static PyTypeObject AccumulateGradClass;
  addClass<AccumulateGrad, NoCtor>(module, AccumulateGradClass, "AccumulateGrad");

  static PyTypeObject AddClass, AddBackwardClass;
  addClass<Add, NoCtor>(module, AddClass, "Add");
  addClass<AddBackward, NoCtor>(module, AddBackwardClass, "AddBackward");

  static PyTypeObject ErrorClass;
  addClass<Error, NoCtor>(module, ErrorClass, "Error");

  static PyTypeObject DelayedErrorClass;
  addClass<DelayedError, DelayedErrorCtor>(module, DelayedErrorClass, "DelayedError");

  static PyTypeObject CloneClass;
  addClass<Clone, NoCtor>(module, CloneClass, "Clone");

  static PyTypeObject IdentityClass;
  addClass<Identity, NoCtor>(module, IdentityClass, "Identity");

  THPObjectPtr parent = PyImport_ImportModule("torch._C");
  if (!parent) return false;
  PyModule_AddObject(parent.get(), "_functions", module.release());
  return true;
}
