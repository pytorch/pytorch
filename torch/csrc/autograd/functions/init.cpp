#include <Python.h>
#include "batch_normalization.h"
#include "convolution.h"
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
  THPObjectPtr module = PyModule_New("torch._C._functions");
  if (!module) return false;


  static PyTypeObject BatchNormClass, BatchNormBackwardClass;
  addClass<BatchNormForward, BatchNormCtor>(module, BatchNormClass, "BatchNorm");
  addClass<BatchNormBackward, NoCtor>(module, BatchNormBackwardClass, "BatchNormBackward");

  static PyTypeObject ConvClass, ConvBackwardClass;
  addClass<ConvForward, ConvCtor>(module, ConvClass, "ConvNd");
  addClass<ConvBackward, NoCtor>(module, ConvBackwardClass, "ConvNdBackward");

  THPObjectPtr parent = PyImport_ImportModule("torch._C");
  if (!parent) return false;
  PyModule_AddObject(parent.get(), "_functions", module.release());
  return true;
}
