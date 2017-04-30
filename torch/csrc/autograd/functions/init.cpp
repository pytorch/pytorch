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

template<typename C, typename T>
static void addClass(PyObject* module, PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties=NULL, PyMethodDef* function_methods=NULL)
{
  createForwardFunctionPyTypeObject<T>(type, name, function_properties, function_methods);
  Py_INCREF(&type);
  PyModule_AddObject(module, name, (PyObject*)&type);
  registerCppFunction(typeid(C), &type);
}

template<typename T, typename M, typename P, M P::*ptr, typename V, PyObject* (*Convert)(V)>
PyObject* getTupleAttr(PyObject* obj, void* _unused)
{
  THPCppFunction* self = (THPCppFunction*)obj;
  auto& arr = std::static_pointer_cast<T>(self->cdata).get()->*ptr;
  auto num_elems = arr.size();
  THPObjectPtr py_tuple = PyTuple_New(num_elems);
  if (!py_tuple) return NULL;
  for (size_t i = 0; i < num_elems; ++i) {
    PyTuple_SET_ITEM(py_tuple.get(), i, Convert(arr[i]));
  }
  return py_tuple.release();
}

template<typename T, typename M, typename P, M P::*ptr, typename V, PyObject* (*Convert)(V)>
PyObject* getValueAttr(PyObject* obj, void* _unused)
{
  THPCppFunction* self = (THPCppFunction*)obj;
  auto& val = std::static_pointer_cast<T>(self->cdata).get()->*ptr;
  return Convert(val);
}

static struct PyGetSetDef conv_forward_properties[] = {
  {(char*)"stride", (getter)getTupleAttr<ConvForward,std::vector<int>,ConvParams,&ConvParams::stride,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"padding", (getter)getTupleAttr<ConvForward,std::vector<int>,ConvParams,&ConvParams::padding,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"dilation", (getter)getTupleAttr<ConvForward,std::vector<int>,ConvParams,&ConvParams::dilation,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"transposed", (getter)getValueAttr<ConvForward,bool,ConvParams,&ConvParams::transposed,long,PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"output_padding", (getter)getTupleAttr<ConvForward,std::vector<int>,ConvParams,&ConvParams::output_padding,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"groups", (getter)getValueAttr<ConvForward,int,ConvParams,&ConvParams::groups,long,PyLong_FromLong>, NULL, NULL, NULL},
  {(char*)"next_functions", (getter)next_functions, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef conv_backward_properties[] = {
  {(char*)"stride", (getter)getTupleAttr<ConvBackward,std::vector<int>,ConvParams,&ConvParams::stride,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"padding", (getter)getTupleAttr<ConvBackward,std::vector<int>,ConvParams,&ConvParams::padding,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"dilation", (getter)getTupleAttr<ConvBackward,std::vector<int>,ConvParams,&ConvParams::dilation,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"transposed", (getter)getValueAttr<ConvBackward,bool,ConvParams,&ConvParams::transposed,long,PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"output_padding", (getter)getTupleAttr<ConvBackward,std::vector<int>,ConvParams,&ConvParams::output_padding,long,PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"groups", (getter)getValueAttr<ConvBackward,int,ConvParams,&ConvParams::groups,long,PyLong_FromLong>, NULL, NULL, NULL},
  {(char*)"next_functions", (getter)next_functions, NULL, NULL, NULL},
  {NULL}
};

bool THPAutograd_initFunctions(PyObject* _unused)
{
  THPObjectPtr module = PyModule_New("torch._C._functions");
  if (!module) return false;

  static PyTypeObject BatchNormClass, BatchNormBackwardClass;
  addClass<BatchNormForward, BatchNormCtor>(module, BatchNormClass, "BatchNorm");
  addClass<BatchNormBackward, NoCtor>(module, BatchNormBackwardClass, "BatchNormBackward");

  static PyTypeObject ConvClass, ConvBackwardClass;
  addClass<ConvForward, ConvCtor>(module, ConvClass, "ConvNd", conv_forward_properties);
  addClass<ConvBackward, NoCtor>(module, ConvBackwardClass, "ConvNdBackward", conv_backward_properties);

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
