#include <Python.h>
#include "batch_normalization.h"
#include "convolution.h"
#include "accumulate_grad.h"
#include "basic_ops.h"
#include "tensor.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/utils/tuple_parser.h"
#include "torch/csrc/DynamicTypes.h"

using namespace torch::autograd;
using torch::TupleParser;

struct BatchNormCtor {
  BatchNormForward* operator()(PyObject* args) {
    BatchNormParams params;

    TupleParser parser(args, 6);
    parser.parse(params.running_mean, "running_mean");
    parser.parse(params.running_var, "running_var");
    parser.parse(params.training, "training");
    parser.parse(params.momentum, "momentum");
    parser.parse(params.eps, "eps");
    parser.parse(params.cudnn_enabled, "cudnn_enabled");

    return new BatchNormForward(std::move(params));
  }
};

struct ConvCtor {
  ConvForward* operator()(PyObject* args) {
    ConvParams params;

    TupleParser parser(args, 8);
    parser.parse(params.stride, "stride");
    parser.parse(params.padding, "padding");
    parser.parse(params.dilation, "dilation");
    parser.parse(params.transposed, "transposed");
    parser.parse(params.output_padding, "output_padding");
    parser.parse(params.groups, "groups");
    parser.parse(params.benchmark, "benchmark");
    parser.parse(params.cudnn_enabled, "cudnn_enabled");

    return new ConvForward(std::move(params));
  }
};

struct DelayedErrorCtor {
  DelayedError* operator()(PyObject* args) {
    std::string msg;

    TupleParser parser(args, 1);
    parser.parse(msg, "msg");

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

template<typename T, typename ValueT, typename ParamsT, ValueT ParamsT::*ptr,
         typename ConvertArgT, PyObject* (*Convert)(ConvertArgT)>
PyObject* getTupleAttr(PyObject* obj, void* _unused)
{
  HANDLE_TH_ERRORS
  THPCppFunction* self = (THPCppFunction*)obj;
  auto& arr = ((T*)(self->cdata.get()))->*ptr;
  auto num_elems = arr.size();
  THPObjectPtr py_tuple(PyTuple_New(num_elems));
  if (!py_tuple) return NULL;
  for (size_t i = 0; i < num_elems; ++i) {
    PyTuple_SET_ITEM(py_tuple.get(), i, Convert(arr[i]));
  }
  return py_tuple.release();
  END_HANDLE_TH_ERRORS
}

template<typename T, typename ValueT, typename ParamsT, ValueT ParamsT::*ptr,
         typename ConvertArgT, PyObject* (*Convert)(ConvertArgT)>
PyObject* getValueAttr(PyObject* obj, void* _unused)
{
  HANDLE_TH_ERRORS
  THPCppFunction* self = (THPCppFunction*)obj;
  auto& val = ((T*)(self->cdata.get()))->*ptr;
  return Convert(val);
  END_HANDLE_TH_ERRORS
}

template<typename T, typename ParamsT, at::Tensor ParamsT::*ptr>
PyObject* getTensorAttr(PyObject* obj, void* _unused)
{
  HANDLE_TH_ERRORS
  THPCppFunction* self = (THPCppFunction*)obj;
  auto& val = ((T*)(self->cdata.get()))->*ptr;
  THPObjectPtr py_tensor;
  if (!val.defined()) {
    Py_INCREF(Py_None);
    py_tensor = Py_None;
  } else {
    py_tensor = torch::createPyObject(val);
  }
  return py_tensor.release();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef batch_norm_forward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"running_mean", (getter)getTensorAttr<BatchNormForward, BatchNormParams,
                                         &BatchNormParams::running_mean>, NULL, NULL, NULL},
  {(char*)"running_var", (getter)getTensorAttr<BatchNormForward, BatchNormParams,
                                         &BatchNormParams::running_var>, NULL, NULL, NULL},
  {(char*)"training", (getter)getValueAttr<BatchNormForward, bool, BatchNormParams,
                                         &BatchNormParams::training, long, PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"momentum", (getter)getValueAttr<BatchNormForward, double, BatchNormParams,
                                         &BatchNormParams::momentum, double, PyFloat_FromDouble>, NULL, NULL, NULL},
  {(char*)"eps", (getter)getValueAttr<BatchNormForward, double, BatchNormParams,
                                         &BatchNormParams::eps, double, PyFloat_FromDouble>, NULL, NULL, NULL},
  {(char*)"cudnn_enabled", (getter)getValueAttr<BatchNormForward, bool, BatchNormParams,
                                         &BatchNormParams::cudnn_enabled, long, PyBool_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef batch_norm_backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"running_mean", (getter)getTensorAttr<BatchNormBackward, BatchNormParams,
                                         &BatchNormParams::running_mean>, NULL, NULL, NULL},
  {(char*)"running_var", (getter)getTensorAttr<BatchNormBackward, BatchNormParams,
                                         &BatchNormParams::running_var>, NULL, NULL, NULL},
  {(char*)"training", (getter)getValueAttr<BatchNormBackward, bool, BatchNormParams,
                                         &BatchNormParams::training, long, PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"momentum", (getter)getValueAttr<BatchNormBackward, double, BatchNormParams,
                                         &BatchNormParams::momentum, double, PyFloat_FromDouble>, NULL, NULL, NULL},
  {(char*)"eps", (getter)getValueAttr<BatchNormBackward, double, BatchNormParams,
                                         &BatchNormParams::eps, double, PyFloat_FromDouble>, NULL, NULL, NULL},
  {(char*)"cudnn_enabled", (getter)getValueAttr<BatchNormBackward, bool, BatchNormParams,
                                         &BatchNormParams::cudnn_enabled, long, PyBool_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef batch_norm_backward_backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"running_mean", (getter)getTensorAttr<BatchNormBackwardBackward, BatchNormParams,
                                         &BatchNormParams::running_mean>, NULL, NULL, NULL},
  {(char*)"running_var", (getter)getTensorAttr<BatchNormBackwardBackward, BatchNormParams,
                                         &BatchNormParams::running_var>, NULL, NULL, NULL},
  {(char*)"training", (getter)getValueAttr<BatchNormBackwardBackward, bool, BatchNormParams,
                                         &BatchNormParams::training, long, PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"momentum", (getter)getValueAttr<BatchNormBackwardBackward, double, BatchNormParams,
                                         &BatchNormParams::momentum, double, PyFloat_FromDouble>, NULL, NULL, NULL},
  {(char*)"eps", (getter)getValueAttr<BatchNormBackwardBackward, double, BatchNormParams,
                                         &BatchNormParams::eps, double, PyFloat_FromDouble>, NULL, NULL, NULL},
  {(char*)"cudnn_enabled", (getter)getValueAttr<BatchNormBackwardBackward, bool, BatchNormParams,
                                         &BatchNormParams::cudnn_enabled, long, PyBool_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef conv_forward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"stride", (getter)getTupleAttr<ConvForward, std::vector<int>, ConvParams,
                                         &ConvParams::stride, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"padding", (getter)getTupleAttr<ConvForward, std::vector<int>, ConvParams,
                                         &ConvParams::padding, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"dilation", (getter)getTupleAttr<ConvForward, std::vector<int>, ConvParams,
                                         &ConvParams::dilation, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"transposed", (getter)getValueAttr<ConvForward, bool, ConvParams,
                                         &ConvParams::transposed, long, PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"output_padding", (getter)getTupleAttr<ConvForward, std::vector<int>, ConvParams,
                                         &ConvParams::output_padding, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"groups", (getter)getValueAttr<ConvForward, int, ConvParams,
                                         &ConvParams::groups, long, PyInt_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef conv_backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"stride", (getter)getTupleAttr<ConvBackward, std::vector<int>, ConvParams,
                                         &ConvParams::stride, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"padding", (getter)getTupleAttr<ConvBackward, std::vector<int>, ConvParams,
                                         &ConvParams::padding, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"dilation", (getter)getTupleAttr<ConvBackward, std::vector<int>, ConvParams,
                                         &ConvParams::dilation, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"transposed", (getter)getValueAttr<ConvBackward, bool, ConvParams,
                                         &ConvParams::transposed, long, PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"output_padding", (getter)getTupleAttr<ConvBackward, std::vector<int>, ConvParams,
                                         &ConvParams::output_padding, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"groups", (getter)getValueAttr<ConvBackward, int, ConvParams,
                                         &ConvParams::groups, long, PyInt_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef conv_backward_backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"stride", (getter)getTupleAttr<ConvBackwardBackward, std::vector<int>, ConvParams,
                                         &ConvParams::stride, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"padding", (getter)getTupleAttr<ConvBackwardBackward, std::vector<int>, ConvParams,
                                         &ConvParams::padding, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"dilation", (getter)getTupleAttr<ConvBackwardBackward, std::vector<int>, ConvParams,
                                         &ConvParams::dilation, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"transposed", (getter)getValueAttr<ConvBackwardBackward, bool, ConvParams,
                                         &ConvParams::transposed, long, PyBool_FromLong>, NULL, NULL, NULL},
  {(char*)"output_padding", (getter)getTupleAttr<ConvBackwardBackward, std::vector<int>, ConvParams,
                                         &ConvParams::output_padding, long, PyInt_FromLong>, NULL, NULL, NULL},
  {(char*)"groups", (getter)getValueAttr<ConvBackwardBackward, int, ConvParams,
                                         &ConvParams::groups, long, PyInt_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static PyObject* accumulateGradVar(PyObject *_self, void* _unused)
{
  THPCppFunction* self = (THPCppFunction*)_self;
  auto grad_acc = (AccumulateGrad*)self->cdata.get();
  auto var = grad_acc->variable.lock();
  if (!var) Py_RETURN_NONE;
  return THPVariable_Wrap(var);
}

static struct PyGetSetDef accumulate_grad_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"variable", accumulateGradVar, NULL, NULL, NULL},
  {NULL}
};

bool THPAutograd_initFunctions(PyObject* _unused)
{
  THPObjectPtr module(PyModule_New("torch._C._functions"));
  if (!module) return false;

  static PyTypeObject BatchNormClass, BatchNormBackwardClass, BatchNormBackwardBackwardClass;
  addClass<BatchNormForward, BatchNormCtor>(module, BatchNormClass, "BatchNorm", batch_norm_forward_properties);
  addClass<BatchNormBackward, NoCtor>(module, BatchNormBackwardClass, "BatchNormBackward", batch_norm_backward_properties);
  addClass<BatchNormBackwardBackward, NoCtor>(module, BatchNormBackwardBackwardClass, "BatchNormBackwardBackward", batch_norm_backward_backward_properties);

  static PyTypeObject ConvClass, ConvBackwardClass, ConvBackwardBackwardClass;
  addClass<ConvForward, ConvCtor>(module, ConvClass, "ConvNd", conv_forward_properties);
  addClass<ConvBackward, NoCtor>(module, ConvBackwardClass, "ConvNdBackward", conv_backward_properties);
  addClass<ConvBackwardBackward, NoCtor>(module, ConvBackwardBackwardClass, "ConvNdBackwardBackward", conv_backward_backward_properties);

  static PyTypeObject AccumulateGradClass;
  addClass<AccumulateGrad, NoCtor>(module, AccumulateGradClass, "AccumulateGrad", accumulate_grad_properties);

  static PyTypeObject AddClass, AddBackwardClass;
  addClass<Add, NoCtor>(module, AddClass, "Add");
  addClass<AddBackward, NoCtor>(module, AddBackwardClass, "AddBackward");

  static PyTypeObject ErrorClass;
  addClass<Error, NoCtor>(module, ErrorClass, "Error");

  static PyTypeObject DelayedErrorClass;
  addClass<DelayedError, DelayedErrorCtor>(module, DelayedErrorClass, "DelayedError");

  static PyTypeObject CloneClass;
  addClass<Clone, NoCtor>(module, CloneClass, "Clone");
  static PyTypeObject ContiguousClass;
  addClass<Contiguous, NoCtor>(module, ContiguousClass, "Contiguous");
  static PyTypeObject IdentityClass;
  addClass<Identity, NoCtor>(module, IdentityClass, "Identity");
  static PyTypeObject TransposeClass;
  addClass<Transpose, NoCtor>(module, TransposeClass, "Transpose");
  static PyTypeObject ViewClass;
  addClass<View, NoCtor>(module, ViewClass, "View");
  static PyTypeObject ExpandClass;
  addClass<Expand, NoCtor>(module, ExpandClass, "Expand");
  static PyTypeObject NarrowClass;
  addClass<Narrow, NoCtor>(module, NarrowClass, "Narrow");
  static PyTypeObject CatClass;
  addClass<Cat, NoCtor>(module, CatClass, "Cat");

  THPObjectPtr parent(PyImport_ImportModule("torch._C"));
  if (!parent) return false;
  PyModule_AddObject(parent.get(), "_functions", module.release());
  return true;
}
