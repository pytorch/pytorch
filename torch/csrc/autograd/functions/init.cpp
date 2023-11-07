#include <Python.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/pybind.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/generated/python_functions.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_variable.h>
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#endif
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include <utility>

using namespace torch::autograd;

struct DelayedErrorCtor {
  DelayedError* operator()(PyObject* args) {
    TORCH_CHECK(
        PyTuple_GET_SIZE(args) == 2,
        "Requires two arguments, got ",
        PyTuple_GET_SIZE(args));
    auto arg1 = PyTuple_GET_ITEM(args, 0);
    TORCH_CHECK(THPUtils_checkString(arg1), "argument 'msg' must be a string");
    std::string msg = THPUtils_unpackString(arg1);
    auto arg2 = PyTuple_GET_ITEM(args, 1);
    TORCH_CHECK(
        THPUtils_checkLong(arg2), "argument 'num_inputs' must be an int");
    auto num_inputs = THPUtils_unpackLong(arg2);
    return new DelayedError(std::move(msg), num_inputs);
  }
};

struct UndefinedGradCtor {
  UndefinedGrad* operator()(PyObject* args) {
    TORCH_CHECK(
        PyTuple_GET_SIZE(args) == 0,
        "Requires zero arguments, got ",
        PyTuple_GET_SIZE(args));
    return new UndefinedGrad();
  }
};

struct NoCtor {
  Node* operator()(PyObject* args) {
    throw std::runtime_error("Cannot construct");
  }
};

template <typename C, typename T>
static void addClass(
    PyObject* module,
    PyTypeObject& type,
    const char* name,
    PyGetSetDef* function_properties = nullptr,
    PyMethodDef* function_methods = nullptr) {
  createForwardFunctionPyTypeObject<T>(
      type, name, function_properties, function_methods);
  Py_INCREF(&type);
  PyModule_AddObject(module, name, (PyObject*)&type);
  registerCppFunction(typeid(C), &type);
}

template <
    typename T,
    typename ValueT,
    typename ParamsT,
    ValueT ParamsT::*ptr,
    typename ConvertArgT,
    PyObject* (*Convert)(ConvertArgT)>
PyObject* getTupleAttr(PyObject* obj, void* _unused) {
  HANDLE_TH_ERRORS
  THPCppFunction* self = (THPCppFunction*)obj;
  auto& arr = ((T*)(self->cdata.get()))->*ptr;
  auto num_elems = arr.size();
  THPObjectPtr py_tuple(PyTuple_New(num_elems));
  if (!py_tuple)
    return nullptr;
  for (const auto i : c10::irange(num_elems)) {
    PyTuple_SET_ITEM(py_tuple.get(), i, Convert(arr[i]));
  }
  return py_tuple.release();
  END_HANDLE_TH_ERRORS
}

template <
    typename T,
    typename ValueT,
    typename ParamsT,
    ValueT ParamsT::*ptr,
    typename ConvertArgT,
    PyObject* (*Convert)(ConvertArgT)>
PyObject* getValueAttr(PyObject* obj, void* _unused) {
  HANDLE_TH_ERRORS
  THPCppFunction* self = (THPCppFunction*)obj;
  auto& val = ((T*)(self->cdata.get()))->*ptr;
  return Convert(val);
  END_HANDLE_TH_ERRORS
}

static PyObject* accumulateGradVar(PyObject* _self, void* _unused) {
  THPCppFunction* self = (THPCppFunction*)_self;
  auto grad_acc = (AccumulateGrad*)self->cdata.get();
  return THPVariable_Wrap(grad_acc->variable);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyGetSetDef accumulate_grad_properties[] = {
    THP_FUNCTION_DEFAULT_PROPERTIES,
    {(char*)"variable", accumulateGradVar, nullptr, nullptr, nullptr},
    {nullptr}};

void THPAutograd_initFunctions() {
  THPObjectPtr module(PyModule_New("torch._C._functions"));
  if (!module)
    throw python_error();

  static PyTypeObject AccumulateGradClass;
  addClass<AccumulateGrad, NoCtor>(
      module,
      AccumulateGradClass,
      "AccumulateGrad",
      accumulate_grad_properties);

  static PyTypeObject ErrorClass;
  addClass<Error, NoCtor>(module, ErrorClass, "Error");

  static PyTypeObject NotImplementedClass;
  addClass<NotImplemented, NoCtor>(
      module, NotImplementedClass, "NotImplemented");

  static PyTypeObject DelayedErrorClass;
  addClass<DelayedError, DelayedErrorCtor>(
      module, DelayedErrorClass, "DelayedError");

  static PyTypeObject UndefinedGradBackwardClass;
  addClass<UndefinedGradBackward, NoCtor>(
      module, UndefinedGradBackwardClass, "UndefinedGradBackward");

  static PyTypeObject UndefinedGradClass;
  addClass<UndefinedGrad, UndefinedGradCtor>(
      module, UndefinedGradClass, "UndefinedGrad");

  static PyTypeObject CopyBackwardsClass;
  addClass<CopyBackwards, NoCtor>(module, CopyBackwardsClass, "CopyBackwards");

#ifdef USE_DISTRIBUTED
  static PyTypeObject SendRpcBackwardClass;
  addClass<torch::distributed::autograd::SendRpcBackward, NoCtor>(
      module, SendRpcBackwardClass, "SendRpcBackward");
#endif

  static PyTypeObject CopySlicesClass;
  addClass<CopySlices, NoCtor>(module, CopySlicesClass, "CopySlices");

  generated::initialize_autogenerated_functions(module);

  auto c_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!c_module)
    throw python_error();

  Py_INCREF(module.get());
  if (PyModule_AddObject(c_module, "_functions", module) < 0) {
    Py_DECREF(module.get());
    throw python_error();
  }
}
