#include <Python.h>
#include <sys/types.h>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include <stdbool.h>
#include <unordered_map>
#include <libshm.h>
#include <TH/TH.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Device.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DataLoader.h"
#include "torch/csrc/Generator.h"
#include "torch/csrc/Layout.h"
#include "torch/csrc/autograd/generated/python_nn_functions.h"
#include "torch/csrc/autograd/python_legacy_variable.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/tensor/python_tensor.h"
#include "torch/csrc/utils/tensor_dtypes.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_layouts.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/init.h"
#include "torch/csrc/jit/python_ir.h"

#ifdef WITH_CUDNN
#include "cudnn.h"
#endif

#define WITH_NUMPY_IMPORT_ARRAY
#include "torch/csrc/utils/numpy_stub.h"

namespace py = pybind11;

PyObject* module;

THPGenerator *THPDefaultGenerator   = NULL;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static PyObject * THPModule_initNames(PyObject *self, PyObject *arg)
{
  static std::vector<std::string> names;

  THPObjectPtr types(PySequence_Fast(arg, "expected a sequence"));
  if (!types) return NULL;

  int num_classes = PySequence_Fast_GET_SIZE(types.get());
  names.reserve(names.size() + num_classes);
  for (int i = 0; i < num_classes; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(types.get(), i);
    THPUtils_assert(PyType_Check(obj), "expected a PyTypeObject");
    PyTypeObject* type = (PyTypeObject*)obj;

    THPObjectPtr module_name(PyObject_GetAttrString(obj, "__module__"));
    if (!module_name) return NULL;
    THPUtils_assert(THPUtils_checkString(module_name.get()),
        "expected __module__ to be a string");
    std::string name = THPUtils_unpackString(module_name.get());
    names.push_back(name + "." + type->tp_name);
    type->tp_name = names.back().c_str();
  }
  Py_RETURN_NONE;
}
//
// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *_unused, PyObject *shm_manager_path)
{
  HANDLE_TH_ERRORS
  if (!THPUtils_checkString(shm_manager_path)) {
    THPUtils_setError("initialization error - expected bytes/string object as shm_manager_path!");
    return NULL;
  }
  torch::utils::initializeLayouts();
  torch::utils::initializeDtypes();
  torch::tensor::initialize_python_bindings();
  std::string path = THPUtils_unpackString(shm_manager_path);
  libshm_init(path.c_str());

  auto module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!module) throw python_error();

  THPDoubleStorage_postInit(module);
  THPFloatStorage_postInit(module);
  THPHalfStorage_postInit(module);
  THPLongStorage_postInit(module);
  THPIntStorage_postInit(module);
  THPShortStorage_postInit(module);
  THPCharStorage_postInit(module);
  THPByteStorage_postInit(module);
  THPAutograd_initFunctions();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPModule_getNumThreads(PyObject *module)
{
  return PyLong_FromLong(THGetNumThreads());
}

static PyObject * THPModule_setNumThreads(PyObject *module, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_num_threads expects an int, "
          "but got %s", THPUtils_typename(arg));
  THSetNumThreads((int)THPUtils_unpackLong(arg));
  at::set_num_threads((int)THPUtils_unpackLong(arg));
  Py_RETURN_NONE;
}

PyObject * THPModule_setDefaultTensorType(PyObject *_unused, PyObject *type)
{
  HANDLE_TH_ERRORS
  torch::tensor::py_set_default_tensor_type(type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_safeCall(PyObject *_unused, PyObject *args, PyObject *kwargs)
{
  PyObject *result = NULL;
  PyObject *args_slice = NULL;
  PyThreadState *thread_state = PyThreadState_Get();
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;
  THPUtils_assert(num_args > 0, "expected at least one argument");
  try {
    args_slice = PyTuple_GetSlice(args, 1, num_args);
    result = PyObject_Call(PyTuple_GET_ITEM(args, 0), args_slice, kwargs);
  } catch (std::exception &e) {
    PyEval_RestoreThread(thread_state);
    Py_DECREF(args_slice);
    PyErr_SetString(THPException_FatalError, e.what());
    Py_LeaveRecursiveCall();
  }
  Py_DECREF(args_slice);
  return result;
}

PyObject *THPModule_addDocStr(PyObject *_unused, PyObject *args)
{
  // adds a __doc__ string to a function, similar to numpy's arr_add_docstring
  static std::vector<std::string> all_docs;
  PyObject *obj;
  PyObject *doc_obj;
  if (!PyArg_ParseTuple(args, "OO", &obj, &doc_obj)) {
    return NULL;
  }

  const char* doc_str = "<invalid string>";
  if (THPUtils_checkString(doc_obj)) {
    all_docs.push_back(THPUtils_unpackString(doc_obj));
    doc_str = all_docs.back().c_str();
  }

  if (Py_TYPE(obj) == &PyCFunction_Type) {
    PyCFunctionObject* f = (PyCFunctionObject *)obj;
    if (f->m_ml->ml_doc) {
      return PyErr_Format(PyExc_RuntimeError,
          "function '%s' already has a docstring", f->m_ml->ml_name);
    }
    f->m_ml->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m = (PyMethodDescrObject *)obj;
    if (m->d_method->ml_doc) {
      return PyErr_Format(PyExc_RuntimeError,
          "method '%s' already has a docstring", m->d_method->ml_name);
    }
    m->d_method->ml_doc = doc_str;
  } else {
    return PyErr_Format(PyExc_TypeError,
        "don't know how to add docstring to type '%s'", Py_TYPE(obj)->tp_name);
  }

  Py_INCREF(obj);
  return obj;
}


PyObject *THPModule_inferSize(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? (Py_ssize_t) PyTuple_Size(args) : 0;
  THPUtils_assert(num_args == 2, "expected exactly 2 arguments");
  PyObject *arg1 = PyTuple_GET_ITEM(args, 0);
  THPUtils_assert(THPSize_Check(arg1), "expected a torch.Size as argument 1");
  PyObject *arg2 = PyTuple_GET_ITEM(args, 1);
  THPUtils_assert(THPSize_Check(arg2), "expected a torch.Size as argument 2");

  auto size1 = THPUtils_unpackLongs(arg1);
  auto size2 = THPUtils_unpackLongs(arg2);
  auto sizes = at::infer_size(size1, size2);
  return THPSize_New(sizes.size(), sizes.data());
  END_HANDLE_TH_ERRORS
}

static PyObject *THPModule_setBackcompatBroadcastWarn(PyObject *module, PyObject *arg) {
  THPUtils_assert(PyBool_Check(arg), "set_backcompat_broadcast_warn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  setBackCompatBroadcastWarn(arg == Py_True);
  Py_RETURN_NONE;
}

static PyObject *THPModule_getBackcompatBroadcastWarn(PyObject *module)
{
  if (getBackCompatBroadcastWarn()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

static PyObject *THPModule_setBackcompatKeepdimWarn(PyObject *module, PyObject *arg) {
  THPUtils_assert(PyBool_Check(arg), "set_backcompat_keepdim_warn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  setBackCompatKeepdimWarn(arg == Py_True);
  Py_RETURN_NONE;
}

static PyObject *THPModule_getBackcompatKeepdimWarn(PyObject *module)
{
  if (getBackCompatKeepdimWarn()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_hasDistributed(PyObject *_unused)
{
#ifdef WITH_DISTRIBUTED
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

PyObject *THPModule_toDLPack(PyObject *_unused, PyObject *data)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPVariable_Check(data), "data must be a Tensor");
  DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_UnpackData(data));
  return PyCapsule_New(dlMTensor, "dltensor", NULL);
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_fromDLPack(PyObject *_unused, PyObject *data)
{
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
  THPUtils_assert(dlMTensor, "from_dlpack received an invalid capsule. "
    "Note that DLTensor capsules can be consumed only once, "
    "so you might have already constructed a tensor from it once.")
  // atensor steals the ownership of the underlying storage. It also passes a
  // destructor function that will be called when the underlying storage goes
  // out of scope. When the destructor is called, the dlMTensor is destructed too.
  auto atensor = make_variable(at::fromDLPack(dlMTensor), false);

  // It is possible that the call to at::fromDLPack is the very first
  // call to create a Tensor in PyTorch. If so, then _lazy_init has
  // not been called, and the attempt to call createPyObject will fail
  // because cuda ATen types have not been registered in Python yet.
  // so if we have a cuda tensor, then we need to make sure
  // we have called _lazy_init here
  if(atensor.is_cuda()) {
    py::module::import("torch.cuda").attr("init")();
  }
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(data, "used_dltensor");
  return THPVariable_Wrap(std::move(atensor));
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_setUserEnabledCuDNN(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_enabled_cudnn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setUserEnabledCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_userEnabledCuDNN(PyObject *_unused)
{
  if (at::globalContext().userEnabledCuDNN()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setDeterministicCuDNN(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_deterministic_cudnn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setDeterministicCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_deterministicCuDNN(PyObject *_unused)
{
  if (at::globalContext().deterministicCuDNN()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setBenchmarkCuDNN(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_benchmark_cudnn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setBenchmarkCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_benchmarkCuDNN(PyObject *_unused)
{
  if (at::globalContext().benchmarkCuDNN()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setFlushDenormal(PyObject *_unused, PyObject *arg) {
  THPUtils_assert(PyBool_Check(arg), "flush_denormal expects a bool, "
          "but got %s", THPUtils_typename(arg));
  if (!at::globalContext().setFlushDenormal(arg == Py_True)) {
    Py_RETURN_FALSE;
  };
  Py_RETURN_TRUE;
}

PyObject *THPModule_getDefaultDtype(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  auto& type = torch::tensor::get_default_tensor_type();
  auto dtype = (PyObject*)torch::getDtype(type.scalarType());
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef TorchMethods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,   METH_O,       NULL},
  {"_autograd_init",  (PyCFunction)THPAutograd_initExtension, METH_NOARGS,  NULL},
  {"_add_docstr",     (PyCFunction)THPModule_addDocStr,       METH_VARARGS, NULL},
  {"_init_names",     (PyCFunction)THPModule_initNames,       METH_O,       NULL},
  {"_has_distributed",(PyCFunction)THPModule_hasDistributed,  METH_NOARGS,  NULL},
  {"_safe_call",      (PyCFunction)THPModule_safeCall,          METH_VARARGS | METH_KEYWORDS, NULL},
  {"_set_default_tensor_type", (PyCFunction)THPModule_setDefaultTensorType, METH_O, NULL},
  {"_infer_size",     (PyCFunction)THPModule_inferSize,         METH_VARARGS, NULL},
  {"_set_backcompat_broadcast_warn", (PyCFunction)THPModule_setBackcompatBroadcastWarn, METH_O, NULL},
  {"_get_backcompat_broadcast_warn", (PyCFunction)THPModule_getBackcompatBroadcastWarn, METH_NOARGS, NULL},
  {"_set_backcompat_keepdim_warn", (PyCFunction)THPModule_setBackcompatKeepdimWarn, METH_O, NULL},
  {"_get_backcompat_keepdim_warn", (PyCFunction)THPModule_getBackcompatKeepdimWarn, METH_NOARGS, NULL},
  {"get_num_threads", (PyCFunction)THPModule_getNumThreads,     METH_NOARGS,  NULL},
  {"set_num_threads", (PyCFunction)THPModule_setNumThreads,     METH_O,       NULL},
  {"_get_cudnn_enabled", (PyCFunction)THPModule_userEnabledCuDNN, METH_NOARGS,     NULL},
  {"_set_cudnn_enabled", (PyCFunction)THPModule_setUserEnabledCuDNN, METH_O,  NULL},
  {"_get_cudnn_benchmark", (PyCFunction)THPModule_benchmarkCuDNN, METH_NOARGS,     NULL},
  {"_set_cudnn_benchmark", (PyCFunction)THPModule_setBenchmarkCuDNN, METH_O,  NULL},
  {"_get_cudnn_deterministic", (PyCFunction)THPModule_deterministicCuDNN, METH_NOARGS,     NULL},
  {"_set_cudnn_deterministic", (PyCFunction)THPModule_setDeterministicCuDNN, METH_O,  NULL},
  {"_to_dlpack",      (PyCFunction)THPModule_toDLPack,          METH_O,       NULL},
  {"_from_dlpack",    (PyCFunction)THPModule_fromDLPack,        METH_O,       NULL},
  {"set_flush_denormal", (PyCFunction)THPModule_setFlushDenormal, METH_O,     NULL},
  {"get_default_dtype", (PyCFunction)THPModule_getDefaultDtype, METH_NOARGS,  NULL},
  {NULL, NULL, 0, NULL}
};

bool THCPDoubleStorage_init(PyObject *module);
bool THCPFloatStorage_init(PyObject *module);
bool THCPHalfStorage_init(PyObject *module);
bool THCPLongStorage_init(PyObject *module);
bool THCPIntStorage_init(PyObject *module);
bool THCPShortStorage_init(PyObject *module);
bool THCPCharStorage_init(PyObject *module);
bool THCPByteStorage_init(PyObject *module);

bool THCPStream_init(PyObject *module);

#ifdef WITH_CUDA
PyMethodDef* THCPModule_methods();
namespace torch { namespace cuda {

void initModule(PyObject *module);

}} // namespace torch::cuda
#endif

namespace torch { namespace nn {

void init__THNN(PyObject*);
#ifdef WITH_CUDA
void init__THCUNN(PyObject*);
#endif

}} // namespace torch::nn

bool THDPDoubleStorage_init(PyObject *module);
bool THDPFloatStorage_init(PyObject *module);
//bool THDPHalfStorage_init(PyObject *module);
bool THDPLongStorage_init(PyObject *module);
bool THDPIntStorage_init(PyObject *module);
bool THDPShortStorage_init(PyObject *module);
bool THDPCharStorage_init(PyObject *module);
bool THDPByteStorage_init(PyObject *module);

static std::vector<PyMethodDef> methods;

#ifdef WITH_DISTRIBUTED
PyMethodDef* THDPModule_methods();
#endif

// TODO: Refactor this in some less manual way
#ifdef WITH_CUDNN
static PyObject * THCUDNN_cudnn_version(PyObject *self, PyObject *args)
{
  return PyLong_FromLong(CUDNN_VERSION);
}

static PyMethodDef _THCUDNN_methods[] = {
  {"_cudnn_version", (PyCFunction)THCUDNN_cudnn_version, METH_VARARGS, NULL},
  {NULL}
};

PyMethodDef* THCUDNN_methods() {
  return _THCUDNN_methods;
}
#endif

static PyObject* initModule() {
  HANDLE_TH_ERRORS
  THInferNumThreads();

#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL

  THPUtils_addPyMethodDefs(methods, TorchMethods);
  THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
  THPUtils_addPyMethodDefs(methods, torch::autograd::python_functions());
#ifdef WITH_CUDA
  THPUtils_addPyMethodDefs(methods, THCPModule_methods());
#endif
#ifdef WITH_CUDNN
  THPUtils_addPyMethodDefs(methods, THCUDNN_methods());
#endif
#ifdef WITH_DISTRIBUTED
  THPUtils_addPyMethodDefs(methods, THDPModule_methods());
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", methods.data()));
#else
  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     NULL,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
  ASSERT_TRUE(THPWrapper_init(module));
  ASSERT_TRUE(THPGenerator_init(module));
  ASSERT_TRUE(THPException_init(module));
  THPSize_init(module);
  THPDtype_init(module);
  THPLayout_init(module);
  THPDevice_init(module);
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
  ASSERT_TRUE(THPEngine_initModule(module));
  torch::autograd::initAutogradClosureBindings(module);
  torch::jit::initJITBindings(module);
  torch::autograd::initNNFunctions(module);
  torch::autograd::init_legacy_variable(module);
#ifdef WITH_CUDA
  torch::cuda::initModule(module);
#endif
  ASSERT_TRUE(THPDoubleStorage_init(module));
  ASSERT_TRUE(THPFloatStorage_init(module));
  ASSERT_TRUE(THPHalfStorage_init(module));
  ASSERT_TRUE(THPLongStorage_init(module));
  ASSERT_TRUE(THPIntStorage_init(module));
  ASSERT_TRUE(THPShortStorage_init(module));
  ASSERT_TRUE(THPCharStorage_init(module));
  ASSERT_TRUE(THPByteStorage_init(module));

#ifdef WITH_CUDA
  // This will only initialise base classes and attach them to library namespace
  // They won't be ready for real usage until importing cuda module, that will
  // complete the process (but it defines Python classes before calling back into
  // C, so these lines have to execute first)..
  ASSERT_TRUE(THCPDoubleStorage_init(module));
  ASSERT_TRUE(THCPFloatStorage_init(module));
  ASSERT_TRUE(THCPHalfStorage_init(module));
  ASSERT_TRUE(THCPLongStorage_init(module));
  ASSERT_TRUE(THCPIntStorage_init(module));
  ASSERT_TRUE(THCPShortStorage_init(module));
  ASSERT_TRUE(THCPCharStorage_init(module));
  ASSERT_TRUE(THCPByteStorage_init(module));

  ASSERT_TRUE(THCPStream_init(module));
#endif

#ifdef WITH_CUDNN
  PyObject *has_cudnn = Py_True;
#else
  PyObject *has_cudnn = Py_False;
#endif
  Py_INCREF(has_cudnn);
  ASSERT_TRUE(PyModule_AddObject(module, "has_cudnn", has_cudnn) == 0);

#ifdef WITH_DISTRIBUTED_MW
  // See comment on CUDA objects
  ASSERT_TRUE(THDPDoubleStorage_init(module));
  ASSERT_TRUE(THDPFloatStorage_init(module));
  //ASSERT_TRUE(THDPHalfStorage_init(module));
  ASSERT_TRUE(THDPLongStorage_init(module));
  ASSERT_TRUE(THDPIntStorage_init(module));
  ASSERT_TRUE(THDPShortStorage_init(module));
  ASSERT_TRUE(THDPCharStorage_init(module));
  ASSERT_TRUE(THDPByteStorage_init(module));
#endif

  // force ATen to initialize because it handles
  // setting up TH Errors so that they throw C++ exceptions
  at::init();

  ASSERT_TRUE(PyModule_AddObject(module, "has_mkl", at::hasMKL() ? Py_True : Py_False) == 0);

  auto& defaultGenerator = at::globalContext().defaultGenerator(at::kCPU);
  THPDefaultGenerator = (THPGenerator*)THPGenerator_NewWithGenerator(
    defaultGenerator);
  ASSERT_TRUE(PyModule_AddObject(module, "default_generator", (PyObject*)THPDefaultGenerator) == 0);

#ifdef WITH_NUMPY
  if (_import_array() < 0) return NULL;
#endif

  torch::nn::init__THNN(module);
#ifdef WITH_CUDA
  torch::nn::init__THCUNN(module);
#endif

  return module;
  END_HANDLE_TH_ERRORS
}

// Checks that the _C shared library isn't initialized multiple times. This
// can happen if the same csrc files are compiled into multiple shared
// libraries.
inline void pytorch_duplicate_guard() {
  static int initialized = 0;
  if (initialized) {
    fprintf(stderr, "pytorch: _C shared library re-initialized\n");
    abort();
  }
  initialized = 1;
;}

struct call_duplicate_guard {
  call_duplicate_guard() { pytorch_duplicate_guard(); }
};

static call_duplicate_guard _call_duplicate_guard;

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_C()
#else
PyMODINIT_FUNC PyInit__C()
#endif
{
#if PY_MAJOR_VERSION == 2
  initModule();
#else
  return initModule();
#endif
}
