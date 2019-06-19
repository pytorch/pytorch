#include <torch/csrc/python_headers.h>
#include <sys/types.h>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include <unordered_map>
#include <cstdlib>
#include <libshm.h>
#include <TH/TH.h>
#include <c10/util/Logging.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DataLoader.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/TypeInfo.h>
#include <torch/csrc/autograd/generated/python_nn_functions.h>
#include <torch/csrc/autograd/python_legacy_variable.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/multiprocessing/init.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/csrc/utils/tensor_qschemes.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/jit/init.h>
#include <torch/csrc/jit/python_ir.h>
#include <torch/csrc/onnx/init.h>
#include <torch/csrc/api/include/torch/python/init.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#ifdef USE_DISTRIBUTED
#ifdef USE_C10D
#include <torch/csrc/distributed/c10d/c10d.h>
#endif
#endif

#define WITH_NUMPY_IMPORT_ARRAY
#include <torch/csrc/utils/numpy_stub.h>

namespace py = pybind11;

PyObject* module;

THPGenerator *THPDefaultCPUGenerator = nullptr;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static PyObject * THPModule_initNames(PyObject *self, PyObject *arg)
{
  static std::vector<std::string> names;

  THPObjectPtr types(PySequence_Fast(arg, "expected a sequence"));
  if (!types) return nullptr;

  int num_classes = PySequence_Fast_GET_SIZE(types.get());
  names.reserve(names.size() + num_classes);
  for (int i = 0; i < num_classes; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(types.get(), i);
    THPUtils_assert(PyType_Check(obj), "expected a PyTypeObject");
    PyTypeObject* type = (PyTypeObject*)obj;

    THPObjectPtr module_name(PyObject_GetAttrString(obj, "__module__"));
    if (!module_name) return nullptr;
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
    return nullptr;
  }
  torch::utils::initializeLayouts();
  torch::utils::initializeMemoryFormats();
  torch::utils::initializeQSchemes();
  torch::utils::initializeDtypes();
  torch::tensors::initialize_python_bindings();
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
  THPBoolStorage_postInit(module);
  THPQUInt8Storage_postInit(module);
  THPQInt8Storage_postInit(module);
  THPQInt32Storage_postInit(module);
  THPAutograd_initFunctions();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// The idea behind these two functions is to make it easy to test if we are
// built with ASAN: they're designed not to crash if ASAN is not enabled, but
// to trigger ASAN if it is enabled.  This lets us run a "canary" tests which
// checks if our build environment is misconfigured.

static PyObject * THPModule_crashIfCsrcASAN(PyObject *module, PyObject *arg) {
  THPUtils_assert(THPUtils_checkLong(arg), "crash_if_csrc_asan expects an int, "
          "but got %s", THPUtils_typename(arg));
  volatile char x[3];
  x[static_cast<int>(THPUtils_unpackLong(arg))] = 0;
  return PyLong_FromLong(x[0]);
}

static PyObject * THPModule_crashIfCsrcUBSAN(PyObject *module, PyObject *arg) {
  THPUtils_assert(THPUtils_checkLong(arg), "crash_if_csrc_ubsan expects an int, "
          "but got %s", THPUtils_typename(arg));
  int32_t x = static_cast<int>(THPUtils_unpackLong(arg));
  double y = 1.0 / x;
  return PyLong_FromLong((int)y);
}

static PyObject * THPModule_crashIfATenASAN(PyObject *module, PyObject *arg) {
  THPUtils_assert(THPUtils_checkLong(arg), "crash_if_aten_asan expects an int, "
          "but got %s", THPUtils_typename(arg));
  return PyLong_FromLong(at::_crash_if_asan(static_cast<int>(THPUtils_unpackLong(arg))));
}

static PyObject * THPModule_getNumThreads(PyObject *module)
{
  return PyLong_FromLong(at::get_num_threads());
}

static PyObject * THPModule_setNumThreads(PyObject *module, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_num_threads expects an int, "
          "but got %s", THPUtils_typename(arg));
  int nthreads = (int)THPUtils_unpackLong(arg);
  THPUtils_assert(nthreads > 0, "set_num_threads expects a positive integer");
  at::set_num_threads(nthreads);
  Py_RETURN_NONE;
}

static PyObject * THPModule_getNumInteropThreads(PyObject *module)
{
  return PyLong_FromLong(at::get_num_interop_threads());
}

static PyObject * THPModule_setNumInteropThreads(PyObject *module, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_num_interop_threads expects an int, "
          "but got %s", THPUtils_typename(arg));
  int nthreads = (int)THPUtils_unpackLong(arg);
  THPUtils_assert(nthreads > 0, "set_num_interop_threads expects a positive integer");
  at::set_num_interop_threads(nthreads);
  Py_RETURN_NONE;
}

PyObject * THPModule_setDefaultTensorType(PyObject *_unused, PyObject *type)
{
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_tensor_type(type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_setDefaultDtype(PyObject *_unused, PyObject *dtype)
{
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_dtype(dtype);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_safeCall(PyObject *_unused, PyObject *args, PyObject *kwargs)
{
  PyObject *result = nullptr;
  PyObject *args_slice = nullptr;
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
    return nullptr;
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
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    PyGetSetDescrObject* m = (PyGetSetDescrObject *)obj;
    if (m->d_getset->doc) {
      //NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      return PyErr_Format(PyExc_RuntimeError,
          "attribute '%s' already has a docstring", m->d_getset->name);
    }
    // This field is not const for python < 3.7 yet the content is
    // never modified.
    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    m->d_getset->doc = const_cast<char *>(doc_str);
  } else if (Py_TYPE(obj) == &PyType_Type) {
    PyTypeObject* t = (PyTypeObject *)obj;
    if (t->tp_doc) {
      return PyErr_Format(PyExc_RuntimeError,
          "Type '%s' already has a docstring", t->tp_name);
    }
    t->tp_doc = doc_str;
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
  return THPSize_NewFromSizes(sizes.size(), sizes.data());
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
#ifdef USE_DISTRIBUTED
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyObject *THPModule_showConfig(PyObject *module)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::show_config());
  END_HANDLE_TH_ERRORS
}

static PyObject *THPModule_parallelInfo(PyObject *module)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_parallel_info());
  END_HANDLE_TH_ERRORS
}

void DLPack_Capsule_Destructor(PyObject* data) {
  HANDLE_TH_ERRORS
  DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
  if (dlMTensor) {
    // the dlMTensor has not been consumed, call deleter ourselves
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
  } else {
    // the dlMTensor has been consumed
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
  END_HANDLE_TH_ERRORS_RET()
}

PyObject *THPModule_toDLPack(PyObject *_unused, PyObject *data)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPVariable_Check(data), "data must be a Tensor");
  DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(data));
  return PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
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
  auto scalar_type = at::typeMetaToScalarType(torch::tensors::get_default_tensor_options().dtype());
  auto dtype = (PyObject*)torch::getDtype(scalar_type);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_isDefaultTypeCuda(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (torch::tensors::get_default_tensor_options().device().is_cuda()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef TorchMethods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,   METH_O,       nullptr},
  {"_autograd_init",  (PyCFunction)THPAutograd_initExtension, METH_NOARGS,  nullptr},
  {"_add_docstr",     (PyCFunction)THPModule_addDocStr,       METH_VARARGS, nullptr},
  {"_init_names",     (PyCFunction)THPModule_initNames,       METH_O,       nullptr},
  {"_has_distributed",(PyCFunction)THPModule_hasDistributed,  METH_NOARGS,  nullptr},
  {"_safe_call",      (PyCFunction)THPModule_safeCall,          METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_set_default_tensor_type", (PyCFunction)THPModule_setDefaultTensorType, METH_O, nullptr},
  {"_set_default_dtype", (PyCFunction)THPModule_setDefaultDtype, METH_O, nullptr},
  {"_infer_size",     (PyCFunction)THPModule_inferSize,         METH_VARARGS, nullptr},
  {"_crash_if_csrc_asan", (PyCFunction)THPModule_crashIfCsrcASAN, METH_O, nullptr},
  {"_crash_if_csrc_ubsan", (PyCFunction)THPModule_crashIfCsrcUBSAN, METH_O, nullptr},
  {"_crash_if_aten_asan", (PyCFunction)THPModule_crashIfATenASAN, METH_O, nullptr},
  {"_show_config",    (PyCFunction)THPModule_showConfig, METH_NOARGS, nullptr},
  {"_parallel_info",    (PyCFunction)THPModule_parallelInfo, METH_NOARGS, nullptr},
  {"_set_backcompat_broadcast_warn", (PyCFunction)THPModule_setBackcompatBroadcastWarn, METH_O, nullptr},
  {"_get_backcompat_broadcast_warn", (PyCFunction)THPModule_getBackcompatBroadcastWarn, METH_NOARGS, nullptr},
  {"_set_backcompat_keepdim_warn", (PyCFunction)THPModule_setBackcompatKeepdimWarn, METH_O, nullptr},
  {"_get_backcompat_keepdim_warn", (PyCFunction)THPModule_getBackcompatKeepdimWarn, METH_NOARGS, nullptr},
  {"get_num_threads", (PyCFunction)THPModule_getNumThreads,     METH_NOARGS,  nullptr},
  {"set_num_threads", (PyCFunction)THPModule_setNumThreads,     METH_O,       nullptr},
  {"get_num_interop_threads", (PyCFunction)THPModule_getNumInteropThreads,     METH_NOARGS,  nullptr},
  {"set_num_interop_threads", (PyCFunction)THPModule_setNumInteropThreads,     METH_O,       nullptr},
  {"_get_cudnn_enabled", (PyCFunction)THPModule_userEnabledCuDNN, METH_NOARGS,     nullptr},
  {"_set_cudnn_enabled", (PyCFunction)THPModule_setUserEnabledCuDNN, METH_O,  nullptr},
  {"_get_cudnn_benchmark", (PyCFunction)THPModule_benchmarkCuDNN, METH_NOARGS,     nullptr},
  {"_set_cudnn_benchmark", (PyCFunction)THPModule_setBenchmarkCuDNN, METH_O,  nullptr},
  {"_get_cudnn_deterministic", (PyCFunction)THPModule_deterministicCuDNN, METH_NOARGS,     nullptr},
  {"_set_cudnn_deterministic", (PyCFunction)THPModule_setDeterministicCuDNN, METH_O,  nullptr},
  {"_to_dlpack",      (PyCFunction)THPModule_toDLPack,          METH_O,       nullptr},
  {"_from_dlpack",    (PyCFunction)THPModule_fromDLPack,        METH_O,       nullptr},
  {"set_flush_denormal", (PyCFunction)THPModule_setFlushDenormal, METH_O,     nullptr},
  {"get_default_dtype", (PyCFunction)THPModule_getDefaultDtype, METH_NOARGS,  nullptr},
  {"_is_default_type_cuda", (PyCFunction)THPModule_isDefaultTypeCuda, METH_NOARGS,  nullptr},
  {nullptr, nullptr, 0, nullptr}
};

bool THCPDoubleStorage_init(PyObject *module);
bool THCPFloatStorage_init(PyObject *module);
bool THCPHalfStorage_init(PyObject *module);
bool THCPLongStorage_init(PyObject *module);
bool THCPIntStorage_init(PyObject *module);
bool THCPShortStorage_init(PyObject *module);
bool THCPCharStorage_init(PyObject *module);
bool THCPByteStorage_init(PyObject *module);
bool THCPBoolStorage_init(PyObject *module);

void THCPStream_init(PyObject *module);
void THCPEvent_init(PyObject *module);

#ifdef USE_CUDA
PyMethodDef* THCPModule_methods();
namespace torch { namespace cuda {

void initModule(PyObject *module);

}} // namespace torch::cuda
#endif

namespace torch { namespace nn {

void init__THNN(PyObject*);
#ifdef USE_CUDA
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
bool THDPBoolStorage_init(PyObject *module);

static std::vector<PyMethodDef> methods;

#ifdef USE_DISTRIBUTED
PyMethodDef* THDPModule_methods();
#endif

// TODO: Refactor this in some less manual way
#ifdef USE_CUDNN
static PyObject * THCUDNN_cudnn_version(PyObject *self, PyObject *args)
{
  return PyLong_FromLong(CUDNN_VERSION);
}

static PyMethodDef _THCUDNN_methods[] = {
  {"_cudnn_version", (PyCFunction)THCUDNN_cudnn_version, METH_VARARGS, nullptr},
  {nullptr}
};

PyMethodDef* THCUDNN_methods() {
  return _THCUDNN_methods;
}
#endif

// ATen warning handler for Python
static void warning_handler(
    const c10::SourceLocation& source_location,
    const char* msg) {
  AutoGIL gil;
  auto result = -1;
  if (source_location.file == nullptr) {
    result = PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1);
  } else {
    result = PyErr_WarnExplicit(
        /*category=*/PyExc_UserWarning,
        /*message=*/msg,
        /*filename=*/source_location.file,
        /*lineno=*/source_location.line,
        /*module=*/nullptr,
        /*registry=*/nullptr);
  }
  if (result < 0) {
    throw python_error();
  }
}

// In Python we can't use the trick of C10_LOG_API_USAGE_ONCE
// Guaranteed to be invoked from Python under GIL, no locking on map needed
static void LogAPIUsageOnceFromPython(const std::string& event) {
  static std::unordered_set<std::string> seen;
  if (!seen.count(event)) {
    seen.insert(event);
    c10::LogAPIUsage(event);
  }
}


#ifdef _WIN32
__declspec(dllexport)
#endif
PyObject* initModule() {
  HANDLE_TH_ERRORS
  at::init_num_threads();

  C10_LOG_API_USAGE_ONCE("torch.python.import");

#define ASSERT_TRUE(cmd) if (!(cmd)) return nullptr

  THPUtils_addPyMethodDefs(methods, TorchMethods);
  THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
  THPUtils_addPyMethodDefs(methods, torch::autograd::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::multiprocessing::python_functions());
#ifdef USE_CUDA
  THPUtils_addPyMethodDefs(methods, THCPModule_methods());
#endif
#ifdef USE_CUDNN
  THPUtils_addPyMethodDefs(methods, THCUDNN_methods());
#endif
#ifdef USE_DISTRIBUTED
  THPUtils_addPyMethodDefs(methods, THDPModule_methods());
#ifdef USE_C10D
  THPUtils_addPyMethodDefs(methods, torch::distributed::c10d::python_functions());
#endif
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", methods.data()));
#else
  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     nullptr,
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
  THPDTypeInfo_init(module);
  THPLayout_init(module);
  THPMemoryFormat_init(module);
  THPQScheme_init(module);
  THPDevice_init(module);
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
  ASSERT_TRUE(THPEngine_initModule(module));
  // NOTE: We need to be able to access OperatorExportTypes from ONNX for use in
  // the export side of JIT, so this ONNX init needs to appear before the JIT
  // init.
  torch::onnx::initONNXBindings(module);
  torch::jit::initJITBindings(module);
  torch::autograd::initNNFunctions(module);
  torch::autograd::init_legacy_variable(module);
  torch::python::init_bindings(module);
#ifdef USE_CUDA
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
  ASSERT_TRUE(THPBoolStorage_init(module));
  ASSERT_TRUE(THPQUInt8Storage_init(module));
  ASSERT_TRUE(THPQInt8Storage_init(module));
  ASSERT_TRUE(THPQInt32Storage_init(module));

#ifdef USE_CUDA
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
  ASSERT_TRUE(THCPBoolStorage_init(module));

  THCPStream_init(module);
  THCPEvent_init(module);
#endif

  auto set_module_attr = [&](const char* name, PyObject* v, bool incref = true) {
    // PyModule_AddObject steals reference
    if (incref) {
      Py_INCREF(v);
    }
    return PyModule_AddObject(module, name, v) == 0;
  };

#ifdef USE_CUDNN
  PyObject *has_cudnn = Py_True;
#else
  PyObject *has_cudnn = Py_False;
#endif
 ASSERT_TRUE(set_module_attr("has_cudnn", has_cudnn));

  // force ATen to initialize because it handles
  // setting up TH Errors so that they throw C++ exceptions
  at::init();

  auto py_module = py::reinterpret_borrow<py::module>(module);
  py_module.def("_demangle", &c10::demangle);
  py_module.def("_log_api_usage_once", &LogAPIUsageOnceFromPython);

  // Set ATen warnings to issue Python warnings
  ::c10::Warning::set_warning_handler(&warning_handler);

  ASSERT_TRUE(set_module_attr("has_openmp", at::hasOpenMP() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_mkl", at::hasMKL() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_lapack", at::hasLAPACK() ? Py_True : Py_False));

#ifdef USE_CUDA
  PyObject *has_cuda = Py_True;
#else
  PyObject *has_cuda = Py_False;
#endif
  ASSERT_TRUE(set_module_attr("has_cuda", has_cuda));

  ASSERT_TRUE(set_module_attr("has_mkldnn", at::hasMKLDNN() ? Py_True : Py_False));

#ifdef _GLIBCXX_USE_CXX11_ABI
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False));
#else
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", Py_False));
#endif

  auto defaultGenerator = at::detail::getDefaultCPUGenerator();
  THPDefaultCPUGenerator = (THPGenerator*)THPGenerator_initDefaultGenerator(defaultGenerator);
  // This reference is meant to be given away, so no need to incref here.
  ASSERT_TRUE(set_module_attr("default_generator", (PyObject*)THPDefaultCPUGenerator, /* incref= */ false));

#ifdef USE_NUMPY
  if (_import_array() < 0) return nullptr;
#endif

  torch::nn::init__THNN(module);
#ifdef USE_CUDA
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
