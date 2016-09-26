#include <Python.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <stdbool.h>
#include <unordered_map>
#include <libshm.h>
#include <TH/TH.h>

#define WITH_NUMPY_IMPORT_ARRAY
#include "THP.h"

PyObject* module;
PyObject* tensor_classes;

PyObject *THPDefaultTensorClass = NULL;
THPGenerator *THPDefaultGenerator   = NULL;

static bool THPModule_loadClasses(PyObject *self)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  PyObject *torch_module = PyImport_ImportModule("torch");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch module");
    return false;
  }
  PyObject* module_dict = PyModule_GetDict(torch_module);

  ASSERT_NOT_NULL(tensor_classes = PyMapping_GetItemString(module_dict, (char*)"_tensor_classes"));

  ASSERT_NOT_NULL(THPDoubleStorageClass = PyMapping_GetItemString(module_dict,(char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THPFloatStorageClass  = PyMapping_GetItemString(module_dict,(char*)"FloatStorage"));
  ASSERT_NOT_NULL(THPLongStorageClass   = PyMapping_GetItemString(module_dict,(char*)"LongStorage"));
  ASSERT_NOT_NULL(THPIntStorageClass    = PyMapping_GetItemString(module_dict,(char*)"IntStorage"));
  ASSERT_NOT_NULL(THPShortStorageClass  = PyMapping_GetItemString(module_dict,(char*)"ShortStorage"));
  ASSERT_NOT_NULL(THPCharStorageClass   = PyMapping_GetItemString(module_dict,(char*)"CharStorage"));
  ASSERT_NOT_NULL(THPByteStorageClass   = PyMapping_GetItemString(module_dict,(char*)"ByteStorage"));

  ASSERT_NOT_NULL(THPDoubleTensorClass  = PyMapping_GetItemString(module_dict,(char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THPFloatTensorClass   = PyMapping_GetItemString(module_dict,(char*)"FloatTensor"));
  ASSERT_NOT_NULL(THPLongTensorClass    = PyMapping_GetItemString(module_dict,(char*)"LongTensor"));
  ASSERT_NOT_NULL(THPIntTensorClass     = PyMapping_GetItemString(module_dict,(char*)"IntTensor"));
  ASSERT_NOT_NULL(THPShortTensorClass   = PyMapping_GetItemString(module_dict,(char*)"ShortTensor"));
  ASSERT_NOT_NULL(THPCharTensorClass    = PyMapping_GetItemString(module_dict,(char*)"CharTensor"));
  ASSERT_NOT_NULL(THPByteTensorClass    = PyMapping_GetItemString(module_dict,(char*)"ByteTensor"));

  return true;
#undef ASSERT_NOT_NULL
}

////////////////////////////////////////////////////////////////////////////////
// Copy handlers
////////////////////////////////////////////////////////////////////////////////

#include "ModuleCopy.h"

std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> tensor_copy_handlers;
std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> storage_copy_handlers;
std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> tensor_async_copy_handlers;
std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> storage_async_copy_handlers;

#define COPY_METHODS(name) TH_CONCAT_2(name,_copy_handlers)
#define IMPLEMENT_COPY_WITH_WRAPPER(name)                                      \
bool TH_CONCAT_3(THPModule_,name,Copy)(PyObject *dst, PyObject *src)           \
{                                                                              \
  auto it = COPY_METHODS(name).find(std::make_pair((PyObject*)Py_TYPE(dst), (PyObject*)Py_TYPE(src))); \
  if (it == COPY_METHODS(name).end()) {                                        \
    THPUtils_setError("Copy function from %s to %s isn't implemented!", Py_TYPE(src)->tp_name, Py_TYPE(dst)->tp_name); \
    return false;                                                              \
  }                                                                            \
  (it->second)(dst, src);                                                      \
  return true;                                                                 \
}                                                                              \
                                                                               \
static PyObject * TH_CONCAT_3(THPModule_,name,CopyWrapper)(PyObject *unused, PyObject *args)\
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;                         \
  THPUtils_assert(num_args == 2, #name "Copy expected exactly two arguments, " \
          "but got %ld", (long)num_args);                                      \
  PyObject *dst = PyTuple_GET_ITEM(args, 0);                                   \
  PyObject *src = PyTuple_GET_ITEM(args, 1);                                   \
  if (!TH_CONCAT_3(THPModule_,name,Copy)(dst, src)) {                          \
    return NULL;                                                               \
  }                                                                            \
  Py_INCREF(dst);                                                              \
  return dst;                                                                  \
  END_HANDLE_TH_ERRORS                                                         \
}

IMPLEMENT_COPY_WITH_WRAPPER(tensor)
IMPLEMENT_COPY_WITH_WRAPPER(storage)
IMPLEMENT_COPY_WITH_WRAPPER(tensor_async)
IMPLEMENT_COPY_WITH_WRAPPER(storage_async)
#undef COPY_METHODS
#undef IMPLEMENT_COPY_WITH_WRAPPER

#include "ModuleCopy.cpp"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static bool THPModule_assignStateless(PyObject *self)
{
#define INIT_STATELESS(type)                                                   \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_2(type, TensorStatelessType), arg, NULL); \
  if (!stateless) {                                                            \
    THPUtils_setError("stateless method initialization error");                \
    return false;                                                              \
  }                                                                            \
  if (PyObject_SetAttrString(TH_CONCAT_3(THP,type,TensorClass), THP_STATELESS_ATTRIBUTE_NAME, stateless) == -1) { \
    THPUtils_setError("stateless method initialization error (on assignment)");\
  }
  PyObject *arg = PyTuple_New(0);
  PyObject *stateless;
  INIT_STATELESS(Double);
  INIT_STATELESS(Float);
  INIT_STATELESS(Long);
  INIT_STATELESS(Int);
  INIT_STATELESS(Short);
  INIT_STATELESS(Char);
  INIT_STATELESS(Byte);
  Py_DECREF(arg);
  return true;
#undef INIT_STATELESS
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *self, PyObject *shm_manager_path)
{
  if (!THPUtils_checkBytes(shm_manager_path)) {
    THPUtils_setError("initialization error - expected bytes/string object as shm_manager_path!");
    return NULL;
  }
  libshm_init(THPUtils_bytesAsString(shm_manager_path));
  if (!THPModule_loadClasses(self))         return NULL;
  if (!THPModule_assignStateless(self))     return NULL;
  if (!THPModule_initCopy(self))            return NULL;
  return PyBool_FromLong(true);
}

static PyObject * THPModule_getNumThreads(PyObject *module)
{
#ifdef _OPENMP
  return PyLong_FromLong(omp_get_max_threads());
#else
  return PyLong_FromLong(1);
#endif
}

static PyObject * THPModule_setNumThreads(PyObject *module, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_num_threads expects an int, "
          "but got %s", THPUtils_typename(arg));
#ifdef _OPENMP
  omp_set_num_threads(THPUtils_unpackLong(arg));
#else
  PyErr_WarnEx(PyExc_RuntimeWarning, "set_num_threads is a no-op - torch was "
          "compiled without OpenMP support", 1);
#endif
  return 0;
}

static PyObject * THPModule_getRNGState(PyObject *module)
{
  HANDLE_TH_ERRORS
  THPGenerator *self = THPDefaultGenerator;
  THGenerator *generator = self->cdata;
  THByteTensorPtr _t = THByteTensor_new();
  THByteTensor_getRNGState(generator, _t.get());
  PyObject *_ret =  THPByteTensor_New(_t);
  _t.release();
  return _ret;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPModule_setRNGState(PyObject *_unused, PyObject *_new_state)
{
  HANDLE_TH_ERRORS
  THPGenerator *self = THPDefaultGenerator;
  THGenerator *generator = self->cdata;
  THPUtils_assert(THPByteTensor_Check(_new_state), "set_rng_state expects a "
          "torch.ByteTensor, but got %s", THPUtils_typename(_new_state));
  THByteTensor *new_state = ((THPByteTensor*)_new_state)->cdata;;
  THByteTensor_setRNGState(generator, new_state);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPModule_manualSeed(PyObject *_unused, PyObject *seed)
{
  HANDLE_TH_ERRORS
  THPGenerator *self = THPDefaultGenerator;
  THGenerator *generator = self->cdata;
  THPUtils_assert(THPUtils_checkLong(seed), "manual_seed expected a long, "
          "but got %s", THPUtils_typename(seed));
  THRandom_manualSeed(generator, THPUtils_unpackLong(seed));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

bool THPModule_isTensor(PyObject *obj)
{
  int result = PySet_Contains(tensor_classes, (PyObject*)Py_TYPE(obj));
  if (result == -1)
    throw std::logic_error("FATAL: tensor_classes isn't a set!");
  return result;
}

// Following two functions are taken from Python 2.7 source
// https://github.com/python/cpython/blob/2.7/Modules/_multiprocessing/multiprocessing.c
static PyObject * THPModule_sendfd(PyObject *self, PyObject *args)
{
  int conn, fd, res;
  struct iovec dummy_iov;
  char dummy_char;
  struct msghdr msg;
  struct cmsghdr *cmsg;
  union {
    struct cmsghdr hdr;
    unsigned char buf[CMSG_SPACE(sizeof(int))];
  } cmsgbuf;

  if (!PyArg_ParseTuple(args, "ii", &conn, &fd))
    return NULL;

  dummy_iov.iov_base = &dummy_char;
  dummy_iov.iov_len = 1;

  memset(&msg, 0, sizeof(msg));
  msg.msg_control = &cmsgbuf.buf;
  msg.msg_controllen = sizeof(cmsgbuf.buf);
  msg.msg_iov = &dummy_iov;
  msg.msg_iovlen = 1;

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  * (int *) CMSG_DATA(cmsg) = fd;

  Py_BEGIN_ALLOW_THREADS
  res = sendmsg(conn, &msg, 0);
  Py_END_ALLOW_THREADS

  if (res < 0)
    return PyErr_SetFromErrno(PyExc_OSError);
  Py_RETURN_NONE;
}

static PyObject * THPModule_recvfd(PyObject *self, PyObject *args)
{
  int conn, fd, res;
  char dummy_char;
  struct iovec dummy_iov;
  struct msghdr msg = {0};
  struct cmsghdr *cmsg;
  union {
    struct cmsghdr hdr;
    unsigned char buf[CMSG_SPACE(sizeof(int))];
  } cmsgbuf;

  if (!PyArg_ParseTuple(args, "i", &conn))
    return NULL;

  dummy_iov.iov_base = &dummy_char;
  dummy_iov.iov_len = 1;

  memset(&msg, 0, sizeof(msg));
  msg.msg_control = &cmsgbuf.buf;
  msg.msg_controllen = sizeof(cmsgbuf.buf);
  msg.msg_iov = &dummy_iov;
  msg.msg_iovlen = 1;

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  msg.msg_controllen = cmsg->cmsg_len;

  Py_BEGIN_ALLOW_THREADS
  res = recvmsg(conn, &msg, 0);
  Py_END_ALLOW_THREADS

  if (res < 0)
    return PyErr_SetFromErrno(PyExc_OSError);

  if (msg.msg_controllen < CMSG_LEN(sizeof(int)) ||
      (cmsg = CMSG_FIRSTHDR(&msg)) == NULL ||
      cmsg->cmsg_level != SOL_SOCKET ||
      cmsg->cmsg_type != SCM_RIGHTS ||
      cmsg->cmsg_len < CMSG_LEN(sizeof(int))) {
    /* If at least one control message is present, there should be
        no room for any further data in the buffer. */
    PyErr_SetString(PyExc_RuntimeError, "No file descriptor received");
    return NULL;
  }

  fd = * (int *) CMSG_DATA(cmsg);
  return Py_BuildValue("i", fd);
}

PyObject * THPModule_setDefaultTensorType(PyObject *_unused, PyObject *type)
{
  THPDefaultTensorClass = type;
  Py_RETURN_NONE;
}


#define IMPLEMENT_STATELESS(name)                                              \
static PyObject * TH_CONCAT_2(THPModule_, name)(PyObject *_unused, PyObject *args) \
{                                                                              \
  PyObject *tensor = THPDefaultTensorClass;                                    \
  for (int i = 0; i < PyTuple_Size(args); i++) {                               \
    PyObject *item = PyTuple_GET_ITEM(args, i);                                \
    if (THPModule_isTensor(item)) {                                            \
      tensor = item;                                                           \
      break;                                                                   \
    }                                                                          \
  }                                                                            \
                                                                               \
  PyObject *methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME);     \
  THPUtils_assert(methods, "Type %s doesn't implement statless methods",       \
      Py_TYPE(tensor)->tp_name);                                               \
  PyObject *method = PyObject_GetAttrString(methods, #name);                   \
  THPUtils_assert(method, "Type %s doesn't implement stateless method " #name, \
      Py_TYPE(tensor)->tp_name);                                               \
  return PyObject_Call(method, args, NULL);                                    \
}

IMPLEMENT_STATELESS(sigmoid)
IMPLEMENT_STATELESS(log)
IMPLEMENT_STATELESS(log1p)
IMPLEMENT_STATELESS(exp)
IMPLEMENT_STATELESS(cos)
IMPLEMENT_STATELESS(acos)
IMPLEMENT_STATELESS(cosh)
IMPLEMENT_STATELESS(sin)
IMPLEMENT_STATELESS(asin)
IMPLEMENT_STATELESS(sinh)
IMPLEMENT_STATELESS(tan)
IMPLEMENT_STATELESS(atan)
IMPLEMENT_STATELESS(tanh)
IMPLEMENT_STATELESS(sqrt)
IMPLEMENT_STATELESS(rsqrt)
IMPLEMENT_STATELESS(ceil)
IMPLEMENT_STATELESS(floor)
IMPLEMENT_STATELESS(round)
IMPLEMENT_STATELESS(abs)
IMPLEMENT_STATELESS(trunc)
IMPLEMENT_STATELESS(frac)
IMPLEMENT_STATELESS(mean)
IMPLEMENT_STATELESS(std)
IMPLEMENT_STATELESS(var)
IMPLEMENT_STATELESS(norm)
IMPLEMENT_STATELESS(cinv)
IMPLEMENT_STATELESS(neg)
IMPLEMENT_STATELESS(add)
IMPLEMENT_STATELESS(csub)
IMPLEMENT_STATELESS(mul)
IMPLEMENT_STATELESS(div)
IMPLEMENT_STATELESS(fmod)
IMPLEMENT_STATELESS(cmul)
IMPLEMENT_STATELESS(cdiv)
IMPLEMENT_STATELESS(cfmod)
IMPLEMENT_STATELESS(min)
IMPLEMENT_STATELESS(max)
IMPLEMENT_STATELESS(cmax)
IMPLEMENT_STATELESS(cmin)
IMPLEMENT_STATELESS(cpow)
IMPLEMENT_STATELESS(dot)
IMPLEMENT_STATELESS(sum)
IMPLEMENT_STATELESS(prod)
IMPLEMENT_STATELESS(remainder)
IMPLEMENT_STATELESS(cremainder)
IMPLEMENT_STATELESS(cumsum)
IMPLEMENT_STATELESS(cumprod)
IMPLEMENT_STATELESS(clamp)
IMPLEMENT_STATELESS(equal)
IMPLEMENT_STATELESS(eye)
IMPLEMENT_STATELESS(fill)
IMPLEMENT_STATELESS(diag)
IMPLEMENT_STATELESS(numel)
IMPLEMENT_STATELESS(sign)
IMPLEMENT_STATELESS(trace)
IMPLEMENT_STATELESS(tril)
IMPLEMENT_STATELESS(triu)
IMPLEMENT_STATELESS(zero)
IMPLEMENT_STATELESS(kthvalue)
IMPLEMENT_STATELESS(mode)
IMPLEMENT_STATELESS(median)
IMPLEMENT_STATELESS(cross)
IMPLEMENT_STATELESS(sort)
IMPLEMENT_STATELESS(topk)
IMPLEMENT_STATELESS(t)
IMPLEMENT_STATELESS(transpose)
IMPLEMENT_STATELESS(squeeze)
IMPLEMENT_STATELESS(renorm)
IMPLEMENT_STATELESS(dist)
IMPLEMENT_STATELESS(linspace)
IMPLEMENT_STATELESS(logspace)
IMPLEMENT_STATELESS(histc)
IMPLEMENT_STATELESS(atan2)
IMPLEMENT_STATELESS(pow)
IMPLEMENT_STATELESS(lerp)
IMPLEMENT_STATELESS(reshape)
IMPLEMENT_STATELESS(zeros)
IMPLEMENT_STATELESS(ones)
IMPLEMENT_STATELESS(index_select)
IMPLEMENT_STATELESS(narrow)
IMPLEMENT_STATELESS(addmm)
IMPLEMENT_STATELESS(addmv)
IMPLEMENT_STATELESS(addr)
IMPLEMENT_STATELESS(ger)
IMPLEMENT_STATELESS(mv)
IMPLEMENT_STATELESS(addbmm)
IMPLEMENT_STATELESS(baddbmm)
IMPLEMENT_STATELESS(addcmul)
IMPLEMENT_STATELESS(addcdiv)
IMPLEMENT_STATELESS(mm)
IMPLEMENT_STATELESS(bmm)
// TODO: this doesn't implement options that return numbers!
IMPLEMENT_STATELESS(multinomial)
IMPLEMENT_STATELESS(uniform)
IMPLEMENT_STATELESS(normal)
IMPLEMENT_STATELESS(cauchy)
IMPLEMENT_STATELESS(log_normal)
IMPLEMENT_STATELESS(exponential)
IMPLEMENT_STATELESS(random)
IMPLEMENT_STATELESS(geometric)
IMPLEMENT_STATELESS(bernoulli)
IMPLEMENT_STATELESS(randperm)
IMPLEMENT_STATELESS(unfold)
IMPLEMENT_STATELESS(range)
IMPLEMENT_STATELESS(gather)
IMPLEMENT_STATELESS(scatter)
IMPLEMENT_STATELESS(rand)
IMPLEMENT_STATELESS(randn)
IMPLEMENT_STATELESS(all)
IMPLEMENT_STATELESS(any)
IMPLEMENT_STATELESS(masked_select)

#undef IMPLEMENT_STATELESS

// For logical functions a reverse type search is required (if the first argument
// is a ByteTensor (result), it shouldn't pick it's version).
#define IMPLEMENT_STATELESS_REVERSED(name)                                     \
static PyObject * TH_CONCAT_2(THPModule_, name)(PyObject *_unused, PyObject *args) \
{                                                                              \
  PyObject *tensor = THPDefaultTensorClass;                                    \
  for (int i = PyTuple_Size(args)-1; i >= 0; i--) {                            \
    PyObject *item = PyTuple_GET_ITEM(args, i);                                \
    if (THPModule_isTensor(item)) {                                            \
      tensor = item;                                                           \
      break;                                                                   \
    }                                                                          \
  }                                                                            \
                                                                               \
  PyObject *methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME);     \
  THPUtils_assert(methods, "Type %s doesn't implement statless methods",       \
      Py_TYPE(tensor)->tp_name);                                               \
  PyObject *method = PyObject_GetAttrString(methods, #name);                   \
  THPUtils_assert(method, "Type %s doesn't implement stateless method " #name, \
      Py_TYPE(tensor)->tp_name);                                               \
  return PyObject_Call(method, args, NULL);                                    \
}

IMPLEMENT_STATELESS_REVERSED(gt)
IMPLEMENT_STATELESS_REVERSED(lt)
IMPLEMENT_STATELESS_REVERSED(ge)
IMPLEMENT_STATELESS_REVERSED(le)
IMPLEMENT_STATELESS_REVERSED(eq)
IMPLEMENT_STATELESS_REVERSED(ne)

#undef IMPLEMENT_STATELESS

// In nonzero, the first argument might be a LongTensor that will be used
// for indices output, so we should pick a function based on second
// tensor's type.
static PyObject * THPModule_nonzero(PyObject *_unused, PyObject *args)
{
  PyObject *tensor = THPDefaultTensorClass;
  if (PyTuple_Size(args) == 1)
    tensor = PyTuple_GET_ITEM(args, 0);
  else if (PyTuple_Size(args) == 2)
    tensor = PyTuple_GET_ITEM(args, 1);

  PyObject *methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME);
  THPUtils_assert(methods, "Type %s doesn't implement statless methods",
      Py_TYPE(tensor)->tp_name);
  PyObject *method = PyObject_GetAttrString(methods, "nonzero");
  THPUtils_assert(method, "Type %s doesn't implement stateless method nonzero",
      Py_TYPE(tensor)->tp_name);
  return PyObject_Call(method, args, NULL);
}

// In nonzero, the first argument might be a LongTensor that will be used
// for indices output, so we should pick a function based on second
// tensor's type.
static PyObject * THPModule_cat(PyObject *_unused, PyObject *args)
{
  PyObject *tensor = THPDefaultTensorClass;
  THPObjectPtr iterator;
  THPObjectPtr item;
  if (args && PyTuple_Size(args) > 0) {
    if (THPModule_isTensor(PyTuple_GET_ITEM(args, 0))) {
      tensor = PyTuple_GET_ITEM(args, 0);
    } else if ((iterator = PyObject_GetIter(PyTuple_GET_ITEM(args, 0)))) {
      item = PyIter_Next(iterator);
      if (item && THPModule_isTensor(item)) {
        tensor = item;
      }
    }
    PyErr_Clear();
  }

  PyObject *methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME);
  THPUtils_assert(methods, "Type %s doesn't implement statless methods",
      Py_TYPE(tensor)->tp_name);
  PyObject *method = PyObject_GetAttrString(methods, "cat");
  THPUtils_assert(method, "Type %s doesn't implement stateless method nonzero",
      Py_TYPE(tensor)->tp_name);
  return PyObject_Call(method, args, NULL);
}

PyObject *THPModule_safeCall(PyObject *_unused, PyObject *args, PyObject *kwargs)
{
  PyObject *result = NULL;
  PyThreadState *thread_state = PyThreadState_Get();
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;
  THPUtils_assert(num_args > 0, "expected at least one argument");
  try {
    THPObjectPtr args_slice = PyTuple_GetSlice(args, 1, num_args);
    result = PyObject_Call(PyTuple_GET_ITEM(args, 0), args_slice, kwargs);
  } catch (std::exception &e) {
    PyEval_RestoreThread(thread_state);
    PyErr_SetString(THPException_FatalError, e.what());
    Py_LeaveRecursiveCall();
  }
  return result;
}

#ifdef WITH_CUDA
extern PyObject * THCPModule_initExtension(PyObject *self);
extern PyObject * THCPModule_setDevice_wrap(PyObject *self, PyObject *arg);
extern PyObject * THCPModule_getDevice_wrap(PyObject *self);
extern PyObject * THCPModule_getDeviceCount_wrap(PyObject *self);
extern PyObject * THCPModule_getDriverVersion(PyObject *self);
extern PyObject * THCPModule_isDriverSufficient(PyObject *self);
#endif

static PyMethodDef TorchMethods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,     METH_O,  NULL},
#ifdef WITH_CUDA
  {"_cuda_init",      (PyCFunction)THCPModule_initExtension,    METH_NOARGS,  NULL},
  {"_cuda_setDevice", (PyCFunction)THCPModule_setDevice_wrap,   METH_O,       NULL},
  {"_cuda_getDevice", (PyCFunction)THCPModule_getDevice_wrap,   METH_NOARGS,  NULL},
  {"_cuda_getDeviceCount", (PyCFunction)THCPModule_getDeviceCount_wrap, METH_NOARGS, NULL},
  {"_cuda_isDriverSufficient", (PyCFunction)THCPModule_isDriverSufficient, METH_NOARGS, NULL},
  {"_cuda_getDriverVersion", (PyCFunction)THCPModule_getDriverVersion, METH_NOARGS, NULL},
#endif
  {"_safe_call",      (PyCFunction)THPModule_safeCall,          METH_VARARGS | METH_KEYWORDS, NULL},
  {"_sendfd",         (PyCFunction)THPModule_sendfd,            METH_VARARGS, NULL},
  {"_recvfd",         (PyCFunction)THPModule_recvfd,            METH_VARARGS, NULL},
  {"_set_default_tensor_type", (PyCFunction)THPModule_setDefaultTensorType, METH_O, NULL},
  {"_tensorCopy",     (PyCFunction)THPModule_tensorCopyWrapper, METH_VARARGS, NULL},
  {"_storageCopy",    (PyCFunction)THPModule_storageCopyWrapper, METH_VARARGS, NULL},
  {"_tensorCopyAsync", (PyCFunction)THPModule_tensor_asyncCopyWrapper, METH_VARARGS, NULL},
  {"_storageCopyAsync", (PyCFunction)THPModule_storage_asyncCopyWrapper, METH_VARARGS, NULL},
  {"get_num_threads", (PyCFunction)THPModule_getNumThreads,     METH_NOARGS,  NULL},
  {"set_num_threads", (PyCFunction)THPModule_setNumThreads,     METH_O,       NULL},
  {"get_rng_state",   (PyCFunction)THPModule_getRNGState,       METH_NOARGS,  NULL},
  {"set_rng_state",   (PyCFunction)THPModule_setRNGState,       METH_O,       NULL},
  {"manual_seed",     (PyCFunction)THPModule_manualSeed,        METH_O,       NULL},

  {"sigmoid",         (PyCFunction)THPModule_sigmoid,           METH_VARARGS, NULL},
  {"log",             (PyCFunction)THPModule_log,               METH_VARARGS, NULL},
  {"log1p",           (PyCFunction)THPModule_log1p,             METH_VARARGS, NULL},
  {"exp",             (PyCFunction)THPModule_exp,               METH_VARARGS, NULL},
  {"cos",             (PyCFunction)THPModule_cos,               METH_VARARGS, NULL},
  {"acos",            (PyCFunction)THPModule_acos,              METH_VARARGS, NULL},
  {"cosh",            (PyCFunction)THPModule_cosh,              METH_VARARGS, NULL},
  {"sin",             (PyCFunction)THPModule_sin,               METH_VARARGS, NULL},
  {"asin",            (PyCFunction)THPModule_asin,              METH_VARARGS, NULL},
  {"sinh",            (PyCFunction)THPModule_sinh,              METH_VARARGS, NULL},
  {"tan",             (PyCFunction)THPModule_tan,               METH_VARARGS, NULL},
  {"atan",            (PyCFunction)THPModule_atan,              METH_VARARGS, NULL},
  {"tanh",            (PyCFunction)THPModule_tanh,              METH_VARARGS, NULL},
  {"sqrt",            (PyCFunction)THPModule_sqrt,              METH_VARARGS, NULL},
  {"rsqrt",           (PyCFunction)THPModule_rsqrt,             METH_VARARGS, NULL},
  {"ceil",            (PyCFunction)THPModule_ceil,              METH_VARARGS, NULL},
  {"floor",           (PyCFunction)THPModule_floor,             METH_VARARGS, NULL},
  {"round",           (PyCFunction)THPModule_round,             METH_VARARGS, NULL},
  {"abs",             (PyCFunction)THPModule_abs,               METH_VARARGS, NULL},
  {"trunc",           (PyCFunction)THPModule_trunc,             METH_VARARGS, NULL},
  {"frac",            (PyCFunction)THPModule_frac,              METH_VARARGS, NULL},
  {"mean",            (PyCFunction)THPModule_mean,              METH_VARARGS, NULL},
  {"std",             (PyCFunction)THPModule_std,               METH_VARARGS, NULL},
  {"var",             (PyCFunction)THPModule_var,               METH_VARARGS, NULL},
  {"norm",            (PyCFunction)THPModule_norm,              METH_VARARGS, NULL},
  {"cinv",            (PyCFunction)THPModule_cinv,              METH_VARARGS, NULL},
  {"neg",             (PyCFunction)THPModule_neg,               METH_VARARGS, NULL},
  {"add",             (PyCFunction)THPModule_add,               METH_VARARGS, NULL},
  {"csub",            (PyCFunction)THPModule_csub,              METH_VARARGS, NULL},
  {"mul",             (PyCFunction)THPModule_mul,               METH_VARARGS, NULL},
  {"div",             (PyCFunction)THPModule_div,               METH_VARARGS, NULL},
  {"fmod",            (PyCFunction)THPModule_fmod,              METH_VARARGS, NULL},
  {"mod",             (PyCFunction)THPModule_fmod,              METH_VARARGS, NULL},
  {"cmul",            (PyCFunction)THPModule_cmul,              METH_VARARGS, NULL},
  {"cdiv",            (PyCFunction)THPModule_cdiv,              METH_VARARGS, NULL},
  {"cfmod",           (PyCFunction)THPModule_cfmod,             METH_VARARGS, NULL},
  {"cmod",            (PyCFunction)THPModule_cfmod,             METH_VARARGS, NULL},
  {"min",             (PyCFunction)THPModule_min,               METH_VARARGS, NULL},
  {"max",             (PyCFunction)THPModule_max,               METH_VARARGS, NULL},
  {"cmax",            (PyCFunction)THPModule_cmax,              METH_VARARGS, NULL},
  {"cmin",            (PyCFunction)THPModule_cmin,              METH_VARARGS, NULL},
  {"cpow",            (PyCFunction)THPModule_cpow,              METH_VARARGS, NULL},
  {"dot",             (PyCFunction)THPModule_dot,               METH_VARARGS, NULL},
  {"sum",             (PyCFunction)THPModule_sum,               METH_VARARGS, NULL},
  {"prod",            (PyCFunction)THPModule_prod,              METH_VARARGS, NULL},
  {"remainder",       (PyCFunction)THPModule_remainder,         METH_VARARGS, NULL},
  {"cremainder",      (PyCFunction)THPModule_cremainder,        METH_VARARGS, NULL},
  {"cumsum",          (PyCFunction)THPModule_cumsum,            METH_VARARGS, NULL},
  {"cumprod",         (PyCFunction)THPModule_cumprod,           METH_VARARGS, NULL},
  {"clamp",           (PyCFunction)THPModule_clamp,             METH_VARARGS, NULL},
  {"equal",           (PyCFunction)THPModule_equal,             METH_VARARGS, NULL},
  {"eye",             (PyCFunction)THPModule_eye,               METH_VARARGS, NULL},
  {"fill",            (PyCFunction)THPModule_fill,              METH_VARARGS, NULL},
  {"diag",            (PyCFunction)THPModule_diag,              METH_VARARGS, NULL},
  {"numel",           (PyCFunction)THPModule_numel,             METH_VARARGS, NULL},
  {"sign",            (PyCFunction)THPModule_sign,              METH_VARARGS, NULL},
  {"trace",           (PyCFunction)THPModule_trace,             METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPModule_tril,              METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPModule_triu,              METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPModule_zero,              METH_VARARGS, NULL},
  {"gt",              (PyCFunction)THPModule_gt,                METH_VARARGS, NULL},
  {"lt",              (PyCFunction)THPModule_lt,                METH_VARARGS, NULL},
  {"ge",              (PyCFunction)THPModule_ge,                METH_VARARGS, NULL},
  {"le",              (PyCFunction)THPModule_le,                METH_VARARGS, NULL},
  {"eq",              (PyCFunction)THPModule_eq,                METH_VARARGS, NULL},
  {"ne",              (PyCFunction)THPModule_ne,                METH_VARARGS, NULL},
  {"kthvalue",        (PyCFunction)THPModule_kthvalue,          METH_VARARGS, NULL},
  {"mode",            (PyCFunction)THPModule_mode,              METH_VARARGS, NULL},
  {"median",          (PyCFunction)THPModule_median,            METH_VARARGS, NULL},
  {"cross",           (PyCFunction)THPModule_cross,             METH_VARARGS, NULL},
  {"sort",            (PyCFunction)THPModule_sort,              METH_VARARGS, NULL},
  {"topk",            (PyCFunction)THPModule_topk,              METH_VARARGS, NULL},
  {"t",               (PyCFunction)THPModule_t,                 METH_VARARGS, NULL},
  {"transpose",       (PyCFunction)THPModule_transpose,         METH_VARARGS, NULL},
  {"squeeze",         (PyCFunction)THPModule_squeeze,           METH_VARARGS, NULL},
  {"nonzero",         (PyCFunction)THPModule_nonzero,           METH_VARARGS, NULL},
  {"renorm",          (PyCFunction)THPModule_renorm,            METH_VARARGS, NULL},
  {"dist",            (PyCFunction)THPModule_dist,              METH_VARARGS, NULL},
  {"linspace",        (PyCFunction)THPModule_linspace,          METH_VARARGS, NULL},
  {"logspace",        (PyCFunction)THPModule_logspace,          METH_VARARGS, NULL},
  {"histc",           (PyCFunction)THPModule_histc,             METH_VARARGS, NULL},
  {"atan2",           (PyCFunction)THPModule_atan2,             METH_VARARGS, NULL},
  {"pow",             (PyCFunction)THPModule_pow,               METH_VARARGS, NULL},
  {"lerp",            (PyCFunction)THPModule_lerp,              METH_VARARGS, NULL},
  {"reshape",         (PyCFunction)THPModule_reshape,           METH_VARARGS, NULL},
  {"zeros",           (PyCFunction)THPModule_zeros,             METH_VARARGS, NULL},
  {"ones",            (PyCFunction)THPModule_ones,              METH_VARARGS, NULL},
  {"index_select",    (PyCFunction)THPModule_index_select,      METH_VARARGS, NULL},
  {"narrow",          (PyCFunction)THPModule_narrow,            METH_VARARGS, NULL},
  {"addmm",           (PyCFunction)THPModule_addmm,             METH_VARARGS, NULL},
  {"addmv",           (PyCFunction)THPModule_addmv,             METH_VARARGS, NULL},
  {"addr",            (PyCFunction)THPModule_addr,              METH_VARARGS, NULL},
  {"ger",             (PyCFunction)THPModule_ger,               METH_VARARGS, NULL},
  {"mv",              (PyCFunction)THPModule_mv,                METH_VARARGS, NULL},
  {"addbmm",          (PyCFunction)THPModule_addbmm,            METH_VARARGS, NULL},
  {"baddbmm",         (PyCFunction)THPModule_baddbmm,           METH_VARARGS, NULL},
  {"addcmul",         (PyCFunction)THPModule_addcmul,           METH_VARARGS, NULL},
  {"addcdiv",         (PyCFunction)THPModule_addcdiv,           METH_VARARGS, NULL},
  {"mm",              (PyCFunction)THPModule_mm,                METH_VARARGS, NULL},
  {"bmm",             (PyCFunction)THPModule_bmm,               METH_VARARGS, NULL},
  {"multinomial",     (PyCFunction)THPModule_multinomial,       METH_VARARGS, NULL},
  {"uniform",         (PyCFunction)THPModule_uniform,           METH_VARARGS, NULL},
  {"normal",          (PyCFunction)THPModule_normal,            METH_VARARGS, NULL},
  {"cauchy",          (PyCFunction)THPModule_cauchy,            METH_VARARGS, NULL},
  {"log_normal",      (PyCFunction)THPModule_log_normal,        METH_VARARGS, NULL},
  {"exponential",     (PyCFunction)THPModule_exponential,       METH_VARARGS, NULL},
  {"random",          (PyCFunction)THPModule_random,            METH_VARARGS, NULL},
  {"geometric",       (PyCFunction)THPModule_geometric,         METH_VARARGS, NULL},
  {"bernoulli",       (PyCFunction)THPModule_bernoulli,         METH_VARARGS, NULL},
  {"rand",            (PyCFunction)THPModule_rand,              METH_VARARGS, NULL},
  {"randn",           (PyCFunction)THPModule_randn,             METH_VARARGS, NULL},
  {"randperm",        (PyCFunction)THPModule_randperm,          METH_VARARGS, NULL},
  {"unfold",          (PyCFunction)THPModule_unfold,            METH_VARARGS, NULL},
  {"range",           (PyCFunction)THPModule_range,             METH_VARARGS, NULL},
  {"gather",          (PyCFunction)THPModule_gather,            METH_VARARGS, NULL},
  {"scatter",         (PyCFunction)THPModule_scatter,           METH_VARARGS, NULL},
  {"all",             (PyCFunction)THPModule_all,               METH_VARARGS, NULL},
  {"any",             (PyCFunction)THPModule_any,               METH_VARARGS, NULL},
  {"cat",             (PyCFunction)THPModule_cat,               METH_VARARGS, NULL},
  {"masked_select",   (PyCFunction)THPModule_masked_select,     METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION != 2
static struct PyModuleDef torchmodule = {
   PyModuleDef_HEAD_INIT,
   "torch._C",
   NULL,
   -1,
   TorchMethods
};
#endif

static void errorHandler(const char *msg, void *data)
{
  throw THException(msg);
}

static void errorHandlerArg(int argNumber, const char *msg, void *data)
{
  throw THArgException(msg, argNumber);
}

static void updateErrorHandlers()
{
  THSetErrorHandler(errorHandler, NULL);
  THSetArgErrorHandler(errorHandlerArg, NULL);
}

bool THCPDoubleStorage_init(PyObject *module);
bool THCPFloatStorage_init(PyObject *module);
bool THCPHalfStorage_init(PyObject *module);
bool THCPLongStorage_init(PyObject *module);
bool THCPIntStorage_init(PyObject *module);
bool THCPShortStorage_init(PyObject *module);
bool THCPCharStorage_init(PyObject *module);
bool THCPByteStorage_init(PyObject *module);

bool THCPDoubleTensor_init(PyObject *module);
bool THCPFloatTensor_init(PyObject *module);
bool THCPHalfTensor_init(PyObject *module);
bool THCPLongTensor_init(PyObject *module);
bool THCPIntTensor_init(PyObject *module);
bool THCPShortTensor_init(PyObject *module);
bool THCPCharTensor_init(PyObject *module);
bool THCPByteTensor_init(PyObject *module);

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_C()
#else
PyMODINIT_FUNC PyInit__C()
#endif
{

#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", TorchMethods));
#else
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
  ASSERT_TRUE(THPGenerator_init(module));
  ASSERT_TRUE(THPException_init(module));

  ASSERT_TRUE(THPDoubleStorage_init(module));
  ASSERT_TRUE(THPFloatStorage_init(module));
  ASSERT_TRUE(THPLongStorage_init(module));
  ASSERT_TRUE(THPIntStorage_init(module));
  ASSERT_TRUE(THPShortStorage_init(module));
  ASSERT_TRUE(THPCharStorage_init(module));
  ASSERT_TRUE(THPByteStorage_init(module));

  ASSERT_TRUE(THPDoubleTensor_init(module));
  ASSERT_TRUE(THPFloatTensor_init(module));
  ASSERT_TRUE(THPLongTensor_init(module));
  ASSERT_TRUE(THPIntTensor_init(module));
  ASSERT_TRUE(THPShortTensor_init(module));
  ASSERT_TRUE(THPCharTensor_init(module));
  ASSERT_TRUE(THPByteTensor_init(module));

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
  ASSERT_TRUE(THCPHalfStorage_init(module));

  ASSERT_TRUE(THCPDoubleTensor_init(module));
  ASSERT_TRUE(THCPFloatTensor_init(module));
  ASSERT_TRUE(THCPHalfTensor_init(module));
  ASSERT_TRUE(THCPLongTensor_init(module));
  ASSERT_TRUE(THCPIntTensor_init(module));
  ASSERT_TRUE(THCPShortTensor_init(module));
  ASSERT_TRUE(THCPCharTensor_init(module));
  ASSERT_TRUE(THCPByteTensor_init(module));
#endif

  THPDefaultGenerator = (THPGenerator*)THPGenerator_New();
  ASSERT_TRUE(THPDefaultGenerator != nullptr);

  updateErrorHandlers();

#ifdef WITH_NUMPY
  import_array();
#endif

#if PY_MAJOR_VERSION == 2
#else
  return module;
#endif

#undef ASSERT_TRUE
}
