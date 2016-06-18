#include <Python.h>

#include <stdbool.h>
#include <TH/TH.h>

#include "THP.h"

#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif

static PyObject* module;
static PyObject* tensor_classes;

PyObject *THPDoubleStorageClass = NULL;
PyObject *THPFloatStorageClass  = NULL;
PyObject *THPLongStorageClass   = NULL;
PyObject *THPIntStorageClass    = NULL;
PyObject *THPShortStorageClass  = NULL;
PyObject *THPCharStorageClass   = NULL;
PyObject *THPByteStorageClass   = NULL;

PyObject *THPDoubleTensorClass  = NULL;
PyObject *THPFloatTensorClass   = NULL;
PyObject *THPLongTensorClass    = NULL;
PyObject *THPIntTensorClass     = NULL;
PyObject *THPShortTensorClass   = NULL;
PyObject *THPCharTensorClass    = NULL;
PyObject *THPByteTensorClass    = NULL;

PyObject *THPDefaultTensorClass = NULL;
PyObject *THPGeneratorClass     = NULL;

// Used if no other generator is provided
// TODO: this won't be thread-safe anymore
THPGenerator *THPDefaultGenerator   = NULL;

static bool THPModule_loadClasses(PyObject *self)
{
  // TODO: error checking
  PyObject *torch_module = PyImport_ImportModule("torch");
  PyObject* module_dict = PyModule_GetDict(torch_module);

  THPDoubleStorageClass = PyMapping_GetItemString(module_dict,(char*)"DoubleStorage");
  THPFloatStorageClass  = PyMapping_GetItemString(module_dict,(char*)"FloatStorage");
  THPLongStorageClass   = PyMapping_GetItemString(module_dict,(char*)"LongStorage");
  THPIntStorageClass    = PyMapping_GetItemString(module_dict,(char*)"IntStorage");
  THPShortStorageClass  = PyMapping_GetItemString(module_dict,(char*)"ShortStorage");
  THPCharStorageClass   = PyMapping_GetItemString(module_dict,(char*)"CharStorage");
  THPByteStorageClass   = PyMapping_GetItemString(module_dict,(char*)"ByteStorage");

  THPDoubleTensorClass  = PyMapping_GetItemString(module_dict,(char*)"DoubleTensor");
  THPFloatTensorClass   = PyMapping_GetItemString(module_dict,(char*)"FloatTensor");
  THPLongTensorClass    = PyMapping_GetItemString(module_dict,(char*)"LongTensor");
  THPIntTensorClass     = PyMapping_GetItemString(module_dict,(char*)"IntTensor");
  THPShortTensorClass   = PyMapping_GetItemString(module_dict,(char*)"ShortTensor");
  THPCharTensorClass    = PyMapping_GetItemString(module_dict,(char*)"CharTensor");
  THPByteTensorClass    = PyMapping_GetItemString(module_dict,(char*)"ByteTensor");
  PySet_Add(tensor_classes, THPDoubleTensorClass);
  PySet_Add(tensor_classes, THPFloatTensorClass);
  PySet_Add(tensor_classes, THPLongTensorClass);
  PySet_Add(tensor_classes, THPIntTensorClass);
  PySet_Add(tensor_classes, THPShortTensorClass);
  PySet_Add(tensor_classes, THPCharTensorClass);
  PySet_Add(tensor_classes, THPByteTensorClass);

  THPDefaultTensorClass = THPDoubleTensorClass;

  return true;
}

#define INIT_STATELESS(type)                                                   \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_2(type, TensorStatelessType), arg, NULL); \
  PyObject_SetAttrString(TH_CONCAT_3(THP,type,TensorClass), "_torch", stateless);
static bool THPModule_assignStateless(PyObject *self)
{
  // TODO: error checking
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
}
#undef INIT_STATELESS

static PyObject * THPModule_initExtension(PyObject *self)
{
  THPModule_loadClasses(self);
  THPModule_assignStateless(self);
  return PyBool_FromLong(true);
}

static bool THPModule_isTensor(PyObject *obj)
{
  // TODO: error handling
  return PySet_Contains(tensor_classes, (PyObject*)Py_TYPE(obj));
}

static PyObject * THPModule_tensorCheck(PyObject *module, PyObject *obj)
{
  return PyBool_FromLong(THPModule_isTensor(obj));
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
  PyObject *methods = PyObject_GetAttrString(tensor, "_torch");                \
  PyObject *method = PyObject_GetAttrString(methods, #name);                   \
  if (!method)                                                                 \
    return NULL;                                                               \
  return PyObject_Call(method, args, NULL);                                    \
  /* TODO: error handling */                                                   \
  return NULL;                                                                 \
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
IMPLEMENT_STATELESS(gt)
IMPLEMENT_STATELESS(lt)
IMPLEMENT_STATELESS(ge)
IMPLEMENT_STATELESS(le)
IMPLEMENT_STATELESS(eq)
IMPLEMENT_STATELESS(ne)
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
IMPLEMENT_STATELESS(index)
IMPLEMENT_STATELESS(indexCopy)
IMPLEMENT_STATELESS(indexAdd)
IMPLEMENT_STATELESS(indexFill)
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
IMPLEMENT_STATELESS(logNormal)
IMPLEMENT_STATELESS(exponential)
IMPLEMENT_STATELESS(random)
IMPLEMENT_STATELESS(geometric)
IMPLEMENT_STATELESS(bernoulli)
IMPLEMENT_STATELESS(randperm)
IMPLEMENT_STATELESS(unfold)
IMPLEMENT_STATELESS(range)
IMPLEMENT_STATELESS(gather)
IMPLEMENT_STATELESS(scatter)

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

  PyObject *methods = PyObject_GetAttrString(tensor, "_torch");
  PyObject *method = PyObject_GetAttrString(methods, "nonzero");
  if (!method)
    return NULL;
  return PyObject_Call(method, args, NULL);
  /* TODO: error handling */
  return NULL;
}

static PyMethodDef TorchMethods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,     METH_NOARGS,  NULL},
  {"isTensor",        (PyCFunction)THPModule_tensorCheck,       METH_O,       NULL},

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
  {"index",           (PyCFunction)THPModule_index,             METH_VARARGS, NULL},
  {"indexCopy",       (PyCFunction)THPModule_indexCopy,         METH_VARARGS, NULL},
  {"indexAdd",        (PyCFunction)THPModule_indexAdd,          METH_VARARGS, NULL},
  {"indexFill",       (PyCFunction)THPModule_indexFill,         METH_VARARGS, NULL},
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
  {"logNormal",       (PyCFunction)THPModule_logNormal,         METH_VARARGS, NULL},
  {"exponential",     (PyCFunction)THPModule_exponential,       METH_VARARGS, NULL},
  {"random",          (PyCFunction)THPModule_random,            METH_VARARGS, NULL},
  {"geometric",       (PyCFunction)THPModule_geometric,         METH_VARARGS, NULL},
  {"bernoulli",       (PyCFunction)THPModule_bernoulli,         METH_VARARGS, NULL},
  {"randperm",        (PyCFunction)THPModule_randperm,          METH_VARARGS, NULL},
  {"unfold",          (PyCFunction)THPModule_unfold,            METH_VARARGS, NULL},
  {"range",           (PyCFunction)THPModule_range,             METH_VARARGS, NULL},
  {"gather",          (PyCFunction)THPModule_gather,            METH_VARARGS, NULL},
  {"scatter",         (PyCFunction)THPModule_scatter,           METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION != 2
static struct PyModuleDef torchmodule = {
   PyModuleDef_HEAD_INIT,
   "torch.C",
   NULL,
   -1,
   TorchMethods
};
#endif

static void errorHandler(const char *msg, void *data)
{
  fprintf(stderr, "%s\n", msg);
  throw THException(msg);
}

static void errorHandlerArg(int argNumber, const char *msg, void *data)
{
  fprintf(stderr, "%s\n", msg);
  throw THArgException(msg, argNumber);
}

static void updateErrorHandlers()
{
  THSetErrorHandler(errorHandler, NULL);
  THSetArgErrorHandler(errorHandlerArg, NULL);
}

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC initC()
#else
PyMODINIT_FUNC PyInit_C()
#endif
{
#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch.C", TorchMethods));
#else
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
  ASSERT_TRUE(tensor_classes = PySet_New(NULL));
  ASSERT_TRUE(PyObject_SetAttrString(module, "_tensorclasses", tensor_classes) == 0);

  ASSERT_TRUE(THPGenerator_init(module));

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

  THPDefaultGenerator = (THPGenerator*)THPGenerator_newObject();
  ASSERT_TRUE(THPDefaultGenerator != nullptr);

  updateErrorHandlers();

#if PY_MAJOR_VERSION == 2
#else
  return module;
#endif
}

#undef ASSERT_TRUE
