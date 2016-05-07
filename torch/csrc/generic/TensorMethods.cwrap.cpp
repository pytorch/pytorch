#define PARSE_TUPLE(...) if (!PyArg_ParseTuple(__VA_ARGS__)) return NULL
#define RETURN_SELF Py_INCREF(self); return (PyObject*)self

#define POINTWISE_OP(name)                                                     \
static PyObject * THPTensor_(name)(THPTensor *self, PyObject *args)            \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  THPTensor *source = self;                                                    \
  PARSE_TUPLE(args, "|O!", &THPTensorType, &source);                           \
  THTensor_(name)(self->cdata, source->cdata);                                 \
  Py_INCREF(self);                                                             \
  return (PyObject*)self;                                                      \
  END_HANDLE_TH_ERRORS                                                         \
}

#define SIMPLE_OP(name, expr)                                                  \
static PyObject * THPTensor_(name)(THPTensor *self)                            \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  return (expr);                                                               \
  END_HANDLE_TH_ERRORS                                                         \
}

#define SIMPLE_RETURN_SELF(name, expr)                                         \
static PyObject * THPTensor_(name)(THPTensor *self)                            \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  expr;                                                                        \
  Py_INCREF(self);                                                             \
  return (PyObject*)self;                                                      \
  END_HANDLE_TH_ERRORS                                                         \
}

#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
POINTWISE_OP(abs)
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
POINTWISE_OP(sigmoid)
POINTWISE_OP(log)
POINTWISE_OP(log1p)
POINTWISE_OP(exp)
POINTWISE_OP(cos)
POINTWISE_OP(acos)
POINTWISE_OP(cosh)
POINTWISE_OP(sin)
POINTWISE_OP(asin)
POINTWISE_OP(sinh)
POINTWISE_OP(tan)
POINTWISE_OP(atan)
POINTWISE_OP(tanh)
POINTWISE_OP(sqrt)
POINTWISE_OP(rsqrt)
POINTWISE_OP(ceil)
POINTWISE_OP(floor)
POINTWISE_OP(round)
POINTWISE_OP(abs)
POINTWISE_OP(trunc)
POINTWISE_OP(frac)
#endif

SIMPLE_OP(elementSize,      PyLong_FromLong(THStorage_(elementSize)()))
SIMPLE_OP(storage,          THPStorage_(newObject)(THTensor_(storage)(self->cdata)))
SIMPLE_OP(storageOffset,    PyLong_FromLong(THTensor_(storageOffset)(self->cdata)))
SIMPLE_OP(numel,            PyLong_FromLong(THTensor_(numel)(self->cdata)))
SIMPLE_OP(nDimension,       PyLong_FromLong(THTensor_(nDimension)(self->cdata)))

SIMPLE_RETURN_SELF(free,    THTensor_(free)(self->cdata))
SIMPLE_RETURN_SELF(retain,  THTensor_(retain)(self->cdata))
SIMPLE_RETURN_SELF(zero,    THTensor_(zero)(self->cdata))

[[
  size
  size -> long
    - self
    - long dim
  newSizeOf -> THLongStorage
    - self
]]

[[
  stride
  stride -> long
    - self
    - long dim
  newStrideOf -> THLongStorage
    - self
]]

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
[[
  mean
  meanall -> double
    - self
  mean -> self
    - self
    - self
    - long dim
  mean -> self
    - self
    - THTensor other
    - long dim
]]

[[
  var
  varall -> double
    - self
  var -> self
    - self
    - self
    - long dim
    - CONSTANT false
  var -> self
    - self
    - THTensor other
    - long dim
    - CONSTANT false
]]

[[
  std
  stdall -> double
    - self
  std -> self
    - self
    - self
    - long dim
    - CONSTANT false
  std -> self
    - self
    - THTensor other
    - long dim
    - CONSTANT false
]]
#endif

[[
  fill
  fill -> self
    - self
    - real value
]]

[[
  isSameSizeAs
  isSameSizeAs -> bool
    - self
    - THTensor other
]]

// Declared in TensorCopy.cpp
static PyObject * THPTensor_(copy)(THPTensor *self, PyObject *other);

static PyMethodDef THPTensor_(methods)[] = {
#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
  {"abs",             (PyCFunction)THPTensor_(abs),             METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"sigmoid",         (PyCFunction)THPTensor_(sigmoid),         METH_VARARGS, NULL},
  {"log",             (PyCFunction)THPTensor_(log),             METH_VARARGS, NULL},
  {"log1p",           (PyCFunction)THPTensor_(log1p),           METH_VARARGS, NULL},
  {"exp",             (PyCFunction)THPTensor_(exp),             METH_VARARGS, NULL},
  {"cos",             (PyCFunction)THPTensor_(cos),             METH_VARARGS, NULL},
  {"acos",            (PyCFunction)THPTensor_(acos),            METH_VARARGS, NULL},
  {"cosh",            (PyCFunction)THPTensor_(cosh),            METH_VARARGS, NULL},
  {"sin",             (PyCFunction)THPTensor_(sin),             METH_VARARGS, NULL},
  {"asin",            (PyCFunction)THPTensor_(asin),            METH_VARARGS, NULL},
  {"sinh",            (PyCFunction)THPTensor_(sinh),            METH_VARARGS, NULL},
  {"tan",             (PyCFunction)THPTensor_(tan),             METH_VARARGS, NULL},
  {"atan",            (PyCFunction)THPTensor_(atan),            METH_VARARGS, NULL},
  {"tanh",            (PyCFunction)THPTensor_(tanh),            METH_VARARGS, NULL},
  {"sqrt",            (PyCFunction)THPTensor_(sqrt),            METH_VARARGS, NULL},
  {"rsqrt",           (PyCFunction)THPTensor_(rsqrt),           METH_VARARGS, NULL},
  {"ceil",            (PyCFunction)THPTensor_(ceil),            METH_VARARGS, NULL},
  {"floor",           (PyCFunction)THPTensor_(floor),           METH_VARARGS, NULL},
  {"round",           (PyCFunction)THPTensor_(round),           METH_VARARGS, NULL},
  {"abs",             (PyCFunction)THPTensor_(abs),             METH_VARARGS, NULL},
  {"trunc",           (PyCFunction)THPTensor_(trunc),           METH_VARARGS, NULL},
  {"frac",            (PyCFunction)THPTensor_(frac),            METH_VARARGS, NULL},
  {"mean",            (PyCFunction)THPTensor_(mean),            METH_VARARGS, NULL},
  {"std",             (PyCFunction)THPTensor_(std),             METH_VARARGS, NULL},
  {"var",             (PyCFunction)THPTensor_(var),             METH_VARARGS, NULL},
#endif
  {"elementSize",     (PyCFunction)THPTensor_(elementSize),     METH_NOARGS,  NULL},
  {"fill",            (PyCFunction)THPTensor_(fill),            METH_VARARGS, NULL},
  {"free",            (PyCFunction)THPTensor_(free),            METH_NOARGS,  NULL},
  {"dim",             (PyCFunction)THPTensor_(nDimension),      METH_NOARGS,  NULL},
  {"copy",            (PyCFunction)THPTensor_(copy),            METH_O,       NULL},
  {"isSameSizeAs",    (PyCFunction)THPTensor_(isSameSizeAs),    METH_VARARGS, NULL},
  {"numel",           (PyCFunction)THPTensor_(numel),           METH_NOARGS,  NULL},
  {"nElement",        (PyCFunction)THPTensor_(numel),           METH_NOARGS,  NULL},
  {"nDimension",      (PyCFunction)THPTensor_(nDimension),      METH_NOARGS,  NULL},
  {"size",            (PyCFunction)THPTensor_(size),            METH_VARARGS, NULL},
  {"storage",         (PyCFunction)THPTensor_(storage),         METH_NOARGS,  NULL},
  {"storageOffset",   (PyCFunction)THPTensor_(storageOffset),   METH_NOARGS,  NULL},
  {"stride",          (PyCFunction)THPTensor_(stride),          METH_VARARGS, NULL},
  {"retain",          (PyCFunction)THPTensor_(retain),          METH_NOARGS,  NULL},
  {"zero",            (PyCFunction)THPTensor_(zero),            METH_NOARGS,  NULL},
  {NULL}
};
