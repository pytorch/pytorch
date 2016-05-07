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
  meanall -> accreal
    - self
  mean -> self
    - self
    - self
    - long dim
  mean -> self
    - self
    - THTensor source
    - long dim
]]

[[
  var
  varall -> accreal
    - self
  var -> self
    - self
    - self
    - long dim
    - CONSTANT false
  var -> self
    - self
    - THTensor source
    - long dim
    - CONSTANT false
]]

[[
  std
  stdall -> accreal
    - self
  std -> self
    - self
    - self
    - long dim
    - CONSTANT false
  std -> self
    - self
    - THTensor source
    - long dim
    - CONSTANT false
]]

[[
  norm
  normall -> accreal
    - self
    - real p
  norm -> self
    - self
    - self
    - real p
    - long dim
  norm -> self
    - self
    - THTensor source
    - real p
    - long dim
]]

[[
  cinv
  cinv -> self
    - self
    - self
  cinv -> self
    - self
    - THTensor source
]]

[[
  neg
  neg -> self
    - self
    - self
  neg -> self
    - self
    - THTensor source
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
    - THTensor source
]]

[[
  cmax
  cmax -> self
    - self
    - self
    - THTensor a
  cmax -> self
    - self
    - THTensor a
    - THTensor b
  cmaxValue -> self
    - self
    - THTensor b
    - real value
  cmaxValue -> self
    - self
    - self
    - real value
]]

[[
  cmin
  cmin -> self
    - self
    - self
    - THTensor a
  cmin -> self
    - self
    - THTensor a
    - THTensor b
  cminValue -> self
    - self
    - THTensor b
    - real value
  cminValue -> self
    - self
    - self
    - real value
]]

[[
  sum
  sumall -> accreal
    - self
  sum -> self
    - self
    - self
    - long dim
  sum -> self
    - self
    - THTensor source
    - long dim
]]

[[
  prod
  prodall -> accreal
    - self
  prod -> self
    - self
    - self
    - long dim
  prod -> self
    - self
    - THTensor source
    - long dim
]]

[[
  cumsum
  cumsum -> self
    - self
    - THTensor source
    - long dim
  cumsum -> self
    - self
    - self
    - long dim
]]

[[
  cumprod
  cumprod -> self
    - self
    - THTensor source
    - long dim
  cumprod -> self
    - self
    - self
    - long dim
]]

[[
  sign
  sign -> self
    - self
    - THTensor source
  sign -> self
    - self
    - self
]]

[[
  trace
  trace -> accreal
    - self
]]

[[
  add
  add -> self
    - self
    - self
    - real value
  add -> self
    - self
    - THTensor a
    - real value
  cadd -> self
    - self
    - self
    - CONSTANT 1
    - THTensor a
  cadd -> self
    - self
    - self
    - real value
    - THTensor a
  cadd -> self
    - self
    - THTensor a
    - CONSTANT 1
    - THTensor b
  cadd -> self
    - self
    - THTensor a
    - real value
    - THTensor b
]]

[[
  csub
  sub -> self
    - self
    - self
    - real value
  sub -> self
    - self
    - THTensor a
    - real value
  csub -> self
    - self
    - self
    - CONSTANT 1
    - THTensor a
  csub -> self
    - self
    - self
    - real value
    - THTensor a
  csub -> self
    - self
    - THTensor a
    - CONSTANT 1
    - THTensor b
  csub -> self
    - self
    - THTensor a
    - real value
    - THTensor b
]]

[[
  mul
  mul -> self
    - self
    - self
    - real value
  mul -> self
    - self
    - THTensor a
    - real value
]]

[[
  cmul
  cmul -> self
    - self
    - self
    - THTensor a
  cmul -> self
    - self
    - THTensor a
    - THTensor b
]]

[[
  div
  div -> self
    - self
    - self
    - real value
  div -> self
    - self
    - THTensor a
    - real value
]]

[[
  cdiv
  cdiv -> self
    - self
    - self
    - THTensor a
  cdiv -> self
    - self
    - THTensor a
    - THTensor b
]]

[[
  fmod
  fmod -> self
    - self
    - self
    - real value
  fmod -> self
    - self
    - THTensor source
    - real value
]]

[[
  cfmod
  cfmod -> self
    - self
    - self
    - THTensor div
  cfmod -> self
    - self
    - THTensor source
    - THTensor div
]]

[[
  remainder
  remainder -> self
    - self
    - self
    - real value
  remainder -> self
    - self
    - THTensor source
    - real value
]]

[[
  cremainder
  cremainder -> self
    - self
    - self
    - THTensor div
  cremainder -> self
    - self
    - THTensor source
    - THTensor div
]]

// TODO: why pow isn't always available
[[
  cpow
  cpow -> self
    - self
    - self
    - THTensor pow
  cpow -> self
    - self
    - THTensor source
    - THTensor pow
]]

[[
  clamp
  clamp -> self
    - self
    - self
    - real min
    - real max
  clamp -> self
    - self
    - THTensor source
    - real min
    - real max
]]

[[
  dot
  dot -> self
    - self
    - THTensor a
  dot -> self
    - THTensor a
    - THTensor b
]]

[[
  equal
  equal -> bool
    - self
    - THTensor other
]]

[[
  tril
  tril -> self
    - self
    - self
    - long k
  tril -> self
    - self
    - THTensor source
    - long k
]]

[[
  triu
  triu -> self
    - self
    - self
    - long k
  triu -> self
    - self
    - THTensor source
    - long k
]]

[[
  eye
  eye -> self
    - self
    - long n
    - long n
  eye -> self
    - self
    - long n
    - long m
]]

[[
  diag
  diag -> self
    - self
    - self
    - CONSTANT 0
  diag -> self
    - self
    - THTensor other
    - CONSTANT 0
  diag -> self
    - self
    - self
    - long k
  diag -> self
    - self
    - THTensor other
    - long k
]]

// TODO: fmod, reminder, clamp

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
  {"norm",            (PyCFunction)THPTensor_(norm),            METH_VARARGS, NULL},
  {"cinv",            (PyCFunction)THPTensor_(cinv),            METH_VARARGS, NULL},
  {"neg",             (PyCFunction)THPTensor_(neg),             METH_VARARGS, NULL},
#endif
  {"add",             (PyCFunction)THPTensor_(add),             METH_VARARGS, NULL},
  {"csub",            (PyCFunction)THPTensor_(csub),            METH_VARARGS, NULL},
  {"mul",             (PyCFunction)THPTensor_(mul),             METH_VARARGS, NULL},
  {"div",             (PyCFunction)THPTensor_(div),             METH_VARARGS, NULL},
  {"fmod",            (PyCFunction)THPTensor_(fmod),            METH_VARARGS, NULL},
  {"mod",             (PyCFunction)THPTensor_(fmod),            METH_VARARGS, NULL},
  {"cmul",            (PyCFunction)THPTensor_(cmul),            METH_VARARGS, NULL},
  {"cdiv",            (PyCFunction)THPTensor_(cdiv),            METH_VARARGS, NULL},
  {"cfmod",           (PyCFunction)THPTensor_(cfmod),           METH_VARARGS, NULL},
  {"cmod",            (PyCFunction)THPTensor_(cfmod),           METH_VARARGS, NULL},
  {"cmax",            (PyCFunction)THPTensor_(cmax),            METH_VARARGS, NULL},
  {"cmin",            (PyCFunction)THPTensor_(cmin),            METH_VARARGS, NULL},
  {"cpow",            (PyCFunction)THPTensor_(cpow),            METH_VARARGS, NULL},
  {"dot",             (PyCFunction)THPTensor_(dot),             METH_VARARGS, NULL},
  {"sum",             (PyCFunction)THPTensor_(sum),             METH_VARARGS, NULL},
  {"prod",            (PyCFunction)THPTensor_(prod),            METH_VARARGS, NULL},
  {"remainder",       (PyCFunction)THPTensor_(remainder),       METH_VARARGS, NULL},
  {"cremainder",      (PyCFunction)THPTensor_(cremainder),      METH_VARARGS, NULL},
  {"cumsum",          (PyCFunction)THPTensor_(cumsum),          METH_VARARGS, NULL},
  {"cumprod",         (PyCFunction)THPTensor_(cumprod),         METH_VARARGS, NULL},
  {"clamp",           (PyCFunction)THPTensor_(clamp),           METH_VARARGS, NULL},
  {"equal",           (PyCFunction)THPTensor_(equal),           METH_VARARGS, NULL},
  {"eye",             (PyCFunction)THPTensor_(eye),             METH_VARARGS, NULL},
  {"elementSize",     (PyCFunction)THPTensor_(elementSize),     METH_NOARGS,  NULL},
  {"fill",            (PyCFunction)THPTensor_(fill),            METH_VARARGS, NULL},
  {"free",            (PyCFunction)THPTensor_(free),            METH_NOARGS,  NULL},
  {"dim",             (PyCFunction)THPTensor_(nDimension),      METH_NOARGS,  NULL},
  {"diag",            (PyCFunction)THPTensor_(diag),            METH_NOARGS,  NULL},
  {"copy",            (PyCFunction)THPTensor_(copy),            METH_O,       NULL},
  {"isSameSizeAs",    (PyCFunction)THPTensor_(isSameSizeAs),    METH_VARARGS, NULL},
  {"numel",           (PyCFunction)THPTensor_(numel),           METH_NOARGS,  NULL},
  {"nElement",        (PyCFunction)THPTensor_(numel),           METH_NOARGS,  NULL},
  {"nDimension",      (PyCFunction)THPTensor_(nDimension),      METH_NOARGS,  NULL},
  {"sign",            (PyCFunction)THPTensor_(sign),            METH_VARARGS, NULL},
  {"size",            (PyCFunction)THPTensor_(size),            METH_VARARGS, NULL},
  {"storage",         (PyCFunction)THPTensor_(storage),         METH_NOARGS,  NULL},
  {"storageOffset",   (PyCFunction)THPTensor_(storageOffset),   METH_NOARGS,  NULL},
  {"stride",          (PyCFunction)THPTensor_(stride),          METH_VARARGS, NULL},
  {"retain",          (PyCFunction)THPTensor_(retain),          METH_NOARGS,  NULL},
  {"trace",           (PyCFunction)THPTensor_(trace),           METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPTensor_(tril),            METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPTensor_(triu),            METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPTensor_(zero),            METH_NOARGS,  NULL},
  {NULL}
};
