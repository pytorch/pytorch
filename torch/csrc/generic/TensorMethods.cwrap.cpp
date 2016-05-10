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

SIMPLE_OP(elementSize,      PyLong_FromLong(THStorage_(elementSize)()))
SIMPLE_OP(storage,          THPStorage_(newObject)(THTensor_(storage)(self->cdata)))
SIMPLE_OP(storageOffset,    PyLong_FromLong(THTensor_(storageOffset)(self->cdata)))
SIMPLE_OP(nDimension,       PyLong_FromLong(THTensor_(nDimension)(self->cdata)))

SIMPLE_RETURN_SELF(free,    THTensor_(free)(self->cdata))
SIMPLE_RETURN_SELF(retain,  THTensor_(retain)(self->cdata))

static PyObject * THPTensor_(select)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  long dim, idx;
  if (!PyArg_ParseTuple(args, "ll", &dim, &idx))
    return NULL;

  int ndim = THTensor_(nDimension)(self->cdata);
  if(ndim > 1) {
    THTensor *selected = THTensor_(newWithTensor)(self->cdata);
    THTensor_(select)(selected, NULL, dim, idx);
    return THPTensor_(newObject)(selected);
  }
  else {
    THArgCheck(ndim == 1, 1, "empty Tensor");
    return THPUtils_(newReal)(THTensor_(get1d)(self->cdata, idx));
  }
  END_HANDLE_TH_ERRORS
}

[[
  numel
  numel -> long
    - self
]]

#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
[[
  abs
  abs -> self
    - self
    - self
  abs -> self OPTIONAL_SELF
    - self
    - THTensor source
]]
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
[[
  sigmoid
  sigmoid -> self
    - self
    - self
  sigmoid -> self OPTIONAL_SELF
    - self
    - THTensor source
]]

[[
  log
  log -> self
    - self
    - self
  log -> self OPTIONAL_SELF
    - self
    - THTensor source
]]


[[
  log1p
  log1p -> self
    - self
    - self
  log1p -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  exp
  exp -> self
    - self
    - self
  exp -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  cos
  cos -> self
    - self
    - self
  cos -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  acos
  acos -> self
    - self
    - self
  acos -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  cosh
  cosh -> self
    - self
    - self
  cosh -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  sin
  sin -> self
    - self
    - self
  sin -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  asin
  asin -> self
    - self
    - self
  asin -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  sinh
  sinh -> self
    - self
    - self
  sinh -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  tan
  tan -> self
    - self
    - self
  tan -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  atan
  atan -> self
    - self
    - self
  atan -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  tanh
  tanh -> self
    - self
    - self
  tanh -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  sqrt
  sqrt -> self
    - self
    - self
  sqrt -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  rsqrt
  rsqrt -> self
    - self
    - self
  rsqrt -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  ceil
  ceil -> self
    - self
    - self
  ceil -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  floor
  floor -> self
    - self
    - self
  floor -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  round
  round -> self
    - self
    - self
  round -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  abs
  abs -> self
    - self
    - self
  abs -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  trunc
  trunc -> self
    - self
    - self
  trunc -> self OPTIONAL_SELF
    - self
    - THTensor other
]]


[[
  frac
  frac -> self
    - self
    - self
  frac -> self OPTIONAL_SELF
    - self
    - THTensor other
]]

[[
  mean
  meanall -> accreal
    - self
  mean -> self
    - self
    - self
    - long dim
  mean -> self OPTIONAL_SELF
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
  var -> self OPTIONAL_SELF
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
  std -> self OPTIONAL_SELF
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
  norm -> self OPTIONAL_SELF
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
  cinv -> self OPTIONAL_SELF
    - self
    - THTensor source
]]

[[
  neg
  neg -> self
    - self
    - self
  neg -> self OPTIONAL_SELF
    - self
    - THTensor source
]]
#endif

[[
  zero
  zero -> self
    - self
]]

[[
  size STATEFUL_ONLY
  size -> long
    - self
    - long dim
  newSizeOf -> THLongStorage
    - self
]]

[[
  stride STATEFUL_ONLY
  stride -> long
    - self
    - long dim
  newStrideOf -> THLongStorage
    - self
]]

[[
  fill
  fill -> self
    - self
    - real value
]]

[[
  isSameSizeAs STATEFUL_ONLY
  isSameSizeAs -> bool
    - self
    - THTensor other
]]

[[
  cmax
  cmax -> self
    - self
    - self
    - THTensor a
  cmax -> self OPTIONAL_SELF
    - self
    - THTensor a
    - THTensor b
  cmaxValue -> self OPTIONAL_SELF
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
  cmin -> self OPTIONAL_SELF
    - self
    - THTensor a
    - THTensor b
  cminValue -> self OPTIONAL_SELF
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
  sum -> self OPTIONAL_SELF
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
  prod -> self OPTIONAL_SELF
    - self
    - THTensor source
    - long dim
]]

[[
  cumsum
  cumsum -> self OPTIONAL_SELF
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
  cumprod -> self OPTIONAL_SELF
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
  sign -> self OPTIONAL_SELF
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
  add -> self OPTIONAL_SELF
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
  cadd -> self OPTIONAL_SELF
    - self
    - THTensor a
    - CONSTANT 1
    - THTensor b
  cadd -> self OPTIONAL_SELF
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
  sub -> self OPTIONAL_SELF
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
  csub -> self OPTIONAL_SELF
    - self
    - THTensor a
    - CONSTANT 1
    - THTensor b
  csub -> self OPTIONAL_SELF
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
  mul -> self OPTIONAL_SELF
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
  cmul -> self OPTIONAL_SELF
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
  div -> self OPTIONAL_SELF
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
  cdiv -> self OPTIONAL_SELF
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
  fmod -> self OPTIONAL_SELF
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
  cfmod -> self OPTIONAL_SELF
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
  remainder -> self OPTIONAL_SELF
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
  cremainder -> self OPTIONAL_SELF
    - self
    - THTensor source
    - THTensor div
]]

[[
  cpow
  cpow -> self
    - self
    - self
    - THTensor pow
  cpow -> self OPTIONAL_SELF
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
  clamp -> self OPTIONAL_SELF
    - self
    - THTensor source
    - real min
    - real max
]]

[[
  dot
  dot -> accreal
    - self
    - THTensor a
  dot -> accreal
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
  tril -> self OPTIONAL_SELF
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
  triu -> self OPTIONAL_SELF
    - self
    - THTensor source
    - long k
]]

[[
  eye
  eye -> self OPTIONAL_SELF
    - self
    - long n
    - long n
  eye -> self OPTIONAL_SELF
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
  diag -> self OPTIONAL_SELF
    - self
    - THTensor other
    - CONSTANT 0
  diag -> self
    - self
    - self
    - long k
  diag -> self OPTIONAL_SELF
    - self
    - THTensor other
    - long k
]]

// Declared in TensorCopy.cpp
static PyObject * THPTensor_(copy)(THPTensor *self, PyObject *other);

static PyMethodDef THPTensor_(methods)[] = {
  //////////////////////////////////////////////////////////////////////////////
  // These methods are stateful only
  {"elementSize",     (PyCFunction)THPTensor_(elementSize),     METH_NOARGS,  NULL},
  {"isSameSizeAs",    (PyCFunction)THPTensor_(isSameSizeAs),    METH_VARARGS, NULL},
  {"dim",             (PyCFunction)THPTensor_(nDimension),      METH_NOARGS,  NULL},
  {"stride",          (PyCFunction)THPTensor_(stride),          METH_VARARGS, NULL},
  {"storage",         (PyCFunction)THPTensor_(storage),         METH_NOARGS,  NULL},
  {"storageOffset",   (PyCFunction)THPTensor_(storageOffset),   METH_NOARGS,  NULL},
  {"nElement",        (PyCFunction)THPTensor_(numel),           METH_NOARGS,  NULL},
  {"nDimension",      (PyCFunction)THPTensor_(nDimension),      METH_NOARGS,  NULL},
  {"copy",            (PyCFunction)THPTensor_(copy),            METH_O,       NULL},
  {"free",            (PyCFunction)THPTensor_(free),            METH_NOARGS,  NULL},
  {"retain",          (PyCFunction)THPTensor_(retain),          METH_NOARGS,  NULL},
  {"size",            (PyCFunction)THPTensor_(size),            METH_VARARGS, NULL},
  {"select",          (PyCFunction)THPTensor_(select),          METH_VARARGS, NULL},
  //////////////////////////////////////////////////////////////////////////////
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
  {"fill",            (PyCFunction)THPTensor_(fill),            METH_VARARGS, NULL},
  {"diag",            (PyCFunction)THPTensor_(diag),            METH_VARARGS,  NULL},
  {"numel",           (PyCFunction)THPTensor_(numel),           METH_VARARGS,  NULL},
  {"sign",            (PyCFunction)THPTensor_(sign),            METH_VARARGS, NULL},
  {"trace",           (PyCFunction)THPTensor_(trace),           METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPTensor_(tril),            METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPTensor_(triu),            METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPTensor_(zero),            METH_VARARGS,  NULL},
  {NULL}
};

static PyMethodDef THPTensorStatelessMethods[] = {
#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
  {"abs",             (PyCFunction)THPTensor_stateless_(abs),             METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"sigmoid",         (PyCFunction)THPTensor_stateless_(sigmoid),         METH_VARARGS, NULL},
  {"log",             (PyCFunction)THPTensor_stateless_(log),             METH_VARARGS, NULL},
  {"log1p",           (PyCFunction)THPTensor_stateless_(log1p),           METH_VARARGS, NULL},
  {"exp",             (PyCFunction)THPTensor_stateless_(exp),             METH_VARARGS, NULL},
  {"cos",             (PyCFunction)THPTensor_stateless_(cos),             METH_VARARGS, NULL},
  {"acos",            (PyCFunction)THPTensor_stateless_(acos),            METH_VARARGS, NULL},
  {"cosh",            (PyCFunction)THPTensor_stateless_(cosh),            METH_VARARGS, NULL},
  {"sin",             (PyCFunction)THPTensor_stateless_(sin),             METH_VARARGS, NULL},
  {"asin",            (PyCFunction)THPTensor_stateless_(asin),            METH_VARARGS, NULL},
  {"sinh",            (PyCFunction)THPTensor_stateless_(sinh),            METH_VARARGS, NULL},
  {"tan",             (PyCFunction)THPTensor_stateless_(tan),             METH_VARARGS, NULL},
  {"atan",            (PyCFunction)THPTensor_stateless_(atan),            METH_VARARGS, NULL},
  {"tanh",            (PyCFunction)THPTensor_stateless_(tanh),            METH_VARARGS, NULL},
  {"sqrt",            (PyCFunction)THPTensor_stateless_(sqrt),            METH_VARARGS, NULL},
  {"rsqrt",           (PyCFunction)THPTensor_stateless_(rsqrt),           METH_VARARGS, NULL},
  {"ceil",            (PyCFunction)THPTensor_stateless_(ceil),            METH_VARARGS, NULL},
  {"floor",           (PyCFunction)THPTensor_stateless_(floor),           METH_VARARGS, NULL},
  {"round",           (PyCFunction)THPTensor_stateless_(round),           METH_VARARGS, NULL},
  {"abs",             (PyCFunction)THPTensor_stateless_(abs),             METH_VARARGS, NULL},
  {"trunc",           (PyCFunction)THPTensor_stateless_(trunc),           METH_VARARGS, NULL},
  {"frac",            (PyCFunction)THPTensor_stateless_(frac),            METH_VARARGS, NULL},
  {"mean",            (PyCFunction)THPTensor_stateless_(mean),            METH_VARARGS, NULL},
  {"std",             (PyCFunction)THPTensor_stateless_(std),             METH_VARARGS, NULL},
  {"var",             (PyCFunction)THPTensor_stateless_(var),             METH_VARARGS, NULL},
  {"norm",            (PyCFunction)THPTensor_stateless_(norm),            METH_VARARGS, NULL},
  {"cinv",            (PyCFunction)THPTensor_stateless_(cinv),            METH_VARARGS, NULL},
  {"neg",             (PyCFunction)THPTensor_stateless_(neg),             METH_VARARGS, NULL},
#endif
  {"add",             (PyCFunction)THPTensor_stateless_(add),             METH_VARARGS, NULL},
  {"csub",            (PyCFunction)THPTensor_stateless_(csub),            METH_VARARGS, NULL},
  {"mul",             (PyCFunction)THPTensor_stateless_(mul),             METH_VARARGS, NULL},
  {"div",             (PyCFunction)THPTensor_stateless_(div),             METH_VARARGS, NULL},
  {"fmod",            (PyCFunction)THPTensor_stateless_(fmod),            METH_VARARGS, NULL},
  {"mod",             (PyCFunction)THPTensor_stateless_(fmod),            METH_VARARGS, NULL},
  {"cmul",            (PyCFunction)THPTensor_stateless_(cmul),            METH_VARARGS, NULL},
  {"cdiv",            (PyCFunction)THPTensor_stateless_(cdiv),            METH_VARARGS, NULL},
  {"cfmod",           (PyCFunction)THPTensor_stateless_(cfmod),           METH_VARARGS, NULL},
  {"cmod",            (PyCFunction)THPTensor_stateless_(cfmod),           METH_VARARGS, NULL},
  {"cmax",            (PyCFunction)THPTensor_stateless_(cmax),            METH_VARARGS, NULL},
  {"cmin",            (PyCFunction)THPTensor_stateless_(cmin),            METH_VARARGS, NULL},
  {"cpow",            (PyCFunction)THPTensor_stateless_(cpow),            METH_VARARGS, NULL},
  {"dot",             (PyCFunction)THPTensor_stateless_(dot),             METH_VARARGS, NULL},
  {"sum",             (PyCFunction)THPTensor_stateless_(sum),             METH_VARARGS, NULL},
  {"prod",            (PyCFunction)THPTensor_stateless_(prod),            METH_VARARGS, NULL},
  {"remainder",       (PyCFunction)THPTensor_stateless_(remainder),       METH_VARARGS, NULL},
  {"cremainder",      (PyCFunction)THPTensor_stateless_(cremainder),      METH_VARARGS, NULL},
  {"cumsum",          (PyCFunction)THPTensor_stateless_(cumsum),          METH_VARARGS, NULL},
  {"cumprod",         (PyCFunction)THPTensor_stateless_(cumprod),         METH_VARARGS, NULL},
  {"clamp",           (PyCFunction)THPTensor_stateless_(clamp),           METH_VARARGS, NULL},
  {"equal",           (PyCFunction)THPTensor_stateless_(equal),           METH_VARARGS, NULL},
  {"eye",             (PyCFunction)THPTensor_stateless_(eye),             METH_VARARGS, NULL},
  {"fill",            (PyCFunction)THPTensor_stateless_(fill),            METH_VARARGS, NULL},
  {"diag",            (PyCFunction)THPTensor_stateless_(diag),            METH_VARARGS,  NULL},
  {"numel",           (PyCFunction)THPTensor_stateless_(numel),           METH_VARARGS,  NULL},
  {"sign",            (PyCFunction)THPTensor_stateless_(sign),            METH_VARARGS, NULL},
  {"trace",           (PyCFunction)THPTensor_stateless_(trace),           METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPTensor_stateless_(tril),            METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPTensor_stateless_(triu),            METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPTensor_stateless_(zero),            METH_VARARGS,  NULL},
  {NULL}
};
