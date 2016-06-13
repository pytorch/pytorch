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

static PyObject * THPTensor_(resize)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  THLongStoragePtr size = THPUtils_getLongStorage(args);
  THTensor_(resize)(self->cdata, size, NULL);

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static void THPTensor_(doReshape)(THTensor *result, THTensor *src,
        PyObject *args, int indices_offset)
{
  THLongStoragePtr size = THPUtils_getLongStorage(args, indices_offset);
  THTensor_(reshape)(result, src, size);
}

static PyObject * THPTensor_(reshape)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_Size(args) == 0) {
    THPUtils_setError("reshape requires at least one argument");
    return NULL;
  }
  PyObject *first_arg = PyTuple_GET_ITEM(args, 0);
  THPTensorPtr returned;

  // TODO: this behaviour is quite weird...
  // m.reshape(x, 2, 2) will alter m with x elements :/
  if (!THPTensor_(IsSubclass)(first_arg)) {
    THTensorPtr _ret = THTensor_(new)();
    returned = (THPTensor*)THPTensor_(newObject)(_ret);
    _ret.release();
    THPTensor_(doReshape)(returned->cdata, self->cdata, args, 0);
  } else {
    Py_INCREF(self);
    returned = self;
    THPTensor_(doReshape)(self->cdata, ((THPTensor*)first_arg)->cdata, args, 1);
  }

  return (PyObject *)returned.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_stateless_(reshape)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_Size(args) < 2) {
    THPUtils_setError("reshape requires at least two arguments");
    return NULL;
  }
  THPTensor *first_arg = (THPTensor*)PyTuple_GET_ITEM(args, 0);
  PyObject *second_arg = PyTuple_GET_ITEM(args, 1);
  THPTensorPtr returned;
  if (!THPTensor_(IsSubclass)((PyObject*)first_arg)) {
    THPUtils_setError("reshape requires it's first argument to be a Tensor");
    return NULL;
  }

  if (!THPTensor_(IsSubclass)(second_arg)) {
    THTensorPtr _ret = THTensor_(new)();
    returned = (THPTensor*)THPTensor_(newObject)(_ret);
    _ret.release();
    THPTensor_(doReshape)(returned->cdata, first_arg->cdata, args, 1);
  } else {
    Py_INCREF(first_arg);
    returned = first_arg;
    THPTensor_(doReshape)(returned->cdata, ((THPTensor*)second_arg)->cdata, args, 2);
  }

  return (PyObject *)returned.release();
  END_HANDLE_TH_ERRORS
}

#define IMPLEMENT_FILLER(NAME)                                                 \
static PyObject * THPTensor_(NAME)(THPTensor *self, PyObject *args)            \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  THLongStorage *size = THPUtils_getLongStorage(args, 0);                      \
  try {                                                                        \
    THTensor_(NAME)(self->cdata, size);                                        \
  } catch(...) {                                                               \
    THLongStorage_free(size);                                                  \
    throw;                                                                     \
  }                                                                            \
  THLongStorage_free(size);                                                    \
                                                                               \
  Py_INCREF(self);                                                             \
  return (PyObject*)self;                                                      \
  END_HANDLE_TH_ERRORS                                                         \
}

#define IMPLEMENT_STATELESS_FILLER(NAME)                                       \
static PyObject * THPTensor_stateless_(NAME)(PyObject *_unused, PyObject *args)\
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  if (PyTuple_Size(args) == 0) {                                               \
    THPUtils_setError(#NAME " requires at least one argument");                \
    return NULL;                                                               \
  }                                                                            \
  PyObject *first_arg = PyTuple_GET_ITEM(args, 0);                             \
  THPTensor *returned;                                                         \
  THLongStorage *size;                                                         \
  if (THPTensor_(IsSubclass)(first_arg)) {                                     \
    Py_INCREF(first_arg);                                                      \
    returned = (THPTensor*)first_arg;                                          \
    size = THPUtils_getLongStorage(args, 1);                                   \
  } else {                                                                     \
    THTensor *_ret = THTensor_(new)();                                         \
    returned = (THPTensor*)THPTensor_(newObject)(_ret);                        \
    size = THPUtils_getLongStorage(args, 0);                                   \
  }                                                                            \
  try {                                                                        \
    THTensor_(NAME)(returned->cdata, size);                                    \
  } catch(...) {                                                               \
    THLongStorage_free(size);                                                  \
    Py_DECREF(returned);                                                       \
    throw;                                                                     \
  }                                                                            \
  THLongStorage_free(size);                                                    \
                                                                               \
  return (PyObject*)returned;                                                  \
  END_HANDLE_TH_ERRORS                                                         \
}

IMPLEMENT_FILLER(zeros)
IMPLEMENT_FILLER(ones)
IMPLEMENT_STATELESS_FILLER(zeros)
IMPLEMENT_STATELESS_FILLER(ones)

static PyObject * THPTensor_(set)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = PyTuple_Size(args);
  PyObject *first_arg = num_args == 0 ? NULL : PyTuple_GET_ITEM(args, 0);

  if (num_args == 1 && THPTensor_(IsSubclass)(first_arg)) {
    THTensor_(set)(self->cdata, ((THPTensor*)first_arg)->cdata);
    Py_INCREF(self);
    return (PyObject*)self;

  } else if (num_args <= 4 && THPStorage_(IsSubclass)(first_arg)) {
    THLongStoragePtr sizes, strides;
    THPStorage *storage = (THPStorage*)first_arg;
    long storageOffset = 0;

    if (num_args >= 2 && !THPUtils_getLong(PyTuple_GET_ITEM(args, 1), &storageOffset))
      return NULL;

    if (num_args >= 3) {
      PyObject *third_arg = PyTuple_GET_ITEM(args, 2);
      THPUtils_assert(THPLongStorage_IsSubclass(third_arg), "set expects a LongStorage as its third argument");
      sizes = ((THPLongStorage*)third_arg)->cdata;
      THLongStorage_retain(sizes);
    } else {
      sizes = THLongStorage_newWithSize1(THStorage_(size)(storage->cdata));
    }

    if (num_args >= 4) {
      PyObject *fourth_arg = PyTuple_GET_ITEM(args, 2);
      THPUtils_assert(THPLongStorage_IsSubclass(fourth_arg), "set expects a LongStorage as its third argument");
      strides = ((THPLongStorage*)fourth_arg)->cdata;
      THLongStorage_retain(strides);
    }

    THTensor_(setStorage)(self->cdata, storage->cdata, storageOffset, sizes, strides);
    Py_INCREF(self);
    return (PyObject*)self;
  }

  // TODO: Inform about possible arg configurations
  return NULL;
  END_HANDLE_TH_ERRORS
}

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

#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
#define BUILD_REAL_FMT "d"
#else
#define BUILD_REAL_FMT "L"
#endif

static PyObject * THPTensor_(apply)(THPTensor *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  if (!PyCallable_Check(arg)) {
    THPUtils_setError("apply requires a callable as it's first argument");
    return NULL;
  }

  real v;
  THTensor *tensor = self->cdata;
  TH_TENSOR_APPLY(real, tensor,
                  PyObject *ret =
                      PyObject_CallFunction(arg, BUILD_REAL_FMT, *tensor_data);
                  if (!ret)
                    return NULL;
                  bool success = THPUtils_(parseReal)(ret, &v);
                  Py_DECREF(ret);
                  if (!success)
                    THError("given function should return a number");
                  *tensor_data = v;
                  );

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(map)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
	PyObject *fn;
	THPTensor *src_object;
	if (!PyArg_ParseTuple(args, "O!O&", &THPTensorType, &src_object, THPUtils_getCallable, &fn))
	  return NULL;

  real v;
  THTensor *tensor = self->cdata;
  THTensor *src = src_object->cdata;
  TH_TENSOR_APPLY2(real, tensor, real, src,
                  PyObject *ret =
                      PyObject_CallFunction(fn, BUILD_REAL_FMT BUILD_REAL_FMT,
                                            *tensor_data, *src_data);
                  if (!ret)
                    return NULL;
                  bool success = THPUtils_(parseReal)(ret, &v);
                  Py_DECREF(ret);
                  if (!success)
                    THError("given function should return a number");
                  *tensor_data = v;
                  );

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(map2)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
	PyObject *fn;
	THPTensor *src1_object;
	THPTensor *src2_object;
	if (!PyArg_ParseTuple(args, "O!O!O&", &THPTensorType, &src1_object, &THPTensorType, &src2_object, THPUtils_getCallable, &fn))
	  return NULL;

  real v;
  THTensor *tensor = self->cdata;
  THTensor *src1 = src1_object->cdata;
  THTensor *src2 = src2_object->cdata;
  TH_TENSOR_APPLY3(real, tensor, real, src1, real, src2,
                  PyObject *ret =
                      PyObject_CallFunction(fn, BUILD_REAL_FMT BUILD_REAL_FMT BUILD_REAL_FMT,
                                            *tensor_data, *src1_data, *src2_data);
                  if (!ret)
                    return NULL;
                  bool success = THPUtils_(parseReal)(ret, &v);
                  Py_DECREF(ret);
                  if (!success)
                    THError("given function should return a number");
                  *tensor_data = v;
                  );

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

#undef BUILD_REAL_FMT

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
  renorm
  renorm -> self
    - self
    - self
    - real p
    - long dim
    - real maxnorm
  renorm -> self OPTIONAL_SELF
    - self
    - THTensor source
    - real p
    - long dim
    - real maxnorm
]]

[[
  dist
  dist -> accreal
    - self
    - THTensor a
    - CONSTANT 2
  dist -> accreal
    - self
    - THTensor a
    - real p
]]

[[
  linspace
  linspace -> self OPTIONAL_SELF
    - self
    - real start
    - real end
    - CONSTANT 100
  linspace -> self OPTIONAL_SELF
    - self
    - real start
    - real end
    - long steps
]]

[[
  logspace
  logspace -> self OPTIONAL_SELF
    - self
    - real start
    - real end
    - CONSTANT 100
  logspace -> self OPTIONAL_SELF
    - self
    - real start
    - real end
    - long steps
]]

[[
  histc
  histc -> self
    - self
    - THTensor src
    - CONSTANT 100
    - CONSTANT 0
    - CONSTANT 0
  histc -> self
    - self
    - THTensor src
    - long bins
    - CONSTANT 0
    - CONSTANT 0
  histc -> self
    - self
    - THTensor src
    - long bins
    - real min
    - CONSTANT 0
  histc -> self
    - self
    - THTensor src
    - long bins
    - real min
    - real max
  histc -> new THTensor
    - self
    - CONSTANT 100
    - CONSTANT 0
    - CONSTANT 0
  histc -> new THTensor
    - self
    - long bins
    - CONSTANT 0
    - CONSTANT 0
  histc -> new THTensor
    - self
    - long bins
    - real min
    - CONSTANT 0
  histc -> new THTensor
    - self
    - long bins
    - real min
    - real max
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

[[
  atan2 STATEFUL_ONLY
  atan2 -> self
    - self
    - self
    - THTensor other
  atan2 -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  atan2 STATELESS_ONLY
  atan2 -> self OPTIONAL_SELF
    - self
    - THTensor other
    - THTensor other2
  atan2 -> real PLAIN_CALL
    - real a
    - real b
]]

[[
  pow STATEFUL_ONLY
  pow -> self
    - self
    - self
    - real power
  pow -> self
    - self
    - THTensor source
    - real power
  tpow -> self
    - self
    - real value
    - THTensor source
]]

[[
  pow STATELESS_ONLY
  pow -> self OPTIONAL_SELF
    - self
    - THTensor source
    - real power
  tpow -> self OPTIONAL_SELF
    - self
    - real value
    - THTensor source
  pow -> real PLAIN_CALL
    - real value
    - real power
]]

[[
  lerp STATEFUL_ONLY
  lerp -> self
    - self
    - self
    - THTensor a
    - real weight
  lerp -> self
    - self
    - THTensor a
    - THTensor b
    - real weight
]]

[[
  lerp STATELESS_ONLY
  lerp -> self OPTIONAL_SELF
    - self
    - THTensor a
    - THTensor b
    - real weight
  TH_lerp -> real PLAIN_CALL
    - real start
    - real end
    - real weight
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
  isContiguous STATEFUL_ONLY
  isContiguous -> bool
    - self
]]

[[
  isSetTo STATEFUL_ONLY
  isSetTo -> bool
    - self
    - THTensor other
]]

[[
  isSize STATEFUL_ONLY
  isSize -> bool
    - self
    - THLongStorage other
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

[[
  lt
  ltValue -> new THByteTensor
    - self
    - real value
  ltValueT -> self
    - self
    - THTensor other
    - real value
  ltTensor -> new THByteTensor
    - self
    - THTensor other
  ltTensorT -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  gt
  gtValue -> new THByteTensor
    - self
    - real value
  gtValueT -> self
    - self
    - THTensor other
    - real value
  gtTensor -> new THByteTensor
    - self
    - THTensor other
  gtTensorT -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  le
  leValue -> new THByteTensor
    - self
    - real value
  leValueT -> self
    - self
    - THTensor other
    - real value
  leTensor -> new THByteTensor
    - self
    - THTensor other
  leTensorT -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  ge
  geValue -> new THByteTensor
    - self
    - real value
  geValueT -> self
    - self
    - THTensor other
    - real value
  geTensor -> new THByteTensor
    - self
    - THTensor other
  geTensorT -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  eq
  eqValue -> new THByteTensor
    - self
    - real value
  eqValueT -> self
    - self
    - THTensor other
    - real value
  eqTensor -> new THByteTensor
    - self
    - THTensor other
  eqTensorT -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  ne
  neValue -> new THByteTensor
    - self
    - real value
  neValueT -> self
    - self
    - THTensor other
    - real value
  neTensor -> new THByteTensor
    - self
    - THTensor other
  neTensorT -> self
    - self
    - THTensor other
    - THTensor other2
]]


[[
  min
  minall -> real
    - self
  min -> new ValueIndexPair
    - self
    - long index
  min -> new SelfIndexPair
    - THTensor other
    - long index
]]

[[
  max
  maxall -> real
    - self
  max -> new ValueIndexPair
    - self
    - long index
  max -> new SelfIndexPair
    - THTensor other
    - long index
]]

[[
  kthvalue
  kthvalue -> new ValueIndexPair
    - self
    - long k
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
  kthvalue -> new ValueIndexPair
    - self
    - long k
    - long dim
  kthvalue -> new SelfIndexPair
    - THTensor other
    - long k
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
  kthvalue -> new SelfIndexPair
    - THTensor other
    - long k
    - long dim
]]

[[
  mode
  mode -> new ValueIndexPair
    - self
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
  mode -> new ValueIndexPair
    - self
    - long dim
  mode -> new SelfIndexPair
    - THTensor other
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
  mode -> new SelfIndexPair
    - THTensor other
    - long dim
]]

[[
  median
  median -> new ValueIndexPair
    - self
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
  median -> new ValueIndexPair
    - self
    - long dim
  median -> new SelfIndexPair
    - THTensor other
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
  median -> new SelfIndexPair
    - THTensor other
    - long dim
]]

[[
  cross
  cross -> self
    - self
    - self
    - THTensor a
    - CONSTANT 0
  cross -> self
    - self
    - self
    - THTensor a
    - long dim
  cross -> self OPTIONAL_SELF
    - self
    - THTensor a
    - THTensor b
    - CONSTANT 0
  cross -> self OPTIONAL_SELF
    - self
    - THTensor a
    - THTensor b
    - long dim
]]

[[
  sort
  sort -> new ValueIndexPair
    - self
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
    - CONSTANT false
  sort -> new ValueIndexPair
    - self
    - long dim
    - CONSTANT false
  sort -> new ValueIndexPair
    - self
    - long dim
    - bool descending
  sort -> new SelfIndexPair
    - THTensor source
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
    - CONSTANT false
  sort -> new SelfIndexPair
    - THTensor source
    - long dim
    - CONSTANT false
  sort -> new SelfIndexPair
    - THTensor source
    - long dim
    - bool descending
]]

[[
  topk
  topk -> new ValueIndexPair
    - self
    - CONSTANT 1
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new ValueIndexPair
    - self
    - long k
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new ValueIndexPair
    - self
    - long k
    - long dim
    - CONSTANT false
    - CONSTANT false
  topk -> new ValueIndexPair
    - self
    - long k
    - long dim
    - bool smallest
    - CONSTANT false
  topk -> new ValueIndexPair
    - self
    - long k
    - long dim
    - bool smallest
    - bool sorted
  topk -> new SelfIndexPair
    - THTensor source
    - CONSTANT 1
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new SelfIndexPair
    - THTensor source
    - long k
    - EXPRESSION THTensor_(size)({2}->cdata,THTensor_(nDimension)({2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new SelfIndexPair
    - THTensor source
    - long k
    - long dim
    - CONSTANT false
    - CONSTANT false
  topk -> new SelfIndexPair
    - THTensor source
    - long k
    - long dim
    - bool smallest
    - CONSTANT false
  topk -> new SelfIndexPair
    - THTensor source
    - long k
    - long dim
    - bool smallest
    - bool sorted
]]

// TODO: why are these stateful only?
[[
  maskedFill STATEFUL_ONLY
  maskedFill -> self
    - self
    - THByteTensor mask
    - real value
]]

[[
  maskedCopy STATEFUL_ONLY
  maskedCopy -> self
    - self
    - THByteTensor mask
    - THTensor source
]]

[[
  maskedSelect STATEFUL_ONLY
  maskedSelect -> new THTensor
    - self
    - THByteTensor mask
  maskedSelect -> self
    - self
    - THTensor source
    - THByteTensor mask
]]

// TODO: why not inplace?
[[
  t
  newTranspose -> THTensor
    - self
    - long dim0
    - long dim1
]]

[[
  transpose
  newTranspose -> THTensor
    - self
    - CONSTANT 0
    - CONSTANT 1
]]

// TODO: inconsistent with lua - doesn't return a number when one-elemtn tensor
// collapses to one dimension
[[
  squeeze
  squeeze -> self
    - self
    - self
  squeeze -> self OPTIONAL_SELF
    - self
    - THTensor other
  squeeze1d -> self
    - self
    - self
    - long dim
  squeeze1d -> self OPTIONAL_SELF
    - self
    - THTensor other
    - long dim
]]

[[
  nonzero
  nonzero -> new THLongTensor
    - self
  nonzero -> CUSTOM {expr}; Py_INCREF(indices); return (PyObject*)indices
    - THLongTensor indices
    - self
]]

[[
  contiguous
  newContiguous -> THTensor
    - self
]]

[[
  clone
  newClone -> THTensor
    - self
]]

[[
  resizeAs STATEFUL_ONLY
  resizeAs -> self
    - self
    - THTensor other
]]

// TODO: index* methods expect 1-based indexing
// this has to be fixed in TH
[[
  index
  indexSelect -> self
    - self
    - self
    - long dim
    - THLongTensor index
  indexSelect -> self OPTIONAL_SELF
    - self
    - THTensor source
    - long dim
    - THLongTensor index
]]

[[
  indexCopy
  indexCopy -> self
    - self
    - long dim
    - THLongTensor index
    - THTensor source
]]

[[
  indexAdd
  indexAdd -> self
    - self
    - long dim
    - THLongTensor index
    - THTensor source
]]

[[
  indexFill
  indexFill -> self
    - self
    - long dim
    - THLongTensor index
    - real value
]]

// Declared in TensorCopy.cpp
static PyObject * THPTensor_(copy)(THPTensor *self, PyObject *other);

static PyMethodDef THPTensor_(methods)[] = {
  //////////////////////////////////////////////////////////////////////////////
  // These methods are stateful only
  {"elementSize",     (PyCFunction)THPTensor_(elementSize),     METH_NOARGS,  NULL},
  {"isSameSizeAs",    (PyCFunction)THPTensor_(isSameSizeAs),    METH_VARARGS, NULL},
  {"isContiguous",    (PyCFunction)THPTensor_(isContiguous),    METH_VARARGS, NULL},
  {"isSetTo",         (PyCFunction)THPTensor_(isSetTo),         METH_VARARGS, NULL},
  {"isSize",          (PyCFunction)THPTensor_(isSize),          METH_VARARGS, NULL},
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
  {"maskedFill",      (PyCFunction)THPTensor_(maskedFill),      METH_VARARGS, NULL},
  {"maskedCopy",      (PyCFunction)THPTensor_(maskedCopy),      METH_VARARGS, NULL},
  {"maskedSelect",    (PyCFunction)THPTensor_(maskedSelect),    METH_VARARGS, NULL},
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
  {"renorm",          (PyCFunction)THPTensor_(renorm),          METH_VARARGS, NULL},
  {"dist",            (PyCFunction)THPTensor_(dist),            METH_VARARGS, NULL},
  {"linspace",        (PyCFunction)THPTensor_(linspace),        METH_VARARGS, NULL},
  {"logspace",        (PyCFunction)THPTensor_(logspace),        METH_VARARGS, NULL},
  {"histc",           (PyCFunction)THPTensor_(histc),           METH_VARARGS, NULL},
  {"atan2",           (PyCFunction)THPTensor_(atan2),           METH_VARARGS, NULL},
  {"pow",             (PyCFunction)THPTensor_(pow),             METH_VARARGS, NULL},
  {"lerp",            (PyCFunction)THPTensor_(lerp),            METH_VARARGS, NULL},
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
  {"min",             (PyCFunction)THPTensor_(min),             METH_VARARGS, NULL},
  {"max",             (PyCFunction)THPTensor_(max),             METH_VARARGS, NULL},
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
  {"diag",            (PyCFunction)THPTensor_(diag),            METH_VARARGS, NULL},
  {"numel",           (PyCFunction)THPTensor_(numel),           METH_VARARGS, NULL},
  {"sign",            (PyCFunction)THPTensor_(sign),            METH_VARARGS, NULL},
  {"trace",           (PyCFunction)THPTensor_(trace),           METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPTensor_(tril),            METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPTensor_(triu),            METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPTensor_(zero),            METH_VARARGS, NULL},
  {"gt",              (PyCFunction)THPTensor_(gt),              METH_VARARGS, NULL},
  {"lt",              (PyCFunction)THPTensor_(lt),              METH_VARARGS, NULL},
  {"ge",              (PyCFunction)THPTensor_(ge),              METH_VARARGS, NULL},
  {"le",              (PyCFunction)THPTensor_(le),              METH_VARARGS, NULL},
  {"eq",              (PyCFunction)THPTensor_(eq),              METH_VARARGS, NULL},
  {"ne",              (PyCFunction)THPTensor_(ne),              METH_VARARGS, NULL},
  {"kthvalue",        (PyCFunction)THPTensor_(kthvalue),        METH_VARARGS, NULL},
  {"mode",            (PyCFunction)THPTensor_(mode),            METH_VARARGS, NULL},
  {"median",          (PyCFunction)THPTensor_(median),          METH_VARARGS, NULL},
  {"cross",           (PyCFunction)THPTensor_(cross),           METH_VARARGS, NULL},
  {"sort",            (PyCFunction)THPTensor_(sort),            METH_VARARGS, NULL},
  {"topk",            (PyCFunction)THPTensor_(topk),            METH_VARARGS, NULL},
  {"t",               (PyCFunction)THPTensor_(t),               METH_VARARGS, NULL},
  {"transpose",       (PyCFunction)THPTensor_(transpose),       METH_VARARGS, NULL},
  {"squeeze",         (PyCFunction)THPTensor_(squeeze),         METH_VARARGS, NULL},
  {"nonzero",         (PyCFunction)THPTensor_(nonzero),         METH_VARARGS, NULL},
  {"contiguous",      (PyCFunction)THPTensor_(contiguous),      METH_VARARGS, NULL},
  {"clone",           (PyCFunction)THPTensor_(clone),           METH_VARARGS, NULL},
  {"apply",           (PyCFunction)THPTensor_(apply),           METH_O,       NULL},
  {"map",             (PyCFunction)THPTensor_(map),             METH_VARARGS, NULL},
  {"map2",            (PyCFunction)THPTensor_(map2),            METH_VARARGS, NULL},
  {"resize",          (PyCFunction)THPTensor_(resize),          METH_VARARGS, NULL},
  {"resizeAs",        (PyCFunction)THPTensor_(resizeAs),        METH_VARARGS, NULL},
  {"reshape",         (PyCFunction)THPTensor_(reshape),         METH_VARARGS, NULL},
  {"zeros",           (PyCFunction)THPTensor_(zeros),           METH_VARARGS, NULL},
  {"ones",            (PyCFunction)THPTensor_(ones),            METH_VARARGS, NULL},
  {"set",             (PyCFunction)THPTensor_(set),             METH_VARARGS, NULL},
  {"index",           (PyCFunction)THPTensor_(index),           METH_VARARGS, NULL},
  {"indexCopy",       (PyCFunction)THPTensor_(indexCopy),       METH_VARARGS, NULL},
  {"indexAdd",        (PyCFunction)THPTensor_(indexAdd),        METH_VARARGS, NULL},
  {"indexFill",       (PyCFunction)THPTensor_(indexFill),       METH_VARARGS, NULL},
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
  {"renorm",          (PyCFunction)THPTensor_stateless_(renorm),          METH_VARARGS, NULL},
  {"dist",            (PyCFunction)THPTensor_stateless_(dist),            METH_VARARGS, NULL},
  {"linspace",        (PyCFunction)THPTensor_stateless_(linspace),        METH_VARARGS, NULL},
  {"logspace",        (PyCFunction)THPTensor_stateless_(logspace),        METH_VARARGS, NULL},
  {"histc",           (PyCFunction)THPTensor_stateless_(histc),           METH_VARARGS, NULL},
  {"atan2",           (PyCFunction)THPTensor_stateless_(atan2),           METH_VARARGS, NULL},
  {"pow",             (PyCFunction)THPTensor_stateless_(pow),             METH_VARARGS, NULL},
  {"lerp",            (PyCFunction)THPTensor_stateless_(lerp),            METH_VARARGS, NULL},
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
  {"min",             (PyCFunction)THPTensor_stateless_(min),             METH_VARARGS, NULL},
  {"max",             (PyCFunction)THPTensor_stateless_(max),             METH_VARARGS, NULL},
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
  {"diag",            (PyCFunction)THPTensor_stateless_(diag),            METH_VARARGS, NULL},
  {"numel",           (PyCFunction)THPTensor_stateless_(numel),           METH_VARARGS, NULL},
  {"sign",            (PyCFunction)THPTensor_stateless_(sign),            METH_VARARGS, NULL},
  {"trace",           (PyCFunction)THPTensor_stateless_(trace),           METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPTensor_stateless_(tril),            METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPTensor_stateless_(triu),            METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPTensor_stateless_(zero),            METH_VARARGS, NULL},
  {"gt",              (PyCFunction)THPTensor_stateless_(gt),              METH_VARARGS, NULL},
  {"lt",              (PyCFunction)THPTensor_stateless_(lt),              METH_VARARGS, NULL},
  {"ge",              (PyCFunction)THPTensor_stateless_(ge),              METH_VARARGS, NULL},
  {"le",              (PyCFunction)THPTensor_stateless_(le),              METH_VARARGS, NULL},
  {"eq",              (PyCFunction)THPTensor_stateless_(eq),              METH_VARARGS, NULL},
  {"ne",              (PyCFunction)THPTensor_stateless_(ne),              METH_VARARGS, NULL},
  {"kthvalue",        (PyCFunction)THPTensor_stateless_(kthvalue),        METH_VARARGS, NULL},
  {"mode",            (PyCFunction)THPTensor_stateless_(mode),            METH_VARARGS, NULL},
  {"median",          (PyCFunction)THPTensor_stateless_(median),          METH_VARARGS, NULL},
  {"cross",           (PyCFunction)THPTensor_stateless_(cross),           METH_VARARGS, NULL},
  {"sort",            (PyCFunction)THPTensor_stateless_(sort),            METH_VARARGS, NULL},
  {"topk",            (PyCFunction)THPTensor_stateless_(topk),            METH_VARARGS, NULL},
  {"t",               (PyCFunction)THPTensor_stateless_(t),               METH_VARARGS, NULL},
  {"transpose",       (PyCFunction)THPTensor_stateless_(transpose),       METH_VARARGS, NULL},
  {"squeeze",         (PyCFunction)THPTensor_stateless_(squeeze),         METH_VARARGS, NULL},
  {"nonzero",         (PyCFunction)THPTensor_stateless_(nonzero),         METH_VARARGS, NULL},
  {"contiguous",      (PyCFunction)THPTensor_stateless_(contiguous),      METH_VARARGS, NULL},
  {"clone",           (PyCFunction)THPTensor_stateless_(clone),           METH_VARARGS, NULL},
  {"reshape",         (PyCFunction)THPTensor_stateless_(reshape),         METH_VARARGS, NULL},
  {"zeros",           (PyCFunction)THPTensor_stateless_(zeros),           METH_VARARGS, NULL},
  {"ones",            (PyCFunction)THPTensor_stateless_(ones),            METH_VARARGS, NULL},
  {"index",           (PyCFunction)THPTensor_stateless_(index),           METH_VARARGS, NULL},
  {"indexCopy",       (PyCFunction)THPTensor_stateless_(indexCopy),       METH_VARARGS, NULL},
  {"indexAdd",        (PyCFunction)THPTensor_stateless_(indexAdd),        METH_VARARGS, NULL},
  {"indexFill",       (PyCFunction)THPTensor_stateless_(indexFill),       METH_VARARGS, NULL},
  {NULL}
};

