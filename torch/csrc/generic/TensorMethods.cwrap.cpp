#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#define RealStr "float"
#else
#define RealStr "int"
#endif

#ifdef THC_REAL_IS_HALF
#define AS_REAL(x) THC_float2half(x)
#else
#define AS_REAL(x) x
#endif

#ifndef THC_GENERIC_FILE
#define IS_CUDA false
#define CUDA_FLOAT false
#else
#define IS_CUDA true
#define CUDA_FLOAT defined(THC_REAL_IS_FLOAT)
#endif

[[
  elementSize STATEFUL_ONLY
  elementSize -> long STORAGE_CALL
]]

[[
  storage STATEFUL_ONLY
  storage -> THStorage
    - self
]]

[[
  storageOffset STATEFUL_ONLY
  storageOffset -> long
    - self
]]

[[
  nDimension STATEFUL_ONLY
  nDimension -> long
    - self
]]

[[
  free STATEFUL_ONLY
  free -> self
    - self
]]

[[
  retain STATEFUL_ONLY
  retain -> self
    - self
]]

[[
  reshape
  reshape -> self LONG_ARGS
    - self
    - self
    - CONSTANT _long_args
  reshape -> self LONG_ARGS OPTIONAL_SELF
    - self
    - THTensor src
    - CONSTANT _long_args
]]

[[
  resize STATEFUL_ONLY
  resize -> self LONG_ARGS
    - self
    - CONSTANT _long_args
    - CONSTANT NULL
]]

[[
  zeros
  zeros -> self LONG_ARGS OPTIONAL_SELF
    - self
    - CONSTANT _long_args
]]

[[
  ones
  ones -> self LONG_ARGS OPTIONAL_SELF
    - self
    - CONSTANT _long_args
]]

[[
  numel
  numel -> long
    - self
]]

[[
  set STATEFUL_ONLY
  set -> self
    - self
    - THTensor source
  setStorage -> self
    - self
    - THStorage sourceStorage
    - CONSTANT 0
    - EXPRESSION THLongStorage_newWithSize1(THStorage_(size)(LIBRARY_STATE sourceStorage->cdata))
    - CONSTANT NULL
  setStorage -> self
    - self
    - THStorage sourceStorage
    - long storageOffset
    - THLongStorage sizes
    - THLongStorage strides OPTIONAL NULL
  setStorage -> self LONG_ARGS
    - self
    - THStorage sourceStorage
    - long storageOffset
    - CONSTANT _long_args
    - CONSTANT NULL
]]

static PyObject * THPTensor_(select)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  long dim, idx;
  if (!PyArg_ParseTuple(args, "ll", &dim, &idx))
    return NULL;

  int ndim = THTensor_(nDimension)(LIBRARY_STATE self->cdata);
  if(ndim > 1) {
    THTensor *selected = THTensor_(newWithTensor)(LIBRARY_STATE self->cdata);
    THTensor_(select)(LIBRARY_STATE selected, NULL, dim, idx);
    return THPTensor_(newObject)(selected);
  }
  else {
    THArgCheck(ndim == 1, 1, "empty Tensor");
    return THPUtils_(newReal)(THTensor_(get1d)(LIBRARY_STATE self->cdata, idx));
  }
  END_HANDLE_TH_ERRORS
}

#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
#define BUILD_REAL_FMT "d"
#else
#define BUILD_REAL_FMT "L"
#endif

#if !IS_CUDA
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
#endif /* !IS_CUDA */

#undef BUILD_REAL_FMT

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
#endif /* defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG) */

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || \
    CUDA_FLOAT
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
    - real p OPTIONAL 2
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

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
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
#endif /* defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) */

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

#if !IS_CUDA || CUDA_FLOAT
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
#endif /* !IS_CUDA || CUDA_FLOAT */

#if !IS_CUDA
[[
  trace
  trace -> accreal
    - self
]]
#endif

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
    - CONSTANT AS_REAL(1)
    - THTensor a
  cadd -> self
    - self
    - self
    - real value
    - THTensor a
  cadd -> self OPTIONAL_SELF
    - self
    - THTensor a
    - CONSTANT AS_REAL(1)
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
    - CONSTANT AS_REAL(1)
    - THTensor a
  csub -> self
    - self
    - self
    - real value
    - THTensor a
  csub -> self OPTIONAL_SELF
    - self
    - THTensor a
    - CONSTANT AS_REAL(1)
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

#if !IS_CUDA
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
#endif /* !IS_CUDA */

#if !IS_CUDA || CUDA_FLOAT
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
#endif /* !IS_CUDA || CUDA_FLOAT */

#if !IS_CUDA
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
  equal
  equal -> bool
    - self
    - THTensor other
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
#endif /* !IS_CUDA */

#if !IS_CUDA
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
#elif CUDA_FLOAT
[[
  lt
  ltValue -> new THTensor
    - self
    - real value
  ltValue -> self
    - self
    - THTensor other
    - real value
  ltTensor -> new THTensor
    - self
    - THTensor other
  ltTensor -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  gt
  gtValue -> new THTensor
    - self
    - real value
  gtValue -> self
    - self
    - THTensor other
    - real value
  gtTensor -> new THTensor
    - self
    - THTensor other
  gtTensor -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  le
  leValue -> new THTensor
    - self
    - real value
  leValue -> self
    - self
    - THTensor other
    - real value
  leTensor -> new THTensor
    - self
    - THTensor other
  leTensor -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  ge
  geValue -> new THTensor
    - self
    - real value
  geValue -> self
    - self
    - THTensor other
    - real value
  geTensor -> new THTensor
    - self
    - THTensor other
  geTensor -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  eq
  eqValue -> new THTensor
    - self
    - real value
  eqValue -> self
    - self
    - THTensor other
    - real value
  eqTensor -> new THTensor
    - self
    - THTensor other
  eqTensor -> self
    - self
    - THTensor other
    - THTensor other2
]]

[[
  ne
  neValue -> new THTensor
    - self
    - real value
  neValue -> self
    - self
    - THTensor other
    - real value
  neTensor -> new THTensor
    - self
    - THTensor other
  neTensor -> self
    - self
    - THTensor other
    - THTensor other2
]]
#endif

#if !IS_CUDA
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
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
  kthvalue -> new ValueIndexPair
    - self
    - long k
    - long dim
  kthvalue -> new SelfIndexPair
    - THTensor other
    - long k
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
  kthvalue -> new SelfIndexPair
    - THTensor other
    - long k
    - long dim
]]

[[
  mode
  mode -> new ValueIndexPair
    - self
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
  mode -> new ValueIndexPair
    - self
    - long dim
  mode -> new SelfIndexPair
    - THTensor other
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
  mode -> new SelfIndexPair
    - THTensor other
    - long dim
]]

[[
  median
  median -> new ValueIndexPair
    - self
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
  median -> new ValueIndexPair
    - self
    - long dim
  median -> new SelfIndexPair
    - THTensor other
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
  median -> new SelfIndexPair
    - THTensor other
    - long dim
]]

[[
  sort
  sort -> new ValueIndexPair
    - self
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
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
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
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
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new ValueIndexPair
    - self
    - long k
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
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
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new SelfIndexPair
    - THTensor source
    - long k
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
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
#elif CUDA_FLOAT
[[
  min
  minall -> real
    - self
  min -> new ValueValuePair
    - self
    - long index
  min -> new SelfValuePair
    - THTensor other
    - long index
]]

[[
  max
  maxall -> real
    - self
  max -> new ValueValuePair
    - self
    - long index
  max -> new SelfValuePair
    - THTensor other
    - long index
]]

[[
  sort
  sort -> new ValueValuePair
    - self
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
  sort -> new ValueValuePair
    - self
    - long dim
    - CONSTANT false
  sort -> new ValueValuePair
    - self
    - long dim
    - bool descending
  sort -> new SelfValuePair
    - THTensor source
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
  sort -> new SelfValuePair
    - THTensor source
    - long dim
    - CONSTANT false
  sort -> new SelfValuePair
    - THTensor source
    - long dim
    - bool descending
]]

[[
  topk
  topk -> new ValueValuePair
    - self
    - CONSTANT 1
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new ValueValuePair
    - self
    - long k
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new ValueValuePair
    - self
    - long k
    - long dim
    - CONSTANT false
    - CONSTANT false
  topk -> new ValueValuePair
    - self
    - long k
    - long dim
    - bool smallest
    - CONSTANT false
  topk -> new ValueValuePair
    - self
    - long k
    - long dim
    - bool smallest
    - bool sorted
  topk -> new SelfValuePair
    - THTensor source
    - CONSTANT 1
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new SelfValuePair
    - THTensor source
    - long k
    - EXPRESSION THTensor_(size)(LIBRARY_STATE {2}->cdata,THTensor_(nDimension)(LIBRARY_STATE {2}->cdata)-1)
    - CONSTANT false
    - CONSTANT false
  topk -> new SelfValuePair
    - THTensor source
    - long k
    - long dim
    - CONSTANT false
    - CONSTANT false
  topk -> new SelfValuePair
    - THTensor source
    - long k
    - long dim
    - bool smallest
    - CONSTANT false
  topk -> new SelfValuePair
    - THTensor source
    - long k
    - long dim
    - bool smallest
    - bool sorted
]]
#endif

// TODO: why are these stateful only?
#if !IS_CUDA
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
#elif CUDA_FLOAT
[[
  maskedFill STATEFUL_ONLY
  maskedFill -> self
    - self
    - THTensor mask
    - real value
]]

[[
  maskedCopy STATEFUL_ONLY
  maskedCopy -> self
    - self
    - THTensor mask
    - THTensor source
]]

[[
  maskedSelect STATEFUL_ONLY
  maskedSelect -> new THTensor
    - self
    - THTensor mask
  maskedSelect -> self
    - self
    - THTensor source
    - THTensor mask
]]
#endif

[[
  transpose
  newTranspose -> THTensor
    - self
    - long dim0
    - long dim1
]]

[[
  t
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

#if !IS_CUDA
[[
  nonzero
  nonzero -> new THLongTensor
    - self
  nonzero -> CUSTOM {expr}; Py_INCREF(indices); return (PyObject*)indices
    - THLongTensor indices
    - self
]]
#endif /* !IS_CUDA */

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
#if !IS_CUDA
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
#elif CUDA_FLOAT
[[
  index
  indexSelect -> self
    - self
    - self
    - long dim
    - THTensor index
  indexSelect -> self OPTIONAL_SELF
    - self
    - THTensor source
    - long dim
    - THTensor index
]]

[[
  indexCopy
  indexCopy -> self
    - self
    - long dim
    - THTensor index
    - THTensor source
]]

[[
  indexAdd
  indexAdd -> self
    - self
    - long dim
    - THTensor index
    - THTensor source
]]

[[
  indexFill
  indexFill -> self
    - self
    - long dim
    - THTensor index
    - real value
]]
#endif

[[
  narrow
  narrow -> new THTensor
    - self
    - long dimension
    - long firstIndex
    - long size
]]

[[
  unfold
  unfold -> new THTensor
    - self
    - long dimension
    - long size
    - long step
]]

#if !IS_CUDA
[[
  range
  range -> self OPTIONAL_SELF
    - self
    - accreal xmin
    - accreal xmax
    - accreal step
]]
#endif /* !IS_CUDA */

#if !IS_CUDA
[[
  scatter
  scatter -> self OPTIONAL_SELF
    - self
    - long dim
    - THLongTensor index
    - THTensor src
  scatterFill -> self OPTIONAL_SELF
    - self
    - long dim
    - THLongTensor index
    - real value
]]

[[
  gather
  gather -> new THTensor
    - self
    - long dim
    - THLongTensor index
  gather -> self OPTIONAL_SELF
    - self
    - THTensor src
    - long dim
    - THLongTensor index
]]
#elif CUDA_FLOAT
[[
  scatter
  scatter -> self OPTIONAL_SELF
    - self
    - long dim
    - THTensor index
    - THTensor src
  scatterFill -> self OPTIONAL_SELF
    - self
    - long dim
    - THTensor index
    - real value
]]

[[
  gather
  gather -> new THTensor
    - self
    - long dim
    - THTensor index
  gather -> self OPTIONAL_SELF
    - self
    - THTensor src
    - long dim
    - THTensor index
]]
#endif

#if !IS_CUDA || CUDA_FLOAT
// TODO: torch docs provide 7 args
[[
  addmm
  addmm -> self OPTIONAL_SELF
    - self
    - real beta OPTIONAL 1
    - THTensor M
    - real alpha OPTIONAL 1
    - THTensor mat1
    - THTensor mat2
]]

[[
  addmv
  addmv -> self OPTIONAL_SELF
    - self
    - real beta OPTIONAL 1
    - THTensor M
    - real alpha OPTIONAL 1
    - THTensor mat
    - THTensor vec
]]

[[
  addr
  addr -> self OPTIONAL_SELF
    - self
    - real beta OPTIONAL 1
    - THTensor M
    - real alpha OPTIONAL 1
    - THTensor vec1
    - THTensor vec2
]]

#define IMPLEMENT_TWO_MAT_OP(NAME, TH_NAME, PRE_CALL)                          \
static PyObject * THPTensor_(NAME)(THPTensor *self, PyObject *args)            \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  int _argcount = args ? PyTuple_Size(args) : 0;                               \
  if (_argcount != 2) {                                                        \
    THPUtils_setError("Provided %d args, but " #NAME "expects exactly two!", _argcount); \
    return NULL;                                                               \
  }                                                                            \
  THPTensor *t1 = (THPTensor*)PyTuple_GET_ITEM(args, 0);                       \
  THPTensor *t2 = (THPTensor*)PyTuple_GET_ITEM(args, 1);                       \
  if (!THPTensor_(IsSubclass)((PyObject*)t1) || !THPTensor_(IsSubclass)((PyObject*)t2)) { \
    THPUtils_setError("Expected two " THPTensorBaseStr);                       \
    return NULL;                                                               \
  }                                                                            \
  PRE_CALL;                                                                    \
  THTensor_(TH_NAME)(LIBRARY_STATE self->cdata, 0, self->cdata, 1, t1->cdata, t2->cdata); \
                                                                               \
  Py_INCREF(self);                                                             \
  return (PyObject*)self;                                                      \
  END_HANDLE_TH_ERRORS                                                         \
}


#define IMPLEMENT_TWO_MAT_OP_STATELESS(NAME, TH_NAME, PRE_CALL)                \
static PyObject * THPTensor_stateless_(NAME)(PyObject *_unused, PyObject *args)\
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  int _argcount = args ? PyTuple_Size(args) : 0;                               \
  THPTensor *t1;                                                               \
  THPTensor *t2;                                                               \
  THPTensorPtr self_ptr;                                                       \
  if (_argcount < 2 || _argcount > 3) {                                        \
    THPUtils_setError("Provided %d args, but expects two or three!", _argcount); \
    return NULL;                                                               \
  }                                                                            \
  if (_argcount == 2) {                                                        \
    t1 = (THPTensor*)PyTuple_GET_ITEM(args, 0);                                \
    t2 = (THPTensor*)PyTuple_GET_ITEM(args, 1);                                \
    if (!THPTensor_(IsSubclass)((PyObject*)t1) || !THPTensor_(IsSubclass)((PyObject*)t2)) { \
      THPUtils_setError("Expected two " THPTensorBaseStr);                     \
      return NULL;                                                             \
    }                                                                          \
    THTensorPtr _tmp = THTensor_(new)(LIBRARY_STATE_NOARGS);                   \
    self_ptr = (THPTensor*)THPTensor_(newObject)(_tmp);                        \
    _tmp.release();                                                            \
  } else {                                                                     \
    THPTensor *self = (THPTensor*)PyTuple_GET_ITEM(args, 0);                   \
    Py_INCREF(self);                                                           \
    self_ptr = self;                                                           \
    t1 = (THPTensor*)PyTuple_GET_ITEM(args, 1);                                \
    t2 = (THPTensor*)PyTuple_GET_ITEM(args, 2);                                \
    if (!THPTensor_(IsSubclass)((PyObject*)t1) || !THPTensor_(IsSubclass)((PyObject*)t2) || !THPTensor_(IsSubclass)((PyObject*)self)) { \
      THPUtils_setError("Expected three " THPTensorBaseStr);                   \
      return NULL;                                                             \
    }                                                                          \
  }                                                                            \
  PRE_CALL;                                                                    \
  THTensor_(TH_NAME)(LIBRARY_STATE self_ptr->cdata, 0, self_ptr->cdata, 1, t1->cdata, t2->cdata); \
                                                                               \
  return (PyObject*)self_ptr.release();                                        \
  END_HANDLE_TH_ERRORS                                                         \
}

IMPLEMENT_TWO_MAT_OP(ger, addr,
  long s1 = THTensor_(size)(LIBRARY_STATE t1->cdata, 0);
  long s2 = THTensor_(size)(LIBRARY_STATE t2->cdata, 0);
  THTensor_(resize2d)(LIBRARY_STATE self->cdata, s1, s2)
  );
IMPLEMENT_TWO_MAT_OP(mv, addmv,
  long s = THTensor_(size)(LIBRARY_STATE t1->cdata, 0);
  THTensor_(resize1d)(LIBRARY_STATE self->cdata, s)
  );

IMPLEMENT_TWO_MAT_OP_STATELESS(ger, addr,
  long s1 = THTensor_(size)(LIBRARY_STATE t1->cdata, 0);
  long s2 = THTensor_(size)(LIBRARY_STATE t2->cdata, 0);
  THTensor_(resize2d)(LIBRARY_STATE self_ptr->cdata, s1, s2)
  );
IMPLEMENT_TWO_MAT_OP_STATELESS(mv, addmv,
  long s = THTensor_(size)(LIBRARY_STATE t1->cdata, 0);
  THTensor_(resize1d)(LIBRARY_STATE self_ptr->cdata, s)
  );

[[
  addbmm
  addbmm -> self OPTIONAL_SELF
    - self
    - real beta OPTIONAL 1
    - THTensor M
    - real alpha OPTIONAL 1
    - THTensor batch1
    - THTensor batch2
]]

[[
  baddbmm
  baddbmm -> self OPTIONAL_SELF
    - self
    - real beta OPTIONAL 1
    - THTensor M
    - real alpha OPTIONAL 1
    - THTensor batch1
    - THTensor batch2
]]

[[
  addcmul
  addcmul -> self OPTIONAL_SELF
    - self
    - self
    - real value OPTIONAL 1
    - THTensor t1
    - THTensor t2
  addcmul -> self OPTIONAL_SELF
    - self
    - THTensor M
    - real value OPTIONAL 1
    - THTensor t1
    - THTensor t2
]]

[[
  addcdiv
  addcdiv -> self
    - self
    - self
    - real value OPTIONAL 1
    - THTensor t1
    - THTensor t2
  addcdiv -> self OPTIONAL_SELF
    - self
    - THTensor M
    - real value OPTIONAL 1
    - THTensor t1
    - THTensor t2
]]

// TODO: mm and bmm will fail if mat1 and mat2 aren't square
[[
  mm
  addmm -> self OPTIONAL_SELF
    - self
    - CONSTANT 0
    - THTensor mat1
    - CONSTANT 1
    - THTensor mat1
    - THTensor mat2
]]

[[
  bmm
  addbmm -> self OPTIONAL_SELF
    - self
    - CONSTANT 0
    - THTensor mat1
    - CONSTANT 1
    - THTensor mat1
    - THTensor mat2
]]
#endif

#if !IS_CUDA
[[
  randperm
  randperm -> self OPTIONAL_SELF
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - long n
]]

[[
  random
  random -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
]]
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
[[
  multinomial
  multinomial -> new THLongTensor
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - self
    - long n
    - bool replacement OPTIONAL false
]]

[[
  uniform
  uniform -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - real a OPTIONAL 0
    - real b OPTIONAL 1
]]

[[
  normal
  normal -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - real a OPTIONAL 0
    - real b OPTIONAL 1
]]

[[
  cauchy
  cauchy -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - real a OPTIONAL 0
    - real b OPTIONAL 1
]]

[[
  logNormal
  logNormal -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - real a OPTIONAL 1
    - real b OPTIONAL 2
]]

[[
  exponential
  exponential -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - real lambda OPTIONAL 1
]]

[[
  rand
  rand -> self LONG_ARGS OPTIONAL_SELF
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - CONSTANT _long_args
]]

[[
  randn
  randn -> self LONG_ARGS OPTIONAL_SELF
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - CONSTANT _long_args
]]

// TODO: can't handle sampling from [a, b]
[[
  geometric
  geometric -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - double p
]]

[[
  bernoulli
  bernoulli -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - double p OPTIONAL 0.5
  bernoulli_FloatTensor -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - THFloatTensor float_probabilities
  bernoulli_DoubleTensor -> self
    - self
    - THGenerator generator OPTIONAL THPDefaultGenerator->cdata
    - THDoubleTensor double_probabilities
]]
#elif CUDA_FLOAT
[[
  multinomial
  multinomial -> new THTensor
    - self
    - long n
    - bool replacement OPTIONAL false
]]

[[
  uniform
  uniform -> self
    - self
    - real a OPTIONAL 0
    - real b OPTIONAL 1
]]

[[
  normal
  normal -> self
    - self
    - real a OPTIONAL 0
    - real b OPTIONAL 1
]]

[[
  cauchy
  cauchy -> self
    - self
    - real a OPTIONAL 0
    - real b OPTIONAL 1
]]

[[
  logNormal
  logNormal -> self
    - self
    - real a OPTIONAL 1
    - real b OPTIONAL 2
]]

[[
  exponential
  exponential -> self
    - self
    - real lambda OPTIONAL 1
]]

[[
  rand
  rand -> self LONG_ARGS OPTIONAL_SELF
    - self
    - CONSTANT _long_args
]]

[[
  randn
  randn -> self LONG_ARGS OPTIONAL_SELF
    - self
    - CONSTANT _long_args
]]

// TODO: can't handle sampling from [a, b]
[[
  geometric
  geometric -> self
    - self
    - double p
]]

[[
  bernoulli
  bernoulli -> self
    - self
    - double p OPTIONAL 0.5
]]
#endif

// Declared in TensorCopy.cpp
//PyObject * THPTensor_(copy)(THPTensor *self, PyObject *other);

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
  //{"copy",            (PyCFunction)THPTensor_(copy),            METH_O,       NULL},
  {"free",            (PyCFunction)THPTensor_(free),            METH_NOARGS,  NULL},
  {"retain",          (PyCFunction)THPTensor_(retain),          METH_NOARGS,  NULL},
  {"size",            (PyCFunction)THPTensor_(size),            METH_VARARGS, NULL},
  {"select",          (PyCFunction)THPTensor_(select),          METH_VARARGS, NULL},
#if !IS_CUDA || CUDA_FLOAT
  {"maskedFill",      (PyCFunction)THPTensor_(maskedFill),      METH_VARARGS, NULL},
  {"maskedCopy",      (PyCFunction)THPTensor_(maskedCopy),      METH_VARARGS, NULL},
  {"maskedSelect",    (PyCFunction)THPTensor_(maskedSelect),    METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"apply",           (PyCFunction)THPTensor_(apply),           METH_O,       NULL},
  {"map",             (PyCFunction)THPTensor_(map),             METH_VARARGS, NULL},
  {"map2",            (PyCFunction)THPTensor_(map2),            METH_VARARGS, NULL},
#endif
  {"resize",          (PyCFunction)THPTensor_(resize),          METH_VARARGS, NULL},
  {"resizeAs",        (PyCFunction)THPTensor_(resizeAs),        METH_VARARGS, NULL},
  {"set",             (PyCFunction)THPTensor_(set),             METH_VARARGS, NULL},
  //////////////////////////////////////////////////////////////////////////////
#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
  {"abs",             (PyCFunction)THPTensor_(abs),             METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
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
  {"atan2",           (PyCFunction)THPTensor_(atan2),           METH_VARARGS, NULL},
  {"pow",             (PyCFunction)THPTensor_(pow),             METH_VARARGS, NULL},
  {"lerp",            (PyCFunction)THPTensor_(lerp),            METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"linspace",        (PyCFunction)THPTensor_(linspace),        METH_VARARGS, NULL},
  {"logspace",        (PyCFunction)THPTensor_(logspace),        METH_VARARGS, NULL},
  {"histc",           (PyCFunction)THPTensor_(histc),           METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"multinomial",     (PyCFunction)THPTensor_(multinomial),     METH_VARARGS, NULL},
  {"uniform",         (PyCFunction)THPTensor_(uniform),         METH_VARARGS, NULL},
  {"normal",          (PyCFunction)THPTensor_(normal),          METH_VARARGS, NULL},
  {"cauchy",          (PyCFunction)THPTensor_(cauchy),          METH_VARARGS, NULL},
  {"logNormal",       (PyCFunction)THPTensor_(logNormal),       METH_VARARGS, NULL},
  {"exponential",     (PyCFunction)THPTensor_(exponential),     METH_VARARGS, NULL},
  {"rand",            (PyCFunction)THPTensor_(rand),            METH_VARARGS, NULL},
  {"randn",           (PyCFunction)THPTensor_(randn),           METH_VARARGS, NULL},
  {"geometric",       (PyCFunction)THPTensor_(geometric),       METH_VARARGS, NULL},
  {"bernoulli",       (PyCFunction)THPTensor_(bernoulli),       METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"randperm",        (PyCFunction)THPTensor_(randperm),        METH_VARARGS, NULL},
  {"random",          (PyCFunction)THPTensor_(random),          METH_VARARGS, NULL},
  {"fmod",            (PyCFunction)THPTensor_(fmod),            METH_VARARGS, NULL},
  {"mod",             (PyCFunction)THPTensor_(fmod),            METH_VARARGS, NULL},
  {"cfmod",           (PyCFunction)THPTensor_(cfmod),           METH_VARARGS, NULL},
  {"cmod",            (PyCFunction)THPTensor_(cfmod),           METH_VARARGS, NULL},
  {"remainder",       (PyCFunction)THPTensor_(remainder),       METH_VARARGS, NULL},
  {"cremainder",      (PyCFunction)THPTensor_(cremainder),      METH_VARARGS, NULL},
  {"eye",             (PyCFunction)THPTensor_(eye),             METH_VARARGS, NULL},
  {"equal",           (PyCFunction)THPTensor_(equal),           METH_VARARGS, NULL},
  {"diag",            (PyCFunction)THPTensor_(diag),            METH_VARARGS, NULL},
  {"trace",           (PyCFunction)THPTensor_(trace),           METH_VARARGS, NULL},
  {"kthvalue",        (PyCFunction)THPTensor_(kthvalue),        METH_VARARGS, NULL},
  {"mode",            (PyCFunction)THPTensor_(mode),            METH_VARARGS, NULL},
  {"median",          (PyCFunction)THPTensor_(median),          METH_VARARGS, NULL},
  {"nonzero",         (PyCFunction)THPTensor_(nonzero),         METH_VARARGS, NULL},
#endif
#if !IS_CUDA || CUDA_FLOAT
  {"min",             (PyCFunction)THPTensor_(min),             METH_VARARGS, NULL},
  {"max",             (PyCFunction)THPTensor_(max),             METH_VARARGS, NULL},
  {"cmax",            (PyCFunction)THPTensor_(cmax),            METH_VARARGS, NULL},
  {"cmin",            (PyCFunction)THPTensor_(cmin),            METH_VARARGS, NULL},
  {"dot",             (PyCFunction)THPTensor_(dot),             METH_VARARGS, NULL},
  {"sum",             (PyCFunction)THPTensor_(sum),             METH_VARARGS, NULL},
  {"prod",            (PyCFunction)THPTensor_(prod),            METH_VARARGS, NULL},
  {"cumsum",          (PyCFunction)THPTensor_(cumsum),          METH_VARARGS, NULL},
  {"cumprod",         (PyCFunction)THPTensor_(cumprod),         METH_VARARGS, NULL},
  {"clamp",           (PyCFunction)THPTensor_(clamp),           METH_VARARGS, NULL},
  {"sign",            (PyCFunction)THPTensor_(sign),            METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPTensor_(tril),            METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPTensor_(triu),            METH_VARARGS, NULL},
  {"gt",              (PyCFunction)THPTensor_(gt),              METH_VARARGS, NULL},
  {"lt",              (PyCFunction)THPTensor_(lt),              METH_VARARGS, NULL},
  {"ge",              (PyCFunction)THPTensor_(ge),              METH_VARARGS, NULL},
  {"le",              (PyCFunction)THPTensor_(le),              METH_VARARGS, NULL},
  {"eq",              (PyCFunction)THPTensor_(eq),              METH_VARARGS, NULL},
  {"ne",              (PyCFunction)THPTensor_(ne),              METH_VARARGS, NULL},
  {"cross",           (PyCFunction)THPTensor_(cross),           METH_VARARGS, NULL},
  {"sort",            (PyCFunction)THPTensor_(sort),            METH_VARARGS, NULL},
  {"topk",            (PyCFunction)THPTensor_(topk),            METH_VARARGS, NULL},
#endif
  {"add",             (PyCFunction)THPTensor_(add),             METH_VARARGS, NULL},
  {"csub",            (PyCFunction)THPTensor_(csub),            METH_VARARGS, NULL},
  {"mul",             (PyCFunction)THPTensor_(mul),             METH_VARARGS, NULL},
  {"cmul",            (PyCFunction)THPTensor_(cmul),            METH_VARARGS, NULL},
  {"div",             (PyCFunction)THPTensor_(div),             METH_VARARGS, NULL},
  {"cdiv",            (PyCFunction)THPTensor_(cdiv),            METH_VARARGS, NULL},
  {"cpow",            (PyCFunction)THPTensor_(cpow),            METH_VARARGS, NULL},
  {"fill",            (PyCFunction)THPTensor_(fill),            METH_VARARGS, NULL},
  {"numel",           (PyCFunction)THPTensor_(numel),           METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPTensor_(zero),            METH_VARARGS, NULL},
  {"t",               (PyCFunction)THPTensor_(t),               METH_VARARGS, NULL},
  {"transpose",       (PyCFunction)THPTensor_(transpose),       METH_VARARGS, NULL},
  {"squeeze",         (PyCFunction)THPTensor_(squeeze),         METH_VARARGS, NULL},
  {"contiguous",      (PyCFunction)THPTensor_(contiguous),      METH_VARARGS, NULL},
  {"clone",           (PyCFunction)THPTensor_(clone),           METH_VARARGS, NULL},
  {"reshape",         (PyCFunction)THPTensor_(reshape),         METH_VARARGS, NULL},
  {"zeros",           (PyCFunction)THPTensor_(zeros),           METH_VARARGS, NULL},
  {"ones",            (PyCFunction)THPTensor_(ones),            METH_VARARGS, NULL},
#if !IS_CUDA || CUDA_FLOAT
  {"index",           (PyCFunction)THPTensor_(index),           METH_VARARGS, NULL},
  {"indexCopy",       (PyCFunction)THPTensor_(indexCopy),       METH_VARARGS, NULL},
  {"indexAdd",        (PyCFunction)THPTensor_(indexAdd),        METH_VARARGS, NULL},
  {"indexFill",       (PyCFunction)THPTensor_(indexFill),       METH_VARARGS, NULL},
#endif
  {"narrow",          (PyCFunction)THPTensor_(narrow),          METH_VARARGS, NULL},
  {"unfold",          (PyCFunction)THPTensor_(unfold),          METH_VARARGS, NULL},
#if !IS_CUDA || CUDA_FLOAT
  {"addmm",           (PyCFunction)THPTensor_(addmm),           METH_VARARGS, NULL},
  {"addmv",           (PyCFunction)THPTensor_(addmv),           METH_VARARGS, NULL},
  {"addr",            (PyCFunction)THPTensor_(addr),            METH_VARARGS, NULL},
  {"ger",             (PyCFunction)THPTensor_(ger),             METH_VARARGS, NULL},
  {"mv",              (PyCFunction)THPTensor_(mv),              METH_VARARGS, NULL},
  {"addbmm",          (PyCFunction)THPTensor_(addbmm),          METH_VARARGS, NULL},
  {"baddbmm",         (PyCFunction)THPTensor_(baddbmm),         METH_VARARGS, NULL},
  {"addcmul",         (PyCFunction)THPTensor_(addcmul),         METH_VARARGS, NULL},
  {"addcdiv",         (PyCFunction)THPTensor_(addcdiv),         METH_VARARGS, NULL},
  {"mm",              (PyCFunction)THPTensor_(mm),              METH_VARARGS, NULL},
  {"bmm",             (PyCFunction)THPTensor_(bmm),             METH_VARARGS, NULL},
#if !IS_CUDA
  {"range",           (PyCFunction)THPTensor_(range),           METH_VARARGS, NULL},
#endif
  {"gather",          (PyCFunction)THPTensor_(gather),          METH_VARARGS, NULL},
  {"scatter",         (PyCFunction)THPTensor_(scatter),         METH_VARARGS, NULL},
#endif
  {NULL}
};

static PyMethodDef THPTensorStatelessMethods[] = {
#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
  {"abs",             (PyCFunction)THPTensor_stateless_(abs),             METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
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
  {"atan2",           (PyCFunction)THPTensor_stateless_(atan2),           METH_VARARGS, NULL},
  {"pow",             (PyCFunction)THPTensor_stateless_(pow),             METH_VARARGS, NULL},
  {"lerp",            (PyCFunction)THPTensor_stateless_(lerp),            METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"linspace",        (PyCFunction)THPTensor_stateless_(linspace),        METH_VARARGS, NULL},
  {"logspace",        (PyCFunction)THPTensor_stateless_(logspace),        METH_VARARGS, NULL},
  {"histc",           (PyCFunction)THPTensor_stateless_(histc),           METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"multinomial",     (PyCFunction)THPTensor_stateless_(multinomial),     METH_VARARGS, NULL},
  {"uniform",         (PyCFunction)THPTensor_stateless_(uniform),         METH_VARARGS, NULL},
  {"normal",          (PyCFunction)THPTensor_stateless_(normal),          METH_VARARGS, NULL},
  {"cauchy",          (PyCFunction)THPTensor_stateless_(cauchy),          METH_VARARGS, NULL},
  {"logNormal",       (PyCFunction)THPTensor_stateless_(logNormal),       METH_VARARGS, NULL},
  {"exponential",     (PyCFunction)THPTensor_stateless_(exponential),     METH_VARARGS, NULL},
  {"rand",            (PyCFunction)THPTensor_stateless_(rand),            METH_VARARGS, NULL},
  {"randn",           (PyCFunction)THPTensor_stateless_(randn),           METH_VARARGS, NULL},
  {"geometric",       (PyCFunction)THPTensor_stateless_(geometric),       METH_VARARGS, NULL},
  {"bernoulli",       (PyCFunction)THPTensor_stateless_(bernoulli),       METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"randperm",        (PyCFunction)THPTensor_stateless_(randperm),        METH_VARARGS, NULL},
  {"random",          (PyCFunction)THPTensor_stateless_(random),          METH_VARARGS, NULL},
  {"fmod",            (PyCFunction)THPTensor_stateless_(fmod),            METH_VARARGS, NULL},
  {"mod",             (PyCFunction)THPTensor_stateless_(fmod),            METH_VARARGS, NULL},
  {"cfmod",           (PyCFunction)THPTensor_stateless_(cfmod),           METH_VARARGS, NULL},
  {"cmod",            (PyCFunction)THPTensor_stateless_(cfmod),           METH_VARARGS, NULL},
  {"remainder",       (PyCFunction)THPTensor_stateless_(remainder),       METH_VARARGS, NULL},
  {"cremainder",      (PyCFunction)THPTensor_stateless_(cremainder),      METH_VARARGS, NULL},
  {"eye",             (PyCFunction)THPTensor_stateless_(eye),             METH_VARARGS, NULL},
  {"equal",           (PyCFunction)THPTensor_stateless_(equal),           METH_VARARGS, NULL},
  {"diag",            (PyCFunction)THPTensor_stateless_(diag),            METH_VARARGS, NULL},
  {"trace",           (PyCFunction)THPTensor_stateless_(trace),           METH_VARARGS, NULL},
  {"kthvalue",        (PyCFunction)THPTensor_stateless_(kthvalue),        METH_VARARGS, NULL},
  {"mode",            (PyCFunction)THPTensor_stateless_(mode),            METH_VARARGS, NULL},
  {"median",          (PyCFunction)THPTensor_stateless_(median),          METH_VARARGS, NULL},
  {"nonzero",         (PyCFunction)THPTensor_stateless_(nonzero),         METH_VARARGS, NULL},
#endif
#if !IS_CUDA || CUDA_FLOAT
  {"min",             (PyCFunction)THPTensor_stateless_(min),             METH_VARARGS, NULL},
  {"max",             (PyCFunction)THPTensor_stateless_(max),             METH_VARARGS, NULL},
  {"cmax",            (PyCFunction)THPTensor_stateless_(cmax),            METH_VARARGS, NULL},
  {"cmin",            (PyCFunction)THPTensor_stateless_(cmin),            METH_VARARGS, NULL},
  {"dot",             (PyCFunction)THPTensor_stateless_(dot),             METH_VARARGS, NULL},
  {"sum",             (PyCFunction)THPTensor_stateless_(sum),             METH_VARARGS, NULL},
  {"prod",            (PyCFunction)THPTensor_stateless_(prod),            METH_VARARGS, NULL},
  {"cumsum",          (PyCFunction)THPTensor_stateless_(cumsum),          METH_VARARGS, NULL},
  {"cumprod",         (PyCFunction)THPTensor_stateless_(cumprod),         METH_VARARGS, NULL},
  {"clamp",           (PyCFunction)THPTensor_stateless_(clamp),           METH_VARARGS, NULL},
  {"sign",            (PyCFunction)THPTensor_stateless_(sign),            METH_VARARGS, NULL},
  {"tril",            (PyCFunction)THPTensor_stateless_(tril),            METH_VARARGS, NULL},
  {"triu",            (PyCFunction)THPTensor_stateless_(triu),            METH_VARARGS, NULL},
  {"gt",              (PyCFunction)THPTensor_stateless_(gt),              METH_VARARGS, NULL},
  {"lt",              (PyCFunction)THPTensor_stateless_(lt),              METH_VARARGS, NULL},
  {"ge",              (PyCFunction)THPTensor_stateless_(ge),              METH_VARARGS, NULL},
  {"le",              (PyCFunction)THPTensor_stateless_(le),              METH_VARARGS, NULL},
  {"eq",              (PyCFunction)THPTensor_stateless_(eq),              METH_VARARGS, NULL},
  {"ne",              (PyCFunction)THPTensor_stateless_(ne),              METH_VARARGS, NULL},
  {"cross",           (PyCFunction)THPTensor_stateless_(cross),           METH_VARARGS, NULL},
  {"sort",            (PyCFunction)THPTensor_stateless_(sort),            METH_VARARGS, NULL},
  {"topk",            (PyCFunction)THPTensor_stateless_(topk),            METH_VARARGS, NULL},
#endif
  {"add",             (PyCFunction)THPTensor_stateless_(add),             METH_VARARGS, NULL},
  {"csub",            (PyCFunction)THPTensor_stateless_(csub),            METH_VARARGS, NULL},
  {"mul",             (PyCFunction)THPTensor_stateless_(mul),             METH_VARARGS, NULL},
  {"cmul",            (PyCFunction)THPTensor_stateless_(cmul),            METH_VARARGS, NULL},
  {"div",             (PyCFunction)THPTensor_stateless_(div),             METH_VARARGS, NULL},
  {"cdiv",            (PyCFunction)THPTensor_stateless_(cdiv),            METH_VARARGS, NULL},
  {"cpow",            (PyCFunction)THPTensor_stateless_(cpow),            METH_VARARGS, NULL},
  {"fill",            (PyCFunction)THPTensor_stateless_(fill),            METH_VARARGS, NULL},
  {"numel",           (PyCFunction)THPTensor_stateless_(numel),           METH_VARARGS, NULL},
  {"zero",            (PyCFunction)THPTensor_stateless_(zero),            METH_VARARGS, NULL},
  {"t",               (PyCFunction)THPTensor_stateless_(t),               METH_VARARGS, NULL},
  {"transpose",       (PyCFunction)THPTensor_stateless_(transpose),       METH_VARARGS, NULL},
  {"squeeze",         (PyCFunction)THPTensor_stateless_(squeeze),         METH_VARARGS, NULL},
  {"contiguous",      (PyCFunction)THPTensor_stateless_(contiguous),      METH_VARARGS, NULL},
  {"clone",           (PyCFunction)THPTensor_stateless_(clone),           METH_VARARGS, NULL},
  {"reshape",         (PyCFunction)THPTensor_stateless_(reshape),         METH_VARARGS, NULL},
  {"zeros",           (PyCFunction)THPTensor_stateless_(zeros),           METH_VARARGS, NULL},
  {"ones",            (PyCFunction)THPTensor_stateless_(ones),            METH_VARARGS, NULL},
#if !IS_CUDA || CUDA_FLOAT
  {"index",           (PyCFunction)THPTensor_stateless_(index),           METH_VARARGS, NULL},
  {"indexCopy",       (PyCFunction)THPTensor_stateless_(indexCopy),       METH_VARARGS, NULL},
  {"indexAdd",        (PyCFunction)THPTensor_stateless_(indexAdd),        METH_VARARGS, NULL},
  {"indexFill",       (PyCFunction)THPTensor_stateless_(indexFill),       METH_VARARGS, NULL},
#endif
  {"narrow",          (PyCFunction)THPTensor_stateless_(narrow),          METH_VARARGS, NULL},
  {"unfold",          (PyCFunction)THPTensor_stateless_(unfold),          METH_VARARGS, NULL},
#if !IS_CUDA || CUDA_FLOAT
  {"addmm",           (PyCFunction)THPTensor_stateless_(addmm),           METH_VARARGS, NULL},
  {"addmv",           (PyCFunction)THPTensor_stateless_(addmv),           METH_VARARGS, NULL},
  {"addr",            (PyCFunction)THPTensor_stateless_(addr),            METH_VARARGS, NULL},
  {"ger",             (PyCFunction)THPTensor_stateless_(ger),             METH_VARARGS, NULL},
  {"mv",              (PyCFunction)THPTensor_stateless_(mv),              METH_VARARGS, NULL},
  {"addbmm",          (PyCFunction)THPTensor_stateless_(addbmm),          METH_VARARGS, NULL},
  {"baddbmm",         (PyCFunction)THPTensor_stateless_(baddbmm),         METH_VARARGS, NULL},
  {"addcmul",         (PyCFunction)THPTensor_stateless_(addcmul),         METH_VARARGS, NULL},
  {"addcdiv",         (PyCFunction)THPTensor_stateless_(addcdiv),         METH_VARARGS, NULL},
  {"mm",              (PyCFunction)THPTensor_stateless_(mm),              METH_VARARGS, NULL},
  {"bmm",             (PyCFunction)THPTensor_stateless_(bmm),             METH_VARARGS, NULL},
#if !IS_CUDA
  {"range",           (PyCFunction)THPTensor_stateless_(range),           METH_VARARGS, NULL},
#endif
  {"gather",          (PyCFunction)THPTensor_stateless_(gather),          METH_VARARGS, NULL},
  {"scatter",         (PyCFunction)THPTensor_stateless_(scatter),         METH_VARARGS, NULL},
#endif
  {NULL}
};

#undef RealStr
#undef AS_REAL
