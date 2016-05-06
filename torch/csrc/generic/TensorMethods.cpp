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

#define TENSOR_OR_DIM_WISE(name, expr_tensor, expr_dim)                        \
static PyObject * THPTensor_(name)(THPTensor *self, PyObject *arg)             \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  int dim = -1;                                                                \
  PARSE_TUPLE(arg, "|i", &dim);                                                \
  /* TODO: check dim? */                                                       \
  if (dim != -1) {                                                             \
    expr_dim;                                                                  \
  } else {                                                                     \
    expr_tensor;                                                               \
  }                                                                            \
  END_HANDLE_TH_ERRORS                                                         \
}

#define TENSOR_OR_DIM_WISE2(name, expr_tensor, expr_dim)                       \
static PyObject * THPTensor_(name)(THPTensor *self, PyObject *args)            \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  int dim = -1;                                                                \
  THPTensor *source = self;                                                    \
  int argc = PyTuple_Size(args);                                               \
  if (argc == 1) {                                                             \
    PARSE_TUPLE(args, "i", &dim);                                              \
  } else if (argc == 2) {                                                      \
    PARSE_TUPLE(args, "O!i", &THPTensorType, &source, &dim);                   \
  }                                                                            \
  /* TODO: check dim? */                                                       \
  if (dim == -1) {                                                             \
    expr_tensor;                                                               \
  } else {                                                                     \
    expr_dim;                                                                  \
  }                                                                            \
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

// TENSOR_OR_DIM_WISE
// Name
// Operation on the whole tensor
// Operation on selected dim (available in dim variable)
TENSOR_OR_DIM_WISE(
  size,
  return THPLongStorage_newObject(THTensor_(newSizeOf)(self->cdata)),
  return PyLong_FromLong(THTensor_(size)(self->cdata, dim))
)
TENSOR_OR_DIM_WISE(
  stride,
  return THPLongStorage_newObject(THTensor_(newStrideOf)(self->cdata)),
  return PyLong_FromLong(THTensor_(stride)(self->cdata, dim))
)

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TENSOR_OR_DIM_WISE2(
  mean,
  return PyFloat_FromDouble(THTensor_(meanall)(self->cdata)),
  THTensor_(mean)(self->cdata, source->cdata, dim); RETURN_SELF
)
TENSOR_OR_DIM_WISE2(
  var,
  return PyFloat_FromDouble(THTensor_(varall)(self->cdata)),
  THTensor_(var)(self->cdata, source->cdata, dim, false); RETURN_SELF
)
TENSOR_OR_DIM_WISE2(
  std,
  return PyFloat_FromDouble(THTensor_(stdall)(self->cdata)),
  THTensor_(std)(self->cdata, source->cdata, dim, false); RETURN_SELF
)
#endif

static PyObject * THPTensor_(isSameSizeAs)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPTensor *other;
  if (!PyArg_ParseTuple(args, "O!", &THPTensorType, &other))
    return NULL;
  return PyBool_FromLong(THTensor_(isSameSizeAs)(self->cdata, other->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(fill)(THPTensor *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  real rvalue;
  if (!THPUtils_(parseReal)(arg, &rvalue))
    return NULL;
  THTensor_(fill)(self->cdata, rvalue);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

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
  {"fill",            (PyCFunction)THPTensor_(fill),            METH_O,       NULL},
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
