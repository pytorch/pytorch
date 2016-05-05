
////////////////////////////////////////////////////////////////////////////////

#define IMPLEMENT_POINTWISE_OP(name)                                           \
static PyObject * THPTensor_(name)(THPTensor *self, PyObject *args)            \
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  THPTensor *source = self;                                                    \
  if (!PyArg_ParseTuple(args, "|O!", &THPTensorType, &source))                 \
    return NULL;                                                               \
  THTensor_(name)(self->cdata, source->cdata);                                 \
  Py_INCREF(self);                                                             \
  return (PyObject*)self;                                                      \
  END_HANDLE_TH_ERRORS                                                         \
}

#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
IMPLEMENT_POINTWISE_OP(abs)
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
IMPLEMENT_POINTWISE_OP(sigmoid)
IMPLEMENT_POINTWISE_OP(log)
IMPLEMENT_POINTWISE_OP(log1p)
IMPLEMENT_POINTWISE_OP(exp)
IMPLEMENT_POINTWISE_OP(cos)
IMPLEMENT_POINTWISE_OP(acos)
IMPLEMENT_POINTWISE_OP(cosh)
IMPLEMENT_POINTWISE_OP(sin)
IMPLEMENT_POINTWISE_OP(asin)
IMPLEMENT_POINTWISE_OP(sinh)
IMPLEMENT_POINTWISE_OP(tan)
IMPLEMENT_POINTWISE_OP(atan)
IMPLEMENT_POINTWISE_OP(tanh)
IMPLEMENT_POINTWISE_OP(sqrt)
IMPLEMENT_POINTWISE_OP(rsqrt)
IMPLEMENT_POINTWISE_OP(ceil)
IMPLEMENT_POINTWISE_OP(floor)
IMPLEMENT_POINTWISE_OP(round)
IMPLEMENT_POINTWISE_OP(abs)
IMPLEMENT_POINTWISE_OP(trunc)
IMPLEMENT_POINTWISE_OP(frac)
#endif

static PyObject * THPTensor_(size)(THPTensor *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  int dim = -1;
  if (!PyArg_ParseTuple(arg, "|i", &dim))
    return NULL;

  if (dim != -1) {
    return PyLong_FromLong(THTensor_(size)(self->cdata, dim));
  } else {
    return THPLongStorage_newObject(THTensor_(newSizeOf)(self->cdata));
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(storage)(THPTensor *self)
{
  HANDLE_TH_ERRORS
  return THPStorage_(newObject)(THTensor_(storage)(self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(storageOffset)(THPTensor *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THTensor_(storageOffset)(self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(isSameSizeAs)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPTensor *other;
  if (!PyArg_ParseTuple(args, "O!", &THPTensorType, &other))
    return NULL;
  return PyBool_FromLong(THTensor_(isSameSizeAs)(self->cdata, other->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(stride)(THPTensor *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  int dim = -1;
  if (!PyArg_ParseTuple(arg, "|i", &dim))
    return NULL;

  if (dim != -1) {
    return PyLong_FromLong(THTensor_(stride)(self->cdata, dim));
  } else {
    return THPLongStorage_newObject(THTensor_(newStrideOf)(self->cdata));
  }
  END_HANDLE_TH_ERRORS
}

// Declared in TensorCopy.cpp
static PyObject * THPTensor_(copy)(THPTensor *self, PyObject *other);

static PyMethodDef THPTensor_(methods)[] = {
#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
  {"abs", 						(PyCFunction)THPTensor_(abs), 						METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"sigmoid", 				(PyCFunction)THPTensor_(sigmoid),				  METH_VARARGS, NULL},
  {"log", 						(PyCFunction)THPTensor_(log),   					METH_VARARGS, NULL},
  {"log1p", 					(PyCFunction)THPTensor_(log1p), 					METH_VARARGS, NULL},
  {"exp",  						(PyCFunction)THPTensor_(exp),   					METH_VARARGS, NULL},
  {"cos",  						(PyCFunction)THPTensor_(cos),   					METH_VARARGS, NULL},
  {"acos", 						(PyCFunction)THPTensor_(acos),  					METH_VARARGS, NULL},
  {"cosh", 						(PyCFunction)THPTensor_(cosh),  					METH_VARARGS, NULL},
  {"sin",  						(PyCFunction)THPTensor_(sin),   					METH_VARARGS, NULL},
  {"asin", 						(PyCFunction)THPTensor_(asin),  					METH_VARARGS, NULL},
  {"sinh", 						(PyCFunction)THPTensor_(sinh),  					METH_VARARGS, NULL},
  {"tan", 						(PyCFunction)THPTensor_(tan),   					METH_VARARGS, NULL},
  {"atan", 						(PyCFunction)THPTensor_(atan),  					METH_VARARGS, NULL},
  {"tanh", 						(PyCFunction)THPTensor_(tanh),  					METH_VARARGS, NULL},
  {"sqrt", 						(PyCFunction)THPTensor_(sqrt),  					METH_VARARGS, NULL},
  {"rsqrt", 					(PyCFunction)THPTensor_(rsqrt), 					METH_VARARGS, NULL},
  {"ceil", 						(PyCFunction)THPTensor_(ceil),  					METH_VARARGS, NULL},
  {"floor", 					(PyCFunction)THPTensor_(floor), 					METH_VARARGS, NULL},
  {"round", 					(PyCFunction)THPTensor_(round), 					METH_VARARGS, NULL},
  {"abs", 						(PyCFunction)THPTensor_(abs), 						METH_VARARGS, NULL},
  {"trunc", 					(PyCFunction)THPTensor_(trunc), 					METH_VARARGS, NULL},
  {"frac", 						(PyCFunction)THPTensor_(frac), 						METH_VARARGS, NULL},
#endif
  {"copy",            (PyCFunction)THPTensor_(copy),            METH_O, NULL},
  {"isSameSizeAs",    (PyCFunction)THPTensor_(isSameSizeAs),    METH_VARARGS, NULL},
  {"size",            (PyCFunction)THPTensor_(size),            METH_VARARGS, NULL},
  {"storage",         (PyCFunction)THPTensor_(storage),         METH_NOARGS,  NULL},
  {"storageOffset",   (PyCFunction)THPTensor_(storageOffset),   METH_NOARGS,  NULL},
  {"stride",          (PyCFunction)THPTensor_(stride),          METH_VARARGS, NULL},
  {NULL}
};
