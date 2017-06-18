
static void tensorFill(rpc::RPCMessage& raw_message) {
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  thpp::Type type = peekType(raw_message);
  if (thpp::isInteger(type)) {
    auto value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(t)->fill(value);
  } else if (thpp::isFloat(type)) {
    auto value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(t)->fill(value);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorMaskedFill(rpc::RPCMessage& raw_message) {
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  thpp::Tensor *mask = unpackRetrieveTensor(raw_message);
  thpp::Type type = peekType(raw_message);
  if (thpp::isInteger(type)) {
    auto value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(t)->maskedFill(*mask, value);
  } else if (thpp::isFloat(type)) {
    auto value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(t)->maskedFill(*mask, value);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorMaskedCopy(rpc::RPCMessage& raw_message) {
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  thpp::Tensor *mask = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  t->maskedCopy(*mask, *src);
}

static void tensorMaskedSelect(rpc::RPCMessage& raw_message) {
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  thpp::Tensor *mask = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  t->maskedSelect(*src, *mask);
}

static void tensorNonzero(rpc::RPCMessage& raw_message) {
  thpp::Tensor *subscript = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->nonzero(*subscript);
  long long numel = subscript->sizes().size() > 0 ? subscript->sizes()[0] : 0;
  sendValueToMaster(numel);
}

static void tensorIndexSelect(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  thpp::Tensor *index = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->indexSelect(*src, dim, *index);
}

static void tensorIndexCopy(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  thpp::Tensor *index = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->indexCopy(dim, *index, *src);
}

static void tensorIndexAdd(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  thpp::Tensor *index = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->indexAdd(dim, *index, *src);
}

static void tensorIndexFill(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  thpp::Tensor *index = unpackRetrieveTensor(raw_message);
  thpp::Type type = peekType(raw_message);
  if (thpp::isInteger(type)) {
    auto val = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(tensor)->indexFill(dim, *index, val);
  } else if (thpp::isFloat(type)) {
    auto val = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(tensor)->indexFill(dim, *index, val);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorDiag(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  int k = unpackInteger(raw_message);
  finalize(raw_message);
  r->diag(*t, k);
}

static void tensorEye(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  long n = unpackInteger(raw_message);
  long m = unpackInteger(raw_message);
  finalize(raw_message);
  tensor->eye(n, m);
}

static void tensorRange(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Type type = peekType(raw_message);
  if (thpp::isInteger(type)) {
    long long xmin = unpackInteger(raw_message);
    long long xmax = unpackInteger(raw_message);
    long long step = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(r)->range(xmin, xmax, step);
  } else if (thpp::isFloat(type)) {
    double xmin = unpackFloat(raw_message);
    double xmax = unpackFloat(raw_message);
    double step = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(r)->range(xmin, xmax, step);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorRandperm(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  long n = unpackInteger(raw_message);
  finalize(raw_message);
  r->randperm(*_generator, n);
}

static void tensorSort(rpc::RPCMessage& raw_message) {
  thpp::Tensor *rt = unpackRetrieveTensor(raw_message);
  thpp::Tensor *ri = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int desc = unpackInteger(raw_message);
  finalize(raw_message);
  rt->sort(*ri, *tensor, dimension, desc);
}

static void tensorTopk(rpc::RPCMessage& raw_message) {
  thpp::Tensor *rt = unpackRetrieveTensor(raw_message);
  thpp::Tensor *ri = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  long k = unpackInteger(raw_message);
  int dimension = unpackInteger(raw_message);
  int dir = unpackInteger(raw_message);
  int sorted = unpackInteger(raw_message);
  finalize(raw_message);
  rt->topk(*ri, *tensor, k, dimension, dir, sorted);
}

static void tensorTril(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  long k = unpackInteger(raw_message);
  finalize(raw_message);
  r->tril(*t, k);
}

static void tensorTriu(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  long k = unpackInteger(raw_message);
  finalize(raw_message);
  r->triu(*t, k);
}

static void tensorCatArray(rpc::RPCMessage& raw_message) {
  thpp::Tensor *result = unpackRetrieveTensor(raw_message);
  int numInputs = unpackInteger(raw_message);
  std::vector<thpp::Tensor*> inputs(numInputs);
  for (std::size_t i = 0; i < numInputs; i++)
    inputs[i] = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  result->catArray(inputs, dimension);
}

static void tensorEqual(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ta = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tb = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  long long response = ta->equal(*tb);
  sendValueToMaster(response);
}

static void tensorTpow(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  if (thpp::isInteger(r->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(r)->tpow(value, *t);
  } else if (thpp::isFloat(t->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(r)->tpow(value, *t);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME, METHODNAME)                   \
  static void tensor##NAME##Value(rpc::RPCMessage& raw_message) {    \
    thpp::Tensor *r = unpackRetrieveTensor(raw_message);             \
    thpp::Tensor *t = unpackRetrieveTensor(raw_message);             \
    if (thpp::isInteger(t->type())) {                                \
      long long value = unpackInteger(raw_message);                  \
      finalize(raw_message);                                         \
      dynamic_cast<thpp::IntTensor*>(t)->METHODNAME##Value(*r, value);     \
    } else if (thpp::isFloat(t->type())) {                           \
      double value = unpackFloat(raw_message);                       \
      finalize(raw_message);                                         \
      dynamic_cast<thpp::FloatTensor*>(t)->METHODNAME##Value(*r, value);   \
    } else {                                                         \
      throw std::runtime_error("expected scalar type");              \
    }                                                                \
  }                                                                  \
  static void tensor##NAME##ValueT(rpc::RPCMessage& raw_message) {   \
    thpp::Tensor *r = unpackRetrieveTensor(raw_message);             \
    thpp::Tensor *t = unpackRetrieveTensor(raw_message);             \
    if (thpp::isInteger(t->type())) {                                \
      long long value = unpackInteger(raw_message);                  \
      finalize(raw_message);                                         \
      dynamic_cast<thpp::IntTensor*>(r)->METHODNAME##Value(*t, value);     \
    } else if (thpp::isFloat(t->type())) {                           \
      double value = unpackFloat(raw_message);                       \
      finalize(raw_message);                                         \
      dynamic_cast<thpp::FloatTensor*>(r)->METHODNAME##Value(*t, value);   \
    } else {                                                         \
      throw std::runtime_error("expected scalar type");              \
    }                                                                \
  }                                                                  \
  static void tensor##NAME##Tensor(rpc::RPCMessage& raw_message) {   \
    thpp::Tensor *r = unpackRetrieveTensor(raw_message);             \
    thpp::Tensor *ta = unpackRetrieveTensor(raw_message);            \
    thpp::Tensor *tb = unpackRetrieveTensor(raw_message);            \
    finalize(raw_message);                                           \
    ta->METHODNAME##Tensor(*r, *tb);                                 \
  }                                                                  \
  static void tensor##NAME##TensorT(rpc::RPCMessage& raw_message) {  \
    thpp::Tensor *r = unpackRetrieveTensor(raw_message);             \
    thpp::Tensor *ta = unpackRetrieveTensor(raw_message);            \
    thpp::Tensor *tb = unpackRetrieveTensor(raw_message);            \
    finalize(raw_message);                                           \
    r->METHODNAME##TensorT(*ta, *tb);                                \
  }                                                                  \

TENSOR_IMPLEMENT_LOGICAL(Lt,lt)
TENSOR_IMPLEMENT_LOGICAL(Gt,lt)
TENSOR_IMPLEMENT_LOGICAL(Le,le)
TENSOR_IMPLEMENT_LOGICAL(Ge,ge)
TENSOR_IMPLEMENT_LOGICAL(Eq,eq)
TENSOR_IMPLEMENT_LOGICAL(Ne,ne)

#undef TENSOR_IMPLEMENT_LOGICAL

#define TENSOR_IMPLEMENT_POINTWISE_FUNCTION(NAME, METHODNAME) \
  static void tensor##NAME(rpc::RPCMessage& raw_message) {    \
    thpp::Tensor *r = unpackRetrieveTensor(raw_message);      \
    thpp::Tensor *t = unpackRetrieveTensor(raw_message);      \
    finalize(raw_message);                                    \
    r->METHODNAME(*t);                                        \
  }                                                           \

#define TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION(NAME, METHODNAME) \
  static void tensor##NAME(rpc::RPCMessage& raw_message) {          \
    thpp::Tensor *r = unpackRetrieveTensor(raw_message);            \
    thpp::Tensor *t = unpackRetrieveTensor(raw_message);            \
    if (thpp::isInteger(t->type())) {                               \
      long long value = unpackInteger(raw_message);                 \
      finalize(raw_message);                                        \
      dynamic_cast<thpp::IntTensor*>(r)->METHODNAME(*t, value);     \
    } else if (thpp::isFloat(t->type())) {                          \
      double value = unpackFloat(raw_message);                      \
      finalize(raw_message);                                        \
      dynamic_cast<thpp::FloatTensor*>(r)->METHODNAME(*t, value);   \
    } else {                                                        \
      throw std::runtime_error("expected scalar type");             \
    }                                                               \
  }                                                                 \

TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Abs,abs)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Sigmoid,sigmoid)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Log,log)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Log1p,log1p)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Exp,exp)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Cos,cos)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Acos,acos)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Cosh,cosh)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Sin,sin)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Asin,asin)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Sinh,sinh)

TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Tan,tan)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Atan,atan)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Tanh,tanh)
TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION(Pow,pow)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Sqrt,sqrt)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Rsqrt,rsqrt)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Ceil,ceil)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Floor,floor)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Round,round)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Trunc,trunc)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Frac,frac)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Neg,neg)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Cinv,cinv)

#undef TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION
#undef TENSOR_IMPLEMENT_POINTWISE_FUNCTION

static void tensorAtan2(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tx = unpackRetrieveTensor(raw_message);
  thpp::Tensor *ty = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  r->atan2(*tx, *ty);
}

static void tensorLerp(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  thpp::Tensor *b = unpackRetrieveTensor(raw_message);
  if (thpp::isInteger(r->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(r)->lerp(*a, *b, value);
  } else if (thpp::isFloat(r->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(r)->lerp(*a, *b, value);
  } else {
    throw std::runtime_error("expected scalar type");
  }
}

static void tensorMean(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  r->mean(*t, dimension, keepdim);
}

static void tensorStd(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int flag = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  r->std(*t, dimension, flag, keepdim);
}

static void tensorVar(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int flag = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  r->var(*t, dimension, flag, keepdim);
}

static void tensorNorm(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Tensor *t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  if (thpp::isInteger(r->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(r)->norm(*t, value, dimension, keepdim);
  } else if (thpp::isFloat(r->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(r)->norm(*t, value, dimension, keepdim);
  } else {
    throw std::runtime_error("expected scalar type");
  }
}

static void tensorNormall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(r->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);

    long long response = dynamic_cast<thpp::IntTensor*>(r)->normall(value);
    sendValueToMaster(response);
  } else if (thpp::isFloat(r->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);

    double response = dynamic_cast<thpp::FloatTensor*>(r)->normall(value);
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorRenorm(rpc::RPCMessage& raw_message) {
  thpp::Tensor *res = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);

  if (thpp::isInteger(res->type())) {
    long long value = unpackInteger(raw_message);
    long long maxnorm = unpackInteger(raw_message);
    finalize(raw_message);

    dynamic_cast<thpp::IntTensor*>(res)->renorm(*src, value, dimension, maxnorm);
  } else if (thpp::isFloat(res->type())) {
    double value = unpackFloat(raw_message);
    double maxnorm = unpackFloat(raw_message);
    finalize(raw_message);

    dynamic_cast<thpp::FloatTensor*>(res)->renorm(*src, value, dimension, maxnorm);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDist(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Tensor *src = unpackRetrieveTensor(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long value = unpackInteger(raw_message);
    finalize(raw_message);

    long long response = dynamic_cast<thpp::IntTensor*>(tensor)->dist(*src, value);
    sendValueToMaster(response);
  } else if (thpp::isFloat(tensor->type())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);

    double response = dynamic_cast<thpp::FloatTensor*>(tensor)->dist(*src, value);
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMeanall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long response = dynamic_cast<thpp::IntTensor*>(tensor)->meanall();
    sendValueToMaster(response);
  } else if (thpp::isFloat(tensor->type())) {
    double response = dynamic_cast<thpp::FloatTensor*>(tensor)->meanall();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorVarall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long response = dynamic_cast<thpp::IntTensor*>(tensor)->varall();
    sendValueToMaster(response);
  } else if (thpp::isFloat(tensor->type())) {
    double response = dynamic_cast<thpp::FloatTensor*>(tensor)->varall();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorStdall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (thpp::isInteger(tensor->type())) {
    long long response = dynamic_cast<thpp::IntTensor*>(tensor)->stdall();
    sendValueToMaster(response);
  } else if (thpp::isFloat(tensor->type())) {
    double response = dynamic_cast<thpp::FloatTensor*>(tensor)->stdall();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorLinspace(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  long n = unpackInteger(raw_message);

  if (thpp::isInteger(r->type())) {
    long long a = unpackInteger(raw_message);
    long long b = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(r)->linspace(a, b, n);
  } else if (thpp::isFloat(r->type())) {
    double a = unpackFloat(raw_message);
    double b = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(r)->linspace(a, b, n);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorLogspace(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  long n = unpackInteger(raw_message);

  if (thpp::isInteger(r->type())) {
    long long a = unpackInteger(raw_message);
    long long b = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(r)->logspace(a, b, n);
  } else if (thpp::isFloat(r->type())) {
    double a = unpackFloat(raw_message);
    double b = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(r)->logspace(a, b, n);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorRand(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  r->rand(*_generator, size);
  THLongStorage_free(size);
}

static void tensorRandn(rpc::RPCMessage& raw_message) {
  thpp::Tensor *r = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  r->randn(*_generator, size);
  THLongStorage_free(size);
}

static void tensorHistc(rpc::RPCMessage& raw_message) {
  thpp::Tensor *hist = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  long nbins = unpackInteger(raw_message);

  if (thpp::isInteger(hist->type())) {
    long long minvalue = unpackInteger(raw_message);
    long long maxvalue = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(hist)->histc(*tensor, nbins, minvalue, maxvalue);
  } else if (thpp::isFloat(hist->type())) {
    double minvalue = unpackFloat(raw_message);
    double maxvalue = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(hist)->histc(*tensor, nbins, minvalue, maxvalue);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorBhistc(rpc::RPCMessage& raw_message) {
  thpp::Tensor *hist = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  long nbins = unpackInteger(raw_message);

  if (thpp::isInteger(hist->type())) {
    long long minvalue = unpackInteger(raw_message);
    long long maxvalue = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(hist)->bhistc(*tensor, nbins, minvalue, maxvalue);
  } else if (thpp::isFloat(hist->type())) {
    double minvalue = unpackFloat(raw_message);
    double maxvalue = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(hist)->bhistc(*tensor, nbins, minvalue, maxvalue);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorLogicalall(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  long long response = tensor->logicalall();
  sendValueToMaster(response);
}

static void tensorLogicalany(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  long long response = tensor->logicalany();
  sendValueToMaster(response);
}
