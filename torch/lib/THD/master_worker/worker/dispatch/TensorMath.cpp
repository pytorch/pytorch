
static void tensorFill(rpc::RPCMessage& raw_message) {
  at::Tensor t = unpackRetrieveTensor(raw_message);
  RPCType type = peekType(raw_message);
  if (isInteger(type)) {
    auto value = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    t.fill_(value);
  } else if (isFloat(type)) {
    auto value = unpackFloat(raw_message);
    finalize(raw_message);
    t.fill_(value);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorMaskedFill(rpc::RPCMessage& raw_message) {
  at::Tensor t = unpackRetrieveTensor(raw_message);
  at::Tensor mask = unpackRetrieveTensor(raw_message);
  RPCType type = peekType(raw_message);
  if (isInteger(type)) {
    auto value = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    t.masked_fill_(mask, value);
  } else if (isFloat(type)) {
    auto value = unpackFloat(raw_message);
    finalize(raw_message);
    t.masked_fill_(mask, value);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorMaskedCopy(rpc::RPCMessage& raw_message) {
  at::Tensor t = unpackRetrieveTensor(raw_message);
  at::Tensor mask = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  t.masked_scatter_(mask, src);
}

static void tensorMaskedSelect(rpc::RPCMessage& raw_message) {
  at::Tensor t = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  at::Tensor mask = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::masked_select_out(t, src, mask);
}

static void tensorNonzero(rpc::RPCMessage& raw_message) {
  at::Tensor subscript = unpackRetrieveTensor(raw_message);
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::nonzero_out(subscript, tensor);
  int64_t numel = subscript.sizes().size() > 0 ? subscript.sizes()[0] : 0;
  sendValueToMaster(numel);
}

static void tensorIndexSelect(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  at::Tensor index = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::index_select_out(tensor, src, dim, index);
}

static void tensorIndexCopy(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  at::Tensor index = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor.index_copy_(dim, index, src);
}

static void tensorIndexAdd(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  at::Tensor index = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor.index_add_(dim, index, src);
}

static void tensorIndexFill(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int dim = unpackInteger(raw_message);
  at::Tensor index = unpackRetrieveTensor(raw_message);
  RPCType type = peekType(raw_message);
  if (isInteger(type)) {
    auto val = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    tensor.index_fill_(dim, index, val);
  } else if (isFloat(type)) {
    auto val = unpackFloat(raw_message);
    finalize(raw_message);
    tensor.index_fill_(dim, index, val);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorDiag(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  int k = unpackInteger(raw_message);
  finalize(raw_message);
  at::diag_out(r, t, k);
}

static void tensorEye(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int64_t n = unpackInteger(raw_message);
  int64_t m = unpackInteger(raw_message);
  finalize(raw_message);
  at::eye_out(tensor, n, m);
}

static void tensorRange(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  RPCType type = peekType(raw_message);
  if (isInteger(type)) {
    int64_t xmin = unpackInteger(raw_message);
    int64_t xmax = unpackInteger(raw_message);
    int64_t step = unpackInteger(raw_message);
    finalize(raw_message);
    at::range_out(r, xmin, xmax, step);
  } else if (isFloat(type)) {
    double xmin = unpackFloat(raw_message);
    double xmax = unpackFloat(raw_message);
    double step = unpackFloat(raw_message);
    finalize(raw_message);
    at::range_out(r, xmin, xmax, step);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorRandperm(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  int64_t n = unpackInteger(raw_message);
  finalize(raw_message);
  at::randperm_out(r, n, _generator);
}

static void tensorSort(rpc::RPCMessage& raw_message) {
  at::Tensor rt = unpackRetrieveTensor(raw_message);
  at::Tensor ri = unpackRetrieveTensor(raw_message);
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int desc = unpackInteger(raw_message);
  finalize(raw_message);
  at::sort_out(rt, ri, tensor, dimension, desc);
}

static void tensorTopk(rpc::RPCMessage& raw_message) {
  at::Tensor rt = unpackRetrieveTensor(raw_message);
  at::Tensor ri = unpackRetrieveTensor(raw_message);
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int64_t k = unpackInteger(raw_message);
  int dimension = unpackInteger(raw_message);
  int dir = unpackInteger(raw_message);
  int sorted = unpackInteger(raw_message);
  finalize(raw_message);
  at::topk_out(rt, ri, tensor, k, dimension, dir, sorted);
}

static void tensorTril(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  int64_t k = unpackInteger(raw_message);
  finalize(raw_message);
  at::tril_out(r, t, k);
}

static void tensorTriu(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  int64_t k = unpackInteger(raw_message);
  finalize(raw_message);
  at::triu_out(r, t, k);
}

static void tensorCatArray(rpc::RPCMessage& raw_message) {
  at::Tensor result = unpackRetrieveTensor(raw_message);
  int numInputs = unpackInteger(raw_message);
  std::vector<at::Tensor> inputs(numInputs);
  for (std::size_t i = 0; i < numInputs; i++)
    inputs[i] = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  finalize(raw_message);
  at::cat_out(result, inputs, dimension);
}

static void tensorEqual(rpc::RPCMessage& raw_message) {
  at::Tensor ta = unpackRetrieveTensor(raw_message);
  at::Tensor tb = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  int64_t response = ta.equal(tb);
  sendValueToMaster(response);
}

static void tensorTpow(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  if (at::isIntegralType(t.type().scalarType())) {
    int64_t value = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    at::pow_out(r, t, value);
  } else if (at::isFloatingType(t.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::pow_out(r, t, value);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME, METHODNAME)                   \
  static void tensor##NAME##Value(rpc::RPCMessage& raw_message) {    \
    at::Tensor r = unpackRetrieveTensor(raw_message);             \
    at::Tensor t = unpackRetrieveTensor(raw_message);             \
    if (at::isIntegralType(t.type().scalarType())) {                 \
      int64_t value = (int64_t) unpackInteger(raw_message);                  \
      finalize(raw_message);                                         \
      at::METHODNAME##_out(t, r, value);                              \
    } else if (at::isFloatingType(t.type().scalarType())) {          \
      double value = unpackFloat(raw_message);                       \
      finalize(raw_message);                                         \
      at::METHODNAME##_out(t, r, value);                              \
    } else {                                                         \
      throw std::runtime_error("expected scalar type");              \
    }                                                                \
  }                                                                  \
  static void tensor##NAME##ValueT(rpc::RPCMessage& raw_message) {   \
    at::Tensor r = unpackRetrieveTensor(raw_message);             \
    at::Tensor t = unpackRetrieveTensor(raw_message);             \
    if (at::isIntegralType(t.type().scalarType())) {                 \
      int64_t value = (int64_t) unpackInteger(raw_message);                  \
      finalize(raw_message);                                         \
      at::METHODNAME##_out(r, t, value);                              \
    } else if (at::isFloatingType(t.type().scalarType())) {          \
      double value = unpackFloat(raw_message);                       \
      finalize(raw_message);                                         \
      at::METHODNAME##_out(r, t, value);                              \
    } else {                                                         \
      throw std::runtime_error("expected scalar type");              \
    }                                                                \
  }                                                                  \
  static void tensor##NAME##Tensor(rpc::RPCMessage& raw_message) {   \
    at::Tensor r = unpackRetrieveTensor(raw_message);             \
    at::Tensor ta = unpackRetrieveTensor(raw_message);            \
    at::Tensor tb = unpackRetrieveTensor(raw_message);            \
    finalize(raw_message);                                           \
    at::METHODNAME##_out(ta, r, tb);                              \
  }                                                                  \
  static void tensor##NAME##TensorT(rpc::RPCMessage& raw_message) {  \
    at::Tensor r = unpackRetrieveTensor(raw_message);             \
    at::Tensor ta = unpackRetrieveTensor(raw_message);            \
    at::Tensor tb = unpackRetrieveTensor(raw_message);            \
    finalize(raw_message);                                           \
    at::METHODNAME##_out(r, ta, tb);                               \
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
    at::Tensor r = unpackRetrieveTensor(raw_message);      \
    at::Tensor t = unpackRetrieveTensor(raw_message);      \
    finalize(raw_message);                                    \
    at::METHODNAME##_out(r, t);                            \
  }                                                           \

#define TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION(NAME, METHODNAME) \
  static void tensor##NAME(rpc::RPCMessage& raw_message) {          \
    at::Tensor r = unpackRetrieveTensor(raw_message);            \
    at::Tensor t = unpackRetrieveTensor(raw_message);            \
    if (at::isIntegralType(t.type().scalarType())) {                 \
      int64_t value = (int64_t) unpackInteger(raw_message);                 \
      finalize(raw_message);                                        \
      at::METHODNAME##_out(r, t, value);                            \
    } else if (at::isFloatingType(t.type().scalarType())) {          \
      double value = unpackFloat(raw_message);                      \
      finalize(raw_message);                                        \
      at::METHODNAME##_out(r, t, value);                            \
    } else {                                                        \
      throw std::runtime_error("expected scalar type");             \
    }                                                               \
  }                                                                 \

TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Abs,abs)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Sigmoid,sigmoid)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Log,log)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Log10,log10)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Log1p,log1p)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Log2,log2)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Exp,exp)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Expm1,expm1)
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
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(Cinv,inverse)

#undef TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION
#undef TENSOR_IMPLEMENT_POINTWISE_FUNCTION

static void tensorAtan2(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor tx = unpackRetrieveTensor(raw_message);
  at::Tensor ty = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::atan2_out(r, tx, ty);
}

static void tensorLerp(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  at::Tensor b = unpackRetrieveTensor(raw_message);
  if (at::isIntegralType(r.type().scalarType())) {
    int64_t value = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    at::lerp_out(r, a, b, value);
  } else if (at::isFloatingType(r.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::lerp_out(r, a, b, value);
  } else {
    throw std::runtime_error("expected scalar type");
  }
}

static void tensorMean(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::mean_out(r, t, dimension, keepdim);
}

static void tensorStd(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int biased = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::std_out(r, t, dimension, biased, keepdim);
}

static void tensorVar(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int biased = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::var_out(r, t, dimension, biased, keepdim);
}

static void tensorNorm(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Tensor t = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  if (at::isIntegralType(r.type().scalarType())) {
    int64_t value = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    at::norm_out(r, t, value, dimension, keepdim);
  } else if (at::isFloatingType(r.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);
    at::norm_out(r, t, value, dimension, keepdim);
  } else {
    throw std::runtime_error("expected scalar type");
  }
}

static void tensorNormall(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(r.type().scalarType())) {
    int64_t value = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);

    int64_t response = r.norm(value).toCLong();
    sendValueToMaster(response);
  } else if (at::isFloatingType(r.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);

    double response = r.norm(value).toCDouble();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorRenorm(rpc::RPCMessage& raw_message) {
  at::Tensor res = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);

  if (at::isIntegralType(res.type().scalarType())) {
    int64_t value = (int64_t) unpackInteger(raw_message);
    int64_t maxnorm = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);

    at::renorm_out(res, src, value, dimension, maxnorm);
  } else if (at::isFloatingType(res.type().scalarType())) {
    double value = unpackFloat(raw_message);
    double maxnorm = unpackFloat(raw_message);
    finalize(raw_message);

    at::renorm_out(res, src, value, dimension, maxnorm);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorDist(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t value = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);

    int64_t response = src.dist(tensor, value).toCLong();
    sendValueToMaster(response);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double value = unpackFloat(raw_message);
    finalize(raw_message);

    double response = src.dist(tensor, value).toCDouble();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorMeanall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t response = tensor.mean().toCLong();
    sendValueToMaster(response);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double response = tensor.mean().toCLong();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorVarall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int biased = unpackInteger(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t response = tensor.var((bool)biased).toCLong();
    sendValueToMaster(response);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double response = tensor.var((bool)biased).toCDouble();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorStdall(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int biased = unpackInteger(raw_message);
  finalize(raw_message);

  if (at::isIntegralType(tensor.type().scalarType())) {
    int64_t response = tensor.std((bool)biased).toCLong();
    sendValueToMaster(response);
  } else if (at::isFloatingType(tensor.type().scalarType())) {
    double response = tensor.std((bool)biased).toCDouble();
    sendValueToMaster(response);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorLinspace(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  int64_t n = unpackInteger(raw_message);

  if (at::isIntegralType(r.type().scalarType())) {
    int64_t a = (int64_t) unpackInteger(raw_message);
    int64_t b = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    at::linspace_out(r, a, b, n);
  } else if (at::isFloatingType(r.type().scalarType())) {
    double a = unpackFloat(raw_message);
    double b = unpackFloat(raw_message);
    finalize(raw_message);
    at::linspace_out(r, a, b, n);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorLogspace(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  int64_t n = unpackInteger(raw_message);

  if (at::isIntegralType(r.type().scalarType())) {
    int64_t a = (int64_t) unpackInteger(raw_message);
    int64_t b = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    at::logspace_out(r, a, b, n);
  } else if (at::isFloatingType(r.type().scalarType())) {
    double a = unpackFloat(raw_message);
    double b = unpackFloat(raw_message);
    finalize(raw_message);
    at::logspace_out(r, a, b, n);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorRand(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  at::ArrayRef<int64_t> sizeRef(THLongStorage_data(size), THLongStorage_size(size));
  at::rand_out(r, sizeRef, _generator);
  THLongStorage_free(size);
}

static void tensorRandn(rpc::RPCMessage& raw_message) {
  at::Tensor r = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  THLongStorage *size = unpackTHLongStorage(raw_message);
  finalize(raw_message);
  at::ArrayRef<int64_t> sizeRef(THLongStorage_data(size), THLongStorage_size(size));
  at::randn_out(r, sizeRef, _generator);
  THLongStorage_free(size);
}

static void tensorHistc(rpc::RPCMessage& raw_message) {
  at::Tensor hist = unpackRetrieveTensor(raw_message);
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  int64_t nbins = unpackInteger(raw_message);

  if (at::isIntegralType(hist.type().scalarType())) {
    int64_t minvalue = (int64_t) unpackInteger(raw_message);
    int64_t maxvalue = (int64_t) unpackInteger(raw_message);
    finalize(raw_message);
    at::histc_out(hist, tensor, nbins, minvalue, maxvalue);
  } else if (at::isFloatingType(hist.type().scalarType())) {
    double minvalue = unpackFloat(raw_message);
    double maxvalue = unpackFloat(raw_message);
    finalize(raw_message);
    at::histc_out(hist, tensor, nbins, minvalue, maxvalue);
  } else {
    throw std::invalid_argument("expected scalar type");
  }
}

static void tensorLogicalAndAll(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  int64_t response = tensor.all().toCLong();
  sendValueToMaster(response);
}

static void tensorLogicalAnyAll(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  int64_t response = tensor.any().toCLong();
  sendValueToMaster(response);
}

static void tensorLogicalAnd(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::all_out(tensor, src, dimension, keepdim);
}

static void tensorLogicalAny(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Tensor src = unpackRetrieveTensor(raw_message);
  int dimension = unpackInteger(raw_message);
  int keepdim = unpackInteger(raw_message);
  finalize(raw_message);
  at::any_out(tensor, src, dimension, keepdim);
}
