
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
    auto xmin = unpackInteger(raw_message);
    auto xmax = unpackInteger(raw_message);
    auto step = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(r)->range(xmin, xmax, step);
  } else if (thpp::isFloat(type)) {
    auto xmin = unpackFloat(raw_message);
    auto xmax = unpackFloat(raw_message);
    auto step = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(r)->range(xmin, xmax, step);
  } else {
    throw std::runtime_error("expected a scalar type");
  }
}

static void tensorRandperm(rpc::RPCMessage& raw_message) {
  throw std::runtime_error("randperm is not available yet");
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

#define TENSOR_IMPLEMENT_LOGICAL(NAME, METHODNAME)                   \
  static void tensor##NAME##Value(rpc::RPCMessage& raw_message) {    \
    thpp::Tensor *r = unpackRetrieveTensor(raw_message);             \
    thpp::Tensor *t = unpackRetrieveTensor(raw_message);             \
    if (thpp::isInteger(t->type())) {                                \
      long long value = unpackInteger(raw_message);                  \
      finalize(raw_message);                                         \
      dynamic_cast<thpp::IntTensor*>(t)->METHODNAME##Value(*r, value);     \
    } else if (thpp::isFloat(t->type())) {                           \
      double value = unpackInteger(raw_message);                     \
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
      double value = unpackInteger(raw_message);                     \
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

#undef TENSOR_IMPLEMENT_POINTWISE_FUNCTION
