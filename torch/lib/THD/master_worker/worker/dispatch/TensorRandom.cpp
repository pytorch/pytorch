
static void tensorRandom(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  finalize(raw_message);
  tensor.random_(_generator);
}

static void tensorGeometric(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double p = unpackFloat(raw_message);
  finalize(raw_message);
  tensor.geometric_(p, _generator);
}

static void tensorBernoulli(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double p = unpackFloat(raw_message);
  finalize(raw_message);

  throw std::runtime_error("bernoulli not yet wrapped in ATen");
  /* tensor.bernoulli_(p, _generator); */
}

static void tensorBernoulli_FloatTensor(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  at::Tensor p = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  throw std::runtime_error("bernoulli not yet wrapped in ATen");
  /* tensor->bernoulli_FloatTensor(_generator, *p); */
}

static void tensorBernoulli_DoubleTensor(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  at::Tensor p = unpackRetrieveTensor(raw_message);
  finalize(raw_message);

  throw std::runtime_error("bernoulli not yet wrapped in ATen");
  /* tensor->bernoulli_DoubleTensor(_generator, *p); */
}

static void tensorUniform(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double a = unpackFloat(raw_message);
  double b = unpackFloat(raw_message);
  finalize(raw_message);
  tensor.uniform_(a, b, _generator);
}

static void tensorNormal(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double mean = unpackFloat(raw_message);
  double stdv = unpackFloat(raw_message);
  finalize(raw_message);
  tensor.normal_(mean, stdv, _generator);
}

static void tensorExponential(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double lambda = unpackFloat(raw_message);
  finalize(raw_message);
  tensor.exponential_(lambda, _generator);
}

static void tensorCauchy(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double median = unpackFloat(raw_message);
  double sigma = unpackFloat(raw_message);
  finalize(raw_message);
  tensor.cauchy_(median, sigma, _generator);
}

static void tensorLogNormal(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double mean = unpackFloat(raw_message);
  double stdv = unpackFloat(raw_message);
  finalize(raw_message);
  tensor.log_normal_(mean, stdv, _generator);
}

static void tensorMultinomial(rpc::RPCMessage& raw_message) {
  at::Tensor tensor = unpackRetrieveTensor(raw_message);
  at::Generator *_generator = unpackRetrieveGenerator(raw_message);
  at::Tensor prob_dist = unpackRetrieveTensor(raw_message);
  int n_sample = unpackInteger(raw_message);
  int with_replacement = unpackInteger(raw_message);
  finalize(raw_message);
  at::multinomial_out(tensor, prob_dist, n_sample, with_replacement, _generator);
}
