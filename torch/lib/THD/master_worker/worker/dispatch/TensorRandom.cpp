
static void tensorRandom(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  finalize(raw_message);
  tensor->random(*_generator);
}

static void tensorGeometric(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double p = unpackFloat(raw_message);
  finalize(raw_message);
  tensor->geometric(*_generator, p);
}

static void tensorBernoulli(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double p = unpackFloat(raw_message);
  finalize(raw_message);
  tensor->bernoulli(*_generator, p);
}

static void tensorBernoulli_FloatTensor(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  thpp::Tensor *p = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->bernoulli_FloatTensor(*_generator, *p);
}

static void tensorBernoulli_DoubleTensor(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  thpp::Tensor *p = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  tensor->bernoulli_DoubleTensor(*_generator, *p);
}

static void tensorUniform(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double a = unpackFloat(raw_message);
  double b = unpackFloat(raw_message);
  finalize(raw_message);
  tensor->uniform(*_generator, a, b);
}

static void tensorNormal(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double mean = unpackFloat(raw_message);
  double stdv = unpackFloat(raw_message);
  finalize(raw_message);
  tensor->normal(*_generator, mean, stdv);
}

static void tensorExponential(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double lambda = unpackFloat(raw_message);
  finalize(raw_message);
  tensor->exponential(*_generator, lambda);
}

static void tensorCauchy(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double median = unpackFloat(raw_message);
  double sigma = unpackFloat(raw_message);
  finalize(raw_message);
  tensor->cauchy(*_generator, median, sigma);
}

static void tensorLogNormal(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  double mean = unpackFloat(raw_message);
  double stdv = unpackFloat(raw_message);
  finalize(raw_message);
  tensor->logNormal(*_generator, mean, stdv);
}

static void tensorMultinomial(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  thpp::Generator *_generator = unpackRetrieveGenerator(raw_message);
  thpp::Tensor *prob_dist = unpackRetrieveTensor(raw_message);
  int n_sample = unpackInteger(raw_message);
  int with_replacement = unpackInteger(raw_message);
  finalize(raw_message);
  prob_dist->multinomial(*tensor, *_generator, n_sample, with_replacement);
}
