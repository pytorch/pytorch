
static std::unique_ptr<thpp::Generator> createGenerator() {
  return std::unique_ptr<thpp::Generator>();
}

static void generatorNew(rpc::RPCMessage& raw_message) {
  object_id_type generator_id = unpackGenerator(raw_message);
  finalize(raw_message);
  workerGenerators.emplace(generator_id, createGenerator());
}

static void generatorFree(rpc::RPCMessage& raw_message) {
  object_id_type generator_id = unpackGenerator(raw_message);
  workerGenerators.erase(generator_id);
}

static void generatorCopy(rpc::RPCMessage& raw_message) {
  thpp::Generator* self = unpackRetrieveGenerator(raw_message);
  thpp::Generator* from = unpackRetrieveGenerator(raw_message);
  finalize(raw_message);
  self->copy(*from);
}

static void generatorSeed(rpc::RPCMessage& raw_message) {
  thpp::Generator* _generator = unpackRetrieveGenerator(raw_message);
  finalize(raw_message);
  int64_t response = _generator->seed();
  sendValueToMaster(response);
}

static void generatorManualSeed(rpc::RPCMessage& raw_message) {
  thpp::Generator* _generator = unpackRetrieveGenerator(raw_message);
  uint64_t seed = unpackInteger(raw_message);
  finalize(raw_message);
  _generator->manualSeed(seed);
}
