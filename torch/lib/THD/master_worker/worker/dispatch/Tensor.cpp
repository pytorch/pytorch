
static void construct(rpc::RPCMessage& raw_message) {
  // TODO: assert_empty(raw_message)
  Type type = rpc::unpackType(raw_message);
  thd::object_id_type id = rpc::unpackTensor(raw_message);
  workerTensors.emplace(
    id,
    createTensor(type)
  );
}

static void constructWithSize(rpc::RPCMessage& raw_message) {
  // TODO: assert_empty(raw_message)
  Type type = rpc::unpackType(raw_message);
  object_id_type id = rpc::unpackTensor(raw_message);
  THLongStorage *sizes = rpc::unpackTHLongStorage(raw_message);
  THLongStorage *strides = rpc::unpackTHLongStorage(raw_message);
}

static void add(rpc::RPCMessage& raw_message) {
//THTensor& result = parse_tensor(raw_message);
  //THTensor& source = parse_tensor(raw_message);
  //double x = parse_scalar(raw_message);
  //assert_end(raw_message);
  //result.add(source, x);
}

static void free(rpc::RPCMessage& raw_message) {
  object_id_type tensor_id = unpackInteger(raw_message);
  (void)workerTensors.erase(tensor_id);
}

