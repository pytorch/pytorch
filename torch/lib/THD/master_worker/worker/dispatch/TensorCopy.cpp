
static void tensorCopyFromMaster(rpc::RPCMessage& raw_message) {
  thpp::Tensor *data = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  dataChannel->receive(*data, 0);
}

static void tensorCopyFromWorker(rpc::RPCMessage& raw_message) {
  thpp::Tensor *data = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  dataChannel->send(*data, 0);
}