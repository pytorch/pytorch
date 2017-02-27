
static void sendTensor(rpc::RPCMessage& raw_message) {
  thpp::Tensor *tensor = unpackRetrieveTensor(raw_message);
  int dst_rank = unpackInteger(raw_message);
  finalize(raw_message);
  dataChannel->send(*tensor, dst_rank);
}

static void sendStorage(rpc::RPCMessage& raw_message) {
  thpp::Storage *storage = unpackRetrieveStorage(raw_message);
  int dst_rank = unpackInteger(raw_message);
  finalize(raw_message);
  fprintf(stderr, "sending storage (to be implemented)\n");
}
