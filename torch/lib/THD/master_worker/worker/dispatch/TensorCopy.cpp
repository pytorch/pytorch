
static void tensorCopyTH(rpc::RPCMessage& raw_message) {
  thpp::Tensor *data = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  dataChannel->receive(*data, 0);
}

static void tensorCopyTHD(rpc::RPCMessage& raw_message) {
  thpp::Tensor *data = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  dataChannel->send(*data, 0);
}