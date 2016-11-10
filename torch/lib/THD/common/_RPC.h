#pragma once

#include "../master/THDTensor.h"
#include "_Tensor.h"
#include <cstdint>
#include <string>

namespace thd { namespace rpc {

using function_id_type = uint16_t;

class RPCMessage {
public:
  RPCMessage();
  RPCMessage(std::string &str);
  const char *data();
  const char *read(size_t num_bytes);
  bool isEmpty();
private:
  std::string _msg;
  size_t _offset;
};

uint16_t unpackFunctionId(RPCMessage& raw_message);
uint16_t unpackArgCount(RPCMessage& raw_message);
Tensor *unpackTensor(RPCMessage& raw_message);
double unpackFloat(RPCMessage& raw_message);
long long unpackInteger(RPCMessage& raw_message);
template <typename ...Args>
RPCMessage packMessage(function_id_type fid, uint16_t num_args,
    const Args&... args);

}} // namespace rpc, thd 

#include "_RPC-inl.h"
