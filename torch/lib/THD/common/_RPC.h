#pragma once

#include "../master/THDTensor.h"
#include "_ByteArray.h"
#include "_Tensor.h"

#include <cstdint>
#include <string>

namespace thd { namespace rpc {

using function_id_type = uint16_t;

class RPCMessage {
public:
  RPCMessage();
  RPCMessage(char* str, size_t size);
  RPCMessage(const ByteArray& str);
  RPCMessage(ByteArray&& str);

  ByteArray& bytes(); // Raw data.
  const char* data() const; // Offset data.
  bool isEmpty() const;
  size_t remaining() const; // Length of msg left to read.
  const char* read(size_t num_bytes);

  static void freeMessage(void *data, void *hint);
private:

  ByteArray _msg;
  size_t _offset;
};

template <typename ...Args>
RPCMessage packMessage(function_id_type fid,
                       uint16_t num_args,
                       const Args&... args);

uint16_t unpackArgCount(RPCMessage& raw_message);
double unpackFloat(RPCMessage& raw_message);
uint16_t unpackFunctionId(RPCMessage& raw_message);
long long unpackInteger(RPCMessage& raw_message);
Tensor* unpackTensor(RPCMessage& raw_message);

}} // namespace rpc, thd

#include "_RPC-inl.h"
