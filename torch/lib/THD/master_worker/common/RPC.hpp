#pragma once

#include "../../base/Tensor.hpp"
#include "../master/THDTensor.h"
#include "ByteArray.hpp"

#include <cstdint>
#include <string>

namespace thd { namespace rpc {

using function_id_type = std::uint16_t;

class RPCMessage {
public:
  RPCMessage();
  RPCMessage(char* str, std::size_t size);
  RPCMessage(const ByteArray& str);
  RPCMessage(ByteArray&& str);

  ByteArray& bytes(); // Raw data.
  const char* data() const; // Offset data.
  bool isEmpty() const;
  std::size_t remaining() const; // Length of msg left to read.
  const char* read(std::size_t num_bytes);

  static void freeMessage(void *data, void *hint);
private:

  ByteArray _msg;
  std::size_t _offset;
};

template <typename ...Args>
RPCMessage packMessage(function_id_type fid,
                       std::uint16_t num_args,
                       const Args&... args);

std::uint16_t unpackArgCount(RPCMessage& raw_message);
double unpackFloat(RPCMessage& raw_message);
std::uint16_t unpackFunctionId(RPCMessage& raw_message);
long long unpackInteger(RPCMessage& raw_message);
Tensor* unpackTensor(RPCMessage& raw_message);

}} // namespace rpc, thd

#include "RPC-inl.hpp"
