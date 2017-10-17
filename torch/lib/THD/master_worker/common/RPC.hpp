#pragma once
#include "../master/THDTensor.h"
#include "ByteArray.hpp"
#include "TH/THStorage.h"

#include <THPP/Type.hpp>
#include <cstdint>
#include <memory>
#include <string>

namespace thd {

using object_id_type = std::uint64_t;

namespace rpc {

using function_id_type = std::uint16_t;

class RPCMessage {
public:
  using size_type = ByteArray::size_type;
  RPCMessage();
  RPCMessage(char* str, std::size_t size);
  RPCMessage(const ByteArray& str);
  RPCMessage(ByteArray&& str);

  ByteArray& bytes(); // Raw data.
  const char* data() const; // Offset data.
  bool isEmpty() const;
  size_type remaining() const; // Length of the msg left to read.
  const char* read(std::size_t num_bytes);

private:
  ByteArray _msg;
  std::size_t _offset;
};

template <typename ...Args>
std::unique_ptr<RPCMessage> packMessage(function_id_type fid, const Args&... args);

thpp::Type unpackType(RPCMessage& raw_message);
thpp::Type peekType(RPCMessage& raw_message);
double unpackFloat(RPCMessage& raw_message);
function_id_type unpackFunctionId(RPCMessage& raw_message);
int64_t unpackInteger(RPCMessage& raw_message);
object_id_type unpackGenerator(RPCMessage& raw_message);
object_id_type unpackTensor(RPCMessage& raw_message);
object_id_type unpackStorage(RPCMessage& raw_message);
THLongStorage* unpackTHLongStorage(RPCMessage& raw_message);

}} // namespace rpc, thd

#include "RPC-inl.hpp"
