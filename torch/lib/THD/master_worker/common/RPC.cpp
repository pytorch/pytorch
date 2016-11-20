#include "RPC.hpp"
#include "ByteArray.hpp"

#include <cstdarg>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace thd { namespace rpc {

RPCMessage::RPCMessage()
  : _msg(0)
  , _offset(0)
{}

RPCMessage::RPCMessage(char* str, size_t size)
  : _msg(str, size)
  , _offset(0)
{}

RPCMessage::RPCMessage(const ByteArray& str)
  : _msg(str)
  , _offset(0)
{}

RPCMessage::RPCMessage(ByteArray&& str)
  : _msg(std::move(str))
  , _offset(0)
{}

ByteArray& RPCMessage::bytes() {
  return _msg;
}

const char* RPCMessage::data() const {
  return _msg.data() + _offset;
}

bool RPCMessage::isEmpty() const {
  return _offset >= _msg.length();
}

size_t RPCMessage::remaining() const {
  return _msg.length() - _offset;
}

const char* RPCMessage::read(size_t num_bytes) {
  if (_offset + num_bytes > _msg.length())
    throw std::out_of_range("invalid access: out of bounds");
  const char* ret_val = _msg.data() + _offset;
  _offset += num_bytes;
  return ret_val;
}

void RPCMessage::freeMessage(void *data, void *hint) {
  delete static_cast<RPCMessage*>(hint);
}

namespace  {
////////////////////////////////////////////////////////////////////////////////

template <typename real, typename return_type = double>
return_type _readValue(RPCMessage& msg) {
  real ret_val;
  memcpy(&ret_val, msg.read(sizeof(real)), sizeof(real));
  return (return_type)ret_val;
}

////////////////////////////////////////////////////////////////////////////////
} // anonymous namespace

uint16_t unpackArgCount(RPCMessage& raw_message) {
  return _readValue<uint16_t, uint16_t>(raw_message);
}

double unpackFloat(RPCMessage& raw_message) {
  char type = *(raw_message.read(sizeof(char)));
  if (type == 'd')
    return _readValue<double>(raw_message);
  else if (type == 'f')
    return _readValue<float>(raw_message);
  throw std::invalid_argument("wrong real type in the raw message");
}

uint16_t unpackFunctionId(RPCMessage& raw_message) {
  return _readValue<uint16_t, uint16_t>(raw_message);
}

long long unpackInteger(RPCMessage& raw_message) {
  char type = *(raw_message.read(sizeof(char)));
  if (type == 'c')
    return _readValue<char, long long>(raw_message);
  else if (type == 'i')
    return _readValue<int, long long>(raw_message);
  else if (type == 'l')
    return _readValue<long, long long>(raw_message);
  else if (type == 'h')
    return _readValue<short, long long>(raw_message);
  else if (type == 'q')
    return _readValue<long long, long long>(raw_message);
  throw std::invalid_argument("wrong integer type in the raw message");
}

Tensor *unpackTensor(RPCMessage& raw_message) {
  char type = *raw_message.read(sizeof(char));
  if (type == 'T')
    return NULL; //_readValue<long long int>(raw_message); TODO
  throw std::invalid_argument("expected tensor in the raw message");
}

}} // namespace rpc, thd
