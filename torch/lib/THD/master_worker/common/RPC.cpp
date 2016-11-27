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

RPCMessage::RPCMessage(char* str, std::size_t size)
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

std::size_t RPCMessage::remaining() const {
  return _msg.length() - _offset;
}

const char* RPCMessage::read(std::size_t num_bytes) {
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

std::uint16_t unpackArgCount(RPCMessage& raw_message) {
  return _readValue<std::uint16_t, std::uint16_t>(raw_message);
}

double unpackFloat(RPCMessage& raw_message) {
  const TensorType& type = format_to_type.at(*(raw_message.read(sizeof(char))));
  if (type == TensorType::DOUBLE)
    return _readValue<double>(raw_message);
  else if (type == TensorType::FLOAT)
    return _readValue<float>(raw_message);

  throw std::invalid_argument("wrong real type in the raw message");
}

std::uint16_t unpackFunctionId(RPCMessage& raw_message) {
  return _readValue<std::uint16_t, std::uint16_t>(raw_message);
}

long long unpackInteger(RPCMessage& raw_message) {
  const TensorType& type = format_to_type.at(*(raw_message.read(sizeof(char))));
  if (type == TensorType::CHAR)
    return _readValue<char, long long>(raw_message);
  else if (type == TensorType::SHORT)
    return _readValue<short, long long>(raw_message);
  else if (type == TensorType::INT)
    return _readValue<int, long long>(raw_message);
  else if (type == TensorType::LONG)
    return _readValue<long, long long>(raw_message);
  else if (type == TensorType::LONG_LONG)
    return _readValue<long long, long long>(raw_message);

  throw std::invalid_argument("wrong integer type in the raw message");
}

Tensor *unpackTensor(RPCMessage& raw_message) {
  const TensorType& type = format_to_type.at(*raw_message.read(sizeof(char)));
  if (type == TensorType::TENSOR)
    return NULL; //_readValue<long long int>(raw_message); TODO
  throw std::invalid_argument("expected tensor in the raw message");
}

unsigned long long unpackTensorAsId(RPCMessage& raw_message) {
  const TensorType& type = format_to_type.at(*raw_message.read(sizeof(char)));
  if (type == TensorType::TENSOR)
    return _readValue<long long int>(raw_message);
  throw std::invalid_argument("expected tensor in the raw message");
}

THLongStorage* unpackTHLongStorage(RPCMessage& raw_message) {
  // TODO this might leak on errors
  char type = *raw_message.read(sizeof(char));
  if (type != 'F')
    throw std::invalid_argument("expected THLongStorage in the raw message");
  ptrdiff_t size = _readValue<ptrdiff_t, ptrdiff_t>(raw_message);
  THLongStorage* storage = THLongStorage_newWithSize(size);
  long* data = storage->data;
  for (int i = 0; i < size; i++) {
    data[i] = _readValue<long, long>(raw_message);
  }
  return storage;
}

}} // namespace rpc, thd
