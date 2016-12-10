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

////////////////////////////////////////////////////////////////////////////////

namespace {

template<typename T>
inline T unpackScalar(RPCMessage& raw_message) {
  return *reinterpret_cast<const T*>(raw_message.read(sizeof(T)));
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(Type) == sizeof(char), "Type has to be of the "
    "same size as char");
Type unpackType(RPCMessage& raw_message) {
  char _type = *raw_message.read(sizeof(Type));
  return static_cast<Type>(_type);
}

Type peekType(RPCMessage& raw_message) {
  char _type = *raw_message.data();
  return static_cast<Type>(_type);
}

function_id_type unpackFunctionId(RPCMessage& raw_message) {
  return unpackScalar<function_id_type>(raw_message);
}

double unpackFloat(RPCMessage& raw_message) {
  Type type = unpackType(raw_message);
  if (type == Type::DOUBLE)
    return unpackScalar<double>(raw_message);
  else if (type == Type::FLOAT)
    return unpackScalar<float>(raw_message);

  throw std::invalid_argument("wrong real type in the raw message");
}

long long unpackInteger(RPCMessage& raw_message) {
  Type type = unpackType(raw_message);
  if (type == Type::CHAR)
    return unpackScalar<char>(raw_message);
  else if (type == Type::SHORT)
    return unpackScalar<short>(raw_message);
  else if (type == Type::INT)
    return unpackScalar<int>(raw_message);
  else if (type == Type::LONG)
    return unpackScalar<long>(raw_message);
  else if (type == Type::LONG_LONG)
    return unpackScalar<long long>(raw_message);

  throw std::invalid_argument(std::string("wrong integer type in the raw message (") +
          std::to_string(static_cast<char>(type)) + ")");
}

object_id_type unpackTensor(RPCMessage& raw_message) {
  Type type = unpackType(raw_message);
  if (type == Type::TENSOR)
    return unpackScalar<object_id_type>(raw_message);
  throw std::invalid_argument("expected tensor in the raw message");
}

object_id_type unpackStorage(RPCMessage& raw_message) {
  Type type = unpackType(raw_message);
  if (type == Type::STORAGE)
    return unpackScalar<object_id_type>(raw_message);
  throw std::invalid_argument("expected storage in the raw message");
}

THLongStorage* unpackTHLongStorage(RPCMessage& raw_message) {
  // TODO this might leak on errors
  Type type = unpackType(raw_message);
  if (type != Type::LONG_STORAGE)
    throw std::invalid_argument("expected THLongStorage in the raw message");
  ptrdiff_t size = unpackScalar<ptrdiff_t>(raw_message);
  THLongStorage* storage = THLongStorage_newWithSize(size);
  long* data = storage->data;
  for (int i = 0; i < size; i++) {
    data[i] = unpackScalar<long>(raw_message);
  }
  return storage;
}

}} // namespace rpc, thd
