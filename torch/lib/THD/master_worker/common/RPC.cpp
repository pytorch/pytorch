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

RPCMessage::size_type RPCMessage::remaining() const {
  return _msg.length() - _offset;
}

const char* RPCMessage::read(std::size_t num_bytes) {
  if (_offset + num_bytes > _msg.length())
    throw std::out_of_range("invalid access: out of bounds");
  const char* ret_val = _msg.data() + _offset;
  _offset += num_bytes;
  return ret_val;
}

////////////////////////////////////////////////////////////////////////////////

namespace {

template<typename T>
inline T unpackScalar(RPCMessage& raw_message) {
  return *reinterpret_cast<const T*>(raw_message.read(sizeof(T)));
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(thpp::Type) == sizeof(char), "thpp::Type has to be of the "
    "same size as char");
thpp::Type unpackType(RPCMessage& raw_message) {
  char _type = *raw_message.read(sizeof(thpp::Type));
  return static_cast<thpp::Type>(_type);
}

thpp::Type peekType(RPCMessage& raw_message) {
  char _type = *raw_message.data();
  return static_cast<thpp::Type>(_type);
}

function_id_type unpackFunctionId(RPCMessage& raw_message) {
  return unpackScalar<function_id_type>(raw_message);
}

double unpackFloat(RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  if (type == thpp::Type::DOUBLE)
    return unpackScalar<double>(raw_message);
  else if (type == thpp::Type::FLOAT)
    return unpackScalar<float>(raw_message);

  throw std::invalid_argument("wrong real type in the raw message");
}

long long unpackInteger(RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  if (type == thpp::Type::CHAR)
    return unpackScalar<char>(raw_message);
  else if (type == thpp::Type::SHORT)
    return unpackScalar<short>(raw_message);
  else if (type == thpp::Type::INT)
    return unpackScalar<int>(raw_message);
  else if (type == thpp::Type::LONG)
    return unpackScalar<long>(raw_message);
  else if (type == thpp::Type::LONG_LONG)
    return unpackScalar<long long>(raw_message);

  throw std::invalid_argument(std::string("wrong integer type in the raw message (") +
          std::to_string(static_cast<char>(type)) + ")");
}

object_id_type unpackTensor(RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  if (type == thpp::Type::TENSOR)
    return unpackScalar<object_id_type>(raw_message);
  throw std::invalid_argument("expected tensor in the raw message");
}

object_id_type unpackStorage(RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  if (type == thpp::Type::STORAGE)
    return unpackScalar<object_id_type>(raw_message);
  throw std::invalid_argument("expected storage in the raw message");
}

object_id_type unpackGenerator(RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  if (type == thpp::Type::GENERATOR) {
    return unpackScalar<object_id_type>(raw_message);
  }
  throw std::invalid_argument("expected generator in the raw message");
}

THLongStorage* unpackTHLongStorage(RPCMessage& raw_message) {
  thpp::Type type = unpackType(raw_message);
  if (type != thpp::Type::LONG_STORAGE)
    throw std::invalid_argument("expected THLongStorage in the raw message");
  char is_null = unpackScalar<char>(raw_message);
  if (is_null) return NULL;
  ptrdiff_t size = unpackScalar<ptrdiff_t>(raw_message);
  THLongStorage* storage = THLongStorage_newWithSize(size);
  long* data = storage->data;

  try {
    for (int i = 0; i < size; i++) {
      data[i] = unpackScalar<long>(raw_message);
    }
  } catch (std::exception& e) {
    THLongStorage_free(storage);
    throw;
  }

  return storage;
}

}} // namespace rpc, thd
