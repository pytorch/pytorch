#include "RPC.hpp"
#include "ByteArray.hpp"

#include <cstdarg>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace thd {
namespace rpc {

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

static_assert(sizeof(RPCType) == sizeof(char), "RPCType has to be of the "
    "same size as char");
RPCType unpackType(RPCMessage& raw_message) {
  char _type = *raw_message.read(sizeof(RPCType));
  return static_cast<RPCType>(_type);
}

RPCType peekType(RPCMessage& raw_message) {
  char _type = *raw_message.data();
  return static_cast<RPCType>(_type);
}

function_id_type unpackFunctionId(RPCMessage& raw_message) {
  return unpackScalar<function_id_type>(raw_message);
}

double unpackFloat(RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  if (type == RPCType::DOUBLE)
    return unpackScalar<double>(raw_message);
  else if (type == RPCType::FLOAT)
    return unpackScalar<float>(raw_message);

  throw std::invalid_argument("wrong real type in the raw message");
}

int64_t unpackInteger(RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  if (type == RPCType::CHAR)
    return unpackScalar<int8_t>(raw_message);
  else if (type == RPCType::SHORT)
    return unpackScalar<int16_t>(raw_message);
  else if (type == RPCType::INT)
    return unpackScalar<int32_t>(raw_message);
  else if (type == RPCType::LONG)
    return unpackScalar<int64_t>(raw_message);
  else if (type == RPCType::LONG_LONG)
    return unpackScalar<int64_t>(raw_message);

  throw std::invalid_argument(std::string("wrong integer type in the raw message (") +
          std::to_string(static_cast<char>(type)) + ")");
}

object_id_type unpackTensor(RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  if (type == RPCType::TENSOR)
    return unpackScalar<object_id_type>(raw_message);
  throw std::invalid_argument("expected tensor in the raw message");
}

object_id_type unpackStorage(RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  if (type == RPCType::STORAGE)
    return unpackScalar<object_id_type>(raw_message);
  throw std::invalid_argument("expected storage in the raw message");
}

object_id_type unpackGenerator(RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  if (type == RPCType::GENERATOR) {
    return unpackScalar<object_id_type>(raw_message);
  }
  throw std::invalid_argument("expected generator in the raw message");
}

THLongStorage* unpackTHLongStorage(RPCMessage& raw_message) {
  RPCType type = unpackType(raw_message);
  if (type != RPCType::LONG_STORAGE)
    throw std::invalid_argument("expected THLongStorage in the raw message");
  char is_null = unpackScalar<char>(raw_message);
  if (is_null) return NULL;
  ptrdiff_t size = unpackScalar<ptrdiff_t>(raw_message);
  THLongStorage* storage = THLongStorage_newWithSize(size);
  int64_t* data = THLongStorage_data(storage);

  try {
    for (int i = 0; i < size; i++) {
      data[i] = unpackScalar<int64_t>(raw_message);
    }
  } catch (std::exception& e) {
    THLongStorage_free(storage);
    throw;
  }

  return storage;
}

}} // namespace rpc, thd
