#include <cstdint>
#include "TH/THStorage.h"
#include "Traits.hpp"

namespace thd { namespace rpc { namespace detail {
////////////////////////////////////////////////////////////////////////////////

constexpr size_t INITIAL_BUFFER_SIZE = 256;

template<typename real,
         typename = typename std::enable_if<std::is_arithmetic<real>::value>::type>
inline void _appendScalar(ByteArray& str, real data) {
  str.append(reinterpret_cast<char*>(&data), sizeof(data));
}

inline void _appendType(ByteArray& str, RPCType _type) {
  char type = static_cast<char>(_type);
  str.append(&type, sizeof(type));
}

template<typename T>
inline void __appendData(ByteArray& str, const T& arg,
    std::false_type is_generator, std::false_type is_tensor, std::false_type is_storage) {
  _appendType(str, type_traits<T>::type);
  _appendScalar<T>(str, arg);
}

template<typename T>
inline void __appendData(ByteArray& str, const T& arg,
    std::true_type is_generator, std::false_type is_tensor, std::false_type is_storage) {
  _appendType(str, RPCType::GENERATOR);
  _appendScalar<object_id_type>(str, arg->generator_id);
}

template<typename T>
inline void __appendData(ByteArray& str, const T& arg,
    std::false_type is_generator, std::true_type is_tensor, std::false_type is_storage) {
  _appendType(str, RPCType::TENSOR);
  _appendScalar<object_id_type>(str, arg->tensor_id);
}

template<typename T>
inline void __appendData(ByteArray& str, const T& arg,
    std::false_type is_generator, std::false_type is_tensor, std::true_type is_storage) {
  _appendType(str, RPCType::STORAGE);
  _appendScalar<object_id_type>(str, arg->storage_id);
}

template<typename T>
inline void _appendData(ByteArray& str, const T& arg) {
  __appendData(
      str,
      arg,
      is_any_of<T, THDGeneratorPtrTypes>(),
      is_any_of<T, THDTensorPtrTypes>(),
      is_any_of<T, THDStoragePtrTypes>()
  );
}

inline void _appendData(ByteArray& str, THLongStorage* arg) {
  _appendType(str, RPCType::LONG_STORAGE);
  _appendScalar<char>(str, arg == NULL);
  if (!arg) return;
  _appendScalar<ptrdiff_t>(str, THLongStorage_size(arg));
  for (ptrdiff_t i = 0; i < THLongStorage_size(arg); i++)
    _appendScalar<int64_t>(str, THLongStorage_get(arg, i));
}

template<typename T>
inline void _appendData(ByteArray& str, const std::vector<T>& arg) {
  int l = arg.size();
  _appendData(str, l);
  for (std::size_t i = 0; i < l; i++)
    __appendData(
        str,
        arg[i],
        is_any_of<T, THDGeneratorPtrTypes>(),
        is_any_of<T, THDTensorPtrTypes>(),
        is_any_of<T, THDStoragePtrTypes>()
    );
}

inline void _appendData(ByteArray& str, RPCType type) {
  _appendType(str, type);
}

inline void _packIntoString(ByteArray& str) {};

template <typename T, typename ...Args>
inline void _packIntoString(ByteArray& str, const T& arg, const Args&... args) {
  _appendData(str, arg);
  _packIntoString(str, args...);
}

////////////////////////////////////////////////////////////////////////////////
} // namespace detail

template <typename ...Args>
inline std::unique_ptr<RPCMessage> packMessage(
    function_id_type fid,
    const Args&... args
) {
  ByteArray msg(detail::INITIAL_BUFFER_SIZE);
  detail::_appendScalar<function_id_type>(msg, fid);
  detail::_packIntoString(msg, args...);
  return std::unique_ptr<RPCMessage>(new RPCMessage(std::move(msg)));
}

}} // namespace rpc, thd
