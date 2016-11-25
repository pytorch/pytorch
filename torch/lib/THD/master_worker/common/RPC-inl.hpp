#include <cstdint>
#include "TH/THStorage.h"
#include "base/TensorTraits.hpp"

namespace thd { namespace rpc { namespace detail {
////////////////////////////////////////////////////////////////////////////////

constexpr size_t INITIAL_BUFFER_SIZE = 256;

template<typename real>
void _appendScalar(ByteArray& str, real data) {
  str.append((char*)&data, sizeof(real));
}

template <typename T>
void _appendData(ByteArray& str, const T& arg) {
  _appendScalar<TensorType>(str, tensor_type_traits<T>::type);
  _appendScalar<T>(str, arg);
}

template<typename T,
         typename = typename std::enable_if<is_any_of<T, THDTensorTypes>::value>::type>
inline void _appendData(ByteArray& str, const T& arg) {
  _appendScalar<char>(str, 'T'); // TODO store this char somewhere else
  _appendScalar<unsigned long long>(str, arg.tensor_id);
}

inline void _appendData(ByteArray& str, THLongStorage *arg) {
  _appendScalar<char>(str, 'F');    // 'F' stands for THLongStorage
  _appendScalar<ptrdiff_t>(str, arg->size);
  for (ptrdiff_t i = 0; i < arg->size; i++)
    _appendScalar<long>(str, arg->data[i]);
}

inline void _packIntoString(ByteArray& str) {};

template <typename T, typename ...Args>
void _packIntoString(ByteArray& str, const T& arg, const Args&... args) {
  _appendData(str, arg);
  _packIntoString(str, args...);
}

////////////////////////////////////////////////////////////////////////////////
} // namespace detail

template <typename ...Args>
std::unique_ptr<RPCMessage> packMessage(
    function_id_type fid,
    const Args&... args
) {
  ByteArray msg(detail::INITIAL_BUFFER_SIZE);
  detail::_appendScalar<function_id_type>(msg, fid);
  detail::_packIntoString(msg, args...);
  return std::unique_ptr<RPCMessage>(new RPCMessage(std::move(msg)));
}

}} // namespace rpc, thd
