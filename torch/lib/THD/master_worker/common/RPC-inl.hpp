#include <cstdint>

namespace thd { namespace rpc { namespace detail {
////////////////////////////////////////////////////////////////////////////////

constexpr std::size_t MAX_VAR_SIZE = 8;


template<typename real>
void _appendData(ByteArray& str, real data) {
  str.append(reinterpret_cast<char*>(&data), sizeof(real));
}

template <typename T>
void _appendTensorOrScalar(ByteArray& str, const T& arg) {
  _appendData<char>(str, static_cast<char>(tensor_type_traits<T>::type));
  _appendData<T>(str, arg);
}

inline void _appendTensorOrScalar(ByteArray& str, const THDTensor& arg) {
  _appendData<char>(str, static_cast<char>(TensorType::TENSOR));
  _appendData<unsigned long long>(str, arg.tensor_id);
}

inline void packIntoString(ByteArray& str) {};

template <typename T, typename ...Args>
void packIntoString(ByteArray& str, const T& arg, const Args&... args) {
  _appendTensorOrScalar(str, arg);
  packIntoString(str, args...);
}

////////////////////////////////////////////////////////////////////////////////
} // namespace detail

template <typename ...Args>
RPCMessage packMessage(function_id_type fid, std::uint16_t num_args,
    const Args&... args) {
  ByteArray msg(sizeof(fid) + sizeof(std::uint16_t) + num_args * (sizeof(char) +
                detail::MAX_VAR_SIZE));
  detail::_appendData<function_id_type>(msg, fid);
  detail::_appendData<std::uint16_t>(msg, num_args);
  detail::packIntoString(msg, args...);
  return RPCMessage(std::move(msg));
}

}} // namespace rpc, thd
