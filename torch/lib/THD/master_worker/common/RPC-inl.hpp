
namespace thd { namespace rpc { namespace detail {
////////////////////////////////////////////////////////////////////////////////

constexpr size_t MAX_VAR_SIZE = 8;

// The following notation comes from:
// docs.python.org/3/library/struct.html#module-struct
// except from 'T', which stands for Tensor.
template<typename T>
struct rpc_traits {};

template<>
struct rpc_traits<double> {
  static constexpr char scalar_char = 'd';
};

template<>
struct rpc_traits<char> {
  static constexpr char scalar_char = 'c';
};

template<>
struct rpc_traits<float> {
  static constexpr char scalar_char = 'f';
};

template<>
struct rpc_traits<int> {
  static constexpr char scalar_char = 'i';
};

template<>
struct rpc_traits<long> {
  static constexpr char scalar_char = 'l';
};

template<>
struct rpc_traits<long long> {
  static constexpr char scalar_char = 'q';
};

template<>
struct rpc_traits<short> {
  static constexpr char scalar_char = 'h';
};

template<typename real>
void _appendData(ByteArray& str, real data) {
  str.append((char*)&data, sizeof(real));
}

template <typename T>
void _appendTensorOrScalar(ByteArray& str, const T& arg) {
  _appendData<char>(str, rpc_traits<T>::scalar_char);
  _appendData<T>(str, arg);
}

inline void _appendTensorOrScalar(ByteArray& str, const THDTensor& arg) {
  _appendData<char>(str, 'T');
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
RPCMessage packMessage(function_id_type fid, uint16_t num_args,
    const Args&... args) {
  ByteArray msg(sizeof(fid) + sizeof(uint16_t) + num_args * (sizeof(char) +
                detail::MAX_VAR_SIZE));
  detail::_appendData<function_id_type>(msg, fid);
  detail::_appendData<uint16_t>(msg, num_args);
  detail::packIntoString(msg, args...);
  return RPCMessage(std::move(msg));
}

}} // namespace rpc, thd
