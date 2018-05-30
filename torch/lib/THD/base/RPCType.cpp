#include "RPCType.hpp"

namespace thd {

// Static constexpr variables have to be defined out-of-source in C++11.
// https://stackoverflow.com/questions/8016780/undefined-reference-to-static-constexpr-char
constexpr RPCType type_traits<char>::type;
constexpr RPCType type_traits<int8_t>::type;
constexpr RPCType type_traits<uint8_t>::type;
constexpr RPCType type_traits<float>::type;
constexpr RPCType type_traits<double>::type;
constexpr RPCType type_traits<int16_t>::type;
constexpr RPCType type_traits<int32_t>::type;
constexpr RPCType type_traits<uint32_t>::type;
constexpr RPCType type_traits<uint16_t>::type;
constexpr RPCType type_traits<int64_t>::type;
constexpr RPCType type_traits<uint64_t>::type;
constexpr RPCType type_traits<std::conditional<std::is_same<int64_t, long>::value, long long, long>::type>::type;
constexpr RPCType type_traits<std::conditional<std::is_same<uint64_t, unsigned long>::value, unsigned long long, unsigned long>::type>::type;

} // thd
