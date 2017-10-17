#include "Traits.hpp"

namespace thpp {

constexpr Type type_traits<char>::type;
constexpr Type type_traits<int8_t>::type;
constexpr Type type_traits<uint8_t>::type;
constexpr Type type_traits<float>::type;
constexpr Type type_traits<double>::type;
constexpr Type type_traits<int16_t>::type;
constexpr Type type_traits<uint16_t>::type;
constexpr Type type_traits<int32_t>::type;
constexpr Type type_traits<uint32_t>::type;
constexpr Type type_traits<int64_t>::type;
constexpr Type type_traits<uint64_t>::type;
constexpr Type type_traits<std::conditional<std::is_same<long, int64_t>::value, long long, long>::type>::type;
constexpr Type type_traits<std::conditional<std::is_same<unsigned long, uint64_t>::value, unsigned long long, unsigned long>::type>::type;

} // namespace thpp
