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
//constexpr Type type_traits<long>::type;
constexpr Type type_traits<int64_t>::type;
//constexpr Type type_traits<unsigned long>::type;
constexpr Type type_traits<uint64_t>::type;

} // namespace thpp
