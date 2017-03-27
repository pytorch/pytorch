#include "Traits.hpp"

namespace thpp {

constexpr Type type_traits<char>::type;
constexpr Type type_traits<unsigned char>::type;
constexpr Type type_traits<float>::type;
constexpr Type type_traits<double>::type;
constexpr Type type_traits<short>::type;
constexpr Type type_traits<unsigned short>::type;
constexpr Type type_traits<int>::type;
constexpr Type type_traits<unsigned int>::type;
constexpr Type type_traits<long>::type;
constexpr Type type_traits<unsigned long>::type;
constexpr Type type_traits<long long>::type;
constexpr Type type_traits<unsigned long long>::type;

} // namespace thpp
