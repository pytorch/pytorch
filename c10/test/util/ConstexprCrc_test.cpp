#include <c10/util/ConstexprCrc.h>

using c10::util::crc64_t;

namespace {
template <size_t N>
constexpr crc64_t crc64(const char (&input)[N]) {
  return c10::util::crc64(input, N - 1);
}
} // namespace

// generic tests
static_assert(
    crc64("MyTestString") == crc64("MyTestString"),
    "crc64 is deterministic");
static_assert(
    crc64("MyTestString1") != crc64("MyTestString2"),
    "different strings, different result");

// check concrete expected values (for CRC64 with Jones coefficients and an init
// value of 0)
static_assert(crc64_t{0} == crc64(""), "");
static_assert(crc64_t{0xe9c6d914c4b8d9ca} == crc64("123456789"), "");
