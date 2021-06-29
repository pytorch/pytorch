#include <test/cpp/api/support.h>

namespace torch {
namespace test {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex AutoDefaultDtypeMode::default_dtype_mutex;

} // namespace test
} // namespace torch
