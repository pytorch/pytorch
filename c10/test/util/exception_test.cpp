#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
bool throw_func() {
  throw std::runtime_error("I'm throwing...");
}
} // namespace

TEST(ExceptionTest, TORCH_INTERNAL_ASSERT_DEBUG_ONLY) {
#ifdef NDEBUG
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false));
  // Does nothing - `throw_func()` should not be evaluated
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(throw_func()));
#else
  ASSERT_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false), c10::Error);
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(true));
#endif
}
