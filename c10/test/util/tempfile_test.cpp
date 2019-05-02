#include <c10/util/tempfile.h>
#include <gtest/gtest.h>

#if !defined(_WIN32)
TEST(TempFileTest, MatchesExpectedPattern) {
  c10::TempFile pattern = c10::make_tempfile("test-pattern-");
  ASSERT_NE(pattern.name.find("test-pattern-"), std::string::npos);
}
#endif // !defined(_WIN32)
