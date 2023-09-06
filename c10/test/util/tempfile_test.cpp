#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <optional>

TEST(TempFileTest, MatchesExpectedPattern) {
  c10::TempFile pattern = c10::make_tempfile("test-pattern-");
  ASSERT_TRUE(stdfs::is_regular_file(pattern.name));
#if !defined(_WIN32)
  ASSERT_NE(pattern.name.find("test-pattern-"), std::string::npos);
#endif // !defined(_WIN32)
}

TEST(TempDirTest, tryMakeTempdir) {
  std::optional<c10::TempDir> tempdir = c10::make_tempdir("test-dir-");
  auto tempdir_name = tempdir->name;

  // directory should exist while tempdir is alive
  ASSERT_TRUE(stdfs::is_directory(tempdir_name));

  // directory should not exist after tempdir destroyed
  tempdir.reset();
  ASSERT_FALSE(stdfs::is_directory(tempdir_name));
}
