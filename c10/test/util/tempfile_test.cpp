#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#if !defined(_WIN32)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TempFileTest, MatchesExpectedPattern) {
  c10::TempFile pattern = c10::make_tempfile("test-pattern-");
  ASSERT_NE(pattern.name.find("test-pattern-"), std::string::npos);
}
#endif // !defined(_WIN32)

static bool directory_exists(const char* path) {
  struct stat st;
  return (stat(path, &st) == 0 && (st.st_mode & S_IFDIR));
}

TEST(TempDirTest, tryMakeTempdir) {
  c10::optional<c10::TempDir> tempdir = c10::make_tempdir("test-dir-");
  std::string tempdir_name = tempdir->name;

  // directory should exist while tempdir is alive
  ASSERT_TRUE(directory_exists(tempdir_name.c_str()));

  // directory should not exist after tempdir destroyed
  tempdir.reset();
  ASSERT_FALSE(directory_exists(tempdir_name.c_str()));
}
