#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <optional>

#if !defined(_WIN32)
static bool file_exists(const char* path) {
  struct stat st {};
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}
static bool directory_exists(const char* path) {
  struct stat st {};
  return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}
#else
static bool file_exists(const char* path) {
  struct _stat st {};
  return _stat(path, &st) == 0 && ((st.st_mode & _S_IFMT) == _S_IFREG);
}
static bool directory_exists(const char* path) {
  struct _stat st {};
  return _stat(path, &st) == 0 && ((st.st_mode & _S_IFMT) == _S_IFDIR);
}
#endif // !defined(_WIN32)

TEST(TempFileTest, MatchesExpectedPattern) {
  c10::TempFile file = c10::make_tempfile("test-pattern-");

#if defined(_WIN32)
  ASSERT_TRUE(file.open());
#endif
  ASSERT_TRUE(file_exists(file.name.c_str()));
#if !defined(_WIN32)
  ASSERT_NE(file.name.find("test-pattern-"), std::string::npos);
#endif // !defined(_WIN32)
}

TEST(TempDirTest, tryMakeTempdir) {
  std::optional<c10::TempDir> tempdir = c10::make_tempdir("test-dir-");
  std::string tempdir_name = tempdir->name;

  // directory should exist while tempdir is alive
  ASSERT_TRUE(directory_exists(tempdir_name.c_str()));

  // directory should not exist after tempdir destroyed
  tempdir.reset();
  ASSERT_FALSE(directory_exists(tempdir_name.c_str()));
}
