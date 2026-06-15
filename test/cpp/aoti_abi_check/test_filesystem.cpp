#include <gtest/gtest.h>

#include <torch/headeronly/util/FileSystem.h>

TEST(TestFileSystem, TestFileSystem) {
  torch::headeronly::filesystem::path p("/tmp/foo/bar.txt");
  EXPECT_EQ(p.filename().string(), "bar.txt");

  // c10 alias resolves to the same namespace
  c10::filesystem::path parent = p.parent_path();
  EXPECT_EQ(parent.filename().string(), "foo");
}
