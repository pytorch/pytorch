#include <gtest/gtest.h>
#include <torch/nativert/common/FileUtil.h>
#include <fstream>

namespace torch {
namespace nativert {

TEST(FileUtilTest, OpenNoInt) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(fd, 0);

  closeNoInt(fd);
}

TEST(FileUtilTest, CloseNoInt) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(fd, 0);

  int result = closeNoInt(fd);
  ASSERT_EQ(result, 0);
}

TEST(FileUtilTest, WriteFull) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_WRONLY | O_CREAT, 0644);
  ASSERT_GE(fd, 0);

  const char* data = "Hello, World!";
  ssize_t bytesWritten = writeFull(fd, data, strlen(data));
  ASSERT_EQ(bytesWritten, strlen(data));

  closeNoInt(fd);
}

TEST(FileUtilTest, ReadFull) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile << "Hello, World!";
  tmpFile.close();

  int fd = openNoInt("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(fd, 0);

  char buffer[1024];
  ssize_t bytesRead = readFull(fd, buffer, 1024);
  ASSERT_EQ(bytesRead, 13); // length of "Hello, World!"

  closeNoInt(fd);
}

TEST(FileUtilTest, FileConstructor) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  File file("tmp_file.txt", O_RDONLY, 0);
  ASSERT_GE(file.fd(), 0);

  file.close();
}

TEST(FileUtilTest, FileMoveConstructor) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  File file1("tmp_file.txt", O_RDONLY, 0);
  File file2(std::move(file1));

  ASSERT_GE(file2.fd(), 0);
  ASSERT_EQ(file1.fd(), -1);

  file2.close();
}

TEST(FileUtilTest, FileAssignmentOperator) {
  // Create a temporary file
  std::ofstream tmpFile("tmp_file.txt");
  tmpFile.close();

  File file1("tmp_file.txt", O_RDONLY, 0);
  File file2;

  file2 = std::move(file1);

  ASSERT_GE(file2.fd(), 0);
  ASSERT_EQ(file1.fd(), -1);

  file2.close();
}

TEST(FileUtilTest, TemporaryFile) {
  File file = File::temporary();
  ASSERT_GE(file.fd(), 0);

  file.close();
}

} // namespace nativert
} // namespace torch
