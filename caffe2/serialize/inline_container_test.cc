#include <cstdio>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "caffe2/serialize/inline_container.h"

namespace at {
namespace {

TEST(PyTorchFileWriterAndReader, SaveAndLoad) {
  int64_t kFieldAlignment = 64L;
  // create a name for temporary file
  // TODO to have different implementation for Windows and POXIS
  std::string tmp_name = std::tmpnam(nullptr);

  // write records through writers
  torch::jit::PyTorchFileWriter writer{tmp_name};
  char data1[127];
  for (int i = 0; i < sizeof(data1); ++i) {
    data1[i] = sizeof(data1) - i;
  }
  ASSERT_EQ(writer.writeRecord(data1, sizeof(data1)), writer.getCurrentSize());
  char data2[64];
  for (int i = 0; i < sizeof(data2); ++i) {
    data2[i] = sizeof(data2) - i;
  }
  ASSERT_EQ(writer.writeRecord(data2, sizeof(data2)), writer.getCurrentSize());
  writer.writeEndOfFile();
  ASSERT_TRUE(writer.closed());

  // read records through readers
  torch::jit::PyTorchFileReader reader{tmp_name};
  ASSERT_TRUE(reader.hasNextRecord());
  at::DataPtr data_ptr;
  int64_t key;
  int64_t size;
  std::tie(data_ptr, key, size) = reader.getNextRecord();
  ASSERT_EQ(key, kFieldAlignment);
  ASSERT_EQ(size, sizeof(data1));
  ASSERT_EQ(memcmp(data_ptr.get(), data1, sizeof(data1)), 0);

  ASSERT_TRUE(reader.hasNextRecord());
  std::tie(data_ptr, key, size) = reader.getNextRecord();
  ASSERT_EQ(
      key,
      kFieldAlignment * 2 +
          (sizeof(data1) + kFieldAlignment - 1) / kFieldAlignment *
              kFieldAlignment);
  ASSERT_EQ(size, sizeof(data2));
  ASSERT_EQ(memcmp(data_ptr.get(), data2, sizeof(data2)), 0);

  ASSERT_FALSE(reader.hasNextRecord());

  std::tie(data_ptr, size) = reader.getLastRecord();
  ASSERT_EQ(size, sizeof(data2));
  ASSERT_EQ(memcmp(data_ptr.get(), data2, sizeof(data2)), 0);
  ASSERT_FALSE(reader.hasNextRecord());

  std::tie(data_ptr, size) = reader.getRecordWithKey(kFieldAlignment);
  ASSERT_EQ(size, sizeof(data1));
  ASSERT_EQ(memcmp(data_ptr.get(), data1, sizeof(data1)), 0);
  ASSERT_TRUE(reader.hasNextRecord());

  // clean up
  std::remove(tmp_name.c_str());
}

} // namespace
} // namespace at
