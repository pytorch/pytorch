#include <cstdio>
#include <string>
#include <array>

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
  std::array<char, 127> data1;

  for (int i = 0; i < data1.size(); ++i) {
    data1[i] = data1.size() - i;
  }
  uint64_t next_key = writer.getCurrentSize();
  ASSERT_EQ(writer.writeRecord(data1.data(), data1.size()), next_key);
  std::array<char, 64> data2;
  for (int i = 0; i < data2.size(); ++i) {
    data2[i] = data2.size() - i;
  }
  next_key = writer.getCurrentSize();
  ASSERT_EQ(writer.writeRecord(data2.data(), data2.size()), next_key);
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
  ASSERT_EQ(size, data1.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data1.data(), data1.size()), 0);

  ASSERT_TRUE(reader.hasNextRecord());
  std::tie(data_ptr, key, size) = reader.getNextRecord();
  ASSERT_EQ(
      key,
      kFieldAlignment * 2 +
          (data1.size() + kFieldAlignment - 1) / kFieldAlignment *
              kFieldAlignment);
  ASSERT_EQ(size, data2.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data2.data(), data2.size()), 0);

  ASSERT_FALSE(reader.hasNextRecord());

  std::tie(data_ptr, size) = reader.getLastRecord();
  ASSERT_EQ(size, data2.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data2.data(), data2.size()), 0);
  ASSERT_FALSE(reader.hasNextRecord());

  std::tie(data_ptr, size) = reader.getRecordWithKey(kFieldAlignment);
  ASSERT_EQ(size, data1.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data1.data(), data1.size()), 0);
  ASSERT_TRUE(reader.hasNextRecord());

  // clean up
  std::remove(tmp_name.c_str());
}

} // namespace
} // namespace at
