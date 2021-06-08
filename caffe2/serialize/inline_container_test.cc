#include <cstdio>
#include <string>
#include <array>

#include <gtest/gtest.h>

#include "caffe2/serialize/inline_container.h"

namespace caffe2 {
namespace serialize {
namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PyTorchStreamWriterAndReader, SaveAndLoad) {
  int64_t kFieldAlignment = 64L;

  std::ostringstream oss;
  // write records through writers
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;

  for (int i = 0; i < data1.size(); ++i) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (int i = 0; i < data2.size(); ++i) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());

  const std::vector<std::string>& written_records = writer.getAllWrittenRecords();
  ASSERT_EQ(written_records[0], "key1");
  ASSERT_EQ(written_records[1], "key2");

  writer.writeEndOfFile();

  std::string the_file = oss.str();
  std::ofstream foo("output.zip");
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  ASSERT_TRUE(reader.hasRecord("key1"));
  ASSERT_TRUE(reader.hasRecord("key2"));
  ASSERT_FALSE(reader.hasRecord("key2000"));
  at::DataPtr data_ptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t size;
  std::tie(data_ptr, size) = reader.getRecord("key1");
  size_t off1 = reader.getRecordOffset("key1");
  ASSERT_EQ(size, data1.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data1.data(), data1.size()), 0);
  ASSERT_EQ(memcmp(the_file.c_str() + off1, data1.data(), data1.size()), 0);
  ASSERT_EQ(off1 % kFieldAlignment, 0);

  std::tie(data_ptr, size) = reader.getRecord("key2");
  size_t off2 = reader.getRecordOffset("key2");
  ASSERT_EQ(off2 % kFieldAlignment, 0);

  ASSERT_EQ(size, data2.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data2.data(), data2.size()), 0);
  ASSERT_EQ(memcmp(the_file.c_str() + off2, data2.data(), data2.size()), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PytorchStreamWriterAndReader, GetNonexistentRecordThrows) {
  std::ostringstream oss;
  // write records through writers
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;

  for (int i = 0; i < data1.size(); ++i) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (int i = 0; i < data2.size(); ++i) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());

  const std::vector<std::string>& written_records = writer.getAllWrittenRecords();
  ASSERT_EQ(written_records[0], "key1");
  ASSERT_EQ(written_records[1], "key2");

  writer.writeEndOfFile();

  std::string the_file = oss.str();
  std::ofstream foo("output2.zip");
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(reader.getRecord("key3"), c10::Error);

  // Reader should still work after throwing
  EXPECT_TRUE(reader.hasRecord("key1"));
}

} // namespace
} // namespace serialize
} // namespace caffe2
