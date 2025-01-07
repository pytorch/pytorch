#include <array>
#include <cstdio>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include <c10/util/Logging.h>
#include "c10/util/irange.h"
#include "caffe2/serialize/inline_container.h"

namespace caffe2 {
namespace serialize {
namespace {

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
  // Inplace memory buffer
  std::vector<uint8_t> buf(data1.size());

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 2);
  ASSERT_EQ(written_records.count("key1"), 1);
  ASSERT_EQ(written_records.count("key2"), 1);

  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = oss.str();
  const char* file_name = "output.zip";
  std::ofstream foo(file_name);
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
  // inplace getRecord() test
  std::vector<uint8_t> dst(size);
  size_t ret = reader.getRecord("key1", dst.data(), size);
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data1.data(), size), 0);
  // chunked getRecord() test
  ret = reader.getRecord(
      "key1",
      dst.data(),
      size,
      3,
      buf.data(),
      [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); });
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data1.data(), size), 0);

  std::tie(data_ptr, size) = reader.getRecord("key2");
  size_t off2 = reader.getRecordOffset("key2");
  ASSERT_EQ(off2 % kFieldAlignment, 0);

  ASSERT_EQ(size, data2.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data2.data(), data2.size()), 0);
  ASSERT_EQ(memcmp(the_file.c_str() + off2, data2.data(), data2.size()), 0);
  // inplace getRecord() test
  dst.resize(size);
  ret = reader.getRecord("key2", dst.data(), size);
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data2.data(), size), 0);
  // chunked getRecord() test
  ret = reader.getRecord(
      "key2",
      dst.data(),
      size,
      3,
      buf.data(),
      [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); });
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data2.data(), size), 0);
  // clean up
  remove(file_name);
}

TEST(PyTorchStreamWriterAndReader, LoadWithMultiThreads) {
  std::ostringstream oss;
  // write records through writers
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());

  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 2);
  ASSERT_EQ(written_records.count("key1"), 1);
  ASSERT_EQ(written_records.count("key2"), 1);

  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = oss.str();
  const char* file_name = "output.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  // read records through pytorchStreamReader
  std::istringstream iss(the_file);
  PyTorchStreamReader reader(&iss);
  reader.setAdditionalReaderSizeThreshold(0);
  // before testing, sanity check
  int64_t size1, size2, ret;
  at::DataPtr data_ptr;
  std::tie(data_ptr, size1) = reader.getRecord("key1");
  std::tie(data_ptr, size2) = reader.getRecord("key2");

  // Test getRecord(name, additional_readers)
  std::vector<std::shared_ptr<ReadAdapterInterface>> additionalReader;
  for (int i = 0; i < 10; ++i) {
    // Test various sized additional readers.
    std::tie(data_ptr, ret) = reader.getRecord("key1", additionalReader);
    ASSERT_EQ(ret, size1);
    ASSERT_EQ(memcmp(data_ptr.get(), data1.data(), size1), 0);

    std::tie(data_ptr, ret) = reader.getRecord("key2", additionalReader);
    ASSERT_EQ(ret, size2);
    ASSERT_EQ(memcmp(data_ptr.get(), data2.data(), size2), 0);
  }

  // Inplace multi-threading getRecord(name, dst, n, additional_readers) test
  additionalReader.clear();
  std::vector<uint8_t> dst1(size1), dst2(size2);
  for (int i = 0; i < 10; ++i) {
    // Test various sizes of read threads
    additionalReader.push_back(std::make_unique<IStreamAdapter>(&iss));

    ret = reader.getRecord("key1", dst1.data(), size1, additionalReader);
    ASSERT_EQ(ret, size1);
    ASSERT_EQ(memcmp(dst1.data(), data1.data(), size1), 0);

    ret = reader.getRecord("key2", dst2.data(), size2, additionalReader);
    ASSERT_EQ(ret, size2);
    ASSERT_EQ(memcmp(dst2.data(), data2.data(), size2), 0);
  }
  // clean up
  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, GetNonexistentRecordThrows) {
  std::ostringstream oss;
  // write records through writers
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;

  // Inplace memory buffer
  std::vector<uint8_t> buf;

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 2);
  ASSERT_EQ(written_records.count("key1"), 1);
  ASSERT_EQ(written_records.count("key2"), 1);

  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = oss.str();
  const char* file_name = "output2.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(reader.getRecord("key3"), c10::Error);
  std::vector<uint8_t> dst(data1.size());
  EXPECT_THROW(reader.getRecord("key3", dst.data(), data1.size()), c10::Error);
  EXPECT_THROW(
      reader.getRecord(
          "key3",
          dst.data(),
          data1.size(),
          3,
          buf.data(),
          [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); }),
      c10::Error);

  // Reader should still work after throwing
  EXPECT_TRUE(reader.hasRecord("key1"));
  // clean up
  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, SkipDebugRecords) {
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;
  // Inplace memory buffer
  std::vector<uint8_t> buf(data1.size());

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1.debug_pkl", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2.debug_pkl", data2.data(), data2.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 2);
  ASSERT_EQ(written_records.count("key1.debug_pkl"), 1);
  ASSERT_EQ(written_records.count("key2.debug_pkl"), 1);
  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = oss.str();
  const char* file_name = "output3.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)

  reader.setShouldLoadDebugSymbol(false);
  EXPECT_FALSE(reader.hasRecord("key1.debug_pkl"));
  at::DataPtr ptr;
  size_t size;
  std::tie(ptr, size) = reader.getRecord("key1.debug_pkl");
  EXPECT_EQ(size, 0);
  std::vector<uint8_t> dst(data1.size());
  size_t ret = reader.getRecord("key1.debug_pkl", dst.data(), data1.size());
  EXPECT_EQ(ret, 0);
  ret = reader.getRecord(
      "key1.debug_pkl",
      dst.data(),
      data1.size(),
      3,
      buf.data(),
      [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); });
  EXPECT_EQ(ret, 0);
  // clean up
  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, ValidSerializationId) {
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1.debug_pkl", data1.data(), data1.size());
  writer.writeEndOfFile();
  auto writer_serialization_id = writer.serializationId();

  std::string the_file = oss.str();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)

  EXPECT_EQ(reader.serializationId(), writer_serialization_id);

  // write a second time
  PyTorchStreamWriter writer2([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  writer2.writeRecord("key1.debug_pkl", data1.data(), data1.size());
  writer2.writeEndOfFile();
  auto writer2_serialization_id = writer2.serializationId();

  EXPECT_EQ(writer_serialization_id, writer2_serialization_id);
}

TEST(PytorchStreamWriterAndReader, SkipDuplicateSerializationIdRecords) {
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  std::string dup_serialization_id = "dup-serialization-id";
  writer.writeRecord(
      kSerializationIdRecordName,
      dup_serialization_id.c_str(),
      dup_serialization_id.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 0);
  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);
  auto writer_serialization_id = writer.serializationId();

  std::string the_file = oss.str();
  const char* file_name = "output4.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)

  EXPECT_EQ(reader.serializationId(), writer_serialization_id);
  // clean up
  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, LogAPIUsageMetadata) {
  std::map<std::string, std::map<std::string, std::string>> logs;

  SetAPIUsageMetadataLogger(
      [&](const std::string& context,
          const std::map<std::string, std::string>& metadata_map) {
        logs.insert({context, metadata_map});
      });
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  writer.writeEndOfFile();

  std::istringstream iss(oss.str());
  // read records through readers
  PyTorchStreamReader reader(&iss);

  ASSERT_EQ(logs.size(), 2);
  std::map<std::string, std::map<std::string, std::string>> expected_logs = {
      {"pytorch.stream.writer.metadata",
       {{"serialization_id", writer.serializationId()},
        {"file_name", "archive"},
        {"file_size", str(oss.str().length())}}},
      {"pytorch.stream.reader.metadata",
       {{"serialization_id", writer.serializationId()},
        {"file_name", "archive"},
        {"file_size", str(iss.str().length())}}}};
  ASSERT_EQ(expected_logs, logs);

  // reset logger
  SetAPIUsageMetadataLogger(
      [&](const std::string& context,
          const std::map<std::string, std::string>& metadata_map) {});
}

class ChunkRecordIteratorTest : public ::testing::TestWithParam<int64_t> {};
INSTANTIATE_TEST_SUITE_P(
    ChunkRecordIteratorTestGroup,
    ChunkRecordIteratorTest,
    testing::Values(100, 150, 1010));

TEST_P(ChunkRecordIteratorTest, ChunkRead) {
  auto chunkSize = GetParam();
  std::string zipFileName =
      "output_chunk_" + std::to_string(chunkSize) + ".zip";
  const char* fileName = zipFileName.c_str();
  const std::string recordName = "key1";
  const size_t tensorDataSizeInBytes = 1000;

  // write records through writers
  std::ostringstream oss(std::ios::binary);
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  auto tensorData = std::vector<uint8_t>(tensorDataSizeInBytes, 1);
  auto dataPtr = tensorData.data();
  writer.writeRecord(recordName, dataPtr, tensorDataSizeInBytes);
  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 1);
  ASSERT_EQ(written_records.count(recordName), 1);
  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = oss.str();
  std::ofstream foo(fileName, std::ios::binary);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();
  LOG(INFO) << "Finished saving tensor into zip file " << fileName;

  LOG(INFO) << "Testing chunk size " << chunkSize;
  PyTorchStreamReader reader(fileName);
  ASSERT_TRUE(reader.hasRecord(recordName));
  auto chunkIterator = reader.createChunkReaderIter(
      recordName, tensorDataSizeInBytes, chunkSize);
  std::vector<uint8_t> buffer(chunkSize);
  size_t totalReadSize = 0;
  while (auto readSize = chunkIterator.next(buffer.data())) {
    auto expectedData = std::vector<uint8_t>(readSize, 1);
    ASSERT_EQ(memcmp(expectedData.data(), buffer.data(), readSize), 0);
    totalReadSize += readSize;
  }
  ASSERT_EQ(totalReadSize, tensorDataSizeInBytes);
  // clean up
  remove(fileName);
}

} // namespace
} // namespace serialize
} // namespace caffe2
