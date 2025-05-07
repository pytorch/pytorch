#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <thread>

#include <c10/core/Allocator.h>
#include <c10/core/Backend.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/hash.h>
#include <c10/util/string_view.h>

#include "caffe2/core/common.h"
#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/inline_container.h"
#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"

#include "caffe2/serialize/versions.h"
#include "miniz.h"

namespace caffe2 {
namespace serialize {
constexpr std::string_view kDebugPklSuffix(".debug_pkl");

struct MzZipReaderIterWrapper {
  MzZipReaderIterWrapper(mz_zip_reader_extract_iter_state* iter) : impl(iter) {}
  mz_zip_reader_extract_iter_state* impl;
};

ChunkRecordIterator::ChunkRecordIterator(
    size_t recordSize,
    size_t chunkSize,
    std::unique_ptr<MzZipReaderIterWrapper> iter)
    : recordSize_(recordSize),
      chunkSize_(chunkSize),
      offset_(0),
      iter_(std::move(iter)) {}

ChunkRecordIterator::~ChunkRecordIterator() {
  mz_zip_reader_extract_iter_free(iter_->impl);
}

size_t ChunkRecordIterator::next(void* buf) {
  size_t want_size = std::min(chunkSize_, recordSize_ - offset_);
  if (want_size == 0) {
    return 0;
  }
  size_t read_size =
      mz_zip_reader_extract_iter_read(iter_->impl, buf, want_size);
  TORCH_CHECK(read_size > 0, "Read bytes should be larger than 0");
  offset_ += read_size;
  return read_size;
}

size_t
istream_read_func(void* pOpaque, mz_uint64 file_ofs, void* pBuf, size_t n) {
  auto self = static_cast<PyTorchStreamReader*>(pOpaque);
  return self->read(file_ofs, static_cast<char*>(pBuf), n);
}

static std::string basename(const std::string& name) {
  size_t start = 0;
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '\\' || name[i] == '/') {
      start = i + 1;
    }
  }

  if (start >= name.size()) {
    return "";
  }

  size_t end = name.size();
  for (size_t i = end; i > start; --i) {
    if (name[i - 1] == '.') {
      end = i - 1;
      break;
    }
  }
  return name.substr(start, end - start);
}

static std::string parentdir(const std::string& name) {
  size_t end = name.find_last_of('/');
  if (end == std::string::npos) {
    end = name.find_last_of('\\');
  }

#ifdef WIN32
  if (end != std::string::npos && end > 1 && name[end - 1] == ':') {
    // This is a Windows root directory, so include the slash in
    // the parent directory
    end++;
  }
#endif

  if (end == std::string::npos) {
    return "";
  }

  return name.substr(0, end);
}

size_t PyTorchStreamReader::read(uint64_t pos, char* buf, size_t n) {
  return in_->read(pos, buf, n, "reading file");
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
PyTorchStreamReader::PyTorchStreamReader(const std::string& file_name)
    : ar_(std::make_unique<mz_zip_archive>()),
      in_(std::make_unique<FileAdapter>(file_name)) {
  init();
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
PyTorchStreamReader::PyTorchStreamReader(std::istream* in)
    : ar_(std::make_unique<mz_zip_archive>()),
      in_(std::make_unique<IStreamAdapter>(in)) {
  init();
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
PyTorchStreamReader::PyTorchStreamReader(
    std::shared_ptr<ReadAdapterInterface> in)
    : ar_(std::make_unique<mz_zip_archive>()), in_(std::move(in)) {
  init();
}

void PyTorchStreamReader::init() {
  AT_ASSERT(in_ != nullptr);
  AT_ASSERT(ar_ != nullptr);
  memset(ar_.get(), 0, sizeof(mz_zip_archive));

  size_t size = in_->size();

  // check for the old magic number,
  constexpr size_t kMagicValueLength = 8;
  if (size > kMagicValueLength) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    char buf[kMagicValueLength];
    read(0, buf, kMagicValueLength);
    valid("checking magic number");
    TORCH_INTERNAL_ASSERT(
        memcmp("PYTORCH1", buf, kMagicValueLength) != 0,
        "File is an unsupported archive format from the preview release.");
  }

  ar_->m_pIO_opaque = this;
  ar_->m_pRead = istream_read_func;

  mz_zip_reader_init(ar_.get(), size, 0);
  valid("reading zip archive");

  // figure out the archive_name (i.e. the zip folder all the other files are
  // in) all lookups to getRecord will be prefixed by this folder
  mz_uint n = mz_zip_reader_get_num_files(ar_.get());
  if (n == 0) {
    CAFFE_THROW("archive does not contain any files");
  }
  size_t name_size = mz_zip_reader_get_filename(ar_.get(), 0, nullptr, 0);
  valid("getting filename");
  std::string buf(name_size, '\0');
  mz_zip_reader_get_filename(ar_.get(), 0, &buf[0], name_size);
  valid("getting filename");
  auto pos = buf.find_first_of('/');
  if (pos == std::string::npos) {
    CAFFE_THROW("file in archive is not in a subdirectory: ", buf);
  }
  archive_name_ = buf.substr(0, pos);
  archive_name_plus_slash_ = archive_name_ + "/";

  // read serialization id
  if (hasRecord(kSerializationIdRecordName)) {
    at::DataPtr serialization_id_ptr;
    size_t serialization_id_size = 0;
    std::tie(serialization_id_ptr, serialization_id_size) =
        getRecord(kSerializationIdRecordName);
    serialization_id_.assign(
        static_cast<const char*>(serialization_id_ptr.get()),
        serialization_id_size);
  }
  c10::LogAPIUsageMetadata(
      "pytorch.stream.reader.metadata",
      {{"serialization_id", serialization_id_},
       {"file_name", archive_name_},
       {"file_size", str(mz_zip_get_archive_size(ar_.get()))}});

  // version check
  at::DataPtr version_ptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t version_size;
  if (hasRecord(".data/version")) {
    std::tie(version_ptr, version_size) = getRecord(".data/version");
  } else {
    TORCH_CHECK(hasRecord("version"))
    std::tie(version_ptr, version_size) = getRecord("version");
  }
  std::string version(
      static_cast<const char*>(version_ptr.get()), version_size);
  try {
    version_ = std::stoull(version);
  } catch (const std::invalid_argument&) {
    CAFFE_THROW("Couldn't parse the version ", version, " as Long Long.");
  }
  if (version_ <
      static_cast<decltype(version_)>(kMinSupportedFileFormatVersion)) {
    CAFFE_THROW(
        "Attempted to read a PyTorch file with version ",
        std::to_string(version_),
        ", but the minimum supported version for reading is ",
        std::to_string(kMinSupportedFileFormatVersion),
        ". Your PyTorch script module file is too old. Please regenerate it",
        " with latest version of PyTorch to mitigate this issue.");
  }

  if (version_ >
      static_cast<decltype(version_)>(kMaxSupportedFileFormatVersion)) {
    CAFFE_THROW(
        "Attempted to read a PyTorch file with version ",
        version_,
        ", but the maximum supported version for reading is ",
        kMaxSupportedFileFormatVersion,
        ". The version of your PyTorch installation may be too old, ",
        "please upgrade PyTorch to latest version to mitigate this issue.");
  }
}

void PyTorchStreamReader::valid(const char* what, const char* info) {
  const auto err = mz_zip_get_last_error(ar_.get());
  TORCH_CHECK(
      err == MZ_ZIP_NO_ERROR,
      "PytorchStreamReader failed ",
      what,
      info,
      ": ",
      mz_zip_get_error_string(err));
}

constexpr int MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30;
constexpr int MZ_ZIP_LDH_FILENAME_LEN_OFS = 26;
constexpr int MZ_ZIP_LDH_EXTRA_LEN_OFS = 28;
constexpr int MZ_ZIP_DATA_DESCRIPTOR_ID = 0x08074b50;

namespace detail {

std::tuple<size_t, size_t> getOffset(
    size_t cursor,
    size_t filename_size,
    size_t size,
    uint64_t alignment) {
  size_t start = cursor + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename_size +
      sizeof(mz_uint16) * 2;
  if (size >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX) {
    start += sizeof(mz_uint16) * 2;
    if (size >= MZ_UINT32_MAX) {
      start += 2 * sizeof(mz_uint64);
    }
    if (cursor >= MZ_UINT32_MAX) {
      start += sizeof(mz_uint64);
    }
  }
  size_t mod = start % alignment;
  size_t next_offset = (mod == 0) ? start : (start + alignment - mod);
  std::tuple<size_t, size_t> result(next_offset, start);
  return result;
}

size_t getPadding(
    size_t cursor,
    size_t filename_size,
    size_t size,
    std::string& padding_buf,
    uint64_t alignment) {
  auto [next_offset, start] = getOffset(cursor, filename_size, size, alignment);
  size_t padding_size = next_offset - start;
  size_t padding_size_plus_fbxx = padding_size + 4;
  if (padding_buf.size() < padding_size_plus_fbxx) {
    padding_buf.append(padding_size_plus_fbxx - padding_buf.size(), 'Z');
  }
  // zip extra encoding (key, size_of_extra_bytes)
  padding_buf[0] = 'F';
  padding_buf[1] = 'B';
  padding_buf[2] = (uint8_t)padding_size;
  padding_buf[3] = (uint8_t)(padding_size >> 8);
  return padding_size_plus_fbxx;
}
} // namespace detail

bool PyTorchStreamReader::hasRecord(const std::string& name) {
  std::lock_guard<std::mutex> guard(reader_lock_);

  if ((!load_debug_symbol_) &&
      c10::ends_with(std::string_view(name), kDebugPklSuffix)) {
    return false;
  }
  std::string ss = archive_name_plus_slash_ + name;
  mz_zip_reader_locate_file(ar_.get(), ss.c_str(), nullptr, 0);
  const mz_zip_error err = mz_zip_get_last_error(ar_.get());

  if (err == MZ_ZIP_NO_ERROR) {
    return true;
  } else if (err == MZ_ZIP_FILE_NOT_FOUND) {
    return false;
  } else {
    // A different error happened, raise it.
    valid("attempting to locate file ", name.c_str());
  }
  TORCH_INTERNAL_ASSERT(false, "should not reach here");
}

std::vector<std::string> PyTorchStreamReader::getAllRecords() {
  std::lock_guard<std::mutex> guard(reader_lock_);
  mz_uint num_files = mz_zip_reader_get_num_files(ar_.get());
  std::vector<std::string> out;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char buf[MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE];
  for (size_t i = 0; i < num_files; i++) {
    mz_zip_reader_get_filename(
        ar_.get(), i, buf, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE);
    if (strncmp(
            buf,
            archive_name_plus_slash_.data(),
            archive_name_plus_slash_.size()) != 0) {
      CAFFE_THROW(
          "file in archive is not in a subdirectory ",
          archive_name_plus_slash_,
          ": ",
          buf);
    }
    if ((load_debug_symbol_) ||
        (!c10::ends_with(
            std::string_view(buf + archive_name_plus_slash_.size()),
            kDebugPklSuffix))) {
      // NOLINTNEXTLINE(modernize-use-emplace)
      out.push_back(buf + archive_name_plus_slash_.size());
    }
  }
  return out;
}

const std::unordered_set<std::string>&
PyTorchStreamWriter::getAllWrittenRecords() {
  return files_written_;
}

size_t PyTorchStreamReader::getRecordID(const std::string& name) {
  std::string ss = archive_name_plus_slash_ + name;
  size_t result = mz_zip_reader_locate_file(ar_.get(), ss.c_str(), nullptr, 0);
  valid("locating file ", name.c_str());
  return result;
}

// return dataptr, size
std::tuple<at::DataPtr, size_t> PyTorchStreamReader::getRecord(
    const std::string& name,
    std::optional<at::Allocator*> allocator) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  if ((!load_debug_symbol_) && c10::ends_with(name, kDebugPklSuffix)) {
    at::DataPtr retval;
    return std::make_tuple(std::move(retval), 0);
  }
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  valid("retrieving file meta-data for ", name.c_str());
  at::Allocator* allocatorPtr =
      allocator.has_value() ? allocator.value() : c10::GetCPUAllocator();
  at::DataPtr retval = allocatorPtr->allocate(stat.m_uncomp_size);
  mz_zip_reader_extract_to_mem(
      ar_.get(), key, retval.get(), stat.m_uncomp_size, 0);
  valid("reading file ", name.c_str());

  return std::make_tuple(std::move(retval), stat.m_uncomp_size);
}

size_t PyTorchStreamReader::getRecordMultiReaders(
    const std::string& name,
    std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders,
    void* dst,
    size_t n) {
  size_t nthread = additionalReaders.size() + 1;
  size_t recordOff = getRecordOffset(name);
  std::vector<std::thread> loaderThreads;
  size_t perThreadSize = (n + nthread - 1) / nthread;
  std::vector<size_t> readSizes(nthread, 0);
  std::lock_guard<std::mutex> guard(reader_lock_);
  for (size_t i = 0; i < nthread; i++) {
    loaderThreads.emplace_back([this,
                                name,
                                i,
                                n,
                                recordOff,
                                perThreadSize,
                                dst,
                                &additionalReaders,
                                &readSizes] {
      size_t startPos = i * perThreadSize;
      size_t endPos = std::min((i + 1) * perThreadSize, n);
      if (startPos < endPos) {
        size_t threadReadSize = endPos - startPos;
        size_t size = 0;
        if (i == 0) {
          size =
              read(recordOff + startPos, (char*)dst + startPos, threadReadSize);
        } else {
          auto reader = additionalReaders[i - 1];
          size = reader->read(
              recordOff + startPos, (char*)dst + startPos, threadReadSize);
        }
        readSizes[i] = size;
        LOG(INFO) << "Thread " << i << " read [" << startPos << "-" << endPos
                  << "] " << "from " << name << " of size " << n;
        TORCH_CHECK(
            threadReadSize == size,
            "record size ",
            threadReadSize,
            " mismatch with read size ",
            size);
      }
    });
  }

  for (auto& thread : loaderThreads) {
    thread.join();
  }
  loaderThreads.clear();

  size_t total_read_n = 0;
  for (auto& r : readSizes) {
    total_read_n += r;
  }

  TORCH_CHECK(
      n == total_read_n,
      "Multi reader total read size ",
      total_read_n,
      " mismatch with dst size ",
      n);

  return total_read_n;
}

// read record with multi clients
std::tuple<at::DataPtr, size_t> PyTorchStreamReader::getRecord(
    const std::string& name,
    std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders,
    std::optional<at::Allocator*> allocator) {
  if (additionalReaders.empty()) {
    // No additional readers or record too small, use single threaded version
    return getRecord(name, allocator);
  }

  if ((!load_debug_symbol_) && c10::ends_with(name, kDebugPklSuffix)) {
    at::DataPtr retval;
    return std::make_tuple(std::move(retval), 0);
  }
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  auto n = stat.m_uncomp_size;
  valid("retrieving file meta-data for ", name.c_str());
  if (n < additional_reader_size_threshold_) {
    // Reader size too small, use single threaded version
    return getRecord(name);
  }

  at::Allocator* allocatorPtr =
      allocator.has_value() ? allocator.value() : c10::GetCPUAllocator();
  at::DataPtr retval = allocatorPtr->allocate(stat.m_uncomp_size);
  void* dst = retval.get();
  PyTorchStreamReader::getRecordMultiReaders(name, additionalReaders, dst, n);
  return std::make_tuple(std::move(retval), stat.m_uncomp_size);
}

// inplace memory writing
size_t
PyTorchStreamReader::getRecord(const std::string& name, void* dst, size_t n) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  if ((!load_debug_symbol_) && c10::ends_with(name, kDebugPklSuffix)) {
    return 0;
  }
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  TORCH_CHECK(
      n == stat.m_uncomp_size,
      "record size ",
      stat.m_uncomp_size,
      " mismatch with dst size ",
      n);
  valid("retrieving file meta-data for ", name.c_str());
  mz_zip_reader_extract_to_mem(ar_.get(), key, dst, stat.m_uncomp_size, 0);
  valid("reading file ", name.c_str());

  return stat.m_uncomp_size;
}

// inplace memory writing, in-tensor multi-threads, can be used for large
// tensor.
size_t PyTorchStreamReader::getRecord(
    const std::string& name,
    void* dst,
    size_t n,
    std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders) {
  if (additionalReaders.empty()) {
    // No additional readers, use single threaded version
    return getRecord(name, dst, n);
  }

  if ((!load_debug_symbol_) &&
      c10::ends_with(std::string_view(name), kDebugPklSuffix)) {
    return 0;
  }
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  TORCH_CHECK(
      n == stat.m_uncomp_size,
      "record size ",
      stat.m_uncomp_size,
      " mismatch with dst size ",
      n);
  valid("retrieving file meta-data for ", name.c_str());

  if (n < additional_reader_size_threshold_) {
    // Reader size too small, use single threaded version
    return getRecord(name, dst, n);
  }

  PyTorchStreamReader::getRecordMultiReaders(name, additionalReaders, dst, n);
  return stat.m_uncomp_size;
}

size_t PyTorchStreamReader::getRecord(
    const std::string& name,
    void* dst,
    size_t n,
    size_t chunk_size,
    void* buf,
    const std::function<void(void*, const void*, size_t)>& memcpy_func) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  if ((!load_debug_symbol_) && c10::ends_with(name, kDebugPklSuffix)) {
    return 0;
  }
  if (chunk_size <= 0) {
    chunk_size = n;
  }
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  TORCH_CHECK(
      n == stat.m_uncomp_size,
      "record size ",
      stat.m_uncomp_size,
      " mismatch with dst size ",
      n);
  valid("retrieving file meta-data for ", name.c_str());

  std::vector<uint8_t> buffer;
  if (buf == nullptr) {
    buffer.resize(chunk_size);
    buf = buffer.data();
  }

  auto chunkIterator =
      createChunkReaderIter(name, (size_t)stat.m_uncomp_size, chunk_size);
  while (auto readSize = chunkIterator.next(buf)) {
    memcpy_func((char*)dst + chunkIterator.offset_ - readSize, buf, readSize);
  }
  valid("reading file ", name.c_str());

  return stat.m_uncomp_size;
}

ChunkRecordIterator PyTorchStreamReader::createChunkReaderIter(
    const std::string& name,
    const size_t recordSize,
    const size_t chunkSize) {
  // Create zip reader iterator
  size_t key = getRecordID(name);
  mz_zip_reader_extract_iter_state* zipReaderIter =
      mz_zip_reader_extract_iter_new(ar_.get(), key, 0);
  TORCH_CHECK(
      zipReaderIter != nullptr,
      "Failed to create zip reader iter: ",
      mz_zip_get_error_string(mz_zip_get_last_error(ar_.get())));

  return ChunkRecordIterator(
      recordSize,
      chunkSize,
      std::make_unique<MzZipReaderIterWrapper>(zipReaderIter));
}

static int64_t read_le_16(uint8_t* buf) {
  return buf[0] + (buf[1] << 8);
}

size_t PyTorchStreamReader::getRecordHeaderOffset(const std::string& name) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), getRecordID(name), &stat);
  valid("retrieving file meta-data for ", name.c_str());
  return stat.m_local_header_ofs;
}

size_t PyTorchStreamReader::getRecordOffset(const std::string& name) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), getRecordID(name), &stat);
  valid("retrieving file meta-data for ", name.c_str());
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  uint8_t local_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
  in_->read(
      stat.m_local_header_ofs,
      local_header,
      MZ_ZIP_LOCAL_DIR_HEADER_SIZE,
      "reading file header");
  size_t filename_len = read_le_16(local_header + MZ_ZIP_LDH_FILENAME_LEN_OFS);
  size_t extra_len = read_le_16(local_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
  return stat.m_local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename_len +
      extra_len;
}

size_t PyTorchStreamReader::getRecordSize(const std::string& name) {
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), getRecordID(name), &stat);
  return stat.m_uncomp_size;
}

size_t PyTorchStreamReader::getRecordOffsetNoRead(
    size_t cursor,
    std::string filename,
    size_t size,
    uint64_t alignment) {
  std::string full_name = archive_name_plus_slash_ + filename;
  size_t full_name_size = full_name.size();
  std::tuple<size_t, size_t> result =
      detail::getOffset(cursor, full_name_size, size, alignment);
  size_t offset = std::get<0>(result);
  return offset;
}

PyTorchStreamReader::~PyTorchStreamReader() {
  mz_zip_clear_last_error(ar_.get());
  mz_zip_reader_end(ar_.get());
  valid("closing reader for archive ", archive_name_.c_str());
}

size_t ostream_write_func(
    void* pOpaque,
    mz_uint64 file_ofs,
    const void* pBuf,
    size_t n) {
  auto self = static_cast<PyTorchStreamWriter*>(pOpaque);
  if (self->current_pos_ != file_ofs) {
    CAFFE_THROW("unexpected pos ", self->current_pos_, " vs ", file_ofs);
  }
  size_t ret = self->writer_func_(pBuf, n);
  if (n != ret) {
    self->err_seen_ = true;
  }
  self->current_pos_ += ret;

  // Get the CRC32 of uncompressed data from the data descriptor, if the written
  // data is identified as the data descriptor block.
  // See [Note: write_record_metadata] for why we check for non-null pBuf here
  if (pBuf && n >= 8 && MZ_READ_LE32(pBuf) == MZ_ZIP_DATA_DESCRIPTOR_ID) {
    const int8_t* pInt8Buf = (const int8_t*)pBuf;
    const uint32_t uncomp_crc32 = MZ_READ_LE32(pInt8Buf + 4);
    self->combined_uncomp_crc32_ =
        c10::hash_combine(self->combined_uncomp_crc32_, uncomp_crc32);
  }

  return ret;
}

PyTorchStreamWriter::PyTorchStreamWriter(
    const std::string& file_name,
    bool compute_crc32,
    uint64_t alignment)
    : archive_name_(basename(file_name)),
      compute_crc32_(compute_crc32),
      alignment_(alignment) {
  setup(file_name);
}

PyTorchStreamWriter::PyTorchStreamWriter(
    const std::function<size_t(const void*, size_t)> writer_func,
    bool compute_crc32,
    uint64_t alignment)
    : archive_name_("archive"),
      writer_func_(writer_func),
      compute_crc32_(compute_crc32),
      alignment_(alignment) {
  setup(archive_name_);
}

void PyTorchStreamWriter::setup(const string& file_name) {
  ar_ = std::make_unique<mz_zip_archive>();
  memset(ar_.get(), 0, sizeof(mz_zip_archive));
  archive_name_plus_slash_ = archive_name_ + "/"; // for writeRecord().

  if (archive_name_.size() == 0) {
    CAFFE_THROW("invalid file name: ", file_name);
  }
  if (!writer_func_) {
    file_stream_.open(
        file_name,
        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
    valid("opening archive ", file_name.c_str());

    const std::string dir_name = parentdir(file_name);
    if (!dir_name.empty()) {
      struct stat st;
      bool dir_exists =
          (stat(dir_name.c_str(), &st) == 0 && (st.st_mode & S_IFDIR));
      TORCH_CHECK(
          dir_exists, "Parent directory ", dir_name, " does not exist.");
    }
    TORCH_CHECK(file_stream_, "File ", file_name, " cannot be opened.");
    writer_func_ = [this](const void* buf, size_t nbytes) -> size_t {
      if (!buf) {
        // See [Note: write_record_metadata]
        file_stream_.seekp(nbytes, std::ios_base::cur);
      } else {
        file_stream_.write(static_cast<const char*>(buf), nbytes);
      }
      return !file_stream_ ? 0 : nbytes;
    };
  }

  ar_->m_pIO_opaque = this;
  ar_->m_pWrite = ostream_write_func;

  mz_zip_writer_init_v2(ar_.get(), 0, MZ_ZIP_FLAG_WRITE_ZIP64);
  valid("initializing archive ", file_name.c_str());
}

void PyTorchStreamWriter::setMinVersion(const uint64_t version) {
  version_ = std::max(version, version_);
}

void PyTorchStreamWriter::writeRecord(
    const std::string& name,
    const void* data,
    size_t size,
    bool compress) {
  AT_ASSERT(!finalized_);
  AT_ASSERT(!archive_name_plus_slash_.empty());
  TORCH_INTERNAL_ASSERT(
      files_written_.count(name) == 0, "Tried to serialize file twice: ", name);
  if (name == kSerializationIdRecordName && serialization_id_.empty()) {
    // In case of copying records from another file, skip writing a different
    // serialization_id than the one computed in this writer.
    // This is to ensure serialization_id is unique per serialization output.
    return;
  }
  std::string full_name = archive_name_plus_slash_ + name;
  size_t padding_size = detail::getPadding(
      ar_->m_archive_size, full_name.size(), size, padding_, alignment_);
  uint32_t flags = compress ? MZ_BEST_COMPRESSION : 0;
  if (!compute_crc32_) {
#if (!defined(FBCODE_CAFFE2))
    flags |= MZ_ZIP_FLAG_DO_NOT_COMPUTE_CRC32;
#endif
  }
  mz_zip_writer_add_mem_ex_v2(
      /*pZip=*/ar_.get(),
      /*pArchive_name=*/full_name.c_str(),
      /*pBuf=*/data,
      /*buf_size=*/size,
      /*pComment=*/nullptr,
      /*comment_size=*/0,
      /*level_and_flags=*/flags,
      /*uncomp_size=*/0,
      /*uncomp_crc32=*/0,
      /*last_modified=*/nullptr,
      /*user_extra_data=*/padding_.c_str(),
      /*user_extra_data_len=*/padding_size,
      /*user_extra_data_central=*/nullptr,
      /*user_extra_data_central_len=*/0);
  valid("writing file ", name.c_str());
  files_written_.insert(name);
}

void PyTorchStreamWriter::writeEndOfFile() {
  // Ensurers that finalized is set to true even
  // exception is raised during the method call.
  // I.e. even partial call to writeEndOfFile() should mark
  // file as finalized, otherwise double exception raised from
  // destructor would would result in `std::terminate()`
  // See https://github.com/pytorch/pytorch/issues/87997/
  struct Finalizer {
    Finalizer(bool& var) : var_(var) {}
    ~Finalizer() {
      var_ = true;
    }

   private:
    bool& var_;
  } f(finalized_);

  auto allRecords = getAllWrittenRecords();
  // If no ".data/version" or "version" record in the output model, rewrites
  // version info
  if (allRecords.find(".data/version") == allRecords.end() &&
      allRecords.find("version") == allRecords.end()) {
    std::string version = std::to_string(version_);
    version.push_back('\n');
    if (version_ >= 0x6L) {
      writeRecord(".data/version", version.c_str(), version.size());
    } else {
      writeRecord("version", version.c_str(), version.size());
    }
  }

  // If no "byteorder" record in the output model, rewrites byteorder info
  if (allRecords.find("byteorder") == allRecords.end()) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::string byteorder = "little";
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    std::string byteorder = "big";
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
    writeRecord("byteorder", byteorder.c_str(), byteorder.size());
  }

  writeSerializationId();

  AT_ASSERT(!finalized_);
  finalized_ = true;

  mz_zip_writer_finalize_archive(ar_.get());
  mz_zip_writer_end(ar_.get());
  valid("writing central directory for archive ", archive_name_.c_str());
  c10::LogAPIUsageMetadata(
      "pytorch.stream.writer.metadata",
      {{"serialization_id", serialization_id_},
       {"file_name", archive_name_},
       {"file_size", str(mz_zip_get_archive_size(ar_.get()))}});
  if (file_stream_.is_open()) {
    file_stream_.close();
  }
}

void PyTorchStreamWriter::valid(const char* what, const char* info) {
  auto err = mz_zip_get_last_error(ar_.get());
  if (err != MZ_ZIP_NO_ERROR) {
    CAFFE_THROW(
        "PytorchStreamWriter failed ",
        what,
        info,
        ": ",
        mz_zip_get_error_string(err));
  }
  if (err_seen_) {
    CAFFE_THROW("PytorchStreamWriter failed ", what, info, ".");
  }
}

void PyTorchStreamWriter::writeSerializationId() {
  // Serialization id is computed based on all files written, and is composed of
  // 1) a combined hash of record name hashes
  // 2) a combined crc32 of the record uncompressed data
  // This is best effort to create a fixed-length, unique and deterministic id
  // for the serialized files without incurring additional computation overhead.
  if (files_written_.find(kSerializationIdRecordName) == files_written_.end()) {
    uint64_t combined_record_name_hash = 0;
    for (const std::string& record_name : files_written_) {
      size_t record_name_hash = c10::hash<std::string>{}(record_name);
      combined_record_name_hash =
          c10::hash_combine(combined_record_name_hash, record_name_hash);
    }
    std::ostringstream serialization_id_oss;
    serialization_id_oss << std::setfill('0') << std::setw(20)
                         << combined_record_name_hash << std::setfill('0')
                         << std::setw(20) << combined_uncomp_crc32_;
    serialization_id_ = serialization_id_oss.str();
    writeRecord(
        kSerializationIdRecordName,
        serialization_id_.c_str(),
        serialization_id_.size());
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PyTorchStreamWriter::~PyTorchStreamWriter() {
  if (!finalized_) {
    writeEndOfFile();
  }
}

} // namespace serialize
} // namespace caffe2
