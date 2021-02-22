#include <cstdio>
#include <cstring>
#include <cerrno>
#include <istream>
#include <ostream>
#include <fstream>
#include <algorithm>

#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Backend.h>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/inline_container.h"
#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"

#include "miniz.h"

namespace caffe2 {
namespace serialize {

size_t istream_read_func(void *pOpaque, mz_uint64 file_ofs, void *pBuf, size_t n) {
  auto self = static_cast<PyTorchStreamReader*>(pOpaque);
  return self->read(file_ofs, static_cast<char*>(pBuf), n);
}

static std::string basename(const std::string& name) {
  size_t start = 0;
  for(size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '\\' || name[i] == '/') {
      start = i + 1;
    }
  }

  if (start >= name.size())
    return "";

  size_t end = name.size();
  for(size_t i = end; i > start; --i) {
    if (name[i - 1] == '.') {
      end = i - 1;
      break;
    }
  }
  return name.substr(start, end - start);
}

size_t PyTorchStreamReader::read(uint64_t pos, char* buf, size_t n) {
  return in_->read(pos, buf, n, "reading file");
}

PyTorchStreamReader::PyTorchStreamReader(const std::string& file_name)
    : ar_(std::make_unique<mz_zip_archive>()),
      in_(std::make_unique<FileAdapter>(file_name)) {
  init();
}

PyTorchStreamReader::PyTorchStreamReader(std::istream* in)
    : ar_(std::make_unique<mz_zip_archive>()),
      in_(std::make_unique<IStreamAdapter>(in)) {
  init();
}

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
    char buf[kMagicValueLength];
    read(0, buf, kMagicValueLength);
    valid("checking magic number");
    AT_ASSERTM(
        memcmp("PYTORCH1", buf, kMagicValueLength) != 0,
        "File is an unsupported archive format from the preview release.");
  }

  ar_->m_pIO_opaque = this;
  ar_->m_pRead = istream_read_func;

  mz_zip_reader_init(ar_.get(), size, 0);
  valid("reading zip archive");

  // figure out the archive_name (i.e. the zip folder all the other files are in)
  // all lookups to getRecord will be prefixed by this folder
  int n = mz_zip_reader_get_num_files(ar_.get());
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

  // version check
  at::DataPtr version_ptr;
  size_t version_size;
  if (hasRecord(".data/version")) {
    std::tie(version_ptr, version_size) = getRecord(".data/version");
  } else {
    TORCH_CHECK(hasRecord("version"))
    std::tie(version_ptr, version_size) = getRecord("version");
  }
  std::string version(static_cast<const char*>(version_ptr.get()), version_size);
  version_ = caffe2::stoull(version);
  AT_ASSERTM(
      version_ >= kMinSupportedFileFormatVersion,
      "Attempted to read a PyTorch file with version ",
      c10::to_string(version_),
      ", but the minimum supported version for reading is ",
      c10::to_string(kMinSupportedFileFormatVersion),
      ". Your PyTorch script module file is too old. Please re-export it again.");
  AT_ASSERTM(
      version_ <= kMaxSupportedFileFormatVersion,
      "Attempted to read a PyTorch file with version ",
      version_,
      ", but the maximum supported version for reading is ",
      kMaxSupportedFileFormatVersion,
      ". Your PyTorch installation may be too old.");
}

void PyTorchStreamReader::valid(const char* what, const char* info) {
  auto err = mz_zip_get_last_error(ar_.get());
  if (err != MZ_ZIP_NO_ERROR) {
    CAFFE_THROW(
        "PytorchStreamReader failed ",
        what,
        info,
        ": ",
        mz_zip_get_error_string(err));
  }
}

constexpr int MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30;
constexpr int MZ_ZIP_LDH_FILENAME_LEN_OFS = 26;
constexpr int MZ_ZIP_LDH_EXTRA_LEN_OFS = 28;

namespace detail {
size_t getPadding(
    size_t cursor,
    size_t filename_size,
    size_t size,
    std::string& padding_buf) {
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
  size_t mod = start % kFieldAlignment;
  size_t next_offset = (mod == 0) ? start : (start + kFieldAlignment - mod);
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
}

bool PyTorchStreamReader::hasRecord(const std::string& name) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  std::string ss = archive_name_plus_slash_ + name;
  mz_zip_reader_locate_file(ar_.get(), ss.c_str(), nullptr, 0);
  bool result = ar_->m_last_error != MZ_ZIP_FILE_NOT_FOUND;
  if (!result) {
    ar_->m_last_error = MZ_ZIP_NO_ERROR;
  }
  valid("attempting to locate file ", name.c_str());
  return result;
}

std::vector<std::string> PyTorchStreamReader::getAllRecords() {
  std::lock_guard<std::mutex> guard(reader_lock_);
  mz_uint num_files = mz_zip_reader_get_num_files(ar_.get());
  std::vector<std::string> out;
  char buf[MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE];
  for (size_t i = 0; i < num_files; i++) {
    mz_zip_reader_get_filename(ar_.get(), i, buf, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE);
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
    out.push_back(buf + archive_name_plus_slash_.size());
  }
  return out;
}

const std::vector<std::string>& PyTorchStreamWriter::getAllWrittenRecords() {
  return files_written;
}

size_t PyTorchStreamReader::getRecordID(const std::string& name) {
  std::string ss = archive_name_plus_slash_ + name;
  size_t result = mz_zip_reader_locate_file(ar_.get(), ss.c_str(), nullptr, 0);
  if (ar_->m_last_error == MZ_ZIP_FILE_NOT_FOUND) {
    CAFFE_THROW("file not found: ", ss);
  }
  valid("locating file ", name.c_str());
  return result;
}

// return dataptr, size
std::tuple<at::DataPtr, size_t> PyTorchStreamReader::getRecord(const std::string& name) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  valid("retrieving file meta-data for ", name.c_str());
  at::DataPtr retval = c10::GetCPUAllocator()->allocate(stat.m_uncomp_size);
  mz_zip_reader_extract_to_mem(ar_.get(), key, retval.get(), stat.m_uncomp_size, 0);
  valid("reading file ", name.c_str());

  return std::make_tuple(std::move(retval), stat.m_uncomp_size);
}

static int64_t read_le_16(uint8_t* buf) {
  return buf[0] + (buf[1] << 8);
}

size_t PyTorchStreamReader::getRecordOffset(const std::string& name) {
  std::lock_guard<std::mutex> guard(reader_lock_);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), getRecordID(name), &stat);
  valid("retrieving file meta-data for ", name.c_str());
  uint8_t local_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
  in_->read(
      stat.m_local_header_ofs,
      local_header,
      MZ_ZIP_LOCAL_DIR_HEADER_SIZE,
      "reading file header");
  size_t filename_len = read_le_16(local_header + MZ_ZIP_LDH_FILENAME_LEN_OFS);
  size_t extra_len = read_le_16(local_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
  return stat.m_local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename_len + extra_len;
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
  return ret;
}

PyTorchStreamWriter::PyTorchStreamWriter(std::string file_name)
    : archive_name_(basename(file_name)) {
  setup(file_name);
}

PyTorchStreamWriter::PyTorchStreamWriter(
    const std::function<size_t(const void*, size_t)>& writer_func)
    : archive_name_("archive"),
      writer_func_(writer_func) {
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
    TORCH_CHECK(file_stream_, "File ", file_name, " cannot be opened.");
    writer_func_ = [this](const void* buf, size_t nbytes) -> size_t {
      file_stream_.write(static_cast<const char*>(buf), nbytes);
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
  std::string full_name = archive_name_plus_slash_ + name;
  size_t padding_size =
      detail::getPadding(ar_->m_archive_size, full_name.size(), size, padding_);
  uint32_t flags = compress ? MZ_BEST_COMPRESSION : 0;
  mz_zip_writer_add_mem_ex_v2(
      ar_.get(),
      full_name.c_str(),
      data,
      size,
      nullptr,
      0,
      flags,
      0,
      0,
      nullptr,
      padding_.c_str(),
      padding_size,
      nullptr,
      0);
  valid("writing file ", name.c_str());
  files_written.push_back(name);
}

void PyTorchStreamWriter::writeEndOfFile() {
  // Rewrites version info
  std::string version = c10::to_string(version_);
  version.push_back('\n');
  if (version_ >= 0x6L) {
    writeRecord(".data/version", version.c_str(), version.size());
  } else {
    writeRecord("version", version.c_str(), version.size());
  }

  AT_ASSERT(!finalized_);
  finalized_ = true;

  mz_zip_writer_finalize_archive(ar_.get());
  mz_zip_writer_end(ar_.get());
  valid("writing central directory for archive ", archive_name_.c_str());
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

PyTorchStreamWriter::~PyTorchStreamWriter() {
  if (!finalized_) {
    writeEndOfFile();
  }
}

} // namespace serialize
} // namespace caffe2
