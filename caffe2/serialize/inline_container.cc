#include <cstdio>
#include <cstring>
#include <cerrno>
#include <istream>
#include <ostream>
#include <fstream>

#include <c10/core/Allocator.h>
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
    : ar_(caffe2::make_unique<mz_zip_archive>()),
      in_(caffe2::make_unique<FileAdapter>(file_name)) {
  init();
}

PyTorchStreamReader::PyTorchStreamReader(std::istream* in)
    : ar_(caffe2::make_unique<mz_zip_archive>()),
      in_(caffe2::make_unique<IStreamAdapter>(in)) {
  init();
}

PyTorchStreamReader::PyTorchStreamReader(
    std::unique_ptr<ReadAdapterInterface> in)
    : ar_(caffe2::make_unique<mz_zip_archive>()), in_(std::move(in)) {
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

  // version check
  at::DataPtr version_ptr;
  size_t version_size;
  std::tie(version_ptr, version_size) = getRecord("version");
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

void PyTorchStreamReader::valid(const char* what) {
  auto err = mz_zip_get_last_error(ar_.get());
  if (err != MZ_ZIP_NO_ERROR) {
    CAFFE_THROW("PytorchStreamReader failed ", what, ": ", mz_zip_get_error_string(err));
  }
}

constexpr int MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30;
constexpr int MZ_ZIP_LDH_FILENAME_LEN_OFS = 26;
constexpr int MZ_ZIP_LDH_EXTRA_LEN_OFS = 28;

static std::string getPadding(size_t cursor, const std::string& filename, size_t size) {
  size_t start = cursor + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename.size() + sizeof(mz_uint16) * 2;
  if (size >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX) {
    start += sizeof(mz_uint16) * 2;
    if (size >= MZ_UINT32_MAX) {
      start += 2*sizeof(mz_uint64);
    }
    if (cursor >= MZ_UINT32_MAX) {
      start += sizeof(mz_uint64);
    }
  }
  size_t mod = start % kFieldAlignment;
  size_t next_offset = (mod == 0) ? start : (start + kFieldAlignment - mod);
  size_t padding_size = next_offset - start;
  std::string buf(padding_size + 4, 'Z');
  // zip extra encoding (key, size_of_extra_bytes)
  buf[0] = 'F';
  buf[1] = 'B';
  buf[2] = (uint8_t) padding_size;
  buf[3] = (uint8_t) (padding_size >> 8);
  return buf;
}

bool PyTorchStreamReader::hasRecord(const std::string& name) {
  std::stringstream ss;
  ss << archive_name_ << "/" << name;
  mz_zip_reader_locate_file(ar_.get(), ss.str().c_str(), nullptr, 0);
  bool result = ar_->m_last_error != MZ_ZIP_FILE_NOT_FOUND;
  if (!result) {
    ar_->m_last_error = MZ_ZIP_NO_ERROR;
  }
  valid("attempting to locate file");
  return result;
}

size_t PyTorchStreamReader::getRecordID(const std::string& name) {
  std::stringstream ss;
  ss << archive_name_ << "/" << name;
  size_t result = mz_zip_reader_locate_file(ar_.get(), ss.str().c_str(), nullptr, 0);
  if (ar_->m_last_error == MZ_ZIP_FILE_NOT_FOUND) {
    CAFFE_THROW("file not found: ", ss.str());
  }
  valid("locating file");
  return result;
}

// return dataptr, size
std::tuple<at::DataPtr, size_t> PyTorchStreamReader::getRecord(const std::string& name) {
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), key, &stat);
  valid("retrieving file meta-data");
  void * ptr = malloc(stat.m_uncomp_size);
  mz_zip_reader_extract_to_mem(ar_.get(), key, ptr, stat.m_uncomp_size, 0);
  valid("reading file");

  at::DataPtr retval(ptr, ptr, free, at::kCPU);
  return std::make_tuple(std::move(retval), stat.m_uncomp_size);
}

static int64_t read_le_16(uint8_t* buf) {
  return buf[0] + (buf[1] << 8);
}

size_t PyTorchStreamReader::getRecordOffset(const std::string& name) {
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat(ar_.get(), getRecordID(name), &stat);
  valid("retriving file meta-data");
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
  mz_zip_reader_end(ar_.get());
  valid("closing reader");
}

size_t ostream_write_func(void *pOpaque, mz_uint64 file_ofs, const void *pBuf, size_t n) {
  auto self = static_cast<PyTorchStreamWriter*>(pOpaque);
  if (self->current_pos_ != file_ofs) {
    // xxx - windows ostringstream refuses to seek to the end of an empty string
    // so we workaround this by not calling seek unless necessary
    // in the case of the first write (to the empty string) file_ofs and
    // current_pos_ will be 0 and the seek won't occur.
    self->out_->seekp(file_ofs);
    if(!*self->out_)
      return 0;
  }

  self->out_->write(static_cast<const char*>(pBuf), n);
  if(!*self->out_)
    return 0;
  self->current_pos_ = file_ofs + n;
  return n;
}

PyTorchStreamWriter::PyTorchStreamWriter(
    std::string file_name,
    std::ostream* out)
    : ar_(caffe2::make_unique<mz_zip_archive>()),
      archive_name_(basename(file_name)),
      out_(out) {
  memset(ar_.get(), 0, sizeof(mz_zip_archive));

  if (archive_name_.size() == 0) {
    CAFFE_THROW("invalid file name: ", file_name);
  }
  if (!out_) {
    file_stream_.open(file_name, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
    out_ = &file_stream_;
    valid("opening archive");
  }

  ar_->m_pIO_opaque = this;
  ar_->m_pWrite = ostream_write_func;

  mz_zip_writer_init_v2(ar_.get(), 0, MZ_ZIP_FLAG_WRITE_ZIP64);
  valid("initializing archive");

  std::stringstream version;
  version << kMaxSupportedFileFormatVersion << "\n";
  writeRecord("version", version.str().c_str(), version.str().size());
}

void PyTorchStreamWriter::writeRecord(const std::string& name, const void* data, size_t size, bool compress) {
  AT_ASSERT(!finalized_);
  std::stringstream ss;
  ss << archive_name_ << "/" << name;
  const std::string& full_name = ss.str();
  std::string padding = getPadding(ar_->m_archive_size, full_name, size);
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
      padding.c_str(),
      padding.size(),
      nullptr,
      0);
  valid("writing file");
}

void PyTorchStreamWriter::writeEndOfFile() {
  AT_ASSERT(!finalized_);
  finalized_ = true;
  mz_zip_writer_finalize_archive(ar_.get());
  mz_zip_writer_end(ar_.get());
  valid("writing central directory");
  if (file_stream_.is_open())
    file_stream_.close();
}


void PyTorchStreamWriter::valid(const char* what) {
  auto err = mz_zip_get_last_error(ar_.get());
  if (err != MZ_ZIP_NO_ERROR) {
    CAFFE_THROW("PytorchStreamWriter failed ", what, ": ", mz_zip_get_error_string(err));
  }
  if (!*out_) {
    CAFFE_THROW("PytorchStreamWriter failed ", what, ".");
  }
}

PyTorchStreamWriter::~PyTorchStreamWriter() {
  if (!finalized_) {
    writeEndOfFile();
  }
}

} // namespace serialize
} // namespace caffe2
