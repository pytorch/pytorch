#pragma once

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <istream>
#include <ostream>
#include <fstream>

#include <ATen/core/Allocator.h>

#include "caffe2/core/logging.h"

namespace torch { namespace jit {

// This file defines an on-disk serialization format to be used for PyTorch
// model serialization. All integer values are serialized as little-endian.
// Everything in this format is aligned to 64-byte boundaries to allow for direct
// memory mapping and use in, for example, AVX512 instructions.
// The format is as follows:
//
// -- File header --
// [8 bytes] Magic number - little endian integer that spells 'PYTORCH1' in ASCII
// [8 bytes] Version number - The version of this file format that this file is in.
//                            this allows us to revise and extend this format
// [48 bytes] Padding/reserved
//
// After the file header reside N records of the format
// [8 bytes] Tag - this is a tag that identifies the type of this record. The
//                 values are defined in the RecordTags enum below.
// [8 bytes] size - Size in bytes of the payload of this record
// [48 bytes] Pad/reserved - This space pads out the payload to a 64-byte alignment.
// [size bytes] Payload - The actual raw data for the object serialized in this record
// [size - (size % 64) bytes] Pad/reserved - pad out this record so the next
//                                                one is aligned to 64 bytes
//
// Following those records is a special footer:
// [8 bytes] Tag - This tag field should contain the value for RecordTags::FOOTER
//                 to correctly identify the footer
// [8 bytes] Offset of last record - The last record in this format is used
//                                   as an index into the rest of the file, so
//                                   a reader can use this offset to seek to
//                                   the last record and read the index.
// [48 bytes] Pad/reserved - Pad out the footer s.t. the whole file's size is a
//                           multiple of 64 bytes.
//
//
// When developing this format we want to pay particular attention to the
// following use cases:
//
// -- Reading --
// 1) Reading with full random access
//   a) Reading with file api's such as fread()
//   b) mmaping the file and jumping around the mapped region
// 2) Reading with 1-pass sequential access
//      -> A reader will need to build up a data structure of parsed structures
//         as it reads
//
// -- Writing --
// 1) Writing with full random access
// 2) Writing with 1-pass sequential access
//      -> We must take care not to require updating values that have already
//         been written. We place the variable-length index at the end and do
//         not put any indicies into the header to fulfill this constraint.

// The serialized model, which contains all the metadata information,
// should be stored as the last record. Because the size of tensor data is
// usually stable. As long as the shape and type of the tensor do not change,
// the size of the data won't change. On the other sied, the size of the
// serialized model is likely to change, so we store it as the last record, and
// we don't need to move previous records when updating the model data.

namespace {

enum RecordTags {
  STORAGE = 1,
  FOOTER = 2,
};

// Common constants
constexpr uint64_t kFileMagicNumber = 0x314843524f545950L; // PYTORCH1
constexpr uint64_t kFieldAlignment =
    64L; // 64 byte alignment supports up to AVX512 for mmap

// Reader-specific constants
constexpr uint64_t kMaxSupportedFileFormatVersion = 0x1L;

// Writer-specific constants
constexpr uint64_t kFileFormatVersion = 0x1L;
constexpr uint8_t kPadValue = 0xEF;

}  // namespace

class PyTorchStreamReader final {
 public:
  PyTorchStreamReader(std::istream* in) : in_(in) {
    // Store file size so we know when we're done reading because the f* APIs
    // don't do a good job of that
    in_->seekg(0L, in_->end);
    file_size_ = in_->tellg();
    readAndValidateFileFooter();
    // Do this now since we're reasonably sure this is actually a PyT file from
    // the header.
    if (file_size_ % kFieldAlignment != 0) {
      throw std::runtime_error(
          "File length is not a multiple of the alignment"
          " size. Is this a valid PyTorch model file?");
    }
    readAndValidateFileHeader();
  }

  std::tuple<at::DataPtr, size_t> getLastRecord() {
    return getRecordWithKey(last_record_offset_);
  }

  std::tuple<at::DataPtr, size_t> getRecordWithKey(uint64_t key) {
    if (key + kFieldAlignment > file_size_) {
      throw std::runtime_error("Provided key is larger than the size of the file.");
    }
    if (key % kFieldAlignment != 0) {
      throw std::runtime_error("Provided key is not divisible by the alignment size.");
    }
    // Seek to the provided offset
    cursor_ = key;
    in_->seekg(cursor_);
    auto tag = read64BitIntegerLittleEndian();
    if (tag != RecordTags::STORAGE) {
      throw std::runtime_error("Attempted to read a record of non-storage type");
    }
    auto size = read64BitIntegerLittleEndian();
    seekToNextAlignmentBoundary();
    auto* ptr = malloc(size);
    at::DataPtr retval(ptr, ptr, free, at::kCPU);

    in_->read(static_cast<char*>(ptr), size);
    cursor_ += size;
    seekToNextAlignmentBoundary();
    return std::tuple<at::DataPtr, size_t>(std::move(retval), size);
  }

  // return dataptr, key, size
  std::tuple<at::DataPtr, int64_t, int64_t> getNextRecord() {
    int64_t key = cursor_;
    if (!hasNextRecord()) {
      throw std::runtime_error("No more record, but hasNextRecord is called.");
    }
    auto tag = read64BitIntegerLittleEndian();
    if (tag != RecordTags::STORAGE) {
      throw std::runtime_error(
          "Attempted to read a record of non-storage type");
    }
    auto size = read64BitIntegerLittleEndian();
    seekToNextAlignmentBoundary();
    auto* ptr = malloc(size);
    at::DataPtr retval(ptr, ptr, free, at::kCPU);

    in_->read(static_cast<char*>(ptr), size);
    cursor_ += size;
    seekToNextAlignmentBoundary();
    return std::tuple<at::DataPtr, int64_t, int64_t>(
        std::move(retval), key, size);
  }

  bool hasNextRecord() const {
    return cursor_ + kFieldAlignment * 2 < file_size_;
  }

  bool close() {
    if (closed_) {
      return false;
    }
    closed_ = true;
    return true;
  }

  bool closed() const {
    return closed_;
  }

  ~PyTorchStreamReader() {
  }

 private:
  std::istream* in_;
  size_t cursor_ = 0;
  size_t file_size_;
  size_t last_record_offset_;
  bool closed_ = false;

  // Utility functions
  uint64_t read64BitIntegerLittleEndian() {
    uint64_t retval;
    // TODO endian swap on platforms that need it?
    in_->read(reinterpret_cast<char*>(&retval), 8);
    std::streamsize read_bytes = in_->gcount();
    if (read_bytes != 8) {
      std::ostringstream errmsg;
      errmsg << "Expected to read 8 bytes but got " << read_bytes;
      throw std::runtime_error(errmsg.str());
    }
    cursor_ += read_bytes;
    return retval;
  }

  void seekToNextAlignmentBoundary() {
    size_t next_offset =
        (cursor_ + kFieldAlignment) - (cursor_ % kFieldAlignment);
    size_t pad_amount = next_offset - cursor_;
    cursor_ += pad_amount;
    in_->seekg(cursor_);
  }

  // File format deserialization functions
  bool readAndValidateFileHeader() {
    // Validate magic number
    cursor_ = 0;
    in_->seekg(cursor_);
    uint64_t magic = read64BitIntegerLittleEndian();
    if (magic != kFileMagicNumber) {
      LOG(ERROR) << "Magic number mismatch in PyTorch file. File may"
        << " be corrupted or is not actually a PyTorch file.";
      return false;
    }
    // magic number mismatch in PyTorch file.
    uint64_t file_format_version = read64BitIntegerLittleEndian();
    if (file_format_version > kMaxSupportedFileFormatVersion) {
      LOG(ERROR) << "Attempted to read a PyTorch file with version "
        << file_format_version
        << " but the maximum supported version for reading is "
        << kMaxSupportedFileFormatVersion
        << ". Your PyTorch installation may be too old.";
      return false;
    }
    seekToNextAlignmentBoundary();
    return true;
  }

  bool readAndValidateFileFooter() {
    // Seek to location of file footer. We've already validated that the file
    // length is a multiple of the alignment size
    cursor_ = file_size_ - kFieldAlignment;
    in_->seekg(cursor_);
    auto tag = read64BitIntegerLittleEndian();
    if (tag != RecordTags::FOOTER) {
      LOG(ERROR) << "File footer has wrong record type. Is this"
        << " file corrupted?";
      return false;
    }
    last_record_offset_ = read64BitIntegerLittleEndian();
    if (last_record_offset_ > file_size_) {
      LOG(ERROR) << "Offset of last record is higher than the size"
        << " of the file! Is this file corrupted?";
      return false;
    }
    return true;
  }
};

class PyTorchStreamWriter final {
 public:
  PyTorchStreamWriter(std::ostream* out) : out_(out) {
    writeFileHeader();
    // In the case that we do not write any records into this file, the last
    // record index written into the footer will point to the footer itself.
    last_record_idx_ = cursor_;
  }

  uint64_t writeRecord(const void* data, size_t size) {
    CAFFE_ENFORCE(!finalized_, "should not be finalized!");
    uint64_t record_offset = cursor_;
    last_record_idx_ = record_offset;
    write64BitIntegerLittleEndian(RecordTags::STORAGE);
    write64BitIntegerLittleEndian(size);
    padToNextAlignmentBoundary();
    writeBuffer(data, size);
    padToNextAlignmentBoundary();
    return record_offset;
  }

  void writeEndOfFile() {
    CAFFE_ENFORCE(!finalized_, "cannot finalize again!");
    writeFileFooter();
    finalized_ = true;
  }

  int64_t getCurrentSize() const {
    return static_cast<int64_t>(cursor_);
  }

  bool finalized() const {
    return finalized_;
  }

  ~PyTorchStreamWriter() {
    if (!finalized_) {
      writeEndOfFile();
    }
  }

 private:
  std::ostream* out_;
  size_t cursor_ = 0;
  bool finalized_ = false;
  size_t last_record_idx_ = 0;
  // TODO: move this to the .cc file
  std::vector<char> pad_buffer_{
    /*count=*/kFieldAlignment, /*value=*/kPadValue
  };

  // Utility functions
  void write64BitIntegerLittleEndian(const uint64_t value) {
    // TODO endian swap on platforms that need it?
    out_->write(reinterpret_cast<const char*>(&value), 8);
    cursor_ += 8u;
  }

  void writePad(const size_t num_bytes) {
    out_->write(pad_buffer_.data(), num_bytes);
    cursor_ += num_bytes;
  }

  void padToNextAlignmentBoundary() {
    size_t next_offset =
        (cursor_ + kFieldAlignment) - (cursor_ % kFieldAlignment);
    size_t pad_amount = next_offset - cursor_;
    writePad(pad_amount);
  }

  void writeBuffer(const void* data, size_t size) {
    out_->write(static_cast<const char*>(data), size);
    cursor_ += size;
  }

  // File format write functions
  void writeFileHeader() {
    write64BitIntegerLittleEndian(kFileMagicNumber);
    write64BitIntegerLittleEndian(kFileFormatVersion);
    padToNextAlignmentBoundary();
  }

  void writeFileFooter() {
    write64BitIntegerLittleEndian(RecordTags::FOOTER);
    write64BitIntegerLittleEndian(last_record_idx_);
    padToNextAlignmentBoundary();
  }
};

class PyTorchFileReader final {
 public:
  PyTorchFileReader(const std::string& filename)
      : in_(filename, std::ios_base::binary), stream_reader_(&in_) {}

  bool hasNextRecord() const {
    return stream_reader_.hasNextRecord();
  }

  // return dataptr, key, size
  std::tuple<at::DataPtr, int64_t, int64_t> getNextRecord() {
    return stream_reader_.getNextRecord();
  }

  std::tuple<at::DataPtr, size_t> getLastRecord() {
    return stream_reader_.getLastRecord();
  }

  std::tuple<at::DataPtr, size_t> getRecordWithKey(uint64_t key) {
    return stream_reader_.getRecordWithKey(key);
  }

  bool close() {
    return stream_reader_.close();
  }

  bool closed() const {
    return stream_reader_.closed();
  }

 private:
  std::ifstream in_;
  PyTorchStreamReader stream_reader_;
};

class PyTorchFileWriter final {
 public:
  PyTorchFileWriter(const std::string& filename)
      : out_(filename, std::ios_base::binary), stream_writer_(&out_) {}

  uint64_t writeRecord(const void* data, size_t size) {
    CAFFE_ENFORCE(
        !stream_writer_.finalized(),
        "cannot write to a finalized stream writer.");
    return stream_writer_.writeRecord(data, size);
  }

  void writeEndOfFile() {
    CAFFE_ENFORCE(
        !stream_writer_.finalized(),
        "cannot write end to a finalized stream writer.");
    stream_writer_.writeEndOfFile();
    out_.close();
  }

  int64_t getCurrentSize() const {
    return stream_writer_.getCurrentSize();
  }

  bool closed() const {
    return stream_writer_.finalized();
  }

 private:
  std::ofstream out_;
  PyTorchStreamWriter stream_writer_;
};

}}  // namespace torch::jit
