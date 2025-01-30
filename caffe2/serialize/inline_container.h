#pragma once

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <mutex>
#include <ostream>
#include <unordered_set>

#include <c10/core/Allocator.h>
#include <c10/core/Backend.h>

#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"
#include "caffe2/serialize/versions.h"

extern "C" {
typedef struct mz_zip_archive mz_zip_archive;
}

// PyTorch containers are a special zip archive with the following layout
// archive_name.zip contains:
//    archive_name/
//        version # a file with a single decimal number written in ascii,
//                # used to establish the version of the archive format
//        model.json # overall model description, this is a json output of
//                   # ModelDef from torch.proto
//        # the following names are by convention only, model.json will
//        # refer to these files by full names
//        tensors/
//          0 # flat storage for tensor data, meta-data about shapes, etc. is
//            # in model.json
//          1
//          ...
//        # code entries will only exist for modules that have methods attached
//        code/
//          archive_name.py # serialized torch script code (python syntax, using
//          PythonPrint) archive_name_my_submodule.py # submodules have separate
//          files
//
// The PyTorchStreamWriter also ensures additional useful properties for these
// files
// 1. All files are stored uncompressed.
// 2. All files in the archive are aligned to 64 byte boundaries such that
//    it is possible to mmap the entire file and get an aligned pointer to
//    tensor data.
// 3. We universally write in ZIP64 format for consistency.

// The PyTorchStreamReader also provides additional properties:
// 1. It can read zip files that are created with common
//    zip tools. This means that even though our writer doesn't compress files,
//    the reader can still read files that were compressed.
// 2. It provides a getRecordOffset function which returns the offset into the
//    raw file where file data lives. If the file was written with
//    PyTorchStreamWriter it is guaranteed to be 64 byte aligned.

// PyTorchReader/Writer handle checking the version number on the archive format
// and ensure that all files are written to a archive_name directory so they
// unzip cleanly.

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

// The model.json, which contains all the metadata information,
// should be written as the last file. One reason is that the size of tensor
// data is usually stable. As long as the shape and type of the tensor do not
// change, the size of the data won't change. On the other sied, the size of the
// serialized model is likely to change, so we store it as the last record, and
// we don't need to move previous records when updating the model data.

// The zip format is sufficiently flexible to handle the above use-case.
// it puts its central directory at the end of the archive and we write
// model.json as the last file when writing after we have accumulated all
// other information.

namespace caffe2 {
namespace serialize {

static constexpr const char* kSerializationIdRecordName =
    ".data/serialization_id";

struct MzZipReaderIterWrapper;

class TORCH_API ChunkRecordIterator {
 public:
  ~ChunkRecordIterator();

  // Read at most `chunkSize` into `buf`. Return the number of actual bytes
  // read.
  size_t next(void* buf);
  size_t recordSize() const {
    return recordSize_;
  }

 private:
  ChunkRecordIterator(
      size_t recordSize,
      size_t chunkSize,
      std::unique_ptr<MzZipReaderIterWrapper> iter);

  const size_t recordSize_;
  const size_t chunkSize_;
  size_t offset_;
  std::unique_ptr<MzZipReaderIterWrapper> iter_;

  friend class PyTorchStreamReader;
};

class TORCH_API PyTorchStreamReader final {
 public:
  explicit PyTorchStreamReader(const std::string& file_name);
  explicit PyTorchStreamReader(std::istream* in);
  explicit PyTorchStreamReader(std::shared_ptr<ReadAdapterInterface> in);

  // return dataptr, size
  std::tuple<at::DataPtr, size_t> getRecord(const std::string& name);
  // multi-thread getRecord
  std::tuple<at::DataPtr, size_t> getRecord(
      const std::string& name,
      std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders);
  // inplace memory writing
  size_t getRecord(const std::string& name, void* dst, size_t n);
  // inplace memory writing, multi-threads.
  // When additionalReaders is empty, the default behavior is call
  // getRecord(name, dst, n) with default reader This approach can be used for
  // reading large tensors.
  size_t getRecord(
      const std::string& name,
      void* dst,
      size_t n,
      std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders);
  size_t getRecord(
      const std::string& name,
      void* dst,
      size_t n,
      size_t chunk_size,
      void* buf,
      const std::function<void(void*, const void*, size_t)>& memcpy_func =
          nullptr);

  // Concurrent reading records with multiple readers.
  // additionalReaders are additional clients to access the underlying record at
  // different offsets and write to different trunks of buffers. If the overall
  // size of the tensor is 10, and size of additionalReader is 2. The default
  // thread will read [0,4), the additional reader will read [4,8). The default
  // reader will read [8,10). The default reader will write to buffer[0,4), the
  // additional reader will write to buffer[4,8), the additional reader will
  // write to buffer[8,10). When additionalReaders is empty, the default
  // behavior is call getRecord(name) with default reader This approach can be
  // used for reading large tensors.
  size_t getRecordMultiReaders(
      const std::string& name,
      std::vector<std::shared_ptr<ReadAdapterInterface>>& additionalReaders,
      void* dst,
      size_t n);

  size_t getRecordSize(const std::string& name);

  size_t getRecordOffset(const std::string& name);
  bool hasRecord(const std::string& name);
  std::vector<std::string> getAllRecords();

  ChunkRecordIterator createChunkReaderIter(
      const std::string& name,
      const size_t recordSize,
      const size_t chunkSize);

  ~PyTorchStreamReader();
  uint64_t version() const {
    return version_;
  }
  const std::string& serializationId() {
    return serialization_id_;
  }

  void setShouldLoadDebugSymbol(bool should_load_debug_symbol) {
    load_debug_symbol_ = should_load_debug_symbol;
  }
  void setAdditionalReaderSizeThreshold(const size_t& size) {
    additional_reader_size_threshold_ = size;
  }

 private:
  void init();
  size_t read(uint64_t pos, char* buf, size_t n);
  void valid(const char* what, const char* info = "");
  size_t getRecordID(const std::string& name);

  friend size_t
  istream_read_func(void* pOpaque, uint64_t file_ofs, void* pBuf, size_t n);
  std::unique_ptr<mz_zip_archive> ar_;
  std::string archive_name_;
  std::string archive_name_plus_slash_;
  std::shared_ptr<ReadAdapterInterface> in_;
  int64_t version_;
  std::mutex reader_lock_;
  bool load_debug_symbol_ = true;
  std::string serialization_id_;
  size_t additional_reader_size_threshold_;
};

class TORCH_API PyTorchStreamWriter final {
 public:
  explicit PyTorchStreamWriter(
      const std::string& archive_name,
      bool compute_crc32 = true);
  explicit PyTorchStreamWriter(
      const std::function<size_t(const void*, size_t)> writer_func,
      bool compute_crc32 = true);

  void setMinVersion(const uint64_t version);

  void writeRecord(
      const std::string& name,
      const void* data,
      size_t size,
      bool compress = false);
  void writeEndOfFile();

  const std::unordered_set<std::string>& getAllWrittenRecords();

  bool finalized() const {
    return finalized_;
  }

  const std::string& archiveName() {
    return archive_name_;
  }

  const std::string& serializationId() {
    return serialization_id_;
  }

  ~PyTorchStreamWriter();

 private:
  void setup(const std::string& file_name);
  void valid(const char* what, const char* info = "");
  void writeSerializationId();
  size_t current_pos_ = 0;
  std::unordered_set<std::string> files_written_;
  std::unique_ptr<mz_zip_archive> ar_;
  std::string archive_name_;
  std::string archive_name_plus_slash_;
  std::string padding_;
  std::ofstream file_stream_;
  std::function<size_t(const void*, size_t)> writer_func_;
  uint64_t combined_uncomp_crc32_ = 0;
  std::string serialization_id_;
  bool compute_crc32_;

  // This number will be updated when the model has operators
  // that have valid upgraders.
  uint64_t version_ = kMinProducedFileFormatVersion;
  bool finalized_ = false;
  bool err_seen_ = false;
  friend size_t ostream_write_func(
      void* pOpaque,
      uint64_t file_ofs,
      const void* pBuf,
      size_t n);
};

namespace detail {
// Writer-specific constants
constexpr uint64_t kFieldAlignment = 64;

// Returns a record to be appended to the local user extra data entry in order
// to make data beginning aligned at kFieldAlignment bytes boundary.
size_t getPadding(
    size_t cursor,
    size_t filename_size,
    size_t size,
    std::string& padding_buf);
} // namespace detail

} // namespace serialize
} // namespace caffe2
