#pragma once

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>

#include <c10/core/Allocator.h>
#include <c10/core/Backend.h>

#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"

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

constexpr uint64_t kMinSupportedFileFormatVersion = 0x1L;
constexpr uint64_t kMaxSupportedFileFormatVersion = 0x5L;

// Versions (i.e. why was the version number bumped?)

// Note [Dynamic Versions and torch.jit.save vs. torch.save]
//
// Our versioning scheme has a "produced file format version" which
// describes how an archive is to be read. The version written in an archive
// is at least this current produced file format version, but may be greater
// if it includes certain symbols. We refer to these conditional versions
// as "dynamic," since they are identified at runtime.
//
// Dynamic versioning is useful when an operator's semantics are updated.
// When using torch.jit.save we want those semantics to be preserved. If
// we bumped the produced file format version on every change, however,
// then older versions of PyTorch couldn't read even simple archives, like
// a single tensor, from newer versions of PyTorch. Instead, we
// assign dynamic versions to these changes that override the
// produced file format version as needed. That is, when the semantics
// of torch.div changed it was assigned dynamic version 4, and when
// torch.jit.saving modules that use torch.div those archives also have
// (at least) version 4. This prevents earlier versions of PyTorch
// from accidentally performing the wrong kind of division. Modules
// that don't use torch.div or other operators with dynamic versions
// can write the produced file format version, and these programs will
// run as expected on earlier versions of PyTorch.
//
// While torch.jit.save attempts to preserve operator semantics,
// torch.save does not. torch.save is analogous to pickling Python, so
// a function that uses torch.div will have different behavior if torch.saved
// and torch.loaded across PyTorch versions. From a technical perspective,
// torch.save ignores dynamic versioning.

// 1. Initial version
// 2. Removed op_version_set version numbers
// 3. Added type tags to pickle serialization of container types
// 4. (Dynamic) Stopped integer division using torch.div
//      (a versioned symbol preserves the historic behavior of versions 1--3)
// 5. (Dynamic) Stops torch.full inferring a floating point dtype
//      when given bool or integer fill values.
constexpr uint64_t kProducedFileFormatVersion = 0x3L;

// the version we write when the archive contains bytecode.
// It must be higher or eq to kProducedFileFormatVersion.
// Because torchscript changes is likely introduce bytecode change.
// If kProducedFileFormatVersion is increased, kProducedBytecodeVersion
// should be increased too. The relationship is:
// kMaxSupportedFileFormatVersion >= (most likely ==) kProducedBytecodeVersion
//   >= kProducedFileFormatVersion
constexpr uint64_t kProducedBytecodeVersion = 0x4L;

static_assert(kProducedBytecodeVersion >= kProducedFileFormatVersion,
    "kProducedBytecodeVersion must be higher or equal to kProducedFileFormatVersion.");

// Introduce kMinSupportedBytecodeVersion for limited backward compatibility
// support of bytecode. If
// kMinSupportedBytecodeVersion <= model_version <= kProducedBytecodeVersion (in loader),
// we should support this model_version. For example, we provide a wrapper to
// handle an updated operator.
constexpr uint64_t kMinSupportedBytecodeVersion = 0x3L;

class CAFFE2_API PyTorchStreamReader final {
 public:
  explicit PyTorchStreamReader(const std::string& file_name);
  explicit PyTorchStreamReader(std::istream* in);
  explicit PyTorchStreamReader(std::unique_ptr<ReadAdapterInterface> in);

  // return dataptr, size
  std::tuple<at::DataPtr, size_t> getRecord(const std::string& name);
  size_t getRecordOffset(const std::string& name);
  bool hasRecord(const std::string& name);
  std::vector<std::string> getAllRecords();

  ~PyTorchStreamReader();
  uint64_t version() const {
    return version_;
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
  std::unique_ptr<ReadAdapterInterface> in_;
  int64_t version_;
};

class CAFFE2_API PyTorchStreamWriter final {
 public:
  explicit PyTorchStreamWriter(std::string archive_name);
  explicit PyTorchStreamWriter(
      const std::function<size_t(const void*, size_t)>& writer_func);

  void setMinVersion(const uint64_t version);

  void writeRecord(
      const std::string& name,
      const void* data,
      size_t size,
      bool compress = false);
  void writeEndOfFile();

  bool finalized() const {
    return finalized_;
  }

  const std::string& archiveName() {
    return archive_name_;
  }

  ~PyTorchStreamWriter();

 private:
  void setup(const std::string& file_name);
  void valid(const char* what, const char* info = "");
  size_t current_pos_ = 0;
  std::unique_ptr<mz_zip_archive> ar_;
  std::string archive_name_;
  std::string archive_name_plus_slash_;
  std::string padding_;
  std::ofstream file_stream_;
  std::function<size_t(const void*, size_t)> writer_func_;
  uint64_t version_ = kProducedFileFormatVersion;
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
}

} // namespace serialize
} // namespace caffe2
