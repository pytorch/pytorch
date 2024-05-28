#pragma once

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <istream>
#include <memory>

#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <caffe2/serialize/read_adapter_interface.h>

#if defined(HAVE_MMAP)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

/**
 * @file
 *
 * Helpers for identifying file formats when reading serialized data.
 *
 * Note that these functions are declared inline because they will typically
 * only be called from one or two locations per binary.
 */

namespace torch {
namespace jit {

/**
 * The format of a file or data stream.
 */
enum class FileFormat {
  UnknownFileFormat = 0,
  FlatbufferFileFormat,
  ZipFileFormat,
};

/// The size of the buffer to pass to #getFileFormat(), in bytes.
constexpr size_t kFileFormatHeaderSize = 8;
constexpr size_t kMaxAlignment = 16;

/**
 * Returns the likely file format based on the magic header bytes in @p header,
 * which should contain the first bytes of a file or data stream.
 */
// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline FileFormat getFileFormat(const char* data) {
  // The size of magic strings to look for in the buffer.
  static constexpr size_t kMagicSize = 4;

  // Bytes 4..7 of a Flatbuffer-encoded file produced by
  // `flatbuffer_serializer.h`. (The first four bytes contain an offset to the
  // actual Flatbuffer data.)
  static constexpr std::array<char, kMagicSize> kFlatbufferMagicString = {
      'P', 'T', 'M', 'F'};
  static constexpr size_t kFlatbufferMagicOffset = 4;

  // The first four bytes of a ZIP file.
  static constexpr std::array<char, kMagicSize> kZipMagicString = {
      'P', 'K', '\x03', '\x04'};

  // Note that we check for Flatbuffer magic first. Since the first four bytes
  // of flatbuffer data contain an offset to the root struct, it's theoretically
  // possible to construct a file whose offset looks like the ZIP magic. On the
  // other hand, bytes 4-7 of ZIP files are constrained to a small set of values
  // that do not typically cross into the printable ASCII range, so a ZIP file
  // should never have a header that looks like a Flatbuffer file.
  if (std::memcmp(
          data + kFlatbufferMagicOffset,
          kFlatbufferMagicString.data(),
          kMagicSize) == 0) {
    // Magic header for a binary file containing a Flatbuffer-serialized mobile
    // Module.
    return FileFormat::FlatbufferFileFormat;
  } else if (std::memcmp(data, kZipMagicString.data(), kMagicSize) == 0) {
    // Magic header for a zip file, which we use to store pickled sub-files.
    return FileFormat::ZipFileFormat;
  }
  return FileFormat::UnknownFileFormat;
}

/**
 * Returns the likely file format based on the magic header bytes of @p data.
 * If the stream position changes while inspecting the data, this function will
 * restore the stream position to its original offset before returning.
 */
// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline FileFormat getFileFormat(std::istream& data) {
  FileFormat format = FileFormat::UnknownFileFormat;
  std::streampos orig_pos = data.tellg();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<char, kFileFormatHeaderSize> header;
  data.read(header.data(), header.size());
  if (data.good()) {
    format = getFileFormat(header.data());
  }
  data.seekg(orig_pos, data.beg);
  return format;
}

/**
 * Returns the likely file format based on the magic header bytes of the file
 * named @p filename.
 */
// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline FileFormat getFileFormat(const std::string& filename) {
  std::ifstream data(filename, std::ifstream::binary);
  return getFileFormat(data);
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static void file_not_found_error() {
  std::stringstream message;
  message << "Error while opening file: ";
  if (errno == ENOENT) {
    message << "no such file or directory" << std::endl;
  } else {
    message << "error no is: " << errno << std::endl;
  }
  TORCH_CHECK(false, message.str());
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline std::tuple<std::shared_ptr<char>, size_t> get_file_content(
    const char* filename) {
#if defined(HAVE_MMAP)
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    // failed to open file, chances are it's no such file or directory.
    file_not_found_error();
  }
  struct stat statbuf {};
  fstat(fd, &statbuf);
  size_t size = statbuf.st_size;
  void* ptr = mmap(nullptr, statbuf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  auto deleter = [statbuf](char* ptr) { munmap(ptr, statbuf.st_size); };
  std::shared_ptr<char> data(reinterpret_cast<char*>(ptr), deleter);
#else
  FILE* f = fopen(filename, "rb");
  if (f == nullptr) {
    file_not_found_error();
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  // make sure buffer size is multiple of alignment
  size_t buffer_size = (size / kMaxAlignment + 1) * kMaxAlignment;
  std::shared_ptr<char> data(
      static_cast<char*>(c10::alloc_cpu(buffer_size)), c10::free_cpu);
  fread(data.get(), size, 1, f);
  fclose(f);
#endif
  return std::make_tuple(data, size);
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline std::tuple<std::shared_ptr<char>, size_t> get_stream_content(
    std::istream& in) {
  // get size of the stream and reset to orig
  std::streampos orig_pos = in.tellg();
  in.seekg(orig_pos, std::ios::end);
  const long size = in.tellg();
  in.seekg(orig_pos, in.beg);

  // read stream
  // NOLINT make sure buffer size is multiple of alignment
  size_t buffer_size = (size / kMaxAlignment + 1) * kMaxAlignment;
  std::shared_ptr<char> data(
      static_cast<char*>(c10::alloc_cpu(buffer_size)), c10::free_cpu);
  in.read(data.get(), size);

  // reset stream to original position
  in.seekg(orig_pos, in.beg);
  return std::make_tuple(data, size);
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline std::tuple<std::shared_ptr<char>, size_t> get_rai_content(
    caffe2::serialize::ReadAdapterInterface* rai) {
  size_t buffer_size = (rai->size() / kMaxAlignment + 1) * kMaxAlignment;
  std::shared_ptr<char> data(
      static_cast<char*>(c10::alloc_cpu(buffer_size)), c10::free_cpu);
  rai->read(
      0, data.get(), rai->size(), "Loading ReadAdapterInterface to bytes");
  return std::make_tuple(data, buffer_size);
}

} // namespace jit
} // namespace torch
