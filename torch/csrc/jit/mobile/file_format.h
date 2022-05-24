#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <istream>

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

namespace internal {

/// The size of the buffer to pass to #getFileFormat(), in bytes.
constexpr size_t kFileFormatHeaderSize = 8;

/**
 * Returns the likely file format based on the magic header bytes in @p header,
 * which should contain the first bytes of a file or data stream.
 */
// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline FileFormat getFileFormat(
    const std::array<char, kFileFormatHeaderSize>& header) {
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
          header.data() + kFlatbufferMagicOffset,
          kFlatbufferMagicString.data(),
          kMagicSize) == 0) {
    // Magic header for a binary file containing a Flatbuffer-serialized mobile
    // Module.
    return FileFormat::FlatbufferFileFormat;
  } else if (
      std::memcmp(header.data(), kZipMagicString.data(), kMagicSize) == 0) {
    // Magic header for a zip file, which we use to store pickled sub-files.
    return FileFormat::ZipFileFormat;
  }
  return FileFormat::UnknownFileFormat;
}

} // namespace internal

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
  std::array<char, internal::kFileFormatHeaderSize> header;
  data.read(header.data(), header.size());
  if (data.good()) {
    format = internal::getFileFormat(header);
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

} // namespace jit
} // namespace torch
