#include <torch/csrc/jit/mobile/file_format.h>

#include <gtest/gtest.h>

#include <sstream>

// Tests go in torch::jit
namespace torch {
namespace jit {

TEST(FileFormatTest, IdentifiesFlatbufferStream) {
  // Create data whose initial bytes look like a Flatbuffer stream.
  std::stringstream data;
  data << "abcd" // First four bytes don't matter.
       << "PTMF" // Magic string.
       << "efgh"; // Trailing bytes don't matter.

  // The data should be identified as Flatbuffer.
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);
}

TEST(FileFormatTest, IdentifiesZipStream) {
  // Create data whose initial bytes look like a ZIP stream.
  std::stringstream data;
  data << "PK\x03\x04" // Magic string.
       << "abcd" // Trailing bytes don't matter.
       << "efgh";

  // The data should be identified as ZIP.
  EXPECT_EQ(getFileFormat(data), FileFormat::ZipFileFormat);
}

TEST(FileFormatTest, FlatbufferTakesPrecedence) {
  // Since the Flatbuffer and ZIP magic bytes are at different offsets,
  // the same data could be identified as both. Demonstrate that Flatbuffer
  // takes precedence. (See details in file_format.h)
  std::stringstream data;
  data << "PK\x03\x04" // ZIP magic string.
       << "PTMF" // Flatbuffer magic string.
       << "abcd"; // Trailing bytes don't matter.

  // The data should be identified as Flatbuffer.
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);
}

TEST(FileFormatTest, HandlesUnknownStream) {
  // Create data that doesn't look like any known format.
  std::stringstream data;
  data << "abcd"
       << "efgh"
       << "ijkl";

  // The data should be classified as unknown.
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

TEST(FileFormatTest, ShortStreamIsUnknown) {
  // Create data with fewer than kFileFormatHeaderSize (8) bytes.
  std::stringstream data;
  data << "ABCD";

  // The data should be classified as unknown.
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

TEST(FileFormatTest, EmptyStreamIsUnknown) {
  // Create an empty stream.
  std::stringstream data;

  // The data should be classified as unknown.
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

TEST(FileFormatTest, BadStreamIsUnknown) {
  // Create a stream with valid Flatbuffer data.
  std::stringstream data;
  data << "abcd"
       << "PTMF" // Flatbuffer magic string.
       << "efgh";

  // Demonstrate that the data would normally be identified as Flatbuffer.
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);

  // Mark the stream as bad, and demonstrate that it is in an error state.
  data.setstate(std::stringstream::badbit);
  // Demonstrate that the stream is in an error state.
  EXPECT_FALSE(data.good());

  // The data should now be classified as unknown.
  EXPECT_EQ(getFileFormat(data), FileFormat::UnknownFileFormat);
}

TEST(FileFormatTest, StreamOffsetIsObservedAndRestored) {
  // Create data with a Flatbuffer header at a non-zero offset into the stream.
  std::stringstream data;
  // Add initial padding.
  data << "PADDING";
  size_t offset = data.str().size();
  // Add a valid Flatbuffer header.
  data << "abcd"
       << "PTMF" // Flatbuffer magic string.
       << "efgh";
  // Seek just after the padding.
  data.seekg(static_cast<std::stringstream::off_type>(offset), data.beg);
  // Demonstrate that the stream points to the beginning of the Flatbuffer data,
  // not to the padding.
  EXPECT_EQ(data.peek(), 'a');

  // The data should be identified as Flatbuffer.
  EXPECT_EQ(getFileFormat(data), FileFormat::FlatbufferFileFormat);

  // The stream position should be where it was before identification.
  EXPECT_EQ(offset, data.tellg());
}

TEST(FileFormatTest, HandlesMissingFile) {
  // A missing file should be classified as unknown.
  EXPECT_EQ(
      getFileFormat("NON_EXISTENT_FILE_4965c363-44a7-443c-983a-8895eead0277"),
      FileFormat::UnknownFileFormat);
}

} // namespace jit
} // namespace torch
