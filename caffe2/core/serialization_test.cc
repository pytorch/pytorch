#include <gtest/gtest.h>

#include <c10/util/Flags.h>
#include <c10/util/string_view.h>
#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"

// NOLINTNEXTLINE: cppcoreguidelines-avoid-c-arrays
C10_DEFINE_bool(
    caffe2_test_generate_unknown_dtype_blob,
    false,
    "Recompute and log the serialized blob data for the "
    "TensorSerialization.TestUnknownDType test");

using namespace caffe2;

namespace {

// This data was computed by serializing a 10-element int32_t tensor,
// but with the data_type field set to 4567.  This allows us to test the
// behavior of the code when deserializing data from a future version of the
// code that has new data types that our code does not understand.
constexpr c10::string_view kFutureDtypeBlob(
    "\x0a\x09\x74\x65\x73\x74\x5f\x62\x6c\x6f\x62\x12\x06\x54\x65\x6e"
    "\x73\x6f\x72\x1a\x28\x08\x0a\x08\x01\x10\xd7\x23\x22\x0a\x00\x01"
    "\x02\x03\x04\x05\x06\x07\x08\x09\x3a\x09\x74\x65\x73\x74\x5f\x62"
    "\x6c\x6f\x62\x42\x02\x08\x00\x5a\x04\x08\x00\x10\x0a",
    61);
// The same tensor with the data_type actually set to TensorProto_DataType_INT32
constexpr c10::string_view kInt32DtypeBlob(
    "\x0a\x09\x74\x65\x73\x74\x5f\x62\x6c\x6f\x62\x12\x06\x54\x65\x6e"
    "\x73\x6f\x72\x1a\x27\x08\x0a\x08\x01\x10\x02\x22\x0a\x00\x01\x02"
    "\x03\x04\x05\x06\x07\x08\x09\x3a\x09\x74\x65\x73\x74\x5f\x62\x6c"
    "\x6f\x62\x42\x02\x08\x00\x5a\x04\x08\x00\x10\x0a",
    60);

void logBlob(c10::string_view data) {
  constexpr size_t kBytesPerLine = 16;
  constexpr size_t kCharsPerEncodedByte = 4;
  std::vector<char> hexStr;
  hexStr.resize((kBytesPerLine * kCharsPerEncodedByte) + 1);
  hexStr[kBytesPerLine * kCharsPerEncodedByte] = '\0';
  size_t lineIdx = 0;
  for (char c : data) {
    snprintf(
        hexStr.data() + (kCharsPerEncodedByte * lineIdx),
        kCharsPerEncodedByte + 1,
        "\\x%02x",
        static_cast<unsigned int>(c));
    ++lineIdx;
    if (lineIdx >= kBytesPerLine) {
      LOG(INFO) << "    \"" << hexStr.data() << "\"";
      lineIdx = 0;
    }
  }
  if (lineIdx > 0) {
    hexStr[lineIdx * kCharsPerEncodedByte] = '\0';
    LOG(INFO) << "    \"" << hexStr.data() << "\"";
  }
}

} // namespace

TEST(TensorSerialization, TestUnknownDType) {
  // This code was used to generate the blob data listed above.
  constexpr size_t kTestTensorSize = 10;
  if (FLAGS_caffe2_test_generate_unknown_dtype_blob) {
    Blob blob;
    auto* blobTensor = BlobGetMutableTensor(&blob, CPU);
    blobTensor->Resize(kTestTensorSize, 1);
    auto *tensorData = blobTensor->mutable_data<int32_t>();
    for (int n = 0; n < kTestTensorSize; ++n) {
      tensorData[n] = n;
    }
    auto data = SerializeBlob(blob, "test_blob");
    LOG(INFO) << "test blob: size=" << data.size();
    logBlob(data);
  }

  // Test deserializing the normal INT32 data,
  // just to santity check that deserialization works
  Blob i32Blob;
  DeserializeBlob(std::string(kInt32DtypeBlob), &i32Blob);
  const auto& tensor = BlobGetTensor(i32Blob, c10::DeviceType::CPU);
  EXPECT_EQ(kTestTensorSize, tensor.numel());
  EXPECT_EQ(TypeMeta::Make<int32_t>(), tensor.dtype());
  const auto* tensor_data = tensor.template data<int32_t>();
  for (int i = 0; i < kTestTensorSize; ++i) {
    EXPECT_EQ(static_cast<float>(i), tensor_data[i]);
  }

  // Now test deserializing our blob with an unknown data type
  Blob futureDtypeBlob;
  try {
    DeserializeBlob(std::string(kFutureDtypeBlob), &futureDtypeBlob);
    FAIL() << "DeserializeBlob() should have failed";
  } catch (const std::exception& ex) {
    EXPECT_STREQ(
        "Cannot deserialize tensor: unrecognized data type", ex.what());
  }
}
