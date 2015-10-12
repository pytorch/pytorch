// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <google/protobuf/util/json_util.h>

#include <list>
#include <string>

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/util/json_format_proto3.pb.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/util/type_resolver_util.h>
#include <gtest/gtest.h>

namespace google {
namespace protobuf {
namespace util {
namespace {

using proto3::FOO;
using proto3::BAR;
using proto3::TestMessage;
using proto3::TestMap;

static const char kTypeUrlPrefix[] = "type.googleapis.com";

static string GetTypeUrl(const Descriptor* message) {
  return string(kTypeUrlPrefix) + "/" + message->full_name();
}

// As functions defined in json_util.h are just thin wrappers around the
// JSON conversion code in //net/proto2/util/converter, in this test we
// only cover some very basic cases to make sure the wrappers have forwarded
// parameters to the underlying implementation correctly. More detailed
// tests are contained in the //net/proto2/util/converter directory.
class JsonUtilTest : public testing::Test {
 protected:
  JsonUtilTest() {
    resolver_.reset(NewTypeResolverForDescriptorPool(
        kTypeUrlPrefix, DescriptorPool::generated_pool()));
  }

  string ToJson(const Message& message, const JsonOptions& options) {
    string result;
    GOOGLE_CHECK_OK(BinaryToJsonString(resolver_.get(),
                                GetTypeUrl(message.GetDescriptor()),
                                message.SerializeAsString(), &result, options));
    return result;
  }

  bool FromJson(const string& json, Message* message) {
    string binary;
    GOOGLE_CHECK_OK(JsonToBinaryString(
        resolver_.get(), GetTypeUrl(message->GetDescriptor()), json, &binary));
    return message->ParseFromString(binary);
  }

  google::protobuf::scoped_ptr<TypeResolver> resolver_;
};

TEST_F(JsonUtilTest, TestWhitespaces) {
  TestMessage m;
  m.mutable_message_value();

  JsonOptions options;
  EXPECT_EQ("{\"messageValue\":{}}", ToJson(m, options));
  options.add_whitespace = true;
  EXPECT_EQ(
      "{\n"
      " \"messageValue\": {}\n"
      "}\n",
      ToJson(m, options));
}

// TODO(skarvaje): Uncomment after cl/96232915 is submitted.
// TEST_F(JsonUtilTest, TestDefaultValues) {
  // TestMessage m;
  // JsonOptions options;
  // EXPECT_EQ("{}", ToJson(m, options));
  // options.always_print_primitive_fields = true;
  // EXPECT_EQ(
      // "{\"boolValue\":false,"
      // "\"int32Value\":0,"
      // "\"int64Value\":\"0\","
      // "\"uint32Value\":0,"
      // "\"uint64Value\":\"0\","
      // "\"floatValue\":0,"
      // "\"doubleValue\":0,"
      // "\"stringValue\":\"\","
      // "\"bytesValue\":\"\","
      // // TODO(xiaofeng): The default enum value should be FOO. I believe
      // // this is a bug in DefaultValueObjectWriter.
      // "\"enumValue\":null"
      // "}",
      // ToJson(m, options));
// }

TEST_F(JsonUtilTest, ParseMessage) {
  // Some random message but good enough to verify that the parsing warpper
  // functions are working properly.
  string input =
      "{\n"
      "  \"int32Value\": 1024,\n"
      "  \"repeatedInt32Value\": [1, 2],\n"
      "  \"messageValue\": {\n"
      "    \"value\": 2048\n"
      "  },\n"
      "  \"repeatedMessageValue\": [\n"
      "    {\"value\": 40}, {\"value\": 96}\n"
      "  ]\n"
      "}\n";
  TestMessage m;
  ASSERT_TRUE(FromJson(input, &m));
  EXPECT_EQ(1024, m.int32_value());
  ASSERT_EQ(2, m.repeated_int32_value_size());
  EXPECT_EQ(1, m.repeated_int32_value(0));
  EXPECT_EQ(2, m.repeated_int32_value(1));
  EXPECT_EQ(2048, m.message_value().value());
  ASSERT_EQ(2, m.repeated_message_value_size());
  EXPECT_EQ(40, m.repeated_message_value(0).value());
  EXPECT_EQ(96, m.repeated_message_value(1).value());
}

TEST_F(JsonUtilTest, ParseMap) {
  TestMap message;
  (*message.mutable_string_map())["hello"] = 1234;
  JsonOptions options;
  EXPECT_EQ("{\"stringMap\":{\"hello\":1234}}", ToJson(message, options));
  TestMap other;
  ASSERT_TRUE(FromJson(ToJson(message, options), &other));
  EXPECT_EQ(message.DebugString(), other.DebugString());
}

typedef pair<char*, int> Segment;
// A ZeroCopyOutputStream that writes to multiple buffers.
class SegmentedZeroCopyOutputStream : public io::ZeroCopyOutputStream {
 public:
  explicit SegmentedZeroCopyOutputStream(list<Segment> segments)
      : segments_(segments), last_segment_(static_cast<char*>(NULL), 0), byte_count_(0) {}

  virtual bool Next(void** buffer, int* length) {
    if (segments_.empty()) {
      return false;
    }
    last_segment_ = segments_.front();
    segments_.pop_front();
    *buffer = last_segment_.first;
    *length = last_segment_.second;
    byte_count_ += *length;
    return true;
  }

  virtual void BackUp(int length) {
    GOOGLE_CHECK(length <= last_segment_.second);
    segments_.push_front(
        Segment(last_segment_.first + last_segment_.second - length, length));
    last_segment_ = Segment(last_segment_.first, last_segment_.second - length);
    byte_count_ -= length;
  }

  virtual int64 ByteCount() const { return byte_count_; }

 private:
  list<Segment> segments_;
  Segment last_segment_;
  int64 byte_count_;
};

// This test splits the output buffer and also the input data into multiple
// segments and checks that the implementation of ZeroCopyStreamByteSink
// handles all possible cases correctly.
TEST(ZeroCopyStreamByteSinkTest, TestAllInputOutputPatterns) {
  static const int kOutputBufferLength = 10;
  // An exhaustive test takes too long, skip some combinations to make the test
  // run faster.
  static const int kSkippedPatternCount = 7;

  char buffer[kOutputBufferLength];
  for (int split_pattern = 0; split_pattern < (1 << (kOutputBufferLength - 1));
       split_pattern += kSkippedPatternCount) {
    // Split the buffer into small segments according to the split_pattern.
    list<Segment> segments;
    int segment_start = 0;
    for (int i = 0; i < kOutputBufferLength - 1; ++i) {
      if (split_pattern & (1 << i)) {
        segments.push_back(
            Segment(buffer + segment_start, i - segment_start + 1));
        segment_start = i + 1;
      }
    }
    segments.push_back(
        Segment(buffer + segment_start, kOutputBufferLength - segment_start));

    // Write exactly 10 bytes through the ByteSink.
    string input_data = "0123456789";
    for (int input_pattern = 0; input_pattern < (1 << (input_data.size() - 1));
         input_pattern += kSkippedPatternCount) {
      memset(buffer, 0, sizeof(buffer));
      {
        SegmentedZeroCopyOutputStream output_stream(segments);
        internal::ZeroCopyStreamByteSink byte_sink(&output_stream);
        int start = 0;
        for (int j = 0; j < input_data.length() - 1; ++j) {
          if (input_pattern & (1 << j)) {
            byte_sink.Append(&input_data[start], j - start + 1);
            start = j + 1;
          }
        }
        byte_sink.Append(&input_data[start], input_data.length() - start);
      }
      EXPECT_EQ(input_data, string(buffer, input_data.length()));
    }

    // Write only 9 bytes through the ByteSink.
    input_data = "012345678";
    for (int input_pattern = 0; input_pattern < (1 << (input_data.size() - 1));
         input_pattern += kSkippedPatternCount) {
      memset(buffer, 0, sizeof(buffer));
      {
        SegmentedZeroCopyOutputStream output_stream(segments);
        internal::ZeroCopyStreamByteSink byte_sink(&output_stream);
        int start = 0;
        for (int j = 0; j < input_data.length() - 1; ++j) {
          if (input_pattern & (1 << j)) {
            byte_sink.Append(&input_data[start], j - start + 1);
            start = j + 1;
          }
        }
        byte_sink.Append(&input_data[start], input_data.length() - start);
      }
      EXPECT_EQ(input_data, string(buffer, input_data.length()));
      EXPECT_EQ(0, buffer[input_data.length()]);
    }

    // Write 11 bytes through the ByteSink. The extra byte will just
    // be ignored.
    input_data = "0123456789A";
    for (int input_pattern = 0; input_pattern < (1 << (input_data.size() - 1));
         input_pattern += kSkippedPatternCount) {
      memset(buffer, 0, sizeof(buffer));
      {
        SegmentedZeroCopyOutputStream output_stream(segments);
        internal::ZeroCopyStreamByteSink byte_sink(&output_stream);
        int start = 0;
        for (int j = 0; j < input_data.length() - 1; ++j) {
          if (input_pattern & (1 << j)) {
            byte_sink.Append(&input_data[start], j - start + 1);
            start = j + 1;
          }
        }
        byte_sink.Append(&input_data[start], input_data.length() - start);
      }
      EXPECT_EQ(input_data.substr(0, kOutputBufferLength),
                string(buffer, kOutputBufferLength));
    }
  }
}

}  // namespace
}  // namespace util
}  // namespace protobuf
}  // namespace google
