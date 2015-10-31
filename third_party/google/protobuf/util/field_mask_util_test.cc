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

#include <google/protobuf/util/field_mask_util.h>

#include <google/protobuf/field_mask.pb.h>
#include <google/protobuf/unittest.pb.h>
#include <google/protobuf/test_util.h>
#include <gtest/gtest.h>

namespace google {
namespace protobuf {
namespace util {
namespace {

using protobuf_unittest::TestAllTypes;
using protobuf_unittest::NestedTestAllTypes;
using google::protobuf::FieldMask;

TEST(FieldMaskUtilTest, StringFormat) {
  FieldMask mask;
  EXPECT_EQ("", FieldMaskUtil::ToString(mask));
  mask.add_paths("foo");
  EXPECT_EQ("foo", FieldMaskUtil::ToString(mask));
  mask.add_paths("bar");
  EXPECT_EQ("foo,bar", FieldMaskUtil::ToString(mask));

  FieldMaskUtil::FromString("", &mask);
  EXPECT_EQ(0, mask.paths_size());
  FieldMaskUtil::FromString("foo", &mask);
  EXPECT_EQ(1, mask.paths_size());
  EXPECT_EQ("foo", mask.paths(0));
  FieldMaskUtil::FromString("foo,bar", &mask);
  EXPECT_EQ(2, mask.paths_size());
  EXPECT_EQ("foo", mask.paths(0));
  EXPECT_EQ("bar", mask.paths(1));
}

TEST(FieldMaskUtilTest, TestIsVaildPath) {
  EXPECT_TRUE(FieldMaskUtil::IsValidPath<TestAllTypes>("optional_int32"));
  EXPECT_FALSE(FieldMaskUtil::IsValidPath<TestAllTypes>("optional_nonexist"));
  EXPECT_TRUE(
      FieldMaskUtil::IsValidPath<TestAllTypes>("optional_nested_message.bb"));
  EXPECT_FALSE(FieldMaskUtil::IsValidPath<TestAllTypes>(
      "optional_nested_message.nonexist"));
  // FieldMask cannot be used to specify sub-fields of a repeated message.
  EXPECT_FALSE(
      FieldMaskUtil::IsValidPath<TestAllTypes>("repeated_nested_message.bb"));
}

TEST(FieldMaskUtilTest, TestIsValidFieldMask) {
  FieldMask mask;
  FieldMaskUtil::FromString("optional_int32,optional_nested_message.bb", &mask);
  EXPECT_TRUE(FieldMaskUtil::IsValidFieldMask<TestAllTypes>(mask));

  FieldMaskUtil::FromString(
      "optional_int32,optional_nested_message.bb,optional_nonexist", &mask);
  EXPECT_FALSE(FieldMaskUtil::IsValidFieldMask<TestAllTypes>(mask));
}

TEST(FieldMaskUtilTest, TestGetFieldMaskForAllFields) {
  FieldMask mask;
  FieldMaskUtil::GetFieldMaskForAllFields<TestAllTypes::NestedMessage>(&mask);
  EXPECT_EQ(1, mask.paths_size());
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("bb", mask));

  FieldMaskUtil::GetFieldMaskForAllFields<TestAllTypes>(&mask);
  EXPECT_EQ(76, mask.paths_size());
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_int32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_int64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_uint32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_uint64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_sint32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_sint64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_fixed32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_fixed64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_sfixed32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_sfixed64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_float", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_double", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_bool", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_string", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_bytes", mask));
  EXPECT_TRUE(
      FieldMaskUtil::IsPathInFieldMask("optional_nested_message", mask));
  EXPECT_TRUE(
      FieldMaskUtil::IsPathInFieldMask("optional_foreign_message", mask));
  EXPECT_TRUE(
      FieldMaskUtil::IsPathInFieldMask("optional_import_message", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_nested_enum", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_foreign_enum", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("optional_import_enum", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_int32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_int64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_uint32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_uint64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_sint32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_sint64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_fixed32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_fixed64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_sfixed32", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_sfixed64", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_float", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_double", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_bool", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_string", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_bytes", mask));
  EXPECT_TRUE(
      FieldMaskUtil::IsPathInFieldMask("repeated_nested_message", mask));
  EXPECT_TRUE(
      FieldMaskUtil::IsPathInFieldMask("repeated_foreign_message", mask));
  EXPECT_TRUE(
      FieldMaskUtil::IsPathInFieldMask("repeated_import_message", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_nested_enum", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_foreign_enum", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("repeated_import_enum", mask));
}

TEST(FieldMaskUtilTest, TestToCanonicalForm) {
  FieldMask in, out;
  // Paths will be sorted.
  FieldMaskUtil::FromString("baz.quz,bar,foo", &in);
  FieldMaskUtil::ToCanonicalForm(in, &out);
  EXPECT_EQ("bar,baz.quz,foo", FieldMaskUtil::ToString(out));
  // Duplicated paths will be removed.
  FieldMaskUtil::FromString("foo,bar,foo", &in);
  FieldMaskUtil::ToCanonicalForm(in, &out);
  EXPECT_EQ("bar,foo", FieldMaskUtil::ToString(out));
  // Sub-paths of other paths will be removed.
  FieldMaskUtil::FromString("foo.b1,bar.b1,foo.b2,bar", &in);
  FieldMaskUtil::ToCanonicalForm(in, &out);
  EXPECT_EQ("bar,foo.b1,foo.b2", FieldMaskUtil::ToString(out));

  // Test more deeply nested cases.
  FieldMaskUtil::FromString(
      "foo.bar.baz1,"
      "foo.bar.baz2.quz,"
      "foo.bar.baz2",
      &in);
  FieldMaskUtil::ToCanonicalForm(in, &out);
  EXPECT_EQ("foo.bar.baz1,foo.bar.baz2", FieldMaskUtil::ToString(out));
  FieldMaskUtil::FromString(
      "foo.bar.baz1,"
      "foo.bar.baz2,"
      "foo.bar.baz2.quz",
      &in);
  FieldMaskUtil::ToCanonicalForm(in, &out);
  EXPECT_EQ("foo.bar.baz1,foo.bar.baz2", FieldMaskUtil::ToString(out));
  FieldMaskUtil::FromString(
      "foo.bar.baz1,"
      "foo.bar.baz2,"
      "foo.bar.baz2.quz,"
      "foo.bar",
      &in);
  FieldMaskUtil::ToCanonicalForm(in, &out);
  EXPECT_EQ("foo.bar", FieldMaskUtil::ToString(out));
  FieldMaskUtil::FromString(
      "foo.bar.baz1,"
      "foo.bar.baz2,"
      "foo.bar.baz2.quz,"
      "foo",
      &in);
  FieldMaskUtil::ToCanonicalForm(in, &out);
  EXPECT_EQ("foo", FieldMaskUtil::ToString(out));
}

TEST(FieldMaskUtilTest, TestUnion) {
  FieldMask mask1, mask2, out;
  // Test cases without overlapping.
  FieldMaskUtil::FromString("foo,baz", &mask1);
  FieldMaskUtil::FromString("bar,quz", &mask2);
  FieldMaskUtil::Union(mask1, mask2, &out);
  EXPECT_EQ("bar,baz,foo,quz", FieldMaskUtil::ToString(out));
  // Overlap with duplicated paths.
  FieldMaskUtil::FromString("foo,baz.bb", &mask1);
  FieldMaskUtil::FromString("baz.bb,quz", &mask2);
  FieldMaskUtil::Union(mask1, mask2, &out);
  EXPECT_EQ("baz.bb,foo,quz", FieldMaskUtil::ToString(out));
  // Overlap with paths covering some other paths.
  FieldMaskUtil::FromString("foo.bar.baz,quz", &mask1);
  FieldMaskUtil::FromString("foo.bar,bar", &mask2);
  FieldMaskUtil::Union(mask1, mask2, &out);
  EXPECT_EQ("bar,foo.bar,quz", FieldMaskUtil::ToString(out));
}

TEST(FieldMaskUtilTest, TestIntersect) {
  FieldMask mask1, mask2, out;
  // Test cases without overlapping.
  FieldMaskUtil::FromString("foo,baz", &mask1);
  FieldMaskUtil::FromString("bar,quz", &mask2);
  FieldMaskUtil::Intersect(mask1, mask2, &out);
  EXPECT_EQ("", FieldMaskUtil::ToString(out));
  // Overlap with duplicated paths.
  FieldMaskUtil::FromString("foo,baz.bb", &mask1);
  FieldMaskUtil::FromString("baz.bb,quz", &mask2);
  FieldMaskUtil::Intersect(mask1, mask2, &out);
  EXPECT_EQ("baz.bb", FieldMaskUtil::ToString(out));
  // Overlap with paths covering some other paths.
  FieldMaskUtil::FromString("foo.bar.baz,quz", &mask1);
  FieldMaskUtil::FromString("foo.bar,bar", &mask2);
  FieldMaskUtil::Intersect(mask1, mask2, &out);
  EXPECT_EQ("foo.bar.baz", FieldMaskUtil::ToString(out));
}

TEST(FieldMaskUtilTest, TestIspathInFieldMask) {
  FieldMask mask;
  FieldMaskUtil::FromString("foo.bar", &mask);
  EXPECT_FALSE(FieldMaskUtil::IsPathInFieldMask("", mask));
  EXPECT_FALSE(FieldMaskUtil::IsPathInFieldMask("foo", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("foo.bar", mask));
  EXPECT_TRUE(FieldMaskUtil::IsPathInFieldMask("foo.bar.baz", mask));
  EXPECT_FALSE(FieldMaskUtil::IsPathInFieldMask("foo.bar0.baz", mask));
}

TEST(FieldMaskUtilTest, MergeMessage) {
  TestAllTypes src, dst;
  TestUtil::SetAllFields(&src);
  FieldMaskUtil::MergeOptions options;

#define TEST_MERGE_ONE_PRIMITIVE_FIELD(field_name)           \
  {                                                          \
    TestAllTypes tmp;                                        \
    tmp.set_##field_name(src.field_name());                  \
    FieldMask mask;                                          \
    mask.add_paths(#field_name);                             \
    dst.Clear();                                             \
    FieldMaskUtil::MergeMessageTo(src, mask, options, &dst); \
    EXPECT_EQ(tmp.DebugString(), dst.DebugString());         \
  }
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_int32)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_int64)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_uint32)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_uint64)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_sint32)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_sint64)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_fixed32)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_fixed64)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_sfixed32)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_sfixed64)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_float)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_double)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_bool)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_string)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_bytes)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_nested_enum)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_foreign_enum)
  TEST_MERGE_ONE_PRIMITIVE_FIELD(optional_import_enum)
#undef TEST_MERGE_ONE_PRIMITIVE_FIELD

#define TEST_MERGE_ONE_FIELD(field_name)                     \
  {                                                          \
    TestAllTypes tmp;                                        \
    *tmp.mutable_##field_name() = src.field_name();          \
    FieldMask mask;                                          \
    mask.add_paths(#field_name);                             \
    dst.Clear();                                             \
    FieldMaskUtil::MergeMessageTo(src, mask, options, &dst); \
    EXPECT_EQ(tmp.DebugString(), dst.DebugString());         \
  }
  TEST_MERGE_ONE_FIELD(optional_nested_message)
  TEST_MERGE_ONE_FIELD(optional_foreign_message)
  TEST_MERGE_ONE_FIELD(optional_import_message)

  TEST_MERGE_ONE_FIELD(repeated_int32)
  TEST_MERGE_ONE_FIELD(repeated_int64)
  TEST_MERGE_ONE_FIELD(repeated_uint32)
  TEST_MERGE_ONE_FIELD(repeated_uint64)
  TEST_MERGE_ONE_FIELD(repeated_sint32)
  TEST_MERGE_ONE_FIELD(repeated_sint64)
  TEST_MERGE_ONE_FIELD(repeated_fixed32)
  TEST_MERGE_ONE_FIELD(repeated_fixed64)
  TEST_MERGE_ONE_FIELD(repeated_sfixed32)
  TEST_MERGE_ONE_FIELD(repeated_sfixed64)
  TEST_MERGE_ONE_FIELD(repeated_float)
  TEST_MERGE_ONE_FIELD(repeated_double)
  TEST_MERGE_ONE_FIELD(repeated_bool)
  TEST_MERGE_ONE_FIELD(repeated_string)
  TEST_MERGE_ONE_FIELD(repeated_bytes)
  TEST_MERGE_ONE_FIELD(repeated_nested_message)
  TEST_MERGE_ONE_FIELD(repeated_foreign_message)
  TEST_MERGE_ONE_FIELD(repeated_import_message)
  TEST_MERGE_ONE_FIELD(repeated_nested_enum)
  TEST_MERGE_ONE_FIELD(repeated_foreign_enum)
  TEST_MERGE_ONE_FIELD(repeated_import_enum)
#undef TEST_MERGE_ONE_FIELD

  // Test merge nested fields.
  NestedTestAllTypes nested_src, nested_dst;
  nested_src.mutable_child()->mutable_payload()->set_optional_int32(1234);
  nested_src.mutable_child()
      ->mutable_child()
      ->mutable_payload()
      ->set_optional_int32(5678);
  FieldMask mask;
  FieldMaskUtil::FromString("child.payload", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_EQ(1234, nested_dst.child().payload().optional_int32());
  EXPECT_EQ(0, nested_dst.child().child().payload().optional_int32());

  FieldMaskUtil::FromString("child.child.payload", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_EQ(1234, nested_dst.child().payload().optional_int32());
  EXPECT_EQ(5678, nested_dst.child().child().payload().optional_int32());

  nested_dst.Clear();
  FieldMaskUtil::FromString("child.child.payload", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_EQ(0, nested_dst.child().payload().optional_int32());
  EXPECT_EQ(5678, nested_dst.child().child().payload().optional_int32());

  nested_dst.Clear();
  FieldMaskUtil::FromString("child", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_EQ(1234, nested_dst.child().payload().optional_int32());
  EXPECT_EQ(5678, nested_dst.child().child().payload().optional_int32());

  // Test MergeOptions.

  nested_dst.Clear();
  nested_dst.mutable_child()->mutable_payload()->set_optional_int64(4321);
  // Message fields will be merged by default.
  FieldMaskUtil::FromString("child.payload", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_EQ(1234, nested_dst.child().payload().optional_int32());
  EXPECT_EQ(4321, nested_dst.child().payload().optional_int64());
  // Change the behavior to replace message fields.
  options.set_replace_message_fields(true);
  FieldMaskUtil::FromString("child.payload", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_EQ(1234, nested_dst.child().payload().optional_int32());
  EXPECT_EQ(0, nested_dst.child().payload().optional_int64());

  // By default, fields missing in source are not cleared in destination.
  options.set_replace_message_fields(false);
  nested_dst.mutable_payload();
  EXPECT_TRUE(nested_dst.has_payload());
  FieldMaskUtil::FromString("payload", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_TRUE(nested_dst.has_payload());
  // But they are cleared when replacing message fields.
  options.set_replace_message_fields(true);
  nested_dst.Clear();
  nested_dst.mutable_payload();
  FieldMaskUtil::FromString("payload", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  EXPECT_FALSE(nested_dst.has_payload());

  nested_src.mutable_payload()->add_repeated_int32(1234);
  nested_dst.mutable_payload()->add_repeated_int32(5678);
  // Repeated fields will be appended by default.
  FieldMaskUtil::FromString("payload.repeated_int32", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  ASSERT_EQ(2, nested_dst.payload().repeated_int32_size());
  EXPECT_EQ(5678, nested_dst.payload().repeated_int32(0));
  EXPECT_EQ(1234, nested_dst.payload().repeated_int32(1));
  // Change the behavior to replace repeated fields.
  options.set_replace_repeated_fields(true);
  FieldMaskUtil::FromString("payload.repeated_int32", &mask);
  FieldMaskUtil::MergeMessageTo(nested_src, mask, options, &nested_dst);
  ASSERT_EQ(1, nested_dst.payload().repeated_int32_size());
  EXPECT_EQ(1234, nested_dst.payload().repeated_int32(0));
}


}  // namespace
}  // namespace util
}  // namespace protobuf
}  // namespace google
