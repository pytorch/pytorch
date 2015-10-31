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

#include <google/protobuf/util/internal/protostream_objectsource.h>

#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <sstream>

#include <google/protobuf/stubs/casts.h>
#include <google/protobuf/any.pb.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/util/internal/expecting_objectwriter.h>
#include <google/protobuf/util/internal/testdata/books.pb.h>
#include <google/protobuf/util/internal/testdata/field_mask.pb.h>
#include <google/protobuf/util/internal/type_info_test_helper.h>
#include <google/protobuf/util/internal/constants.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/util/internal/testdata/anys.pb.h>
#include <google/protobuf/util/internal/testdata/maps.pb.h>
#include <google/protobuf/util/internal/testdata/struct.pb.h>
#include <gtest/gtest.h>


namespace google {
namespace protobuf {
namespace util {
namespace converter {

using google::protobuf::Descriptor;
using google::protobuf::DescriptorPool;
using google::protobuf::FileDescriptorProto;
using google::protobuf::Message;
using google::protobuf::io::ArrayInputStream;
using google::protobuf::io::CodedInputStream;
using util::Status;
using google::protobuf::testing::Author;
using google::protobuf::testing::BadAuthor;
using google::protobuf::testing::BadNestedBook;
using google::protobuf::testing::Book;
using google::protobuf::testing::Book_Label;
using google::protobuf::testing::NestedBook;
using google::protobuf::testing::PackedPrimitive;
using google::protobuf::testing::Primitive;
using google::protobuf::testing::more_author;
using google::protobuf::testing::maps::MapOut;
using google::protobuf::testing::anys::AnyOut;
using google::protobuf::testing::anys::AnyM;
using google::protobuf::testing::FieldMaskTest;
using google::protobuf::testing::NestedFieldMask;
using google::protobuf::testing::structs::StructType;
using ::testing::_;


namespace {
string GetTypeUrl(const Descriptor* descriptor) {
  return string(kTypeServiceBaseUrl) + "/" + descriptor->full_name();
}
}  // namespace

class ProtostreamObjectSourceTest
    : public ::testing::TestWithParam<testing::TypeInfoSource> {
 protected:
  ProtostreamObjectSourceTest() : helper_(GetParam()), mock_(), ow_(&mock_) {
    helper_.ResetTypeInfo(Book::descriptor());
  }

  virtual ~ProtostreamObjectSourceTest() {}

  void DoTest(const Message& msg, const Descriptor* descriptor) {
    Status status = ExecuteTest(msg, descriptor);
    EXPECT_EQ(Status::OK, status);
  }

  Status ExecuteTest(const Message& msg, const Descriptor* descriptor) {
    ostringstream oss;
    msg.SerializePartialToOstream(&oss);
    string proto = oss.str();
    ArrayInputStream arr_stream(proto.data(), proto.size());
    CodedInputStream in_stream(&arr_stream);

    google::protobuf::scoped_ptr<ProtoStreamObjectSource> os(
        helper_.NewProtoSource(&in_stream, GetTypeUrl(descriptor)));
    return os->WriteTo(&mock_);
  }

  void PrepareExpectingObjectWriterForRepeatedPrimitive() {
    ow_.StartObject("")
        ->StartList("repFix32")
        ->RenderUint32("", bit_cast<uint32>(3201))
        ->RenderUint32("", bit_cast<uint32>(0))
        ->RenderUint32("", bit_cast<uint32>(3202))
        ->EndList()
        ->StartList("repU32")
        ->RenderUint32("", bit_cast<uint32>(3203))
        ->RenderUint32("", bit_cast<uint32>(0))
        ->EndList()
        ->StartList("repI32")
        ->RenderInt32("", 0)
        ->RenderInt32("", 3204)
        ->RenderInt32("", 3205)
        ->EndList()
        ->StartList("repSf32")
        ->RenderInt32("", 3206)
        ->RenderInt32("", 0)
        ->EndList()
        ->StartList("repS32")
        ->RenderInt32("", 0)
        ->RenderInt32("", 3207)
        ->RenderInt32("", 3208)
        ->EndList()
        ->StartList("repFix64")
        ->RenderUint64("", bit_cast<uint64>(6401LL))
        ->RenderUint64("", bit_cast<uint64>(0LL))
        ->EndList()
        ->StartList("repU64")
        ->RenderUint64("", bit_cast<uint64>(0LL))
        ->RenderUint64("", bit_cast<uint64>(6402LL))
        ->RenderUint64("", bit_cast<uint64>(6403LL))
        ->EndList()
        ->StartList("repI64")
        ->RenderInt64("", 6404L)
        ->RenderInt64("", 0L)
        ->EndList()
        ->StartList("repSf64")
        ->RenderInt64("", 0L)
        ->RenderInt64("", 6405L)
        ->RenderInt64("", 6406L)
        ->EndList()
        ->StartList("repS64")
        ->RenderInt64("", 6407L)
        ->RenderInt64("", 0L)
        ->EndList()
        ->StartList("repFloat")
        ->RenderFloat("", 0.0f)
        ->RenderFloat("", 32.1f)
        ->RenderFloat("", 32.2f)
        ->EndList()
        ->StartList("repDouble")
        ->RenderDouble("", 64.1L)
        ->RenderDouble("", 0.0L)
        ->EndList()
        ->StartList("repBool")
        ->RenderBool("", true)
        ->RenderBool("", false)
        ->EndList()
        ->EndObject();
  }

  Primitive PrepareRepeatedPrimitive() {
    Primitive primitive;
    primitive.add_rep_fix32(3201);
    primitive.add_rep_fix32(0);
    primitive.add_rep_fix32(3202);
    primitive.add_rep_u32(3203);
    primitive.add_rep_u32(0);
    primitive.add_rep_i32(0);
    primitive.add_rep_i32(3204);
    primitive.add_rep_i32(3205);
    primitive.add_rep_sf32(3206);
    primitive.add_rep_sf32(0);
    primitive.add_rep_s32(0);
    primitive.add_rep_s32(3207);
    primitive.add_rep_s32(3208);
    primitive.add_rep_fix64(6401L);
    primitive.add_rep_fix64(0L);
    primitive.add_rep_u64(0L);
    primitive.add_rep_u64(6402L);
    primitive.add_rep_u64(6403L);
    primitive.add_rep_i64(6404L);
    primitive.add_rep_i64(0L);
    primitive.add_rep_sf64(0L);
    primitive.add_rep_sf64(6405L);
    primitive.add_rep_sf64(6406L);
    primitive.add_rep_s64(6407L);
    primitive.add_rep_s64(0L);
    primitive.add_rep_float(0.0f);
    primitive.add_rep_float(32.1f);
    primitive.add_rep_float(32.2f);
    primitive.add_rep_double(64.1L);
    primitive.add_rep_double(0.0);
    primitive.add_rep_bool(true);
    primitive.add_rep_bool(false);

    PrepareExpectingObjectWriterForRepeatedPrimitive();
    return primitive;
  }

  PackedPrimitive PreparePackedPrimitive() {
    PackedPrimitive primitive;
    primitive.add_rep_fix32(3201);
    primitive.add_rep_fix32(0);
    primitive.add_rep_fix32(3202);
    primitive.add_rep_u32(3203);
    primitive.add_rep_u32(0);
    primitive.add_rep_i32(0);
    primitive.add_rep_i32(3204);
    primitive.add_rep_i32(3205);
    primitive.add_rep_sf32(3206);
    primitive.add_rep_sf32(0);
    primitive.add_rep_s32(0);
    primitive.add_rep_s32(3207);
    primitive.add_rep_s32(3208);
    primitive.add_rep_fix64(6401L);
    primitive.add_rep_fix64(0L);
    primitive.add_rep_u64(0L);
    primitive.add_rep_u64(6402L);
    primitive.add_rep_u64(6403L);
    primitive.add_rep_i64(6404L);
    primitive.add_rep_i64(0L);
    primitive.add_rep_sf64(0L);
    primitive.add_rep_sf64(6405L);
    primitive.add_rep_sf64(6406L);
    primitive.add_rep_s64(6407L);
    primitive.add_rep_s64(0L);
    primitive.add_rep_float(0.0f);
    primitive.add_rep_float(32.1f);
    primitive.add_rep_float(32.2f);
    primitive.add_rep_double(64.1L);
    primitive.add_rep_double(0.0);
    primitive.add_rep_bool(true);
    primitive.add_rep_bool(false);

    PrepareExpectingObjectWriterForRepeatedPrimitive();
    return primitive;
  }

  testing::TypeInfoTestHelper helper_;

  ::testing::NiceMock<MockObjectWriter> mock_;
  ExpectingObjectWriter ow_;
};

INSTANTIATE_TEST_CASE_P(DifferentTypeInfoSourceTest,
                        ProtostreamObjectSourceTest,
                        ::testing::Values(
                            testing::USE_TYPE_RESOLVER));

TEST_P(ProtostreamObjectSourceTest, EmptyMessage) {
  Book empty;
  ow_.StartObject("")->EndObject();
  DoTest(empty, Book::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, Primitives) {
  Primitive primitive;
  primitive.set_fix32(3201);
  primitive.set_u32(3202);
  primitive.set_i32(3203);
  primitive.set_sf32(3204);
  primitive.set_s32(3205);
  primitive.set_fix64(6401L);
  primitive.set_u64(6402L);
  primitive.set_i64(6403L);
  primitive.set_sf64(6404L);
  primitive.set_s64(6405L);
  primitive.set_str("String Value");
  primitive.set_bytes("Some Bytes");
  primitive.set_float_(32.1f);
  primitive.set_double_(64.1L);
  primitive.set_bool_(true);

  ow_.StartObject("")
      ->RenderUint32("fix32", bit_cast<uint32>(3201))
      ->RenderUint32("u32", bit_cast<uint32>(3202))
      ->RenderInt32("i32", 3203)
      ->RenderInt32("sf32", 3204)
      ->RenderInt32("s32", 3205)
      ->RenderUint64("fix64", bit_cast<uint64>(6401LL))
      ->RenderUint64("u64", bit_cast<uint64>(6402LL))
      ->RenderInt64("i64", 6403L)
      ->RenderInt64("sf64", 6404L)
      ->RenderInt64("s64", 6405L)
      ->RenderString("str", "String Value")
      ->RenderBytes("bytes", "Some Bytes")
      ->RenderFloat("float", 32.1f)
      ->RenderDouble("double", 64.1L)
      ->RenderBool("bool", true)
      ->EndObject();
  DoTest(primitive, Primitive::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, RepeatingPrimitives) {
  Primitive primitive = PrepareRepeatedPrimitive();
  primitive.add_rep_str("String One");
  primitive.add_rep_str("String Two");
  primitive.add_rep_bytes("Some Bytes");

  ow_.StartList("repStr")
      ->RenderString("", "String One")
      ->RenderString("", "String Two")
      ->EndList()
      ->StartList("repBytes")
      ->RenderBytes("", "Some Bytes")
      ->EndList();
  DoTest(primitive, Primitive::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, NestedMessage) {
  Author* author = new Author();
  author->set_id(101L);
  author->set_name("Tolstoy");
  Book book;
  book.set_title("My Book");
  book.set_allocated_author(author);

  ow_.StartObject("")
      ->RenderString("title", "My Book")
      ->StartObject("author")
      ->RenderUint64("id", bit_cast<uint64>(101LL))
      ->RenderString("name", "Tolstoy")
      ->EndObject()
      ->EndObject();
  DoTest(book, Book::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, RepeatingField) {
  Author author;
  author.set_alive(false);
  author.set_name("john");
  author.add_pseudonym("phil");
  author.add_pseudonym("bob");

  ow_.StartObject("")
      ->RenderBool("alive", false)
      ->RenderString("name", "john")
      ->StartList("pseudonym")
      ->RenderString("", "phil")
      ->RenderString("", "bob")
      ->EndList()
      ->EndObject();
  DoTest(author, Author::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, PackedRepeatingFields) {
  DoTest(PreparePackedPrimitive(), PackedPrimitive::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, NonPackedPackableFieldsActuallyPacked) {
  // Protostream is packed, but parse with non-packed Primitive.
  DoTest(PreparePackedPrimitive(), Primitive::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, PackedPackableFieldNotActuallyPacked) {
  // Protostream is not packed, but parse with PackedPrimitive.
  DoTest(PrepareRepeatedPrimitive(), PackedPrimitive::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, BadAuthor) {
  Author author;
  author.set_alive(false);
  author.set_name("john");
  author.set_id(1234L);
  author.add_pseudonym("phil");
  author.add_pseudonym("bob");

  ow_.StartObject("")
      ->StartList("alive")
      ->RenderBool("", false)
      ->EndList()
      ->StartList("name")
      ->RenderUint64("", static_cast<uint64>('j'))
      ->RenderUint64("", static_cast<uint64>('o'))
      ->RenderUint64("", static_cast<uint64>('h'))
      ->RenderUint64("", static_cast<uint64>('n'))
      ->EndList()
      ->RenderString("pseudonym", "phil")
      ->RenderString("pseudonym", "bob")
      ->EndObject();
  // Protostream created with Author, but parsed with BadAuthor.
  DoTest(author, BadAuthor::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, NestedBookToBadNestedBook) {
  Book* book = new Book();
  book->set_length(250);
  book->set_published(2014L);
  NestedBook nested;
  nested.set_allocated_book(book);

  ow_.StartObject("")
      ->StartList("book")
      ->RenderUint32("", 24)  // tag for field length (3 << 3)
      ->RenderUint32("", 250)
      ->RenderUint32("", 32)  // tag for field published (4 << 3)
      ->RenderUint32("", 2014)
      ->EndList()
      ->EndObject();
  // Protostream created with NestedBook, but parsed with BadNestedBook.
  DoTest(nested, BadNestedBook::descriptor());
}

TEST_P(ProtostreamObjectSourceTest, BadNestedBookToNestedBook) {
  BadNestedBook nested;
  nested.add_book(1);
  nested.add_book(2);
  nested.add_book(3);
  nested.add_book(4);
  nested.add_book(5);
  nested.add_book(6);
  nested.add_book(7);

  ow_.StartObject("")->StartObject("book")->EndObject()->EndObject();
  // Protostream created with BadNestedBook, but parsed with NestedBook.
  DoTest(nested, NestedBook::descriptor());
}

TEST_P(ProtostreamObjectSourceTest,
       LongRepeatedListDoesNotBreakIntoMultipleJsonLists) {
  Book book;

  int repeat = 10000;
  for (int i = 0; i < repeat; ++i) {
    Book_Label* label = book.add_labels();
    label->set_key(StrCat("i", i));
    label->set_value(StrCat("v", i));
  }

  // Make sure StartList and EndList are called exactly once (see b/18227499 for
  // problems when this doesn't happen)
  EXPECT_CALL(mock_, StartList(_)).Times(1);
  EXPECT_CALL(mock_, EndList()).Times(1);

  DoTest(book, Book::descriptor());
}

class ProtostreamObjectSourceMapsTest : public ProtostreamObjectSourceTest {
 protected:
  ProtostreamObjectSourceMapsTest() {
    helper_.ResetTypeInfo(MapOut::descriptor());
  }
};

INSTANTIATE_TEST_CASE_P(DifferentTypeInfoSourceTest,
                        ProtostreamObjectSourceMapsTest,
                        ::testing::Values(
                            testing::USE_TYPE_RESOLVER));

// Tests JSON map.
//
// This is the example expected output.
// {
//   "map1": {
//     "key1": {
//       "foo": "foovalue"
//     },
//     "key2": {
//       "foo": "barvalue"
//     }
//   },
//   "map2": {
//     "nestedself": {
//       "map1": {
//         "nested_key1": {
//           "foo": "nested_foo"
//         }
//       },
//       "bar": "nested_bar_string"
//     }
//   },
//   "map3": {
//     "111": "one one one"
//   },
//   "bar": "top bar"
// }
TEST_P(ProtostreamObjectSourceMapsTest, MapsTest) {
  MapOut out;
  (*out.mutable_map1())["key1"].set_foo("foovalue");
  (*out.mutable_map1())["key2"].set_foo("barvalue");

  MapOut* nested_value = &(*out.mutable_map2())["nestedself"];
  (*nested_value->mutable_map1())["nested_key1"].set_foo("nested_foo");
  nested_value->set_bar("nested_bar_string");

  (*out.mutable_map3())[111] = "one one one";

  out.set_bar("top bar");

  ow_.StartObject("")
      ->StartObject("map1")
      ->StartObject("key1")
      ->RenderString("foo", "foovalue")
      ->EndObject()
      ->StartObject("key2")
      ->RenderString("foo", "barvalue")
      ->EndObject()
      ->StartObject("map2")
      ->StartObject("nestedself")
      ->StartObject("map1")
      ->StartObject("nested_key1")
      ->RenderString("foo", "nested_foo")
      ->EndObject()
      ->EndObject()
      ->RenderString("bar", "nested_bar_string")
      ->EndObject()
      ->EndObject()
      ->StartObject("map3")
      ->RenderString("111", "one one one")
      ->EndObject()
      ->EndObject()
      ->RenderString("bar", "top bar")
      ->EndObject();

  DoTest(out, MapOut::descriptor());
}

class ProtostreamObjectSourceAnysTest : public ProtostreamObjectSourceTest {
 protected:
  ProtostreamObjectSourceAnysTest() {
    helper_.ResetTypeInfo(AnyOut::descriptor(),
                          google::protobuf::Any::descriptor());
  }
};

INSTANTIATE_TEST_CASE_P(DifferentTypeInfoSourceTest,
                        ProtostreamObjectSourceAnysTest,
                        ::testing::Values(
                            testing::USE_TYPE_RESOLVER));

// Tests JSON any support.
//
// This is the example expected output.
// {
//   "any": {
//     "@type": "type.googleapis.com/google.protobuf.testing.anys.AnyM"
//     "foo": "foovalue"
//   }
// }
TEST_P(ProtostreamObjectSourceAnysTest, BasicAny) {
  AnyOut out;
  ::google::protobuf::Any* any = out.mutable_any();

  AnyM m;
  m.set_foo("foovalue");
  any->PackFrom(m);

  ow_.StartObject("")
      ->StartObject("any")
      ->RenderString("@type",
                     "type.googleapis.com/google.protobuf.testing.anys.AnyM")
      ->RenderString("foo", "foovalue")
      ->EndObject()
      ->EndObject();

  DoTest(out, AnyOut::descriptor());
}

TEST_P(ProtostreamObjectSourceAnysTest, RecursiveAny) {
  AnyOut out;
  ::google::protobuf::Any* any = out.mutable_any();
  any->set_type_url("type.googleapis.com/google.protobuf.Any");

  ::google::protobuf::Any nested_any;
  nested_any.set_type_url(
      "type.googleapis.com/google.protobuf.testing.anys.AnyM");

  AnyM m;
  m.set_foo("foovalue");
  nested_any.set_value(m.SerializeAsString());

  any->set_value(nested_any.SerializeAsString());

  ow_.StartObject("")
      ->StartObject("any")
      ->RenderString("@type", "type.googleapis.com/google.protobuf.Any")
      ->StartObject("value")
      ->RenderString("@type",
                     "type.googleapis.com/google.protobuf.testing.anys.AnyM")
      ->RenderString("foo", "foovalue")
      ->EndObject()
      ->EndObject()
      ->EndObject();

  DoTest(out, AnyOut::descriptor());
}

TEST_P(ProtostreamObjectSourceAnysTest, DoubleRecursiveAny) {
  AnyOut out;
  ::google::protobuf::Any* any = out.mutable_any();
  any->set_type_url("type.googleapis.com/google.protobuf.Any");

  ::google::protobuf::Any nested_any;
  nested_any.set_type_url("type.googleapis.com/google.protobuf.Any");

  ::google::protobuf::Any second_nested_any;
  second_nested_any.set_type_url(
      "type.googleapis.com/google.protobuf.testing.anys.AnyM");

  AnyM m;
  m.set_foo("foovalue");
  second_nested_any.set_value(m.SerializeAsString());
  nested_any.set_value(second_nested_any.SerializeAsString());
  any->set_value(nested_any.SerializeAsString());

  ow_.StartObject("")
      ->StartObject("any")
      ->RenderString("@type", "type.googleapis.com/google.protobuf.Any")
      ->StartObject("value")
      ->RenderString("@type", "type.googleapis.com/google.protobuf.Any")
      ->StartObject("value")
      ->RenderString("@type",
                     "type.googleapis.com/google.protobuf.testing.anys.AnyM")
      ->RenderString("foo", "foovalue")
      ->EndObject()
      ->EndObject()
      ->EndObject()
      ->EndObject();

  DoTest(out, AnyOut::descriptor());
}

TEST_P(ProtostreamObjectSourceAnysTest, EmptyAnyOutputsEmptyObject) {
  AnyOut out;
  out.mutable_any();

  ow_.StartObject("")->StartObject("any")->EndObject()->EndObject();

  DoTest(out, AnyOut::descriptor());
}

TEST_P(ProtostreamObjectSourceAnysTest, EmptyWithTypeAndNoValueOutputsType) {
  AnyOut out;
  out.mutable_any()->set_type_url("foo.googleapis.com/my.Type");

  ow_.StartObject("")
      ->StartObject("any")
      ->RenderString("@type", "foo.googleapis.com/my.Type")
      ->EndObject()
      ->EndObject();

  DoTest(out, AnyOut::descriptor());
}

TEST_P(ProtostreamObjectSourceAnysTest, MissingTypeUrlError) {
  AnyOut out;
  ::google::protobuf::Any* any = out.mutable_any();

  AnyM m;
  m.set_foo("foovalue");
  any->set_value(m.SerializeAsString());

  // We start the "AnyOut" part and then fail when we hit the Any object.
  ow_.StartObject("");

  Status status = ExecuteTest(out, AnyOut::descriptor());
  EXPECT_EQ(util::error::INTERNAL, status.error_code());
}

TEST_P(ProtostreamObjectSourceAnysTest, UnknownTypeServiceError) {
  AnyOut out;
  ::google::protobuf::Any* any = out.mutable_any();
  any->set_type_url("foo.googleapis.com/my.own.Type");

  AnyM m;
  m.set_foo("foovalue");
  any->set_value(m.SerializeAsString());

  // We start the "AnyOut" part and then fail when we hit the Any object.
  ow_.StartObject("");

  Status status = ExecuteTest(out, AnyOut::descriptor());
  EXPECT_EQ(util::error::INTERNAL, status.error_code());
}

TEST_P(ProtostreamObjectSourceAnysTest, UnknownTypeError) {
  AnyOut out;
  ::google::protobuf::Any* any = out.mutable_any();
  any->set_type_url("type.googleapis.com/unknown.Type");

  AnyM m;
  m.set_foo("foovalue");
  any->set_value(m.SerializeAsString());

  // We start the "AnyOut" part and then fail when we hit the Any object.
  ow_.StartObject("");

  Status status = ExecuteTest(out, AnyOut::descriptor());
  EXPECT_EQ(util::error::INTERNAL, status.error_code());
}

class ProtostreamObjectSourceStructTest : public ProtostreamObjectSourceTest {
 protected:
  ProtostreamObjectSourceStructTest() {
    helper_.ResetTypeInfo(StructType::descriptor(),
                          google::protobuf::Struct::descriptor());
  }
};

INSTANTIATE_TEST_CASE_P(DifferentTypeInfoSourceTest,
                        ProtostreamObjectSourceStructTest,
                        ::testing::Values(
                            testing::USE_TYPE_RESOLVER));

// Tests struct
//
//  "object": {
//    "k1": 123,
//    "k2": true
//  }
TEST_P(ProtostreamObjectSourceStructTest, StructRenderSuccess) {
  StructType out;
  google::protobuf::Struct* s = out.mutable_object();
  s->mutable_fields()->operator[]("k1").set_number_value(123);
  s->mutable_fields()->operator[]("k2").set_bool_value(true);

  ow_.StartObject("")
      ->StartObject("object")
      ->RenderDouble("k1", 123)
      ->RenderBool("k2", true)
      ->EndObject()
      ->EndObject();

  DoTest(out, StructType::descriptor());
}

TEST_P(ProtostreamObjectSourceStructTest, MissingValueSkipsField) {
  StructType out;
  google::protobuf::Struct* s = out.mutable_object();
  s->mutable_fields()->operator[]("k1");

  ow_.StartObject("")->StartObject("object")->EndObject()->EndObject();

  DoTest(out, StructType::descriptor());
}

class ProtostreamObjectSourceFieldMaskTest
    : public ProtostreamObjectSourceTest {
 protected:
  ProtostreamObjectSourceFieldMaskTest() {
    helper_.ResetTypeInfo(FieldMaskTest::descriptor(),
                          google::protobuf::FieldMask::descriptor());
  }
};

INSTANTIATE_TEST_CASE_P(DifferentTypeInfoSourceTest,
                        ProtostreamObjectSourceFieldMaskTest,
                        ::testing::Values(
                            testing::USE_TYPE_RESOLVER));

TEST_P(ProtostreamObjectSourceFieldMaskTest, FieldMaskRenderSuccess) {
  FieldMaskTest out;
  out.set_id("1");
  out.mutable_single_mask()->add_paths("path1");
  out.mutable_single_mask()->add_paths("snake_case_path2");
  ::google::protobuf::FieldMask* mask = out.add_repeated_mask();
  mask->add_paths("path3");
  mask = out.add_repeated_mask();
  mask->add_paths("snake_case_path4");
  mask->add_paths("path5");
  NestedFieldMask* nested = out.add_nested_mask();
  nested->set_data("data");
  nested->mutable_single_mask()->add_paths("nested.path1");
  nested->mutable_single_mask()->add_paths("nested_field.snake_case_path2");
  mask = nested->add_repeated_mask();
  mask->add_paths("nested_field.path3");
  mask->add_paths("nested.snake_case_path4");
  mask = nested->add_repeated_mask();
  mask->add_paths("nested.path5");
  mask = nested->add_repeated_mask();
  mask->add_paths(
      "snake_case.map_field[\"map_key_should_be_ignored\"].nested_snake_case."
      "map_field[\"map_key_sho\\\"uld_be_ignored\"]");

  ow_.StartObject("")
      ->RenderString("id", "1")
      ->RenderString("singleMask", "path1,snakeCasePath2")
      ->StartList("repeatedMask")
      ->RenderString("", "path3")
      ->RenderString("", "snakeCasePath4,path5")
      ->EndList()
      ->StartList("nestedMask")
      ->StartObject("")
      ->RenderString("data", "data")
      ->RenderString("singleMask", "nested.path1,nestedField.snakeCasePath2")
      ->StartList("repeatedMask")
      ->RenderString("", "nestedField.path3,nested.snakeCasePath4")
      ->RenderString("", "nested.path5")
      ->RenderString("",
                     "snakeCase.mapField[\"map_key_should_be_ignored\"]."
                     "nestedSnakeCase.mapField[\"map_key_sho\\\"uld_be_"
                     "ignored\"]")
      ->EndList()
      ->EndObject()
      ->EndList()
      ->EndObject();

  DoTest(out, FieldMaskTest::descriptor());
}

}  // namespace converter
}  // namespace util
}  // namespace protobuf
}  // namespace google
