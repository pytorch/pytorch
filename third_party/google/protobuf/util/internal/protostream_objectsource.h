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

#ifndef GOOGLE_PROTOBUF_UTIL_CONVERTER_PROTOSTREAM_OBJECTSOURCE_H__
#define GOOGLE_PROTOBUF_UTIL_CONVERTER_PROTOSTREAM_OBJECTSOURCE_H__

#include <functional>
#include <google/protobuf/stubs/hash.h>
#include <string>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/type.pb.h>
#include <google/protobuf/util/internal/object_source.h>
#include <google/protobuf/util/internal/object_writer.h>
#include <google/protobuf/util/internal/type_info.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/stubs/stringpiece.h>
#include <google/protobuf/stubs/status.h>
#include <google/protobuf/stubs/statusor.h>


namespace google {
namespace protobuf {
class Field;
class Type;
}  // namespace protobuf


namespace protobuf {
namespace util {
namespace converter {

class TypeInfo;

// An ObjectSource that can parse a stream of bytes as a protocol buffer.
// Its WriteTo() method can be given an ObjectWriter.
// This implementation uses a google.protobuf.Type for tag and name lookup.
// The field names are converted into lower camel-case when writing to the
// ObjectWriter.
//
// Sample usage: (suppose input is: string proto)
//   ArrayInputStream arr_stream(proto.data(), proto.size());
//   CodedInputStream in_stream(&arr_stream);
//   ProtoStreamObjectSource os(&in_stream, /*ServiceTypeInfo*/ typeinfo,
//                              <your message google::protobuf::Type>);
//
//   Status status = os.WriteTo(<some ObjectWriter>);
class LIBPROTOBUF_EXPORT ProtoStreamObjectSource : public ObjectSource {
 public:
  ProtoStreamObjectSource(google::protobuf::io::CodedInputStream* stream,
                          TypeResolver* type_resolver,
                          const google::protobuf::Type& type);

  virtual ~ProtoStreamObjectSource();

  virtual util::Status NamedWriteTo(StringPiece name, ObjectWriter* ow) const;

 protected:
  // Writes a proto2 Message to the ObjectWriter. When the given end_tag is
  // found this method will complete, allowing it to be used for parsing both
  // nested messages (end with 0) and nested groups (end with group end tag).
  // The include_start_and_end parameter allows this method to be called when
  // already inside of an object, and skip calling StartObject and EndObject.
  virtual util::Status WriteMessage(const google::protobuf::Type& descriptor,
                                      StringPiece name, const uint32 end_tag,
                                      bool include_start_and_end,
                                      ObjectWriter* ow) const;

 private:
  ProtoStreamObjectSource(google::protobuf::io::CodedInputStream* stream,
                          const TypeInfo* typeinfo,
                          const google::protobuf::Type& type);
  // Function that renders a well known type with a modified behavior.
  typedef util::Status (*TypeRenderer)(const ProtoStreamObjectSource*,
                                         const google::protobuf::Type&,
                                         StringPiece, ObjectWriter*);

  // Looks up a field and verify its consistency with wire type in tag.
  const google::protobuf::Field* FindAndVerifyField(
      const google::protobuf::Type& type, uint32 tag) const;

  // TODO(skarvaje): Mark these methods as non-const as they modify internal
  // state (stream_).
  //
  // Renders a repeating field (packed or unpacked).
  // Returns the next tag after reading all sequential repeating elements. The
  // caller should use this tag before reading more tags from the stream.
  util::StatusOr<uint32> RenderList(const google::protobuf::Field* field,
                                      StringPiece name, uint32 list_tag,
                                      ObjectWriter* ow) const;
  // Renders a NWP map.
  // Returns the next tag after reading all map entries. The caller should use
  // this tag before reading more tags from the stream.
  util::StatusOr<uint32> RenderMap(const google::protobuf::Field* field,
                                     StringPiece name, uint32 list_tag,
                                     ObjectWriter* ow) const;

  // Renders an entry in a map, advancing stream pointers appropriately.
  util::Status RenderMapEntry(const google::protobuf::Type* type,
                                ObjectWriter* ow) const;

  // Renders a packed repeating field. A packed field is stored as:
  // {tag length item1 item2 item3} instead of the less efficient
  // {tag item1 tag item2 tag item3}.
  util::Status RenderPacked(const google::protobuf::Field* field,
                              ObjectWriter* ow) const;

  // Equivalent of RenderPacked, but for map entries.
  util::Status RenderPackedMapEntry(const google::protobuf::Type* type,
                                      ObjectWriter* ow) const;

  // Renders a google.protobuf.Timestamp value to ObjectWriter
  static util::Status RenderTimestamp(const ProtoStreamObjectSource* os,
                                        const google::protobuf::Type& type,
                                        StringPiece name, ObjectWriter* ow);

  // Renders a google.protobuf.Duration value to ObjectWriter
  static util::Status RenderDuration(const ProtoStreamObjectSource* os,
                                       const google::protobuf::Type& type,
                                       StringPiece name, ObjectWriter* ow);

  // Following RenderTYPE functions render well known types in
  // google/protobuf/wrappers.proto corresponding to TYPE.
  static util::Status RenderDouble(const ProtoStreamObjectSource* os,
                                     const google::protobuf::Type& type,
                                     StringPiece name, ObjectWriter* ow);
  static util::Status RenderFloat(const ProtoStreamObjectSource* os,
                                    const google::protobuf::Type& type,
                                    StringPiece name, ObjectWriter* ow);
  static util::Status RenderInt64(const ProtoStreamObjectSource* os,
                                    const google::protobuf::Type& type,
                                    StringPiece name, ObjectWriter* ow);
  static util::Status RenderUInt64(const ProtoStreamObjectSource* os,
                                     const google::protobuf::Type& type,
                                     StringPiece name, ObjectWriter* ow);
  static util::Status RenderInt32(const ProtoStreamObjectSource* os,
                                    const google::protobuf::Type& type,
                                    StringPiece name, ObjectWriter* ow);
  static util::Status RenderUInt32(const ProtoStreamObjectSource* os,
                                     const google::protobuf::Type& type,
                                     StringPiece name, ObjectWriter* ow);
  static util::Status RenderBool(const ProtoStreamObjectSource* os,
                                   const google::protobuf::Type& type,
                                   StringPiece name, ObjectWriter* ow);
  static util::Status RenderString(const ProtoStreamObjectSource* os,
                                     const google::protobuf::Type& type,
                                     StringPiece name, ObjectWriter* ow);
  static util::Status RenderBytes(const ProtoStreamObjectSource* os,
                                    const google::protobuf::Type& type,
                                    StringPiece name, ObjectWriter* ow);

  // Renders a google.protobuf.Struct to ObjectWriter.
  static util::Status RenderStruct(const ProtoStreamObjectSource* os,
                                     const google::protobuf::Type& type,
                                     StringPiece name, ObjectWriter* ow);

  // Helper to render google.protobuf.Struct's Value fields to ObjectWriter.
  static util::Status RenderStructValue(const ProtoStreamObjectSource* os,
                                          const google::protobuf::Type& type,
                                          StringPiece name, ObjectWriter* ow);

  // Helper to render google.protobuf.Struct's ListValue fields to ObjectWriter.
  static util::Status RenderStructListValue(
      const ProtoStreamObjectSource* os, const google::protobuf::Type& type,
      StringPiece name, ObjectWriter* ow);

  // Render the "Any" type.
  static util::Status RenderAny(const ProtoStreamObjectSource* os,
                                  const google::protobuf::Type& type,
                                  StringPiece name, ObjectWriter* ow);

  // Render the "FieldMask" type.
  static util::Status RenderFieldMask(const ProtoStreamObjectSource* os,
                                        const google::protobuf::Type& type,
                                        StringPiece name, ObjectWriter* ow);

  static hash_map<string, TypeRenderer>* renderers_;
  static void InitRendererMap();
  static void DeleteRendererMap();
  static TypeRenderer* FindTypeRenderer(const string& type_url);

  // Renders a field value to the ObjectWriter.
  util::Status RenderField(const google::protobuf::Field* field,
                             StringPiece field_name, ObjectWriter* ow) const;


  // Reads field value according to Field spec in 'field' and returns the read
  // value as string. This only works for primitive datatypes (no message
  // types).
  const string ReadFieldValueAsString(
      const google::protobuf::Field& field) const;

  // Utility function to detect proto maps. The 'field' MUST be repeated.
  bool IsMap(const google::protobuf::Field& field) const;

  // Utility to read int64 and int32 values from a message type in stream_.
  // Used for reading google.protobuf.Timestamp and Duration messages.
  std::pair<int64, int32> ReadSecondsAndNanos(
      const google::protobuf::Type& type) const;

  // Input stream to read from. Ownership rests with the caller.
  google::protobuf::io::CodedInputStream* stream_;

  // Type information for all the types used in the descriptor. Used to find
  // google::protobuf::Type of nested messages/enums.
  const TypeInfo* typeinfo_;
  // Whether this class owns the typeinfo_ object. If true the typeinfo_ object
  // should be deleted in the destructor.
  bool own_typeinfo_;

  // google::protobuf::Type of the message source.
  const google::protobuf::Type& type_;

  GOOGLE_DISALLOW_IMPLICIT_CONSTRUCTORS(ProtoStreamObjectSource);
};

}  // namespace converter
}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_CONVERTER_PROTOSTREAM_OBJECTSOURCE_H__
