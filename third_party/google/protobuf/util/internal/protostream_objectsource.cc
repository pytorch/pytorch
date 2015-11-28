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

#include <utility>

#include <google/protobuf/stubs/casts.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/stringprintf.h>
#include <google/protobuf/stubs/time.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/util/internal/field_mask_utility.h>
#include <google/protobuf/util/internal/constants.h>
#include <google/protobuf/util/internal/utility.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/map_util.h>
#include <google/protobuf/stubs/status_macros.h>


namespace google {
namespace protobuf {
namespace util {
using util::Status;
using util::StatusOr;
namespace error {
using util::error::Code;
using util::error::INTERNAL;
}
namespace converter {

using google::protobuf::Descriptor;
using google::protobuf::EnumValueDescriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::internal::WireFormat;
using google::protobuf::internal::WireFormatLite;
using util::Status;
using util::StatusOr;

namespace {
// Finds a field with the given number. NULL if none found.
const google::protobuf::Field* FindFieldByNumber(
    const google::protobuf::Type& type, int number);

// Returns true if the field is packable.
bool IsPackable(const google::protobuf::Field& field);

// Finds an enum value with the given number. NULL if none found.
const google::protobuf::EnumValue* FindEnumValueByNumber(
    const google::protobuf::Enum& tech_enum, int number);

// Utility function to format nanos.
const string FormatNanos(uint32 nanos);
}  // namespace


ProtoStreamObjectSource::ProtoStreamObjectSource(
    google::protobuf::io::CodedInputStream* stream, TypeResolver* type_resolver,
    const google::protobuf::Type& type)
    : stream_(stream),
      typeinfo_(TypeInfo::NewTypeInfo(type_resolver)),
      own_typeinfo_(true),
      type_(type) {
  GOOGLE_LOG_IF(DFATAL, stream == NULL) << "Input stream is NULL.";
}

ProtoStreamObjectSource::ProtoStreamObjectSource(
    google::protobuf::io::CodedInputStream* stream, const TypeInfo* typeinfo,
    const google::protobuf::Type& type)
    : stream_(stream), typeinfo_(typeinfo), own_typeinfo_(false), type_(type) {
  GOOGLE_LOG_IF(DFATAL, stream == NULL) << "Input stream is NULL.";
}

ProtoStreamObjectSource::~ProtoStreamObjectSource() {
  if (own_typeinfo_) {
    delete typeinfo_;
  }
}

Status ProtoStreamObjectSource::NamedWriteTo(StringPiece name,
                                             ObjectWriter* ow) const {
  return WriteMessage(type_, name, 0, true, ow);
}

const google::protobuf::Field* ProtoStreamObjectSource::FindAndVerifyField(
    const google::protobuf::Type& type, uint32 tag) const {
  // Lookup the new field in the type by tag number.
  const google::protobuf::Field* field = FindFieldByNumber(type, tag >> 3);
  // Verify if the field corresponds to the wire type in tag.
  // If there is any discrepancy, mark the field as not found.
  if (field != NULL) {
    WireFormatLite::WireType expected_type =
        WireFormatLite::WireTypeForFieldType(
            static_cast<WireFormatLite::FieldType>(field->kind()));
    WireFormatLite::WireType actual_type = WireFormatLite::GetTagWireType(tag);
    if (actual_type != expected_type &&
        (!IsPackable(*field) ||
         actual_type != WireFormatLite::WIRETYPE_LENGTH_DELIMITED)) {
      field = NULL;
    }
  }
  return field;
}

Status ProtoStreamObjectSource::WriteMessage(const google::protobuf::Type& type,
                                             StringPiece name,
                                             const uint32 end_tag,
                                             bool include_start_and_end,
                                             ObjectWriter* ow) const {

  const TypeRenderer* type_renderer = FindTypeRenderer(type.name());
  if (type_renderer != NULL) {
    return (*type_renderer)(this, type, name, ow);
  }

  const google::protobuf::Field* field = NULL;
  string field_name;
  // last_tag set to dummy value that is different from tag.
  uint32 tag = stream_->ReadTag(), last_tag = tag + 1;

  if (include_start_and_end) {
    ow->StartObject(name);
  }
  while (tag != end_tag) {
    if (tag != last_tag) {  // Update field only if tag is changed.
      last_tag = tag;
      field = FindAndVerifyField(type, tag);
      if (field != NULL) {
        field_name = field->json_name();
      }
    }
    if (field == NULL) {
      // If we didn't find a field, skip this unknown tag.
      // TODO(wpoon): Check return boolean value.
      WireFormat::SkipField(stream_, tag, NULL);
      tag = stream_->ReadTag();
      continue;
    }

    if (field->cardinality() ==
        google::protobuf::Field_Cardinality_CARDINALITY_REPEATED) {
      if (IsMap(*field)) {
        ow->StartObject(field_name);
        ASSIGN_OR_RETURN(tag, RenderMap(field, field_name, tag, ow));
        ow->EndObject();
      } else {
        ASSIGN_OR_RETURN(tag, RenderList(field, field_name, tag, ow));
      }
    } else {
      // Render the field.
      RETURN_IF_ERROR(RenderField(field, field_name, ow));
      tag = stream_->ReadTag();
    }
  }
  if (include_start_and_end) {
    ow->EndObject();
  }
  return Status::OK;
}

StatusOr<uint32> ProtoStreamObjectSource::RenderList(
    const google::protobuf::Field* field, StringPiece name, uint32 list_tag,
    ObjectWriter* ow) const {
  uint32 tag_to_return = 0;
  ow->StartList(name);
  if (IsPackable(*field) &&
      list_tag ==
          WireFormatLite::MakeTag(field->number(),
                                  WireFormatLite::WIRETYPE_LENGTH_DELIMITED)) {
    RETURN_IF_ERROR(RenderPacked(field, ow));
    // Since packed fields have a single tag, read another tag from stream to
    // return.
    tag_to_return = stream_->ReadTag();
  } else {
    do {
      RETURN_IF_ERROR(RenderField(field, "", ow));
    } while ((tag_to_return = stream_->ReadTag()) == list_tag);
  }
  ow->EndList();
  return tag_to_return;
}

StatusOr<uint32> ProtoStreamObjectSource::RenderMap(
    const google::protobuf::Field* field, StringPiece name, uint32 list_tag,
    ObjectWriter* ow) const {
  const google::protobuf::Type* field_type =
      typeinfo_->GetTypeByTypeUrl(field->type_url());
  uint32 tag_to_return = 0;
  if (IsPackable(*field) &&
      list_tag ==
          WireFormatLite::MakeTag(field->number(),
                                  WireFormatLite::WIRETYPE_LENGTH_DELIMITED)) {
    RETURN_IF_ERROR(RenderPackedMapEntry(field_type, ow));
    tag_to_return = stream_->ReadTag();
  } else {
    do {
      RETURN_IF_ERROR(RenderMapEntry(field_type, ow));
    } while ((tag_to_return = stream_->ReadTag()) == list_tag);
  }
  return tag_to_return;
}

Status ProtoStreamObjectSource::RenderMapEntry(
    const google::protobuf::Type* type, ObjectWriter* ow) const {
  uint32 buffer32;
  stream_->ReadVarint32(&buffer32);  // message length
  int old_limit = stream_->PushLimit(buffer32);
  string map_key;
  for (uint32 tag = stream_->ReadTag(); tag != 0; tag = stream_->ReadTag()) {
    const google::protobuf::Field* field = FindAndVerifyField(*type, tag);
    if (field == NULL) {
      WireFormat::SkipField(stream_, tag, NULL);
      continue;
    }
    // Map field numbers are key = 1 and value = 2
    if (field->number() == 1) {
      map_key = ReadFieldValueAsString(*field);
    } else if (field->number() == 2) {
      if (map_key.empty()) {
        return Status(util::error::INTERNAL, "Map key must be non-empty");
      }
      // Disable case normalization for map keys as they are just data. We
      // retain them intact.
      ow->DisableCaseNormalizationForNextKey();
      RETURN_IF_ERROR(RenderField(field, map_key, ow));
    }
  }
  stream_->PopLimit(old_limit);

  return Status::OK;
}

Status ProtoStreamObjectSource::RenderPacked(
    const google::protobuf::Field* field, ObjectWriter* ow) const {
  uint32 length;
  stream_->ReadVarint32(&length);
  int old_limit = stream_->PushLimit(length);
  while (stream_->BytesUntilLimit() > 0) {
    RETURN_IF_ERROR(RenderField(field, StringPiece(), ow));
  }
  stream_->PopLimit(old_limit);
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderPackedMapEntry(
    const google::protobuf::Type* type, ObjectWriter* ow) const {
  uint32 length;
  stream_->ReadVarint32(&length);
  int old_limit = stream_->PushLimit(length);
  while (stream_->BytesUntilLimit() > 0) {
    RETURN_IF_ERROR(RenderMapEntry(type, ow));
  }
  stream_->PopLimit(old_limit);
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderTimestamp(
    const ProtoStreamObjectSource* os, const google::protobuf::Type& type,
    StringPiece field_name, ObjectWriter* ow) {
  pair<int64, int32> p = os->ReadSecondsAndNanos(type);
  int64 seconds = p.first;
  int32 nanos = p.second;
  if (seconds > kMaxSeconds || seconds < kMinSeconds) {
    return Status(
        util::error::INTERNAL,
        StrCat("Timestamp seconds exceeds limit for field: ", field_name));
  }

  if (nanos < 0 || nanos >= kNanosPerSecond) {
    return Status(
        util::error::INTERNAL,
        StrCat("Timestamp nanos exceeds limit for field: ", field_name));
  }

  ow->RenderString(field_name,
                   ::google::protobuf::internal::FormatTime(seconds, nanos));

  return Status::OK;
}

Status ProtoStreamObjectSource::RenderDuration(
    const ProtoStreamObjectSource* os, const google::protobuf::Type& type,
    StringPiece field_name, ObjectWriter* ow) {
  pair<int64, int32> p = os->ReadSecondsAndNanos(type);
  int64 seconds = p.first;
  int32 nanos = p.second;
  if (seconds > kMaxSeconds || seconds < kMinSeconds) {
    return Status(
        util::error::INTERNAL,
        StrCat("Duration seconds exceeds limit for field: ", field_name));
  }

  if (nanos <= -kNanosPerSecond || nanos >= kNanosPerSecond) {
    return Status(
        util::error::INTERNAL,
        StrCat("Duration nanos exceeds limit for field: ", field_name));
  }

  string sign = "";
  if (seconds < 0) {
    if (nanos > 0) {
      return Status(util::error::INTERNAL,
                    StrCat("Duration nanos is non-negative, but seconds is "
                           "negative for field: ",
                           field_name));
    }
    sign = "-";
    seconds = -seconds;
    nanos = -nanos;
  } else if (seconds == 0 && nanos < 0) {
    sign = "-";
    nanos = -nanos;
  }
  string formatted_duration = StringPrintf("%s%lld%ss", sign.c_str(), seconds,
                                           FormatNanos(nanos).c_str());
  ow->RenderString(field_name, formatted_duration);
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderDouble(const ProtoStreamObjectSource* os,
                                             const google::protobuf::Type& type,
                                             StringPiece field_name,
                                             ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint64 buffer64 = 0;  // default value of Double wrapper value
  if (tag != 0) {
    os->stream_->ReadLittleEndian64(&buffer64);
    os->stream_->ReadTag();
  }
  ow->RenderDouble(field_name, bit_cast<double>(buffer64));
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderFloat(const ProtoStreamObjectSource* os,
                                            const google::protobuf::Type& type,
                                            StringPiece field_name,
                                            ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint32 buffer32 = 0;  // default value of Float wrapper value
  if (tag != 0) {
    os->stream_->ReadLittleEndian32(&buffer32);
    os->stream_->ReadTag();
  }
  ow->RenderFloat(field_name, bit_cast<float>(buffer32));
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderInt64(const ProtoStreamObjectSource* os,
                                            const google::protobuf::Type& type,
                                            StringPiece field_name,
                                            ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint64 buffer64 = 0;  // default value of Int64 wrapper value
  if (tag != 0) {
    os->stream_->ReadVarint64(&buffer64);
    os->stream_->ReadTag();
  }
  ow->RenderInt64(field_name, bit_cast<int64>(buffer64));
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderUInt64(const ProtoStreamObjectSource* os,
                                             const google::protobuf::Type& type,
                                             StringPiece field_name,
                                             ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint64 buffer64 = 0;  // default value of UInt64 wrapper value
  if (tag != 0) {
    os->stream_->ReadVarint64(&buffer64);
    os->stream_->ReadTag();
  }
  ow->RenderUint64(field_name, bit_cast<uint64>(buffer64));
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderInt32(const ProtoStreamObjectSource* os,
                                            const google::protobuf::Type& type,
                                            StringPiece field_name,
                                            ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint32 buffer32 = 0;  // default value of Int32 wrapper value
  if (tag != 0) {
    os->stream_->ReadVarint32(&buffer32);
    os->stream_->ReadTag();
  }
  ow->RenderInt32(field_name, bit_cast<int32>(buffer32));
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderUInt32(const ProtoStreamObjectSource* os,
                                             const google::protobuf::Type& type,
                                             StringPiece field_name,
                                             ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint32 buffer32 = 0;  // default value of UInt32 wrapper value
  if (tag != 0) {
    os->stream_->ReadVarint32(&buffer32);
    os->stream_->ReadTag();
  }
  ow->RenderUint32(field_name, bit_cast<uint32>(buffer32));
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderBool(const ProtoStreamObjectSource* os,
                                           const google::protobuf::Type& type,
                                           StringPiece field_name,
                                           ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint64 buffer64 = 0;  // results in 'false' value as default, which is the
                        // default value of Bool wrapper
  if (tag != 0) {
    os->stream_->ReadVarint64(&buffer64);
    os->stream_->ReadTag();
  }
  ow->RenderBool(field_name, buffer64 != 0);
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderString(const ProtoStreamObjectSource* os,
                                             const google::protobuf::Type& type,
                                             StringPiece field_name,
                                             ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint32 buffer32;
  string str;  // default value of empty for String wrapper
  if (tag != 0) {
    os->stream_->ReadVarint32(&buffer32);  // string size.
    os->stream_->ReadString(&str, buffer32);
    os->stream_->ReadTag();
  }
  ow->RenderString(field_name, str);
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderBytes(const ProtoStreamObjectSource* os,
                                            const google::protobuf::Type& type,
                                            StringPiece field_name,
                                            ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();
  uint32 buffer32;
  string str;
  if (tag != 0) {
    os->stream_->ReadVarint32(&buffer32);
    os->stream_->ReadString(&str, buffer32);
    os->stream_->ReadTag();
  }
  ow->RenderBytes(field_name, str);
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderStruct(const ProtoStreamObjectSource* os,
                                             const google::protobuf::Type& type,
                                             StringPiece field_name,
                                             ObjectWriter* ow) {
  const google::protobuf::Field* field = NULL;
  uint32 tag = os->stream_->ReadTag();
  ow->StartObject(field_name);
  while (tag != 0) {
    field = os->FindAndVerifyField(type, tag);
    // google.protobuf.Struct has only one field that is a map. Hence we use
    // RenderMap to render that field.
    if (os->IsMap(*field)) {
      ASSIGN_OR_RETURN(tag, os->RenderMap(field, field_name, tag, ow));
    }
  }
  ow->EndObject();
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderStructValue(
    const ProtoStreamObjectSource* os, const google::protobuf::Type& type,
    StringPiece field_name, ObjectWriter* ow) {
  const google::protobuf::Field* field = NULL;
  for (uint32 tag = os->stream_->ReadTag(); tag != 0;
       tag = os->stream_->ReadTag()) {
    field = os->FindAndVerifyField(type, tag);
    if (field == NULL) {
      WireFormat::SkipField(os->stream_, tag, NULL);
      continue;
    }
    RETURN_IF_ERROR(os->RenderField(field, field_name, ow));
  }
  return Status::OK;
}

// TODO(skarvaje): Avoid code duplication of for loops and SkipField logic.
Status ProtoStreamObjectSource::RenderStructListValue(
    const ProtoStreamObjectSource* os, const google::protobuf::Type& type,
    StringPiece field_name, ObjectWriter* ow) {
  uint32 tag = os->stream_->ReadTag();

  // Render empty list when we find empty ListValue message.
  if (tag == 0) {
    ow->StartList(field_name);
    ow->EndList();
    return Status::OK;
  }

  while (tag != 0) {
    const google::protobuf::Field* field = os->FindAndVerifyField(type, tag);
    if (field == NULL) {
      WireFormat::SkipField(os->stream_, tag, NULL);
      tag = os->stream_->ReadTag();
      continue;
    }
    ASSIGN_OR_RETURN(tag, os->RenderList(field, field_name, tag, ow));
  }
  return Status::OK;
}

Status ProtoStreamObjectSource::RenderAny(const ProtoStreamObjectSource* os,
                                          const google::protobuf::Type& type,
                                          StringPiece field_name,
                                          ObjectWriter* ow) {
  // An Any is of the form { string type_url = 1; bytes value = 2; }
  uint32 tag;
  string type_url;
  string value;

  // First read out the type_url and value from the proto stream
  for (tag = os->stream_->ReadTag(); tag != 0; tag = os->stream_->ReadTag()) {
    const google::protobuf::Field* field = os->FindAndVerifyField(type, tag);
    if (field == NULL) {
      WireFormat::SkipField(os->stream_, tag, NULL);
      continue;
    }
    // 'type_url' has field number of 1 and 'value' has field number 2
    // //google/protobuf/any.proto
    if (field->number() == 1) {
      // read type_url
      uint32 type_url_size;
      os->stream_->ReadVarint32(&type_url_size);
      os->stream_->ReadString(&type_url, type_url_size);
    } else if (field->number() == 2) {
      // read value
      uint32 value_size;
      os->stream_->ReadVarint32(&value_size);
      os->stream_->ReadString(&value, value_size);
    }
  }

  // If there is no value, we don't lookup the type, we just output it (if
  // present). If both type and value are empty we output an empty object.
  if (value.empty()) {
    ow->StartObject(field_name);
    if (!type_url.empty()) {
      ow->RenderString("@type", type_url);
    }
    ow->EndObject();
    return util::Status::OK;
  }

  // If there is a value but no type, we cannot render it, so report an error.
  if (type_url.empty()) {
    // TODO(sven): Add an external message once those are ready.
    return util::Status(util::error::INTERNAL,
                        "Invalid Any, the type_url is missing.");
  }

  util::StatusOr<const google::protobuf::Type*> resolved_type =
      os->typeinfo_->ResolveTypeUrl(type_url);

  if (!resolved_type.ok()) {
    // Convert into an internal error, since this means the backend gave us
    // an invalid response (missing or invalid type information).
    return util::Status(util::error::INTERNAL,
                        resolved_type.status().error_message());
  }
  // nested_type cannot be null at this time.
  const google::protobuf::Type* nested_type = resolved_type.ValueOrDie();

  // We know the type so we can render it. Recursively parse the nested stream
  // using a nested ProtoStreamObjectSource using our nested type information.
  google::protobuf::io::ArrayInputStream zero_copy_stream(value.data(), value.size());
  google::protobuf::io::CodedInputStream in_stream(&zero_copy_stream);
  ProtoStreamObjectSource nested_os(&in_stream, os->typeinfo_, *nested_type);

  // We manually call start and end object here so we can inject the @type.
  ow->StartObject(field_name);
  ow->RenderString("@type", type_url);
  util::Status result =
      nested_os.WriteMessage(nested_os.type_, "value", 0, false, ow);
  ow->EndObject();
  return result;
}

Status ProtoStreamObjectSource::RenderFieldMask(
    const ProtoStreamObjectSource* os, const google::protobuf::Type& type,
    StringPiece field_name, ObjectWriter* ow) {
  string combined;
  uint32 buffer32;
  uint32 paths_field_tag = 0;
  for (uint32 tag = os->stream_->ReadTag(); tag != 0;
       tag = os->stream_->ReadTag()) {
    if (paths_field_tag == 0) {
      const google::protobuf::Field* field = os->FindAndVerifyField(type, tag);
      if (field != NULL && field->number() == 1 &&
          field->name() == "paths") {
        paths_field_tag = tag;
      }
    }
    if (paths_field_tag != tag) {
      return util::Status(util::error::INTERNAL,
                          "Invalid FieldMask, unexpected field.");
    }
    string str;
    os->stream_->ReadVarint32(&buffer32);  // string size.
    os->stream_->ReadString(&str, buffer32);
    if (!combined.empty()) {
      combined.append(",");
    }
    combined.append(ConvertFieldMaskPath(str, &ToCamelCase));
  }
  ow->RenderString(field_name, combined);
  return Status::OK;
}


hash_map<string, ProtoStreamObjectSource::TypeRenderer>*
    ProtoStreamObjectSource::renderers_ = NULL;
GOOGLE_PROTOBUF_DECLARE_ONCE(source_renderers_init_);

void ProtoStreamObjectSource::InitRendererMap() {
  renderers_ = new hash_map<string, ProtoStreamObjectSource::TypeRenderer>();
  (*renderers_)["google.protobuf.Timestamp"] =
      &ProtoStreamObjectSource::RenderTimestamp;
  (*renderers_)["google.protobuf.Duration"] =
      &ProtoStreamObjectSource::RenderDuration;
  (*renderers_)["google.protobuf.DoubleValue"] =
      &ProtoStreamObjectSource::RenderDouble;
  (*renderers_)["google.protobuf.FloatValue"] =
      &ProtoStreamObjectSource::RenderFloat;
  (*renderers_)["google.protobuf.Int64Value"] =
      &ProtoStreamObjectSource::RenderInt64;
  (*renderers_)["google.protobuf.UInt64Value"] =
      &ProtoStreamObjectSource::RenderUInt64;
  (*renderers_)["google.protobuf.Int32Value"] =
      &ProtoStreamObjectSource::RenderInt32;
  (*renderers_)["google.protobuf.UInt32Value"] =
      &ProtoStreamObjectSource::RenderUInt32;
  (*renderers_)["google.protobuf.BoolValue"] =
      &ProtoStreamObjectSource::RenderBool;
  (*renderers_)["google.protobuf.StringValue"] =
      &ProtoStreamObjectSource::RenderString;
  (*renderers_)["google.protobuf.BytesValue"] =
      &ProtoStreamObjectSource::RenderBytes;
  (*renderers_)["google.protobuf.Any"] =
      &ProtoStreamObjectSource::RenderAny;
  (*renderers_)["google.protobuf.Struct"] =
      &ProtoStreamObjectSource::RenderStruct;
  (*renderers_)["google.protobuf.Value"] =
      &ProtoStreamObjectSource::RenderStructValue;
  (*renderers_)["google.protobuf.ListValue"] =
      &ProtoStreamObjectSource::RenderStructListValue;
  (*renderers_)["google.protobuf.FieldMask"] =
      &ProtoStreamObjectSource::RenderFieldMask;
  ::google::protobuf::internal::OnShutdown(&DeleteRendererMap);
}

void ProtoStreamObjectSource::DeleteRendererMap() {
  delete ProtoStreamObjectSource::renderers_;
  renderers_ = NULL;
}

// static
ProtoStreamObjectSource::TypeRenderer*
ProtoStreamObjectSource::FindTypeRenderer(const string& type_url) {
  ::google::protobuf::GoogleOnceInit(&source_renderers_init_, &InitRendererMap);
  return FindOrNull(*renderers_, type_url);
}

Status ProtoStreamObjectSource::RenderField(
    const google::protobuf::Field* field, StringPiece field_name,
    ObjectWriter* ow) const {
  switch (field->kind()) {
    case google::protobuf::Field_Kind_TYPE_BOOL: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      ow->RenderBool(field_name, buffer64 != 0);
      break;
    }
    case google::protobuf::Field_Kind_TYPE_INT32: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);
      ow->RenderInt32(field_name, bit_cast<int32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_INT64: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      ow->RenderInt64(field_name, bit_cast<int64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_UINT32: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);
      ow->RenderUint32(field_name, bit_cast<uint32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_UINT64: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      ow->RenderUint64(field_name, bit_cast<uint64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SINT32: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);
      ow->RenderInt32(field_name, WireFormatLite::ZigZagDecode32(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SINT64: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      ow->RenderInt64(field_name, WireFormatLite::ZigZagDecode64(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SFIXED32: {
      uint32 buffer32;
      stream_->ReadLittleEndian32(&buffer32);
      ow->RenderInt32(field_name, bit_cast<int32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SFIXED64: {
      uint64 buffer64;
      stream_->ReadLittleEndian64(&buffer64);
      ow->RenderInt64(field_name, bit_cast<int64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FIXED32: {
      uint32 buffer32;
      stream_->ReadLittleEndian32(&buffer32);
      ow->RenderUint32(field_name, bit_cast<uint32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FIXED64: {
      uint64 buffer64;
      stream_->ReadLittleEndian64(&buffer64);
      ow->RenderUint64(field_name, bit_cast<uint64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FLOAT: {
      uint32 buffer32;
      stream_->ReadLittleEndian32(&buffer32);
      ow->RenderFloat(field_name, bit_cast<float>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_DOUBLE: {
      uint64 buffer64;
      stream_->ReadLittleEndian64(&buffer64);
      ow->RenderDouble(field_name, bit_cast<double>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_ENUM: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);

      // If the field represents an explicit NULL value, render null.
      if (field->type_url() == kStructNullValueTypeUrl) {
        ow->RenderNull(field_name);
        break;
      }

      // Get the nested enum type for this field.
      // TODO(skarvaje): Avoid string manipulation. Find ways to speed this
      // up.
      const google::protobuf::Enum* en =
          typeinfo_->GetEnumByTypeUrl(field->type_url());
      // Lookup the name of the enum, and render that. Skips unknown enums.
      if (en != NULL) {
        const google::protobuf::EnumValue* enum_value =
            FindEnumValueByNumber(*en, buffer32);
        if (enum_value != NULL) {
          ow->RenderString(field_name, enum_value->name());
        }
      } else {
        GOOGLE_LOG(INFO) << "Unknown enum skipped: " << field->type_url();
      }
      break;
    }
    case google::protobuf::Field_Kind_TYPE_STRING: {
      uint32 buffer32;
      string str;
      stream_->ReadVarint32(&buffer32);  // string size.
      stream_->ReadString(&str, buffer32);
      ow->RenderString(field_name, str);
      break;
    }
    case google::protobuf::Field_Kind_TYPE_BYTES: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);  // bytes size.
      string value;
      stream_->ReadString(&value, buffer32);
      ow->RenderBytes(field_name, value);
      break;
    }
    case google::protobuf::Field_Kind_TYPE_MESSAGE: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);  // message length
      int old_limit = stream_->PushLimit(buffer32);
      // Get the nested message type for this field.
      const google::protobuf::Type* type =
          typeinfo_->GetTypeByTypeUrl(field->type_url());
      if (type == NULL) {
        return Status(util::error::INTERNAL,
                      StrCat("Invalid configuration. Could not find the type: ",
                             field->type_url()));
      }

      RETURN_IF_ERROR(WriteMessage(*type, field_name, 0, true, ow));
      if (!stream_->ConsumedEntireMessage()) {
        return Status(util::error::INVALID_ARGUMENT,
                      "Nested protocol message not parsed in its entirety.");
      }
      stream_->PopLimit(old_limit);
      break;
    }
    default:
      break;
  }
  return Status::OK;
}

// TODO(skarvaje): Fix this to avoid code duplication.
const string ProtoStreamObjectSource::ReadFieldValueAsString(
    const google::protobuf::Field& field) const {
  string result;
  switch (field.kind()) {
    case google::protobuf::Field_Kind_TYPE_BOOL: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      result = buffer64 != 0 ? "true" : "false";
      break;
    }
    case google::protobuf::Field_Kind_TYPE_INT32: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);
      result = SimpleItoa(bit_cast<int32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_INT64: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      result = SimpleItoa(bit_cast<int64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_UINT32: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);
      result = SimpleItoa(bit_cast<uint32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_UINT64: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      result = SimpleItoa(bit_cast<uint64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SINT32: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);
      result = SimpleItoa(WireFormatLite::ZigZagDecode32(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SINT64: {
      uint64 buffer64;
      stream_->ReadVarint64(&buffer64);
      result = SimpleItoa(WireFormatLite::ZigZagDecode64(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SFIXED32: {
      uint32 buffer32;
      stream_->ReadLittleEndian32(&buffer32);
      result = SimpleItoa(bit_cast<int32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SFIXED64: {
      uint64 buffer64;
      stream_->ReadLittleEndian64(&buffer64);
      result = SimpleItoa(bit_cast<int64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FIXED32: {
      uint32 buffer32;
      stream_->ReadLittleEndian32(&buffer32);
      result = SimpleItoa(bit_cast<uint32>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FIXED64: {
      uint64 buffer64;
      stream_->ReadLittleEndian64(&buffer64);
      result = SimpleItoa(bit_cast<uint64>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FLOAT: {
      uint32 buffer32;
      stream_->ReadLittleEndian32(&buffer32);
      result = SimpleFtoa(bit_cast<float>(buffer32));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_DOUBLE: {
      uint64 buffer64;
      stream_->ReadLittleEndian64(&buffer64);
      result = SimpleDtoa(bit_cast<double>(buffer64));
      break;
    }
    case google::protobuf::Field_Kind_TYPE_ENUM: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);
      // Get the nested enum type for this field.
      // TODO(skarvaje): Avoid string manipulation. Find ways to speed this
      // up.
      const google::protobuf::Enum* en =
          typeinfo_->GetEnumByTypeUrl(field.type_url());
      // Lookup the name of the enum, and render that. Skips unknown enums.
      if (en != NULL) {
        const google::protobuf::EnumValue* enum_value =
            FindEnumValueByNumber(*en, buffer32);
        if (enum_value != NULL) {
          result = enum_value->name();
        }
      }
      break;
    }
    case google::protobuf::Field_Kind_TYPE_STRING: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);  // string size.
      stream_->ReadString(&result, buffer32);
      break;
    }
    case google::protobuf::Field_Kind_TYPE_BYTES: {
      uint32 buffer32;
      stream_->ReadVarint32(&buffer32);  // bytes size.
      stream_->ReadString(&result, buffer32);
      break;
    }
    default:
      break;
  }
  return result;
}

// Field is a map if it is a repeated message and it has an option "map_type".
// TODO(skarvaje): Consider pre-computing the IsMap() into Field directly.
bool ProtoStreamObjectSource::IsMap(
    const google::protobuf::Field& field) const {
  const google::protobuf::Type* field_type =
      typeinfo_->GetTypeByTypeUrl(field.type_url());

  // TODO(xiaofeng): Unify option names.
  return field.kind() == google::protobuf::Field_Kind_TYPE_MESSAGE &&
         (GetBoolOptionOrDefault(field_type->options(),
                                 "google.protobuf.MessageOptions.map_entry", false) ||
          GetBoolOptionOrDefault(field_type->options(), "map_entry", false));
}

std::pair<int64, int32> ProtoStreamObjectSource::ReadSecondsAndNanos(
    const google::protobuf::Type& type) const {
  uint64 seconds = 0;
  uint32 nanos = 0;
  uint32 tag = 0;
  int64 signed_seconds = 0;
  int32 signed_nanos = 0;

  for (tag = stream_->ReadTag(); tag != 0; tag = stream_->ReadTag()) {
    const google::protobuf::Field* field = FindAndVerifyField(type, tag);
    if (field == NULL) {
      WireFormat::SkipField(stream_, tag, NULL);
      continue;
    }
    // 'seconds' has field number of 1 and 'nanos' has field number 2
    // //google/protobuf/timestamp.proto & duration.proto
    if (field->number() == 1) {
      // read seconds
      stream_->ReadVarint64(&seconds);
      signed_seconds = bit_cast<int64>(seconds);
    } else if (field->number() == 2) {
      // read nanos
      stream_->ReadVarint32(&nanos);
      signed_nanos = bit_cast<int32>(nanos);
    }
  }
  return std::pair<int64, int32>(signed_seconds, signed_nanos);
}

namespace {
// TODO(skarvaje): Speed this up by not doing a linear scan.
const google::protobuf::Field* FindFieldByNumber(
    const google::protobuf::Type& type, int number) {
  for (int i = 0; i < type.fields_size(); ++i) {
    if (type.fields(i).number() == number) {
      return &type.fields(i);
    }
  }
  return NULL;
}

// TODO(skarvaje): Replace FieldDescriptor by implementing IsTypePackable()
// using tech Field.
bool IsPackable(const google::protobuf::Field& field) {
  return field.cardinality() ==
             google::protobuf::Field_Cardinality_CARDINALITY_REPEATED &&
         google::protobuf::FieldDescriptor::IsTypePackable(
             static_cast<google::protobuf::FieldDescriptor::Type>(field.kind()));
}

// TODO(skarvaje): Speed this up by not doing a linear scan.
const google::protobuf::EnumValue* FindEnumValueByNumber(
    const google::protobuf::Enum& tech_enum, int number) {
  for (int i = 0; i < tech_enum.enumvalue_size(); ++i) {
    const google::protobuf::EnumValue& ev = tech_enum.enumvalue(i);
    if (ev.number() == number) {
      return &ev;
    }
  }
  return NULL;
}

// TODO(skarvaje): Look into optimizing this by not doing computation on
// double.
const string FormatNanos(uint32 nanos) {
  const char* format =
      (nanos % 1000 != 0) ? "%.9f" : (nanos % 1000000 != 0) ? "%.6f" : "%.3f";
  string formatted =
      StringPrintf(format, static_cast<double>(nanos) / kNanosPerSecond);
  // remove the leading 0 before decimal.
  return formatted.substr(1);
}
}  // namespace

}  // namespace converter
}  // namespace util
}  // namespace protobuf
}  // namespace google
