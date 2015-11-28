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

#include <google/protobuf/util/internal/protostream_objectwriter.h>

#include <functional>
#include <stack>

#include <google/protobuf/stubs/once.h>
#include <google/protobuf/stubs/time.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/util/internal/field_mask_utility.h>
#include <google/protobuf/util/internal/object_location_tracker.h>
#include <google/protobuf/util/internal/constants.h>
#include <google/protobuf/util/internal/utility.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/map_util.h>
#include <google/protobuf/stubs/statusor.h>


namespace google {
namespace protobuf {
namespace util {
namespace converter {

using google::protobuf::internal::WireFormatLite;
using google::protobuf::io::CodedOutputStream;
using util::error::INVALID_ARGUMENT;
using util::Status;
using util::StatusOr;


ProtoStreamObjectWriter::ProtoStreamObjectWriter(
    TypeResolver* type_resolver, const google::protobuf::Type& type,
    strings::ByteSink* output, ErrorListener* listener)
    : master_type_(type),
      typeinfo_(TypeInfo::NewTypeInfo(type_resolver)),
      own_typeinfo_(true),
      done_(false),
      element_(NULL),
      size_insert_(),
      output_(output),
      buffer_(),
      adapter_(&buffer_),
      stream_(new CodedOutputStream(&adapter_)),
      listener_(listener),
      invalid_depth_(0),
      tracker_(new ObjectLocationTracker()) {}

ProtoStreamObjectWriter::ProtoStreamObjectWriter(
    const TypeInfo* typeinfo, const google::protobuf::Type& type,
    strings::ByteSink* output, ErrorListener* listener)
    : master_type_(type),
      typeinfo_(typeinfo),
      own_typeinfo_(false),
      done_(false),
      element_(NULL),
      size_insert_(),
      output_(output),
      buffer_(),
      adapter_(&buffer_),
      stream_(new CodedOutputStream(&adapter_)),
      listener_(listener),
      invalid_depth_(0),
      tracker_(new ObjectLocationTracker()) {}

ProtoStreamObjectWriter::~ProtoStreamObjectWriter() {
  if (own_typeinfo_) {
    delete typeinfo_;
  }
  if (element_ == NULL) return;
  // Cleanup explicitly in order to avoid destructor stack overflow when input
  // is deeply nested.
  // Cast to BaseElement to avoid doing additional checks (like missing fields)
  // during pop().
  google::protobuf::scoped_ptr<BaseElement> element(
      static_cast<BaseElement*>(element_.get())->pop<BaseElement>());
  while (element != NULL) {
    element.reset(element->pop<BaseElement>());
  }
}

namespace {

// Writes an INT32 field, including tag to the stream.
inline Status WriteInt32(int field_number, const DataPiece& data,
                         CodedOutputStream* stream) {
  StatusOr<int32> i32 = data.ToInt32();
  if (i32.ok()) {
    WireFormatLite::WriteInt32(field_number, i32.ValueOrDie(), stream);
  }
  return i32.status();
}

// writes an SFIXED32 field, including tag, to the stream.
inline Status WriteSFixed32(int field_number, const DataPiece& data,
                            CodedOutputStream* stream) {
  StatusOr<int32> i32 = data.ToInt32();
  if (i32.ok()) {
    WireFormatLite::WriteSFixed32(field_number, i32.ValueOrDie(), stream);
  }
  return i32.status();
}

// Writes an SINT32 field, including tag, to the stream.
inline Status WriteSInt32(int field_number, const DataPiece& data,
                          CodedOutputStream* stream) {
  StatusOr<int32> i32 = data.ToInt32();
  if (i32.ok()) {
    WireFormatLite::WriteSInt32(field_number, i32.ValueOrDie(), stream);
  }
  return i32.status();
}

// Writes a FIXED32 field, including tag, to the stream.
inline Status WriteFixed32(int field_number, const DataPiece& data,
                           CodedOutputStream* stream) {
  StatusOr<uint32> u32 = data.ToUint32();
  if (u32.ok()) {
    WireFormatLite::WriteFixed32(field_number, u32.ValueOrDie(), stream);
  }
  return u32.status();
}

// Writes a UINT32 field, including tag, to the stream.
inline Status WriteUInt32(int field_number, const DataPiece& data,
                          CodedOutputStream* stream) {
  StatusOr<uint32> u32 = data.ToUint32();
  if (u32.ok()) {
    WireFormatLite::WriteUInt32(field_number, u32.ValueOrDie(), stream);
  }
  return u32.status();
}

// Writes an INT64 field, including tag, to the stream.
inline Status WriteInt64(int field_number, const DataPiece& data,
                         CodedOutputStream* stream) {
  StatusOr<int64> i64 = data.ToInt64();
  if (i64.ok()) {
    WireFormatLite::WriteInt64(field_number, i64.ValueOrDie(), stream);
  }
  return i64.status();
}

// Writes an SFIXED64 field, including tag, to the stream.
inline Status WriteSFixed64(int field_number, const DataPiece& data,
                            CodedOutputStream* stream) {
  StatusOr<int64> i64 = data.ToInt64();
  if (i64.ok()) {
    WireFormatLite::WriteSFixed64(field_number, i64.ValueOrDie(), stream);
  }
  return i64.status();
}

// Writes an SINT64 field, including tag, to the stream.
inline Status WriteSInt64(int field_number, const DataPiece& data,
                          CodedOutputStream* stream) {
  StatusOr<int64> i64 = data.ToInt64();
  if (i64.ok()) {
    WireFormatLite::WriteSInt64(field_number, i64.ValueOrDie(), stream);
  }
  return i64.status();
}

// Writes a FIXED64 field, including tag, to the stream.
inline Status WriteFixed64(int field_number, const DataPiece& data,
                           CodedOutputStream* stream) {
  StatusOr<uint64> u64 = data.ToUint64();
  if (u64.ok()) {
    WireFormatLite::WriteFixed64(field_number, u64.ValueOrDie(), stream);
  }
  return u64.status();
}

// Writes a UINT64 field, including tag, to the stream.
inline Status WriteUInt64(int field_number, const DataPiece& data,
                          CodedOutputStream* stream) {
  StatusOr<uint64> u64 = data.ToUint64();
  if (u64.ok()) {
    WireFormatLite::WriteUInt64(field_number, u64.ValueOrDie(), stream);
  }
  return u64.status();
}

// Writes a DOUBLE field, including tag, to the stream.
inline Status WriteDouble(int field_number, const DataPiece& data,
                          CodedOutputStream* stream) {
  StatusOr<double> d = data.ToDouble();
  if (d.ok()) {
    WireFormatLite::WriteDouble(field_number, d.ValueOrDie(), stream);
  }
  return d.status();
}

// Writes a FLOAT field, including tag, to the stream.
inline Status WriteFloat(int field_number, const DataPiece& data,
                         CodedOutputStream* stream) {
  StatusOr<float> f = data.ToFloat();
  if (f.ok()) {
    WireFormatLite::WriteFloat(field_number, f.ValueOrDie(), stream);
  }
  return f.status();
}

// Writes a BOOL field, including tag, to the stream.
inline Status WriteBool(int field_number, const DataPiece& data,
                        CodedOutputStream* stream) {
  StatusOr<bool> b = data.ToBool();
  if (b.ok()) {
    WireFormatLite::WriteBool(field_number, b.ValueOrDie(), stream);
  }
  return b.status();
}

// Writes a BYTES field, including tag, to the stream.
inline Status WriteBytes(int field_number, const DataPiece& data,
                         CodedOutputStream* stream) {
  StatusOr<string> c = data.ToBytes();
  if (c.ok()) {
    WireFormatLite::WriteBytes(field_number, c.ValueOrDie(), stream);
  }
  return c.status();
}

// Writes a STRING field, including tag, to the stream.
inline Status WriteString(int field_number, const DataPiece& data,
                          CodedOutputStream* stream) {
  StatusOr<string> s = data.ToString();
  if (s.ok()) {
    WireFormatLite::WriteString(field_number, s.ValueOrDie(), stream);
  }
  return s.status();
}

// Writes an ENUM field, including tag, to the stream.
inline Status WriteEnum(int field_number, const DataPiece& data,
                        const google::protobuf::Enum* enum_type,
                        CodedOutputStream* stream) {
  StatusOr<int> e = data.ToEnum(enum_type);
  if (e.ok()) {
    WireFormatLite::WriteEnum(field_number, e.ValueOrDie(), stream);
  }
  return e.status();
}

// Given a google::protobuf::Type, returns the set of all required fields.
std::set<const google::protobuf::Field*> GetRequiredFields(
    const google::protobuf::Type& type) {
  std::set<const google::protobuf::Field*> required;
  for (int i = 0; i < type.fields_size(); i++) {
    const google::protobuf::Field& field = type.fields(i);
    if (field.cardinality() ==
        google::protobuf::Field_Cardinality_CARDINALITY_REQUIRED) {
      required.insert(&field);
    }
  }
  return required;
}

// Utility method to split a string representation of Timestamp or Duration and
// return the parts.
void SplitSecondsAndNanos(StringPiece input, StringPiece* seconds,
                          StringPiece* nanos) {
  size_t idx = input.rfind('.');
  if (idx != string::npos) {
    *seconds = input.substr(0, idx);
    *nanos = input.substr(idx + 1);
  } else {
    *seconds = input;
    *nanos = StringPiece();
  }
}

}  // namespace

ProtoStreamObjectWriter::AnyWriter::AnyWriter(ProtoStreamObjectWriter* parent)
    : parent_(parent),
      ow_(),
      invalid_(false),
      data_(),
      output_(&data_),
      depth_(0),
      has_injected_value_message_(false) {}

ProtoStreamObjectWriter::AnyWriter::~AnyWriter() {}

void ProtoStreamObjectWriter::AnyWriter::StartObject(StringPiece name) {
  ++depth_;
  // If an object writer is absent, that means we have not called StartAny()
  // before reaching here. This is an invalid state. StartAny() gets called
  // whenever we see an "@type" being rendered (see AnyWriter::RenderDataPiece).
  if (ow_ == NULL) {
    // Make sure we are not already in an invalid state. This avoids making
    // multiple unnecessary InvalidValue calls.
    if (!invalid_) {
      parent_->InvalidValue("Any",
                            StrCat("Missing or invalid @type for any field in ",
                                   parent_->master_type_.name()));
      invalid_ = true;
    }
  } else if (!has_injected_value_message_ || depth_ != 1 || name != "value") {
    // We don't propagate to ow_ StartObject("value") calls for nested Anys or
    // Struct at depth 1 as they are nested one level deep with an injected
    // "value" field.
    ow_->StartObject(name);
  }
}

bool ProtoStreamObjectWriter::AnyWriter::EndObject() {
  --depth_;
  // As long as depth_ >= 0, we know we haven't reached the end of Any.
  // Propagate these EndObject() calls to the contained ow_.  If we are in a
  // nested Any or Struct type, ignore the second to last EndObject call (depth_
  // == -1)
  if (ow_ != NULL && (!has_injected_value_message_ || depth_ >= 0)) {
    ow_->EndObject();
  }
  // A negative depth_ implies that we have reached the end of Any
  // object. Now we write out its contents.
  if (depth_ < 0) {
    WriteAny();
    return false;
  }
  return true;
}

void ProtoStreamObjectWriter::AnyWriter::StartList(StringPiece name) {
  ++depth_;
  // We expect ow_ to be present as this call only makes sense inside an Any.
  if (ow_ == NULL) {
    if (!invalid_) {
      parent_->InvalidValue("Any",
                            StrCat("Missing or invalid @type for any field in ",
                                   parent_->master_type_.name()));
      invalid_ = true;
    }
  } else {
    ow_->StartList(name);
  }
}

void ProtoStreamObjectWriter::AnyWriter::EndList() {
  --depth_;
  if (depth_ < 0) {
    GOOGLE_LOG(DFATAL) << "Mismatched EndList found, should not be possible";
    depth_ = 0;
  }
  // We don't write an error on the close, only on the open
  if (ow_ != NULL) {
    ow_->EndList();
  }
}

void ProtoStreamObjectWriter::AnyWriter::RenderDataPiece(
    StringPiece name, const DataPiece& value) {
  // Start an Any only at depth_ 0. Other RenderDataPiece calls with "@type"
  // should go to the contained ow_ as they indicate nested Anys.
  if (depth_ == 0 && ow_ == NULL && name == "@type") {
    StartAny(value);
  } else if (ow_ == NULL) {
    if (!invalid_) {
      parent_->InvalidValue("Any",
                            StrCat("Missing or invalid @type for any field in ",
                                   parent_->master_type_.name()));
      invalid_ = true;
    }
  } else {
    // Check to see if the data needs to be rendered with well-known-type
    // renderer.
    const TypeRenderer* type_renderer =
        FindTypeRenderer(GetFullTypeWithUrl(ow_->master_type_.name()));
    if (type_renderer) {
      Status status = (*type_renderer)(ow_.get(), value);
      if (!status.ok()) ow_->InvalidValue("Any", status.error_message());
    } else {
      ow_->RenderDataPiece(name, value);
    }
  }
}

void ProtoStreamObjectWriter::AnyWriter::StartAny(const DataPiece& value) {
  // Figure out the type url. This is a copy-paste from WriteString but we also
  // need the value, so we can't just call through to that.
  if (value.type() == DataPiece::TYPE_STRING) {
    type_url_ = value.str().ToString();
  } else {
    StatusOr<string> s = value.ToString();
    if (!s.ok()) {
      parent_->InvalidValue("String", s.status().error_message());
      invalid_ = true;
      return;
    }
    type_url_ = s.ValueOrDie();
  }
  // Resolve the type url, and report an error if we failed to resolve it.
  StatusOr<const google::protobuf::Type*> resolved_type =
      parent_->typeinfo_->ResolveTypeUrl(type_url_);
  if (!resolved_type.ok()) {
    parent_->InvalidValue("Any", resolved_type.status().error_message());
    invalid_ = true;
    return;
  }
  // At this point, type is never null.
  const google::protobuf::Type* type = resolved_type.ValueOrDie();

  // If this is the case of an Any in an Any or Struct in an Any, we need to
  // expect a StartObject call with "value" while we're at depth_ 0, which we
  // should ignore (not propagate to our nested object writer). We also need to
  // ignore the second-to-last EndObject call, and not propagate that either.
  if (type->name() == kAnyType || type->name() == kStructType) {
    has_injected_value_message_ = true;
  }

  // Create our object writer and initialize it with the first StartObject
  // call.
  ow_.reset(new ProtoStreamObjectWriter(parent_->typeinfo_, *type, &output_,
                                        parent_->listener_));
  ow_->StartObject("");
}

void ProtoStreamObjectWriter::AnyWriter::WriteAny() {
  if (ow_ == NULL) {
    // If we had no object writer, we never got any content, so just return
    // immediately, which is equivalent to writing an empty Any.
    return;
  }
  // Render the type_url and value fields directly to the stream.
  // type_url has tag 1 and value has tag 2.
  WireFormatLite::WriteString(1, type_url_, parent_->stream_.get());
  if (!data_.empty()) {
    WireFormatLite::WriteBytes(2, data_, parent_->stream_.get());
  }
}

ProtoStreamObjectWriter::ProtoElement::ProtoElement(
    const TypeInfo* typeinfo, const google::protobuf::Type& type,
    ProtoStreamObjectWriter* enclosing)
    : BaseElement(NULL),
      ow_(enclosing),
      any_(),
      field_(NULL),
      typeinfo_(typeinfo),
      type_(type),
      required_fields_(GetRequiredFields(type)),
      is_repeated_type_(false),
      size_index_(-1),
      array_index_(-1),
      element_type_(GetElementType(type_)) {
  if (element_type_ == ANY) {
    any_.reset(new AnyWriter(ow_));
  }
}

ProtoStreamObjectWriter::ProtoElement::ProtoElement(
    ProtoStreamObjectWriter::ProtoElement* parent,
    const google::protobuf::Field* field, const google::protobuf::Type& type,
    ElementType element_type)
    : BaseElement(parent),
      ow_(this->parent()->ow_),
      any_(),
      field_(field),
      typeinfo_(this->parent()->typeinfo_),
      type_(type),
      is_repeated_type_(element_type == ProtoElement::LIST ||
                        element_type == ProtoElement::STRUCT_LIST ||
                        element_type == ProtoElement::MAP ||
                        element_type == ProtoElement::STRUCT_MAP),
      size_index_(!is_repeated_type_ &&
                          field->kind() ==
                              google::protobuf::Field_Kind_TYPE_MESSAGE
                      ? ow_->size_insert_.size()
                      : -1),
      array_index_(is_repeated_type_ ? 0 : -1),
      element_type_(element_type) {
  if (!is_repeated_type_) {
    if (field->cardinality() ==
        google::protobuf::Field_Cardinality_CARDINALITY_REPEATED) {
      // Update array_index_ if it is an explicit list.
      if (this->parent()->array_index_ >= 0) this->parent()->array_index_++;
    } else {
      this->parent()->RegisterField(field);
    }
    if (field->kind() == google::protobuf::Field_Kind_TYPE_MESSAGE) {
      required_fields_ = GetRequiredFields(type_);
      int start_pos = ow_->stream_->ByteCount();
      // length of serialized message is the final buffer position minus
      // starting buffer position, plus length adjustments for size fields
      // of any nested messages. We start with -start_pos here, so we only
      // need to add the final buffer position to it at the end.
      SizeInfo info = {start_pos, -start_pos};
      ow_->size_insert_.push_back(info);
    }
  }
  if (element_type == ANY) {
    any_.reset(new AnyWriter(ow_));
  }
}

ProtoStreamObjectWriter::ProtoElement*
ProtoStreamObjectWriter::ProtoElement::pop() {
  // Calls the registered error listener for any required field(s) not yet
  // seen.
  for (set<const google::protobuf::Field*>::iterator it =
           required_fields_.begin();
       it != required_fields_.end(); ++it) {
    ow_->MissingField((*it)->name());
  }
  // Computes the total number of proto bytes used by a message, also adjusts
  // the size of all parent messages by the length of this size field.
  // If size_index_ < 0, this is not a message, so no size field is added.
  if (size_index_ >= 0) {
    // Add the final buffer position to compute the total length of this
    // serialized message. The stored value (before this addition) already
    // contains the total length of the size fields of all nested messages
    // minus the initial buffer position.
    ow_->size_insert_[size_index_].size += ow_->stream_->ByteCount();
    // Calculate the length required to serialize the size field of the
    // message, and propagate this additional size information upward to
    // all enclosing messages.
    int size = ow_->size_insert_[size_index_].size;
    int length = CodedOutputStream::VarintSize32(size);
    for (ProtoElement* e = parent(); e != NULL; e = e->parent()) {
      // Only nested messages have size field, lists do not have size field.
      if (e->size_index_ >= 0) {
        ow_->size_insert_[e->size_index_].size += length;
      }
    }
  }
  return BaseElement::pop<ProtoElement>();
}

void ProtoStreamObjectWriter::ProtoElement::RegisterField(
    const google::protobuf::Field* field) {
  if (!required_fields_.empty() &&
      field->cardinality() ==
          google::protobuf::Field_Cardinality_CARDINALITY_REQUIRED) {
    required_fields_.erase(field);
  }
}

string ProtoStreamObjectWriter::ProtoElement::ToString() const {
  if (parent() == NULL) return "";
  string loc = parent()->ToString();
  if (field_->cardinality() !=
          google::protobuf::Field_Cardinality_CARDINALITY_REPEATED ||
      parent()->field_ != field_) {
    string name = field_->name();
    int i = 0;
    while (i < name.size() && (ascii_isalnum(name[i]) || name[i] == '_')) ++i;
    if (i > 0 && i == name.size()) {  // safe field name
      if (loc.empty()) {
        loc = name;
      } else {
        StrAppend(&loc, ".", name);
      }
    } else {
      StrAppend(&loc, "[\"", CEscape(name), "\"]");
    }
  }
  if (field_->cardinality() ==
          google::protobuf::Field_Cardinality_CARDINALITY_REPEATED &&
      array_index_ > 0) {
    StrAppend(&loc, "[", array_index_ - 1, "]");
  }
  return loc.empty() ? "." : loc;
}

bool ProtoStreamObjectWriter::ProtoElement::OneofIndexTaken(int32 index) {
  return ContainsKey(oneof_indices_, index);
}

void ProtoStreamObjectWriter::ProtoElement::TakeOneofIndex(int32 index) {
  InsertIfNotPresent(&oneof_indices_, index);
}

bool ProtoStreamObjectWriter::ProtoElement::InsertMapKeyIfNotPresent(
    StringPiece map_key) {
  return InsertIfNotPresent(&map_keys_, map_key.ToString());
}

inline void ProtoStreamObjectWriter::InvalidName(StringPiece unknown_name,
                                                 StringPiece message) {
  listener_->InvalidName(location(), ToSnakeCase(unknown_name), message);
}

inline void ProtoStreamObjectWriter::InvalidValue(StringPiece type_name,
                                                  StringPiece value) {
  listener_->InvalidValue(location(), type_name, value);
}

inline void ProtoStreamObjectWriter::MissingField(StringPiece missing_name) {
  listener_->MissingField(location(), missing_name);
}

ProtoStreamObjectWriter* ProtoStreamObjectWriter::StartObject(
    StringPiece name) {
  // Starting the root message. Create the root ProtoElement and return.
  if (element_ == NULL) {
    if (!name.empty()) {
      InvalidName(name, "Root element should not be named.");
    }
    element_.reset(new ProtoElement(typeinfo_, master_type_, this));

    // If master type is a special type that needs extra values to be written to
    // stream, we write those values.
    if (master_type_.name() == kStructType) {
      StartStruct(NULL);
    } else if (master_type_.name() == kStructValueType) {
      // We got a StartObject call with google.protobuf.Value field. This means
      // we are starting an object within google.protobuf.Value type. The only
      // object within that type is a struct type. So start a struct.
      const google::protobuf::Field* field = StartStructValueInStruct(NULL);
      StartStruct(field);
    }
    return this;
  }

  const google::protobuf::Field* field = NULL;
  if (element_ != NULL && element_->IsAny()) {
    element_->any()->StartObject(name);
    return this;
  } else if (element_ != NULL &&
             (element_->IsMap() || element_->IsStructMap())) {
    if (!ValidMapKey(name)) {
      ++invalid_depth_;
      return this;
    }

    field = StartMapEntry(name);
    if (element_->IsStructMapEntry()) {
      // If the top element is a map entry, this means we are starting another
      // struct within a struct.
      field = StartStructValueInStruct(field);
    }
  } else if (element_ != NULL && element_->IsStructList()) {
    // If the top element is a list, then we are starting a list field within a
    // struct.
    field = Lookup(name);
    field = StartStructValueInStruct(field);
  } else {
    field = BeginNamed(name, false);
  }
  if (field == NULL) {
    return this;
  }

  const google::protobuf::Type* type = LookupType(field);
  if (type == NULL) {
    ++invalid_depth_;
    InvalidName(name,
                StrCat("Missing descriptor for field: ", field->type_url()));
    return this;
  }

  // Check to see if this field is a oneof and that no oneof in that group has
  // already been set.
  if (!ValidOneof(*field, name)) {
    ++invalid_depth_;
    return this;
  }

  if (field->type_url() == GetFullTypeWithUrl(kStructType)) {
    // Start a struct object.
    StartStruct(field);
  } else if (field->type_url() == GetFullTypeWithUrl(kStructValueType)) {
    // We got a StartObject call with google.protobuf.Value field. This means we
    // are starting an object within google.protobuf.Value type. The only object
    // within that type is a struct type. So start a struct.
    field = StartStructValueInStruct(field);
    StartStruct(field);
  } else if (field->type_url() == GetFullTypeWithUrl(kAnyType)) {
    // Begin an Any. We can't do the real work till we get the @type field.
    WriteTag(*field);
    element_.reset(
        new ProtoElement(element_.release(), field, *type, ProtoElement::ANY));
  } else if (IsMap(*field)) {
    // Begin a map.
    // A map is triggered by a StartObject() call if the current field has a map
    // type. Map values are written to proto in a manner detailed in comments
    // above StartMapEntry() function.
    element_.reset(
        new ProtoElement(element_.release(), field, *type, ProtoElement::MAP));
  } else {
      WriteTag(*field);
    element_.reset(new ProtoElement(element_.release(), field, *type,
                                    ProtoElement::MESSAGE));
  }
  return this;
}

// Proto3 maps are represented on the wire as a message with
// "key" and a "value".
//
// For example, the syntax:
// map<key_type, value_type> map_field = N;
//
// is represented as:
// message MapFieldEntry {
//   option map_entry = true;   // marks the map construct in the descriptor
//
//   key_type key = 1;
//   value_type value = 2;
// }
// repeated MapFieldEntry map_field = N;
//
// See go/proto3-maps for more information.
const google::protobuf::Field* ProtoStreamObjectWriter::StartMapEntry(
    StringPiece name) {
  // top of stack is already a map field
  const google::protobuf::Field* field = element_->field();
  const google::protobuf::Type& type = element_->type();
  // If we come from a regular map, use MAP_ENTRY or if we come from a struct,
  // use STRUCT_MAP_ENTRY. These values are used later in StartObject/StartList
  // or RenderDataPiece for making appropriate decisions.
  ProtoElement::ElementType element_type = element_->IsStructMap()
                                               ? ProtoElement::STRUCT_MAP_ENTRY
                                               : ProtoElement::MAP_ENTRY;
  WriteTag(*field);
  element_.reset(
      new ProtoElement(element_.release(), field, type, element_type));
  RenderDataPiece("key", DataPiece(name));
  return BeginNamed("value", false);
}

// Starts a google.protobuf.Struct.
// 'field' represents a field in a message of type google.protobuf.Struct.  A
// struct contains a map with name 'fields'. This function starts this map as
// well.
// When 'field' is NULL, it means that the top level message is of struct
// type.
void ProtoStreamObjectWriter::StartStruct(
    const google::protobuf::Field* field) {
  const google::protobuf::Type* type = NULL;
  if (field) {
    type = LookupType(field);
    WriteTag(*field);
    element_.reset(new ProtoElement(element_.release(), field, *type,
                                    ProtoElement::STRUCT));
  }
  const google::protobuf::Field* struct_field = BeginNamed("fields", false);

  if (!struct_field) {
    // It is a programmatic error if this happens. Log an error.
    GOOGLE_LOG(ERROR) << "Invalid internal state. Cannot find 'fields' within "
               << (field ? field->type_url() : "google.protobuf.Struct");
    return;
  }

  type = LookupType(struct_field);
  element_.reset(new ProtoElement(element_.release(), struct_field, *type,
                                  ProtoElement::STRUCT_MAP));
}

// Starts a "struct_value" within struct.proto's google.protobuf.Value type.
// 'field' should be of the type google.protobuf.Value.
// Returns the field identifying "struct_value" within the given field.
//
// If field is NULL, then we are starting struct_value at the top-level, in
// this case skip writing any tag information for the passed field.
const google::protobuf::Field*
ProtoStreamObjectWriter::StartStructValueInStruct(
    const google::protobuf::Field* field) {
  if (field) {
    const google::protobuf::Type* type = LookupType(field);
    WriteTag(*field);
    element_.reset(new ProtoElement(element_.release(), field, *type,
                                    ProtoElement::STRUCT_VALUE));
  }
  return BeginNamed("struct_value", false);
}

// Starts a "list_value" within struct.proto's google.protobuf.Value type.
// 'field' should be of the type google.protobuf.Value.
// Returns the field identifying "list_value" within the given field.
//
// If field is NULL, then we are starting list_value at the top-level, in
// this case skip writing any tag information for the passed field.
const google::protobuf::Field* ProtoStreamObjectWriter::StartListValueInStruct(
    const google::protobuf::Field* field) {
  if (field) {
    const google::protobuf::Type* type = LookupType(field);
    WriteTag(*field);
    element_.reset(new ProtoElement(element_.release(), field, *type,
                                    ProtoElement::STRUCT_VALUE));
  }
  const google::protobuf::Field* list_value = BeginNamed("list_value", false);

  if (!list_value) {
    // It is a programmatic error if this happens. Log an error.
    GOOGLE_LOG(ERROR) << "Invalid internal state. Cannot find 'list_value' within "
               << (field ? field->type_url() : "google.protobuf.Value");
    return field;
  }

  return StartRepeatedValuesInListValue(list_value);
}

// Starts the repeated "values" field in struct.proto's
// google.protobuf.ListValue type. 'field' should be of type
// google.protobuf.ListValue.
//
// If field is NULL, then we are starting ListValue at the top-level, in
// this case skip writing any tag information for the passed field.
const google::protobuf::Field*
ProtoStreamObjectWriter::StartRepeatedValuesInListValue(
    const google::protobuf::Field* field) {
  if (field) {
    const google::protobuf::Type* type = LookupType(field);
    WriteTag(*field);
    element_.reset(new ProtoElement(element_.release(), field, *type,
                                    ProtoElement::STRUCT_LIST_VALUE));
  }
  return BeginNamed("values", true);
}

void ProtoStreamObjectWriter::SkipElements() {
  if (element_ == NULL) return;

  ProtoElement::ElementType element_type = element_->element_type();
  while (element_type == ProtoElement::STRUCT ||
         element_type == ProtoElement::STRUCT_LIST_VALUE ||
         element_type == ProtoElement::STRUCT_VALUE ||
         element_type == ProtoElement::STRUCT_MAP_ENTRY ||
         element_type == ProtoElement::MAP_ENTRY) {
    element_.reset(element_->pop());
    element_type =
        element_ != NULL ? element_->element_type() : ProtoElement::MESSAGE;
  }
}

ProtoStreamObjectWriter* ProtoStreamObjectWriter::EndObject() {
  if (invalid_depth_ > 0) {
    --invalid_depth_;
    return this;
  }
  if (element_ != NULL && element_->IsAny()) {
    if (element_->any()->EndObject()) {
      return this;
    }
  }
  if (element_ != NULL) {
    element_.reset(element_->pop());
  }

  // Skip sentinel elements added to keep track of new proto3 types - map,
  // struct.
  SkipElements();


  // If ending the root element,
  // then serialize the full message with calculated sizes.
  if (element_ == NULL) {
    WriteRootMessage();
  }
  return this;
}

ProtoStreamObjectWriter* ProtoStreamObjectWriter::StartList(StringPiece name) {
  const google::protobuf::Field* field = NULL;
  // Since we cannot have a top-level repeated item in protobuf, the only way
  // element_ can be null when here is when we start a top-level list
  // google.protobuf.ListValue.
  if (element_ == NULL) {
    if (!name.empty()) {
      InvalidName(name, "Root element should not be named.");
    }
    element_.reset(new ProtoElement(typeinfo_, master_type_, this));

    // If master type is a special type that needs extra values to be written to
    // stream, we write those values.
    if (master_type_.name() == kStructValueType) {
      // We got a StartList with google.protobuf.Value master type. This means
      // we have to start the "list_value" within google.protobuf.Value.
      field = StartListValueInStruct(NULL);
    } else if (master_type_.name() == kStructListValueType) {
      // We got a StartList with google.protobuf.ListValue master type. This
      // means we have to start the "values" within google.protobuf.ListValue.
      field = StartRepeatedValuesInListValue(NULL);
    }

    // field is NULL when master_type_ is anything other than
    // google.protobuf.Value or google.protobuf.ListValue.
    if (field) {
      const google::protobuf::Type* type = LookupType(field);
      element_.reset(new ProtoElement(element_.release(), field, *type,
                                      ProtoElement::STRUCT_LIST));
    }
    return this;
  }

  if (element_->IsAny()) {
    element_->any()->StartList(name);
    return this;
  }
  // The type of element we push to stack.
  ProtoElement::ElementType element_type = ProtoElement::LIST;

  // Check if we need to start a map. This can heppen when there is either a map
  // or a struct type within a list.
  if (element_->IsMap() || element_->IsStructMap()) {
    if (!ValidMapKey(name)) {
      ++invalid_depth_;
      return this;
    }

    field = StartMapEntry(name);
    if (field == NULL) return this;

    if (element_->IsStructMapEntry()) {
      // If the top element is a map entry, this means we are starting a list
      // within a struct or a map.
      // An example sequence of calls would be
      //    StartObject -> StartList
      field = StartListValueInStruct(field);
      if (field == NULL) return this;
    }

    element_type = ProtoElement::STRUCT_LIST;
  } else if (element_->IsStructList()) {
    // If the top element is a STRUCT_LIST, this means we are starting a list
    // within the current list (inside a struct).
    // An example call sequence would be
    //    StartObject -> StartList -> StartList
    // with StartObject starting a struct.

    // Lookup the last list type in element stack as we are adding an element of
    // the same type.
    field = Lookup(name);
    if (field == NULL) return this;

    field = StartListValueInStruct(field);
    if (field == NULL) return this;

    element_type = ProtoElement::STRUCT_LIST;
  } else {
    // Lookup field corresponding to 'name'. If it is a google.protobuf.Value
    // or google.protobuf.ListValue type, then StartList is a valid call, start
    // this list.
    // We cannot use Lookup() here as it will produce InvalidName() error if the
    // field is not found. We do not want to error here as it would cause us to
    // report errors twice, once here and again later in BeginNamed() call.
    // Also we ignore if the field is not found here as it is caught later.
    field = typeinfo_->FindField(&element_->type(), name);

    // Only check for oneof collisions on the first StartList call. We identify
    // the first call with !name.empty() check. Subsequent list element calls
    // will not have the name filled.
    if (!name.empty() && field && !ValidOneof(*field, name)) {
      ++invalid_depth_;
      return this;
    }

    // It is an error to try to bind to map, which behind the scenes is a list.
    if (field && IsMap(*field)) {
      // Push field to stack for error location tracking & reporting.
      element_.reset(new ProtoElement(element_.release(), field,
                                      *LookupType(field),
                                      ProtoElement::MESSAGE));
      InvalidValue("Map", "Cannot bind a list to map.");
      ++invalid_depth_;
      element_.reset(element_->pop());
      return this;
    }

    if (field && field->type_url() == GetFullTypeWithUrl(kStructValueType)) {
      // There are 2 cases possible:
      //   a. g.p.Value is repeated
      //   b. g.p.Value is not repeated
      //
      // For case (a), the StartList should bind to the repeated g.p.Value.
      // For case (b), the StartList should bind to g.p.ListValue within the
      // g.p.Value.
      //
      // This means, for case (a), we treat it just like any other repeated
      // message, except we would apply an appropriate element_type so future
      // Start or Render calls are routed appropriately.
      if (field->cardinality() !=
          google::protobuf::Field_Cardinality_CARDINALITY_REPEATED) {
        field = StartListValueInStruct(field);
      }
      element_type = ProtoElement::STRUCT_LIST;
    } else if (field &&
               field->type_url() == GetFullTypeWithUrl(kStructListValueType)) {
      // We got a StartList with google.protobuf.ListValue master type. This
      // means we have to start the "values" within google.protobuf.ListValue.
      field = StartRepeatedValuesInListValue(field);
    } else {
      // If no special types are to be bound, fall back to normal processing of
      // StartList.
      field = BeginNamed(name, true);
    }
    if (field == NULL) return this;
  }

  const google::protobuf::Type* type = LookupType(field);
  if (type == NULL) {
    ++invalid_depth_;
    InvalidName(name,
                StrCat("Missing descriptor for field: ", field->type_url()));
    return this;
  }

  element_.reset(
      new ProtoElement(element_.release(), field, *type, element_type));
  return this;
}

ProtoStreamObjectWriter* ProtoStreamObjectWriter::EndList() {
  if (invalid_depth_ > 0) {
    --invalid_depth_;
  } else if (element_ != NULL) {
    if (element_->IsAny()) {
      element_->any()->EndList();
    } else {
      element_.reset(element_->pop());
      // Skip sentinel elements added to keep track of new proto3 types - map,
      // struct.
      SkipElements();
    }
  }

  // When element_ is NULL, we have reached the root message type. Write out
  // the bytes.
  if (element_ == NULL) {
    WriteRootMessage();
  }
  return this;
}

Status ProtoStreamObjectWriter::RenderStructValue(ProtoStreamObjectWriter* ow,
                                                  const DataPiece& data) {
  string struct_field_name;
  switch (data.type()) {
    // Our JSON parser parses numbers as either int64, uint64, or double.
    case DataPiece::TYPE_INT64:
    case DataPiece::TYPE_UINT64:
    case DataPiece::TYPE_DOUBLE: {
      struct_field_name = "number_value";
      break;
    }
    case DataPiece::TYPE_STRING: {
      struct_field_name = "string_value";
      break;
    }
    case DataPiece::TYPE_BOOL: {
      struct_field_name = "bool_value";
      break;
    }
    case DataPiece::TYPE_NULL: {
      struct_field_name = "null_value";
      break;
    }
    default: {
      return Status(INVALID_ARGUMENT,
                    "Invalid struct data type. Only number, string, boolean or "
                    "null values are supported.");
    }
  }
  ow->RenderDataPiece(struct_field_name, data);
  return Status::OK;
}

Status ProtoStreamObjectWriter::RenderTimestamp(ProtoStreamObjectWriter* ow,
                                                const DataPiece& data) {
  if (data.type() != DataPiece::TYPE_STRING) {
    return Status(INVALID_ARGUMENT,
                  StrCat("Invalid data type for timestamp, value is ",
                         data.ValueAsStringOrDefault("")));
  }

  StringPiece value(data.str());

  int64 seconds;
  int32 nanos;
  if (!::google::protobuf::internal::ParseTime(value.ToString(), &seconds,
                                               &nanos)) {
    return Status(INVALID_ARGUMENT, StrCat("Invalid time format: ", value));
  }


  ow->RenderDataPiece("seconds", DataPiece(seconds));
  ow->RenderDataPiece("nanos", DataPiece(nanos));
  return Status::OK;
}

static inline util::Status RenderOneFieldPath(ProtoStreamObjectWriter* ow,
                                                StringPiece path) {
  ow->RenderDataPiece("paths",
                      DataPiece(ConvertFieldMaskPath(path, &ToSnakeCase)));
  return Status::OK;
}

Status ProtoStreamObjectWriter::RenderFieldMask(ProtoStreamObjectWriter* ow,
                                                const DataPiece& data) {
  if (data.type() != DataPiece::TYPE_STRING) {
    return Status(INVALID_ARGUMENT,
                  StrCat("Invalid data type for field mask, value is ",
                         data.ValueAsStringOrDefault("")));
  }

// TODO(tsun): figure out how to do proto descriptor based snake case
// conversions as much as possible. Because ToSnakeCase sometimes returns the
// wrong value.
  google::protobuf::scoped_ptr<ResultCallback1<util::Status, StringPiece> > callback(
      google::protobuf::internal::NewPermanentCallback(&RenderOneFieldPath, ow));
  return DecodeCompactFieldMaskPaths(data.str(), callback.get());
}

Status ProtoStreamObjectWriter::RenderDuration(ProtoStreamObjectWriter* ow,
                                               const DataPiece& data) {
  if (data.type() != DataPiece::TYPE_STRING) {
    return Status(INVALID_ARGUMENT,
                  StrCat("Invalid data type for duration, value is ",
                         data.ValueAsStringOrDefault("")));
  }

  StringPiece value(data.str());

  if (!value.ends_with("s")) {
    return Status(INVALID_ARGUMENT,
                  "Illegal duration format; duration must end with 's'");
  }
  value = value.substr(0, value.size() - 1);
  int sign = 1;
  if (value.starts_with("-")) {
    sign = -1;
    value = value.substr(1);
  }

  StringPiece s_secs, s_nanos;
  SplitSecondsAndNanos(value, &s_secs, &s_nanos);
  uint64 unsigned_seconds;
  if (!safe_strtou64(s_secs, &unsigned_seconds)) {
    return Status(INVALID_ARGUMENT,
                  "Invalid duration format, failed to parse seconds");
  }

  double d_nanos = 0;
  if (!safe_strtod("0." + s_nanos.ToString(), &d_nanos)) {
    return Status(INVALID_ARGUMENT,
                  "Invalid duration format, failed to parse nanos seconds");
  }

  int32 nanos = sign * static_cast<int32>(d_nanos * kNanosPerSecond);
  int64 seconds = sign * unsigned_seconds;

  if (seconds > kMaxSeconds || seconds < kMinSeconds ||
      nanos <= -kNanosPerSecond || nanos >= kNanosPerSecond) {
    return Status(INVALID_ARGUMENT, "Duration value exceeds limits");
  }

  ow->RenderDataPiece("seconds", DataPiece(seconds));
  ow->RenderDataPiece("nanos", DataPiece(nanos));
  return Status::OK;
}

Status ProtoStreamObjectWriter::RenderWrapperType(ProtoStreamObjectWriter* ow,
                                                  const DataPiece& data) {
  ow->RenderDataPiece("value", data);
  return Status::OK;
}

ProtoStreamObjectWriter* ProtoStreamObjectWriter::RenderDataPiece(
    StringPiece name, const DataPiece& data) {
  Status status;
  if (invalid_depth_ > 0) return this;
  if (element_ != NULL && element_->IsAny()) {
    element_->any()->RenderDataPiece(name, data);
    return this;
  }

  const google::protobuf::Field* field = NULL;
  string type_url;
  bool is_map_entry = false;
  // We are at the root when element_ == NULL.
  if (element_ == NULL) {
    type_url = GetFullTypeWithUrl(master_type_.name());
  } else {
    if (element_->IsMap() || element_->IsStructMap()) {
      if (!ValidMapKey(name)) return this;
      is_map_entry = true;
      field = StartMapEntry(name);
    } else {
      field = Lookup(name);
    }
    if (field == NULL) {
      return this;
    }

    // Check to see if this field is a oneof and that no oneof in that group has
    // already been set.
    if (!ValidOneof(*field, name)) return this;

    type_url = field->type_url();
  }

  // Check if there are any well known type renderers available for type_url.
  const TypeRenderer* type_renderer = FindTypeRenderer(type_url);
  if (type_renderer != NULL) {
    // Push the current element to stack so lookups in type_renderer will
    // find the fields. We do an EndObject soon after, which pops this. This is
    // safe because all well-known types are messages.
    if (element_ == NULL) {
      element_.reset(new ProtoElement(typeinfo_, master_type_, this));
    } else {
      if (field) {
        WriteTag(*field);
        const google::protobuf::Type* type = LookupType(field);
        element_.reset(new ProtoElement(element_.release(), field, *type,
                                        ProtoElement::MESSAGE));
      }
    }
    status = (*type_renderer)(this, data);
    if (!status.ok()) {
      InvalidValue(type_url,
                   StrCat("Field '", name, "', ", status.error_message()));
    }
    EndObject();
    return this;
  } else if (element_ == NULL) {  // no message type found at root
    element_.reset(new ProtoElement(typeinfo_, master_type_, this));
    InvalidName(name, "Root element must be a message.");
    return this;
  }

  if (field == NULL) {
    return this;
  }
  const google::protobuf::Type* type = LookupType(field);
  if (type == NULL) {
    InvalidName(name,
                StrCat("Missing descriptor for field: ", field->type_url()));
    return this;
  }

  // Whether we should pop at the end. Set to true if the data field is a
  // message type, which can happen in case of struct values.
  bool should_pop = false;

  RenderSimpleDataPiece(*field, *type, data);

  if (should_pop && element_ != NULL) {
    element_.reset(element_->pop());
  }

  if (is_map_entry) {
    // Ending map is the same as ending an object.
    EndObject();
  }
  return this;
}

void ProtoStreamObjectWriter::RenderSimpleDataPiece(
    const google::protobuf::Field& field, const google::protobuf::Type& type,
    const DataPiece& data) {
  // If we are rendering explicit null values and the backend proto field is not
  // of the google.protobuf.NullType type, we do nothing.
  if (data.type() == DataPiece::TYPE_NULL &&
      field.type_url() != kStructNullValueTypeUrl) {
    return;
  }

  // Pushing a ProtoElement and then pop it off at the end for 2 purposes:
  // error location reporting and required field accounting.
  element_.reset(new ProtoElement(element_.release(), &field, type,
                                  ProtoElement::MESSAGE));

  // Make sure that field represents a simple data type.
  if (field.kind() == google::protobuf::Field_Kind_TYPE_UNKNOWN ||
      field.kind() == google::protobuf::Field_Kind_TYPE_MESSAGE) {
    InvalidValue(field.type_url().empty()
                     ? google::protobuf::Field_Kind_Name(field.kind())
                     : field.type_url(),
                 data.ValueAsStringOrDefault(""));
    return;
  }

  Status status;
  switch (field.kind()) {
    case google::protobuf::Field_Kind_TYPE_INT32: {
      status = WriteInt32(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SFIXED32: {
      status = WriteSFixed32(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SINT32: {
      status = WriteSInt32(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FIXED32: {
      status = WriteFixed32(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_UINT32: {
      status = WriteUInt32(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_INT64: {
      status = WriteInt64(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SFIXED64: {
      status = WriteSFixed64(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_SINT64: {
      status = WriteSInt64(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FIXED64: {
      status = WriteFixed64(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_UINT64: {
      status = WriteUInt64(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_DOUBLE: {
      status = WriteDouble(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_FLOAT: {
      status = WriteFloat(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_BOOL: {
      status = WriteBool(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_BYTES: {
      status = WriteBytes(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_STRING: {
      status = WriteString(field.number(), data, stream_.get());
      break;
    }
    case google::protobuf::Field_Kind_TYPE_ENUM: {
      status = WriteEnum(field.number(), data,
                         typeinfo_->GetEnumByTypeUrl(field.type_url()),
                         stream_.get());
      break;
    }
    default:  // TYPE_GROUP or TYPE_MESSAGE
      status = Status(INVALID_ARGUMENT, data.ToString().ValueOrDie());
  }
  if (!status.ok()) {
    InvalidValue(google::protobuf::Field_Kind_Name(field.kind()),
                 status.error_message());
  }
  element_.reset(element_->pop());
}

// Map of functions that are responsible for rendering well known type
// represented by the key.
hash_map<string, ProtoStreamObjectWriter::TypeRenderer>*
    ProtoStreamObjectWriter::renderers_ = NULL;
GOOGLE_PROTOBUF_DECLARE_ONCE(writer_renderers_init_);

void ProtoStreamObjectWriter::InitRendererMap() {
  renderers_ = new hash_map<string, ProtoStreamObjectWriter::TypeRenderer>();
  (*renderers_)["type.googleapis.com/google.protobuf.Timestamp"] =
      &ProtoStreamObjectWriter::RenderTimestamp;
  (*renderers_)["type.googleapis.com/google.protobuf.Duration"] =
      &ProtoStreamObjectWriter::RenderDuration;
  (*renderers_)["type.googleapis.com/google.protobuf.FieldMask"] =
      &ProtoStreamObjectWriter::RenderFieldMask;
  (*renderers_)["type.googleapis.com/google.protobuf.Double"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Float"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Int64"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.UInt64"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Int32"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.UInt32"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Bool"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.String"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Bytes"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.DoubleValue"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.FloatValue"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Int64Value"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.UInt64Value"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Int32Value"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.UInt32Value"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.BoolValue"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.StringValue"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.BytesValue"] =
      &ProtoStreamObjectWriter::RenderWrapperType;
  (*renderers_)["type.googleapis.com/google.protobuf.Value"] =
      &ProtoStreamObjectWriter::RenderStructValue;
  ::google::protobuf::internal::OnShutdown(&DeleteRendererMap);
}

void ProtoStreamObjectWriter::DeleteRendererMap() {
  delete ProtoStreamObjectWriter::renderers_;
  renderers_ = NULL;
}

ProtoStreamObjectWriter::TypeRenderer*
ProtoStreamObjectWriter::FindTypeRenderer(const string& type_url) {
  ::google::protobuf::GoogleOnceInit(&writer_renderers_init_, &InitRendererMap);
  return FindOrNull(*renderers_, type_url);
}

ProtoStreamObjectWriter::ProtoElement::ElementType
ProtoStreamObjectWriter::GetElementType(const google::protobuf::Type& type) {
  if (type.name() == kAnyType) {
    return ProtoElement::ANY;
  } else if (type.name() == kStructType) {
    return ProtoElement::STRUCT;
  } else if (type.name() == kStructValueType) {
    return ProtoElement::STRUCT_VALUE;
  } else if (type.name() == kStructListValueType) {
    return ProtoElement::STRUCT_LIST_VALUE;
  } else {
    return ProtoElement::MESSAGE;
  }
}

bool ProtoStreamObjectWriter::ValidOneof(const google::protobuf::Field& field,
                                         StringPiece unnormalized_name) {
  if (element_ == NULL) return true;

  if (field.oneof_index() > 0) {
    if (element_->OneofIndexTaken(field.oneof_index())) {
      InvalidValue(
          "oneof",
          StrCat("oneof field '",
                 element_->type().oneofs(field.oneof_index() - 1),
                 "' is already set. Cannot set '", unnormalized_name, "'"));
      return false;
    }
    element_->TakeOneofIndex(field.oneof_index());
  }
  return true;
}

bool ProtoStreamObjectWriter::ValidMapKey(StringPiece unnormalized_name) {
  if (element_ == NULL) return true;

  if (!element_->InsertMapKeyIfNotPresent(unnormalized_name)) {
    InvalidName(
        unnormalized_name,
        StrCat("Repeated map key: '", unnormalized_name, "' is already set."));
    return false;
  }

  return true;
}

const google::protobuf::Field* ProtoStreamObjectWriter::BeginNamed(
    StringPiece name, bool is_list) {
  if (invalid_depth_ > 0) {
    ++invalid_depth_;
    return NULL;
  }
  const google::protobuf::Field* field = Lookup(name);
  if (field == NULL) {
    ++invalid_depth_;
    // InvalidName() already called in Lookup().
    return NULL;
  }
  if (is_list &&
      field->cardinality() !=
          google::protobuf::Field_Cardinality_CARDINALITY_REPEATED) {
    ++invalid_depth_;
    InvalidName(name, "Proto field is not repeating, cannot start list.");
    return NULL;
  }
  return field;
}

const google::protobuf::Field* ProtoStreamObjectWriter::Lookup(
    StringPiece unnormalized_name) {
  ProtoElement* e = element();
  if (e == NULL) {
    InvalidName(unnormalized_name, "Root element must be a message.");
    return NULL;
  }
  if (unnormalized_name.empty()) {
    // Objects in repeated field inherit the same field descriptor.
    if (e->field() == NULL) {
      InvalidName(unnormalized_name, "Proto fields must have a name.");
    } else if (e->field()->cardinality() !=
               google::protobuf::Field_Cardinality_CARDINALITY_REPEATED) {
      InvalidName(unnormalized_name, "Proto fields must have a name.");
      return NULL;
    }
    return e->field();
  }
  const google::protobuf::Field* field =
      typeinfo_->FindField(&e->type(), unnormalized_name);
  if (field == NULL) InvalidName(unnormalized_name, "Cannot find field.");
  return field;
}

const google::protobuf::Type* ProtoStreamObjectWriter::LookupType(
    const google::protobuf::Field* field) {
  return (field->kind() == google::protobuf::Field_Kind_TYPE_MESSAGE
              ? typeinfo_->GetTypeByTypeUrl(field->type_url())
              : &element_->type());
}

// Looks up the oneof struct field based on the data type.
StatusOr<const google::protobuf::Field*>
ProtoStreamObjectWriter::LookupStructField(DataPiece::Type type) {
  const google::protobuf::Field* field = NULL;
  switch (type) {
    // Our JSON parser parses numbers as either int64, uint64, or double.
    case DataPiece::TYPE_INT64:
    case DataPiece::TYPE_UINT64:
    case DataPiece::TYPE_DOUBLE: {
      field = Lookup("number_value");
      break;
    }
    case DataPiece::TYPE_STRING: {
      field = Lookup("string_value");
      break;
    }
    case DataPiece::TYPE_BOOL: {
      field = Lookup("bool_value");
      break;
    }
    case DataPiece::TYPE_NULL: {
      field = Lookup("null_value");
      break;
    }
    default: { return Status(INVALID_ARGUMENT, "Invalid struct data type"); }
  }
  if (field == NULL) {
    return Status(INVALID_ARGUMENT, "Could not lookup struct field");
  }
  return field;
}

void ProtoStreamObjectWriter::WriteRootMessage() {
  GOOGLE_DCHECK(!done_);
  int curr_pos = 0;
  // Calls the destructor of CodedOutputStream to remove any uninitialized
  // memory from the Cord before we read it.
  stream_.reset(NULL);
  const void* data;
  int length;
  google::protobuf::io::ArrayInputStream input_stream(buffer_.data(), buffer_.size());
  while (input_stream.Next(&data, &length)) {
    if (length == 0) continue;
    int num_bytes = length;
    // Write up to where we need to insert the size field.
    // The number of bytes we may write is the smaller of:
    //   - the current fragment size
    //   - the distance to the next position where a size field needs to be
    //     inserted.
    if (!size_insert_.empty() &&
        size_insert_.front().pos - curr_pos < num_bytes) {
      num_bytes = size_insert_.front().pos - curr_pos;
    }
    output_->Append(static_cast<const char*>(data), num_bytes);
    if (num_bytes < length) {
      input_stream.BackUp(length - num_bytes);
    }
    curr_pos += num_bytes;
    // Insert the size field.
    //   size_insert_.front():      the next <index, size> pair to be written.
    //   size_insert_.front().pos:  position of the size field.
    //   size_insert_.front().size: the size (integer) to be inserted.
    if (!size_insert_.empty() && curr_pos == size_insert_.front().pos) {
      // Varint32 occupies at most 10 bytes.
      uint8 insert_buffer[10];
      uint8* insert_buffer_pos = CodedOutputStream::WriteVarint32ToArray(
          size_insert_.front().size, insert_buffer);
      output_->Append(reinterpret_cast<const char*>(insert_buffer),
                      insert_buffer_pos - insert_buffer);
      size_insert_.pop_front();
    }
  }
  output_->Flush();
  stream_.reset(new CodedOutputStream(&adapter_));
  done_ = true;
}

bool ProtoStreamObjectWriter::IsMap(const google::protobuf::Field& field) {
  if (field.type_url().empty() ||
      field.kind() != google::protobuf::Field_Kind_TYPE_MESSAGE ||
      field.cardinality() !=
          google::protobuf::Field_Cardinality_CARDINALITY_REPEATED) {
    return false;
  }
  const google::protobuf::Type* field_type =
      typeinfo_->GetTypeByTypeUrl(field.type_url());

  // TODO(xiaofeng): Unify option names.
  return GetBoolOptionOrDefault(field_type->options(),
                                "google.protobuf.MessageOptions.map_entry", false) ||
         GetBoolOptionOrDefault(field_type->options(), "map_entry", false);
}

void ProtoStreamObjectWriter::WriteTag(const google::protobuf::Field& field) {
  WireFormatLite::WireType wire_type = WireFormatLite::WireTypeForFieldType(
      static_cast<WireFormatLite::FieldType>(field.kind()));
  stream_->WriteTag(WireFormatLite::MakeTag(field.number(), wire_type));
}


}  // namespace converter
}  // namespace util
}  // namespace protobuf
}  // namespace google
