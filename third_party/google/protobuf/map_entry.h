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

#ifndef GOOGLE_PROTOBUF_MAP_ENTRY_H__
#define GOOGLE_PROTOBUF_MAP_ENTRY_H__

#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/map_entry_lite.h>
#include <google/protobuf/map_type_handler.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/wire_format_lite_inl.h>

namespace google {
namespace protobuf {
class Arena;
namespace internal {
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapField;
}
}

namespace protobuf {
namespace internal {

// Register all MapEntry default instances so we can delete them in
// ShutdownProtobufLibrary().
void LIBPROTOBUF_EXPORT RegisterMapEntryDefaultInstance(
    MessageLite* default_instance);

// This is the common base class for MapEntry. It is used by MapFieldBase in
// reflection api, in which the static type of key and value is unknown.
class LIBPROTOBUF_EXPORT MapEntryBase : public Message {
 public:
  ::google::protobuf::Metadata GetMetadata() const {
    ::google::protobuf::Metadata metadata;
    metadata.descriptor = descriptor_;
    metadata.reflection = reflection_;
    return metadata;
  }

 protected:
  MapEntryBase() : descriptor_(NULL), reflection_(NULL) {  }
  virtual ~MapEntryBase() {}

  const Descriptor* descriptor_;
  const Reflection* reflection_;
};

// MapEntry is the returned google::protobuf::Message when calling AddMessage of
// google::protobuf::Reflection. In order to let it work with generated message
// reflection, its in-memory type is the same as generated message with the same
// fields. However, in order to decide the in-memory type of key/value, we need
// to know both their cpp type in generated api and proto type. In
// implmentation, all in-memory types have related wire format functions to
// support except ArenaStringPtr. Therefore, we need to define another type with
// supporting wire format functions. Since this type is only used as return type
// of MapEntry accessors, it's named MapEntry accessor type.
//
// cpp type:               the type visible to users in public API.
// proto type:             WireFormatLite::FieldType of the field.
// in-memory type:         type of the data member used to stored this field.
// MapEntry accessor type: type used in MapEntry getters/mutators to access the
//                         field.
//
// cpp type | proto type  | in-memory type | MapEntry accessor type
// int32      TYPE_INT32    int32            int32
// int32      TYPE_FIXED32  int32            int32
// string     TYPE_STRING   ArenaStringPtr   string
// FooEnum    TYPE_ENUM     int              int
// FooMessage TYPE_MESSAGE  FooMessage*      FooMessage
//
// The in-memory types of primitive types can be inferred from its proto type,
// while we need to explicitly specify the cpp type if proto type is
// TYPE_MESSAGE to infer the in-memory type.  Moreover, default_enum_value is
// used to initialize enum field in proto2.
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapEntry : public MapEntryBase {
  // Provide utilities to parse/serialize key/value.  Provide utilities to
  // manipulate internal stored type.
  typedef MapTypeHandler<kKeyFieldType, Key> KeyTypeHandler;
  typedef MapTypeHandler<kValueFieldType, Value> ValueTypeHandler;

  // Enum type cannot be used for MapTypeHandler::Read. Define a type
  // which will replace Enum with int.
  typedef typename KeyTypeHandler::MapEntryAccessorType KeyMapEntryAccessorType;
  typedef typename ValueTypeHandler::MapEntryAccessorType
      ValueMapEntryAccessorType;

  // Abbreviation for MapEntry
  typedef typename google::protobuf::internal::MapEntry<
      Key, Value, kKeyFieldType, kValueFieldType, default_enum_value> EntryType;

  // Abbreviation for MapEntryLite
  typedef typename google::protobuf::internal::MapEntryLite<
      Key, Value, kKeyFieldType, kValueFieldType, default_enum_value>
      EntryLiteType;

 public:
  ~MapEntry() {
    if (this == default_instance_) {
      delete reflection_;
    }
  }

  // accessors ======================================================

  virtual inline const KeyMapEntryAccessorType& key() const {
    return entry_lite_.key();
  }
  inline KeyMapEntryAccessorType* mutable_key() {
    return entry_lite_.mutable_key();
  }
  virtual inline const ValueMapEntryAccessorType& value() const {
    return entry_lite_.value();
  }
  inline ValueMapEntryAccessorType* mutable_value() {
    return entry_lite_.mutable_value();
  }

  // implements Message =============================================

  bool MergePartialFromCodedStream(::google::protobuf::io::CodedInputStream* input) {
    return entry_lite_.MergePartialFromCodedStream(input);
  }

  int ByteSize() const {
    return entry_lite_.ByteSize();
  }

  void SerializeWithCachedSizes(::google::protobuf::io::CodedOutputStream* output) const {
    entry_lite_.SerializeWithCachedSizes(output);
  }

  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return entry_lite_.SerializeWithCachedSizesToArray(output);
  }

  int GetCachedSize() const {
    return entry_lite_.GetCachedSize();
  }

  bool IsInitialized() const {
    return entry_lite_.IsInitialized();
  }

  Message* New() const {
    MapEntry* entry = new MapEntry;
    entry->descriptor_ = descriptor_;
    entry->reflection_ = reflection_;
    entry->set_default_instance(default_instance_);
    return entry;
  }

  Message* New(Arena* arena) const {
    MapEntry* entry = Arena::CreateMessage<MapEntry>(arena);
    entry->descriptor_ = descriptor_;
    entry->reflection_ = reflection_;
    entry->set_default_instance(default_instance_);
    return entry;
  }

  int SpaceUsed() const {
    int size = sizeof(MapEntry);
    size += entry_lite_.SpaceUsed();
    return size;
  }

  void CopyFrom(const ::google::protobuf::Message& from) {
    Clear();
    MergeFrom(from);
  }

  void MergeFrom(const ::google::protobuf::Message& from) {
    GOOGLE_CHECK_NE(&from, this);
    const MapEntry* source = dynamic_cast_if_available<const MapEntry*>(&from);
    if (source == NULL) {
      ReflectionOps::Merge(from, this);
    } else {
      MergeFrom(*source);
    }
  }

  void CopyFrom(const MapEntry& from) {
    Clear();
    MergeFrom(from);
  }

  void MergeFrom(const MapEntry& from) {
    entry_lite_.MergeFrom(from.entry_lite_);
  }

  void Clear() {
    entry_lite_.Clear();
  }

  void InitAsDefaultInstance() {
    entry_lite_.InitAsDefaultInstance();
  }

  Arena* GetArena() const {
    return entry_lite_.GetArena();
  }

  // Create default MapEntry instance for given descriptor. Descriptor has to be
  // given when creating default MapEntry instance because different map field
  // may have the same type and MapEntry class. The given descriptor is needed
  // to distinguish instances of the same MapEntry class.
  static MapEntry* CreateDefaultInstance(const Descriptor* descriptor) {
    MapEntry* entry = new MapEntry;
    const Reflection* reflection = new GeneratedMessageReflection(
        descriptor, entry, offsets_,
        GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MapEntry, entry_lite_._has_bits_),
        GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MapEntry, _unknown_fields_), -1,
        DescriptorPool::generated_pool(),
        ::google::protobuf::MessageFactory::generated_factory(),
        sizeof(MapEntry),
        GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MapEntry, _internal_metadata_));
    entry->descriptor_ = descriptor;
    entry->reflection_ = reflection;
    entry->set_default_instance(entry);
    entry->InitAsDefaultInstance();
    RegisterMapEntryDefaultInstance(entry);
    return entry;
  }

 private:
  MapEntry()
      : _internal_metadata_(NULL), default_instance_(NULL), entry_lite_() {}

  explicit MapEntry(Arena* arena)
      : _internal_metadata_(arena),
        default_instance_(NULL),
        entry_lite_(arena) {}

  inline Arena* GetArenaNoVirtual() const {
    return entry_lite_.GetArenaNoVirtual();
  }

  void set_default_instance(MapEntry* default_instance) {
    default_instance_ = default_instance;
    entry_lite_.set_default_instance(&default_instance->entry_lite_);
  }

  static int offsets_[2];
  UnknownFieldSet _unknown_fields_;
  InternalMetadataWithArena _internal_metadata_;
  MapEntry* default_instance_;
  EntryLiteType entry_lite_;

  friend class ::google::protobuf::Arena;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  template <typename K, typename V, WireFormatLite::FieldType k_wire_type,
            WireFormatLite::FieldType, int default_enum>
  friend class internal::MapField;
  friend class internal::GeneratedMessageReflection;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(MapEntry);
};

template <typename Key, typename Value, WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType, int default_enum_value>
int MapEntry<Key, Value, kKeyFieldType, kValueFieldType,
             default_enum_value>::offsets_[2] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MapEntry, entry_lite_.key_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MapEntry, entry_lite_.value_),
};

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_MAP_ENTRY_H__
