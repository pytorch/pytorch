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

#ifndef GOOGLE_PROTOBUF_MAP_ENTRY_LITE_H__
#define GOOGLE_PROTOBUF_MAP_ENTRY_LITE_H__

#include <google/protobuf/map_type_handler.h>
#include <google/protobuf/wire_format_lite_inl.h>

namespace google {
namespace protobuf {
class Arena;
namespace internal {
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapEntry;
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapFieldLite;
}  // namespace internal
}  // namespace protobuf

namespace protobuf {
namespace internal {

// MapEntryLite is used to implement parsing and serialization of map for lite
// runtime.
template <typename Key, typename Value,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
class MapEntryLite : public MessageLite {
  // Provide utilities to parse/serialize key/value.  Provide utilities to
  // manipulate internal stored type.
  typedef MapTypeHandler<kKeyFieldType, Key> KeyTypeHandler;
  typedef MapTypeHandler<kValueFieldType, Value> ValueTypeHandler;

  // Define internal memory layout. Strings and messages are stored as
  // pointers, while other types are stored as values.
  typedef typename KeyTypeHandler::TypeOnMemory KeyOnMemory;
  typedef typename ValueTypeHandler::TypeOnMemory ValueOnMemory;

  // Enum type cannot be used for MapTypeHandler::Read. Define a type
  // which will replace Enum with int.
  typedef typename KeyTypeHandler::MapEntryAccessorType KeyMapEntryAccessorType;
  typedef typename ValueTypeHandler::MapEntryAccessorType
      ValueMapEntryAccessorType;

  // Constants for field number.
  static const int kKeyFieldNumber = 1;
  static const int kValueFieldNumber = 2;

  // Constants for field tag.
  static const uint8 kKeyTag = GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(
      kKeyFieldNumber, KeyTypeHandler::kWireType);
  static const uint8 kValueTag = GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(
      kValueFieldNumber, ValueTypeHandler::kWireType);
  static const int kTagSize = 1;

 public:
  ~MapEntryLite() {
    if (this != default_instance_) {
      if (GetArenaNoVirtual() != NULL) return;
      KeyTypeHandler::DeleteNoArena(key_);
      ValueTypeHandler::DeleteNoArena(value_);
    }
  }

  // accessors ======================================================

  virtual inline const KeyMapEntryAccessorType& key() const {
    return KeyTypeHandler::GetExternalReference(key_);
  }
  virtual inline const ValueMapEntryAccessorType& value() const {
    GOOGLE_CHECK(default_instance_ != NULL);
    return ValueTypeHandler::DefaultIfNotInitialized(value_,
                                                    default_instance_->value_);
  }
  inline KeyMapEntryAccessorType* mutable_key() {
    set_has_key();
    return KeyTypeHandler::EnsureMutable(&key_, GetArenaNoVirtual());
  }
  inline ValueMapEntryAccessorType* mutable_value() {
    set_has_value();
    return ValueTypeHandler::EnsureMutable(&value_, GetArenaNoVirtual());
  }

  // implements MessageLite =========================================

  // MapEntryLite is for implementation only and this function isn't called
  // anywhere. Just provide a fake implementation here for MessageLite.
  string GetTypeName() const { return ""; }

  void CheckTypeAndMergeFrom(const MessageLite& other) {
    MergeFrom(*::google::protobuf::down_cast<const MapEntryLite*>(&other));
  }

  bool MergePartialFromCodedStream(::google::protobuf::io::CodedInputStream* input) {
    uint32 tag;

    for (;;) {
      // 1) corrupted data: return false;
      // 2) unknown field: skip without putting into unknown field set;
      // 3) unknown enum value: keep it in parsing. In proto2, caller should
      // check the value and put this entry into containing message's unknown
      // field set if the value is an unknown enum. In proto3, caller doesn't
      // need to care whether the value is unknown enum;
      // 4) missing key/value: missed key/value will have default value. caller
      // should take this entry as if key/value is set to default value.
      tag = input->ReadTag();
      switch (tag) {
        case kKeyTag:
          if (!KeyTypeHandler::Read(input, mutable_key())) {
            return false;
          }
          set_has_key();
          if (!input->ExpectTag(kValueTag)) break;
          GOOGLE_FALLTHROUGH_INTENDED;

        case kValueTag:
          if (!ValueTypeHandler::Read(input, mutable_value())) {
            return false;
          }
          set_has_value();
          if (input->ExpectAtEnd()) return true;
          break;

        default:
          if (tag == 0 ||
              WireFormatLite::GetTagWireType(tag) ==
              WireFormatLite::WIRETYPE_END_GROUP) {
            return true;
          }
          if (!WireFormatLite::SkipField(input, tag)) return false;
          break;
      }
    }
  }

  int ByteSize() const {
    int size = 0;
    size += has_key() ? kTagSize + KeyTypeHandler::ByteSize(key()) : 0;
    size += has_value() ? kTagSize + ValueTypeHandler::ByteSize(value()) : 0;
    return size;
  }

  void SerializeWithCachedSizes(::google::protobuf::io::CodedOutputStream* output) const {
    KeyTypeHandler::Write(kKeyFieldNumber, key(), output);
    ValueTypeHandler::Write(kValueFieldNumber, value(), output);
  }

  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    output = KeyTypeHandler::WriteToArray(kKeyFieldNumber, key(), output);
    output = ValueTypeHandler::WriteToArray(kValueFieldNumber, value(), output);
    return output;
  }

  int GetCachedSize() const {
    int size = 0;
    size += has_key()
        ? kTagSize + KeyTypeHandler::GetCachedSize(key())
        : 0;
    size += has_value()
        ? kTagSize + ValueTypeHandler::GetCachedSize(
            value())
        : 0;
    return size;
  }

  bool IsInitialized() const { return ValueTypeHandler::IsInitialized(value_); }

  MessageLite* New() const {
    MapEntryLite* entry = new MapEntryLite;
    entry->default_instance_ = default_instance_;
    return entry;
  }

  MessageLite* New(Arena* arena) const {
    MapEntryLite* entry = Arena::CreateMessage<MapEntryLite>(arena);
    entry->default_instance_ = default_instance_;
    return entry;
  }

  int SpaceUsed() const {
    int size = sizeof(MapEntryLite);
    size += KeyTypeHandler::SpaceUsedInMapEntry(key_);
    size += ValueTypeHandler::SpaceUsedInMapEntry(value_);
    return size;
  }

  void MergeFrom(const MapEntryLite& from) {
    if (from._has_bits_[0]) {
      if (from.has_key()) {
        KeyTypeHandler::EnsureMutable(&key_, GetArenaNoVirtual());
        KeyTypeHandler::Merge(from.key(), &key_, GetArenaNoVirtual());
        set_has_key();
      }
      if (from.has_value()) {
        ValueTypeHandler::EnsureMutable(&value_, GetArenaNoVirtual());
        ValueTypeHandler::Merge(from.value(), &value_, GetArenaNoVirtual());
        set_has_value();
      }
    }
  }

  void Clear() {
    KeyTypeHandler::Clear(&key_, GetArenaNoVirtual());
    ValueTypeHandler::ClearMaybeByDefaultEnum(
        &value_, GetArenaNoVirtual(), default_enum_value);
    clear_has_key();
    clear_has_value();
  }

  void InitAsDefaultInstance() {
    KeyTypeHandler::AssignDefaultValue(&key_);
    ValueTypeHandler::AssignDefaultValue(&value_);
  }

  Arena* GetArena() const {
    return GetArenaNoVirtual();
  }

  // Create a MapEntryLite for given key and value from google::protobuf::Map in
  // serialization. This function is only called when value is enum. Enum is
  // treated differently because its type in MapEntry is int and its type in
  // google::protobuf::Map is enum. We cannot create a reference to int from an enum.
  static MapEntryLite* EnumWrap(const Key& key, const Value value,
                                Arena* arena) {
    return Arena::CreateMessage<MapEnumEntryWrapper<
        Key, Value, kKeyFieldType, kValueFieldType, default_enum_value> >(
        arena, key, value);
  }

  // Like above, but for all the other types. This avoids value copy to create
  // MapEntryLite from google::protobuf::Map in serialization.
  static MapEntryLite* Wrap(const Key& key, const Value& value, Arena* arena) {
    return Arena::CreateMessage<MapEntryWrapper<Key, Value, kKeyFieldType,
                                                kValueFieldType,
                                                default_enum_value> >(
        arena, key, value);
  }

 protected:
  void set_has_key() { _has_bits_[0] |= 0x00000001u; }
  bool has_key() const { return (_has_bits_[0] & 0x00000001u) != 0; }
  void clear_has_key() { _has_bits_[0] &= ~0x00000001u; }
  void set_has_value() { _has_bits_[0] |= 0x00000002u; }
  bool has_value() const { return (_has_bits_[0] & 0x00000002u) != 0; }
  void clear_has_value() { _has_bits_[0] &= ~0x00000002u; }

 private:
  // Serializing a generated message containing map field involves serializing
  // key-value pairs from google::protobuf::Map. The wire format of each key-value pair
  // after serialization should be the same as that of a MapEntry message
  // containing the same key and value inside it.  However, google::protobuf::Map doesn't
  // store key and value as MapEntry message, which disables us to use existing
  // code to serialize message. In order to use existing code to serialize
  // message, we need to construct a MapEntry from key-value pair. But it
  // involves copy of key and value to construct a MapEntry. In order to avoid
  // this copy in constructing a MapEntry, we need the following class which
  // only takes references of given key and value.
  template <typename K, typename V, WireFormatLite::FieldType k_wire_type,
            WireFormatLite::FieldType v_wire_type, int default_enum>
  class MapEntryWrapper
      : public MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> {
    typedef MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> Base;
    typedef typename Base::KeyMapEntryAccessorType KeyMapEntryAccessorType;
    typedef typename Base::ValueMapEntryAccessorType ValueMapEntryAccessorType;

   public:
    MapEntryWrapper(Arena* arena, const K& key, const V& value)
        : MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum>(arena),
          key_(key),
          value_(value) {
      Base::set_has_key();
      Base::set_has_value();
    }
    inline const KeyMapEntryAccessorType& key() const { return key_; }
    inline const ValueMapEntryAccessorType& value() const { return value_; }

   private:
    const Key& key_;
    const Value& value_;

    friend class ::google::protobuf::Arena;
    typedef void InternalArenaConstructable_;
    typedef void DestructorSkippable_;
  };

  // Like above, but for enum value only, which stores value instead of
  // reference of value field inside. This is needed because the type of value
  // field in constructor is an enum, while we need to store it as an int. If we
  // initialize a reference to int with a reference to enum, compiler will
  // generate a temporary int from enum and initialize the reference to int with
  // the temporary.
  template <typename K, typename V, WireFormatLite::FieldType k_wire_type,
            WireFormatLite::FieldType v_wire_type, int default_enum>
  class MapEnumEntryWrapper
      : public MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> {
    typedef MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum> Base;
    typedef typename Base::KeyMapEntryAccessorType KeyMapEntryAccessorType;
    typedef typename Base::ValueMapEntryAccessorType ValueMapEntryAccessorType;

   public:
    MapEnumEntryWrapper(Arena* arena, const K& key, const V& value)
        : MapEntryLite<K, V, k_wire_type, v_wire_type, default_enum>(arena),
          key_(key),
          value_(value) {
      Base::set_has_key();
      Base::set_has_value();
    }
    inline const KeyMapEntryAccessorType& key() const { return key_; }
    inline const ValueMapEntryAccessorType& value() const { return value_; }

   private:
    const KeyMapEntryAccessorType& key_;
    const ValueMapEntryAccessorType value_;

    friend class google::protobuf::Arena;
    typedef void DestructorSkippable_;
  };

  MapEntryLite() : default_instance_(NULL), arena_(NULL) {
    KeyTypeHandler::Initialize(&key_, NULL);
    ValueTypeHandler::InitializeMaybeByDefaultEnum(
        &value_, default_enum_value, NULL);
    _has_bits_[0] = 0;
  }

  explicit MapEntryLite(Arena* arena)
      : default_instance_(NULL), arena_(arena) {
    KeyTypeHandler::Initialize(&key_, arena);
    ValueTypeHandler::InitializeMaybeByDefaultEnum(
        &value_, default_enum_value, arena);
    _has_bits_[0] = 0;
  }

  inline Arena* GetArenaNoVirtual() const {
    return arena_;
  }

  void set_default_instance(MapEntryLite* default_instance) {
    default_instance_ = default_instance;
  }

  MapEntryLite* default_instance_;

  KeyOnMemory key_;
  ValueOnMemory value_;
  Arena* arena_;
  uint32 _has_bits_[1];

  friend class ::google::protobuf::Arena;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  template <typename K, typename V, WireFormatLite::FieldType,
            WireFormatLite::FieldType, int>
  friend class internal::MapEntry;
  template <typename K, typename V, WireFormatLite::FieldType,
            WireFormatLite::FieldType, int>
  friend class internal::MapFieldLite;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(MapEntryLite);
};

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_MAP_ENTRY_LITE_H__
