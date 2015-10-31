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

#ifndef GOOGLE_PROTOBUF_MAP_FIELD_LITE_H__
#define GOOGLE_PROTOBUF_MAP_FIELD_LITE_H__

#include <google/protobuf/map.h>
#include <google/protobuf/map_entry_lite.h>

namespace google {
namespace protobuf {
namespace internal {

// This class provides accesss to map field using generated api. It is used for
// internal generated message implentation only. Users should never use this
// directly.
template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value = 0>
class MapFieldLite {
  // Define message type for internal repeated field.
  typedef MapEntryLite<Key, T, key_wire_type, value_wire_type,
                       default_enum_value> EntryType;

 public:
  MapFieldLite();
  explicit MapFieldLite(Arena* arena);
  virtual ~MapFieldLite();

  // Accessors
  virtual const Map<Key, T>& GetMap() const;
  virtual Map<Key, T>* MutableMap();

  // Convenient methods for generated message implementation.
  virtual int size() const;
  virtual void Clear();
  virtual void MergeFrom(const MapFieldLite& other);
  virtual void Swap(MapFieldLite* other);

  // Set default enum value only for proto2 map field whose value is enum type.
  void SetDefaultEnumValue();

  // Used in the implementation of parsing. Caller should take the ownership.
  EntryType* NewEntry() const;
  // Used in the implementation of serializing enum value type. Caller should
  // take the ownership.
  EntryType* NewEnumEntryWrapper(const Key& key, const T t) const;
  // Used in the implementation of serializing other value types. Caller should
  // take the ownership.
  EntryType* NewEntryWrapper(const Key& key, const T& t) const;

 protected:
  // Convenient methods to get internal google::protobuf::Map
  virtual const Map<Key, T>& GetInternalMap() const;
  virtual Map<Key, T>* MutableInternalMap();

 private:
  typedef void DestructorSkippable_;

  Arena* arena_;
  Map<Key, T>* map_;

  friend class ::google::protobuf::Arena;
};

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::MapFieldLite()
    : arena_(NULL) {
  map_ = new Map<Key, T>;
  SetDefaultEnumValue();
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::MapFieldLite(Arena* arena)
  : arena_(arena) {
  map_ = Arena::CreateMessage<Map<Key, T> >(arena);
  SetDefaultEnumValue();
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::~MapFieldLite() {
  delete map_;
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
const Map<Key, T>&
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::GetMap() const {
  return *map_;
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
Map<Key, T>*
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::MutableMap() {
  return map_;
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
int
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::size() const {
  return map_->size();
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
void
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::Clear() {
  map_->clear();
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
void
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::MergeFrom(
    const MapFieldLite& other) {
  for (typename Map<Key, T>::const_iterator it = other.map_->begin();
       it != other.map_->end(); ++it) {
    (*map_)[it->first] = it->second;
  }
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
void
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::Swap(
    MapFieldLite* other) {
  std::swap(map_, other->map_);
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
void
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::SetDefaultEnumValue() {
  MutableInternalMap()->SetDefaultEnumValue(default_enum_value);
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
const Map<Key, T>&
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::GetInternalMap() const {
  return *map_;
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
Map<Key, T>*
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::MutableInternalMap() {
  return map_;
}

#define EntryType \
  MapEntryLite<Key, T, key_wire_type, value_wire_type, default_enum_value>

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
EntryType*
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::NewEntry() const {
  if (arena_ == NULL) {
    return new EntryType();
  } else {
    return Arena::CreateMessage<EntryType>(arena_);
  }
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
EntryType*
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::NewEnumEntryWrapper(const Key& key,
                                                      const T t) const {
  return EntryType::EnumWrap(key, t, arena_);
}

template <typename Key, typename T,
          WireFormatLite::FieldType key_wire_type,
          WireFormatLite::FieldType value_wire_type,
          int default_enum_value>
EntryType*
MapFieldLite<Key, T, key_wire_type, value_wire_type,
             default_enum_value>::NewEntryWrapper(const Key& key,
                                                  const T& t) const {
  return EntryType::Wrap(key, t, arena_);
}

#undef EntryType

// True if IsInitialized() is true for value field in all elements of t. T is
// expected to be message.  It's useful to have this helper here to keep the
// protobuf compiler from ever having to emit loops in IsInitialized() methods.
// We want the C++ compiler to inline this or not as it sees fit.
template <typename Key, typename T>
bool AllAreInitialized(const Map<Key, T>& t) {
  for (typename Map<Key, T>::const_iterator it = t.begin(); it != t.end();
       ++it) {
    if (!it->second.IsInitialized()) return false;
  }
  return true;
}

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_MAP_FIELD_LITE_H__
