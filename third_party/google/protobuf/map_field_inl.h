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

#ifndef GOOGLE_PROTOBUF_MAP_FIELD_INL_H__
#define GOOGLE_PROTOBUF_MAP_FIELD_INL_H__

#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif

#include <google/protobuf/map.h>
#include <google/protobuf/map_field.h>
#include <google/protobuf/map_type_handler.h>

namespace google {
namespace protobuf {
namespace internal {
// UnwrapMapKey template
template<typename T>
T UnwrapMapKey(const MapKey& map_key);
template<>
inline int32 UnwrapMapKey<int32>(const MapKey& map_key) {
  return map_key.GetInt32Value();
}
template<>
inline uint32 UnwrapMapKey<uint32>(const MapKey& map_key) {
  return map_key.GetUInt32Value();
}
template<>
inline int64 UnwrapMapKey<int64>(const MapKey& map_key) {
  return map_key.GetInt64Value();
}
template<>
inline uint64 UnwrapMapKey<uint64>(const MapKey& map_key) {
  return map_key.GetUInt64Value();
}
template<>
inline bool UnwrapMapKey<bool>(const MapKey& map_key) {
  return map_key.GetBoolValue();
}
template<>
inline string UnwrapMapKey<string>(const MapKey& map_key) {
  return map_key.GetStringValue();
}

// SetMapKey template
template<typename T>
inline void SetMapKey(MapKey* map_key, const T& value);
template<>
inline void SetMapKey<int32>(MapKey* map_key, const int32& value) {
  map_key->SetInt32Value(value);
}
template<>
inline void SetMapKey<uint32>(MapKey* map_key, const uint32& value) {
  map_key->SetUInt32Value(value);
}
template<>
inline void SetMapKey<int64>(MapKey* map_key, const int64& value) {
  map_key->SetInt64Value(value);
}
template<>
inline void SetMapKey<uint64>(MapKey* map_key, const uint64& value) {
  map_key->SetUInt64Value(value);
}
template<>
inline void SetMapKey<bool>(MapKey* map_key, const bool& value) {
  map_key->SetBoolValue(value);
}
template<>
inline void SetMapKey<string>(MapKey* map_key, const string& value) {
  map_key->SetStringValue(value);
}

// ------------------------TypeDefinedMapFieldBase---------------
template <typename Key, typename T>
typename Map<Key, T>::const_iterator&
TypeDefinedMapFieldBase<Key, T>::InternalGetIterator(
    const MapIterator* map_iter) const {
  return *reinterpret_cast<typename Map<Key, T>::const_iterator *>(
      map_iter->iter_);
}

template <typename Key, typename T>
void TypeDefinedMapFieldBase<Key, T>::MapBegin(MapIterator* map_iter) const {
  InternalGetIterator(map_iter) = GetMap().begin();
  SetMapIteratorValue(map_iter);
}

template <typename Key, typename T>
void TypeDefinedMapFieldBase<Key, T>::MapEnd(MapIterator* map_iter) const {
  InternalGetIterator(map_iter) = GetMap().end();
}

template <typename Key, typename T>
bool TypeDefinedMapFieldBase<Key, T>::EqualIterator(const MapIterator& a,
                                                    const MapIterator& b)
    const {
  return InternalGetIterator(&a) == InternalGetIterator(&b);
}

template <typename Key, typename T>
void TypeDefinedMapFieldBase<Key, T>::IncreaseIterator(MapIterator* map_iter)
    const {
  ++InternalGetIterator(map_iter);
  SetMapIteratorValue(map_iter);
}

template <typename Key, typename T>
void TypeDefinedMapFieldBase<Key, T>::InitializeIterator(
    MapIterator* map_iter) const {
  map_iter->iter_ = new typename Map<Key, T>::const_iterator;
  GOOGLE_CHECK(map_iter->iter_ != NULL);
}

template <typename Key, typename T>
void TypeDefinedMapFieldBase<Key, T>::DeleteIterator(MapIterator* map_iter)
    const {
  delete reinterpret_cast<typename Map<Key, T>::const_iterator *>(
      map_iter->iter_);
}

template <typename Key, typename T>
void TypeDefinedMapFieldBase<Key, T>::CopyIterator(
    MapIterator* this_iter,
    const MapIterator& that_iter) const {
  InternalGetIterator(this_iter) = InternalGetIterator(&that_iter);
  this_iter->key_.SetType(that_iter.key_.type());
  // MapValueRef::type() fails when containing data is null. However, if
  // this_iter points to MapEnd, data can be null.
  this_iter->value_.SetType(
      static_cast<FieldDescriptor::CppType>(that_iter.value_.type_));
  SetMapIteratorValue(this_iter);
}

// ----------------------------------------------------------------------

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
MapField<Key, T, kKeyFieldType, kValueFieldType, default_enum_value>::MapField()
    : default_entry_(NULL) {}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
MapField<Key, T, kKeyFieldType, kValueFieldType, default_enum_value>::MapField(
    Arena* arena)
    : TypeDefinedMapFieldBase<Key, T>(arena),
      MapFieldLite<Key, T, kKeyFieldType, kValueFieldType, default_enum_value>(
          arena),
      default_entry_(NULL) {}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
MapField<Key, T, kKeyFieldType, kValueFieldType, default_enum_value>::MapField(
    const Message* default_entry)
    : default_entry_(down_cast<const EntryType*>(default_entry)) {}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
MapField<Key, T, kKeyFieldType, kValueFieldType, default_enum_value>::MapField(
    Arena* arena, const Message* default_entry)
    : TypeDefinedMapFieldBase<Key, T>(arena),
      MapFieldLite<Key, T, kKeyFieldType, kValueFieldType, default_enum_value>(
          arena),
      default_entry_(down_cast<const EntryType*>(default_entry)) {}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::~MapField() {}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
int
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::size() const {
  MapFieldBase::SyncMapWithRepeatedField();
  return MapFieldLiteType::GetInternalMap().size();
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::Clear() {
  MapFieldBase::SyncMapWithRepeatedField();
  MapFieldLiteType::MutableInternalMap()->clear();
  MapFieldBase::SetMapDirty();
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void MapField<Key, T, kKeyFieldType, kValueFieldType,
              default_enum_value>::SetMapIteratorValue(
                  MapIterator* map_iter) const {
  const Map<Key, T>& map = GetMap();
  typename Map<Key, T>::const_iterator iter =
      TypeDefinedMapFieldBase<Key, T>::InternalGetIterator(map_iter);
  if (iter == map.end()) return;
  SetMapKey(&map_iter->key_, iter->first);
  map_iter->value_.SetValue(&iter->second);
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
bool MapField<Key, T, kKeyFieldType, kValueFieldType,
              default_enum_value>::ContainsMapKey(
                  const MapKey& map_key) const {
  const Map<Key, T>& map = GetMap();
  const Key& key = UnwrapMapKey<Key>(map_key);
  typename Map<Key, T>::const_iterator iter = map.find(key);
  return iter != map.end();
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
bool MapField<Key, T, kKeyFieldType, kValueFieldType,
              default_enum_value>::InsertMapValue(const MapKey& map_key,
                                                  MapValueRef* val) {
  Map<Key, T>* map = MutableMap();
  bool result = false;
  const Key& key = UnwrapMapKey<Key>(map_key);
  if (map->end() == map->find(key)) {
    result = true;
  }
  val->SetValue(&((*map)[key]));
  return result;
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
bool MapField<Key, T, kKeyFieldType, kValueFieldType,
              default_enum_value>::DeleteMapValue(
                  const MapKey& map_key) {
  const Key& key = UnwrapMapKey<Key>(map_key);
  return MutableMap()->erase(key);
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
const Map<Key, T>&
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::GetMap() const {
  MapFieldBase::SyncMapWithRepeatedField();
  return MapFieldLiteType::GetInternalMap();
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
Map<Key, T>*
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::MutableMap() {
  MapFieldBase::SyncMapWithRepeatedField();
  Map<Key, T>* result = MapFieldLiteType::MutableInternalMap();
  MapFieldBase::SetMapDirty();
  return result;
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::MergeFrom(
    const MapFieldLiteType& other) {
  const MapField& down_other = down_cast<const MapField&>(other);
  MapFieldBase::SyncMapWithRepeatedField();
  down_other.SyncMapWithRepeatedField();
  MapFieldLiteType::MergeFrom(other);
  MapFieldBase::SetMapDirty();
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::Swap(
    MapFieldLiteType* other) {
  MapField* down_other = down_cast<MapField*>(other);
  std::swap(MapFieldBase::repeated_field_, down_other->repeated_field_);
  MapFieldLiteType::Swap(other);
  std::swap(MapFieldBase::state_, down_other->state_);
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::SetEntryDescriptor(
    const Descriptor** descriptor) {
  MapFieldBase::entry_descriptor_ = descriptor;
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::SetAssignDescriptorCallback(void (*callback)()) {
  MapFieldBase::assign_descriptor_callback_ = callback;
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
const Map<Key, T>&
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::GetInternalMap() const {
  return MapFieldLiteType::GetInternalMap();
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
Map<Key, T>*
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::MutableInternalMap() {
  return MapFieldLiteType::MutableInternalMap();
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::SyncRepeatedFieldWithMapNoLock() const {
  if (MapFieldBase::repeated_field_ == NULL) {
    if (MapFieldBase::arena_ == NULL) {
      MapFieldBase::repeated_field_ = new RepeatedPtrField<Message>();
    } else {
      MapFieldBase::repeated_field_ =
          Arena::CreateMessage<RepeatedPtrField<Message> >(
              MapFieldBase::arena_);
    }
  }
  const Map<Key, T>& map = GetInternalMap();
  RepeatedPtrField<EntryType>* repeated_field =
      reinterpret_cast<RepeatedPtrField<EntryType>*>(
          MapFieldBase::repeated_field_);

  repeated_field->Clear();

  for (typename Map<Key, T>::const_iterator it = map.begin();
       it != map.end(); ++it) {
    InitDefaultEntryOnce();
    GOOGLE_CHECK(default_entry_ != NULL);
    EntryType* new_entry =
        down_cast<EntryType*>(default_entry_->New(MapFieldBase::arena_));
    repeated_field->AddAllocated(new_entry);
    (*new_entry->mutable_key()) = it->first;
    (*new_entry->mutable_value()) = it->second;
  }
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::SyncMapWithRepeatedFieldNoLock() const {
  Map<Key, T>* map = const_cast<MapField*>(this)->MutableInternalMap();
  RepeatedPtrField<EntryType>* repeated_field =
      reinterpret_cast<RepeatedPtrField<EntryType>*>(
          MapFieldBase::repeated_field_);
  GOOGLE_CHECK(MapFieldBase::repeated_field_ != NULL);
  map->clear();
  for (typename RepeatedPtrField<EntryType>::iterator it =
           repeated_field->begin(); it != repeated_field->end(); ++it) {
    // Cast is needed because Map's api and internal storage is different when
    // value is enum. For enum, we cannot cast an int to enum. Thus, we have to
    // copy value. For other types, they have same exposed api type and internal
    // stored type. We should not introduce value copy for them. We achieve this
    // by casting to value for enum while casting to reference for other types.
    (*map)[it->key()] = static_cast<CastValueType>(it->value());
  }
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
int
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::SpaceUsedExcludingSelfNoLock() const {
  int size = 0;
  if (MapFieldBase::repeated_field_ != NULL) {
    size += MapFieldBase::repeated_field_->SpaceUsedExcludingSelf();
  }
  Map<Key, T>* map = const_cast<MapField*>(this)->MutableInternalMap();
  size += sizeof(*map);
  for (typename Map<Key, T>::iterator it = map->begin();
       it != map->end(); ++it) {
    size += KeyTypeHandler::SpaceUsedInMap(it->first);
    size += ValueTypeHandler::SpaceUsedInMap(it->second);
  }
  return size;
}

template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value>
void
MapField<Key, T, kKeyFieldType, kValueFieldType,
         default_enum_value>::InitDefaultEntryOnce()
    const {
  if (default_entry_ == NULL) {
    MapFieldBase::InitMetadataOnce();
    GOOGLE_CHECK(*MapFieldBase::entry_descriptor_ != NULL);
    default_entry_ = down_cast<const EntryType*>(
        MessageFactory::generated_factory()->GetPrototype(
            *MapFieldBase::entry_descriptor_));
  }
}

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_MAP_FIELD_INL_H__
