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

#ifndef GOOGLE_PROTOBUF_MAP_FIELD_H__
#define GOOGLE_PROTOBUF_MAP_FIELD_H__

#include <google/protobuf/stubs/atomicops.h>
#include <google/protobuf/stubs/mutex.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_lite.h>
#include <google/protobuf/map_type_handler.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/unknown_field_set.h>


namespace google {
namespace protobuf {
class DynamicMessage;
class MapKey;
namespace internal {

class ContendedMapCleanTest;
class GeneratedMessageReflection;
class MapFieldAccessor;

// This class provides accesss to map field using reflection, which is the same
// as those provided for RepeatedPtrField<Message>. It is used for internal
// reflection implentation only. Users should never use this directly.
class LIBPROTOBUF_EXPORT MapFieldBase {
 public:
  MapFieldBase()
      : arena_(NULL),
        repeated_field_(NULL),
        entry_descriptor_(NULL),
        assign_descriptor_callback_(NULL),
        state_(STATE_MODIFIED_MAP) {}
  explicit MapFieldBase(Arena* arena)
      : arena_(arena),
        repeated_field_(NULL),
        entry_descriptor_(NULL),
        assign_descriptor_callback_(NULL),
        state_(STATE_MODIFIED_MAP) {
    // Mutex's destructor needs to be called explicitly to release resources
    // acquired in its constructor.
    arena->OwnDestructor(&mutex_);
  }
  virtual ~MapFieldBase();

  // Returns reference to internal repeated field. Data written using
  // google::protobuf::Map's api prior to calling this function is guarantted to be
  // included in repeated field.
  const RepeatedPtrFieldBase& GetRepeatedField() const;

  // Like above. Returns mutable pointer to the internal repeated field.
  RepeatedPtrFieldBase* MutableRepeatedField();

  // Pure virtual map APIs for Map Reflection.
  virtual bool ContainsMapKey(const MapKey& map_key) const = 0;
  virtual bool InsertMapValue(const MapKey& map_key, MapValueRef* val) = 0;
  virtual bool DeleteMapValue(const MapKey& map_key) = 0;
  virtual bool EqualIterator(const MapIterator& a,
                             const MapIterator& b) const = 0;
  virtual void MapBegin(MapIterator* map_iter) const = 0;
  virtual void MapEnd(MapIterator* map_iter) const = 0;
  // Sync Map with repeated field and returns the size of map.
  virtual int size() const = 0;

  // Returns the number of bytes used by the repeated field, excluding
  // sizeof(*this)
  int SpaceUsedExcludingSelf() const;

 protected:
  // Gets the size of space used by map field.
  virtual int SpaceUsedExcludingSelfNoLock() const;

  // Synchronizes the content in Map to RepeatedPtrField if there is any change
  // to Map after last synchronization.
  void SyncRepeatedFieldWithMap() const;
  virtual void SyncRepeatedFieldWithMapNoLock() const;

  // Synchronizes the content in RepeatedPtrField to Map if there is any change
  // to RepeatedPtrField after last synchronization.
  void SyncMapWithRepeatedField() const;
  virtual void SyncMapWithRepeatedFieldNoLock() const {}

  // Tells MapFieldBase that there is new change to Map.
  void SetMapDirty();

  // Tells MapFieldBase that there is new change to RepeatedPTrField.
  void SetRepeatedDirty();

  // Provides derived class the access to repeated field.
  void* MutableRepeatedPtrField() const;

  // Creates descriptor for only one time.
  void InitMetadataOnce() const;

  enum State {
    STATE_MODIFIED_MAP = 0,       // map has newly added data that has not been
                                  // synchronized to repeated field
    STATE_MODIFIED_REPEATED = 1,  // repeated field has newly added data that
                                  // has not been synchronized to map
    CLEAN = 2,  // data in map and repeated field are same
  };

  Arena* arena_;
  mutable RepeatedPtrField<Message>* repeated_field_;
  // MapEntry can only be created from MapField. To create MapEntry, MapField
  // needs to know its descriptor, because MapEntry is not generated class which
  // cannot initialize its own descriptor by calling generated
  // descriptor-assign-function. Thus, we need to register a callback to
  // initialize MapEntry's descriptor.
  const Descriptor** entry_descriptor_;
  void (*assign_descriptor_callback_)();

  mutable Mutex mutex_;  // The thread to synchronize map and repeated field
                         // needs to get lock first;
  mutable volatile Atomic32 state_;  // 0: STATE_MODIFIED_MAP
                                     // 1: STATE_MODIFIED_REPEATED
                                     // 2: CLEAN

 private:
  friend class ContendedMapCleanTest;
  friend class GeneratedMessageReflection;
  friend class MapFieldAccessor;
  friend class ::google::protobuf::DynamicMessage;

  // Virtual helper methods for MapIterator. MapIterator doesn't have the
  // type helper for key and value. Call these help methods to deal with
  // different types. Real helper methods are implemented in
  // TypeDefinedMapFieldBase.
  friend class ::google::protobuf::MapIterator;
  // Allocate map<...>::iterator for MapIterator.
  virtual void InitializeIterator(MapIterator* map_iter) const = 0;

  // DeleteIterator() is called by the destructor of MapIterator only.
  // It deletes map<...>::iterator for MapIterator.
  virtual void DeleteIterator(MapIterator* map_iter) const = 0;

  // Copy the map<...>::iterator from other_iterator to
  // this_iterator.
  virtual void CopyIterator(MapIterator* this_iterator,
                            const MapIterator& other_iterator) const = 0;

  // IncreaseIterator() is called by operator++() of MapIterator only.
  // It implements the ++ operator of MapIterator.
  virtual void IncreaseIterator(MapIterator* map_iter) const = 0;
};

// This class provides common Map Reflection implementations for generated
// message and dynamic message.
template<typename Key, typename T>
class TypeDefinedMapFieldBase : public MapFieldBase {
 public:
  TypeDefinedMapFieldBase() {}
  explicit TypeDefinedMapFieldBase(Arena* arena) : MapFieldBase(arena) {}
  ~TypeDefinedMapFieldBase() {}
  void MapBegin(MapIterator* map_iter) const;
  void MapEnd(MapIterator* map_iter) const;
  bool EqualIterator(const MapIterator& a, const MapIterator& b) const;

  virtual const Map<Key, T>& GetMap() const = 0;
  virtual Map<Key, T>* MutableMap() = 0;

 protected:
  typename Map<Key, T>::const_iterator& InternalGetIterator(
      const MapIterator* map_iter) const;

 private:
  void InitializeIterator(MapIterator* map_iter) const;
  void DeleteIterator(MapIterator* map_iter) const;
  void CopyIterator(MapIterator* this_iteratorm,
                    const MapIterator& that_iterator) const;
  void IncreaseIterator(MapIterator* map_iter) const;

  virtual void SetMapIteratorValue(MapIterator* map_iter) const = 0;
};

// This class provides accesss to map field using generated api. It is used for
// internal generated message implentation only. Users should never use this
// directly.
template <typename Key, typename T,
          WireFormatLite::FieldType kKeyFieldType,
          WireFormatLite::FieldType kValueFieldType,
          int default_enum_value = 0>
class MapField : public TypeDefinedMapFieldBase<Key, T>,
                 public MapFieldLite<Key, T, kKeyFieldType, kValueFieldType,
                                     default_enum_value> {
  // Provide utilities to parse/serialize key/value.  Provide utilities to
  // manipulate internal stored type.
  typedef MapTypeHandler<kKeyFieldType, Key> KeyTypeHandler;
  typedef MapTypeHandler<kValueFieldType, T> ValueTypeHandler;

  // Define message type for internal repeated field.
  typedef MapEntry<Key, T, kKeyFieldType, kValueFieldType, default_enum_value>
      EntryType;
  typedef MapEntryLite<Key, T, kKeyFieldType, kValueFieldType,
                       default_enum_value> EntryLiteType;

  // Define abbreviation for parent MapFieldLite
  typedef MapFieldLite<Key, T, kKeyFieldType, kValueFieldType,
                       default_enum_value> MapFieldLiteType;

  // Enum needs to be handled differently from other types because it has
  // different exposed type in google::protobuf::Map's api and repeated field's api. For
  // details see the comment in the implementation of
  // SyncMapWithRepeatedFieldNoLock.
  static const bool kIsValueEnum = ValueTypeHandler::kIsEnum;
  typedef typename MapIf<kIsValueEnum, T, const T&>::type CastValueType;

 public:
  MapField();
  explicit MapField(Arena* arena);
  // MapField doesn't own the default_entry, which means default_entry must
  // outlive the lifetime of MapField.
  MapField(const Message* default_entry);
  // For tests only.
  MapField(Arena* arena, const Message* default_entry);
  ~MapField();

  // Implement MapFieldBase
  bool ContainsMapKey(const MapKey& map_key) const;
  bool InsertMapValue(const MapKey& map_key, MapValueRef* val);
  bool DeleteMapValue(const MapKey& map_key);

  // Accessors
  const Map<Key, T>& GetMap() const;
  Map<Key, T>* MutableMap();

  // Convenient methods for generated message implementation.
  int size() const;
  void Clear();
  void MergeFrom(const MapFieldLiteType& other);
  void Swap(MapFieldLiteType* other);

  // Allocates metadata only if this MapField is part of a generated message.
  void SetEntryDescriptor(const Descriptor** descriptor);
  void SetAssignDescriptorCallback(void (*callback)());

 private:
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;

  // MapField needs MapEntry's default instance to create new MapEntry.
  void InitDefaultEntryOnce() const;

  // Manually set default entry instance. For test only.
  void SetDefaultEntryOnce(const EntryType* default_entry) const;

  // Convenient methods to get internal google::protobuf::Map
  const Map<Key, T>& GetInternalMap() const;
  Map<Key, T>* MutableInternalMap();

  // Implements MapFieldBase
  void SyncRepeatedFieldWithMapNoLock() const;
  void SyncMapWithRepeatedFieldNoLock() const;
  int SpaceUsedExcludingSelfNoLock() const;

  void SetMapIteratorValue(MapIterator* map_iter) const;

  mutable const EntryType* default_entry_;

  friend class ::google::protobuf::Arena;
};

class LIBPROTOBUF_EXPORT DynamicMapField: public TypeDefinedMapFieldBase<MapKey, MapValueRef> {
 public:
  explicit DynamicMapField(const Message* default_entry);
  DynamicMapField(const Message* default_entry, Arena* arena);
  ~DynamicMapField();

  // Implement MapFieldBase
  bool ContainsMapKey(const MapKey& map_key) const;
  bool InsertMapValue(const MapKey& map_key, MapValueRef* val);
  bool DeleteMapValue(const MapKey& map_key);

  const Map<MapKey, MapValueRef>& GetMap() const;
  Map<MapKey, MapValueRef>* MutableMap();

  int size() const;

 private:
  Map<MapKey, MapValueRef> map_;
  const Message* default_entry_;

  // Implements MapFieldBase
  void SyncRepeatedFieldWithMapNoLock() const;
  void SyncMapWithRepeatedFieldNoLock() const;
  int SpaceUsedExcludingSelfNoLock() const;
  void SetMapIteratorValue(MapIterator* map_iter) const;
};

}  // namespace internal

class LIBPROTOBUF_EXPORT MapIterator {
 public:
  MapIterator(Message* message, const FieldDescriptor* field) {
    const Reflection* reflection = message->GetReflection();
    map_ = reflection->MapData(message, field);
    key_.SetType(field->message_type()->FindFieldByName("key")->cpp_type());
    value_.SetType(field->message_type()->FindFieldByName("value")->cpp_type());
    map_->InitializeIterator(this);
  }
  MapIterator(const MapIterator& other) {
    map_ = other.map_;
    map_->InitializeIterator(this);
    map_->CopyIterator(this, other);
  }
  ~MapIterator() {
    map_->DeleteIterator(this);
  }
  friend bool operator==(const MapIterator& a, const MapIterator& b) {
    return a.map_->EqualIterator(a, b);
  }
  friend bool operator!=(const MapIterator& a, const MapIterator& b) {
    return !a.map_->EqualIterator(a, b);
  }
  MapIterator& operator++() {
    map_->IncreaseIterator(this);
    return *this;
  }
  MapIterator operator++(int) {
    // iter_ is copied from Map<...>::iterator, no need to
    // copy from its self again. Use the same implementation
    // with operator++()
    map_->IncreaseIterator(this);
    return *this;
  }
  const MapKey& GetKey() {
    return key_;
  }
  const MapValueRef& GetValueRef() {
    return value_;
  }
  MapValueRef* MutableValueRef() {
    map_->SetMapDirty();
    return &value_;
  }

 private:
  template <typename Key, typename T>
  friend class internal::TypeDefinedMapFieldBase;
  friend class internal::DynamicMapField;
  template <typename Key, typename T,
            internal::WireFormatLite::FieldType kKeyFieldType,
            internal::WireFormatLite::FieldType kValueFieldType,
            int default_enum_value>
  friend class internal::MapField;

  // reinterpret_cast from heap-allocated Map<...>::iterator*. MapIterator owns
  // the iterator. It is allocated by MapField<...>::InitializeIterator() called
  // in constructor and deleted by MapField<...>::DeleteIterator() called in
  // destructor.
  void* iter_;
  // Point to a MapField to call helper methods implemented in MapField.
  // MapIterator does not own this object.
  internal::MapFieldBase* map_;
  MapKey key_;
  MapValueRef value_;
};

}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_MAP_FIELD_H__
