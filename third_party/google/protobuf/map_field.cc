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

#include <google/protobuf/map_field.h>
#include <google/protobuf/map_field_inl.h>

#include <vector>

namespace google {
namespace protobuf {
namespace internal {

ProtobufOnceType map_entry_default_instances_once_;
Mutex* map_entry_default_instances_mutex_;
vector<MessageLite*>* map_entry_default_instances_;

void DeleteMapEntryDefaultInstances() {
  for (int i = 0; i < map_entry_default_instances_->size(); ++i) {
    delete map_entry_default_instances_->at(i);
  }
  delete map_entry_default_instances_mutex_;
  delete map_entry_default_instances_;
}

void InitMapEntryDefaultInstances() {
  map_entry_default_instances_mutex_ = new Mutex();
  map_entry_default_instances_ = new vector<MessageLite*>();
  OnShutdown(&DeleteMapEntryDefaultInstances);
}

void RegisterMapEntryDefaultInstance(MessageLite* default_instance) {
  ::google::protobuf::GoogleOnceInit(&map_entry_default_instances_once_,
                 &InitMapEntryDefaultInstances);
  MutexLock lock(map_entry_default_instances_mutex_);
  map_entry_default_instances_->push_back(default_instance);
}

MapFieldBase::~MapFieldBase() {
  if (repeated_field_ != NULL && arena_ == NULL) delete repeated_field_;
}

const RepeatedPtrFieldBase& MapFieldBase::GetRepeatedField() const {
  SyncRepeatedFieldWithMap();
  return *repeated_field_;
}

RepeatedPtrFieldBase* MapFieldBase::MutableRepeatedField() {
  SyncRepeatedFieldWithMap();
  SetRepeatedDirty();
  return repeated_field_;
}

int MapFieldBase::SpaceUsedExcludingSelf() const {
  mutex_.Lock();
  int size = SpaceUsedExcludingSelfNoLock();
  mutex_.Unlock();
  return size;
}

int MapFieldBase::SpaceUsedExcludingSelfNoLock() const {
  if (repeated_field_ != NULL) {
    return repeated_field_->SpaceUsedExcludingSelf();
  } else {
    return 0;
  }
}

void MapFieldBase::InitMetadataOnce() const {
  GOOGLE_CHECK(entry_descriptor_ != NULL);
  GOOGLE_CHECK(assign_descriptor_callback_ != NULL);
  (*assign_descriptor_callback_)();
}

void MapFieldBase::SetMapDirty() { state_ = STATE_MODIFIED_MAP; }

void MapFieldBase::SetRepeatedDirty() { state_ = STATE_MODIFIED_REPEATED; }

void* MapFieldBase::MutableRepeatedPtrField() const { return repeated_field_; }

void MapFieldBase::SyncRepeatedFieldWithMap() const {
  // "Acquire" insures the operation after SyncRepeatedFieldWithMap won't get
  // executed before state_ is checked.
  Atomic32 state = google::protobuf::internal::Acquire_Load(&state_);
  if (state == STATE_MODIFIED_MAP) {
    mutex_.Lock();
    // Double check state, because another thread may have seen the same state
    // and done the synchronization before the current thread.
    if (state_ == STATE_MODIFIED_MAP) {
      SyncRepeatedFieldWithMapNoLock();
      // "Release" insures state_ can only be changed "after"
      // SyncRepeatedFieldWithMapNoLock is finished.
      google::protobuf::internal::Release_Store(&state_, CLEAN);
    }
    mutex_.Unlock();
  }
}

void MapFieldBase::SyncRepeatedFieldWithMapNoLock() const {
  if (repeated_field_ == NULL) {
    repeated_field_ = Arena::CreateMessage<RepeatedPtrField<Message> >(arena_);
  }
}

void MapFieldBase::SyncMapWithRepeatedField() const {
  // "Acquire" insures the operation after SyncMapWithRepeatedField won't get
  // executed before state_ is checked.
  Atomic32 state = google::protobuf::internal::Acquire_Load(&state_);
  if (state == STATE_MODIFIED_REPEATED) {
    mutex_.Lock();
    // Double check state, because another thread may have seen the same state
    // and done the synchronization before the current thread.
    if (state_ == STATE_MODIFIED_REPEATED) {
      SyncMapWithRepeatedFieldNoLock();
      // "Release" insures state_ can only be changed "after"
      // SyncRepeatedFieldWithMapNoLock is finished.
      google::protobuf::internal::Release_Store(&state_, CLEAN);
    }
    mutex_.Unlock();
  }
}

// ------------------DynamicMapField------------------
DynamicMapField::DynamicMapField(const Message* default_entry)
    : default_entry_(default_entry) {
}

DynamicMapField::DynamicMapField(const Message* default_entry,
                                 Arena* arena)
    : TypeDefinedMapFieldBase<MapKey, MapValueRef>(arena),
      default_entry_(default_entry) {
}

DynamicMapField::~DynamicMapField() {
  // DynamicMapField owns map values. Need to delete them before clearing
  // the map.
  for (Map<MapKey, MapValueRef>::iterator iter = map_.begin();
       iter != map_.end(); ++iter) {
    iter->second.DeleteData();
  }
  map_.clear();
}

int DynamicMapField::size() const {
  return GetMap().size();
}

bool DynamicMapField::ContainsMapKey(
    const MapKey& map_key) const {
  const Map<MapKey, MapValueRef>& map = GetMap();
  Map<MapKey, MapValueRef>::const_iterator iter = map.find(map_key);
  return iter != map.end();
}

bool DynamicMapField::InsertMapValue(
    const MapKey& map_key, MapValueRef* val) {
  bool result = false;

  MapValueRef& map_val = (*MutableMap())[map_key];
  // If map_val.data_ is not set, it is newly inserted by map_[map_key].
  if (map_val.data_ == NULL) {
    result = true;
    const FieldDescriptor* val_des =
        default_entry_->GetDescriptor()->FindFieldByName("value");
    map_val.SetType(val_des->cpp_type());
    // Allocate momery for the inserted MapValueRef, and initialize to
    // default value.
    switch (val_des->cpp_type()) {
#define HANDLE_TYPE(CPPTYPE, TYPE)                              \
      case google::protobuf::FieldDescriptor::CPPTYPE_##CPPTYPE: {        \
        TYPE * value = new TYPE();                              \
        map_val.SetValue(value);                                \
        break;                                                  \
      }
      HANDLE_TYPE(INT32, int32);
      HANDLE_TYPE(INT64, int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(STRING, string);
      HANDLE_TYPE(ENUM, int32);
#undef HANDLE_TYPE
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
        const Message& message = default_entry_->GetReflection()->GetMessage(
            *default_entry_, val_des);
        Message* value = message.New();
        map_val.SetValue(value);
        break;
      }
    }
  }
  val->CopyFrom(map_val);
  return result;
}

bool DynamicMapField::DeleteMapValue(const MapKey& map_key) {
  MapFieldBase::SyncMapWithRepeatedField();
  Map<MapKey, MapValueRef>::iterator iter = map_.find(map_key);
  if (iter == map_.end()) {
    return false;
  }
  // Set map dirty only if the delete is successful.
  MapFieldBase::SetMapDirty();
  iter->second.DeleteData();
  map_.erase(iter);
  return true;
}

const Map<MapKey, MapValueRef>& DynamicMapField::GetMap() const {
  MapFieldBase::SyncMapWithRepeatedField();
  return map_;
}

Map<MapKey, MapValueRef>* DynamicMapField::MutableMap() {
  MapFieldBase::SyncMapWithRepeatedField();
  MapFieldBase::SetMapDirty();
  return &map_;
}

void DynamicMapField::SetMapIteratorValue(MapIterator* map_iter) const {
  Map<MapKey, MapValueRef>::const_iterator iter =
      TypeDefinedMapFieldBase<MapKey, MapValueRef>::InternalGetIterator(
          map_iter);
  if (iter == map_.end()) return;
  map_iter->key_.CopyFrom(iter->first);
  map_iter->value_.CopyFrom(iter->second);
}

void DynamicMapField::SyncRepeatedFieldWithMapNoLock() const {
  const Reflection* reflection = default_entry_->GetReflection();
  const FieldDescriptor* key_des =
      default_entry_->GetDescriptor()->FindFieldByName("key");
  const FieldDescriptor* val_des =
      default_entry_->GetDescriptor()->FindFieldByName("value");
  if (MapFieldBase::repeated_field_ == NULL) {
    if (MapFieldBase::arena_ == NULL) {
      MapFieldBase::repeated_field_ = new RepeatedPtrField<Message>();
    } else {
      MapFieldBase::repeated_field_ =
          Arena::CreateMessage<RepeatedPtrField<Message> >(
              MapFieldBase::arena_);
    }
  }

  MapFieldBase::repeated_field_->Clear();

  for (Map<MapKey, MapValueRef>::const_iterator it = map_.begin();
       it != map_.end(); ++it) {
    Message* new_entry = default_entry_->New();
    MapFieldBase::repeated_field_->AddAllocated(new_entry);
    const MapKey& map_key = it->first;
    switch (key_des->cpp_type()) {
      case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
        reflection->SetString(new_entry, key_des, map_key.GetStringValue());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
        reflection->SetInt64(new_entry, key_des, map_key.GetInt64Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
        reflection->SetInt32(new_entry, key_des, map_key.GetInt32Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
        reflection->SetUInt64(new_entry, key_des, map_key.GetUInt64Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
        reflection->SetUInt32(new_entry, key_des, map_key.GetUInt32Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
        reflection->SetBool(new_entry, key_des, map_key.GetBoolValue());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Can't get here.";
        break;
    }
    const MapValueRef& map_val = it->second;
    switch (val_des->cpp_type()) {
      case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
        reflection->SetString(new_entry, val_des, map_val.GetStringValue());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
        reflection->SetInt64(new_entry, val_des, map_val.GetInt64Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
        reflection->SetInt32(new_entry, val_des, map_val.GetInt32Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
        reflection->SetUInt64(new_entry, val_des, map_val.GetUInt64Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
        reflection->SetUInt32(new_entry, val_des, map_val.GetUInt32Value());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
        reflection->SetBool(new_entry, val_des, map_val.GetBoolValue());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
        reflection->SetDouble(new_entry, val_des, map_val.GetDoubleValue());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
        reflection->SetFloat(new_entry, val_des, map_val.GetFloatValue());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
        reflection->SetEnumValue(new_entry, val_des, map_val.GetEnumValue());
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
        const Message& message = map_val.GetMessageValue();
        reflection->MutableMessage(new_entry, val_des)->CopyFrom(message);
        break;
      }
    }
  }
}

void DynamicMapField::SyncMapWithRepeatedFieldNoLock() const {
  Map<MapKey, MapValueRef>* map = &const_cast<DynamicMapField*>(this)->map_;
  const Reflection* reflection = default_entry_->GetReflection();
  const FieldDescriptor* key_des =
      default_entry_->GetDescriptor()->FindFieldByName("key");
  const FieldDescriptor* val_des =
      default_entry_->GetDescriptor()->FindFieldByName("value");
  // DynamicMapField owns map values. Need to delete them before clearing
  // the map.
  for (Map<MapKey, MapValueRef>::iterator iter = map->begin();
       iter != map->end(); ++iter) {
    iter->second.DeleteData();
  }
  map->clear();
  for (RepeatedPtrField<Message>::iterator it =
           MapFieldBase::repeated_field_->begin();
       it != MapFieldBase::repeated_field_->end(); ++it) {
    MapKey map_key;
    switch (key_des->cpp_type()) {
      case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
        map_key.SetStringValue(reflection->GetString(*it, key_des));
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
        map_key.SetInt64Value(reflection->GetInt64(*it, key_des));
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
        map_key.SetInt32Value(reflection->GetInt32(*it, key_des));
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
        map_key.SetUInt64Value(reflection->GetUInt64(*it, key_des));
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
        map_key.SetUInt32Value(reflection->GetUInt32(*it, key_des));
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
        map_key.SetBoolValue(reflection->GetBool(*it, key_des));
        break;
      case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
        GOOGLE_LOG(FATAL) << "Can't get here.";
        break;
    }
    MapValueRef& map_val = (*map)[map_key];
    map_val.SetType(val_des->cpp_type());
    switch (val_des->cpp_type()) {
#define HANDLE_TYPE(CPPTYPE, TYPE, METHOD)                      \
      case google::protobuf::FieldDescriptor::CPPTYPE_##CPPTYPE: {        \
        TYPE * value = new TYPE;                                \
        *value = reflection->Get##METHOD(*it, val_des);         \
            map_val.SetValue(value);                            \
            break;                                              \
      }
      HANDLE_TYPE(INT32, int32, Int32);
      HANDLE_TYPE(INT64, int64, Int64);
      HANDLE_TYPE(UINT32, uint32, UInt32);
      HANDLE_TYPE(UINT64, uint64, UInt64);
      HANDLE_TYPE(DOUBLE, double, Double);
      HANDLE_TYPE(FLOAT, float, Float);
      HANDLE_TYPE(BOOL, bool, Bool);
      HANDLE_TYPE(STRING, string, String);
      HANDLE_TYPE(ENUM, int32, EnumValue);
#undef HANDLE_TYPE
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
        const Message& message = reflection->GetMessage(*it, val_des);
        Message* value = message.New();
        value->CopyFrom(message);
        map_val.SetValue(value);
        break;
      }
    }
  }
}

int DynamicMapField::SpaceUsedExcludingSelfNoLock() const {
  int size = 0;
  if (MapFieldBase::repeated_field_ != NULL) {
    size += MapFieldBase::repeated_field_->SpaceUsedExcludingSelf();
  }
  size += sizeof(map_);
  int map_size = map_.size();
  if (map_size) {
    Map<MapKey, MapValueRef>::const_iterator it = map_.begin();
    size += sizeof(it->first) * map_size;
    size += sizeof(it->second) * map_size;
    // If key is string, add the allocated space.
    if (it->first.type() == google::protobuf::FieldDescriptor::CPPTYPE_STRING) {
      size += sizeof(string) * map_size;
    }
    // Add the allocated space in MapValueRef.
    switch (it->second.type()) {
#define HANDLE_TYPE(CPPTYPE, TYPE)                              \
      case google::protobuf::FieldDescriptor::CPPTYPE_##CPPTYPE: {        \
        size += sizeof(TYPE) * map_size;                        \
        break;                                                  \
      }
      HANDLE_TYPE(INT32, int32);
      HANDLE_TYPE(INT64, int64);
      HANDLE_TYPE(UINT32, uint32);
      HANDLE_TYPE(UINT64, uint64);
      HANDLE_TYPE(DOUBLE, double);
      HANDLE_TYPE(FLOAT, float);
      HANDLE_TYPE(BOOL, bool);
      HANDLE_TYPE(STRING, string);
      HANDLE_TYPE(ENUM, int32);
#undef HANDLE_TYPE
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
        while (it != map_.end()) {
          const Message& message = it->second.GetMessageValue();
          size += message.GetReflection()->SpaceUsed(message);
          ++it;
        }
        break;
      }
    }
  }
  return size;
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google
