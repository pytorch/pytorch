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

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.
//
// Contains classes used to keep track of unrecognized fields seen while
// parsing a protocol message.

#ifndef GOOGLE_PROTOBUF_UNKNOWN_FIELD_SET_H__
#define GOOGLE_PROTOBUF_UNKNOWN_FIELD_SET_H__

#include <assert.h>
#include <string>
#include <vector>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/logging.h>

namespace google {
namespace protobuf {
  namespace io {
    class CodedInputStream;         // coded_stream.h
    class CodedOutputStream;        // coded_stream.h
    class ZeroCopyInputStream;      // zero_copy_stream.h
  }
  namespace internal {
    class WireFormat;               // wire_format.h
    class MessageSetFieldSkipperUsingCord;
                                    // extension_set_heavy.cc
  }

class Message;                      // message.h
class UnknownField;                 // below

// An UnknownFieldSet contains fields that were encountered while parsing a
// message but were not defined by its type.  Keeping track of these can be
// useful, especially in that they may be written if the message is serialized
// again without being cleared in between.  This means that software which
// simply receives messages and forwards them to other servers does not need
// to be updated every time a new field is added to the message definition.
//
// To get the UnknownFieldSet attached to any message, call
// Reflection::GetUnknownFields().
//
// This class is necessarily tied to the protocol buffer wire format, unlike
// the Reflection interface which is independent of any serialization scheme.
class LIBPROTOBUF_EXPORT UnknownFieldSet {
 public:
  UnknownFieldSet();
  ~UnknownFieldSet();

  // Remove all fields.
  inline void Clear();

  // Remove all fields and deallocate internal data objects
  void ClearAndFreeMemory();

  // Is this set empty?
  inline bool empty() const;

  // Merge the contents of some other UnknownFieldSet with this one.
  void MergeFrom(const UnknownFieldSet& other);

  // Similar to above, but this function will destroy the contents of other.
  void MergeFromAndDestroy(UnknownFieldSet* other);

  // Swaps the contents of some other UnknownFieldSet with this one.
  inline void Swap(UnknownFieldSet* x);

  // Computes (an estimate of) the total number of bytes currently used for
  // storing the unknown fields in memory. Does NOT include
  // sizeof(*this) in the calculation.
  int SpaceUsedExcludingSelf() const;

  // Version of SpaceUsed() including sizeof(*this).
  int SpaceUsed() const;

  // Returns the number of fields present in the UnknownFieldSet.
  inline int field_count() const;
  // Get a field in the set, where 0 <= index < field_count().  The fields
  // appear in the order in which they were added.
  inline const UnknownField& field(int index) const;
  // Get a mutable pointer to a field in the set, where
  // 0 <= index < field_count().  The fields appear in the order in which
  // they were added.
  inline UnknownField* mutable_field(int index);

  // Adding fields ---------------------------------------------------

  void AddVarint(int number, uint64 value);
  void AddFixed32(int number, uint32 value);
  void AddFixed64(int number, uint64 value);
  void AddLengthDelimited(int number, const string& value);
  string* AddLengthDelimited(int number);
  UnknownFieldSet* AddGroup(int number);

  // Adds an unknown field from another set.
  void AddField(const UnknownField& field);

  // Delete fields with indices in the range [start .. start+num-1].
  // Caution: implementation moves all fields with indices [start+num .. ].
  void DeleteSubrange(int start, int num);

  // Delete all fields with a specific field number. The order of left fields
  // is preserved.
  // Caution: implementation moves all fields after the first deleted field.
  void DeleteByNumber(int number);

  // Parsing helpers -------------------------------------------------
  // These work exactly like the similarly-named methods of Message.

  bool MergeFromCodedStream(io::CodedInputStream* input);
  bool ParseFromCodedStream(io::CodedInputStream* input);
  bool ParseFromZeroCopyStream(io::ZeroCopyInputStream* input);
  bool ParseFromArray(const void* data, int size);
  inline bool ParseFromString(const string& data) {
    return ParseFromArray(data.data(), static_cast<int>(data.size()));
  }

  static const UnknownFieldSet* default_instance();
 private:
  // For InternalMergeFrom
  friend class UnknownField;
  // Merges from other UnknownFieldSet. This method assumes, that this object
  // is newly created and has fields_ == NULL;
  void InternalMergeFrom(const UnknownFieldSet& other);
  void ClearFallback();

  // fields_ is either NULL, or a pointer to a vector that is *non-empty*. We
  // never hold the empty vector because we want the 'do we have any unknown
  // fields' check to be fast, and avoid a cache miss: the UFS instance gets
  // embedded in the message object, so 'fields_ != NULL' tests a member
  // variable hot in the cache, without the need to go touch a vector somewhere
  // else in memory.
  std::vector<UnknownField>* fields_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(UnknownFieldSet);
};

// Represents one field in an UnknownFieldSet.
class LIBPROTOBUF_EXPORT UnknownField {
 public:
  enum Type {
    TYPE_VARINT,
    TYPE_FIXED32,
    TYPE_FIXED64,
    TYPE_LENGTH_DELIMITED,
    TYPE_GROUP
  };

  // The field's tag number, as seen on the wire.
  inline int number() const;

  // The field type.
  inline Type type() const;

  // Accessors -------------------------------------------------------
  // Each method works only for UnknownFields of the corresponding type.

  inline uint64 varint() const;
  inline uint32 fixed32() const;
  inline uint64 fixed64() const;
  inline const string& length_delimited() const;
  inline const UnknownFieldSet& group() const;

  inline void set_varint(uint64 value);
  inline void set_fixed32(uint32 value);
  inline void set_fixed64(uint64 value);
  inline void set_length_delimited(const string& value);
  inline string* mutable_length_delimited();
  inline UnknownFieldSet* mutable_group();

  // Serialization API.
  // These methods can take advantage of the underlying implementation and may
  // archieve a better performance than using getters to retrieve the data and
  // do the serialization yourself.
  void SerializeLengthDelimitedNoTag(io::CodedOutputStream* output) const;
  uint8* SerializeLengthDelimitedNoTagToArray(uint8* target) const;

  inline int GetLengthDelimitedSize() const;

 private:
  friend class UnknownFieldSet;

  // If this UnknownField contains a pointer, delete it.
  void Delete();

  // Reset all the underlying pointers to NULL. A special function to be only
  // used while merging from a temporary UFS.
  void Reset();

  // Make a deep copy of any pointers in this UnknownField.
  void DeepCopy(const UnknownField& other);

  // Set the wire type of this UnknownField. Should only be used when this
  // UnknownField is being created.
  inline void SetType(Type type);

  union LengthDelimited {
    string* string_value_;
  };

  uint32 number_;
  uint32 type_;
  union {
    uint64 varint_;
    uint32 fixed32_;
    uint64 fixed64_;
    mutable union LengthDelimited length_delimited_;
    UnknownFieldSet* group_;
  };
};

// ===================================================================
// inline implementations

inline void UnknownFieldSet::Clear() {
  if (fields_) {
    ClearFallback();
  }
}

inline bool UnknownFieldSet::empty() const {
  // Invariant: fields_ is never empty if present.
  return !fields_;
}

inline void UnknownFieldSet::Swap(UnknownFieldSet* x) {
  std::swap(fields_, x->fields_);
}

inline int UnknownFieldSet::field_count() const {
  return fields_ ? static_cast<int>(fields_->size()) : 0;
}
inline const UnknownField& UnknownFieldSet::field(int index) const {
  GOOGLE_DCHECK(fields_ != NULL);
  return (*fields_)[index];
}
inline UnknownField* UnknownFieldSet::mutable_field(int index) {
  return &(*fields_)[index];
}

inline void UnknownFieldSet::AddLengthDelimited(
    int number, const string& value) {
  AddLengthDelimited(number)->assign(value);
}


inline int UnknownField::number() const { return number_; }
inline UnknownField::Type UnknownField::type() const {
  return static_cast<Type>(type_);
}

inline uint64 UnknownField::varint() const {
  assert(type() == TYPE_VARINT);
  return varint_;
}
inline uint32 UnknownField::fixed32() const {
  assert(type() == TYPE_FIXED32);
  return fixed32_;
}
inline uint64 UnknownField::fixed64() const {
  assert(type() == TYPE_FIXED64);
  return fixed64_;
}
inline const string& UnknownField::length_delimited() const {
  assert(type() == TYPE_LENGTH_DELIMITED);
  return *length_delimited_.string_value_;
}
inline const UnknownFieldSet& UnknownField::group() const {
  assert(type() == TYPE_GROUP);
  return *group_;
}

inline void UnknownField::set_varint(uint64 value) {
  assert(type() == TYPE_VARINT);
  varint_ = value;
}
inline void UnknownField::set_fixed32(uint32 value) {
  assert(type() == TYPE_FIXED32);
  fixed32_ = value;
}
inline void UnknownField::set_fixed64(uint64 value) {
  assert(type() == TYPE_FIXED64);
  fixed64_ = value;
}
inline void UnknownField::set_length_delimited(const string& value) {
  assert(type() == TYPE_LENGTH_DELIMITED);
  length_delimited_.string_value_->assign(value);
}
inline string* UnknownField::mutable_length_delimited() {
  assert(type() == TYPE_LENGTH_DELIMITED);
  return length_delimited_.string_value_;
}
inline UnknownFieldSet* UnknownField::mutable_group() {
  assert(type() == TYPE_GROUP);
  return group_;
}

inline int UnknownField::GetLengthDelimitedSize() const {
  GOOGLE_DCHECK_EQ(TYPE_LENGTH_DELIMITED, type());
  return static_cast<int>(length_delimited_.string_value_->size());
}

inline void UnknownField::SetType(Type type) {
  type_ = type;
}


}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UNKNOWN_FIELD_SET_H__
