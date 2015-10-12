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

#include <google/protobuf/unknown_field_set.h>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/stl_util.h>

namespace google {
namespace protobuf {

namespace {
// This global instance is returned by unknown_fields() on any message class
// when the object has no unknown fields. This is necessary because we now
// instantiate the UnknownFieldSet dynamically only when required.
UnknownFieldSet* default_unknown_field_set_instance_ = NULL;

void DeleteDefaultUnknownFieldSet() {
  delete default_unknown_field_set_instance_;
}

void InitDefaultUnknownFieldSet() {
  default_unknown_field_set_instance_ = new UnknownFieldSet();
  internal::OnShutdown(&DeleteDefaultUnknownFieldSet);
}

GOOGLE_PROTOBUF_DECLARE_ONCE(default_unknown_field_set_once_init_);
}

const UnknownFieldSet* UnknownFieldSet::default_instance() {
  ::google::protobuf::GoogleOnceInit(&default_unknown_field_set_once_init_,
                 &InitDefaultUnknownFieldSet);
  return default_unknown_field_set_instance_;
}

UnknownFieldSet::UnknownFieldSet()
    : fields_(NULL) {}

UnknownFieldSet::~UnknownFieldSet() {
  Clear();
  delete fields_;
}

void UnknownFieldSet::ClearFallback() {
  if (fields_ != NULL) {
    for (int i = 0; i < fields_->size(); i++) {
      (*fields_)[i].Delete();
    }
    delete fields_;
    fields_ = NULL;
  }
}

void UnknownFieldSet::ClearAndFreeMemory() {
  if (fields_ != NULL) {
    Clear();
  }
}

void UnknownFieldSet::InternalMergeFrom(const UnknownFieldSet& other) {
  int other_field_count = other.field_count();
  if (other_field_count > 0) {
    fields_ = new vector<UnknownField>();
    for (int i = 0; i < other_field_count; i++) {
      fields_->push_back((*other.fields_)[i]);
      fields_->back().DeepCopy((*other.fields_)[i]);
    }
  }
}

void UnknownFieldSet::MergeFrom(const UnknownFieldSet& other) {
  int other_field_count = other.field_count();
  if (other_field_count > 0) {
    if (fields_ == NULL) fields_ = new vector<UnknownField>();
    for (int i = 0; i < other_field_count; i++) {
      fields_->push_back((*other.fields_)[i]);
      fields_->back().DeepCopy((*other.fields_)[i]);
    }
  }
}

// A specialized MergeFrom for performance when we are merging from an UFS that
// is temporary and can be destroyed in the process.
void UnknownFieldSet::MergeFromAndDestroy(UnknownFieldSet* other) {
  int other_field_count = other->field_count();
  if (other_field_count > 0) {
    if (fields_ == NULL) fields_ = new vector<UnknownField>();
    for (int i = 0; i < other_field_count; i++) {
      fields_->push_back((*other->fields_)[i]);
      (*other->fields_)[i].Reset();
    }
  }
  delete other->fields_;
  other->fields_ = NULL;
}

int UnknownFieldSet::SpaceUsedExcludingSelf() const {
  if (fields_ == NULL) return 0;

  int total_size = sizeof(*fields_) + sizeof(UnknownField) * fields_->size();

  for (int i = 0; i < fields_->size(); i++) {
    const UnknownField& field = (*fields_)[i];
    switch (field.type()) {
      case UnknownField::TYPE_LENGTH_DELIMITED:
        total_size += sizeof(*field.length_delimited_.string_value_) +
                      internal::StringSpaceUsedExcludingSelf(
                          *field.length_delimited_.string_value_);
        break;
      case UnknownField::TYPE_GROUP:
        total_size += field.group_->SpaceUsed();
        break;
      default:
        break;
    }
  }
  return total_size;
}

int UnknownFieldSet::SpaceUsed() const {
  return sizeof(*this) + SpaceUsedExcludingSelf();
}

void UnknownFieldSet::AddVarint(int number, uint64 value) {
  UnknownField field;
  field.number_ = number;
  field.SetType(UnknownField::TYPE_VARINT);
  field.varint_ = value;
  if (fields_ == NULL) fields_ = new vector<UnknownField>();
  fields_->push_back(field);
}

void UnknownFieldSet::AddFixed32(int number, uint32 value) {
  UnknownField field;
  field.number_ = number;
  field.SetType(UnknownField::TYPE_FIXED32);
  field.fixed32_ = value;
  if (fields_ == NULL) fields_ = new vector<UnknownField>();
  fields_->push_back(field);
}

void UnknownFieldSet::AddFixed64(int number, uint64 value) {
  UnknownField field;
  field.number_ = number;
  field.SetType(UnknownField::TYPE_FIXED64);
  field.fixed64_ = value;
  if (fields_ == NULL) fields_ = new vector<UnknownField>();
  fields_->push_back(field);
}

string* UnknownFieldSet::AddLengthDelimited(int number) {
  UnknownField field;
  field.number_ = number;
  field.SetType(UnknownField::TYPE_LENGTH_DELIMITED);
  field.length_delimited_.string_value_ = new string;
  if (fields_ == NULL) fields_ = new vector<UnknownField>();
  fields_->push_back(field);
  return field.length_delimited_.string_value_;
}


UnknownFieldSet* UnknownFieldSet::AddGroup(int number) {
  UnknownField field;
  field.number_ = number;
  field.SetType(UnknownField::TYPE_GROUP);
  field.group_ = new UnknownFieldSet;
  if (fields_ == NULL) fields_ = new vector<UnknownField>();
  fields_->push_back(field);
  return field.group_;
}

void UnknownFieldSet::AddField(const UnknownField& field) {
  if (fields_ == NULL) fields_ = new vector<UnknownField>();
  fields_->push_back(field);
  fields_->back().DeepCopy(field);
}

void UnknownFieldSet::DeleteSubrange(int start, int num) {
  // Delete the specified fields.
  for (int i = 0; i < num; ++i) {
    (*fields_)[i + start].Delete();
  }
  // Slide down the remaining fields.
  for (int i = start + num; i < fields_->size(); ++i) {
    (*fields_)[i - num] = (*fields_)[i];
  }
  // Pop off the # of deleted fields.
  for (int i = 0; i < num; ++i) {
    fields_->pop_back();
  }
  if (fields_ && fields_->size() == 0) {
    // maintain invariant: never hold fields_ if empty.
    delete fields_;
    fields_ = NULL;
  }
}

void UnknownFieldSet::DeleteByNumber(int number) {
  if (fields_ == NULL) return;
  int left = 0;  // The number of fields left after deletion.
  for (int i = 0; i < fields_->size(); ++i) {
    UnknownField* field = &(*fields_)[i];
    if (field->number() == number) {
      field->Delete();
    } else {
      if (i != left) {
        (*fields_)[left] = (*fields_)[i];
      }
      ++left;
    }
  }
  fields_->resize(left);
  if (left == 0) {
    // maintain invariant: never hold fields_ if empty.
    delete fields_;
    fields_ = NULL;
  }
}

bool UnknownFieldSet::MergeFromCodedStream(io::CodedInputStream* input) {
  UnknownFieldSet other;
  if (internal::WireFormat::SkipMessage(input, &other) &&
      input->ConsumedEntireMessage()) {
    MergeFromAndDestroy(&other);
    return true;
  } else {
    return false;
  }
}

bool UnknownFieldSet::ParseFromCodedStream(io::CodedInputStream* input) {
  Clear();
  return MergeFromCodedStream(input);
}

bool UnknownFieldSet::ParseFromZeroCopyStream(io::ZeroCopyInputStream* input) {
  io::CodedInputStream coded_input(input);
  return (ParseFromCodedStream(&coded_input) &&
          coded_input.ConsumedEntireMessage());
}

bool UnknownFieldSet::ParseFromArray(const void* data, int size) {
  io::ArrayInputStream input(data, size);
  return ParseFromZeroCopyStream(&input);
}

void UnknownField::Delete() {
  switch (type()) {
    case UnknownField::TYPE_LENGTH_DELIMITED:
      delete length_delimited_.string_value_;
      break;
    case UnknownField::TYPE_GROUP:
      delete group_;
      break;
    default:
      break;
  }
}

// Reset all owned ptrs, a special function for performance, to avoid double
// owning the ptrs, when we merge from a temporary UnknownFieldSet objects.
void UnknownField::Reset() {
  switch (type()) {
    case UnknownField::TYPE_LENGTH_DELIMITED:
      length_delimited_.string_value_ = NULL;
      break;
    case UnknownField::TYPE_GROUP: {
      group_ = NULL;
      break;
    }
    default:
      break;
  }
}

void UnknownField::DeepCopy(const UnknownField& other) {
  switch (type()) {
    case UnknownField::TYPE_LENGTH_DELIMITED:
      length_delimited_.string_value_ = new string(
          *length_delimited_.string_value_);
      break;
    case UnknownField::TYPE_GROUP: {
      UnknownFieldSet* group = new UnknownFieldSet();
      group->InternalMergeFrom(*group_);
      group_ = group;
      break;
    }
    default:
      break;
  }
}


void UnknownField::SerializeLengthDelimitedNoTag(
    io::CodedOutputStream* output) const {
  GOOGLE_DCHECK_EQ(TYPE_LENGTH_DELIMITED, type());
  const string& data = *length_delimited_.string_value_;
  output->WriteVarint32(data.size());
  output->WriteRawMaybeAliased(data.data(), data.size());
}

uint8* UnknownField::SerializeLengthDelimitedNoTagToArray(uint8* target) const {
  GOOGLE_DCHECK_EQ(TYPE_LENGTH_DELIMITED, type());
  const string& data = *length_delimited_.string_value_;
  target = io::CodedOutputStream::WriteVarint32ToArray(data.size(), target);
  target = io::CodedOutputStream::WriteStringToArray(data, target);
  return target;
}

}  // namespace protobuf
}  // namespace google
