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

#ifndef GOOGLE_PROTOBUF_UTIL_UNKNOWN_ENUM_IMPL_H__
#define GOOGLE_PROTOBUF_UTIL_UNKNOWN_ENUM_IMPL_H__

#include <stdlib.h>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/bridge/compatibility_mode_support.h>

namespace google {
namespace protobuf {

// google/protobuf/message.h
class Message;

namespace util {

// NOTE: You should not call these functions directly.  Instead use either
// HAS_UNKNOWN_ENUM() or GET_UNKNOWN_ENUM(), defined in the public header.
// The macro-versions operate in a type-safe manner and behave appropriately
// for the proto version of the message, whereas these versions assume a
// specific proto version and allow the caller to pass in any arbitrary integer
// value as a field number.
//
// Returns whether the message has unrecognized the enum value for a given
// field. It also stores the value into the unknown_value parameter if the
// function returns true and the pointer is not NULL.
//
// In proto2, invalid enum values will be treated as unknown fields. This
// function checks that case.
bool HasUnknownEnum(const Message& message, int32 field_number,
                    int32* unknown_value = NULL);
// Same as above, but returns all unknown enums.
bool GetRepeatedEnumUnknowns(const Message& message, int32 field_number,
                             vector<int32>* unknown_values = NULL);
// In proto1, invalue enum values are stored in the same way as valid enum
// values.
// TODO(karner): Delete this once the migration to proto2 is complete.
bool HasUnknownEnumProto1(const Message& message, int32 field_number,
                          int32* unknown_value);
// Same as above, but returns all unknown enums.
bool GetRepeatedEnumUnknownsProto1(const Message& message, int32 field_number,
                                   vector<int32>* unknown_values);
// Invokes the appropriate version based on whether the message is proto1
// or proto2.
template <typename T>
bool HasUnknownEnum_Template(const T& message, int32 field_number,
                             int32* unknown_value = NULL) {
  if (internal::is_base_of<bridge::internal::Proto1CompatibleMessage, T>::value ||
      !internal::is_base_of<ProtocolMessage, T>::value) {
    return HasUnknownEnum(message, field_number, unknown_value);
  } else {
    return HasUnknownEnumProto1(message, field_number, unknown_value);
  }
}
// Invokes the appropriate version based on whether the message is proto1
// or proto2.
template <typename T>
bool GetRepeatedEnumUnknowns_Template(
    const T& message, int32 field_number,
    vector<int32>* unknown_values = NULL) {
  if (internal::is_base_of<bridge::internal::Proto1CompatibleMessage, T>::value ||
      !internal::is_base_of<ProtocolMessage, T>::value) {
    return GetRepeatedEnumUnknowns(message, field_number, unknown_values);
  } else {
    return GetRepeatedEnumUnknownsProto1(message, field_number,
                                         unknown_values);
  }
}

// NOTE: You should not call these functions directly.  Instead use
// CLEAR_UNKNOWN_ENUM(), defined in the public header.  The macro-versions
// operate in a type-safe manner and behave appropriately for the proto
// version of the message, whereas these versions assume a specific proto
// version and allow the caller to pass in any arbitrary integer value as a
// field number.
//
// Clears the unknown entries of the given field of the message.
void ClearUnknownEnum(Message* message, int32 field_number);
// In proto1, clears the field if the value is out of range.
// TODO(karner): Delete this or make it proto2-only once the migration
// to proto2 is complete.
void ClearUnknownEnumProto1(Message* message, int32 field_number);
template <typename T>
void ClearUnknownEnum_Template(T* message, int32 field_number) {
  if (internal::is_base_of<bridge::internal::Proto1CompatibleMessage, T>::value ||
      !internal::is_base_of<ProtocolMessage, T>::value) {
    ClearUnknownEnum(message, field_number);
  } else {
    ClearUnknownEnumProto1(message, field_number);
  }
}

// NOTE: You should not call these functions directly.  Instead use
// SET_UNKNOWN_ENUM(), defined in the public header.  The macro-versions
// operate in a type-safe manner and behave appropriately for the proto
// version of the message, whereas these versions assume a specific proto
// version and allow the caller to pass in any arbitrary integer value as a
// field number.
//
// Sets the given value in the unknown fields of the message.
void SetUnknownEnum(Message* message, int32 field_number, int32 unknown_value);
// In proto1, invalue enum values are stored in the same way as valid enum
// values.
// TODO(karner): Delete this once the migration to proto2 is complete.
void SetUnknownEnumProto1(Message* message, int32 field_number,
                          int32 unknown_value);
// Invokes the appropriate version based on whether the message is proto1
// or proto2.
template <typename T>
void SetUnknownEnum_Template(T* message, int32 field_number,
                             int32 unknown_value) {
  if (internal::is_base_of<bridge::internal::Proto1CompatibleMessage, T>::value ||
      !internal::is_base_of<ProtocolMessage, T>::value) {
    SetUnknownEnum(message, field_number, unknown_value);
  } else {
    SetUnknownEnumProto1(message, field_number, unknown_value);
  }
}

}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_UNKNOWN_ENUM_IMPL_H__
