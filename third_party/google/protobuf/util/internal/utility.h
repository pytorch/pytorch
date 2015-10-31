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

#ifndef GOOGLE_PROTOBUF_UTIL_CONVERTER_UTILITY_H__
#define GOOGLE_PROTOBUF_UTIL_CONVERTER_UTILITY_H__

#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <string>
#include <utility>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/type.pb.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/stubs/stringpiece.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/status.h>
#include <google/protobuf/stubs/statusor.h>


namespace google {
namespace protobuf {
class Method;
class Any;
class Bool;
class Option;
class Field;
class Type;
class Enum;
class EnumValue;
}  // namespace protobuf


namespace protobuf {
namespace util {
namespace converter {
// Finds the tech option identified by option_name. Parses the boolean value and
// returns it.
// When the option with the given name is not found, default_value is returned.
LIBPROTOBUF_EXPORT bool GetBoolOptionOrDefault(
    const google::protobuf::RepeatedPtrField<google::protobuf::Option>& options,
    const string& option_name, bool default_value);

// Returns int64 option value. If the option isn't found, returns the
// default_value.
LIBPROTOBUF_EXPORT int64 GetInt64OptionOrDefault(
    const google::protobuf::RepeatedPtrField<google::protobuf::Option>& options,
    const string& option_name, int64 default_value);

// Returns double option value. If the option isn't found, returns the
// default_value.
LIBPROTOBUF_EXPORT double GetDoubleOptionOrDefault(
    const google::protobuf::RepeatedPtrField<google::protobuf::Option>& options,
    const string& option_name, double default_value);

// Returns string option value. If the option isn't found, returns the
// default_value.
LIBPROTOBUF_EXPORT string GetStringOptionOrDefault(
    const google::protobuf::RepeatedPtrField<google::protobuf::Option>& options,
    const string& option_name, const string& default_value);

// Returns a boolean value contained in Any type.
// TODO(skarvaje): Make these utilities dealing with Any types more generic,
// add more error checking and move to a more public/sharable location so others
// can use.
LIBPROTOBUF_EXPORT bool GetBoolFromAny(const google::protobuf::Any& any);

// Returns int64 value contained in Any type.
LIBPROTOBUF_EXPORT int64 GetInt64FromAny(const google::protobuf::Any& any);

// Returns double value contained in Any type.
LIBPROTOBUF_EXPORT double GetDoubleFromAny(const google::protobuf::Any& any);

// Returns string value contained in Any type.
LIBPROTOBUF_EXPORT string GetStringFromAny(const google::protobuf::Any& any);

// Returns the type string without the url prefix. e.g.: If the passed type is
// 'type.googleapis.com/tech.type.Bool', the returned value is 'tech.type.Bool'.
LIBPROTOBUF_EXPORT const StringPiece GetTypeWithoutUrl(StringPiece type_url);

// Returns the simple_type with the base type url (kTypeServiceBaseUrl)
// prefixed.
//
// E.g:
// GetFullTypeWithUrl("google.protobuf.Timestamp") returns the string
// "type.googleapis.com/google.protobuf.Timestamp".
LIBPROTOBUF_EXPORT const string GetFullTypeWithUrl(StringPiece simple_type);

// Finds and returns option identified by name and option_name within the
// provided map. Returns NULL if none found.
const google::protobuf::Option* FindOptionOrNull(
    const google::protobuf::RepeatedPtrField<google::protobuf::Option>& options,
    const string& option_name);

// Finds and returns the field identified by field_name in the passed tech Type
// object. Returns NULL if none found.
const google::protobuf::Field* FindFieldInTypeOrNull(
    const google::protobuf::Type* type, StringPiece field_name);

// Finds and returns the EnumValue identified by enum_name in the passed tech
// Enum object. Returns NULL if none found.
const google::protobuf::EnumValue* FindEnumValueByNameOrNull(
    const google::protobuf::Enum* enum_type, StringPiece enum_name);

// Finds and returns the EnumValue identified by value in the passed tech
// Enum object. Returns NULL if none found.
const google::protobuf::EnumValue* FindEnumValueByNumberOrNull(
    const google::protobuf::Enum* enum_type, int32 value);

// Converts input to camel-case and returns it.
LIBPROTOBUF_EXPORT string ToCamelCase(const StringPiece input);

// Converts input to snake_case and returns it.
LIBPROTOBUF_EXPORT string ToSnakeCase(StringPiece input);

// Returns true if type_name represents a well-known type.
LIBPROTOBUF_EXPORT bool IsWellKnownType(const string& type_name);

// Returns true if 'bool_string' represents a valid boolean value. Only "true",
// "false", "0" and "1" are allowed.
LIBPROTOBUF_EXPORT bool IsValidBoolString(const string& bool_string);

// Returns true if "field" is a protobuf map field based on its type.
LIBPROTOBUF_EXPORT bool IsMap(const google::protobuf::Field& field,
           const google::protobuf::Type& type);

// Returns true if the given type has special MessageSet wire format.
bool IsMessageSetWireFormat(const google::protobuf::Type& type);

// Infinity/NaN-aware conversion to string.
LIBPROTOBUF_EXPORT string DoubleAsString(double value);
LIBPROTOBUF_EXPORT string FloatAsString(float value);

// Convert from int32, int64, uint32, uint64, double or float to string.
template <typename T>
string ValueAsString(T value) {
  return SimpleItoa(value);
}

template <>
inline string ValueAsString(float value) {
  return FloatAsString(value);
}

template <>
inline string ValueAsString(double value) {
  return DoubleAsString(value);
}

// Converts a string to float. Unlike safe_strtof, conversion will fail if the
// value fits into double but not float (e.g., DBL_MAX).
LIBPROTOBUF_EXPORT bool SafeStrToFloat(StringPiece str, float* value);
}  // namespace converter
}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_CONVERTER_UTILITY_H__
