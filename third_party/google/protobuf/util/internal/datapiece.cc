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

#include <google/protobuf/util/internal/datapiece.h>

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/type.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/util/internal/utility.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/mathutil.h>
#include <google/protobuf/stubs/mathlimits.h>

namespace google {
namespace protobuf {
namespace util {
namespace converter {

using google::protobuf::EnumDescriptor;
using google::protobuf::EnumValueDescriptor;
;
;
using util::error::Code;
using util::Status;
using util::StatusOr;

namespace {

inline Status InvalidArgument(StringPiece value_str) {
  return Status(util::error::INVALID_ARGUMENT, value_str);
}

// For general conversion between
//     int32, int64, uint32, uint64, double and float
// except conversion between double and float.
template <typename To, typename From>
StatusOr<To> NumberConvertAndCheck(From before) {
  if (::google::protobuf::internal::is_same<From, To>::value) return before;
  To after = static_cast<To>(before);
  if (after == before &&
      MathUtil::Sign<From>(before) == MathUtil::Sign<To>(after)) {
    return after;
  } else {
    return InvalidArgument(::google::protobuf::internal::is_integral<From>::value
                               ? ValueAsString(before)
                               : ::google::protobuf::internal::is_same<From, double>::value
                                     ? DoubleAsString(before)
                                     : FloatAsString(before));
  }
}

// For conversion between double and float only.
template <typename To, typename From>
StatusOr<To> FloatingPointConvertAndCheck(From before) {
  if (MathLimits<From>::IsNaN(before)) {
    return std::numeric_limits<To>::quiet_NaN();
  }

  To after = static_cast<To>(before);
  if (MathUtil::AlmostEquals<To>(after, before)) {
    return after;
  } else {
    return InvalidArgument(::google::protobuf::internal::is_same<From, double>::value
                               ? DoubleAsString(before)
                               : FloatAsString(before));
  }
}

}  // namespace

StatusOr<int32> DataPiece::ToInt32() const {
  if (type_ == TYPE_STRING) {
    return StringToNumber<int32>(safe_strto32);
  }
  return GenericConvert<int32>();
}

StatusOr<uint32> DataPiece::ToUint32() const {
  if (type_ == TYPE_STRING) {
    return StringToNumber<uint32>(safe_strtou32);
  }
  return GenericConvert<uint32>();
}

StatusOr<int64> DataPiece::ToInt64() const {
  if (type_ == TYPE_STRING) {
    return StringToNumber<int64>(safe_strto64);
  }
  return GenericConvert<int64>();
}

StatusOr<uint64> DataPiece::ToUint64() const {
  if (type_ == TYPE_STRING) {
    return StringToNumber<uint64>(safe_strtou64);
  }
  return GenericConvert<uint64>();
}

StatusOr<double> DataPiece::ToDouble() const {
  if (type_ == TYPE_FLOAT) {
    return FloatingPointConvertAndCheck<double, float>(float_);
  }
  if (type_ == TYPE_STRING) {
    if (str_ == "Infinity") return std::numeric_limits<double>::infinity();
    if (str_ == "-Infinity") return -std::numeric_limits<double>::infinity();
    if (str_ == "NaN") return std::numeric_limits<double>::quiet_NaN();
    return StringToNumber<double>(safe_strtod);
  }
  return GenericConvert<double>();
}

StatusOr<float> DataPiece::ToFloat() const {
  if (type_ == TYPE_DOUBLE) {
    return FloatingPointConvertAndCheck<float, double>(double_);
  }
  if (type_ == TYPE_STRING) {
    if (str_ == "Infinity") return std::numeric_limits<float>::infinity();
    if (str_ == "-Infinity") return -std::numeric_limits<float>::infinity();
    if (str_ == "NaN") return std::numeric_limits<float>::quiet_NaN();
    // SafeStrToFloat() is used instead of safe_strtof() because the later
    // does not fail on inputs like SimpleDtoa(DBL_MAX).
    return StringToNumber<float>(SafeStrToFloat);
  }
  return GenericConvert<float>();
}

StatusOr<bool> DataPiece::ToBool() const {
  switch (type_) {
    case TYPE_BOOL:
      return bool_;
    case TYPE_STRING:
      return StringToNumber<bool>(safe_strtob);
    default:
      return InvalidArgument(
          ValueAsStringOrDefault("Wrong type. Cannot convert to Bool."));
  }
}

StatusOr<string> DataPiece::ToString() const {
  switch (type_) {
    case TYPE_STRING:
      return str_.ToString();
    case TYPE_BYTES: {
      string base64;
      Base64Escape(str_, &base64);
      return base64;
    }
    default:
      return InvalidArgument(
          ValueAsStringOrDefault("Cannot convert to string."));
  }
}

string DataPiece::ValueAsStringOrDefault(StringPiece default_string) const {
  switch (type_) {
    case TYPE_INT32:
      return SimpleItoa(i32_);
    case TYPE_INT64:
      return SimpleItoa(i64_);
    case TYPE_UINT32:
      return SimpleItoa(u32_);
    case TYPE_UINT64:
      return SimpleItoa(u64_);
    case TYPE_DOUBLE:
      return DoubleAsString(double_);
    case TYPE_FLOAT:
      return FloatAsString(float_);
    case TYPE_BOOL:
      return SimpleBtoa(bool_);
    case TYPE_STRING:
      return StrCat("\"", str_.ToString(), "\"");
    case TYPE_BYTES: {
      string base64;
      WebSafeBase64Escape(str_, &base64);
      return StrCat("\"", base64, "\"");
    }
    case TYPE_NULL:
      return "null";
    default:
      return default_string.ToString();
  }
}

StatusOr<string> DataPiece::ToBytes() const {
  if (type_ == TYPE_BYTES) return str_.ToString();
  if (type_ == TYPE_STRING) {
    string decoded;
    if (!WebSafeBase64Unescape(str_, &decoded)) {
      if (!Base64Unescape(str_, &decoded)) {
        return InvalidArgument(
            ValueAsStringOrDefault("Invalid data in input."));
      }
    }
    return decoded;
  } else {
    return InvalidArgument(ValueAsStringOrDefault(
        "Wrong type. Only String or Bytes can be converted to Bytes."));
  }
}

StatusOr<int> DataPiece::ToEnum(const google::protobuf::Enum* enum_type) const {
  if (type_ == TYPE_NULL) return google::protobuf::NULL_VALUE;

  if (type_ == TYPE_STRING) {
    // First try the given value as a name.
    string enum_name = str_.ToString();
    const google::protobuf::EnumValue* value =
        FindEnumValueByNameOrNull(enum_type, enum_name);
    if (value != NULL) return value->number();
    // Next try a normalized name.
    for (string::iterator it = enum_name.begin(); it != enum_name.end(); ++it) {
      *it = *it == '-' ? '_' : ascii_toupper(*it);
    }
    value = FindEnumValueByNameOrNull(enum_type, enum_name);
    if (value != NULL) return value->number();
  } else {
    StatusOr<int32> value = ToInt32();
    if (value.ok()) {
      if (const google::protobuf::EnumValue* enum_value =
              FindEnumValueByNumberOrNull(enum_type, value.ValueOrDie())) {
        return enum_value->number();
      }
    }
  }
  return InvalidArgument(
      ValueAsStringOrDefault("Cannot find enum with given value."));
}

template <typename To>
StatusOr<To> DataPiece::GenericConvert() const {
  switch (type_) {
    case TYPE_INT32:
      return NumberConvertAndCheck<To, int32>(i32_);
    case TYPE_INT64:
      return NumberConvertAndCheck<To, int64>(i64_);
    case TYPE_UINT32:
      return NumberConvertAndCheck<To, uint32>(u32_);
    case TYPE_UINT64:
      return NumberConvertAndCheck<To, uint64>(u64_);
    case TYPE_DOUBLE:
      return NumberConvertAndCheck<To, double>(double_);
    case TYPE_FLOAT:
      return NumberConvertAndCheck<To, float>(float_);
    default:  // TYPE_ENUM, TYPE_STRING, TYPE_CORD, TYPE_BOOL
      return InvalidArgument(ValueAsStringOrDefault(
          "Wrong type. Bool, Enum, String and Cord not supported in "
          "GenericConvert."));
  }
}

template <typename To>
StatusOr<To> DataPiece::StringToNumber(bool (*func)(StringPiece, To*)) const {
  To result;
  if (func(str_, &result)) return result;
  return InvalidArgument(StrCat("\"", str_.ToString(), "\""));
}

}  // namespace converter
}  // namespace util
}  // namespace protobuf
}  // namespace google
