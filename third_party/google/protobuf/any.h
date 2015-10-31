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

#ifndef GOOGLE_PROTOBUF_ANY_H__
#define GOOGLE_PROTOBUF_ANY_H__

#include <string>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/arenastring.h>

namespace google {
namespace protobuf {
namespace internal {

// Helper class used to implement google::protobuf::Any.
class LIBPROTOBUF_EXPORT AnyMetadata {
  typedef ArenaStringPtr UrlType;
  typedef ArenaStringPtr ValueType;
 public:
  // AnyMetadata does not take ownership of "type_url" and "value".
  AnyMetadata(UrlType* type_url, ValueType* value);

  void PackFrom(const Message& message);

  bool UnpackTo(Message* message) const;

  template<typename T>
  bool Is() const {
    return InternalIs(T::default_instance().GetDescriptor());
  }

 private:
  bool InternalIs(const Descriptor* message) const;

  UrlType* type_url_;
  ValueType* value_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(AnyMetadata);
};

extern const char kAnyFullTypeName[];          // "google.protobuf.Any".
extern const char kTypeGoogleApisComPrefix[];  // "type.googleapis.com/".
extern const char kTypeGoogleProdComPrefix[];  // "type.googleprod.com/".

// Get the proto type name from Any::type_url value. For example, passing
// "type.googleapis.com/rpc.QueryOrigin" will return "rpc.QueryOrigin" in
// *full_type_name. Returns false if type_url does not start with
// "type.googleapis.com".
bool ParseAnyTypeUrl(const string& type_url, string* full_type_name);

// See if message is of type google.protobuf.Any, if so, return the descriptors
// for "type_url" and "value" fields.
bool GetAnyFieldDescriptors(const Message& message,
                            const FieldDescriptor** type_url_field,
                            const FieldDescriptor** value_field);

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_ANY_H__
