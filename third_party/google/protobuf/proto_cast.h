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

#ifndef GOOGLE_PROTOBUF_UTIL_PROTO_CAST_H__
#define GOOGLE_PROTOBUF_UTIL_PROTO_CAST_H__

#include <string>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>

// proto_cast<> is used to simulate over-the-wire conversion of one
// proto message into another.  This is primarily useful for unit tests
// which validate the version-compatibility semantics of protobufs.
// Usage is similar to C++-style typecasts:
//
// OldMessage old_message = /*...*/;
// NewMessage new_message = proto_cast<NewMessage>(old_message);
namespace google {
template<typename NewProto,
         typename OldProto>
NewProto proto_cast(const OldProto& old_proto) {
  string wire_format;
  GOOGLE_CHECK(old_proto.SerializeToString(&wire_format));

  NewProto new_proto;
  GOOGLE_CHECK(new_proto.ParseFromString(wire_format));
  return new_proto;
}

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_PROTO_CAST_H__
