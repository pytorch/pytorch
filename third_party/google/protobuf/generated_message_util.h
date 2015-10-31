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
// This file contains miscellaneous helper code used by generated code --
// including lite types -- but which should not be used directly by users.

#ifndef GOOGLE_PROTOBUF_GENERATED_MESSAGE_UTIL_H__
#define GOOGLE_PROTOBUF_GENERATED_MESSAGE_UTIL_H__

#include <assert.h>
#include <string>

#include <google/protobuf/stubs/once.h>

#include <google/protobuf/stubs/common.h>
namespace google {

namespace protobuf {

class Arena;
namespace io { class CodedInputStream; }

namespace internal {


// Annotation for the compiler to emit a deprecation message if a field marked
// with option 'deprecated=true' is used in the code, or for other things in
// generated code which are deprecated.
//
// For internal use in the pb.cc files, deprecation warnings are suppressed
// there.
#undef DEPRECATED_PROTOBUF_FIELD
#define PROTOBUF_DEPRECATED


// Constants for special floating point values.
LIBPROTOBUF_EXPORT double Infinity();
LIBPROTOBUF_EXPORT double NaN();

// TODO(jieluo): Change to template. We have tried to use template,
// but it causes net/rpc/python:rpcutil_test fail (the empty string will
// init twice). It may related to swig. Change to template after we
// found the solution.

// Default empty string object. Don't use the pointer directly. Instead, call
// GetEmptyString() to get the reference.
LIBPROTOBUF_EXPORT extern const ::std::string* empty_string_;
LIBPROTOBUF_EXPORT extern ProtobufOnceType empty_string_once_init_;
LIBPROTOBUF_EXPORT void InitEmptyString();


LIBPROTOBUF_EXPORT inline const ::std::string& GetEmptyStringAlreadyInited() {
  assert(empty_string_ != NULL);
  return *empty_string_;
}
LIBPROTOBUF_EXPORT inline const ::std::string& GetEmptyString() {
  ::google::protobuf::GoogleOnceInit(&empty_string_once_init_, &InitEmptyString);
  return GetEmptyStringAlreadyInited();
}

LIBPROTOBUF_EXPORT int StringSpaceUsedExcludingSelf(const string& str);


// True if IsInitialized() is true for all elements of t.  Type is expected
// to be a RepeatedPtrField<some message type>.  It's useful to have this
// helper here to keep the protobuf compiler from ever having to emit loops in
// IsInitialized() methods.  We want the C++ compiler to inline this or not
// as it sees fit.
template <class Type> bool AllAreInitialized(const Type& t) {
  for (int i = t.size(); --i >= 0; ) {
    if (!t.Get(i).IsInitialized()) return false;
  }
  return true;
}

class ArenaString;

// Read a length (varint32), followed by a string, from *input.  Return a
// pointer to a copy of the string that resides in *arena.  Requires both
// args to be non-NULL.  If something goes wrong while reading the data
// then NULL is returned (e.g., input does not start with a valid varint).
ArenaString* ReadArenaString(::google::protobuf::io::CodedInputStream* input,
                             ::google::protobuf::Arena* arena);

}  // namespace internal
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_GENERATED_MESSAGE_UTIL_H__
