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

// arenastring_unittest.cc is not open-sourced. Do not include in open-source
// distribution.

// Based on mvels@'s frankenstring.

#include <google/protobuf/arenastring.h>

#include <string>
#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <cstdlib>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <gtest/gtest.h>

namespace google {
using google::protobuf::internal::ArenaString;
using google::protobuf::internal::ArenaStringPtr;

namespace protobuf {


static string WrapString(const char* value) {
  return value;
}

// Test ArenaStringPtr with arena == NULL.
TEST(ArenaStringPtrTest, ArenaStringPtrOnHeap) {
  ArenaStringPtr field;
  ::std::string default_value = "default";
  field.UnsafeSetDefault(&default_value);
  EXPECT_EQ(string("default"), field.Get(&default_value));
  field.Set(&default_value, WrapString("Test short"), NULL);
  EXPECT_EQ(string("Test short"), field.Get(&default_value));
  field.Set(&default_value, WrapString("Test long long long long value"), NULL);
  EXPECT_EQ(string("Test long long long long value"), field.Get(&default_value));
  field.Set(&default_value, string(""), NULL);
  field.Destroy(&default_value, NULL);

  ArenaStringPtr field2;
  field2.UnsafeSetDefault(&default_value);
  ::std::string* mut = field2.Mutable(&default_value, NULL);
  EXPECT_EQ(mut, field2.Mutable(&default_value, NULL));
  EXPECT_EQ(mut, &field2.Get(&default_value));
  EXPECT_NE(&default_value, mut);
  EXPECT_EQ(string("default"), *mut);
  *mut = "Test long long long long value";  // ensure string allocates storage
  EXPECT_EQ(string("Test long long long long value"), field2.Get(&default_value));
  field2.Destroy(&default_value, NULL);
}

TEST(ArenaStringPtrTest, ArenaStringPtrOnArena) {
  google::protobuf::Arena arena;
  ArenaStringPtr field;
  ::std::string default_value = "default";
  field.UnsafeSetDefault(&default_value);
  EXPECT_EQ(string("default"), field.Get(&default_value));
  field.Set(&default_value, WrapString("Test short"), &arena);
  EXPECT_EQ(string("Test short"), field.Get(&default_value));
  field.Set(&default_value, WrapString("Test long long long long value"), &arena);
  EXPECT_EQ(string("Test long long long long value"),
            field.Get(&default_value));
  field.Set(&default_value, string(""), &arena);
  field.Destroy(&default_value, &arena);

  ArenaStringPtr field2;
  field2.UnsafeSetDefault(&default_value);
  ::std::string* mut = field2.Mutable(&default_value, &arena);
  EXPECT_EQ(mut, field2.Mutable(&default_value, &arena));
  EXPECT_EQ(mut, &field2.Get(&default_value));
  EXPECT_NE(&default_value, mut);
  EXPECT_EQ(string("default"), *mut);
  *mut = "Test long long long long value";  // ensure string allocates storage
  EXPECT_EQ(string("Test long long long long value"),
            field2.Get(&default_value));
  field2.Destroy(&default_value, &arena);
}

}  // namespace protobuf
}  // namespace google
