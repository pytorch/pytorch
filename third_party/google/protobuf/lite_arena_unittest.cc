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

#include <google/protobuf/arena_test_util.h>
#include <google/protobuf/map_lite_test_util.h>
#include <google/protobuf/testing/googletest.h>
#include <gtest/gtest.h>

namespace google {
namespace protobuf {
namespace {

TEST(LiteArenaTest, MapNoHeapAllocation) {
  // Allocate a large initial block to avoid mallocs during hooked test.
  std::vector<char> arena_block(128 * 1024);
  google::protobuf::ArenaOptions options;
  options.initial_block = &arena_block[0];
  options.initial_block_size = arena_block.size();
  google::protobuf::Arena arena(options);
  string data;
  data.reserve(128 * 1024);

  {
    // TODO(teboring): Enable no heap check when ArenaStringPtr is used in
    // Map.
    // google::protobuf::internal::NoHeapChecker no_heap;

    protobuf_unittest::TestArenaMapLite* from =
        google::protobuf::Arena::CreateMessage<protobuf_unittest::TestArenaMapLite>(&arena);
    google::protobuf::MapLiteTestUtil::SetArenaMapFields(from);
    from->SerializeToString(&data);

    protobuf_unittest::TestArenaMapLite* to =
        google::protobuf::Arena::CreateMessage<protobuf_unittest::TestArenaMapLite>(&arena);
    to->ParseFromString(data);
    google::protobuf::MapLiteTestUtil::ExpectArenaMapFieldsSet(*to);
  }
}

TEST(LiteArenaTest, UnknownFieldMemLeak) {
  google::protobuf::Arena arena;
  protobuf_unittest::ForeignMessageArenaLite* message =
      google::protobuf::Arena::CreateMessage<protobuf_unittest::ForeignMessageArenaLite>(
          &arena);
  string data = "\012\000";
  int original_capacity = data.capacity();
  while (data.capacity() <= original_capacity) {
    data.append("a");
  }
  data[1] = data.size() - 2;
  message->ParseFromString(data);
}

}  // namespace
}  // namespace protobuf
}  // namespace google
