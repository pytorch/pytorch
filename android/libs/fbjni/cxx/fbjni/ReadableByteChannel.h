/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#pragma once

#include <fbjni/fbjni.h>
#include <fbjni/ByteBuffer.h>

namespace facebook {
namespace jni {

class JReadableByteChannel : public JavaClass<JReadableByteChannel> {
public:
  static constexpr const char* kJavaDescriptor = "Ljava/nio/channels/ReadableByteChannel;";

  int read(alias_ref<JByteBuffer> dest) const;
};

}}
