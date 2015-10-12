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

#ifndef GOOGLE_PROTOBUF_UTIL_CONVERTER_SNAKE2CAMEL_OBJECTWRITER_H__
#define GOOGLE_PROTOBUF_UTIL_CONVERTER_SNAKE2CAMEL_OBJECTWRITER_H__

#include <string>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/stringpiece.h>
#include <google/protobuf/util/internal/object_writer.h>
#include <google/protobuf/util/internal/utility.h>

namespace google {
namespace protobuf {
namespace util {
namespace converter {

// Snake2CamelObjectWriter is an ObjectWriter than translates each field name
// from snake_case to camelCase. Typical usage is:
//     ProtoStreamObjectSource psos(...);
//     JsonObjectWriter jow(...);
//     Snake2CamelObjectWriter snake_to_camel(&jow);
//     psos.writeTo(&snake_to_camel);
class Snake2CamelObjectWriter : public ObjectWriter {
 public:
  explicit Snake2CamelObjectWriter(ObjectWriter* ow)
      : ow_(ow), normalize_case_(true) {}
  virtual ~Snake2CamelObjectWriter() {}

  // ObjectWriter methods.
  virtual Snake2CamelObjectWriter* StartObject(StringPiece name) {
    ow_->StartObject(name);
    return this;
  }

  virtual Snake2CamelObjectWriter* EndObject() {
    ow_->EndObject();
    return this;
  }

  virtual Snake2CamelObjectWriter* StartList(StringPiece name) {
    ow_->StartList(name);
    return this;
  }

  virtual Snake2CamelObjectWriter* EndList() {
    ow_->EndList();
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderBool(StringPiece name, bool value) {
    ow_->RenderBool(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderInt32(StringPiece name, int32 value) {
    ow_->RenderInt32(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderUint32(StringPiece name,
                                                uint32 value) {
    ow_->RenderUint32(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderInt64(StringPiece name, int64 value) {
    ow_->RenderInt64(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderUint64(StringPiece name,
                                                uint64 value) {
    ow_->RenderUint64(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderDouble(StringPiece name,
                                                double value) {
    ow_->RenderDouble(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderFloat(StringPiece name, float value) {
    ow_->RenderFloat(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderString(StringPiece name,
                                                StringPiece value) {
    ow_->RenderString(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderBytes(StringPiece name,
                                               StringPiece value) {
    ow_->RenderBytes(name, value);
    return this;
  }

  virtual Snake2CamelObjectWriter* RenderNull(StringPiece name) {
    ow_->RenderNull(name);
    return this;
  }

  virtual Snake2CamelObjectWriter* DisableCaseNormalizationForNextKey() {
    normalize_case_ = false;
    return this;
  }

 private:
  ObjectWriter* ow_;
  bool normalize_case_;

  bool ShouldNormalizeCase(StringPiece name) {
    if (normalize_case_) {
      return !IsCamel(name);
    } else {
      normalize_case_ = true;
      return false;
    }
  }

  bool IsCamel(StringPiece name) {
    return name.empty() || (ascii_islower(name[0]) && !name.contains("_"));
  }

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Snake2CamelObjectWriter);
};

}  // namespace converter
}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_CONVERTER_SNAKE2CAMEL_OBJECTWRITER_H__
