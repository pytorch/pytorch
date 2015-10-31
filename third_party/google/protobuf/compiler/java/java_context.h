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

#ifndef GOOGLE_PROTOBUF_COMPILER_JAVA_CONTEXT_H__
#define GOOGLE_PROTOBUF_COMPILER_JAVA_CONTEXT_H__

#include <map>
#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <vector>

#include <google/protobuf/stubs/common.h>

namespace google {
namespace protobuf {
  class FileDescriptor;
  class FieldDescriptor;
  class OneofDescriptor;
  class Descriptor;
  namespace compiler {
    namespace java {
      class ClassNameResolver;  // name_resolver.h
    }
  }
}  // namespace protobuf

namespace protobuf {
namespace compiler {
namespace java {

struct FieldGeneratorInfo;
struct OneofGeneratorInfo;
// A context object holds the information that is shared among all code
// generators.
class Context {
 public:
  explicit Context(const FileDescriptor* file);
  ~Context();

  // Get the name resolver associated with this context. The resolver
  // can be used to map descriptors to Java class names.
  ClassNameResolver* GetNameResolver();

  // Get the FieldGeneratorInfo for a given field.
  const FieldGeneratorInfo* GetFieldGeneratorInfo(
      const FieldDescriptor* field) const;

  // Get the OneofGeneratorInfo for a given oneof.
  const OneofGeneratorInfo* GetOneofGeneratorInfo(
      const OneofDescriptor* oneof) const;

 private:
  void InitializeFieldGeneratorInfo(const FileDescriptor* file);
  void InitializeFieldGeneratorInfoForMessage(const Descriptor* message);
  void InitializeFieldGeneratorInfoForFields(
      const vector<const FieldDescriptor*>& fields);

  google::protobuf::scoped_ptr<ClassNameResolver> name_resolver_;
  map<const FieldDescriptor*, FieldGeneratorInfo> field_generator_info_map_;
  map<const OneofDescriptor*, OneofGeneratorInfo> oneof_generator_info_map_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Context);
};

}  // namespace java
}  // namespace compiler
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_JAVA_CONTEXT_H__
