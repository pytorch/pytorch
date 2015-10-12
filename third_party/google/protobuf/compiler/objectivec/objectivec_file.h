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

#ifndef GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_FILE_H__
#define GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_FILE_H__

#include <string>
#include <set>
#include <vector>
#include <google/protobuf/compiler/objectivec/objectivec_helpers.h>
#include <google/protobuf/stubs/common.h>

namespace google {
namespace protobuf {
class FileDescriptor;  // descriptor.h
namespace io {
class Printer;  // printer.h
}
}

namespace protobuf {
namespace compiler {
namespace objectivec {

class EnumGenerator;
class ExtensionGenerator;
class MessageGenerator;

class FileGenerator {
 public:
  explicit FileGenerator(const FileDescriptor* file);
  ~FileGenerator();

  void GenerateSource(io::Printer* printer);
  void GenerateHeader(io::Printer* printer);

  const string& RootClassName() const { return root_class_name_; }
  const string Path() const;

  bool IsPublicDependency() const { return is_public_dep_; }

 protected:
  void SetIsPublicDependency(bool is_public_dep) {
    is_public_dep_ = is_public_dep;
  }

 private:
  const FileDescriptor* file_;
  string root_class_name_;

  // Access this field through the DependencyGenerators accessor call below.
  // Do not reference it directly.
  vector<FileGenerator*> dependency_generators_;

  vector<EnumGenerator*> enum_generators_;
  vector<MessageGenerator*> message_generators_;
  vector<ExtensionGenerator*> extension_generators_;
  bool is_public_dep_;

  const vector<FileGenerator*>& DependencyGenerators();

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FileGenerator);
};

}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_FILE_H__
