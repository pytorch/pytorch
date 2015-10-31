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

#include <iostream>
#include <google/protobuf/compiler/objectivec/objectivec_generator.h>
#include <google/protobuf/compiler/objectivec/objectivec_file.h>
#include <google/protobuf/compiler/objectivec/objectivec_helpers.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace objectivec {

ObjectiveCGenerator::ObjectiveCGenerator() {}

ObjectiveCGenerator::~ObjectiveCGenerator() {}

bool ObjectiveCGenerator::Generate(const FileDescriptor* file,
                                   const string& parameter,
                                   OutputDirectory* output_directory,
                                   string* error) const {
  // ObjC doesn't have any options at the moment, error if passed one.
  vector<pair<string, string> > options;
  ParseGeneratorParameter(parameter, &options);
  for (int i = 0; i < options.size(); i++) {
    *error = "error:: Unknown generator option: " + options[i].first;
    return false;
  }

  // Validate the objc prefix/package pairing.
  if (!ValidateObjCClassPrefix(file, error)) {
    // *error will have been filled in.
    return false;
  }

  FileGenerator file_generator(file);
  string filepath = FilePath(file);

  // Generate header.
  {
    scoped_ptr<io::ZeroCopyOutputStream> output(
        output_directory->Open(filepath + ".pbobjc.h"));
    io::Printer printer(output.get(), '$');
    file_generator.GenerateHeader(&printer);
  }

  // Generate m file.
  {
    scoped_ptr<io::ZeroCopyOutputStream> output(
        output_directory->Open(filepath + ".pbobjc.m"));
    io::Printer printer(output.get(), '$');
    file_generator.GenerateSource(&printer);
  }

  return true;
}

}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
