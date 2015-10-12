// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// http://code.google.com/p/protobuf/
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

#ifndef GOOGLE_PROTOBUF_COMPILER_JAVANANO_FIELD_H__
#define GOOGLE_PROTOBUF_COMPILER_JAVANANO_FIELD_H__

#include <map>
#include <string>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/compiler/javanano/javanano_params.h>

namespace google {
namespace protobuf {
  namespace io {
    class Printer;             // printer.h
  }
}

namespace protobuf {
namespace compiler {
namespace javanano {

class FieldGenerator {
 public:
  FieldGenerator(const Params& params) : params_(params) {}
  virtual ~FieldGenerator();

  virtual bool SavedDefaultNeeded() const;
  virtual void GenerateInitSavedDefaultCode(io::Printer* printer) const;

  // Generates code for Java fields and methods supporting this field.
  // If this field needs a saved default (SavedDefaultNeeded() is true),
  // then @lazy_init controls how the static field for that default value
  // and its initialization code should be generated. If @lazy_init is
  // true, the static field is not declared final and the initialization
  // code is generated only when GenerateInitSavedDefaultCode is called;
  // otherwise, the static field is declared final and initialized inline.
  // GenerateInitSavedDefaultCode will not be called in the latter case.
  virtual void GenerateMembers(
      io::Printer* printer, bool lazy_init) const = 0;

  virtual void GenerateClearCode(io::Printer* printer) const = 0;
  virtual void GenerateMergingCode(io::Printer* printer) const = 0;

  // Generates code to merge from packed serialized form. The default
  // implementation will fail; subclasses which can handle packed serialized
  // forms will override this and print appropriate code to the printer.
  virtual void GenerateMergingCodeFromPacked(io::Printer* printer) const;

  virtual void GenerateSerializationCode(io::Printer* printer) const = 0;
  virtual void GenerateSerializedSizeCode(io::Printer* printer) const = 0;
  virtual void GenerateEqualsCode(io::Printer* printer) const = 0;
  virtual void GenerateHashCodeCode(io::Printer* printer) const = 0;
  virtual void GenerateFixClonedCode(io::Printer* printer) const {}

 protected:
  const Params& params_;
 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FieldGenerator);
};

// Convenience class which constructs FieldGenerators for a Descriptor.
class FieldGeneratorMap {
 public:
  explicit FieldGeneratorMap(const Descriptor* descriptor, const Params &params);
  ~FieldGeneratorMap();

  const FieldGenerator& get(const FieldDescriptor* field) const;
  int total_bits() const { return total_bits_; }
  bool saved_defaults_needed() const { return saved_defaults_needed_; }

 private:
  const Descriptor* descriptor_;
  scoped_array<scoped_ptr<FieldGenerator> > field_generators_;
  int total_bits_;
  bool saved_defaults_needed_;

  static FieldGenerator* MakeGenerator(const FieldDescriptor* field,
      const Params &params, int* next_has_bit_index);

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FieldGeneratorMap);
};

void SetCommonOneofVariables(const FieldDescriptor* descriptor,
                             map<string, string>* variables);
void GenerateOneofFieldEquals(const FieldDescriptor* descriptor,
                              const map<string, string>& variables,
                              io::Printer* printer);
void GenerateOneofFieldHashCode(const FieldDescriptor* descriptor,
                                const map<string, string>& variables,
                                io::Printer* printer);

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_JAVANANO_FIELD_H__
