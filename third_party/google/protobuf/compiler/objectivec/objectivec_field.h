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

#ifndef GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_FIELD_H__
#define GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_FIELD_H__

#include <map>
#include <string>
#include <google/protobuf/compiler/objectivec/objectivec_helpers.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/descriptor.h>

namespace google {
namespace protobuf {

namespace io {
class Printer;  // printer.h
}  // namespace io

namespace compiler {
namespace objectivec {

class FieldGenerator {
 public:
  static FieldGenerator* Make(const FieldDescriptor* field);

  virtual ~FieldGenerator();

  virtual void GenerateFieldStorageDeclaration(io::Printer* printer) const = 0;
  virtual void GeneratePropertyDeclaration(io::Printer* printer) const = 0;

  virtual void GeneratePropertyImplementation(io::Printer* printer) const = 0;

  virtual void GenerateFieldDescription(io::Printer* printer) const;
  virtual void GenerateFieldDescriptionTypeSpecific(io::Printer* printer) const;
  virtual void GenerateFieldNumberConstant(io::Printer* printer) const;

  virtual void GenerateCFunctionDeclarations(io::Printer* printer) const;
  virtual void GenerateCFunctionImplementations(io::Printer* printer) const;

  virtual void DetermineForwardDeclarations(set<string>* fwd_decls) const;

  void SetOneofIndexBase(int index_base);

  string variable(const char* key) const {
    return variables_.find(key)->second;
  }

  bool needs_textformat_name_support() const {
    const string& field_flags = variable("fieldflags");
    return field_flags.find("GPBFieldTextFormatNameCustom") != string::npos;
  }
  string generated_objc_name() const { return variable("name"); }
  string raw_field_name() const { return variable("raw_field_name"); }

 protected:
  explicit FieldGenerator(const FieldDescriptor* descriptor);

  virtual void FinishInitialization(void);
  virtual bool WantsHasProperty(void) const = 0;

  const FieldDescriptor* descriptor_;
  map<string, string> variables_;

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FieldGenerator);
};

class SingleFieldGenerator : public FieldGenerator {
 public:
  virtual ~SingleFieldGenerator();

  virtual void GenerateFieldStorageDeclaration(io::Printer* printer) const;
  virtual void GeneratePropertyDeclaration(io::Printer* printer) const;

  virtual void GeneratePropertyImplementation(io::Printer* printer) const;

 protected:
  explicit SingleFieldGenerator(const FieldDescriptor* descriptor);
  virtual bool WantsHasProperty(void) const;

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(SingleFieldGenerator);
};

// Subclass with common support for when the field ends up as an ObjC Object.
class ObjCObjFieldGenerator : public SingleFieldGenerator {
 public:
  virtual ~ObjCObjFieldGenerator();

  virtual void GenerateFieldStorageDeclaration(io::Printer* printer) const;
  virtual void GeneratePropertyDeclaration(io::Printer* printer) const;

 protected:
  explicit ObjCObjFieldGenerator(const FieldDescriptor* descriptor);

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ObjCObjFieldGenerator);
};

class RepeatedFieldGenerator : public ObjCObjFieldGenerator {
 public:
  virtual ~RepeatedFieldGenerator();

  virtual void GenerateFieldStorageDeclaration(io::Printer* printer) const;
  virtual void GeneratePropertyDeclaration(io::Printer* printer) const;

  virtual void GeneratePropertyImplementation(io::Printer* printer) const;

 protected:
  explicit RepeatedFieldGenerator(const FieldDescriptor* descriptor);
  virtual void FinishInitialization(void);
  virtual bool WantsHasProperty(void) const;

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(RepeatedFieldGenerator);
};

// Convenience class which constructs FieldGenerators for a Descriptor.
class FieldGeneratorMap {
 public:
  explicit FieldGeneratorMap(const Descriptor* descriptor);
  ~FieldGeneratorMap();

  const FieldGenerator& get(const FieldDescriptor* field) const;
  const FieldGenerator& get_extension(int index) const;

  void SetOneofIndexBase(int index_base);

 private:
  const Descriptor* descriptor_;
  scoped_array<scoped_ptr<FieldGenerator> > field_generators_;
  scoped_array<scoped_ptr<FieldGenerator> > extension_generators_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FieldGeneratorMap);
};
}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_FIELD_H__
