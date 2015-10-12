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

#ifndef GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_HELPERS_H__
#define GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_HELPERS_H__

#include <string>
#include <vector>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace objectivec {

// Strips ".proto" or ".protodevel" from the end of a filename.
string StripProto(const string& filename);

// Returns true if the name requires a ns_returns_not_retained attribute applied
// to it.
bool IsRetainedName(const string& name);

// Returns true if the name starts with "init" and will need to have special
// handling under ARC.
bool IsInitName(const string& name);

// Gets the name of the file we're going to generate (sans the .pb.h
// extension).  This does not include the path to that file.
string FileName(const FileDescriptor* file);

// Gets the path of the file we're going to generate (sans the .pb.h
// extension).  The path will be dependent on the objectivec package
// declared in the proto package.
string FilePath(const FileDescriptor* file);

// Gets the name of the root class we'll generate in the file.  This class
// is not meant for external consumption, but instead contains helpers that
// the rest of the the classes need
string FileClassName(const FileDescriptor* file);

// These return the fully-qualified class name corresponding to the given
// descriptor.
string ClassName(const Descriptor* descriptor);
string EnumName(const EnumDescriptor* descriptor);

// Returns the fully-qualified name of the enum value corresponding to the
// the descriptor.
string EnumValueName(const EnumValueDescriptor* descriptor);

// Returns the name of the enum value corresponding to the descriptor.
string EnumValueShortName(const EnumValueDescriptor* descriptor);

// Reverse what an enum does.
string UnCamelCaseEnumShortName(const string& name);

// Returns the name to use for the extension (used as the method off the file's
// Root class).
string ExtensionMethodName(const FieldDescriptor* descriptor);

// Returns the transformed field name.
string FieldName(const FieldDescriptor* field);
string FieldNameCapitalized(const FieldDescriptor* field);

// Returns the transformed oneof name.
string OneofEnumName(const OneofDescriptor* descriptor);
string OneofName(const OneofDescriptor* descriptor);
string OneofNameCapitalized(const OneofDescriptor* descriptor);

inline bool HasFieldPresence(const FileDescriptor* file) {
  return file->syntax() != FileDescriptor::SYNTAX_PROTO3;
}

inline bool HasPreservingUnknownEnumSemantics(const FileDescriptor* file) {
  return file->syntax() == FileDescriptor::SYNTAX_PROTO3;
}

inline bool IsMapEntryMessage(const Descriptor* descriptor) {
  return descriptor->options().map_entry();
}

// Reverse of the above.
string UnCamelCaseFieldName(const string& name, const FieldDescriptor* field);

enum ObjectiveCType {
  OBJECTIVECTYPE_INT32,
  OBJECTIVECTYPE_UINT32,
  OBJECTIVECTYPE_INT64,
  OBJECTIVECTYPE_UINT64,
  OBJECTIVECTYPE_FLOAT,
  OBJECTIVECTYPE_DOUBLE,
  OBJECTIVECTYPE_BOOLEAN,
  OBJECTIVECTYPE_STRING,
  OBJECTIVECTYPE_DATA,
  OBJECTIVECTYPE_ENUM,
  OBJECTIVECTYPE_MESSAGE
};

string GetCapitalizedType(const FieldDescriptor* field);

ObjectiveCType GetObjectiveCType(FieldDescriptor::Type field_type);

inline ObjectiveCType GetObjectiveCType(const FieldDescriptor* field) {
  return GetObjectiveCType(field->type());
}

bool IsPrimitiveType(const FieldDescriptor* field);
bool IsReferenceType(const FieldDescriptor* field);

string GPBGenericValueFieldName(const FieldDescriptor* field);
string DefaultValue(const FieldDescriptor* field);

string BuildFlagsString(const vector<string>& strings);

string BuildCommentsString(const SourceLocation& location);

// Checks the prefix for a given file and outputs any warnings needed, if
// there are flat out errors, then out_error is filled in and the result is
// false.
bool ValidateObjCClassPrefix(const FileDescriptor* file, string *out_error);

// Generate decode data needed for ObjC's GPBDecodeTextFormatName() to transform
// the input into the the expected output.
class LIBPROTOC_EXPORT TextFormatDecodeData {
 public:
  TextFormatDecodeData() {}

  void AddString(int32 key, const string& input_for_decode,
                 const string& desired_output);
  size_t num_entries() const { return entries_.size(); }
  string Data() const;

  static string DecodeDataForString(const string& input_for_decode,
                                    const string& desired_output);

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(TextFormatDecodeData);

  typedef std::pair<int32, string> DataEntry;
  vector<DataEntry> entries_;
};

}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_HELPERS_H__
