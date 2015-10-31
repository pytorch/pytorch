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

#ifndef GOOGLE_PROTOBUF_COMPILER_JAVANANO_HELPERS_H__
#define GOOGLE_PROTOBUF_COMPILER_JAVANANO_HELPERS_H__

#include <string>
#include <google/protobuf/compiler/javanano/javanano_params.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/printer.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

// Commonly-used separator comments.  Thick is a line of '=', thin is a line
// of '-'.
extern const char kThickSeparator[];
extern const char kThinSeparator[];

// Converts the field's name to camel-case, e.g. "foo_bar_baz" becomes
// "fooBarBaz" or "FooBarBaz", respectively.
string UnderscoresToCamelCase(const FieldDescriptor* field);
string UnderscoresToCamelCase(const OneofDescriptor* oneof);
string UnderscoresToCapitalizedCamelCase(const FieldDescriptor* field);
string UnderscoresToCapitalizedCamelCase(const OneofDescriptor* oneof);

// Appends an "_" to the end of a field where the name is a reserved java
// keyword.  For example int32 public = 1 will generate int public_.
string RenameJavaKeywords(const string& input);

// Similar, but for method names.  (Typically, this merely has the effect
// of lower-casing the first letter of the name.)
string UnderscoresToCamelCase(const MethodDescriptor* method);

// Strips ".proto" or ".protodevel" from the end of a filename.
string StripProto(const string& filename);

// Gets the unqualified class name for the file.  Each .proto file becomes a
// single Java class, with all its contents nested in that class.
string FileClassName(const Params& params, const FileDescriptor* file);

// Returns the file's Java package name.
string FileJavaPackage(const Params& params, const FileDescriptor* file);

// Returns whether the Java outer class is needed, i.e. whether the option
// java_multiple_files is false, or the proto file contains any file-scope
// enums/extensions.
bool IsOuterClassNeeded(const Params& params, const FileDescriptor* file);

// Converts the given simple name of a proto entity to its fully-qualified name
// in the Java namespace, given that it is in the given file enclosed in the
// given parent message (or NULL for file-scope entities). Whether the file's
// outer class name should be included in the return value depends on factors
// inferrable from the given arguments, including is_class which indicates
// whether the entity translates to a Java class.
string ToJavaName(const Params& params, const string& name, bool is_class,
    const Descriptor* parent, const FileDescriptor* file);

// These return the fully-qualified class name corresponding to the given
// descriptor.
inline string ClassName(const Params& params, const Descriptor* descriptor) {
  return ToJavaName(params, descriptor->name(), true,
                    descriptor->containing_type(), descriptor->file());
}
string ClassName(const Params& params, const EnumDescriptor* descriptor);
inline string ClassName(const Params& params,
    const ServiceDescriptor* descriptor) {
  return ToJavaName(params, descriptor->name(), true, NULL, descriptor->file());
}
inline string ExtensionIdentifierName(const Params& params,
    const FieldDescriptor* descriptor) {
  return ToJavaName(params, descriptor->name(), false,
                    descriptor->extension_scope(), descriptor->file());
}
string ClassName(const Params& params, const FileDescriptor* descriptor);

// Get the unqualified name that should be used for a field's field
// number constant.
string FieldConstantName(const FieldDescriptor *field);

string FieldDefaultConstantName(const FieldDescriptor *field);

// Print the field's proto-syntax definition as a comment.
void PrintFieldComment(io::Printer* printer, const FieldDescriptor* field);

enum JavaType {
  JAVATYPE_INT,
  JAVATYPE_LONG,
  JAVATYPE_FLOAT,
  JAVATYPE_DOUBLE,
  JAVATYPE_BOOLEAN,
  JAVATYPE_STRING,
  JAVATYPE_BYTES,
  JAVATYPE_ENUM,
  JAVATYPE_MESSAGE
};

JavaType GetJavaType(FieldDescriptor::Type field_type);

inline JavaType GetJavaType(const FieldDescriptor* field) {
  return GetJavaType(field->type());
}

string PrimitiveTypeName(JavaType type);

// Get the fully-qualified class name for a boxed primitive type, e.g.
// "java.lang.Integer" for JAVATYPE_INT.  Returns NULL for enum and message
// types.
string BoxedPrimitiveTypeName(JavaType type);

string EmptyArrayName(const Params& params, const FieldDescriptor* field);

string DefaultValue(const Params& params, const FieldDescriptor* field);


// Methods for shared bitfields.

// Gets the name of the shared bitfield for the given field index.
string GetBitFieldName(int index);

// Gets the name of the shared bitfield for the given bit index.
// Effectively, GetBitFieldName(bit_index / 32)
string GetBitFieldNameForBit(int bit_index);

// Generates the java code for the expression that returns whether the bit at
// the given bit index is set.
// Example: "((bitField1_ & 0x04000000) != 0)"
string GenerateGetBit(int bit_index);

// Generates the java code for the expression that sets the bit at the given
// bit index.
// Example: "bitField1_ |= 0x04000000"
string GenerateSetBit(int bit_index);

// Generates the java code for the expression that clears the bit at the given
// bit index.
// Example: "bitField1_ = (bitField1_ & ~0x04000000)"
string GenerateClearBit(int bit_index);

// Generates the java code for the expression that returns whether the bit at
// the given bit index contains different values in the current object and
// another object accessible via the variable 'other'.
// Example: "((bitField1_ & 0x04000000) != (other.bitField1_ & 0x04000000))"
string GenerateDifferentBit(int bit_index);

// Sets the 'get_*', 'set_*', 'clear_*' and 'different_*' variables, where * is
// the given name of the bit, to the appropriate Java expressions for the given
// bit index.
void SetBitOperationVariables(const string name,
    int bitIndex, map<string, string>* variables);

inline bool IsMapEntry(const Descriptor* descriptor) {
  // TODO(liujisi): Add an option to turn on maps for proto2 syntax as well.
  return descriptor->options().map_entry() &&
      descriptor->file()->syntax() == FileDescriptor::SYNTAX_PROTO3;
}

bool HasMapField(const Descriptor* descriptor);

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_JAVANANO_HELPERS_H__
