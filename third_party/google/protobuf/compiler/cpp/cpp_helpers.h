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

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#ifndef GOOGLE_PROTOBUF_COMPILER_CPP_HELPERS_H__
#define GOOGLE_PROTOBUF_COMPILER_CPP_HELPERS_H__

#include <map>
#include <string>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>

namespace google {
namespace protobuf {

namespace io {
class Printer;
}

namespace compiler {
namespace cpp {

// Commonly-used separator comments.  Thick is a line of '=', thin is a line
// of '-'.
extern const char kThickSeparator[];
extern const char kThinSeparator[];

// Returns the non-nested type name for the given type.  If "qualified" is
// true, prefix the type with the full namespace.  For example, if you had:
//   package foo.bar;
//   message Baz { message Qux {} }
// Then the qualified ClassName for Qux would be:
//   ::foo::bar::Baz_Qux
// While the non-qualified version would be:
//   Baz_Qux
string ClassName(const Descriptor* descriptor, bool qualified);
string ClassName(const EnumDescriptor* enum_descriptor, bool qualified);

// Name of the CRTP class template (for use with proto_h).
// This is a class name, like "ProtoName_InternalBase".
string DependentBaseClassTemplateName(const Descriptor* descriptor);

// Name of the base class: either the dependent base class (for use with
// proto_h) or google::protobuf::Message.
string SuperClassName(const Descriptor* descriptor);

// Returns a string that down-casts from the dependent base class to the
// derived class.
string DependentBaseDownCast();
string DependentBaseConstDownCast();

// Get the (unqualified) name that should be used for this field in C++ code.
// The name is coerced to lower-case to emulate proto1 behavior.  People
// should be using lowercase-with-underscores style for proto field names
// anyway, so normally this just returns field->name().
string FieldName(const FieldDescriptor* field);

// Get the sanitized name that should be used for the given enum in C++ code.
string EnumValueName(const EnumValueDescriptor* enum_value);

// Get the unqualified name that should be used for a field's field
// number constant.
string FieldConstantName(const FieldDescriptor *field);

// Returns the scope where the field was defined (for extensions, this is
// different from the message type to which the field applies).
inline const Descriptor* FieldScope(const FieldDescriptor* field) {
  return field->is_extension() ?
    field->extension_scope() : field->containing_type();
}

// Returns true if the given 'field_descriptor' has a message type that is
// a dependency of the file where the field is defined (i.e., the field
// type is defined in a different file than the message holding the field).
//
// This only applies to Message-typed fields. Enum-typed fields may refer
// to an enum in a dependency; however, enums are specified and
// forward-declared with an enum-base, so the definition is not required to
// manipulate the field value.
bool IsFieldDependent(const FieldDescriptor* field_descriptor);

// Returns the name that should be used for forcing dependent lookup from a
// dependent base class.
string DependentTypeName(const FieldDescriptor* field);

// Returns the fully-qualified type name field->message_type().  Usually this
// is just ClassName(field->message_type(), true);
string FieldMessageTypeName(const FieldDescriptor* field);

// Strips ".proto" or ".protodevel" from the end of a filename.
string StripProto(const string& filename);

// Get the C++ type name for a primitive type (e.g. "double", "::google::protobuf::int32", etc.).
// Note:  non-built-in type names will be qualified, meaning they will start
// with a ::.  If you are using the type as a template parameter, you will
// need to insure there is a space between the < and the ::, because the
// ridiculous C++ standard defines "<:" to be a synonym for "[".
const char* PrimitiveTypeName(FieldDescriptor::CppType type);

// Get the declared type name in CamelCase format, as is used e.g. for the
// methods of WireFormat.  For example, TYPE_INT32 becomes "Int32".
const char* DeclaredTypeMethodName(FieldDescriptor::Type type);

// Return the code that evaluates to the number when compiled.
string Int32ToString(int number);

// Return the code that evaluates to the number when compiled.
string Int64ToString(int64 number);

// Get code that evaluates to the field's default value.
string DefaultValue(const FieldDescriptor* field);

// Convert a file name into a valid identifier.
string FilenameIdentifier(const string& filename);

// Return the name of the AddDescriptors() function for a given file.
string GlobalAddDescriptorsName(const string& filename);

// Return the name of the AssignDescriptors() function for a given file.
string GlobalAssignDescriptorsName(const string& filename);

// Return the qualified C++ name for a file level symbol.
string QualifiedFileLevelSymbol(const string& package, const string& name);

// Return the name of the ShutdownFile() function for a given file.
string GlobalShutdownFileName(const string& filename);

// Escape C++ trigraphs by escaping question marks to \?
string EscapeTrigraphs(const string& to_escape);

// Escaped function name to eliminate naming conflict.
string SafeFunctionName(const Descriptor* descriptor,
                        const FieldDescriptor* field,
                        const string& prefix);

// Returns true if unknown fields are preseved after parsing.
inline bool PreserveUnknownFields(const Descriptor* message) {
  return message->file()->syntax() != FileDescriptor::SYNTAX_PROTO3;
}

// If PreserveUnknownFields() is true, determines whether unknown
// fields will be stored in an UnknownFieldSet or a string.
// If PreserveUnknownFields() is false, this method will not be
// used.
inline bool UseUnknownFieldSet(const FileDescriptor* file) {
  return file->options().optimize_for() != FileOptions::LITE_RUNTIME;
}


// Does the file have any map fields, necessitating the file to include
// map_field_inl.h and map.h.
bool HasMapFields(const FileDescriptor* file);

// Does this file have any enum type definitions?
bool HasEnumDefinitions(const FileDescriptor* file);

// Does this file have generated parsing, serialization, and other
// standard methods for which reflection-based fallback implementations exist?
inline bool HasGeneratedMethods(const FileDescriptor* file) {
  return file->options().optimize_for() != FileOptions::CODE_SIZE;
}

// Do message classes in this file have descriptor and reflection methods?
inline bool HasDescriptorMethods(const FileDescriptor* file) {
  return file->options().optimize_for() != FileOptions::LITE_RUNTIME;
}

// Should we generate generic services for this file?
inline bool HasGenericServices(const FileDescriptor* file) {
  return file->service_count() > 0 &&
         file->options().optimize_for() != FileOptions::LITE_RUNTIME &&
         file->options().cc_generic_services();
}

// Should we generate a separate, super-optimized code path for serializing to
// flat arrays?  We don't do this in Lite mode because we'd rather reduce code
// size.
inline bool HasFastArraySerialization(const FileDescriptor* file) {
  return file->options().optimize_for() == FileOptions::SPEED;
}

// Returns whether we have to generate code with static initializers.
bool StaticInitializersForced(const FileDescriptor* file);

// Prints 'with_static_init' if static initializers have to be used for the
// provided file. Otherwise emits both 'with_static_init' and
// 'without_static_init' using #ifdef.
void PrintHandlingOptionalStaticInitializers(
    const FileDescriptor* file, io::Printer* printer,
    const char* with_static_init, const char* without_static_init,
    const char* var1 = NULL, const string& val1 = "",
    const char* var2 = NULL, const string& val2 = "");

void PrintHandlingOptionalStaticInitializers(
    const map<string, string>& vars, const FileDescriptor* file,
    io::Printer* printer, const char* with_static_init,
    const char* without_static_init);


inline bool IsMapEntryMessage(const Descriptor* descriptor) {
  return descriptor->options().map_entry();
}

// Returns true if the field's CPPTYPE is string or message.
bool IsStringOrMessage(const FieldDescriptor* field);

// For a string field, returns the effective ctype.  If the actual ctype is
// not supported, returns the default of STRING.
FieldOptions::CType EffectiveStringCType(const FieldDescriptor* field);

string UnderscoresToCamelCase(const string& input, bool cap_next_letter);

inline bool HasFieldPresence(const FileDescriptor* file) {
  return file->syntax() != FileDescriptor::SYNTAX_PROTO3;
}

// Returns true if 'enum' semantics are such that unknown values are preserved
// in the enum field itself, rather than going to the UnknownFieldSet.
inline bool HasPreservingUnknownEnumSemantics(const FileDescriptor* file) {
  return file->syntax() == FileDescriptor::SYNTAX_PROTO3;
}

inline bool SupportsArenas(const FileDescriptor* file) {
  return file->options().cc_enable_arenas();
}

inline bool SupportsArenas(const Descriptor* desc) {
  return SupportsArenas(desc->file());
}

inline bool SupportsArenas(const FieldDescriptor* field) {
  return SupportsArenas(field->file());
}

bool IsAnyMessage(const FileDescriptor* descriptor);
bool IsAnyMessage(const Descriptor* descriptor);

bool IsWellKnownMessage(const FileDescriptor* descriptor);

void GenerateUtf8CheckCodeForString(
    const FieldDescriptor* field,
    bool for_parse,
    const map<string, string>& variables,
    const char* parameters,
    io::Printer* printer);

void GenerateUtf8CheckCodeForCord(
    const FieldDescriptor* field,
    bool for_parse,
    const map<string, string>& variables,
    const char* parameters,
    io::Printer* printer);

}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_CPP_HELPERS_H__
