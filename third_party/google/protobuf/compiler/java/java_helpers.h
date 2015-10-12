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

#ifndef GOOGLE_PROTOBUF_COMPILER_JAVA_HELPERS_H__
#define GOOGLE_PROTOBUF_COMPILER_JAVA_HELPERS_H__

#include <string>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/descriptor.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

// Commonly-used separator comments.  Thick is a line of '=', thin is a line
// of '-'.
extern const char kThickSeparator[];
extern const char kThinSeparator[];

// Converts a name to camel-case. If cap_first_letter is true, capitalize the
// first letter.
string UnderscoresToCamelCase(const string& name, bool cap_first_letter);
// Converts the field's name to camel-case, e.g. "foo_bar_baz" becomes
// "fooBarBaz" or "FooBarBaz", respectively.
string UnderscoresToCamelCase(const FieldDescriptor* field);
string UnderscoresToCapitalizedCamelCase(const FieldDescriptor* field);

// Similar, but for method names.  (Typically, this merely has the effect
// of lower-casing the first letter of the name.)
string UnderscoresToCamelCase(const MethodDescriptor* method);

// Get an identifier that uniquely identifies this type within the file.
// This is used to declare static variables related to this type at the
// outermost file scope.
string UniqueFileScopeIdentifier(const Descriptor* descriptor);

// Strips ".proto" or ".protodevel" from the end of a filename.
string StripProto(const string& filename);

// Gets the unqualified class name for the file.  For each .proto file, there
// will be one Java class containing all the immutable messages and another
// Java class containing all the mutable messages.
// TODO(xiaofeng): remove the default value after updating client code.
string FileClassName(const FileDescriptor* file, bool immutable = true);

// Returns the file's Java package name.
string FileJavaPackage(const FileDescriptor* file, bool immutable = true);

// Returns output directory for the given package name.
string JavaPackageToDir(string package_name);

// Converts the given fully-qualified name in the proto namespace to its
// fully-qualified name in the Java namespace, given that it is in the given
// file.
// TODO(xiaofeng): this method is deprecated and should be removed in the
// future.
string ToJavaName(const string& full_name,
                  const FileDescriptor* file);

// TODO(xiaofeng): the following methods are kept for they are exposed
// publicly in //google/protobuf/compiler/java/names.h. They return
// immutable names only and should be removed after mutable API is
// integrated into google3.
string ClassName(const Descriptor* descriptor);
string ClassName(const EnumDescriptor* descriptor);
string ClassName(const ServiceDescriptor* descriptor);
string ClassName(const FileDescriptor* descriptor);

// Comma-separate list of option-specified interfaces implemented by the
// Message, to follow the "implements" declaration of the Message definition.
string ExtraMessageInterfaces(const Descriptor* descriptor);
// Comma-separate list of option-specified interfaces implemented by the
// MutableMessage, to follow the "implements" declaration of the MutableMessage
// definition.
string ExtraMutableMessageInterfaces(const Descriptor* descriptor);
// Comma-separate list of option-specified interfaces implemented by the
// Builder, to follow the "implements" declaration of the Builder definition.
string ExtraBuilderInterfaces(const Descriptor* descriptor);
// Comma-separate list of option-specified interfaces extended by the
// MessageOrBuilder, to follow the "extends" declaration of the
// MessageOrBuilder definition.
string ExtraMessageOrBuilderInterfaces(const Descriptor* descriptor);

// Get the unqualified Java class name for mutable messages. i.e. without
// package or outer classnames.
inline string ShortMutableJavaClassName(const Descriptor* descriptor) {
  return descriptor->name();
}


// Whether we should generate multiple java files for messages.
inline bool MultipleJavaFiles(
    const FileDescriptor* descriptor, bool immutable) {
  return descriptor->options().java_multiple_files();
}

// Get the unqualified name that should be used for a field's field
// number constant.
string FieldConstantName(const FieldDescriptor *field);

// Returns the type of the FieldDescriptor.
// This does nothing interesting for the open source release, but is used for
// hacks that improve compatibility with version 1 protocol buffers at Google.
FieldDescriptor::Type GetType(const FieldDescriptor* field);

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

JavaType GetJavaType(const FieldDescriptor* field);

const char* PrimitiveTypeName(JavaType type);

// Get the fully-qualified class name for a boxed primitive type, e.g.
// "java.lang.Integer" for JAVATYPE_INT.  Returns NULL for enum and message
// types.
const char* BoxedPrimitiveTypeName(JavaType type);

// Get the name of the java enum constant representing this type. E.g.,
// "INT32" for FieldDescriptor::TYPE_INT32. The enum constant's full
// name is "com.google.protobuf.WireFormat.FieldType.INT32".
const char* FieldTypeName(const FieldDescriptor::Type field_type);

class ClassNameResolver;
string DefaultValue(const FieldDescriptor* field, bool immutable,
                    ClassNameResolver* name_resolver);
inline string ImmutableDefaultValue(const FieldDescriptor* field,
                                    ClassNameResolver* name_resolver) {
  return DefaultValue(field, true, name_resolver);
}
bool IsDefaultValueJavaDefault(const FieldDescriptor* field);

// Does this message class have generated parsing, serialization, and other
// standard methods for which reflection-based fallback implementations exist?
inline bool HasGeneratedMethods(const Descriptor* descriptor) {
  return descriptor->file()->options().optimize_for() !=
           FileOptions::CODE_SIZE;
}

// Does this message have specialized equals() and hashCode() methods?
inline bool HasEqualsAndHashCode(const Descriptor* descriptor) {
  return descriptor->file()->options().java_generate_equals_and_hash();
}

// Does this message class have descriptor and reflection methods?
inline bool HasDescriptorMethods(const Descriptor* descriptor) {
  return descriptor->file()->options().optimize_for() !=
           FileOptions::LITE_RUNTIME;
}
inline bool HasDescriptorMethods(const EnumDescriptor* descriptor) {
  return descriptor->file()->options().optimize_for() !=
           FileOptions::LITE_RUNTIME;
}
inline bool HasDescriptorMethods(const FileDescriptor* descriptor) {
  return descriptor->options().optimize_for() !=
           FileOptions::LITE_RUNTIME;
}

// Should we generate generic services for this file?
inline bool HasGenericServices(const FileDescriptor *file) {
  return file->service_count() > 0 &&
         file->options().optimize_for() != FileOptions::LITE_RUNTIME &&
         file->options().java_generic_services();
}

inline bool IsLazy(const FieldDescriptor* descriptor) {
  // Currently, the proto-lite version suports lazy field.
  // TODO(niwasaki): Support lazy fields also for other proto runtimes.
  if (descriptor->file()->options().optimize_for() !=
      FileOptions::LITE_RUNTIME) {
    return false;
  }
  return descriptor->options().lazy();
}

// Methods for shared bitfields.

// Gets the name of the shared bitfield for the given index.
string GetBitFieldName(int index);

// Gets the name of the shared bitfield for the given bit index.
// Effectively, GetBitFieldName(bitIndex / 32)
string GetBitFieldNameForBit(int bitIndex);

// Generates the java code for the expression that returns the boolean value
// of the bit of the shared bitfields for the given bit index.
// Example: "((bitField1_ & 0x04) == 0x04)"
string GenerateGetBit(int bitIndex);

// Generates the java code for the expression that sets the bit of the shared
// bitfields for the given bit index.
// Example: "bitField1_ = (bitField1_ | 0x04)"
string GenerateSetBit(int bitIndex);

// Generates the java code for the expression that clears the bit of the shared
// bitfields for the given bit index.
// Example: "bitField1_ = (bitField1_ & ~0x04)"
string GenerateClearBit(int bitIndex);

// Does the same as GenerateGetBit but operates on the bit field on a local
// variable. This is used by the builder to copy the value in the builder to
// the message.
// Example: "((from_bitField1_ & 0x04) == 0x04)"
string GenerateGetBitFromLocal(int bitIndex);

// Does the same as GenerateSetBit but operates on the bit field on a local
// variable. This is used by the builder to copy the value in the builder to
// the message.
// Example: "to_bitField1_ = (to_bitField1_ | 0x04)"
string GenerateSetBitToLocal(int bitIndex);

// Does the same as GenerateGetBit but operates on the bit field on a local
// variable. This is used by the parsing constructor to record if a repeated
// field is mutable.
// Example: "((mutable_bitField1_ & 0x04) == 0x04)"
string GenerateGetBitMutableLocal(int bitIndex);

// Does the same as GenerateSetBit but operates on the bit field on a local
// variable. This is used by the parsing constructor to record if a repeated
// field is mutable.
// Example: "mutable_bitField1_ = (mutable_bitField1_ | 0x04)"
string GenerateSetBitMutableLocal(int bitIndex);

// Returns whether the JavaType is a reference type.
bool IsReferenceType(JavaType type);

// Returns the capitalized name for calling relative functions in
// CodedInputStream
const char* GetCapitalizedType(const FieldDescriptor* field, bool immutable);

// For encodings with fixed sizes, returns that size in bytes.  Otherwise
// returns -1.
int FixedSize(FieldDescriptor::Type type);

// Comparators used to sort fields in MessageGenerator
struct FieldOrderingByNumber {
  inline bool operator()(const FieldDescriptor* a,
                         const FieldDescriptor* b) const {
    return a->number() < b->number();
  }
};

struct ExtensionRangeOrdering {
  bool operator()(const Descriptor::ExtensionRange* a,
                  const Descriptor::ExtensionRange* b) const {
    return a->start < b->start;
  }
};

// Sort the fields of the given Descriptor by number into a new[]'d array
// and return it. The caller should delete the returned array.
const FieldDescriptor** SortFieldsByNumber(const Descriptor* descriptor);

// Does this message class have any packed fields?
inline bool HasPackedFields(const Descriptor* descriptor) {
  for (int i = 0; i < descriptor->field_count(); i++) {
    if (descriptor->field(i)->is_packed()) {
      return true;
    }
  }
  return false;
}

// Check a message type and its sub-message types recursively to see if any of
// them has a required field. Return true if a required field is found.
bool HasRequiredFields(const Descriptor* descriptor);

// Whether a .proto file supports field presence test for non-message types.
inline bool SupportFieldPresence(const FileDescriptor* descriptor) {
  return descriptor->syntax() != FileDescriptor::SYNTAX_PROTO3;
}

// Whether generate classes expose public PARSER instances.
inline bool ExposePublicParser(const FileDescriptor* descriptor) {
  // TODO(liujisi): Mark the PARSER private in 3.1.x releases.
  return descriptor->syntax() == FileDescriptor::SYNTAX_PROTO2;
}

// Whether unknown enum values are kept (i.e., not stored in UnknownFieldSet
// but in the message and can be queried using additional getters that return
// ints.
inline bool SupportUnknownEnumValue(const FileDescriptor* descriptor) {
  return descriptor->syntax() == FileDescriptor::SYNTAX_PROTO3;
}

// Check whether a mesasge has repeated fields.
bool HasRepeatedFields(const Descriptor* descriptor);

inline bool IsMapEntry(const Descriptor* descriptor) {
  return descriptor->options().map_entry();
}

inline bool IsMapField(const FieldDescriptor* descriptor) {
  return descriptor->is_map();
}

inline bool PreserveUnknownFields(const Descriptor* descriptor) {
  return descriptor->file()->syntax() != FileDescriptor::SYNTAX_PROTO3;
}

inline bool IsAnyMessage(const Descriptor* descriptor) {
  return descriptor->full_name() == "google.protobuf.Any";
}

inline bool CheckUtf8(const FieldDescriptor* descriptor) {
  return descriptor->file()->syntax() == FileDescriptor::SYNTAX_PROTO3 ||
      descriptor->file()->options().java_string_check_utf8();
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_COMPILER_JAVA_HELPERS_H__
