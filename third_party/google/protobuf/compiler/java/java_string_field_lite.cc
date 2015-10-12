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
// Author: jonp@google.com (Jon Perlow)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <map>
#include <string>

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/compiler/java/java_context.h>
#include <google/protobuf/compiler/java/java_doc_comment.h>
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/compiler/java/java_name_resolver.h>
#include <google/protobuf/compiler/java/java_string_field_lite.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

using internal::WireFormat;
using internal::WireFormatLite;

namespace {

void SetPrimitiveVariables(const FieldDescriptor* descriptor,
                           int messageBitIndex,
                           int builderBitIndex,
                           const FieldGeneratorInfo* info,
                           ClassNameResolver* name_resolver,
                           map<string, string>* variables) {
  SetCommonFieldVariables(descriptor, info, variables);

  (*variables)["empty_list"] =
      "com.google.protobuf.GeneratedMessageLite.emptyProtobufList()";

  (*variables)["default"] = ImmutableDefaultValue(descriptor, name_resolver);
  (*variables)["default_init"] =
      "= " + ImmutableDefaultValue(descriptor, name_resolver);
  (*variables)["capitalized_type"] = "String";
  (*variables)["tag"] = SimpleItoa(WireFormat::MakeTag(descriptor));
  (*variables)["tag_size"] = SimpleItoa(
      WireFormat::TagSize(descriptor->number(), GetType(descriptor)));
  (*variables)["null_check"] =
      "  if (value == null) {\n"
      "    throw new NullPointerException();\n"
      "  }\n";

  // TODO(birdo): Add @deprecated javadoc when generating javadoc is supported
  // by the proto compiler
  (*variables)["deprecation"] = descriptor->options().deprecated()
      ? "@java.lang.Deprecated " : "";
  (*variables)["on_changed"] =
      HasDescriptorMethods(descriptor->containing_type()) ? "onChanged();" : "";

  if (SupportFieldPresence(descriptor->file())) {
    // For singular messages and builders, one bit is used for the hasField bit.
    (*variables)["get_has_field_bit_message"] = GenerateGetBit(messageBitIndex);

    // Note that these have a trailing ";".
    (*variables)["set_has_field_bit_message"] =
        GenerateSetBit(messageBitIndex) + ";";
    (*variables)["clear_has_field_bit_message"] =
        GenerateClearBit(messageBitIndex) + ";";

    (*variables)["is_field_present_message"] = GenerateGetBit(messageBitIndex);
  } else {
    (*variables)["set_has_field_bit_message"] = "";
    (*variables)["clear_has_field_bit_message"] = "";

    (*variables)["is_field_present_message"] =
        "!get" + (*variables)["capitalized_name"] + ".isEmpty()";
  }

  // For repeated builders, the underlying list tracks mutability state.
  (*variables)["is_mutable"] = (*variables)["name"] + "_.isModifiable()";

  (*variables)["get_has_field_bit_from_local"] =
      GenerateGetBitFromLocal(builderBitIndex);
  (*variables)["set_has_field_bit_to_local"] =
      GenerateSetBitToLocal(messageBitIndex);
}

}  // namespace

// ===================================================================

ImmutableStringFieldLiteGenerator::
ImmutableStringFieldLiteGenerator(const FieldDescriptor* descriptor,
                              int messageBitIndex,
                              int builderBitIndex,
                              Context* context)
  : descriptor_(descriptor), messageBitIndex_(messageBitIndex),
    builderBitIndex_(builderBitIndex), context_(context),
    name_resolver_(context->GetNameResolver()) {
  SetPrimitiveVariables(descriptor, messageBitIndex, builderBitIndex,
                        context->GetFieldGeneratorInfo(descriptor),
                        name_resolver_, &variables_);
}

ImmutableStringFieldLiteGenerator::~ImmutableStringFieldLiteGenerator() {}

int ImmutableStringFieldLiteGenerator::GetNumBitsForMessage() const {
  return 1;
}

int ImmutableStringFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

// A note about how strings are handled. In the SPEED and CODE_SIZE runtimes,
// strings are not stored as java.lang.String in the Message because of two
// issues:
//
//  1. It wouldn't roundtrip byte arrays that were not vaid UTF-8 encoded
//     strings, but rather fields that were raw bytes incorrectly marked
//     as strings in the proto file. This is common because in the proto1
//     syntax, string was the way to indicate bytes and C++ engineers can
//     easily make this mistake without affecting the C++ API. By converting to
//     strings immediately, some java code might corrupt these byte arrays as
//     it passes through a java server even if the field was never accessed by
//     application code.
//
//  2. There's a performance hit to converting between bytes and strings and
//     it many cases, the field is never even read by the application code. This
//     avoids unnecessary conversions in the common use cases.
//
// In the LITE_RUNTIME, we store strings as java.lang.String because we assume
// that the users of this runtime are not subject to proto1 constraints and are
// running code on devices that are user facing. That is, the developers are
// properly incentivized to only fetch the data they need to read and wish to
// reduce the number of allocations incurred when running on a user's device.

// TODO(dweis): Consider dropping all of the *Bytes() methods. They really
//     shouldn't be necessary or used on devices.
void ImmutableStringFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$boolean has$capitalized_name$();\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$java.lang.String get$capitalized_name$();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes();\n");
}

void ImmutableStringFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private java.lang.String $name$_;\n");
  PrintExtraFieldInfo(variables_, printer);

  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return $get_has_field_bit_message$;\n"
      "}\n");
  }

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.lang.String get$capitalized_name$() {\n"
    "  return $name$_;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes() {\n"
    "  return com.google.protobuf.ByteString.copyFromUtf8($name$_);\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    java.lang.String value) {\n"
    "$null_check$"
    "  $set_has_field_bit_message$\n"
    "  $name$_ = value;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $clear_has_field_bit_message$\n"
    // The default value is not a simple literal so we want to avoid executing
    // it multiple times.  Instead, get the default out of the default instance.
    "  $name$_ = getDefaultInstance().get$capitalized_name$();\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$Bytes(\n"
    "    com.google.protobuf.ByteString value) {\n"
    "$null_check$");
  if (CheckUtf8(descriptor_)) {
    printer->Print(variables_,
      "  checkByteStringIsUtf8(value);\n");
  }
  printer->Print(variables_,
    "  $set_has_field_bit_message$\n"
    "  $name$_ = value.toStringUtf8();\n"
    "}\n");
}

void ImmutableStringFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return instance.has$capitalized_name$();\n"
      "}\n");
  }

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.lang.String get$capitalized_name$() {\n"
    "  return instance.get$capitalized_name$();\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes() {\n"
    "  return instance.get$capitalized_name$Bytes();\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    java.lang.String value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder clear$capitalized_name$() {\n"
    "  copyOnWrite();\n"
    "  instance.clear$capitalized_name$();\n"
    "  return this;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$Bytes(\n"
    "    com.google.protobuf.ByteString value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$Bytes(value);\n"
    "  return this;\n"
    "}\n");
}

void ImmutableStringFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  // noop for strings
}

void ImmutableStringFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = $default$;\n");
}

void ImmutableStringFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    // Allow a slight breach of abstraction here in order to avoid forcing
    // all string fields to Strings when copying fields from a Message.
    printer->Print(variables_,
      "if (other.has$capitalized_name$()) {\n"
      "  $set_has_field_bit_message$\n"
      "  $name$_ = other.$name$_;\n"
      "  $on_changed$\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "if (!other.get$capitalized_name$().isEmpty()) {\n"
      "  $name$_ = other.$name$_;\n"
      "  $on_changed$\n"
      "}\n");
  }
}

void ImmutableStringFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  // noop for scalars
}

void ImmutableStringFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  if (CheckUtf8(descriptor_)) {
    printer->Print(variables_,
      "String s = input.readStringRequireUtf8();\n"
      "$set_has_field_bit_message$\n"
      "$name$_ = s;\n");
  } else {
    // Lite runtime should attempt to reduce allocations by attempting to
    // construct the string directly from the input stream buffer. This avoids
    // spurious intermediary ByteString allocations, cutting overall allocations
    // in half.
    printer->Print(variables_,
      "String s = input.readString();\n"
      "$set_has_field_bit_message$\n"
      "$name$_ = s;\n");
  }
}

void ImmutableStringFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  // noop for strings
}

void ImmutableStringFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  // Lite runtime should reduce allocations by serializing the string directly.
  // This avoids spurious intermediary ByteString allocations, cutting overall
  // allocations in half.
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  output.writeString($number$, get$capitalized_name$());\n"
    "}\n");
}

void ImmutableStringFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  // Lite runtime should reduce allocations by computing on the string directly.
  // This avoids spurious intermediary ByteString allocations, cutting overall
  // allocations in half.
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeStringSize($number$, get$capitalized_name$());\n"
    "}\n");
}

void ImmutableStringFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = result && get$capitalized_name$()\n"
    "    .equals(other.get$capitalized_name$());\n");
}

void ImmutableStringFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "hash = (37 * hash) + $constant_name$;\n");
  printer->Print(variables_,
    "hash = (53 * hash) + get$capitalized_name$().hashCode();\n");
}

string ImmutableStringFieldLiteGenerator::GetBoxedType() const {
  return "java.lang.String";
}

// ===================================================================

ImmutableStringOneofFieldLiteGenerator::
ImmutableStringOneofFieldLiteGenerator(const FieldDescriptor* descriptor,
                                   int messageBitIndex,
                                   int builderBitIndex,
                                   Context* context)
    : ImmutableStringFieldLiteGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
  const OneofGeneratorInfo* info =
      context->GetOneofGeneratorInfo(descriptor->containing_oneof());
  SetCommonOneofVariables(descriptor, info, &variables_);
}

ImmutableStringOneofFieldLiteGenerator::
~ImmutableStringOneofFieldLiteGenerator() {}

void ImmutableStringOneofFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  PrintExtraFieldInfo(variables_, printer);

  if (SupportFieldPresence(descriptor_->file())) {
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return $has_oneof_case_message$;\n"
    "}\n");
  }

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.lang.String get$capitalized_name$() {\n"
    "  java.lang.String ref $default_init$;\n"
    "  if ($has_oneof_case_message$) {\n"
    "    ref = (java.lang.String) $oneof_name$_;\n"
    "  }\n"
    "  return ref;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);

  printer->Print(variables_,
    "$deprecation$public com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes() {\n"
    "  java.lang.String ref $default_init$;\n"
    "  if ($has_oneof_case_message$) {\n"
    "    ref = (java.lang.String) $oneof_name$_;\n"
    "  }\n"
    "  return com.google.protobuf.ByteString.copyFromUtf8(ref);\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    java.lang.String value) {\n"
    "$null_check$"
    "  $set_oneof_case_message$;\n"
    "  $oneof_name$_ = value;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  if ($has_oneof_case_message$) {\n"
    "    $clear_oneof_case_message$;\n"
    "    $oneof_name$_ = null;\n"
    "  }\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$Bytes(\n"
    "    com.google.protobuf.ByteString value) {\n"
    "$null_check$");
  if (CheckUtf8(descriptor_)) {
    printer->Print(variables_,
      "  checkByteStringIsUtf8(value);\n");
  }
  printer->Print(variables_,
    "  $set_oneof_case_message$;\n"
    "  $oneof_name$_ = value.toStringUtf8();\n"
    "}\n");
}

void ImmutableStringOneofFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return instance.has$capitalized_name$();\n"
      "}\n");
  }

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.lang.String get$capitalized_name$() {\n"
    "  return instance.get$capitalized_name$();\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes() {\n"
    "  return instance.get$capitalized_name$Bytes();\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    java.lang.String value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder clear$capitalized_name$() {\n"
    "  copyOnWrite();\n"
    "  instance.clear$capitalized_name$();\n"
    "  return this;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$Bytes(\n"
    "    com.google.protobuf.ByteString value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$Bytes(value);\n"
    "  return this;\n"
    "}\n");
}

void ImmutableStringOneofFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  // Allow a slight breach of abstraction here in order to avoid forcing
  // all string fields to Strings when copying fields from a Message.
  printer->Print(variables_,
    "$set_oneof_case_message$;\n"
    "$oneof_name$_ = other.$oneof_name$_;\n"
    "$on_changed$\n");
}

void ImmutableStringOneofFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  if (CheckUtf8(descriptor_)) {
    printer->Print(variables_,
      "String s = input.readStringRequireUtf8();\n"
      "$set_oneof_case_message$;\n"
      "$oneof_name$_ = s;\n");
  } else {
    // Lite runtime should attempt to reduce allocations by attempting to
    // construct the string directly from the input stream buffer. This avoids
    // spurious intermediary ByteString allocations, cutting overall allocations
    // in half.
    printer->Print(variables_,
      "String s = input.readString();\n"
      "$set_oneof_case_message$;\n"
      "$oneof_name$_ = s;\n");
  }
}

void ImmutableStringOneofFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  // Lite runtime should reduce allocations by serializing the string directly.
  // This avoids spurious intermediary ByteString allocations, cutting overall
  // allocations in half.
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  output.writeString($number$, get$capitalized_name$());\n"
    "}\n");
}

void ImmutableStringOneofFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  // Lite runtime should reduce allocations by computing on the string directly.
  // This avoids spurious intermediary ByteString allocations, cutting overall
  // allocations in half.
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeStringSize($number$, get$capitalized_name$());\n"
    "}\n");
}

// ===================================================================

RepeatedImmutableStringFieldLiteGenerator::
RepeatedImmutableStringFieldLiteGenerator(const FieldDescriptor* descriptor,
                                      int messageBitIndex,
                                      int builderBitIndex,
                                      Context* context)
  : descriptor_(descriptor), messageBitIndex_(messageBitIndex),
    builderBitIndex_(builderBitIndex), context_(context),
    name_resolver_(context->GetNameResolver()) {
  SetPrimitiveVariables(descriptor, messageBitIndex, builderBitIndex,
                        context->GetFieldGeneratorInfo(descriptor),
                        name_resolver_, &variables_);
}

RepeatedImmutableStringFieldLiteGenerator::
~RepeatedImmutableStringFieldLiteGenerator() {}

int RepeatedImmutableStringFieldLiteGenerator::GetNumBitsForMessage() const {
  return 0;
}

int RepeatedImmutableStringFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$java.util.List<String>\n"
    "    get$capitalized_name$List();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$int get$capitalized_name$Count();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$java.lang.String get$capitalized_name$(int index);\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes(int index);\n");
}


void RepeatedImmutableStringFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private com.google.protobuf.Internal.ProtobufList<String> $name$_;\n");
  PrintExtraFieldInfo(variables_, printer);
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<String> get$capitalized_name$List() {\n"
    "  return $name$_;\n"   // note:  unmodifiable list
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return $name$_.size();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.lang.String get$capitalized_name$(int index) {\n"
    "  return $name$_.get(index);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes(int index) {\n"
    "  return com.google.protobuf.ByteString.copyFromUtf8(\n"
    "      $name$_.get(index));\n"
    "}\n");

  if (descriptor_->options().packed() &&
      HasGeneratedMethods(descriptor_->containing_type())) {
    printer->Print(variables_,
      "private int $name$MemoizedSerializedSize = -1;\n");
  }

  printer->Print(variables_,
    "private void ensure$capitalized_name$IsMutable() {\n"
    "  if (!$is_mutable$) {\n"
    "    $name$_ = com.google.protobuf.GeneratedMessageLite.newProtobufList(\n"
    "        $name$_);\n"
    "   }\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    int index, java.lang.String value) {\n"
    "$null_check$"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.set(index, value);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$(\n"
    "    java.lang.String value) {\n"
    "$null_check$"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(value);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void addAll$capitalized_name$(\n"
    "    java.lang.Iterable<java.lang.String> values) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  com.google.protobuf.AbstractMessageLite.addAll(\n"
    "      values, $name$_);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $name$_ = $empty_list$;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$Bytes(\n"
    "    com.google.protobuf.ByteString value) {\n"
    "$null_check$");
  if (CheckUtf8(descriptor_)) {
    printer->Print(variables_,
      "  checkByteStringIsUtf8(value);\n");
  }
  printer->Print(variables_,
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(value.toStringUtf8());\n"
    "}\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<String>\n"
    "    get$capitalized_name$List() {\n"
    "  return java.util.Collections.unmodifiableList(\n"
    "      instance.get$capitalized_name$List());\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return instance.get$capitalized_name$Count();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.lang.String get$capitalized_name$(int index) {\n"
    "  return instance.get$capitalized_name$(index);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public com.google.protobuf.ByteString\n"
    "    get$capitalized_name$Bytes(int index) {\n"
    "  return instance.get$capitalized_name$Bytes(index);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    int index, java.lang.String value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(index, value);\n"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder add$capitalized_name$(\n"
    "    java.lang.String value) {\n"
    "  copyOnWrite();\n"
    "  instance.add$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder addAll$capitalized_name$(\n"
    "    java.lang.Iterable<java.lang.String> values) {\n"
    "  copyOnWrite();\n"
    "  instance.addAll$capitalized_name$(values);\n"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder clear$capitalized_name$() {\n"
    "  copyOnWrite();\n"
    "  instance.clear$capitalized_name$();\n"
    "  return this;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder add$capitalized_name$Bytes(\n"
    "    com.google.protobuf.ByteString value) {\n"
    "  copyOnWrite();\n"
    "  instance.add$capitalized_name$Bytes(value);\n"
    "  return this;\n"
    "}\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  // noop for strings
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = $empty_list$;\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  // The code below does two optimizations:
  //   1. If the other list is empty, there's nothing to do. This ensures we
  //      don't allocate a new array if we already have an immutable one.
  //   2. If the other list is non-empty and our current list is empty, we can
  //      reuse the other list which is guaranteed to be immutable.
  printer->Print(variables_,
    "if (!other.$name$_.isEmpty()) {\n"
    "  if ($name$_.isEmpty()) {\n"
    "    $name$_ = other.$name$_;\n"
    "  } else {\n"
    "    ensure$capitalized_name$IsMutable();\n"
    "    $name$_.addAll(other.$name$_);\n"
    "  }\n"
    "  $on_changed$\n"
    "}\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_.makeImmutable();\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  if (CheckUtf8(descriptor_)) {
    printer->Print(variables_,
    "String s = input.readStringRequireUtf8();\n");
  } else {
    // Lite runtime should attempt to reduce allocations by attempting to
    // construct the string directly from the input stream buffer. This avoids
    // spurious intermediary ByteString allocations, cutting overall allocations
    // in half.
    printer->Print(variables_,
    "String s = input.readString();\n");
  }
  printer->Print(variables_,
    "if (!$is_mutable$) {\n"
    "  $name$_ = com.google.protobuf.GeneratedMessageLite.newProtobufList();\n"
    "}\n");
  if (CheckUtf8(descriptor_) || !HasDescriptorMethods(descriptor_->file())) {
    printer->Print(variables_,
      "$name$_.add(s);\n");
  } else {
    printer->Print(variables_,
      "$name$_.add(bs);\n");
  }
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateParsingCodeFromPacked(io::Printer* printer) const {
  printer->Print(variables_,
    "int length = input.readRawVarint32();\n"
    "int limit = input.pushLimit(length);\n"
    "if (!$is_mutable$ && input.getBytesUntilLimit() > 0) {\n"
    "  $name$_ = com.google.protobuf.GeneratedMessageLite.newProtobufList();\n"
    "}\n"
    "while (input.getBytesUntilLimit() > 0) {\n");
  if (CheckUtf8(descriptor_)) {
    printer->Print(variables_,
      "  String s = input.readStringRequireUtf8();\n");
  } else {
    printer->Print(variables_,
      "  String s = input.readString();\n");
  }
  printer->Print(variables_,
    "  $name$.add(s);\n");
  printer->Print(variables_,
    "}\n"
    "input.popLimit(limit);\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_mutable$) {\n"
    "  $name$_.makeImmutable();\n"
    "}\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  // Lite runtime should reduce allocations by serializing the string directly.
  // This avoids spurious intermediary ByteString allocations, cutting overall
  // allocations in half.
  if (descriptor_->options().packed()) {
    printer->Print(variables_,
      "if (get$capitalized_name$List().size() > 0) {\n"
      "  output.writeRawVarint32($tag$);\n"
      "  output.writeRawVarint32($name$MemoizedSerializedSize);\n"
      "}\n"
      "for (int i = 0; i < $name$_.size(); i++) {\n"
      "  output.writeStringNoTag($name$_.get(i));\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "for (int i = 0; i < $name$_.size(); i++) {\n"
      "  output.writeString($number$, $name$_.get(i));\n"
      "}\n");
  }
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  // Lite runtime should reduce allocations by computing on the string directly.
  // This avoids spurious intermediary ByteString allocations, cutting overall
  // allocations in half.
  printer->Print(variables_,
    "{\n"
    "  int dataSize = 0;\n");
  printer->Indent();

  printer->Print(variables_,
    "for (int i = 0; i < $name$_.size(); i++) {\n"
    "  dataSize += com.google.protobuf.CodedOutputStream\n"
    "    .computeStringSizeNoTag($name$_.get(i));\n"
    "}\n");

  printer->Print(
      "size += dataSize;\n");

  if (descriptor_->options().packed()) {
    printer->Print(variables_,
      "if (!get$capitalized_name$List().isEmpty()) {\n"
      "  size += $tag_size$;\n"
      "  size += com.google.protobuf.CodedOutputStream\n"
      "      .computeInt32SizeNoTag(dataSize);\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "size += $tag_size$ * get$capitalized_name$List().size();\n");
  }

  // cache the data size for packed fields.
  if (descriptor_->options().packed()) {
    printer->Print(variables_,
      "$name$MemoizedSerializedSize = dataSize;\n");
  }

  printer->Outdent();
  printer->Print("}\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = result && get$capitalized_name$List()\n"
    "    .equals(other.get$capitalized_name$List());\n");
}

void RepeatedImmutableStringFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (get$capitalized_name$Count() > 0) {\n"
    "  hash = (37 * hash) + $constant_name$;\n"
    "  hash = (53 * hash) + get$capitalized_name$List().hashCode();\n"
    "}\n");
}

string RepeatedImmutableStringFieldLiteGenerator::GetBoxedType() const {
  return "String";
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
