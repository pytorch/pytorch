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

#include <map>
#include <string>

#include <google/protobuf/compiler/java/java_context.h>
#include <google/protobuf/compiler/java/java_message_field_lite.h>
#include <google/protobuf/compiler/java/java_doc_comment.h>
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/compiler/java/java_name_resolver.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

namespace {

void SetMessageVariables(const FieldDescriptor* descriptor,
                         int messageBitIndex,
                         int builderBitIndex,
                         const FieldGeneratorInfo* info,
                         ClassNameResolver* name_resolver,
                         map<string, string>* variables) {
  SetCommonFieldVariables(descriptor, info, variables);

  (*variables)["type"] =
      name_resolver->GetImmutableClassName(descriptor->message_type());
  (*variables)["mutable_type"] =
      name_resolver->GetMutableClassName(descriptor->message_type());
  (*variables)["group_or_message"] =
    (GetType(descriptor) == FieldDescriptor::TYPE_GROUP) ?
    "Group" : "Message";
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
        (*variables)["name"] + "_ != null";
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

ImmutableMessageFieldLiteGenerator::
ImmutableMessageFieldLiteGenerator(const FieldDescriptor* descriptor,
                      int messageBitIndex,
                      int builderBitIndex,
                      Context* context)
  : descriptor_(descriptor), messageBitIndex_(messageBitIndex),
    builderBitIndex_(builderBitIndex), context_(context),
    name_resolver_(context->GetNameResolver()) {
    SetMessageVariables(descriptor, messageBitIndex, builderBitIndex,
                        context->GetFieldGeneratorInfo(descriptor),
                        name_resolver_, &variables_);
}

ImmutableMessageFieldLiteGenerator::~ImmutableMessageFieldLiteGenerator() {}

int ImmutableMessageFieldLiteGenerator::GetNumBitsForMessage() const {
  return 1;
}

int ImmutableMessageFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void ImmutableMessageFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  // TODO(jonp): In the future, consider having a method specific to the
  // interface so that builders can choose dynamically to either return a
  // message or a nested builder, so that asking for the interface doesn't
  // cause a message to ever be built.
  if (SupportFieldPresence(descriptor_->file()) ||
      descriptor_->containing_oneof() == NULL) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$boolean has$capitalized_name$();\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$$type$ get$capitalized_name$();\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private $type$ $name$_;\n");
  PrintExtraFieldInfo(variables_, printer);

  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return $get_has_field_bit_message$;\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public $type$ get$capitalized_name$() {\n"
      "  return $name$_ == null ? $type$.getDefaultInstance() : $name$_;\n"
      "}\n");
  } else {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return $name$_ != null;\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public $type$ get$capitalized_name$() {\n"
      "  return $name$_ == null ? $type$.getDefaultInstance() : $name$_;\n"
      "}\n");
  }

  // Field.Builder setField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  $name$_ = value;\n"
    "  $set_has_field_bit_message$\n"
    "  }\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  $name$_ = builderForValue.build();\n"
    "  $set_has_field_bit_message$\n"
    "}\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void merge$capitalized_name$($type$ value) {\n"
    "  if ($name$_ != null &&\n"
    "      $name$_ != $type$.getDefaultInstance()) {\n"
    "    $name$_ =\n"
    "      $type$.newBuilder($name$_).mergeFrom(value).buildPartial();\n"
    "  } else {\n"
    "    $name$_ = value;\n"
    "  }\n"
    "  $set_has_field_bit_message$\n"
    "}\n");

  // Field.Builder clearField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {"
    "  $name$_ = null;\n"
    "  $clear_has_field_bit_message$\n"
    "}\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // The comments above the methods below are based on a hypothetical
  // field of type "Field" called "Field".

  // boolean hasField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return instance.has$capitalized_name$();\n"
    "}\n");

  // Field getField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  return instance.get$capitalized_name$();\n"
    "}\n");

  // Field.Builder setField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$($type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(value);\n"
    "  return this;\n"
    "  }\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(builderForValue);\n"
    "  return this;\n"
    "}\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder merge$capitalized_name$($type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.merge$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");

  // Field.Builder clearField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder clear$capitalized_name$() {"
    "  copyOnWrite();\n"
    "  instance.clear$capitalized_name$();\n"
    "  return this;\n"
    "}\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  if (SupportFieldPresence(descriptor_->file())) {
    printer->Print(variables_,
      "get$capitalized_name$FieldBuilder();\n");
  }
}


void ImmutableMessageFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {}

void ImmutableMessageFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (other.has$capitalized_name$()) {\n"
    "  merge$capitalized_name$(other.get$capitalized_name$());\n"
    "}\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  // noop for scalars
}

void ImmutableMessageFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$type$.Builder subBuilder = null;\n"
    "if ($is_field_present_message$) {\n"
    "  subBuilder = $name$_.toBuilder();\n"
    "}\n");

    if (GetType(descriptor_) == FieldDescriptor::TYPE_GROUP) {
      printer->Print(variables_,
        "$name$_ = input.readGroup($number$, $type$.parser(),\n"
        "    extensionRegistry);\n");
    } else {
      printer->Print(variables_,
        "$name$_ = input.readMessage($type$.parser(), extensionRegistry);\n");
    }

  printer->Print(variables_,
    "if (subBuilder != null) {\n"
    "  subBuilder.mergeFrom($name$_);\n"
    "  $name$_ = subBuilder.buildPartial();\n"
    "}\n"
    "$set_has_field_bit_message$\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  // noop for messages.
}

void ImmutableMessageFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  output.write$group_or_message$($number$, get$capitalized_name$());\n"
    "}\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .compute$group_or_message$Size($number$, get$capitalized_name$());\n"
    "}\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = result && get$capitalized_name$()\n"
    "    .equals(other.get$capitalized_name$());\n");
}

void ImmutableMessageFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "hash = (37 * hash) + $constant_name$;\n"
    "hash = (53 * hash) + get$capitalized_name$().hashCode();\n");
}

string ImmutableMessageFieldLiteGenerator::GetBoxedType() const {
  return name_resolver_->GetImmutableClassName(descriptor_->message_type());
}

// ===================================================================

ImmutableMessageOneofFieldLiteGenerator::
ImmutableMessageOneofFieldLiteGenerator(const FieldDescriptor* descriptor,
                                 int messageBitIndex,
                                 int builderBitIndex,
                                 Context* context)
    : ImmutableMessageFieldLiteGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
  const OneofGeneratorInfo* info =
      context->GetOneofGeneratorInfo(descriptor->containing_oneof());
  SetCommonOneofVariables(descriptor, info, &variables_);
}

ImmutableMessageOneofFieldLiteGenerator::
~ImmutableMessageOneofFieldLiteGenerator() {}

void ImmutableMessageOneofFieldLiteGenerator::
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
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  if ($has_oneof_case_message$) {\n"
    "     return ($type$) $oneof_name$_;\n"
    "  }\n"
    "  return $type$.getDefaultInstance();\n"
    "}\n");

  // Field.Builder setField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  $oneof_name$_ = value;\n"
    "  $set_oneof_case_message$;\n"
    "}\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  $oneof_name$_ = builderForValue.build();\n"
    "  $set_oneof_case_message$;\n"
    "}\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void merge$capitalized_name$($type$ value) {\n"
    "  if ($has_oneof_case_message$ &&\n"
    "      $oneof_name$_ != $type$.getDefaultInstance()) {\n"
    "    $oneof_name$_ = $type$.newBuilder(($type$) $oneof_name$_)\n"
    "        .mergeFrom(value).buildPartial();\n"
    "  } else {\n"
    "    $oneof_name$_ = value;\n"
    "  }\n"
    "  $set_oneof_case_message$;\n"
    "}\n");

  // Field.Builder clearField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  if ($has_oneof_case_message$) {\n"
    "    $clear_oneof_case_message$;\n"
    "    $oneof_name$_ = null;\n"
    "  }\n"
    "}\n");
}

void ImmutableMessageOneofFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // The comments above the methods below are based on a hypothetical
  // field of type "Field" called "Field".

  if (SupportFieldPresence(descriptor_->file())) {
    // boolean hasField()
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return instance.has$capitalized_name$();\n"
      "}\n");
  }

  // Field getField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  return instance.get$capitalized_name$();\n"
    "}\n");

  // Field.Builder setField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$($type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(builderForValue);\n"
    "  return this;\n"
    "}\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder merge$capitalized_name$($type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.merge$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");

  // Field.Builder clearField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder clear$capitalized_name$() {\n"
    "  copyOnWrite();\n"
    "  instance.clear$capitalized_name$();\n"
    "  return this;\n"
    "}\n");
}

void ImmutableMessageOneofFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "merge$capitalized_name$(other.get$capitalized_name$());\n");
}

void ImmutableMessageOneofFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$type$.Builder subBuilder = null;\n"
    "if ($has_oneof_case_message$) {\n"
    "  subBuilder = (($type$) $oneof_name$_).toBuilder();\n"
    "}\n");

    if (GetType(descriptor_) == FieldDescriptor::TYPE_GROUP) {
      printer->Print(variables_,
        "$oneof_name$_ = input.readGroup($number$, $type$.parser(),\n"
        "    extensionRegistry);\n");
    } else {
      printer->Print(variables_,
        "$oneof_name$_ =\n"
        "     input.readMessage($type$.parser(), extensionRegistry);\n");
    }

  printer->Print(variables_,
    "if (subBuilder != null) {\n"
    "  subBuilder.mergeFrom(($type$) $oneof_name$_);\n"
    "  $oneof_name$_ = subBuilder.buildPartial();\n"
    "}\n");
  printer->Print(variables_,
    "$set_oneof_case_message$;\n");
}

void ImmutableMessageOneofFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  output.write$group_or_message$($number$, ($type$) $oneof_name$_);\n"
    "}\n");
}

void ImmutableMessageOneofFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .compute$group_or_message$Size($number$, ($type$) $oneof_name$_);\n"
    "}\n");
}

// ===================================================================

RepeatedImmutableMessageFieldLiteGenerator::
RepeatedImmutableMessageFieldLiteGenerator(const FieldDescriptor* descriptor,
                                       int messageBitIndex,
                                       int builderBitIndex,
                                       Context* context)
  : descriptor_(descriptor), messageBitIndex_(messageBitIndex),
    builderBitIndex_(builderBitIndex), context_(context),
    name_resolver_(context->GetNameResolver())  {
  SetMessageVariables(descriptor, messageBitIndex, builderBitIndex,
                      context->GetFieldGeneratorInfo(descriptor),
                      name_resolver_, &variables_);
}

RepeatedImmutableMessageFieldLiteGenerator::
~RepeatedImmutableMessageFieldLiteGenerator() {}

int RepeatedImmutableMessageFieldLiteGenerator::GetNumBitsForMessage() const {
  return 0;
}

int RepeatedImmutableMessageFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  // TODO(jonp): In the future, consider having methods specific to the
  // interface so that builders can choose dynamically to either return a
  // message or a nested builder, so that asking for the interface doesn't
  // cause a message to ever be built.
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$java.util.List<$type$> \n"
    "    get$capitalized_name$List();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$$type$ get$capitalized_name$(int index);\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$int get$capitalized_name$Count();\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private com.google.protobuf.Internal.ProtobufList<$type$> $name$_;\n");
  PrintExtraFieldInfo(variables_, printer);
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<$type$> get$capitalized_name$List() {\n"
    "  return $name$_;\n"   // note:  unmodifiable list
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<? extends $type$OrBuilder> \n"
    "    get$capitalized_name$OrBuilderList() {\n"
    "  return $name$_;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return $name$_.size();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$(int index) {\n"
    "  return $name$_.get(index);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$OrBuilder get$capitalized_name$OrBuilder(\n"
    "    int index) {\n"
    "  return $name$_.get(index);\n"
    "}\n");

  printer->Print(variables_,
    "private void ensure$capitalized_name$IsMutable() {\n"
    "  if (!$is_mutable$) {\n"
    "    $name$_ = newProtobufList($name$_);\n"
    "   }\n"
    "}\n"
    "\n");

  // Builder setRepeatedField(int index, Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    int index, $type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.set(index, value);\n"
    "}\n");

  // Builder setRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.set(index, builderForValue.build());\n"
    "}\n");

  // Builder addRepeatedField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(value);\n"
    "}\n");

  // Builder addRepeatedField(int index, Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$(\n"
    "    int index, $type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(index, value);\n"
    "}\n");
  // Builder addRepeatedField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(builderForValue.build());\n"
    "}\n");

  // Builder addRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(index, builderForValue.build());\n"
    "}\n");

  // Builder addAllRepeatedField(Iterable<Field> values)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void addAll$capitalized_name$(\n"
    "    java.lang.Iterable<? extends $type$> values) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  com.google.protobuf.AbstractMessageLite.addAll(\n"
    "      values, $name$_);\n"
    "}\n");

  // Builder clearAllRepeatedField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $name$_ = emptyProtobufList();\n"
    "}\n");

  // Builder removeRepeatedField(int index)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void remove$capitalized_name$(int index) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.remove(index);\n"
    "}\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // The comments above the methods below are based on a hypothetical
  // repeated field of type "Field" called "RepeatedField".

  // List<Field> getRepeatedFieldList()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<$type$> get$capitalized_name$List() {\n"
    "  return java.util.Collections.unmodifiableList(\n"
    "      instance.get$capitalized_name$List());\n"
    "}\n");

  // int getRepeatedFieldCount()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return instance.get$capitalized_name$Count();\n"
    "}");

  // Field getRepeatedField(int index)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$(int index) {\n"
    "  return instance.get$capitalized_name$(index);\n"
    "}\n");

  // Builder setRepeatedField(int index, Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    int index, $type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(index, value);\n"
    "  return this;\n"
    "}\n");

  // Builder setRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(index, builderForValue);\n"
    "  return this;\n"
    "}\n");

  // Builder addRepeatedField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder add$capitalized_name$($type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.add$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");

  // Builder addRepeatedField(int index, Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder add$capitalized_name$(\n"
    "    int index, $type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.add$capitalized_name$(index, value);\n"
    "  return this;\n"
    "}\n");
  // Builder addRepeatedField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder add$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  copyOnWrite();\n"
    "  instance.add$capitalized_name$(builderForValue);\n"
    "  return this;\n"
    "}\n");

  // Builder addRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder add$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue) {\n"
    "  copyOnWrite();\n"
    "  instance.add$capitalized_name$(index, builderForValue);\n"
    "  return this;\n"
    "}\n");

  // Builder addAllRepeatedField(Iterable<Field> values)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder addAll$capitalized_name$(\n"
    "    java.lang.Iterable<? extends $type$> values) {\n"
    "  copyOnWrite();\n"
    "  instance.addAll$capitalized_name$(values);\n"
    "  return this;\n"
    "}\n");

  // Builder clearAllRepeatedField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder clear$capitalized_name$() {\n"
    "  copyOnWrite();\n"
    "  instance.clear$capitalized_name$();\n"
    "  return this;\n"
    "}\n");

  // Builder removeRepeatedField(int index)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder remove$capitalized_name$(int index) {\n"
    "  copyOnWrite();\n"
    "  instance.remove$capitalized_name$(index);\n"
    "  return this;\n"
    "}\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  printer->Print(variables_,
    "get$capitalized_name$FieldBuilder();\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = emptyProtobufList();\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  // The code below does two optimizations (non-nested builder case):
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

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_.makeImmutable();\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!$is_mutable$) {\n"
    "  $name$_ = newProtobufList();\n"
    "}\n");

    if (GetType(descriptor_) == FieldDescriptor::TYPE_GROUP) {
      printer->Print(variables_,
        "$name$_.add(input.readGroup($number$, $type$.parser(),\n"
        "    extensionRegistry));\n");
    } else {
      printer->Print(variables_,
        "$name$_.add(\n"
        "    input.readMessage($type$.parser(), extensionRegistry));\n");
    }
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_mutable$) {\n"
    "  $name$_.makeImmutable();\n"
    "}\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "for (int i = 0; i < $name$_.size(); i++) {\n"
    "  output.write$group_or_message$($number$, $name$_.get(i));\n"
    "}\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "for (int i = 0; i < $name$_.size(); i++) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .compute$group_or_message$Size($number$, $name$_.get(i));\n"
    "}\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = result && get$capitalized_name$List()\n"
    "    .equals(other.get$capitalized_name$List());\n");
}

void RepeatedImmutableMessageFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (get$capitalized_name$Count() > 0) {\n"
    "  hash = (37 * hash) + $constant_name$;\n"
    "  hash = (53 * hash) + get$capitalized_name$List().hashCode();\n"
    "}\n");
}

string RepeatedImmutableMessageFieldLiteGenerator::GetBoxedType() const {
  return name_resolver_->GetImmutableClassName(descriptor_->message_type());
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
