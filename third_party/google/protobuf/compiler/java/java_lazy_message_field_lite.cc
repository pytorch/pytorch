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

// Author: niwasaki@google.com (Naoki Iwasaki)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <google/protobuf/compiler/java/java_context.h>
#include <google/protobuf/compiler/java/java_lazy_message_field_lite.h>
#include <google/protobuf/compiler/java/java_doc_comment.h>
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/io/printer.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

ImmutableLazyMessageFieldLiteGenerator::
ImmutableLazyMessageFieldLiteGenerator(
    const FieldDescriptor* descriptor,
    int messageBitIndex,
    int builderBitIndex,
    Context* context)
    : ImmutableMessageFieldLiteGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
}

ImmutableLazyMessageFieldLiteGenerator::
~ImmutableLazyMessageFieldLiteGenerator() {}

void ImmutableLazyMessageFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private com.google.protobuf.LazyFieldLite $name$_ =\n"
    "    new com.google.protobuf.LazyFieldLite();\n");

  PrintExtraFieldInfo(variables_, printer);
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return $get_has_field_bit_message$;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  return ($type$) $name$_.getValue($type$.getDefaultInstance());\n"
   "}\n");

  // Field.Builder setField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  $name$_.setValue(value);\n"
    "  $set_has_field_bit_message$;\n"
    "}\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  $name$_.setValue(builderForValue.build());\n"
    "  $set_has_field_bit_message$;\n"
    "}\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void merge$capitalized_name$($type$ value) {\n"
    "  if ($get_has_field_bit_message$ &&\n"
    "      !$name$_.containsDefaultInstance()) {\n"
    "    $name$_.setValue(\n"
    "      $type$.newBuilder(\n"
    "          get$capitalized_name$()).mergeFrom(value).buildPartial());\n"
    "  } else {\n"
    "    $name$_.setValue(value);\n"
    "  }\n"
    "  $set_has_field_bit_message$;\n"
    "}\n");

  // Field.Builder clearField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $name$_.clear();\n"
    "  $clear_has_field_bit_message$;\n"
    "}\n");
}

void ImmutableLazyMessageFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // The comments above the methods below are based on a hypothetical
  // field of type "Field" called "Field".

  // boolean hasField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return instance.has$capitalized_name$();\n"
    "}\n");

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


void ImmutableLazyMessageFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.clear();\n");
}

void ImmutableLazyMessageFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (other.has$capitalized_name$()) {\n"
    "  $name$_.merge(other.$name$_);\n"
    "  $set_has_field_bit_message$;\n"
    "}\n");
}

void ImmutableLazyMessageFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_.setByteString(input.readBytes(), extensionRegistry);\n");
  printer->Print(variables_,
    "$set_has_field_bit_message$;\n");
}

void ImmutableLazyMessageFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  // Do not de-serialize lazy fields.
  printer->Print(variables_,
    "if ($get_has_field_bit_message$) {\n"
    "  output.writeBytes($number$, $name$_.toByteString());\n"
    "}\n");
}

void ImmutableLazyMessageFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($get_has_field_bit_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeLazyFieldSize($number$, $name$_);\n"
    "}\n");
}

// ===================================================================

ImmutableLazyMessageOneofFieldLiteGenerator::
ImmutableLazyMessageOneofFieldLiteGenerator(const FieldDescriptor* descriptor,
                                            int messageBitIndex,
                                            int builderBitIndex,
                                            Context* context)
    : ImmutableLazyMessageFieldLiteGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
  const OneofGeneratorInfo* info =
      context->GetOneofGeneratorInfo(descriptor->containing_oneof());
  SetCommonOneofVariables(descriptor, info, &variables_);
  variables_["lazy_type"] = "com.google.protobuf.LazyFieldLite";
}

ImmutableLazyMessageOneofFieldLiteGenerator::
~ImmutableLazyMessageOneofFieldLiteGenerator() {}

void ImmutableLazyMessageOneofFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  PrintExtraFieldInfo(variables_, printer);
  WriteFieldDocComment(printer, descriptor_);

  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return $has_oneof_case_message$;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);

  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  if ($has_oneof_case_message$) {\n"
    "    return ($type$) (($lazy_type$) $oneof_name$_).getValue(\n"
    "        $type$.getDefaultInstance());\n"
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
    "  if (!($has_oneof_case_message$)) {\n"
    "    $oneof_name$_ = new $lazy_type$();\n"
    "    $set_oneof_case_message$;\n"
    "  }\n"
    "  (($lazy_type$) $oneof_name$_).setValue(value);\n"
    "}\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  if (!($has_oneof_case_message$)) {\n"
    "    $oneof_name$_ = new $lazy_type$();\n"
    "    $set_oneof_case_message$;\n"
    "  }\n"
    "  (($lazy_type$) $oneof_name$_).setValue(builderForValue.build());\n"
    "}\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void merge$capitalized_name$($type$ value) {\n"
    "  if ($has_oneof_case_message$ &&\n"
    "      !(($lazy_type$) $oneof_name$_).containsDefaultInstance()) {\n"
    "    (($lazy_type$) $oneof_name$_).setValue(\n"
    "       $type$.newBuilder(\n"
    "          get$capitalized_name$()).mergeFrom(value).buildPartial());\n"
    "  } else {\n"
    "    if (!($has_oneof_case_message$)) {\n"
    "      $oneof_name$_ = new $lazy_type$();\n"
    "      $set_oneof_case_message$;\n"
    "    }\n"
    "    (($lazy_type$) $oneof_name$_).setValue(value);\n"
    "  }\n"
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

void ImmutableLazyMessageOneofFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // boolean hasField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return instance.has$capitalized_name$();\n"
    "}\n");

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

void ImmutableLazyMessageOneofFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!($has_oneof_case_message$)) {\n"
    "  $oneof_name$_ = new $lazy_type$();\n"
    "}\n"
    "(($lazy_type$) $oneof_name$_).merge(\n"
    "    ($lazy_type$) other.$oneof_name$_);\n"
    "$set_oneof_case_message$;\n");
}

void ImmutableLazyMessageOneofFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!($has_oneof_case_message$)) {\n"
    "  $oneof_name$_ = new $lazy_type$();\n"
    "}\n"
    "(($lazy_type$) $oneof_name$_).setByteString(\n"
    "    input.readBytes(), extensionRegistry);\n"
    "$set_oneof_case_message$;\n");
}

void ImmutableLazyMessageOneofFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  // Do not de-serialize lazy fields.
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  output.writeBytes(\n"
    "      $number$, (($lazy_type$) $oneof_name$_).toByteString());\n"
    "}\n");
}

void ImmutableLazyMessageOneofFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeLazyFieldSize($number$, ($lazy_type$) $oneof_name$_);\n"
    "}\n");
}

// ===================================================================

RepeatedImmutableLazyMessageFieldLiteGenerator::
RepeatedImmutableLazyMessageFieldLiteGenerator(
    const FieldDescriptor* descriptor,
    int messageBitIndex,
    int builderBitIndex,
    Context* context)
    : RepeatedImmutableMessageFieldLiteGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
}


RepeatedImmutableLazyMessageFieldLiteGenerator::
~RepeatedImmutableLazyMessageFieldLiteGenerator() {}

void RepeatedImmutableLazyMessageFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private com.google.protobuf.Internal.ProtobufList<\n"
    "    com.google.protobuf.LazyFieldLite> $name$_;\n");
  PrintExtraFieldInfo(variables_, printer);
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<$type$>\n"
    "    get$capitalized_name$List() {\n"
    "  java.util.List<$type$> list =\n"
    "      new java.util.ArrayList<$type$>($name$_.size());\n"
    "  for (com.google.protobuf.LazyFieldLite lf : $name$_) {\n"
    "    list.add(($type$) lf.getValue($type$.getDefaultInstance()));\n"
    "  }\n"
    // TODO(dweis): Make this list immutable?
    "  return list;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<? extends $type$OrBuilder>\n"
    "    get$capitalized_name$OrBuilderList() {\n"
    "  return get$capitalized_name$List();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return $name$_.size();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$(int index) {\n"
    "  return ($type$)\n"
    "      $name$_.get(index).getValue($type$.getDefaultInstance());\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$OrBuilder get$capitalized_name$OrBuilder(\n"
    "    int index) {\n"
    "  return ($type$OrBuilder)\n"
    "      $name$_.get(index).getValue($type$.getDefaultInstance());\n"
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
    "  $name$_.set(\n"
    "      index, com.google.protobuf.LazyFieldLite.fromValue(value));\n"
    "}\n");

  // Builder setRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.set(index, com.google.protobuf.LazyFieldLite.fromValue(\n"
    "      builderForValue.build()));\n"
    "}\n");

  // Builder addRepeatedField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(com.google.protobuf.LazyFieldLite.fromValue(value));\n"
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
    "  $name$_.add(\n"
    "      index, com.google.protobuf.LazyFieldLite.fromValue(value));\n"
    "}\n");

  // Builder addRepeatedField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$(\n"
    "    $type$.Builder builderForValue) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(com.google.protobuf.LazyFieldLite.fromValue(\n"
    "      builderForValue.build()));\n"
    "}\n");

  // Builder addRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(index, com.google.protobuf.LazyFieldLite.fromValue(\n"
    "      builderForValue.build()));\n"
    "}\n");

  // Builder addAllRepeatedField(Iterable<Field> values)
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void addAll$capitalized_name$(\n"
    "    java.lang.Iterable<? extends $type$> values) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  for (com.google.protobuf.MessageLite v : values) {\n"
    "    $name$_.add(com.google.protobuf.LazyFieldLite.fromValue(v));\n"
    "  }\n"
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

void RepeatedImmutableLazyMessageFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
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
    "}\n");

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

void RepeatedImmutableLazyMessageFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!$is_mutable$) {\n"
    "  $name$_ = newProtobufList();\n"
    "}\n"
    "$name$_.add(new com.google.protobuf.LazyFieldLite(\n"
    "    extensionRegistry, input.readBytes()));\n");
}

void RepeatedImmutableLazyMessageFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "for (int i = 0; i < $name$_.size(); i++) {\n"
    "  output.writeBytes($number$, $name$_.get(i).toByteString());\n"
    "}\n");
}

void RepeatedImmutableLazyMessageFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "for (int i = 0; i < $name$_.size(); i++) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeLazyFieldSize($number$, $name$_.get(i));\n"
    "}\n");
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
