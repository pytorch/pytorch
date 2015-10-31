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
#include <google/protobuf/compiler/java/java_lazy_message_field.h>
#include <google/protobuf/compiler/java/java_doc_comment.h>
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/io/printer.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

ImmutableLazyMessageFieldGenerator::
ImmutableLazyMessageFieldGenerator(
    const FieldDescriptor* descriptor,
    int messageBitIndex,
    int builderBitIndex,
    Context* context)
    : ImmutableMessageFieldGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
}

ImmutableLazyMessageFieldGenerator::~ImmutableLazyMessageFieldGenerator() {}

void ImmutableLazyMessageFieldGenerator::
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
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$OrBuilder get$capitalized_name$OrBuilder() {\n"
    "  return $name$_;\n"
    "}\n");
}

void ImmutableLazyMessageFieldGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // When using nested-builders, the code initially works just like the
  // non-nested builder case. It only creates a nested builder lazily on
  // demand and then forever delegates to it after creation.

  printer->Print(variables_,
    "private com.google.protobuf.LazyFieldLite $name$_ =\n"
    "    new com.google.protobuf.LazyFieldLite();\n");

  printer->Print(variables_,
    // If this builder is non-null, it is used and the other fields are
    // ignored.
    "private com.google.protobuf.SingleFieldBuilder<\n"
    "    $type$, $type$.Builder, $type$OrBuilder> $name$Builder_;"
    "\n");

  // The comments above the methods below are based on a hypothetical
  // field of type "Field" called "Field".

  // boolean hasField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return $get_has_field_bit_builder$;\n"
    "}\n");

    printer->Print(variables_,
      "$deprecation$public $type$ get$capitalized_name$() {\n"
      "  return ($type$) $name$_.getValue($type$.getDefaultInstance());\n"
      "}\n");

  // Field.Builder setField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder set$capitalized_name$($type$ value)",

    "if (value == null) {\n"
    "  throw new NullPointerException();\n"
    "}\n"
    "$name$_.setValue(value);\n"
    "$on_changed$\n",

     NULL,  // Lazy fields are supported only for lite-runtime.

    "$set_has_field_bit_builder$;\n"
    "return this;\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    $type$.Builder builderForValue)",

    "$name$_.setValue(builderForValue.build());\n"
    "$on_changed$\n",

    NULL,

    "$set_has_field_bit_builder$;\n"
    "return this;\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder merge$capitalized_name$($type$ value)",

    "if ($get_has_field_bit_builder$ &&\n"
    "    !$name$_.containsDefaultInstance()) {\n"
    "  $name$_.setValue(\n"
    "    $type$.newBuilder(\n"
    "        get$capitalized_name$()).mergeFrom(value).buildPartial());\n"
    "} else {\n"
    "  $name$_.setValue(value);\n"
    "}\n"
    "$on_changed$\n",

    NULL,

    "$set_has_field_bit_builder$;\n"
    "return this;\n");

  // Field.Builder clearField()
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder clear$capitalized_name$()",

    "$name$_.clear();\n"
    "$on_changed$\n",

    NULL,

    "$clear_has_field_bit_builder$;\n"
    "return this;\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$.Builder get$capitalized_name$Builder() {\n"
    "  $set_has_field_bit_builder$;\n"
    "  $on_changed$\n"
    "  return get$capitalized_name$FieldBuilder().getBuilder();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$OrBuilder get$capitalized_name$OrBuilder() {\n"
    "  if ($name$Builder_ != null) {\n"
    "    return $name$Builder_.getMessageOrBuilder();\n"
    "  } else {\n"
    "    return $name$_;\n"
    "  }\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private com.google.protobuf.SingleFieldBuilder<\n"
    "    $type$, $type$.Builder, $type$OrBuilder> \n"
    "    get$capitalized_name$FieldBuilder() {\n"
    "  if ($name$Builder_ == null) {\n"
    "    $name$Builder_ = new com.google.protobuf.SingleFieldBuilder<\n"
    "        $type$, $type$.Builder, $type$OrBuilder>(\n"
    "            $name$_,\n"
    "            getParentForChildren(),\n"
    "            isClean());\n"
    "    $name$_ = null;\n"
    "  }\n"
    "  return $name$Builder_;\n"
    "}\n");
}


void ImmutableLazyMessageFieldGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.clear();\n");
}

void ImmutableLazyMessageFieldGenerator::
GenerateBuilderClearCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.clear();\n");
  printer->Print(variables_, "$clear_has_field_bit_builder$;\n");
}

void ImmutableLazyMessageFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (other.has$capitalized_name$()) {\n"
    "  $name$_.merge(other.$name$_);\n"
    "  $set_has_field_bit_builder$;\n"
    "}\n");
}

void ImmutableLazyMessageFieldGenerator::
GenerateBuildingCode(io::Printer* printer) const {
  printer->Print(variables_,
      "if ($get_has_field_bit_from_local$) {\n"
      "  $set_has_field_bit_to_local$;\n"
      "}\n");

  printer->Print(variables_,
      "result.$name$_.set(\n"
      "    $name$_);\n");
}

void ImmutableLazyMessageFieldGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_.setByteString(input.readBytes(), extensionRegistry);\n");
  printer->Print(variables_,
    "$set_has_field_bit_message$;\n");
}

void ImmutableLazyMessageFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  // Do not de-serialize lazy fields.
  printer->Print(variables_,
    "if ($get_has_field_bit_message$) {\n"
    "  output.writeBytes($number$, $name$_.toByteString());\n"
    "}\n");
}

void ImmutableLazyMessageFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($get_has_field_bit_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeLazyFieldSize($number$, $name$_);\n"
    "}\n");
}

// ===================================================================

ImmutableLazyMessageOneofFieldGenerator::
ImmutableLazyMessageOneofFieldGenerator(const FieldDescriptor* descriptor,
                                        int messageBitIndex,
                                        int builderBitIndex,
                                        Context* context)
    : ImmutableLazyMessageFieldGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
  const OneofGeneratorInfo* info =
      context->GetOneofGeneratorInfo(descriptor->containing_oneof());
  SetCommonOneofVariables(descriptor, info, &variables_);
  variables_["lazy_type"] = "com.google.protobuf.LazyFieldLite";
}

ImmutableLazyMessageOneofFieldGenerator::
~ImmutableLazyMessageOneofFieldGenerator() {}

void ImmutableLazyMessageOneofFieldGenerator::
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
}

void ImmutableLazyMessageOneofFieldGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // boolean hasField()
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public boolean has$capitalized_name$() {\n"
    "  return $has_oneof_case_message$;\n"
    "}\n");

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
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder set$capitalized_name$($type$ value)",

    "if (value == null) {\n"
    "  throw new NullPointerException();\n"
    "}\n"
    "if (!($has_oneof_case_message$)) {\n"
    "  $oneof_name$_ = new $lazy_type$();\n"
    "  $set_oneof_case_message$;\n"
    "}\n"
    "(($lazy_type$) $oneof_name$_).setValue(value);\n"
    "$on_changed$\n",

     NULL,  // Lazy fields are supported only for lite-runtime.

    "return this;\n");

  // Field.Builder setField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    $type$.Builder builderForValue)",

    "if (!($has_oneof_case_message$)) {\n"
    "  $oneof_name$_ = new $lazy_type$();\n"
    "  $set_oneof_case_message$;\n"
    "}\n"
    "(($lazy_type$) $oneof_name$_).setValue(builderForValue.build());\n"
    "$on_changed$\n",

    NULL,

    "return this;\n");

  // Field.Builder mergeField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder merge$capitalized_name$($type$ value)",

    "if ($has_oneof_case_message$ &&\n"
    "    !(($lazy_type$) $oneof_name$_).containsDefaultInstance()) {\n"
    "  (($lazy_type$) $oneof_name$_).setValue(\n"
    "    $type$.newBuilder(\n"
    "        get$capitalized_name$()).mergeFrom(value).buildPartial());\n"
    "} else {\n"
    "  if (!($has_oneof_case_message$)) {\n"
    "    $oneof_name$_ = new $lazy_type$();\n"
    "    $set_oneof_case_message$;\n"
    "  }\n"
    "  (($lazy_type$) $oneof_name$_).setValue(value);\n"
    "}\n"
    "$on_changed$\n",

    NULL,

    "return this;\n");

  // Field.Builder clearField()
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder clear$capitalized_name$()",

    "if ($has_oneof_case_message$) {\n"
    "  $clear_oneof_case_message$;\n"
    "  $oneof_name$_ = null;\n"
    "  $on_changed$\n"
    "}\n",

    NULL,

    "return this;\n");
}

void ImmutableLazyMessageOneofFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!($has_oneof_case_message$)) {\n"
    "  $oneof_name$_ = new $lazy_type$();\n"
    "}\n"
    "(($lazy_type$) $oneof_name$_).merge(\n"
    "    ($lazy_type$) other.$oneof_name$_);\n"
    "$set_oneof_case_message$;\n");
}

void ImmutableLazyMessageOneofFieldGenerator::
GenerateBuildingCode(io::Printer* printer) const {
  printer->Print(variables_,
                 "if ($has_oneof_case_message$) {\n");
  printer->Indent();

  printer->Print(variables_,
      "result.$oneof_name$_ = new $lazy_type$();\n"
      "(($lazy_type$) result.$oneof_name$_).set(\n"
      "    (($lazy_type$) $oneof_name$_));\n");
  printer->Outdent();
  printer->Print("}\n");
}

void ImmutableLazyMessageOneofFieldGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!($has_oneof_case_message$)) {\n"
    "  $oneof_name$_ = new $lazy_type$();\n"
    "}\n"
    "(($lazy_type$) $oneof_name$_).setByteString(\n"
    "    input.readBytes(), extensionRegistry);\n"
    "$set_oneof_case_message$;\n");
}

void ImmutableLazyMessageOneofFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  // Do not de-serialize lazy fields.
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  output.writeBytes(\n"
    "      $number$, (($lazy_type$) $oneof_name$_).toByteString());\n"
    "}\n");
}

void ImmutableLazyMessageOneofFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeLazyFieldSize($number$, ($lazy_type$) $oneof_name$_);\n"
    "}\n");
}

// ===================================================================

RepeatedImmutableLazyMessageFieldGenerator::
RepeatedImmutableLazyMessageFieldGenerator(
    const FieldDescriptor* descriptor,
    int messageBitIndex,
    int builderBitIndex,
    Context* context)
    : RepeatedImmutableMessageFieldGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
}


RepeatedImmutableLazyMessageFieldGenerator::
~RepeatedImmutableLazyMessageFieldGenerator() {}

void RepeatedImmutableLazyMessageFieldGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private java.util.List<com.google.protobuf.LazyFieldLite> $name$_;\n");
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
}

void RepeatedImmutableLazyMessageFieldGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  // When using nested-builders, the code initially works just like the
  // non-nested builder case. It only creates a nested builder lazily on
  // demand and then forever delegates to it after creation.

  printer->Print(variables_,
    "private java.util.List<com.google.protobuf.LazyFieldLite> $name$_ =\n"
    "  java.util.Collections.emptyList();\n"

    "private void ensure$capitalized_name$IsMutable() {\n"
    "  if (!$get_mutable_bit_builder$) {\n"
    "    $name$_ =\n"
    "        new java.util.ArrayList<com.google.protobuf.LazyFieldLite>(\n"
    "            $name$_);\n"
    "    $set_mutable_bit_builder$;\n"
    "   }\n"
    "}\n"
    "\n");

  printer->Print(variables_,
    // If this builder is non-null, it is used and the other fields are
    // ignored.
    "private com.google.protobuf.RepeatedFieldBuilder<\n"
    "    $type$, $type$.Builder, $type$OrBuilder> $name$Builder_;\n"
    "\n");

  // The comments above the methods below are based on a hypothetical
  // repeated field of type "Field" called "RepeatedField".

  // List<Field> getRepeatedFieldList()
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public java.util.List<$type$> get$capitalized_name$List()",

    "java.util.List<$type$> list =\n"
    "    new java.util.ArrayList<$type$>($name$_.size());\n"
    "for (com.google.protobuf.LazyFieldLite lf : $name$_) {\n"
    "  list.add(($type$) lf.getValue($type$.getDefaultInstance()));\n"
    "}\n"
    "return java.util.Collections.unmodifiableList(list);\n",

    "return $name$Builder_.getMessageList();\n",

    NULL);

  // int getRepeatedFieldCount()
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public int get$capitalized_name$Count()",

    "return $name$_.size();\n",
    "return $name$Builder_.getCount();\n",

    NULL);

  // Field getRepeatedField(int index)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public $type$ get$capitalized_name$(int index)",

    "return ($type$) $name$_.get(index).getValue(\n"
    "    $type$.getDefaultInstance());\n",

    "return $name$Builder_.getMessage(index);\n",

    NULL);

  // Builder setRepeatedField(int index, Field value)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    int index, $type$ value)",
    "if (value == null) {\n"
    "  throw new NullPointerException();\n"
    "}\n"
    "ensure$capitalized_name$IsMutable();\n"
    "$name$_.set(index, com.google.protobuf.LazyFieldLite.fromValue(value));\n"
    "$on_changed$\n",
    "$name$Builder_.setMessage(index, value);\n",
    "return this;\n");

  // Builder setRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue)",

    "ensure$capitalized_name$IsMutable();\n"
    "$name$_.set(index, com.google.protobuf.LazyFieldLite.fromValue(\n"
    "    builderForValue.build()));\n"
    "$on_changed$\n",

    "$name$Builder_.setMessage(index, builderForValue.build());\n",

    "return this;\n");

  // Builder addRepeatedField(Field value)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder add$capitalized_name$($type$ value)",

    "if (value == null) {\n"
    "  throw new NullPointerException();\n"
    "}\n"
    "ensure$capitalized_name$IsMutable();\n"
    "$name$_.add(com.google.protobuf.LazyFieldLite.fromValue(value));\n"

    "$on_changed$\n",

    "$name$Builder_.addMessage(value);\n",

    "return this;\n");

  // Builder addRepeatedField(int index, Field value)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder add$capitalized_name$(\n"
    "    int index, $type$ value)",

    "if (value == null) {\n"
    "  throw new NullPointerException();\n"
    "}\n"
    "ensure$capitalized_name$IsMutable();\n"
    "$name$_.add(index, com.google.protobuf.LazyFieldLite.fromValue(value));\n"
    "$on_changed$\n",

    "$name$Builder_.addMessage(index, value);\n",

    "return this;\n");

  // Builder addRepeatedField(Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder add$capitalized_name$(\n"
    "    $type$.Builder builderForValue)",

    "ensure$capitalized_name$IsMutable();\n"
    "$name$_.add(com.google.protobuf.LazyFieldLite.fromValue(\n"
    "    builderForValue.build()));\n"
    "$on_changed$\n",

    "$name$Builder_.addMessage(builderForValue.build());\n",

    "return this;\n");

  // Builder addRepeatedField(int index, Field.Builder builderForValue)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder add$capitalized_name$(\n"
    "    int index, $type$.Builder builderForValue)",

    "ensure$capitalized_name$IsMutable();\n"
    "$name$_.add(index, com.google.protobuf.LazyFieldLite.fromValue(\n"
    "    builderForValue.build()));\n"
    "$on_changed$\n",

    "$name$Builder_.addMessage(index, builderForValue.build());\n",

    "return this;\n");

  // Builder addAllRepeatedField(Iterable<Field> values)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder addAll$capitalized_name$(\n"
    "    java.lang.Iterable<? extends $type$> values)",

    "ensure$capitalized_name$IsMutable();\n"
    "for (com.google.protobuf.MessageLite v : values) {\n"
    "  $name$_.add(com.google.protobuf.LazyFieldLite.fromValue(v));\n"
    "}\n"
    "$on_changed$\n",

    "$name$Builder_.addAllMessages(values);\n",

    "return this;\n");

  // Builder clearAllRepeatedField()
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder clear$capitalized_name$()",

    "$name$_ = java.util.Collections.emptyList();\n"
    "$clear_mutable_bit_builder$;\n"
    "$on_changed$\n",

    "$name$Builder_.clear();\n",

    "return this;\n");

  // Builder removeRepeatedField(int index)
  WriteFieldDocComment(printer, descriptor_);
  PrintNestedBuilderFunction(printer,
    "$deprecation$public Builder remove$capitalized_name$(int index)",

    "ensure$capitalized_name$IsMutable();\n"
    "$name$_.remove(index);\n"
    "$on_changed$\n",

    "$name$Builder_.remove(index);\n",

    "return this;\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$.Builder get$capitalized_name$Builder(\n"
    "    int index) {\n"
    "  return get$capitalized_name$FieldBuilder().getBuilder(index);\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
      printer->Print(variables_,
    "$deprecation$public $type$OrBuilder get$capitalized_name$OrBuilder(\n"
    "    int index) {\n"
    "  if ($name$Builder_ == null) {\n"
    "    return $name$_.get(index);"
    "  } else {\n"
    "    return $name$Builder_.getMessageOrBuilder(index);\n"
    "  }\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
      printer->Print(variables_,
    "$deprecation$public java.util.List<? extends $type$OrBuilder> \n"
    "     get$capitalized_name$OrBuilderList() {\n"
    "  if ($name$Builder_ != null) {\n"
    "    return $name$Builder_.getMessageOrBuilderList();\n"
    "  } else {\n"
    "    return java.util.Collections.unmodifiableList($name$_);\n"
    "  }\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
      printer->Print(variables_,
    "$deprecation$public $type$.Builder add$capitalized_name$Builder() {\n"
    "  return get$capitalized_name$FieldBuilder().addBuilder(\n"
    "      $type$.getDefaultInstance());\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
      printer->Print(variables_,
    "$deprecation$public $type$.Builder add$capitalized_name$Builder(\n"
    "    int index) {\n"
    "  return get$capitalized_name$FieldBuilder().addBuilder(\n"
    "      index, $type$.getDefaultInstance());\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
      printer->Print(variables_,
    "$deprecation$public java.util.List<$type$.Builder> \n"
    "     get$capitalized_name$BuilderList() {\n"
    "  return get$capitalized_name$FieldBuilder().getBuilderList();\n"
    "}\n"
    "private com.google.protobuf.RepeatedFieldBuilder<\n"
    "    $type$, $type$.Builder, $type$OrBuilder> \n"
    "    get$capitalized_name$FieldBuilder() {\n"
    "  if ($name$Builder_ == null) {\n"
    "    $name$Builder_ = new com.google.protobuf.RepeatedFieldBuilder<\n"
    "        $type$, $type$.Builder, $type$OrBuilder>(\n"
    "            $name$_,\n"
    "            $get_mutable_bit_builder$,\n"
    "            getParentForChildren(),\n"
    "            isClean());\n"
    "    $name$_ = null;\n"
    "  }\n"
    "  return $name$Builder_;\n"
    "}\n");
}

void RepeatedImmutableLazyMessageFieldGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!$get_mutable_bit_parser$) {\n"
    "  $name$_ =\n"
    "      new java.util.ArrayList<com.google.protobuf.LazyFieldLite>();\n"
    "  $set_mutable_bit_parser$;\n"
    "}\n"
    "$name$_.add(new com.google.protobuf.LazyFieldLite(\n"
    "    extensionRegistry, input.readBytes()));\n");
}

void RepeatedImmutableLazyMessageFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "for (int i = 0; i < $name$_.size(); i++) {\n"
    "  output.writeBytes($number$, $name$_.get(i).toByteString());\n"
    "}\n");
}

void RepeatedImmutableLazyMessageFieldGenerator::
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
