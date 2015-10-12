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

#include <map>
#include <string>

#include <google/protobuf/compiler/javanano/javanano_message_field.h>
#include <google/protobuf/compiler/javanano/javanano_helpers.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

using internal::WireFormat;
using internal::WireFormatLite;

namespace {

// TODO(kenton):  Factor out a "SetCommonFieldVariables()" to get rid of
//   repeat code between this and the other field types.
void SetMessageVariables(const Params& params,
    const FieldDescriptor* descriptor, map<string, string>* variables) {
  (*variables)["name"] =
    RenameJavaKeywords(UnderscoresToCamelCase(descriptor));
  (*variables)["capitalized_name"] =
    RenameJavaKeywords(UnderscoresToCapitalizedCamelCase(descriptor));
  (*variables)["number"] = SimpleItoa(descriptor->number());
  (*variables)["type"] = ClassName(params, descriptor->message_type());
  (*variables)["group_or_message"] =
    (descriptor->type() == FieldDescriptor::TYPE_GROUP) ?
    "Group" : "Message";
  (*variables)["message_name"] = descriptor->containing_type()->name();
  //(*variables)["message_type"] = descriptor->message_type()->name();
  (*variables)["tag"] = SimpleItoa(WireFormat::MakeTag(descriptor));
}

}  // namespace

// ===================================================================

MessageFieldGenerator::
MessageFieldGenerator(const FieldDescriptor* descriptor, const Params& params)
  : FieldGenerator(params), descriptor_(descriptor) {
  SetMessageVariables(params, descriptor, &variables_);
}

MessageFieldGenerator::~MessageFieldGenerator() {}

void MessageFieldGenerator::
GenerateMembers(io::Printer* printer, bool /* unused lazy_init */) const {
  printer->Print(variables_,
    "public $type$ $name$;\n");
}

void MessageFieldGenerator::
GenerateClearCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$ = null;\n");
}

void MessageFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ == null) {\n"
    "  this.$name$ = new $type$();\n"
    "}\n");

  if (descriptor_->type() == FieldDescriptor::TYPE_GROUP) {
    printer->Print(variables_,
      "input.readGroup(this.$name$, $number$);\n");
  } else {
    printer->Print(variables_,
      "input.readMessage(this.$name$);\n");
  }
}

void MessageFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null) {\n"
    "  output.write$group_or_message$($number$, this.$name$);\n"
    "}\n");
}

void MessageFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null) {\n"
    "  size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
    "    .compute$group_or_message$Size($number$, this.$name$);\n"
    "}\n");
}

void MessageFieldGenerator::
GenerateFixClonedCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null) {\n"
    "  cloned.$name$ = this.$name$.clone();\n"
    "}\n");
}

void MessageFieldGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ == null) { \n"
    "  if (other.$name$ != null) {\n"
    "    return false;\n"
    "  }\n"
    "} else {\n"
    "  if (!this.$name$.equals(other.$name$)) {\n"
    "    return false;\n"
    "  }\n"
    "}\n");
}

void MessageFieldGenerator::
GenerateHashCodeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = 31 * result +\n"
    "    (this.$name$ == null ? 0 : this.$name$.hashCode());\n");
}
// ===================================================================

MessageOneofFieldGenerator::MessageOneofFieldGenerator(
    const FieldDescriptor* descriptor, const Params& params)
    : FieldGenerator(params), descriptor_(descriptor) {
    SetMessageVariables(params, descriptor, &variables_);
    SetCommonOneofVariables(descriptor, &variables_);
}

MessageOneofFieldGenerator::~MessageOneofFieldGenerator() {}

void MessageOneofFieldGenerator::
GenerateMembers(io::Printer* printer, bool /* unused lazy_init */) const {
  printer->Print(variables_,
    "public boolean has$capitalized_name$() {\n"
    "  return $has_oneof_case$;\n"
    "}\n"
    "public $type$ get$capitalized_name$() {\n"
    "  if ($has_oneof_case$) {\n"
    "    return ($type$) this.$oneof_name$_;\n"
    "  }\n"
    "  return null;\n"
    "}\n"
    "public $message_name$ set$capitalized_name$($type$ value) {\n"
    "  if (value == null) { throw new java.lang.NullPointerException(); }\n"
    "  $set_oneof_case$;\n"
    "  this.$oneof_name$_ = value;\n"
    "  return this;\n"
    "}\n");
}

void MessageOneofFieldGenerator::
GenerateClearCode(io::Printer* printer) const {
  // No clear method for oneof fields.
}

void MessageOneofFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!($has_oneof_case$)) {\n"
    "  this.$oneof_name$_ = new $type$();\n"
    "}\n"
    "input.readMessage(\n"
    "    (com.google.protobuf.nano.MessageNano) this.$oneof_name$_);\n"
    "$set_oneof_case$;\n");
}

void MessageOneofFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case$) {\n"
    "  output.writeMessage($number$,\n"
    "      (com.google.protobuf.nano.MessageNano) this.$oneof_name$_);\n"
    "}\n");
}

void MessageOneofFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case$) {\n"
    "  size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
    "    .computeMessageSize($number$,\n"
    "        (com.google.protobuf.nano.MessageNano) this.$oneof_name$_);\n"
    "}\n");
}

void MessageOneofFieldGenerator::
GenerateFixClonedCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$oneof_name$ != null) {\n"
    "  cloned.$oneof_name$ = this.$oneof_name$.clone();\n"
    "}\n");
}

void MessageOneofFieldGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  GenerateOneofFieldEquals(descriptor_, variables_, printer);
}

void MessageOneofFieldGenerator::
GenerateHashCodeCode(io::Printer* printer) const {
  GenerateOneofFieldHashCode(descriptor_, variables_, printer);
}

// ===================================================================

RepeatedMessageFieldGenerator::RepeatedMessageFieldGenerator(
    const FieldDescriptor* descriptor, const Params& params)
    : FieldGenerator(params), descriptor_(descriptor) {
  SetMessageVariables(params, descriptor, &variables_);
}

RepeatedMessageFieldGenerator::~RepeatedMessageFieldGenerator() {}

void RepeatedMessageFieldGenerator::
GenerateMembers(io::Printer* printer, bool /* unused lazy_init */) const {
  printer->Print(variables_,
    "public $type$[] $name$;\n");
}

void RepeatedMessageFieldGenerator::
GenerateClearCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$ = $type$.emptyArray();\n");
}

void RepeatedMessageFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  // First, figure out the length of the array, then parse.
  printer->Print(variables_,
    "int arrayLength = com.google.protobuf.nano.WireFormatNano\n"
    "    .getRepeatedFieldArrayLength(input, $tag$);\n"
    "int i = this.$name$ == null ? 0 : this.$name$.length;\n"
    "$type$[] newArray =\n"
    "    new $type$[i + arrayLength];\n"
    "if (i != 0) {\n"
    "  java.lang.System.arraycopy(this.$name$, 0, newArray, 0, i);\n"
    "}\n"
    "for (; i < newArray.length - 1; i++) {\n"
    "  newArray[i] = new $type$();\n");

  if (descriptor_->type() == FieldDescriptor::TYPE_GROUP) {
    printer->Print(variables_,
      "  input.readGroup(newArray[i], $number$);\n");
  } else {
    printer->Print(variables_,
      "  input.readMessage(newArray[i]);\n");
  }

  printer->Print(variables_,
    "  input.readTag();\n"
    "}\n"
    "// Last one without readTag.\n"
    "newArray[i] = new $type$();\n");

  if (descriptor_->type() == FieldDescriptor::TYPE_GROUP) {
    printer->Print(variables_,
      "input.readGroup(newArray[i], $number$);\n");
  } else {
    printer->Print(variables_,
      "input.readMessage(newArray[i]);\n");
  }

  printer->Print(variables_,
    "this.$name$ = newArray;\n");
}

void RepeatedMessageFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null && this.$name$.length > 0) {\n"
    "  for (int i = 0; i < this.$name$.length; i++) {\n"
    "    $type$ element = this.$name$[i];\n"
    "    if (element != null) {\n"
    "      output.write$group_or_message$($number$, element);\n"
    "    }\n"
    "  }\n"
    "}\n");
}

void RepeatedMessageFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null && this.$name$.length > 0) {\n"
    "  for (int i = 0; i < this.$name$.length; i++) {\n"
    "    $type$ element = this.$name$[i];\n"
    "    if (element != null) {\n"
    "      size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
    "        .compute$group_or_message$Size($number$, element);\n"
    "    }\n"
    "  }\n"
    "}\n");
}

void RepeatedMessageFieldGenerator::
GenerateFixClonedCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null && this.$name$.length > 0) {\n"
    "  cloned.$name$ = new $type$[this.$name$.length];\n"
    "  for (int i = 0; i < this.$name$.length; i++) {\n"
    "    if (this.$name$[i] != null) {\n"
    "      cloned.$name$[i] = this.$name$[i].clone();\n"
    "    }\n"
    "  }\n"
    "}\n");
}

void RepeatedMessageFieldGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!com.google.protobuf.nano.InternalNano.equals(\n"
    "    this.$name$, other.$name$)) {\n"
    "  return false;\n"
    "}\n");
}

void RepeatedMessageFieldGenerator::
GenerateHashCodeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = 31 * result\n"
    "    + com.google.protobuf.nano.InternalNano.hashCode(this.$name$);\n");
}

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
