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

#include <sstream>

#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>

#include <google/protobuf/compiler/csharp/csharp_doc_comment.h>
#include <google/protobuf/compiler/csharp/csharp_helpers.h>
#include <google/protobuf/compiler/csharp/csharp_wrapper_field.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace csharp {

WrapperFieldGenerator::WrapperFieldGenerator(const FieldDescriptor* descriptor,
                                       int fieldOrdinal)
    : FieldGeneratorBase(descriptor, fieldOrdinal) {
  variables_["has_property_check"] = name() + "_ != null";
  variables_["has_not_property_check"] = name() + "_ == null";
  const FieldDescriptor* wrapped_field = descriptor->message_type()->field(0);
  is_value_type = wrapped_field->type() != FieldDescriptor::TYPE_STRING &&
      wrapped_field->type() != FieldDescriptor::TYPE_BYTES;
  if (is_value_type) {
    variables_["nonnullable_type_name"] = type_name(wrapped_field);
  }
}

WrapperFieldGenerator::~WrapperFieldGenerator() {
}

void WrapperFieldGenerator::GenerateMembers(io::Printer* printer) {
  printer->Print(
        variables_,
        "private static readonly pb::FieldCodec<$type_name$> _single_$name$_codec = ");
  GenerateCodecCode(printer);
  printer->Print(
    variables_,
    ";\n"
    "private $type_name$ $name$_;\n");
  WritePropertyDocComment(printer, descriptor_);
  AddDeprecatedFlag(printer);
  printer->Print(
    variables_,
    "$access_level$ $type_name$ $property_name$ {\n"
    "  get { return $name$_; }\n"
    "  set {\n"
    "    $name$_ = value;\n"
    "  }\n"
    "}\n");
}

void WrapperFieldGenerator::GenerateMergingCode(io::Printer* printer) {
  printer->Print(
    variables_,
    "if (other.$has_property_check$) {\n"
    "  if ($has_not_property_check$ || other.$property_name$ != $default_value$) {\n"
    "    $property_name$ = other.$property_name$;\n"
    "  }\n"
    "}\n");
}

void WrapperFieldGenerator::GenerateParsingCode(io::Printer* printer) {
  printer->Print(
    variables_,
    "$type_name$ value = _single_$name$_codec.Read(input);\n"
    "if ($has_not_property_check$ || value != $default_value$) {\n"
    "  $property_name$ = value;\n"
    "}\n");
}

void WrapperFieldGenerator::GenerateSerializationCode(io::Printer* printer) {
  printer->Print(
    variables_,
    "if ($has_property_check$) {\n"
    "  _single_$name$_codec.WriteTagAndValue(output, $property_name$);\n"
    "}\n");
}

void WrapperFieldGenerator::GenerateSerializedSizeCode(io::Printer* printer) {
  printer->Print(
    variables_,
    "if ($has_property_check$) {\n"
    "  size += _single_$name$_codec.CalculateSizeWithTag($property_name$);\n"
    "}\n");
}

void WrapperFieldGenerator::WriteHash(io::Printer* printer) {
  printer->Print(
    variables_,
    "if ($has_property_check$) hash ^= $property_name$.GetHashCode();\n");
}

void WrapperFieldGenerator::WriteEquals(io::Printer* printer) {
  printer->Print(
    variables_,
    "if ($property_name$ != other.$property_name$) return false;\n");
}

void WrapperFieldGenerator::WriteToString(io::Printer* printer) {
  // TODO: Implement if we ever actually need it...
}

void WrapperFieldGenerator::GenerateCloningCode(io::Printer* printer) {
  printer->Print(variables_,
    "$property_name$ = other.$property_name$;\n");
}

void WrapperFieldGenerator::GenerateCodecCode(io::Printer* printer) {
  if (is_value_type) {
    printer->Print(
      variables_,
      "pb::FieldCodec.ForStructWrapper<$nonnullable_type_name$>($tag$)");
  } else {
    printer->Print(
      variables_,
      "pb::FieldCodec.ForClassWrapper<$type_name$>($tag$)");
  }
}

WrapperOneofFieldGenerator::WrapperOneofFieldGenerator(const FieldDescriptor* descriptor,
    int fieldOrdinal)
    : WrapperFieldGenerator(descriptor, fieldOrdinal) {
    SetCommonOneofFieldVariables(&variables_);
}

WrapperOneofFieldGenerator::~WrapperOneofFieldGenerator() {
}

void WrapperOneofFieldGenerator::GenerateMembers(io::Printer* printer) {
  // Note: deliberately _oneof_$name$_codec, not _$oneof_name$_codec... we have one codec per field.
  printer->Print(
        variables_,
        "private static readonly pb::FieldCodec<$type_name$> _oneof_$name$_codec = ");
  GenerateCodecCode(printer);
  printer->Print(";\n");
  WritePropertyDocComment(printer, descriptor_);
  AddDeprecatedFlag(printer);
  printer->Print(
    variables_,
    "$access_level$ $type_name$ $property_name$ {\n"
    "  get { return $has_property_check$ ? ($type_name$) $oneof_name$_ : ($type_name$) null; }\n"
    "  set {\n"
    "    $oneof_name$_ = value;\n"
    "    $oneof_name$Case_ = value == null ? $oneof_property_name$OneofCase.None : $oneof_property_name$OneofCase.$property_name$;\n"
    "  }\n"
    "}\n");
}

void WrapperOneofFieldGenerator::GenerateParsingCode(io::Printer* printer) {
  printer->Print(
    variables_,
    "$property_name$ = _oneof_$name$_codec.Read(input);\n");
}

void WrapperOneofFieldGenerator::GenerateSerializationCode(io::Printer* printer) {
  // TODO: I suspect this is wrong...
  printer->Print(
    variables_,
    "if ($has_property_check$) {\n"
    "  _oneof_$name$_codec.WriteTagAndValue(output, ($type_name$) $oneof_name$_);\n"
    "}\n");
}

void WrapperOneofFieldGenerator::GenerateSerializedSizeCode(io::Printer* printer) {
  // TODO: I suspect this is wrong...
  printer->Print(
    variables_,
    "if ($has_property_check$) {\n"
    "  size += _oneof_$name$_codec.CalculateSizeWithTag($property_name$);\n"
    "}\n");
}

}  // namespace csharp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
