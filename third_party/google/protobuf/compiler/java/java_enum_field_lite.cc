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

#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/compiler/java/java_context.h>
#include <google/protobuf/compiler/java/java_doc_comment.h>
#include <google/protobuf/compiler/java/java_enum_field_lite.h>
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

void SetEnumVariables(const FieldDescriptor* descriptor,
                      int messageBitIndex,
                      int builderBitIndex,
                      const FieldGeneratorInfo* info,
                      ClassNameResolver* name_resolver,
                      map<string, string>* variables) {
  SetCommonFieldVariables(descriptor, info, variables);

  (*variables)["type"] =
      name_resolver->GetImmutableClassName(descriptor->enum_type());
  (*variables)["mutable_type"] =
      name_resolver->GetMutableClassName(descriptor->enum_type());
  (*variables)["default"] = ImmutableDefaultValue(descriptor, name_resolver);
  (*variables)["default_number"] = SimpleItoa(
      descriptor->default_value_enum()->number());
  (*variables)["tag"] = SimpleItoa(internal::WireFormat::MakeTag(descriptor));
  (*variables)["tag_size"] = SimpleItoa(
      internal::WireFormat::TagSize(descriptor->number(), GetType(descriptor)));
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
        (*variables)["name"] + "_ != " +
        (*variables)["default"] + ".getNumber()";
  }

  // For repeated builders, the underlying list tracks mutability state.
  (*variables)["is_mutable"] = (*variables)["name"] + "_.isModifiable()";

  (*variables)["get_has_field_bit_from_local"] =
      GenerateGetBitFromLocal(builderBitIndex);
  (*variables)["set_has_field_bit_to_local"] =
      GenerateSetBitToLocal(messageBitIndex);

  if (SupportUnknownEnumValue(descriptor->file())) {
    (*variables)["unknown"] = (*variables)["type"] + ".UNRECOGNIZED";
  } else {
    (*variables)["unknown"] = (*variables)["default"];
  }
}

}  // namespace

// ===================================================================

ImmutableEnumFieldLiteGenerator::
ImmutableEnumFieldLiteGenerator(const FieldDescriptor* descriptor,
                            int messageBitIndex,
                            int builderBitIndex,
                            Context* context)
  : descriptor_(descriptor), messageBitIndex_(messageBitIndex),
    builderBitIndex_(builderBitIndex),
    name_resolver_(context->GetNameResolver()) {
  SetEnumVariables(descriptor, messageBitIndex, builderBitIndex,
                   context->GetFieldGeneratorInfo(descriptor),
                   name_resolver_, &variables_);
}

ImmutableEnumFieldLiteGenerator::~ImmutableEnumFieldLiteGenerator() {}

int ImmutableEnumFieldLiteGenerator::GetNumBitsForMessage() const {
  return 1;
}

int ImmutableEnumFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void ImmutableEnumFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$boolean has$capitalized_name$();\n");
  }
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$int get$capitalized_name$Value();\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$$type$ get$capitalized_name$();\n");
}

void ImmutableEnumFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private int $name$_;\n");
  PrintExtraFieldInfo(variables_, printer);
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return $get_has_field_bit_message$;\n"
      "}\n");
  }
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public int get$capitalized_name$Value() {\n"
      "  return $name$_;\n"
      "}\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  $type$ result = $type$.valueOf($name$_);\n"
    "  return result == null ? $unknown$ : result;\n"
    "}\n");

  // Generate private setters for the builder to proxy into.
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "private void set$capitalized_name$Value(int value) {\n"
      "  $set_has_field_bit_message$"
      "  $name$_ = value;\n"
      "}\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  $set_has_field_bit_message$\n"
    "  $name$_ = value.getNumber();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $clear_has_field_bit_message$\n"
    "  $name$_ = $default_number$;\n"
    "}\n");
}

void ImmutableEnumFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return instance.has$capitalized_name$();\n"
      "}\n");
  }
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public int get$capitalized_name$Value() {\n"
      "  return instance.get$capitalized_name$Value();\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public Builder set$capitalized_name$Value(int value) {\n"
      "  copyOnWrite();\n"
      "  instance.set$capitalized_name$Value(int value);\n"
      "  return this;\n"
      "}\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  return instance.get$capitalized_name$();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$($type$ value) {\n"
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
}

void ImmutableEnumFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  // noop for enums
}

void ImmutableEnumFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = $default_number$;\n");
}

void ImmutableEnumFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    printer->Print(variables_,
      "if (other.has$capitalized_name$()) {\n"
      "  set$capitalized_name$(other.get$capitalized_name$());\n"
      "}\n");
  } else if (SupportUnknownEnumValue(descriptor_->file())) {
    printer->Print(variables_,
      "if (other.$name$_ != $default_number$) {\n"
      "  set$capitalized_name$Value(other.get$capitalized_name$Value());\n"
      "}\n");
  } else {
    GOOGLE_LOG(FATAL) << "Can't reach here.";
  }
}

void ImmutableEnumFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  // noop for scalars
}

void ImmutableEnumFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  if (SupportUnknownEnumValue(descriptor_->file())) {
    printer->Print(variables_,
      "int rawValue = input.readEnum();\n"
      "$set_has_field_bit_message$\n"
      "$name$_ = rawValue;\n");
  } else {
    printer->Print(variables_,
      "int rawValue = input.readEnum();\n"
      "$type$ value = $type$.valueOf(rawValue);\n"
      "if (value == null) {\n");
    if (PreserveUnknownFields(descriptor_->containing_type())) {
      printer->Print(variables_,
        "  super.mergeVarintField($number$, rawValue);\n");
    }
    printer->Print(variables_,
      "} else {\n"
      "  $set_has_field_bit_message$\n"
      "  $name$_ = rawValue;\n"
      "}\n");
  }
}

void ImmutableEnumFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  // noop for enums
}

void ImmutableEnumFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  output.writeEnum($number$, $name$_);\n"
    "}\n");
}

void ImmutableEnumFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeEnumSize($number$, $name$_);\n"
    "}\n");
}

void ImmutableEnumFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = result && $name$_ == other.$name$_;\n");
}

void ImmutableEnumFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "hash = (37 * hash) + $constant_name$;\n"
    "hash = (53 * hash) + $name$_;\n");
}

string ImmutableEnumFieldLiteGenerator::GetBoxedType() const {
  return name_resolver_->GetImmutableClassName(descriptor_->enum_type());
}

// ===================================================================

ImmutableEnumOneofFieldLiteGenerator::
ImmutableEnumOneofFieldLiteGenerator(const FieldDescriptor* descriptor,
                                 int messageBitIndex,
                                 int builderBitIndex,
                                 Context* context)
    : ImmutableEnumFieldLiteGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
  const OneofGeneratorInfo* info =
      context->GetOneofGeneratorInfo(descriptor->containing_oneof());
  SetCommonOneofVariables(descriptor, info, &variables_);
}

ImmutableEnumOneofFieldLiteGenerator::
~ImmutableEnumOneofFieldLiteGenerator() {}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  PrintExtraFieldInfo(variables_, printer);
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return $has_oneof_case_message$;\n"
      "}\n");
  }
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public int get$capitalized_name$Value() {\n"
      "  if ($has_oneof_case_message$) {\n"
      "    return (java.lang.Integer) $oneof_name$_;\n"
      "  }\n"
      "  return $default_number$;\n"
      "}\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  if ($has_oneof_case_message$) {\n"
    "    $type$ result =  $type$.valueOf((java.lang.Integer) $oneof_name$_);\n"
    "    return result == null ? $unknown$ : result;\n"
    "  }\n"
    "  return $default$;\n"
    "}\n");

  // Generate private setters for the builder to proxy into.
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "private void set$capitalized_name$Value(int value) {\n"
      "  $set_oneof_case_message$;\n"
      "  $oneof_name$_ = value;\n"
      "}\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  $set_oneof_case_message$;\n"
    "  $oneof_name$_ = value.getNumber();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  if ($has_oneof_case_message$) {\n"
    "    $clear_oneof_case_message$;\n"
    "    $oneof_name$_ = null;\n"
    "  }\n"
    "}\n");
}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public boolean has$capitalized_name$() {\n"
      "  return instance.has$capitalized_name$();\n"
      "}\n");
  }
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public int get$capitalized_name$Value() {\n"
      "  return instance.get$capitalized_name$Value();\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public Builder set$capitalized_name$Value(int value) {\n"
      "  copyOnWrite();\n"
      "  instance.set$capitalized_name$Value(value);\n"
      "  return this;\n"
      "}\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  return instance.get$capitalized_name$();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$($type$ value) {\n"
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
}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  if (SupportUnknownEnumValue(descriptor_->file())) {
    printer->Print(variables_,
      "set$capitalized_name$Value(other.get$capitalized_name$Value());\n");
  } else {
    printer->Print(variables_,
      "set$capitalized_name$(other.get$capitalized_name$());\n");
  }
}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  if (SupportUnknownEnumValue(descriptor_->file())) {
    printer->Print(variables_,
      "int rawValue = input.readEnum();\n"
      "$set_oneof_case_message$;\n"
      "$oneof_name$_ = rawValue;\n");
  } else {
    printer->Print(variables_,
      "int rawValue = input.readEnum();\n"
      "$type$ value = $type$.valueOf(rawValue);\n"
      "if (value == null) {\n");
    if (PreserveUnknownFields(descriptor_->containing_type())) {
      printer->Print(variables_,
        "  super.mergeVarintField($number$, rawValue);\n");
    }
    printer->Print(variables_,
      "} else {\n"
      "  $set_oneof_case_message$;\n"
      "  $oneof_name$_ = rawValue;\n"
      "}\n");
  }
}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  output.writeEnum($number$, ((java.lang.Integer) $oneof_name$_));\n"
    "}\n");
}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .computeEnumSize($number$, ((java.lang.Integer) $oneof_name$_));\n"
    "}\n");
}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  if (SupportUnknownEnumValue(descriptor_->file())) {
    printer->Print(variables_,
      "result = result && get$capitalized_name$Value()\n"
      "    == other.get$capitalized_name$Value();\n");
  } else {
    printer->Print(variables_,
      "result = result && get$capitalized_name$()\n"
      "    .equals(other.get$capitalized_name$());\n");
  }
}

void ImmutableEnumOneofFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  if (SupportUnknownEnumValue(descriptor_->file())) {
    printer->Print(variables_,
      "hash = (37 * hash) + $constant_name$;\n"
      "hash = (53 * hash) + get$capitalized_name$Value();\n");
  } else {
    printer->Print(variables_,
      "hash = (37 * hash) + $constant_name$;\n"
      "hash = (53 * hash) + get$capitalized_name$().getNumber();\n");
  }
}

// ===================================================================

RepeatedImmutableEnumFieldLiteGenerator::
RepeatedImmutableEnumFieldLiteGenerator(const FieldDescriptor* descriptor,
                                    int messageBitIndex,
                                    int builderBitIndex,
                                    Context* context)
  : descriptor_(descriptor), messageBitIndex_(messageBitIndex),
    builderBitIndex_(builderBitIndex), context_(context),
    name_resolver_(context->GetNameResolver()) {
  SetEnumVariables(descriptor, messageBitIndex, builderBitIndex,
                   context->GetFieldGeneratorInfo(descriptor),
                   name_resolver_, &variables_);
}

RepeatedImmutableEnumFieldLiteGenerator::
~RepeatedImmutableEnumFieldLiteGenerator() {}

int RepeatedImmutableEnumFieldLiteGenerator::GetNumBitsForMessage() const {
  return 0;
}

int RepeatedImmutableEnumFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$java.util.List<$type$> get$capitalized_name$List();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$int get$capitalized_name$Count();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$$type$ get$capitalized_name$(int index);\n");
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$java.util.List<java.lang.Integer>\n"
      "get$capitalized_name$ValueList();\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$int get$capitalized_name$Value(int index);\n");
  }
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    // TODO(dweis): Switch to IntList?
    "private com.google.protobuf.Internal.ProtobufList<\n"
    "    java.lang.Integer> $name$_;\n"
    "private static final com.google.protobuf.Internal.ListAdapter.Converter<\n"
    "    java.lang.Integer, $type$> $name$_converter_ =\n"
    "        new com.google.protobuf.Internal.ListAdapter.Converter<\n"
    "            java.lang.Integer, $type$>() {\n"
    "          public $type$ convert(java.lang.Integer from) {\n"
    "            $type$ result = $type$.valueOf(from);\n"
    "            return result == null ? $unknown$ : result;\n"
    "          }\n"
    "        };\n");
  PrintExtraFieldInfo(variables_, printer);
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<$type$> get$capitalized_name$List() {\n"
    "  return new com.google.protobuf.Internal.ListAdapter<\n"
    "      java.lang.Integer, $type$>($name$_, $name$_converter_);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return $name$_.size();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$(int index) {\n"
    "  return $name$_converter_.convert($name$_.get(index));\n"
    "}\n");
  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public java.util.List<java.lang.Integer>\n"
      "get$capitalized_name$ValueList() {\n"
      "  return $name$_;\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public int get$capitalized_name$Value(int index) {\n"
      "  return $name$_.get(index);\n"
      "}\n");
  }

  if (descriptor_->options().packed() &&
      HasGeneratedMethods(descriptor_->containing_type())) {
    printer->Print(variables_,
      "private int $name$MemoizedSerializedSize;\n");
  }

  // Generate private setters for the builder to proxy into.
  printer->Print(variables_,
    "private void ensure$capitalized_name$IsMutable() {\n"
    "  if (!$is_mutable$) {\n"
    "    $name$_ = newProtobufList($name$_);\n"
    "  }\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    int index, $type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.set(index, value.getNumber());\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$($type$ value) {\n"
    "  if (value == null) {\n"
    "    throw new NullPointerException();\n"
    "  }\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $name$_.add(value.getNumber());\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void addAll$capitalized_name$(\n"
    "    java.lang.Iterable<? extends $type$> values) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  for ($type$ value : values) {\n"
    "    $name$_.add(value.getNumber());\n"
    "  }\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $name$_ = emptyProtobufList();\n"
    "}\n");

  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "private void set$capitalized_name$Value(\n"
      "    int index, int value) {\n"
      "  ensure$capitalized_name$IsMutable();\n"
      "  $name$_.set(index, value);\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "private void add$capitalized_name$Value(int value) {\n"
      "  ensure$capitalized_name$IsMutable();\n"
      "  $name$_.add(value);\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "private void addAll$capitalized_name$Value(\n"
      "    java.lang.Iterable<java.lang.Integer> values) {\n"
      "  ensure$capitalized_name$IsMutable();\n"
      "  for (int value : values) {\n"
      "    $name$_.add(value);\n"
      "  }\n"
      "}\n");
  }
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<$type$> get$capitalized_name$List() {\n"
    "  return instance.get$capitalized_name$List();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return instance.get$capitalized_name$Count();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$(int index) {\n"
    "  return instance.get$capitalized_name$(index);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder set$capitalized_name$(\n"
    "    int index, $type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.set$capitalized_name$(index, value);\n"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder add$capitalized_name$($type$ value) {\n"
    "  copyOnWrite();\n"
    "  instance.add$capitalized_name$(value);\n"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder addAll$capitalized_name$(\n"
    "    java.lang.Iterable<? extends $type$> values) {\n"
    "  copyOnWrite();\n"
    "  instance.addAll$capitalized_name$(values);"
    "  return this;\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public Builder clear$capitalized_name$() {\n"
    "  copyOnWrite();\n"
    "  instance.clear$capitalized_name$();\n"
    "  return this;\n"
    "}\n");

  if (SupportUnknownEnumValue(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public java.util.List<java.lang.Integer>\n"
      "get$capitalized_name$ValueList() {\n"
      "  return java.util.Collections.unmodifiableList(\n"
      "      instance.get$capitalized_name$ValueList());\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public int get$capitalized_name$Value(int index) {\n"
      "  return instance.get$capitalized_name$Value(index);\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public Builder set$capitalized_name$Value(\n"
      "    int index, int value) {\n"
      "  copyOnWrite();\n"
      "  instance.set$capitalized_name$Value(index, value);\n"
      "  return this;\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public Builder add$capitalized_name$Value(int value) {\n"
      "  instance.add$capitalized_name$Value(value);\n"
      "  return this;\n"
      "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$public Builder addAll$capitalized_name$Value(\n"
      "    java.lang.Iterable<java.lang.Integer> values) {\n"
      "  copyOnWrite();\n"
      "  instance.addAll$capitalized_name$Value(values);\n"
      "  return this;\n"
      "}\n");
  }
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  // noop for enums
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = emptyProtobufList();\n");
}

void RepeatedImmutableEnumFieldLiteGenerator::
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

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_.makeImmutable();\n");
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  // Read and store the enum
  if (SupportUnknownEnumValue(descriptor_->file())) {
    printer->Print(variables_,
      "int rawValue = input.readEnum();\n"
      "if (!$is_mutable$) {\n"
      "  $name$_ = newProtobufList();\n"
      "}\n"
      "$name$_.add(rawValue);\n");
  } else {
    printer->Print(variables_,
      "int rawValue = input.readEnum();\n"
      "$type$ value = $type$.valueOf(rawValue);\n"
        "if (value == null) {\n");
    if (PreserveUnknownFields(descriptor_->containing_type())) {
      printer->Print(variables_,
        "  super.mergeVarintField($number$, rawValue);\n");
    }
    printer->Print(variables_,
      "} else {\n"
      "  if (!$is_mutable$) {\n"
      "    $name$_ = newProtobufList();\n"
      "  }\n"
      "  $name$_.add(rawValue);\n"
      "}\n");
  }
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateParsingCodeFromPacked(io::Printer* printer) const {
  // Wrap GenerateParsingCode's contents with a while loop.

  printer->Print(variables_,
    "int length = input.readRawVarint32();\n"
    "int oldLimit = input.pushLimit(length);\n"
    "while(input.getBytesUntilLimit() > 0) {\n");
  printer->Indent();

  GenerateParsingCode(printer);

  printer->Outdent();
  printer->Print(variables_,
    "}\n"
    "input.popLimit(oldLimit);\n");
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_mutable$) {\n"
    "  $name$_.makeImmutable();\n"
    "}\n");
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  if (descriptor_->options().packed()) {
    printer->Print(variables_,
      "if (get$capitalized_name$List().size() > 0) {\n"
      "  output.writeRawVarint32($tag$);\n"
      "  output.writeRawVarint32($name$MemoizedSerializedSize);\n"
      "}\n"
      "for (int i = 0; i < $name$_.size(); i++) {\n"
      "  output.writeEnumNoTag($name$_.get(i));\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "for (int i = 0; i < $name$_.size(); i++) {\n"
      "  output.writeEnum($number$, $name$_.get(i));\n"
      "}\n");
  }
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "{\n"
    "  int dataSize = 0;\n");
  printer->Indent();

  printer->Print(variables_,
    "for (int i = 0; i < $name$_.size(); i++) {\n"
    "  dataSize += com.google.protobuf.CodedOutputStream\n"
    "    .computeEnumSizeNoTag($name$_.get(i));\n"
    "}\n");
  printer->Print(
    "size += dataSize;\n");
  if (descriptor_->options().packed()) {
    printer->Print(variables_,
      "if (!get$capitalized_name$List().isEmpty()) {"
      "  size += $tag_size$;\n"
      "  size += com.google.protobuf.CodedOutputStream\n"
      "    .computeRawVarint32Size(dataSize);\n"
      "}");
  } else {
    printer->Print(variables_,
      "size += $tag_size$ * $name$_.size();\n");
  }

  // cache the data size for packed fields.
  if (descriptor_->options().packed()) {
    printer->Print(variables_,
      "$name$MemoizedSerializedSize = dataSize;\n");
  }

  printer->Outdent();
  printer->Print("}\n");
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = result && $name$_.equals(other.$name$_);\n");
}

void RepeatedImmutableEnumFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (get$capitalized_name$Count() > 0) {\n"
    "  hash = (37 * hash) + $constant_name$;\n"
    "  hash = (53 * hash) + $name$_.hashCode();\n"
    "}\n");
}

string RepeatedImmutableEnumFieldLiteGenerator::GetBoxedType() const {
  return name_resolver_->GetImmutableClassName(descriptor_->enum_type());
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
