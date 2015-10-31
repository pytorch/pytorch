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
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/compiler/java/java_name_resolver.h>
#include <google/protobuf/compiler/java/java_primitive_field_lite.h>
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

  (*variables)["type"] = PrimitiveTypeName(GetJavaType(descriptor));
  (*variables)["boxed_type"] = BoxedPrimitiveTypeName(GetJavaType(descriptor));
  (*variables)["field_type"] = (*variables)["type"];
  (*variables)["default"] = ImmutableDefaultValue(descriptor, name_resolver);
  (*variables)["default_init"] = IsDefaultValueJavaDefault(descriptor) ?
      "" : ("= " + ImmutableDefaultValue(descriptor, name_resolver));
  (*variables)["capitalized_type"] =
      GetCapitalizedType(descriptor, /* immutable = */ true);
  (*variables)["tag"] = SimpleItoa(WireFormat::MakeTag(descriptor));
  (*variables)["tag_size"] = SimpleItoa(
      WireFormat::TagSize(descriptor->number(), GetType(descriptor)));

  string capitalized_type = UnderscoresToCamelCase(PrimitiveTypeName(
        GetJavaType(descriptor)), true /* cap_next_letter */);
  switch (GetJavaType(descriptor)) {
    case JAVATYPE_INT:
    case JAVATYPE_LONG:
    case JAVATYPE_FLOAT:
    case JAVATYPE_DOUBLE:
    case JAVATYPE_BOOLEAN:
      (*variables)["field_list_type"] =
          "com.google.protobuf.Internal." + capitalized_type + "List";
      (*variables)["new_list"] = "new" + capitalized_type + "List";
      (*variables)["empty_list"] = "empty" + capitalized_type + "List()";
      (*variables)["make_name_unmodifiable"] =
          (*variables)["name"] + "_.makeImmutable()";
      (*variables)["repeated_index_get"] =
          (*variables)["name"] + "_.get" + capitalized_type + "(index)";
      (*variables)["repeated_add"] =
          (*variables)["name"] + "_.add" + capitalized_type;
      (*variables)["repeated_set"] =
          (*variables)["name"] + "_.set" + capitalized_type;
      break;
    default:
      (*variables)["field_list_type"] =
          "com.google.protobuf.Internal.ProtobufList<" +
          (*variables)["boxed_type"] + ">";
      (*variables)["new_list"] = "newProtobufList";
      (*variables)["empty_list"] = "emptyProtobufList()";
      (*variables)["make_name_unmodifiable"] =
          (*variables)["name"] + "_.makeImmutable()";
      (*variables)["repeated_index_get"] =
          (*variables)["name"] + "_.get(index)";
      (*variables)["repeated_add"] = (*variables)["name"] + "_.add";
      (*variables)["repeated_set"] = (*variables)["name"] + "_.set";
  }

  if (IsReferenceType(GetJavaType(descriptor))) {
    (*variables)["null_check"] =
        "  if (value == null) {\n"
        "    throw new NullPointerException();\n"
        "  }\n";
  } else {
    (*variables)["null_check"] = "";
  }
  // TODO(birdo): Add @deprecated javadoc when generating javadoc is supported
  // by the proto compiler
  (*variables)["deprecation"] = descriptor->options().deprecated()
      ? "@java.lang.Deprecated " : "";
  int fixed_size = FixedSize(GetType(descriptor));
  if (fixed_size != -1) {
    (*variables)["fixed_size"] = SimpleItoa(fixed_size);
  }
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
    (*variables)["set_has_field_bit_message"] = "";
    (*variables)["clear_has_field_bit_message"] = "";

    if (descriptor->type() == FieldDescriptor::TYPE_BYTES) {
      (*variables)["is_field_present_message"] =
          "!" + (*variables)["name"] + "_.isEmpty()";
    } else {
      (*variables)["is_field_present_message"] =
          (*variables)["name"] + "_ != " + (*variables)["default"];
    }
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

ImmutablePrimitiveFieldLiteGenerator::
ImmutablePrimitiveFieldLiteGenerator(const FieldDescriptor* descriptor,
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

ImmutablePrimitiveFieldLiteGenerator::~ImmutablePrimitiveFieldLiteGenerator() {}

int ImmutablePrimitiveFieldLiteGenerator::GetNumBitsForMessage() const {
  return 1;
}

int ImmutablePrimitiveFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(variables_,
      "$deprecation$boolean has$capitalized_name$();\n");
  }
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$$type$ get$capitalized_name$();\n");
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private $field_type$ $name$_;\n");
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
    "$deprecation$public $type$ get$capitalized_name$() {\n"
    "  return $name$_;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$($type$ value) {\n"
    "$null_check$"
    "  $set_has_field_bit_message$\n"
    "  $name$_ = value;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $clear_has_field_bit_message$\n");
  JavaType type = GetJavaType(descriptor_);
  if (type == JAVATYPE_STRING || type == JAVATYPE_BYTES) {
    // The default value is not a simple literal so we want to avoid executing
    // it multiple times.  Instead, get the default out of the default instance.
    printer->Print(variables_,
      "  $name$_ = getDefaultInstance().get$capitalized_name$();\n");
  } else {
    printer->Print(variables_,
      "  $name$_ = $default$;\n");
  }
  printer->Print(variables_,
    "}\n");
}

void ImmutablePrimitiveFieldLiteGenerator::
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

void ImmutablePrimitiveFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  // noop for primitives
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = $default$;\n");
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateBuilderClearCode(io::Printer* printer) const {
  // noop for primitives
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  if (SupportFieldPresence(descriptor_->file())) {
    printer->Print(variables_,
      "if (other.has$capitalized_name$()) {\n"
      "  set$capitalized_name$(other.get$capitalized_name$());\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "if (other.get$capitalized_name$() != $default$) {\n"
      "  set$capitalized_name$(other.get$capitalized_name$());\n"
      "}\n");
  }
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateBuildingCode(io::Printer* printer) const {
  // noop for primitives
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  // noop for scalars
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$set_has_field_bit_message$\n"
    "$name$_ = input.read$capitalized_type$();\n");
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  // noop for primitives.
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  output.write$capitalized_type$($number$, $name$_);\n"
    "}\n");
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_field_present_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .compute$capitalized_type$Size($number$, $name$_);\n"
    "}\n");
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  switch (GetJavaType(descriptor_)) {
    case JAVATYPE_INT:
    case JAVATYPE_LONG:
    case JAVATYPE_BOOLEAN:
      printer->Print(variables_,
        "result = result && (get$capitalized_name$()\n"
        "    == other.get$capitalized_name$());\n");
      break;

    case JAVATYPE_FLOAT:
      printer->Print(variables_,
        "result = result && (\n"
        "    java.lang.Float.floatToIntBits(get$capitalized_name$())\n"
        "    == java.lang.Float.floatToIntBits(\n"
        "        other.get$capitalized_name$()));\n");
      break;

    case JAVATYPE_DOUBLE:
      printer->Print(variables_,
        "result = result && (\n"
        "    java.lang.Double.doubleToLongBits(get$capitalized_name$())\n"
        "    == java.lang.Double.doubleToLongBits(\n"
        "        other.get$capitalized_name$()));\n");
      break;

    case JAVATYPE_STRING:
    case JAVATYPE_BYTES:
      printer->Print(variables_,
        "result = result && get$capitalized_name$()\n"
        "    .equals(other.get$capitalized_name$());\n");
      break;

    case JAVATYPE_ENUM:
    case JAVATYPE_MESSAGE:
    default:
      GOOGLE_LOG(FATAL) << "Can't get here.";
      break;
  }
}

void ImmutablePrimitiveFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "hash = (37 * hash) + $constant_name$;\n");
  switch (GetJavaType(descriptor_)) {
    case JAVATYPE_INT:
      printer->Print(variables_,
        "hash = (53 * hash) + get$capitalized_name$();\n");
      break;

    case JAVATYPE_LONG:
      printer->Print(variables_,
        "hash = (53 * hash) + com.google.protobuf.Internal.hashLong(\n"
        "    get$capitalized_name$());\n");
      break;

    case JAVATYPE_BOOLEAN:
      printer->Print(variables_,
        "hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(\n"
        "    get$capitalized_name$());\n");
      break;

    case JAVATYPE_FLOAT:
      printer->Print(variables_,
        "hash = (53 * hash) + java.lang.Float.floatToIntBits(\n"
        "    get$capitalized_name$());\n");
      break;

    case JAVATYPE_DOUBLE:
      printer->Print(variables_,
        "hash = (53 * hash) + com.google.protobuf.Internal.hashLong(\n"
        "    java.lang.Double.doubleToLongBits(get$capitalized_name$()));\n");
      break;

    case JAVATYPE_STRING:
    case JAVATYPE_BYTES:
      printer->Print(variables_,
        "hash = (53 * hash) + get$capitalized_name$().hashCode();\n");
      break;

    case JAVATYPE_ENUM:
    case JAVATYPE_MESSAGE:
    default:
      GOOGLE_LOG(FATAL) << "Can't get here.";
      break;
  }
}

string ImmutablePrimitiveFieldLiteGenerator::GetBoxedType() const {
  return BoxedPrimitiveTypeName(GetJavaType(descriptor_));
}

// ===================================================================

ImmutablePrimitiveOneofFieldLiteGenerator::
ImmutablePrimitiveOneofFieldLiteGenerator(const FieldDescriptor* descriptor,
                                 int messageBitIndex,
                                 int builderBitIndex,
                                 Context* context)
    : ImmutablePrimitiveFieldLiteGenerator(
          descriptor, messageBitIndex, builderBitIndex, context) {
  const OneofGeneratorInfo* info =
      context->GetOneofGeneratorInfo(descriptor->containing_oneof());
  SetCommonOneofVariables(descriptor, info, &variables_);
}

ImmutablePrimitiveOneofFieldLiteGenerator::
~ImmutablePrimitiveOneofFieldLiteGenerator() {}

void ImmutablePrimitiveOneofFieldLiteGenerator::
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
    "    return ($boxed_type$) $oneof_name$_;\n"
    "  }\n"
    "  return $default$;\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$($type$ value) {\n"
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
}


void ImmutablePrimitiveOneofFieldLiteGenerator::
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

void ImmutablePrimitiveOneofFieldLiteGenerator::
GenerateBuildingCode(io::Printer* printer) const {
  // noop for primitives
}

void ImmutablePrimitiveOneofFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "set$capitalized_name$(other.get$capitalized_name$());\n");
}

void ImmutablePrimitiveOneofFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$set_oneof_case_message$;\n"
    "$oneof_name$_ = input.read$capitalized_type$();\n");
}

void ImmutablePrimitiveOneofFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  output.write$capitalized_type$(\n"
    "      $number$, ($type$)(($boxed_type$) $oneof_name$_));\n"
    "}\n");
}

void ImmutablePrimitiveOneofFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case_message$) {\n"
    "  size += com.google.protobuf.CodedOutputStream\n"
    "    .compute$capitalized_type$Size(\n"
    "        $number$, ($type$)(($boxed_type$) $oneof_name$_));\n"
    "}\n");
}

// ===================================================================

RepeatedImmutablePrimitiveFieldLiteGenerator::
RepeatedImmutablePrimitiveFieldLiteGenerator(const FieldDescriptor* descriptor,
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

RepeatedImmutablePrimitiveFieldLiteGenerator::
~RepeatedImmutablePrimitiveFieldLiteGenerator() {}

int RepeatedImmutablePrimitiveFieldLiteGenerator::GetNumBitsForMessage() const {
  return 0;
}

int RepeatedImmutablePrimitiveFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$java.util.List<$boxed_type$> get$capitalized_name$List();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$int get$capitalized_name$Count();\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$$type$ get$capitalized_name$(int index);\n");
}


void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "private $field_list_type$ $name$_;\n");
  PrintExtraFieldInfo(variables_, printer);
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<$boxed_type$>\n"
    "    get$capitalized_name$List() {\n"
    "  return $name$_;\n"   // note:  unmodifiable list
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public int get$capitalized_name$Count() {\n"
    "  return $name$_.size();\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public $type$ get$capitalized_name$(int index) {\n"
    "  return $repeated_index_get$;\n"
    "}\n");

  if (descriptor_->options().packed() &&
      HasGeneratedMethods(descriptor_->containing_type())) {
    printer->Print(variables_,
      "private int $name$MemoizedSerializedSize = -1;\n");
  }

  printer->Print(variables_,
    "private void ensure$capitalized_name$IsMutable() {\n"
    "  if (!$is_mutable$) {\n"
    "    $name$_ = $new_list$($name$_);\n"
    "   }\n"
    "}\n");

  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void set$capitalized_name$(\n"
    "    int index, $type$ value) {\n"
    "$null_check$"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $repeated_set$(index, value);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void add$capitalized_name$($type$ value) {\n"
    "$null_check$"
    "  ensure$capitalized_name$IsMutable();\n"
    "  $repeated_add$(value);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void addAll$capitalized_name$(\n"
    "    java.lang.Iterable<? extends $boxed_type$> values) {\n"
    "  ensure$capitalized_name$IsMutable();\n"
    "  com.google.protobuf.AbstractMessageLite.addAll(\n"
    "      values, $name$_);\n"
    "}\n");
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "private void clear$capitalized_name$() {\n"
    "  $name$_ = $empty_list$;\n"
    "}\n");
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  WriteFieldDocComment(printer, descriptor_);
  printer->Print(variables_,
    "$deprecation$public java.util.List<$boxed_type$>\n"
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
    "    java.lang.Iterable<? extends $boxed_type$> values) {\n"
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
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  // noop for primitives
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = $empty_list$;\n");
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateBuilderClearCode(io::Printer* printer) const {
  // noop for primitives
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
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

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateBuildingCode(io::Printer* printer) const {
  // noop for primitives
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_.makeImmutable();\n");
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!$is_mutable$) {\n"
    "  $name$_ = $new_list$();\n"
    "}\n"
    "$repeated_add$(input.read$capitalized_type$());\n");
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateParsingCodeFromPacked(io::Printer* printer) const {
  printer->Print(variables_,
    "int length = input.readRawVarint32();\n"
    "int limit = input.pushLimit(length);\n"
    "if (!$is_mutable$ && input.getBytesUntilLimit() > 0) {\n"
    "  $name$_ = $new_list$();\n"
    "}\n"
    "while (input.getBytesUntilLimit() > 0) {\n"
    "  $repeated_add$(input.read$capitalized_type$());\n"
    "}\n"
    "input.popLimit(limit);\n");
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($is_mutable$) {\n"
    "  $make_name_unmodifiable$;\n"
    "}\n");
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  if (descriptor_->options().packed()) {
    // We invoke getSerializedSize in writeTo for messages that have packed
    // fields in ImmutableMessageGenerator::GenerateMessageSerializationMethods.
    // That makes it safe to rely on the memoized size here.
    printer->Print(variables_,
      "if (get$capitalized_name$List().size() > 0) {\n"
      "  output.writeRawVarint32($tag$);\n"
      "  output.writeRawVarint32($name$MemoizedSerializedSize);\n"
      "}\n"
      "for (int i = 0; i < $name$_.size(); i++) {\n"
      "  output.write$capitalized_type$NoTag($name$_.get(i));\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "for (int i = 0; i < $name$_.size(); i++) {\n"
      "  output.write$capitalized_type$($number$, $name$_.get(i));\n"
      "}\n");
  }
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "{\n"
    "  int dataSize = 0;\n");
  printer->Indent();

  if (FixedSize(GetType(descriptor_)) == -1) {
    printer->Print(variables_,
      "for (int i = 0; i < $name$_.size(); i++) {\n"
      "  dataSize += com.google.protobuf.CodedOutputStream\n"
      "    .compute$capitalized_type$SizeNoTag($name$_.get(i));\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "dataSize = $fixed_size$ * get$capitalized_name$List().size();\n");
  }

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

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = result && get$capitalized_name$List()\n"
    "    .equals(other.get$capitalized_name$List());\n");
}

void RepeatedImmutablePrimitiveFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (get$capitalized_name$Count() > 0) {\n"
    "  hash = (37 * hash) + $constant_name$;\n"
    "  hash = (53 * hash) + get$capitalized_name$List().hashCode();\n"
    "}\n");
}

string RepeatedImmutablePrimitiveFieldLiteGenerator::GetBoxedType() const {
  return BoxedPrimitiveTypeName(GetJavaType(descriptor_));
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
