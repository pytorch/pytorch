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

#include <google/protobuf/compiler/java/java_map_field_lite.h>

#include <google/protobuf/compiler/java/java_context.h>
#include <google/protobuf/compiler/java/java_doc_comment.h>
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/compiler/java/java_name_resolver.h>
#include <google/protobuf/io/printer.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

namespace {

const FieldDescriptor* KeyField(const FieldDescriptor* descriptor) {
  GOOGLE_CHECK_EQ(FieldDescriptor::TYPE_MESSAGE, descriptor->type());
  const Descriptor* message = descriptor->message_type();
  GOOGLE_CHECK(message->options().map_entry());
  return message->FindFieldByName("key");
}

const FieldDescriptor* ValueField(const FieldDescriptor* descriptor) {
  GOOGLE_CHECK_EQ(FieldDescriptor::TYPE_MESSAGE, descriptor->type());
  const Descriptor* message = descriptor->message_type();
  GOOGLE_CHECK(message->options().map_entry());
  return message->FindFieldByName("value");
}

string TypeName(const FieldDescriptor* field,
                ClassNameResolver* name_resolver,
                bool boxed) {
  if (GetJavaType(field) == JAVATYPE_MESSAGE) {
    return name_resolver->GetImmutableClassName(field->message_type());
  } else if (GetJavaType(field) == JAVATYPE_ENUM) {
    return name_resolver->GetImmutableClassName(field->enum_type());
  } else {
    return boxed ? BoxedPrimitiveTypeName(GetJavaType(field))
                 : PrimitiveTypeName(GetJavaType(field));
  }
}

string WireType(const FieldDescriptor* field) {
  return "com.google.protobuf.WireFormat.FieldType." +
      string(FieldTypeName(field->type()));
}

void SetMessageVariables(const FieldDescriptor* descriptor,
                         int messageBitIndex,
                         int builderBitIndex,
                         const FieldGeneratorInfo* info,
                         ClassNameResolver* name_resolver,
                         map<string, string>* variables) {
  SetCommonFieldVariables(descriptor, info, variables);

  (*variables)["type"] =
      name_resolver->GetImmutableClassName(descriptor->message_type());
  const FieldDescriptor* key = KeyField(descriptor);
  const FieldDescriptor* value = ValueField(descriptor);
  (*variables)["key_type"] = TypeName(key, name_resolver, false);
  (*variables)["boxed_key_type"] = TypeName(key, name_resolver, true);
  (*variables)["key_wire_type"] = WireType(key);
  (*variables)["key_default_value"] = DefaultValue(key, true, name_resolver);
  if (GetJavaType(value) == JAVATYPE_ENUM) {
    // We store enums as Integers internally.
    (*variables)["value_type"] = "int";
    (*variables)["boxed_value_type"] = "java.lang.Integer";
    (*variables)["value_wire_type"] = WireType(value);
    (*variables)["value_default_value"] =
        DefaultValue(value, true, name_resolver) + ".getNumber()";

    (*variables)["value_enum_type"] = TypeName(value, name_resolver, false);

    if (SupportUnknownEnumValue(descriptor->file())) {
      // Map unknown values to a special UNRECOGNIZED value if supported.
      (*variables)["unrecognized_value"] =
          (*variables)["value_enum_type"] + ".UNRECOGNIZED";
    } else {
      // Map unknown values to the default value if we don't have UNRECOGNIZED.
      (*variables)["unrecognized_value"] =
          DefaultValue(value, true, name_resolver);
    }
  } else {
    (*variables)["value_type"] = TypeName(value, name_resolver, false);
    (*variables)["boxed_value_type"] = TypeName(value, name_resolver, true);
    (*variables)["value_wire_type"] = WireType(value);
    (*variables)["value_default_value"] =
        DefaultValue(value, true, name_resolver);
  }
  (*variables)["type_parameters"] =
      (*variables)["boxed_key_type"] + ", " + (*variables)["boxed_value_type"];
  // TODO(birdo): Add @deprecated javadoc when generating javadoc is supported
  // by the proto compiler
  (*variables)["deprecation"] = descriptor->options().deprecated()
      ? "@java.lang.Deprecated " : "";
  (*variables)["on_changed"] =
      HasDescriptorMethods(descriptor->containing_type()) ? "onChanged();" : "";

  (*variables)["default_entry"] = (*variables)["capitalized_name"] +
      "DefaultEntryHolder.defaultEntry";
  (*variables)["lite"] = "Lite";
  (*variables)["descriptor"] = "";
}

}  // namespace

ImmutableMapFieldLiteGenerator::
ImmutableMapFieldLiteGenerator(const FieldDescriptor* descriptor,
                                       int messageBitIndex,
                                       int builderBitIndex,
                                       Context* context)
  : descriptor_(descriptor), name_resolver_(context->GetNameResolver())  {
  SetMessageVariables(descriptor, messageBitIndex, builderBitIndex,
                      context->GetFieldGeneratorInfo(descriptor),
                      name_resolver_, &variables_);
}

ImmutableMapFieldLiteGenerator::
~ImmutableMapFieldLiteGenerator() {}

int ImmutableMapFieldLiteGenerator::GetNumBitsForMessage() const {
  return 0;
}

int ImmutableMapFieldLiteGenerator::GetNumBitsForBuilder() const {
  return 0;
}

void ImmutableMapFieldLiteGenerator::
GenerateInterfaceMembers(io::Printer* printer) const {
  if (GetJavaType(ValueField(descriptor_)) == JAVATYPE_ENUM) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "$deprecation$java.util.Map<$boxed_key_type$, $value_enum_type$>\n"
        "get$capitalized_name$();\n");
    if (SupportUnknownEnumValue(descriptor_->file())) {
      WriteFieldDocComment(printer, descriptor_);
      printer->Print(
          variables_,
          "$deprecation$java.util.Map<$type_parameters$>\n"
          "get$capitalized_name$Value();\n");
    }
  } else {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "$deprecation$java.util.Map<$type_parameters$>\n"
        "get$capitalized_name$();\n");
  }
}

void ImmutableMapFieldLiteGenerator::
GenerateMembers(io::Printer* printer) const {
  printer->Print(
      variables_,
      "private static final class $capitalized_name$DefaultEntryHolder {\n"
      "  static final com.google.protobuf.MapEntry$lite$<\n"
      "      $type_parameters$> defaultEntry =\n"
      "          com.google.protobuf.MapEntry$lite$\n"
      "          .<$type_parameters$>newDefaultInstance(\n"
      "              $descriptor$\n"
      "              $key_wire_type$,\n"
      "              $key_default_value$,\n"
      "              $value_wire_type$,\n"
      "              $value_default_value$);\n"
      "}\n");
  printer->Print(
      variables_,
      "private com.google.protobuf.MapField$lite$<\n"
      "    $type_parameters$> $name$_ =\n"
      "        com.google.protobuf.MapField$lite$.emptyMapField();\n"
      "private com.google.protobuf.MapField$lite$<$type_parameters$>\n"
      "internalGet$capitalized_name$() {\n"
      "  return $name$_;\n"
      "}\n"
      "private com.google.protobuf.MapField$lite$<$type_parameters$>\n"
      "internalGetMutable$capitalized_name$() {\n"
      "  if (!$name$_.isMutable()) {\n"
      "    $name$_ = $name$_.copy();\n"
      "  }\n"
      "  return $name$_;\n"
      "}\n");
  if (GetJavaType(ValueField(descriptor_)) == JAVATYPE_ENUM) {
    printer->Print(
        variables_,
        "private static final\n"
        "com.google.protobuf.Internal.MapAdapter.Converter<\n"
        "    java.lang.Integer, $value_enum_type$> $name$ValueConverter =\n"
        "        com.google.protobuf.Internal.MapAdapter.newEnumConverter(\n"
        "            $value_enum_type$.internalGetValueMap(),\n"
        "            $unrecognized_value$);\n");
    if (SupportUnknownEnumValue(descriptor_->file())) {
      WriteFieldDocComment(printer, descriptor_);
      printer->Print(
          variables_,
          "$deprecation$\n"
          "public java.util.Map<$boxed_key_type$, $boxed_value_type$>\n"
          "get$capitalized_name$Value() {\n"
          "  return internalGet$capitalized_name$().getMap();\n"
          "}\n");
    }
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "$deprecation$\n"
        "public java.util.Map<$boxed_key_type$, $value_enum_type$>\n"
        "get$capitalized_name$() {\n"
        "  return new com.google.protobuf.Internal.MapAdapter<\n"
        "      $boxed_key_type$, $value_enum_type$, java.lang.Integer>(\n"
        "          internalGet$capitalized_name$().getMap(),\n"
        "          $name$ValueConverter);\n"
        "}\n");
  } else {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "$deprecation$\n"
        "public java.util.Map<$type_parameters$> get$capitalized_name$() {\n"
        "  return internalGet$capitalized_name$().getMap();\n"
        "}\n");
  }

  // Generate private setters for the builder to proxy into.
  if (GetJavaType(ValueField(descriptor_)) == JAVATYPE_ENUM) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "private java.util.Map<$boxed_key_type$, $value_enum_type$>\n"
        "getMutable$capitalized_name$() {\n"
        "  return new com.google.protobuf.Internal.MapAdapter<\n"
        "      $boxed_key_type$, $value_enum_type$, java.lang.Integer>(\n"
        "          internalGetMutable$capitalized_name$().getMutableMap(),\n"
        "          $name$ValueConverter);\n"
        "}\n");
    if (SupportUnknownEnumValue(descriptor_->file())) {
      WriteFieldDocComment(printer, descriptor_);
      printer->Print(
          variables_,
          "private java.util.Map<$boxed_key_type$, $boxed_value_type$>\n"
          "getMutable$capitalized_name$Value() {\n"
          "  return internalGetMutable$capitalized_name$().getMutableMap();\n"
          "}\n");
    }
  } else {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "private java.util.Map<$type_parameters$>\n"
        "getMutable$capitalized_name$() {\n"
        "  return internalGetMutable$capitalized_name$().getMutableMap();\n"
        "}\n");
  }
}

void ImmutableMapFieldLiteGenerator::
GenerateBuilderMembers(io::Printer* printer) const {
  if (GetJavaType(ValueField(descriptor_)) == JAVATYPE_ENUM) {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "$deprecation$\n"
        "public java.util.Map<$boxed_key_type$, $value_enum_type$>\n"
        "get$capitalized_name$() {\n"
        "  return instance.get$capitalized_name$();\n"
        "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "$deprecation$\n"
        "public java.util.Map<$boxed_key_type$, $value_enum_type$>\n"
        "getMutable$capitalized_name$() {\n"
        "  copyOnWrite();\n"
        "  return instance.getMutable$capitalized_name$();\n"
        "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "$deprecation$public Builder putAll$capitalized_name$(\n"
        "    java.util.Map<$boxed_key_type$, $value_enum_type$> values) {\n"
        "  getMutable$capitalized_name$().putAll(values);\n"
        "  return this;\n"
        "}\n");
    if (SupportUnknownEnumValue(descriptor_->file())) {
      WriteFieldDocComment(printer, descriptor_);
      printer->Print(
          variables_,
          "$deprecation$\n"
          "public java.util.Map<$boxed_key_type$, $boxed_value_type$>\n"
          "get$capitalized_name$Value() {\n"
          "  return instance.get$capitalized_name$Value();\n"
          "}\n");
      WriteFieldDocComment(printer, descriptor_);
      printer->Print(
          variables_,
          "$deprecation$\n"
          "public java.util.Map<$boxed_key_type$, $boxed_value_type$>\n"
          "getMutable$capitalized_name$Value() {\n"
          "  copyOnWrite();\n"
          "  return instance.getMutable$capitalized_name$Value();\n"
          "}\n");
      WriteFieldDocComment(printer, descriptor_);
      printer->Print(
          variables_,
          "$deprecation$public Builder putAll$capitalized_name$Value(\n"
          "    java.util.Map<$boxed_key_type$, $boxed_value_type$> values) {\n"
          "  getMutable$capitalized_name$Value().putAll(values);\n"
          "  return this;\n"
          "}\n");
    }
  } else {
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "public java.util.Map<$type_parameters$> get$capitalized_name$() {\n"
        "  return instance.get$capitalized_name$();\n"
        "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "public java.util.Map<$type_parameters$>\n"
        "getMutable$capitalized_name$() {\n"
        "  copyOnWrite();\n"
        "  return instance.getMutable$capitalized_name$();\n"
        "}\n");
    WriteFieldDocComment(printer, descriptor_);
    printer->Print(
        variables_,
        "public Builder putAll$capitalized_name$(\n"
        "    java.util.Map<$type_parameters$> values) {\n"
        "  getMutable$capitalized_name$().putAll(values);\n"
        "  return this;\n"
        "}\n");
  }
}

void ImmutableMapFieldLiteGenerator::
GenerateFieldBuilderInitializationCode(io::Printer* printer)  const {
  // Nothing to initialize.
}

void ImmutableMapFieldLiteGenerator::
GenerateInitializationCode(io::Printer* printer) const {
  // Nothing to initialize.
}

void ImmutableMapFieldLiteGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(
      variables_,
      "internalGetMutable$capitalized_name$().mergeFrom(\n"
      "    other.internalGet$capitalized_name$());\n");
}

void ImmutableMapFieldLiteGenerator::
GenerateDynamicMethodMakeImmutableCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_.makeImmutable();\n");
}

void ImmutableMapFieldLiteGenerator::
GenerateParsingCode(io::Printer* printer) const {
  printer->Print(
      variables_,
      "if (!$name$_.isMutable()) {\n"
      "  $name$_ = $name$_.copy();\n"
      "}\n");
  if (!SupportUnknownEnumValue(descriptor_->file()) &&
      GetJavaType(ValueField(descriptor_)) == JAVATYPE_ENUM) {
    printer->Print(
        variables_,
        "com.google.protobuf.ByteString bytes = input.readBytes();\n"
        "com.google.protobuf.MapEntry$lite$<$type_parameters$>\n"
        "$name$ = $default_entry$.getParserForType().parseFrom(bytes);\n");
    printer->Print(
        variables_,
        "if ($value_enum_type$.valueOf($name$.getValue()) == null) {\n"
        "  super.mergeLengthDelimitedField($number$, bytes);\n"
        "} else {\n"
        "  $name$_.getMutableMap().put($name$.getKey(), $name$.getValue());\n"
        "}\n");
  } else {
    printer->Print(
        variables_,
        "com.google.protobuf.MapEntry$lite$<$type_parameters$>\n"
        "$name$ = input.readMessage(\n"
        "    $default_entry$.getParserForType(), extensionRegistry);\n"
        "$name$_.getMutableMap().put($name$.getKey(), $name$.getValue());\n");
  }
}

void ImmutableMapFieldLiteGenerator::
GenerateParsingDoneCode(io::Printer* printer) const {
  // Nothing to do here.
}

void ImmutableMapFieldLiteGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(
      variables_,
      "for (java.util.Map.Entry<$type_parameters$> entry\n"
      "     : internalGet$capitalized_name$().getMap().entrySet()) {\n"
      "  com.google.protobuf.MapEntry$lite$<$type_parameters$>\n"
      "  $name$ = $default_entry$.newBuilderForType()\n"
      "      .setKey(entry.getKey())\n"
      "      .setValue(entry.getValue())\n"
      "      .build();\n"
      "  output.writeMessage($number$, $name$);\n"
      "}\n");
}

void ImmutableMapFieldLiteGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(
      variables_,
      "for (java.util.Map.Entry<$type_parameters$> entry\n"
      "     : internalGet$capitalized_name$().getMap().entrySet()) {\n"
      "  com.google.protobuf.MapEntry$lite$<$type_parameters$>\n"
      "  $name$ = $default_entry$.newBuilderForType()\n"
      "      .setKey(entry.getKey())\n"
      "      .setValue(entry.getValue())\n"
      "      .build();\n"
      "  size += com.google.protobuf.CodedOutputStream\n"
      "      .computeMessageSize($number$, $name$);\n"
      "}\n");
}

void ImmutableMapFieldLiteGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(
      variables_,
      "result = result && internalGet$capitalized_name$().equals(\n"
      "    other.internalGet$capitalized_name$());\n");
}

void ImmutableMapFieldLiteGenerator::
GenerateHashCode(io::Printer* printer) const {
  printer->Print(
      variables_,
      "if (!internalGet$capitalized_name$().getMap().isEmpty()) {\n"
      "  hash = (37 * hash) + $constant_name$;\n"
      "  hash = (53 * hash) + internalGet$capitalized_name$().hashCode();\n"
      "}\n");
}

string ImmutableMapFieldLiteGenerator::GetBoxedType() const {
  return name_resolver_->GetImmutableClassName(descriptor_->message_type());
}

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
