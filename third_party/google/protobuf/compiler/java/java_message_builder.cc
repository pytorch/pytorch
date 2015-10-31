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

// Author: dweis@google.com (Daniel Weis)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <google/protobuf/compiler/java/java_message_builder.h>

#include <algorithm>
#include <google/protobuf/stubs/hash.h>
#include <map>
#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <vector>

#include <google/protobuf/compiler/java/java_context.h>
#include <google/protobuf/compiler/java/java_doc_comment.h>
#include <google/protobuf/compiler/java/java_enum.h>
#include <google/protobuf/compiler/java/java_extension.h>
#include <google/protobuf/compiler/java/java_generator_factory.h>
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/compiler/java/java_name_resolver.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/substitute.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

namespace {
bool GenerateHasBits(const Descriptor* descriptor) {
  return SupportFieldPresence(descriptor->file()) ||
      HasRepeatedFields(descriptor);
}

string MapValueImmutableClassdName(const Descriptor* descriptor,
                                   ClassNameResolver* name_resolver) {
  const FieldDescriptor* value_field = descriptor->FindFieldByName("value");
  GOOGLE_CHECK_EQ(FieldDescriptor::TYPE_MESSAGE, value_field->type());
  return name_resolver->GetImmutableClassName(value_field->message_type());
}
}  // namespace

MessageBuilderGenerator::MessageBuilderGenerator(
    const Descriptor* descriptor, Context* context)
  : descriptor_(descriptor), context_(context),
    name_resolver_(context->GetNameResolver()),
    field_generators_(descriptor, context_) {
  GOOGLE_CHECK_NE(
      FileOptions::LITE_RUNTIME, descriptor->file()->options().optimize_for());
}

MessageBuilderGenerator::~MessageBuilderGenerator() {}

void MessageBuilderGenerator::
Generate(io::Printer* printer) {
  WriteMessageDocComment(printer, descriptor_);
  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "public static final class Builder extends\n"
      "    com.google.protobuf.GeneratedMessage.ExtendableBuilder<\n"
      "      $classname$, Builder> implements\n"
      "    $extra_interfaces$\n"
      "    $classname$OrBuilder {\n",
      "classname", name_resolver_->GetImmutableClassName(descriptor_),
      "extra_interfaces", ExtraBuilderInterfaces(descriptor_));
  } else {
    printer->Print(
      "public static final class Builder extends\n"
      "    com.google.protobuf.GeneratedMessage.Builder<Builder> implements\n"
      "    $extra_interfaces$\n"
      "    $classname$OrBuilder {\n",
      "classname", name_resolver_->GetImmutableClassName(descriptor_),
      "extra_interfaces", ExtraBuilderInterfaces(descriptor_));
  }
  printer->Indent();

  GenerateDescriptorMethods(printer);
  GenerateCommonBuilderMethods(printer);

  if (HasGeneratedMethods(descriptor_)) {
    GenerateIsInitialized(printer);
    GenerateBuilderParsingMethods(printer);
  }

  // oneof
  map<string, string> vars;
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    vars["oneof_name"] = context_->GetOneofGeneratorInfo(
        descriptor_->oneof_decl(i))->name;
    vars["oneof_capitalized_name"] = context_->GetOneofGeneratorInfo(
        descriptor_->oneof_decl(i))->capitalized_name;
    vars["oneof_index"] = SimpleItoa(descriptor_->oneof_decl(i)->index());
    // oneofCase_ and oneof_
    printer->Print(vars,
      "private int $oneof_name$Case_ = 0;\n"
      "private java.lang.Object $oneof_name$_;\n");
    // oneofCase() and clearOneof()
    printer->Print(vars,
      "public $oneof_capitalized_name$Case\n"
      "    get$oneof_capitalized_name$Case() {\n"
      "  return $oneof_capitalized_name$Case.valueOf(\n"
      "      $oneof_name$Case_);\n"
      "}\n"
      "\n"
      "public Builder clear$oneof_capitalized_name$() {\n"
      "  $oneof_name$Case_ = 0;\n"
      "  $oneof_name$_ = null;\n");
    printer->Print("  onChanged();\n");
    printer->Print(
      "  return this;\n"
      "}\n"
      "\n");
  }

  if (GenerateHasBits(descriptor_)) {
    // Integers for bit fields.
    int totalBits = 0;
    for (int i = 0; i < descriptor_->field_count(); i++) {
      totalBits += field_generators_.get(descriptor_->field(i))
          .GetNumBitsForBuilder();
    }
    int totalInts = (totalBits + 31) / 32;
    for (int i = 0; i < totalInts; i++) {
      printer->Print("private int $bit_field_name$;\n",
        "bit_field_name", GetBitFieldName(i));
    }
  }

  for (int i = 0; i < descriptor_->field_count(); i++) {
    printer->Print("\n");
    field_generators_.get(descriptor_->field(i))
                     .GenerateBuilderMembers(printer);
  }

  if (!PreserveUnknownFields(descriptor_)) {
    printer->Print(
      "public final Builder setUnknownFields(\n"
      "    final com.google.protobuf.UnknownFieldSet unknownFields) {\n"
      "  return this;\n"
      "}\n"
      "\n"
      "public final Builder mergeUnknownFields(\n"
      "    final com.google.protobuf.UnknownFieldSet unknownFields) {\n"
      "  return this;\n"
      "}\n"
      "\n");
  }

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(builder_scope:$full_name$)\n",
    "full_name", descriptor_->full_name());

  printer->Outdent();
  printer->Print("}\n");
}

// ===================================================================

void MessageBuilderGenerator::
GenerateDescriptorMethods(io::Printer* printer) {
  if (!descriptor_->options().no_standard_descriptor_accessor()) {
    printer->Print(
      "public static final com.google.protobuf.Descriptors.Descriptor\n"
      "    getDescriptor() {\n"
      "  return $fileclass$.internal_$identifier$_descriptor;\n"
      "}\n"
      "\n",
      "fileclass", name_resolver_->GetImmutableClassName(descriptor_->file()),
      "identifier", UniqueFileScopeIdentifier(descriptor_));
  }
  vector<const FieldDescriptor*> map_fields;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (GetJavaType(field) == JAVATYPE_MESSAGE &&
        IsMapEntry(field->message_type())) {
      map_fields.push_back(field);
    }
  }
  if (!map_fields.empty()) {
    printer->Print(
      "@SuppressWarnings({\"rawtypes\"})\n"
      "protected com.google.protobuf.MapField internalGetMapField(\n"
      "    int number) {\n"
      "  switch (number) {\n");
    printer->Indent();
    printer->Indent();
    for (int i = 0; i < map_fields.size(); ++i) {
      const FieldDescriptor* field = map_fields[i];
      const FieldGeneratorInfo* info = context_->GetFieldGeneratorInfo(field);
      printer->Print(
        "case $number$:\n"
        "  return internalGet$capitalized_name$();\n",
        "number", SimpleItoa(field->number()),
        "capitalized_name", info->capitalized_name);
    }
    printer->Print(
        "default:\n"
        "  throw new RuntimeException(\n"
        "      \"Invalid map field number: \" + number);\n");
    printer->Outdent();
    printer->Outdent();
    printer->Print(
        "  }\n"
        "}\n");
    printer->Print(
      "@SuppressWarnings({\"rawtypes\"})\n"
      "protected com.google.protobuf.MapField internalGetMutableMapField(\n"
      "    int number) {\n"
      "  switch (number) {\n");
    printer->Indent();
    printer->Indent();
    for (int i = 0; i < map_fields.size(); ++i) {
      const FieldDescriptor* field = map_fields[i];
      const FieldGeneratorInfo* info =
          context_->GetFieldGeneratorInfo(field);
      printer->Print(
        "case $number$:\n"
        "  return internalGetMutable$capitalized_name$();\n",
        "number", SimpleItoa(field->number()),
        "capitalized_name", info->capitalized_name);
    }
    printer->Print(
        "default:\n"
        "  throw new RuntimeException(\n"
        "      \"Invalid map field number: \" + number);\n");
    printer->Outdent();
    printer->Outdent();
    printer->Print(
        "  }\n"
        "}\n");
  }
  printer->Print(
    "protected com.google.protobuf.GeneratedMessage.FieldAccessorTable\n"
    "    internalGetFieldAccessorTable() {\n"
    "  return $fileclass$.internal_$identifier$_fieldAccessorTable\n"
    "      .ensureFieldAccessorsInitialized(\n"
    "          $classname$.class, $classname$.Builder.class);\n"
    "}\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_),
    "fileclass", name_resolver_->GetImmutableClassName(descriptor_->file()),
    "identifier", UniqueFileScopeIdentifier(descriptor_));
}

// ===================================================================

void MessageBuilderGenerator::
GenerateCommonBuilderMethods(io::Printer* printer) {
  printer->Print(
      "// Construct using $classname$.newBuilder()\n"
      "private Builder() {\n"
      "  maybeForceBuilderInitialization();\n"
      "}\n"
      "\n",
      "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Print(
    "private Builder(\n"
    "    com.google.protobuf.GeneratedMessage.BuilderParent parent) {\n"
    "  super(parent);\n"
    "  maybeForceBuilderInitialization();\n"
    "}\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Print(
    "private void maybeForceBuilderInitialization() {\n"
    "  if (com.google.protobuf.GeneratedMessage.alwaysUseFieldBuilders) {\n");

  printer->Indent();
  printer->Indent();
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {
      field_generators_.get(descriptor_->field(i))
          .GenerateFieldBuilderInitializationCode(printer);
    }
  }
  printer->Outdent();
  printer->Outdent();

  printer->Print(
    "  }\n"
    "}\n");

  printer->Print(
    "public Builder clear() {\n"
    "  super.clear();\n");

  printer->Indent();

  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {
      field_generators_.get(descriptor_->field(i))
          .GenerateBuilderClearCode(printer);
    }
  }

  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
      "$oneof_name$Case_ = 0;\n"
      "$oneof_name$_ = null;\n",
      "oneof_name", context_->GetOneofGeneratorInfo(
          descriptor_->oneof_decl(i))->name);
  }

  printer->Outdent();

  printer->Print(
    "  return this;\n"
    "}\n"
    "\n");

  printer->Print(
    "public com.google.protobuf.Descriptors.Descriptor\n"
    "    getDescriptorForType() {\n"
    "  return $fileclass$.internal_$identifier$_descriptor;\n"
    "}\n"
    "\n",
    "fileclass", name_resolver_->GetImmutableClassName(descriptor_->file()),
    "identifier", UniqueFileScopeIdentifier(descriptor_));

  // LITE runtime implements this in GeneratedMessageLite.
  printer->Print(
    "public $classname$ getDefaultInstanceForType() {\n"
    "  return $classname$.getDefaultInstance();\n"
    "}\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Print(
    "public $classname$ build() {\n"
    "  $classname$ result = buildPartial();\n"
    "  if (!result.isInitialized()) {\n"
    "    throw newUninitializedMessageException(result);\n"
    "  }\n"
    "  return result;\n"
    "}\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Print(
    "public $classname$ buildPartial() {\n"
    "  $classname$ result = new $classname$(this);\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Indent();

  int totalBuilderBits = 0;
  int totalMessageBits = 0;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const ImmutableFieldGenerator& field =
        field_generators_.get(descriptor_->field(i));
    totalBuilderBits += field.GetNumBitsForBuilder();
    totalMessageBits += field.GetNumBitsForMessage();
  }
  int totalBuilderInts = (totalBuilderBits + 31) / 32;
  int totalMessageInts = (totalMessageBits + 31) / 32;

  if (GenerateHasBits(descriptor_)) {
    // Local vars for from and to bit fields to avoid accessing the builder and
    // message over and over for these fields. Seems to provide a slight
    // perforamance improvement in micro benchmark and this is also what proto1
    // code does.
    for (int i = 0; i < totalBuilderInts; i++) {
      printer->Print("int from_$bit_field_name$ = $bit_field_name$;\n",
        "bit_field_name", GetBitFieldName(i));
    }
    for (int i = 0; i < totalMessageInts; i++) {
      printer->Print("int to_$bit_field_name$ = 0;\n",
        "bit_field_name", GetBitFieldName(i));
    }
  }

  // Output generation code for each field.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(descriptor_->field(i)).GenerateBuildingCode(printer);
  }

  if (GenerateHasBits(descriptor_)) {
    // Copy the bit field results to the generated message
    for (int i = 0; i < totalMessageInts; i++) {
      printer->Print("result.$bit_field_name$ = to_$bit_field_name$;\n",
        "bit_field_name", GetBitFieldName(i));
    }
  }

  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print("result.$oneof_name$Case_ = $oneof_name$Case_;\n",
                   "oneof_name", context_->GetOneofGeneratorInfo(
                       descriptor_->oneof_decl(i))->name);
  }

  printer->Outdent();

  printer->Print(
    "  onBuilt();\n");

  printer->Print(
    "  return result;\n"
    "}\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  // -----------------------------------------------------------------

  if (HasGeneratedMethods(descriptor_)) {
    printer->Print(
      "public Builder mergeFrom(com.google.protobuf.Message other) {\n"
      "  if (other instanceof $classname$) {\n"
      "    return mergeFrom(($classname$)other);\n"
      "  } else {\n"
      "    super.mergeFrom(other);\n"
      "    return this;\n"
      "  }\n"
      "}\n"
      "\n",
      "classname", name_resolver_->GetImmutableClassName(descriptor_));

    printer->Print(
      "public Builder mergeFrom($classname$ other) {\n"
      // Optimization:  If other is the default instance, we know none of its
      //   fields are set so we can skip the merge.
      "  if (other == $classname$.getDefaultInstance()) return this;\n",
      "classname", name_resolver_->GetImmutableClassName(descriptor_));
    printer->Indent();

    for (int i = 0; i < descriptor_->field_count(); i++) {
      if (!descriptor_->field(i)->containing_oneof()) {
        field_generators_.get(
            descriptor_->field(i)).GenerateMergingCode(printer);
      }
    }

    // Merge oneof fields.
    for (int i = 0; i < descriptor_->oneof_decl_count(); ++i) {
      printer->Print(
        "switch (other.get$oneof_capitalized_name$Case()) {\n",
        "oneof_capitalized_name",
        context_->GetOneofGeneratorInfo(
            descriptor_->oneof_decl(i))->capitalized_name);
      printer->Indent();
      for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
        const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
        printer->Print(
          "case $field_name$: {\n",
          "field_name",
          ToUpper(field->name()));
        printer->Indent();
        field_generators_.get(field).GenerateMergingCode(printer);
        printer->Print(
            "break;\n");
        printer->Outdent();
        printer->Print(
            "}\n");
      }
      printer->Print(
        "case $cap_oneof_name$_NOT_SET: {\n"
        "  break;\n"
        "}\n",
        "cap_oneof_name",
        ToUpper(context_->GetOneofGeneratorInfo(
            descriptor_->oneof_decl(i))->name));
      printer->Outdent();
      printer->Print(
          "}\n");
    }

    printer->Outdent();

    // if message type has extensions
    if (descriptor_->extension_range_count() > 0) {
      printer->Print(
        "  this.mergeExtensionFields(other);\n");
    }

    if (PreserveUnknownFields(descriptor_)) {
      printer->Print(
        "  this.mergeUnknownFields(other.unknownFields);\n");
    }

    printer->Print(
      "  onChanged();\n");

    printer->Print(
      "  return this;\n"
      "}\n"
      "\n");
  }
}

// ===================================================================

void MessageBuilderGenerator::
GenerateBuilderParsingMethods(io::Printer* printer) {
  printer->Print(
    "public Builder mergeFrom(\n"
    "    com.google.protobuf.CodedInputStream input,\n"
    "    com.google.protobuf.ExtensionRegistryLite extensionRegistry)\n"
    "    throws java.io.IOException {\n"
    "  $classname$ parsedMessage = null;\n"
    "  try {\n"
    "    parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);\n"
    "  } catch (com.google.protobuf.InvalidProtocolBufferException e) {\n"
    "    parsedMessage = ($classname$) e.getUnfinishedMessage();\n"
    "    throw e;\n"
    "  } finally {\n"
    "    if (parsedMessage != null) {\n"
    "      mergeFrom(parsedMessage);\n"
    "    }\n"
    "  }\n"
    "  return this;\n"
    "}\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));
}

// ===================================================================

void MessageBuilderGenerator::GenerateIsInitialized(
    io::Printer* printer) {
  printer->Print(
    "public final boolean isInitialized() {\n");
  printer->Indent();

  // Check that all required fields in this message are set.
  // TODO(kenton):  We can optimize this when we switch to putting all the
  //   "has" fields into a single bitfield.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    const FieldGeneratorInfo* info = context_->GetFieldGeneratorInfo(field);

    if (field->is_required()) {
      printer->Print(
        "if (!has$name$()) {\n"
        "  return false;\n"
        "}\n",
        "name", info->capitalized_name);
    }
  }

  // Now check that all embedded messages are initialized.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    const FieldGeneratorInfo* info = context_->GetFieldGeneratorInfo(field);
    if (GetJavaType(field) == JAVATYPE_MESSAGE &&
        HasRequiredFields(field->message_type())) {
      switch (field->label()) {
        case FieldDescriptor::LABEL_REQUIRED:
          printer->Print(
            "if (!get$name$().isInitialized()) {\n"
             "  return false;\n"
             "}\n",
            "type", name_resolver_->GetImmutableClassName(
                field->message_type()),
            "name", info->capitalized_name);
          break;
        case FieldDescriptor::LABEL_OPTIONAL:
          if (!SupportFieldPresence(descriptor_->file()) &&
              field->containing_oneof() != NULL) {
            const OneofDescriptor* oneof = field->containing_oneof();
            const OneofGeneratorInfo* oneof_info =
                context_->GetOneofGeneratorInfo(oneof);
            printer->Print(
              "if ($oneof_name$Case_ == $field_number$) {\n",
              "oneof_name", oneof_info->name,
              "field_number", SimpleItoa(field->number()));
          } else {
            printer->Print(
              "if (has$name$()) {\n",
              "name", info->capitalized_name);
          }
          printer->Print(
            "  if (!get$name$().isInitialized()) {\n"
            "    return false;\n"
            "  }\n"
            "}\n",
            "name", info->capitalized_name);
          break;
        case FieldDescriptor::LABEL_REPEATED:
          if (IsMapEntry(field->message_type())) {
            printer->Print(
              "for ($type$ item : get$name$().values()) {\n"
              "  if (!item.isInitialized()) {\n"
              "    return false;\n"
              "  }\n"
              "}\n",
              "type", MapValueImmutableClassdName(field->message_type(),
                                                  name_resolver_),
              "name", info->capitalized_name);
          } else {
            printer->Print(
              "for (int i = 0; i < get$name$Count(); i++) {\n"
              "  if (!get$name$(i).isInitialized()) {\n"
              "    return false;\n"
              "  }\n"
              "}\n",
              "type", name_resolver_->GetImmutableClassName(
                  field->message_type()),
              "name", info->capitalized_name);
          }
          break;
      }
    }
  }

  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "if (!extensionsAreInitialized()) {\n"
      "  return false;\n"
      "}\n");
  }

  printer->Outdent();

  printer->Print(
    "  return true;\n"
    "}\n"
    "\n");
}

// ===================================================================

}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
