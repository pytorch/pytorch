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

#include <google/protobuf/compiler/java/java_message_lite.h>

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
#include <google/protobuf/compiler/java/java_enum_lite.h>
#include <google/protobuf/compiler/java/java_extension.h>
#include <google/protobuf/compiler/java/java_generator_factory.h>
#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/compiler/java/java_message_builder.h>
#include <google/protobuf/compiler/java/java_message_builder_lite.h>
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

using internal::WireFormat;
using internal::WireFormatLite;

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

// ===================================================================
ImmutableMessageLiteGenerator::ImmutableMessageLiteGenerator(
    const Descriptor* descriptor, Context* context)
  : MessageGenerator(descriptor), context_(context),
    name_resolver_(context->GetNameResolver()),
    field_generators_(descriptor, context_) {
  GOOGLE_CHECK_EQ(
      FileOptions::LITE_RUNTIME, descriptor->file()->options().optimize_for());
}

ImmutableMessageLiteGenerator::~ImmutableMessageLiteGenerator() {}

void ImmutableMessageLiteGenerator::GenerateStaticVariables(
    io::Printer* printer) {
  // Generate static members for all nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // TODO(kenton):  Reuse MessageGenerator objects?
    ImmutableMessageLiteGenerator(descriptor_->nested_type(i), context_)
      .GenerateStaticVariables(printer);
  }
}

int ImmutableMessageLiteGenerator::GenerateStaticVariableInitializers(
    io::Printer* printer) {
  int bytecode_estimate = 0;
  // Generate static member initializers for all nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // TODO(kenton):  Reuse MessageGenerator objects?
    bytecode_estimate +=
        ImmutableMessageLiteGenerator(descriptor_->nested_type(i), context_)
            .GenerateStaticVariableInitializers(printer);
  }
  return bytecode_estimate;
}

// ===================================================================

void ImmutableMessageLiteGenerator::GenerateInterface(io::Printer* printer) {
  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "public interface $classname$OrBuilder extends \n"
      "    $extra_interfaces$\n"
      "     com.google.protobuf.GeneratedMessageLite.\n"
      "          ExtendableMessageOrBuilder<\n"
      "              $classname$, $classname$.Builder> {\n",
      "extra_interfaces", ExtraMessageOrBuilderInterfaces(descriptor_),
      "classname", descriptor_->name());
  } else {
    printer->Print(
      "public interface $classname$OrBuilder extends\n"
      "    $extra_interfaces$\n"
      "    com.google.protobuf.MessageLiteOrBuilder {\n",
      "extra_interfaces", ExtraMessageOrBuilderInterfaces(descriptor_),
      "classname", descriptor_->name());
  }

  printer->Indent();
    for (int i = 0; i < descriptor_->field_count(); i++) {
      printer->Print("\n");
      field_generators_.get(descriptor_->field(i))
                       .GenerateInterfaceMembers(printer);
    }
    for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
      printer->Print(
          "\n"
          "public $classname$.$oneof_capitalized_name$Case "
          "get$oneof_capitalized_name$Case();\n",
          "oneof_capitalized_name",
          context_->GetOneofGeneratorInfo(
              descriptor_->oneof_decl(i))->capitalized_name,
          "classname",
          context_->GetNameResolver()->GetImmutableClassName(descriptor_));
    }
  printer->Outdent();

  printer->Print("}\n");
}

// ===================================================================

void ImmutableMessageLiteGenerator::Generate(io::Printer* printer) {
  bool is_own_file =
    descriptor_->containing_type() == NULL &&
    MultipleJavaFiles(descriptor_->file(), /* immutable = */ true);

  map<string, string> variables;
  variables["static"] = is_own_file ? " " : " static ";
  variables["classname"] = descriptor_->name();
  variables["extra_interfaces"] = ExtraMessageInterfaces(descriptor_);

  WriteMessageDocComment(printer, descriptor_);

  // The builder_type stores the super type name of the nested Builder class.
  string builder_type;
  if (descriptor_->extension_range_count() > 0) {
    printer->Print(variables,
      "public $static$final class $classname$ extends\n"
      "    com.google.protobuf.GeneratedMessageLite.ExtendableMessage<\n"
      "      $classname$, $classname$.Builder> implements\n"
      "    $extra_interfaces$\n"
      "    $classname$OrBuilder {\n");
    builder_type = strings::Substitute(
        "com.google.protobuf.GeneratedMessageLite.ExtendableBuilder<$0, ?>",
        name_resolver_->GetImmutableClassName(descriptor_));
  } else {
    printer->Print(variables,
        "public $static$final class $classname$ extends\n"
        "    com.google.protobuf.GeneratedMessageLite<\n"
        "        $classname$, $classname$.Builder> implements\n"
        "    $extra_interfaces$\n"
        "    $classname$OrBuilder {\n");

    builder_type = "com.google.protobuf.GeneratedMessageLite.Builder";
  }
  printer->Indent();

  GenerateParsingConstructor(printer);

  // Nested types
  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    EnumLiteGenerator(descriptor_->enum_type(i), true, context_)
        .Generate(printer);
  }

  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // Don't generate Java classes for map entry messages.
    if (IsMapEntry(descriptor_->nested_type(i))) continue;
    ImmutableMessageLiteGenerator messageGenerator(
        descriptor_->nested_type(i), context_);
    messageGenerator.GenerateInterface(printer);
    messageGenerator.Generate(printer);
  }

  if (GenerateHasBits(descriptor_)) {
    // Integers for bit fields.
    int totalBits = 0;
    for (int i = 0; i < descriptor_->field_count(); i++) {
      totalBits += field_generators_.get(descriptor_->field(i))
          .GetNumBitsForMessage();
    }
    int totalInts = (totalBits + 31) / 32;
    for (int i = 0; i < totalInts; i++) {
      printer->Print("private int $bit_field_name$;\n",
        "bit_field_name", GetBitFieldName(i));
    }
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
    // OneofCase enum
    printer->Print(vars,
      "public enum $oneof_capitalized_name$Case\n"
      "    implements com.google.protobuf.Internal.EnumLite {\n");
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      printer->Print(
        "$field_name$($field_number$),\n",
        "field_name",
        ToUpper(field->name()),
        "field_number",
        SimpleItoa(field->number()));
    }
    printer->Print(
      "$cap_oneof_name$_NOT_SET(0);\n",
      "cap_oneof_name",
      ToUpper(vars["oneof_name"]));
    printer->Print(vars,
      "private int value = 0;\n"
      "private $oneof_capitalized_name$Case(int value) {\n"
      "  this.value = value;\n"
      "}\n");
    printer->Print(vars,
      "public static $oneof_capitalized_name$Case valueOf(int value) {\n"
      "  switch (value) {\n");
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      printer->Print(
        "    case $field_number$: return $field_name$;\n",
        "field_number",
        SimpleItoa(field->number()),
        "field_name",
        ToUpper(field->name()));
    }
    printer->Print(
      "    case 0: return $cap_oneof_name$_NOT_SET;\n"
      "    default: throw new java.lang.IllegalArgumentException(\n"
      "      \"Value is undefined for this oneof enum.\");\n"
      "  }\n"
      "}\n"
      "public int getNumber() {\n"
      "  return this.value;\n"
      "}\n",
      "cap_oneof_name", ToUpper(vars["oneof_name"]));
    printer->Outdent();
    printer->Print("};\n\n");
    // oneofCase()
    printer->Print(vars,
      "public $oneof_capitalized_name$Case\n"
      "get$oneof_capitalized_name$Case() {\n"
      "  return $oneof_capitalized_name$Case.valueOf(\n"
      "      $oneof_name$Case_);\n"
      "}\n"
      "\n"
      "private void clear$oneof_capitalized_name$() {\n"
      "  $oneof_name$Case_ = 0;\n"
      "  $oneof_name$_ = null;\n"
      "}\n"
      "\n");
  }

  // Fields
  for (int i = 0; i < descriptor_->field_count(); i++) {
    printer->Print("public static final int $constant_name$ = $number$;\n",
      "constant_name", FieldConstantName(descriptor_->field(i)),
      "number", SimpleItoa(descriptor_->field(i)->number()));
    field_generators_.get(descriptor_->field(i)).GenerateMembers(printer);
    printer->Print("\n");
  }

  GenerateMessageSerializationMethods(printer);

  if (HasEqualsAndHashCode(descriptor_)) {
    GenerateEqualsAndHashCode(printer);
  }


  GenerateParseFromMethods(printer);
  GenerateBuilder(printer);

  if (HasRequiredFields(descriptor_)) {
    // Memoizes whether the protocol buffer is fully initialized (has all
    // required fields). -1 means not yet computed. 0 means false and 1 means
    // true.
    printer->Print(
      "private byte memoizedIsInitialized = -1;\n");
  }

  printer->Print(
    "protected final Object dynamicMethod(\n"
    "    com.google.protobuf.GeneratedMessageLite.MethodToInvoke method,\n"
    "    Object arg0, Object arg1) {\n"
    "  switch (method) {\n"
    "    case PARSE_PARTIAL_FROM: {\n"
    "      return new $classname$("
    "          (com.google.protobuf.CodedInputStream) arg0,\n"
    "          (com.google.protobuf.ExtensionRegistryLite) arg1);\n"
    "    }\n"
    "    case NEW_INSTANCE: {\n"
    "      return new $classname$(\n"
    "          com.google.protobuf.Internal.EMPTY_CODED_INPUT_STREAM,\n"
    "          com.google.protobuf.ExtensionRegistryLite\n"
    "              .getEmptyRegistry());\n"
    "    }\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Indent();
  printer->Indent();

  printer->Print(
    "case IS_INITIALIZED: {\n");
  printer->Indent();
  GenerateDynamicMethodIsInitialized(printer);
  printer->Outdent();

  printer->Print(
    "}\n"
    "case MAKE_IMMUTABLE: {\n");

  printer->Indent();
  GenerateDynamicMethodMakeImmutable(printer);
  printer->Outdent();

  printer->Print(
    "}\n"
    "case NEW_BUILDER: {\n");

  printer->Indent();
  GenerateDynamicMethodNewBuilder(printer);
  printer->Outdent();

  printer->Print(
    "}\n"
    "case MERGE_FROM: {\n");

  printer->Indent();
  GenerateDynamicMethodMergeFrom(printer);
  printer->Outdent();

  printer->Print(
    "}\n"
    "case GET_DEFAULT_INSTANCE: {\n"
    "  return DEFAULT_INSTANCE;\n"
    "}\n"
    "case GET_PARSER: {\n"
    // Generally one would use the lazy initialization holder pattern for
    // manipulating static fields but that has exceptional cost on Android as
    // it will generate an extra class for every message. Instead, use the
    // double-check locking pattern which works just as well.
    "  if (PARSER == null) {"
    "    synchronized ($classname$.class) {\n"
    "      if (PARSER == null) {\n"
    "        PARSER = new DefaultInstanceBasedParser(DEFAULT_INSTANCE);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return PARSER;\n"
    "}\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Outdent();
  printer->Outdent();

  printer->Print(
    "  }\n"
    "  throw new UnsupportedOperationException();\n"
    "}\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(class_scope:$full_name$)\n",
    "full_name", descriptor_->full_name());


  // Carefully initialize the default instance in such a way that it doesn't
  // conflict with other initialization.
  printer->Print(
    "private static final $classname$ DEFAULT_INSTANCE;\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Print(
    "static {\n"
    "  DEFAULT_INSTANCE = new $classname$(\n"
    "      com.google.protobuf.Internal\n"
    "          .EMPTY_CODED_INPUT_STREAM,\n"
    "      com.google.protobuf.ExtensionRegistryLite\n"
    "          .getEmptyRegistry());\n"
    "}\n"
    "\n",
    "classname", descriptor_->name());
  printer->Print(
      "public static $classname$ getDefaultInstance() {\n"
      "  return DEFAULT_INSTANCE;\n"
      "}\n"
      "\n",
      "classname", name_resolver_->GetImmutableClassName(descriptor_));

  GenerateParser(printer);

  // Extensions must be declared after the DEFAULT_INSTANCE is initialized
  // because the DEFAULT_INSTANCE is used by the extension to lazily retrieve
  // the outer class's FileDescriptor.
  for (int i = 0; i < descriptor_->extension_count(); i++) {
    ImmutableExtensionGenerator(descriptor_->extension(i), context_)
        .Generate(printer);
  }

  printer->Outdent();
  printer->Print("}\n\n");
}

// ===================================================================

void ImmutableMessageLiteGenerator::
GenerateMessageSerializationMethods(io::Printer* printer) {
  google::protobuf::scoped_array<const FieldDescriptor * > sorted_fields(
      SortFieldsByNumber(descriptor_));

  vector<const Descriptor::ExtensionRange*> sorted_extensions;
  for (int i = 0; i < descriptor_->extension_range_count(); ++i) {
    sorted_extensions.push_back(descriptor_->extension_range(i));
  }
  std::sort(sorted_extensions.begin(), sorted_extensions.end(),
            ExtensionRangeOrdering());

  printer->Print(
    "public void writeTo(com.google.protobuf.CodedOutputStream output)\n"
    "                    throws java.io.IOException {\n");
  printer->Indent();
  if (HasPackedFields(descriptor_)) {
    // writeTo(CodedOutputStream output) might be invoked without
    // getSerializedSize() ever being called, but we need the memoized
    // sizes in case this message has packed fields. Rather than emit checks for
    // each packed field, just call getSerializedSize() up front.
    // In most cases, getSerializedSize() will have already been called anyway
    // by one of the wrapper writeTo() methods, making this call cheap.
    printer->Print(
      "getSerializedSize();\n");
  }

  if (descriptor_->extension_range_count() > 0) {
    if (descriptor_->options().message_set_wire_format()) {
      printer->Print(
        "com.google.protobuf.GeneratedMessageLite\n"
        "  .ExtendableMessage<$classname$, $classname$.Builder>\n"
        "    .ExtensionWriter extensionWriter =\n"
        "      newMessageSetExtensionWriter();\n",
        "classname", name_resolver_->GetImmutableClassName(descriptor_));
    } else {
      printer->Print(
        "com.google.protobuf.GeneratedMessageLite\n"
        "  .ExtendableMessage<$classname$, $classname$.Builder>\n"
        "    .ExtensionWriter extensionWriter =\n"
        "      newExtensionWriter();\n",
        "classname", name_resolver_->GetImmutableClassName(descriptor_));
    }
  }

  // Merge the fields and the extension ranges, both sorted by field number.
  for (int i = 0, j = 0;
       i < descriptor_->field_count() || j < sorted_extensions.size();
       ) {
    if (i == descriptor_->field_count()) {
      GenerateSerializeOneExtensionRange(printer, sorted_extensions[j++]);
    } else if (j == sorted_extensions.size()) {
      GenerateSerializeOneField(printer, sorted_fields[i++]);
    } else if (sorted_fields[i]->number() < sorted_extensions[j]->start) {
      GenerateSerializeOneField(printer, sorted_fields[i++]);
    } else {
      GenerateSerializeOneExtensionRange(printer, sorted_extensions[j++]);
    }
  }

  if (PreserveUnknownFields(descriptor_)) {
    printer->Print(
      "unknownFields.writeTo(output);\n");
  }

  printer->Outdent();
  printer->Print(
    "}\n"
    "\n"
    "public int getSerializedSize() {\n"
    "  int size = memoizedSerializedSize;\n"
    "  if (size != -1) return size;\n"
    "\n"
    "  size = 0;\n");
  printer->Indent();

  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(sorted_fields[i]).GenerateSerializedSizeCode(printer);
  }

  if (descriptor_->extension_range_count() > 0) {
    if (descriptor_->options().message_set_wire_format()) {
      printer->Print(
        "size += extensionsSerializedSizeAsMessageSet();\n");
    } else {
      printer->Print(
        "size += extensionsSerializedSize();\n");
    }
  }

  if (PreserveUnknownFields(descriptor_)) {
    printer->Print(
      "size += unknownFields.getSerializedSize();\n");
  }

  printer->Outdent();
  printer->Print(
    "  memoizedSerializedSize = size;\n"
    "  return size;\n"
    "}\n"
    "\n");

  printer->Print(
    "private static final long serialVersionUID = 0L;\n");
}

void ImmutableMessageLiteGenerator::
GenerateParseFromMethods(io::Printer* printer) {
  // Note:  These are separate from GenerateMessageSerializationMethods()
  //   because they need to be generated even for messages that are optimized
  //   for code size.
  printer->Print(
    "public static $classname$ parseFrom(\n"
    "    com.google.protobuf.ByteString data)\n"
    "    throws com.google.protobuf.InvalidProtocolBufferException {\n"
    "  return parser().parseFrom(data);\n"
    "}\n"
    "public static $classname$ parseFrom(\n"
    "    com.google.protobuf.ByteString data,\n"
    "    com.google.protobuf.ExtensionRegistryLite extensionRegistry)\n"
    "    throws com.google.protobuf.InvalidProtocolBufferException {\n"
    "  return parser().parseFrom(data, extensionRegistry);\n"
    "}\n"
    "public static $classname$ parseFrom(byte[] data)\n"
    "    throws com.google.protobuf.InvalidProtocolBufferException {\n"
    "  return parser().parseFrom(data);\n"
    "}\n"
    "public static $classname$ parseFrom(\n"
    "    byte[] data,\n"
    "    com.google.protobuf.ExtensionRegistryLite extensionRegistry)\n"
    "    throws com.google.protobuf.InvalidProtocolBufferException {\n"
    "  return parser().parseFrom(data, extensionRegistry);\n"
    "}\n"
    "public static $classname$ parseFrom(java.io.InputStream input)\n"
    "    throws java.io.IOException {\n"
    "  return parser().parseFrom(input);\n"
    "}\n"
    "public static $classname$ parseFrom(\n"
    "    java.io.InputStream input,\n"
    "    com.google.protobuf.ExtensionRegistryLite extensionRegistry)\n"
    "    throws java.io.IOException {\n"
    "  return parser().parseFrom(input, extensionRegistry);\n"
    "}\n"
    "public static $classname$ parseDelimitedFrom(java.io.InputStream input)\n"
    "    throws java.io.IOException {\n"
    "  return parser().parseDelimitedFrom(input);\n"
    "}\n"
    "public static $classname$ parseDelimitedFrom(\n"
    "    java.io.InputStream input,\n"
    "    com.google.protobuf.ExtensionRegistryLite extensionRegistry)\n"
    "    throws java.io.IOException {\n"
    "  return parser().parseDelimitedFrom(input, extensionRegistry);\n"
    "}\n"
    "public static $classname$ parseFrom(\n"
    "    com.google.protobuf.CodedInputStream input)\n"
    "    throws java.io.IOException {\n"
    "  return parser().parseFrom(input);\n"
    "}\n"
    "public static $classname$ parseFrom(\n"
    "    com.google.protobuf.CodedInputStream input,\n"
    "    com.google.protobuf.ExtensionRegistryLite extensionRegistry)\n"
    "    throws java.io.IOException {\n"
    "  return parser().parseFrom(input, extensionRegistry);\n"
    "}\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));
}

void ImmutableMessageLiteGenerator::GenerateSerializeOneField(
    io::Printer* printer, const FieldDescriptor* field) {
  field_generators_.get(field).GenerateSerializationCode(printer);
}

void ImmutableMessageLiteGenerator::GenerateSerializeOneExtensionRange(
    io::Printer* printer, const Descriptor::ExtensionRange* range) {
  printer->Print(
    "extensionWriter.writeUntil($end$, output);\n",
    "end", SimpleItoa(range->end));
}

// ===================================================================

void ImmutableMessageLiteGenerator::GenerateBuilder(io::Printer* printer) {
  printer->Print(
    "public static Builder newBuilder() {\n"
    "  return DEFAULT_INSTANCE.toBuilder();\n"
    "}\n"
    "public static Builder newBuilder($classname$ prototype) {\n"
    "  return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);\n"
    "}\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  MessageBuilderLiteGenerator builderGenerator(descriptor_, context_);
  builderGenerator.Generate(printer);
}

// ===================================================================

void ImmutableMessageLiteGenerator::GenerateDynamicMethodIsInitialized(
    io::Printer* printer) {
  // Returns null for false, DEFAULT_INSTANCE for true.
  if (!HasRequiredFields(descriptor_)) {
    printer->Print("return DEFAULT_INSTANCE;\n");
    return;
  }

  // Don't directly compare to -1 to avoid an Android x86 JIT bug.
  printer->Print(
    "byte isInitialized = memoizedIsInitialized;\n"
    "if (isInitialized == 1) return DEFAULT_INSTANCE;\n"
    "if (isInitialized == 0) return null;\n"
    "\n"
    "boolean shouldMemoize = ((Boolean) arg0).booleanValue();\n");

  // Check that all required fields in this message are set.
  // TODO(kenton):  We can optimize this when we switch to putting all the
  //   "has" fields into a single bitfield.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    const FieldGeneratorInfo* info = context_->GetFieldGeneratorInfo(field);

    if (field->is_required()) {
      printer->Print(
        "if (!has$name$()) {\n"
        "  if (shouldMemoize) {\n"
        "    memoizedIsInitialized = 0;\n"
        "  }\n"
        "  return null;\n"
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
            "  if (shouldMemoize) {\n"
            "    memoizedIsInitialized = 0;\n"
            "  }\n"
            "  return null;\n"
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
            "    if (shouldMemoize) {\n"
            "      memoizedIsInitialized = 0;\n"
            "    }\n"
            "    return null;\n"
            "  }\n"
            "}\n",
            "name", info->capitalized_name);
          break;
        case FieldDescriptor::LABEL_REPEATED:
          if (IsMapEntry(field->message_type())) {
            printer->Print(
              "for ($type$ item : get$name$().values()) {\n"
              "  if (!item.isInitialized()) {\n"
              "    if (shouldMemoize) {\n"
              "      memoizedIsInitialized = 0;\n"
              "    }\n"
              "    return null;\n"
              "  }\n"
              "}\n",
              "type", MapValueImmutableClassdName(field->message_type(),
                                                  name_resolver_),
              "name", info->capitalized_name);
          } else {
            printer->Print(
              "for (int i = 0; i < get$name$Count(); i++) {\n"
              "  if (!get$name$(i).isInitialized()) {\n"
              "    if (shouldMemoize) {\n"
              "      memoizedIsInitialized = 0;\n"
              "    }\n"
              "    return null;\n"
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
      "  if (shouldMemoize) {\n"
      "    memoizedIsInitialized = 0;\n"
      "  }\n"
      "  return null;\n"
      "}\n");
  }

  printer->Print(
    "if (shouldMemoize) memoizedIsInitialized = 1;\n");

  printer->Print(
    "return DEFAULT_INSTANCE;\n"
    "\n");
}

// ===================================================================

void ImmutableMessageLiteGenerator::GenerateDynamicMethodMakeImmutable(
    io::Printer* printer) {
  // Output generation code for each field.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(descriptor_->field(i))
        .GenerateDynamicMethodMakeImmutableCode(printer);
  }
  printer->Print(
    "return null;\n");
}

// ===================================================================

void ImmutableMessageLiteGenerator::GenerateDynamicMethodNewBuilder(
    io::Printer* printer) {
  printer->Print(
    "return new Builder();\n");
}

// ===================================================================

void ImmutableMessageLiteGenerator::GenerateDynamicMethodMergeFrom(
    io::Printer* printer) {
  printer->Print(
    // Optimization:  If other is the default instance, we know none of its
    //   fields are set so we can skip the merge.
    "if (arg0 == $classname$.getDefaultInstance()) return this;\n"
    "$classname$ other = ($classname$) arg0;\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

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

  // if message type has extensions
  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "this.mergeExtensionFields(other);\n");
  }

  if (PreserveUnknownFields(descriptor_)) {
    printer->Print(
      "this.mergeUnknownFields(other.unknownFields);\n");
  }

  printer->Print(
    "return this;\n");
}

// ===================================================================

namespace {
bool CheckHasBitsForEqualsAndHashCode(const FieldDescriptor* field) {
  if (field->is_repeated()) {
    return false;
  }
  if (SupportFieldPresence(field->file())) {
    return true;
  }
  return GetJavaType(field) == JAVATYPE_MESSAGE &&
      field->containing_oneof() == NULL;
}
}  // namespace

void ImmutableMessageLiteGenerator::
GenerateEqualsAndHashCode(io::Printer* printer) {
  printer->Print(
    "@java.lang.Override\n"
    "public boolean equals(final java.lang.Object obj) {\n");
  printer->Indent();
  printer->Print(
    "if (obj == this) {\n"
    " return true;\n"
    "}\n"
    "if (!(obj instanceof $classname$)) {\n"
    "  return super.equals(obj);\n"
    "}\n"
    "$classname$ other = ($classname$) obj;\n"
    "\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  printer->Print("boolean result = true;\n");
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    const FieldGeneratorInfo* info = context_->GetFieldGeneratorInfo(field);
    bool check_has_bits = CheckHasBitsForEqualsAndHashCode(field);
    if (check_has_bits) {
      printer->Print(
        "result = result && (has$name$() == other.has$name$());\n"
        "if (has$name$()) {\n",
        "name", info->capitalized_name);
      printer->Indent();
    }
    field_generators_.get(field).GenerateEqualsCode(printer);
    if (check_has_bits) {
      printer->Outdent();
      printer->Print(
        "}\n");
    }
  }
  if (PreserveUnknownFields(descriptor_)) {
    // Always consider unknown fields for equality. This will sometimes return
    // false for non-canonical ordering when running in LITE_RUNTIME but it's
    // the best we can do.
    printer->Print(
      "result = result && unknownFields.equals(other.unknownFields);\n");
  }
  printer->Print(
    "return result;\n");
  printer->Outdent();
  printer->Print(
    "}\n"
    "\n");

  printer->Print(
    "@java.lang.Override\n"
    "public int hashCode() {\n");
  printer->Indent();
  printer->Print(
    "if (memoizedHashCode != 0) {\n");
  printer->Indent();
  printer->Print(
    "return memoizedHashCode;\n");
  printer->Outdent();
  printer->Print(
    "}\n"
    "int hash = 41;\n");

  // Include the hash of the class so that two objects with different types
  // but the same field values will probably have different hashes.
  printer->Print("hash = (19 * hash) + $classname$.class.hashCode();\n",
    "classname", name_resolver_->GetImmutableClassName(descriptor_));

  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    const FieldGeneratorInfo* info = context_->GetFieldGeneratorInfo(field);
    bool check_has_bits = CheckHasBitsForEqualsAndHashCode(field);
    if (check_has_bits) {
      printer->Print(
        "if (has$name$()) {\n",
        "name", info->capitalized_name);
      printer->Indent();
    }
    field_generators_.get(field).GenerateHashCode(printer);
    if (check_has_bits) {
      printer->Outdent();
      printer->Print("}\n");
    }
  }

  printer->Print(
    "hash = (29 * hash) + unknownFields.hashCode();\n");
  printer->Print(
    "memoizedHashCode = hash;\n"
    "return hash;\n");
  printer->Outdent();
  printer->Print(
    "}\n"
    "\n");
}

// ===================================================================

void ImmutableMessageLiteGenerator::
GenerateExtensionRegistrationCode(io::Printer* printer) {
  for (int i = 0; i < descriptor_->extension_count(); i++) {
    ImmutableExtensionGenerator(descriptor_->extension(i), context_)
      .GenerateRegistrationCode(printer);
  }

  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    ImmutableMessageLiteGenerator(descriptor_->nested_type(i), context_)
      .GenerateExtensionRegistrationCode(printer);
  }
}

// ===================================================================
void ImmutableMessageLiteGenerator::
GenerateParsingConstructor(io::Printer* printer) {
  google::protobuf::scoped_array<const FieldDescriptor * > sorted_fields(
      SortFieldsByNumber(descriptor_));

  printer->Print(
      "private $classname$(\n"
      "    com.google.protobuf.CodedInputStream input,\n"
      "    com.google.protobuf.ExtensionRegistryLite extensionRegistry) {\n",
      "classname", descriptor_->name());
  printer->Indent();

  // Initialize all fields to default.
  GenerateInitializers(printer);

  // Use builder bits to track mutable repeated fields.
  int totalBuilderBits = 0;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const ImmutableFieldLiteGenerator& field =
        field_generators_.get(descriptor_->field(i));
    totalBuilderBits += field.GetNumBitsForBuilder();
  }
  int totalBuilderInts = (totalBuilderBits + 31) / 32;
  for (int i = 0; i < totalBuilderInts; i++) {
    printer->Print("int mutable_$bit_field_name$ = 0;\n",
      "bit_field_name", GetBitFieldName(i));
  }

  printer->Print(
      "try {\n");
  printer->Indent();

  printer->Print(
    "boolean done = false;\n"
    "while (!done) {\n");
  printer->Indent();

  printer->Print(
    "int tag = input.readTag();\n"
    "switch (tag) {\n");
  printer->Indent();

  printer->Print(
    "case 0:\n"          // zero signals EOF / limit reached
    "  done = true;\n"
    "  break;\n");

  if (PreserveUnknownFields(descriptor_)) {
    if (descriptor_->extension_range_count() > 0) {
      printer->Print(
        "default: {\n"
        "  if (!parseUnknownField(getDefaultInstanceForType(),\n"
        "                         input, extensionRegistry, tag)) {\n"
        "    done = true;\n"  // it's an endgroup tag
        "  }\n"
        "  break;\n"
        "}\n");
    } else {
      printer->Print(
        "default: {\n"
        "  if (!parseUnknownField(tag, input)) {\n"
        "    done = true;\n"  // it's an endgroup tag
        "  }\n"
        "  break;\n"
        "}\n");
    }
  } else {
    printer->Print(
      "default: {\n"
      "  if (!input.skipField(tag)) {\n"
      "    done = true;\n"  // it's an endgroup tag
      "  }\n"
      "  break;\n"
      "}\n");
  }

  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = sorted_fields[i];
    uint32 tag = WireFormatLite::MakeTag(field->number(),
      WireFormat::WireTypeForFieldType(field->type()));

    printer->Print(
      "case $tag$: {\n",
      "tag", SimpleItoa(tag));
    printer->Indent();

    field_generators_.get(field).GenerateParsingCode(printer);

    printer->Outdent();
    printer->Print(
      "  break;\n"
      "}\n");

    if (field->is_packable()) {
      // To make packed = true wire compatible, we generate parsing code from a
      // packed version of this field regardless of field->options().packed().
      uint32 packed_tag = WireFormatLite::MakeTag(field->number(),
        WireFormatLite::WIRETYPE_LENGTH_DELIMITED);
      printer->Print(
        "case $tag$: {\n",
        "tag", SimpleItoa(packed_tag));
      printer->Indent();

      field_generators_.get(field).GenerateParsingCodeFromPacked(printer);

      printer->Outdent();
      printer->Print(
        "  break;\n"
        "}\n");
    }
  }

  printer->Outdent();
  printer->Outdent();
  printer->Print(
      "  }\n"     // switch (tag)
      "}\n");     // while (!done)

  printer->Outdent();
  printer->Print(
      "} catch (com.google.protobuf.InvalidProtocolBufferException e) {\n"
      "  throw new RuntimeException(e.setUnfinishedMessage(this));\n"
      "} catch (java.io.IOException e) {\n"
      "  throw new RuntimeException(\n"
      "      new com.google.protobuf.InvalidProtocolBufferException(\n"
      "          e.getMessage()).setUnfinishedMessage(this));\n"
      "} finally {\n");
  printer->Indent();

  // Make repeated field list immutable.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = sorted_fields[i];
    field_generators_.get(field).GenerateParsingDoneCode(printer);
  }

  printer->Print(
      "doneParsing();\n");

  printer->Outdent();
  printer->Outdent();
  printer->Print(
      "  }\n"     // finally
      "}\n");
}

// ===================================================================
void ImmutableMessageLiteGenerator::GenerateParser(io::Printer* printer) {
  printer->Print(
      "private static volatile com.google.protobuf.Parser<$classname$> PARSER;\n"
      "\n"
      "public static com.google.protobuf.Parser<$classname$> parser() {\n"
      "  return DEFAULT_INSTANCE.getParserForType();\n"
      "}\n",
      "classname", descriptor_->name());
}

// ===================================================================
void ImmutableMessageLiteGenerator::GenerateInitializers(io::Printer* printer) {
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {
      field_generators_.get(descriptor_->field(i))
          .GenerateInitializationCode(printer);
    }
  }
}


}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
