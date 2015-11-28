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
#include <algorithm>
#include <map>

#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/wire_format_lite.h>

#include <google/protobuf/compiler/csharp/csharp_doc_comment.h>
#include <google/protobuf/compiler/csharp/csharp_enum.h>
#include <google/protobuf/compiler/csharp/csharp_field_base.h>
#include <google/protobuf/compiler/csharp/csharp_helpers.h>
#include <google/protobuf/compiler/csharp/csharp_message.h>
#include <google/protobuf/compiler/csharp/csharp_names.h>

using google::protobuf::internal::scoped_ptr;

namespace google {
namespace protobuf {
namespace compiler {
namespace csharp {

bool CompareFieldNumbers(const FieldDescriptor* d1, const FieldDescriptor* d2) {
  return d1->number() < d2->number();
}

MessageGenerator::MessageGenerator(const Descriptor* descriptor)
    : SourceGeneratorBase(descriptor->file()),
      descriptor_(descriptor) {

  // sorted field names
  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_names_.push_back(descriptor_->field(i)->name());
  }
  std::sort(field_names_.begin(), field_names_.end());

  // fields by number
  for (int i = 0; i < descriptor_->field_count(); i++) {
    fields_by_number_.push_back(descriptor_->field(i));
  }
  std::sort(fields_by_number_.begin(), fields_by_number_.end(),
            CompareFieldNumbers);
}

MessageGenerator::~MessageGenerator() {
}

std::string MessageGenerator::class_name() {
  return descriptor_->name();
}

std::string MessageGenerator::full_class_name() {
  return GetClassName(descriptor_);
}

const std::vector<std::string>& MessageGenerator::field_names() {
  return field_names_;
}

const std::vector<const FieldDescriptor*>& MessageGenerator::fields_by_number() {
  return fields_by_number_;
}

void MessageGenerator::Generate(io::Printer* printer) {
  map<string, string> vars;
  vars["class_name"] = class_name();
  vars["access_level"] = class_access_level();

  WriteMessageDocComment(printer, descriptor_);
  printer->Print(
    "[global::System.Diagnostics.DebuggerNonUserCodeAttribute()]\n");
  WriteGeneratedCodeAttributes(printer);
  printer->Print(
    vars,
    "$access_level$ sealed partial class $class_name$ : pb::IMessage<$class_name$> {\n");
  printer->Indent();

  // All static fields and properties
  printer->Print(
      vars,
      "private static readonly pb::MessageParser<$class_name$> _parser = new pb::MessageParser<$class_name$>(() => new $class_name$());\n"
      "public static pb::MessageParser<$class_name$> Parser { get { return _parser; } }\n\n");

  // Access the message descriptor via the relevant file descriptor or containing message descriptor.
  if (!descriptor_->containing_type()) {
    vars["descriptor_accessor"] = GetReflectionClassName(descriptor_->file())
        + ".Descriptor.MessageTypes[" + SimpleItoa(descriptor_->index()) + "]";
  } else {
    vars["descriptor_accessor"] = GetClassName(descriptor_->containing_type())
        + ".Descriptor.NestedTypes[" + SimpleItoa(descriptor_->index()) + "]";
  }

  printer->Print(
    vars,
    "public static pbr::MessageDescriptor Descriptor {\n"
    "  get { return $descriptor_accessor$; }\n"
    "}\n"
    "\n"
    "pbr::MessageDescriptor pb::IMessage.Descriptor {\n"
    "  get { return Descriptor; }\n"
    "}\n"
    "\n");

  // Parameterless constructor and partial OnConstruction method.
  printer->Print(
    vars,
    "public $class_name$() {\n"
    "  OnConstruction();\n"
    "}\n\n"
    "partial void OnConstruction();\n\n");

  GenerateCloningCode(printer);
  GenerateFreezingCode(printer);

  // Fields/properties
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* fieldDescriptor = descriptor_->field(i);

    // Rats: we lose the debug comment here :(
    printer->Print(
      "/// <summary>Field number for the \"$field_name$\" field.</summary>\n"
      "public const int $field_constant_name$ = $index$;\n",
      "field_name", fieldDescriptor->name(),
      "field_constant_name", GetFieldConstantName(fieldDescriptor),
      "index", SimpleItoa(fieldDescriptor->number()));
    scoped_ptr<FieldGeneratorBase> generator(
        CreateFieldGeneratorInternal(fieldDescriptor));
    generator->GenerateMembers(printer);
    printer->Print("\n");
  }

  // oneof properties
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    vars["name"] = UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), false);
    vars["property_name"] = UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), true);
    vars["original_name"] = descriptor_->oneof_decl(i)->name();
    printer->Print(
      vars,
      "private object $name$_;\n"
      "/// <summary>Enum of possible cases for the \"$original_name$\" oneof.</summary>\n"
      "public enum $property_name$OneofCase {\n");
    printer->Indent();
    printer->Print("None = 0,\n");
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      printer->Print("$field_property_name$ = $index$,\n",
                     "field_property_name", GetPropertyName(field),
                     "index", SimpleItoa(field->number()));
    }
    printer->Outdent();
    printer->Print("}\n");
    // TODO: Should we put the oneof .proto comments here? It's unclear exactly where they should go.
    printer->Print(
      vars,
      "private $property_name$OneofCase $name$Case_ = $property_name$OneofCase.None;\n"
      "public $property_name$OneofCase $property_name$Case {\n"
      "  get { return $name$Case_; }\n"
      "}\n\n"
      "public void Clear$property_name$() {\n"
      "  $name$Case_ = $property_name$OneofCase.None;\n"
      "  $name$_ = null;\n"
      "}\n\n");
  }

  // Standard methods
  GenerateFrameworkMethods(printer);
  GenerateMessageSerializationMethods(printer);
  GenerateMergingMethods(printer);

  // Nested messages and enums
  if (HasNestedGeneratedTypes()) {
    printer->Print(
      vars,
      "#region Nested types\n"
      "/// <summary>Container for nested types declared in the $class_name$ message type.</summary>\n"
      "[global::System.Diagnostics.DebuggerNonUserCodeAttribute()]\n");
    WriteGeneratedCodeAttributes(printer);
    printer->Print("public static partial class Types {\n");
    printer->Indent();
    for (int i = 0; i < descriptor_->enum_type_count(); i++) {
      EnumGenerator enumGenerator(descriptor_->enum_type(i));
      enumGenerator.Generate(printer);
    }
    for (int i = 0; i < descriptor_->nested_type_count(); i++) {
      // Don't generate nested types for maps...
      if (!IsMapEntryMessage(descriptor_->nested_type(i))) {
        MessageGenerator messageGenerator(descriptor_->nested_type(i));
        messageGenerator.Generate(printer);
      }
    }
    printer->Outdent();
    printer->Print("}\n"
                   "#endregion\n"
                   "\n");
  }

  printer->Outdent();
  printer->Print("}\n");
  printer->Print("\n");
}

// Helper to work out whether we need to generate a class to hold nested types/enums.
// Only tricky because we don't want to generate map entry types.
bool MessageGenerator::HasNestedGeneratedTypes()
{
  if (descriptor_->enum_type_count() > 0) {
    return true;
  }
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    if (!IsMapEntryMessage(descriptor_->nested_type(i))) {
      return true;
    }
  }
  return false;
}

void MessageGenerator::GenerateCloningCode(io::Printer* printer) {
  map<string, string> vars;
  vars["class_name"] = class_name();
    printer->Print(
    vars,
    "public $class_name$($class_name$ other) : this() {\n");
  printer->Indent();
  // Clone non-oneof fields first
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {
      scoped_ptr<FieldGeneratorBase> generator(
        CreateFieldGeneratorInternal(descriptor_->field(i)));
      generator->GenerateCloningCode(printer);
    }
  }
  // Clone just the right field for each oneof
  for (int i = 0; i < descriptor_->oneof_decl_count(); ++i) {
    vars["name"] = UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), false);
    vars["property_name"] = UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), true);
    printer->Print(vars, "switch (other.$property_name$Case) {\n");
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      scoped_ptr<FieldGeneratorBase> generator(CreateFieldGeneratorInternal(field));
      vars["field_property_name"] = GetPropertyName(field);
      printer->Print(
          vars,
          "case $property_name$OneofCase.$field_property_name$:\n");
      printer->Indent();
      generator->GenerateCloningCode(printer);
      printer->Print("break;\n");
      printer->Outdent();
    }
    printer->Outdent();
    printer->Print("}\n\n");
  }

  printer->Outdent();
  printer->Print("}\n\n");

  printer->Print(
    vars,
    "public $class_name$ Clone() {\n"
    "  return new $class_name$(this);\n"
    "}\n\n");
}

void MessageGenerator::GenerateFreezingCode(io::Printer* printer) {
}

void MessageGenerator::GenerateFrameworkMethods(io::Printer* printer) {
    map<string, string> vars;
    vars["class_name"] = class_name();

    // Equality
    printer->Print(
        vars,
        "public override bool Equals(object other) {\n"
        "  return Equals(other as $class_name$);\n"
        "}\n\n"
        "public bool Equals($class_name$ other) {\n"
        "  if (ReferenceEquals(other, null)) {\n"
        "    return false;\n"
        "  }\n"
        "  if (ReferenceEquals(other, this)) {\n"
        "    return true;\n"
        "  }\n");
    printer->Indent();
    for (int i = 0; i < descriptor_->field_count(); i++) {
        scoped_ptr<FieldGeneratorBase> generator(
            CreateFieldGeneratorInternal(descriptor_->field(i)));
        generator->WriteEquals(printer);
    }
    for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
        printer->Print("if ($property_name$Case != other.$property_name$Case) return false;\n",
            "property_name", UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), true));
    }
    printer->Outdent();
    printer->Print(
        "  return true;\n"
        "}\n\n");

    // GetHashCode
    // Start with a non-zero value to easily distinguish between null and "empty" messages.
    printer->Print(
        "public override int GetHashCode() {\n"
        "  int hash = 1;\n");
    printer->Indent();
    for (int i = 0; i < descriptor_->field_count(); i++) {
        scoped_ptr<FieldGeneratorBase> generator(
            CreateFieldGeneratorInternal(descriptor_->field(i)));
        generator->WriteHash(printer);
    }
    for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
        printer->Print("hash ^= (int) $name$Case_;\n",
            "name", UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), false));
    }
    printer->Print("return hash;\n");
    printer->Outdent();
    printer->Print("}\n\n");

    printer->Print(
        "public override string ToString() {\n"
        "  return pb::JsonFormatter.Default.Format(this);\n"
        "}\n\n");
}

void MessageGenerator::GenerateMessageSerializationMethods(io::Printer* printer) {
  printer->Print(
      "public void WriteTo(pb::CodedOutputStream output) {\n");
  printer->Indent();

  // Serialize all the fields
  for (int i = 0; i < fields_by_number().size(); i++) {
    scoped_ptr<FieldGeneratorBase> generator(
      CreateFieldGeneratorInternal(fields_by_number()[i]));
    generator->GenerateSerializationCode(printer);
  }

  // TODO(jonskeet): Memoize size of frozen messages?
  printer->Outdent();
  printer->Print(
    "}\n"
    "\n"
    "public int CalculateSize() {\n");
  printer->Indent();
  printer->Print("int size = 0;\n");
  for (int i = 0; i < descriptor_->field_count(); i++) {
    scoped_ptr<FieldGeneratorBase> generator(
        CreateFieldGeneratorInternal(descriptor_->field(i)));
    generator->GenerateSerializedSizeCode(printer);
  }
  printer->Print("return size;\n");
  printer->Outdent();
  printer->Print("}\n\n");
}

void MessageGenerator::GenerateMergingMethods(io::Printer* printer) {
  // Note:  These are separate from GenerateMessageSerializationMethods()
  //   because they need to be generated even for messages that are optimized
  //   for code size.
  map<string, string> vars;
  vars["class_name"] = class_name();

  printer->Print(
    vars,
    "public void MergeFrom($class_name$ other) {\n");
  printer->Indent();
  printer->Print(
    "if (other == null) {\n"
    "  return;\n"
    "}\n");
  // Merge non-oneof fields
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {      
      scoped_ptr<FieldGeneratorBase> generator(
          CreateFieldGeneratorInternal(descriptor_->field(i)));
      generator->GenerateMergingCode(printer);
    }
  }
  // Merge oneof fields
  for (int i = 0; i < descriptor_->oneof_decl_count(); ++i) {
    vars["name"] = UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), false);
    vars["property_name"] = UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), true);
    printer->Print(vars, "switch (other.$property_name$Case) {\n");
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      vars["field_property_name"] = GetPropertyName(field);
      printer->Print(
        vars,
        "case $property_name$OneofCase.$field_property_name$:\n"
        "  $field_property_name$ = other.$field_property_name$;\n"
        "  break;\n");
    }
    printer->Outdent();
    printer->Print("}\n\n");
  }
  printer->Outdent();
  printer->Print("}\n\n");
  printer->Print("public void MergeFrom(pb::CodedInputStream input) {\n");
  printer->Indent();
  printer->Print(
    "uint tag;\n"
    "while ((tag = input.ReadTag()) != 0) {\n"
    "  switch(tag) {\n");
  printer->Indent();
  printer->Indent();
  printer->Print(
    "default:\n"
    "  input.SkipLastField();\n" // We're not storing the data, but we still need to consume it.
    "  break;\n");
  for (int i = 0; i < fields_by_number().size(); i++) {
    const FieldDescriptor* field = fields_by_number()[i];
    internal::WireFormatLite::WireType wt =
        internal::WireFormat::WireTypeForFieldType(field->type());
    uint32 tag = internal::WireFormatLite::MakeTag(field->number(), wt);
    // Handle both packed and unpacked repeated fields with the same Read*Array call;
    // the two generated cases are the packed and unpacked tags.
    // TODO(jonskeet): Check that is_packable is equivalent to is_repeated && wt in { VARINT, FIXED32, FIXED64 }.
    // It looks like it is...
    if (field->is_packable()) {
      printer->Print(
        "case $packed_tag$:\n",
        "packed_tag",
        SimpleItoa(
            internal::WireFormatLite::MakeTag(
                field->number(),
                internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED)));
    }

    printer->Print("case $tag$: {\n", "tag", SimpleItoa(tag));
    printer->Indent();
    scoped_ptr<FieldGeneratorBase> generator(
        CreateFieldGeneratorInternal(field));
    generator->GenerateParsingCode(printer);
    printer->Print("break;\n");
    printer->Outdent();
    printer->Print("}\n");
  }
  printer->Outdent();
  printer->Print("}\n"); // switch
  printer->Outdent();
  printer->Print("}\n"); // while
  printer->Outdent();
  printer->Print("}\n\n"); // method
}

int MessageGenerator::GetFieldOrdinal(const FieldDescriptor* descriptor) {
  for (int i = 0; i < field_names().size(); i++) {
    if (field_names()[i] == descriptor->name()) {
      return i;
    }
  }
  GOOGLE_LOG(DFATAL)<< "Could not find ordinal for field " << descriptor->name();
  return -1;
}

FieldGeneratorBase* MessageGenerator::CreateFieldGeneratorInternal(
    const FieldDescriptor* descriptor) {
  return CreateFieldGenerator(descriptor, GetFieldOrdinal(descriptor));
}

}  // namespace csharp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
