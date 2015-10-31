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

#include <algorithm>
#include <google/protobuf/stubs/hash.h>
#include <google/protobuf/compiler/javanano/javanano_message.h>
#include <google/protobuf/compiler/javanano/javanano_enum.h>
#include <google/protobuf/compiler/javanano/javanano_extension.h>
#include <google/protobuf/compiler/javanano/javanano_helpers.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/descriptor.pb.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

using internal::WireFormat;
using internal::WireFormatLite;

namespace {

struct FieldOrderingByNumber {
  inline bool operator()(const FieldDescriptor* a,
                         const FieldDescriptor* b) const {
    return a->number() < b->number();
  }
};

// Sort the fields of the given Descriptor by number into a new[]'d array
// and return it.
const FieldDescriptor** SortFieldsByNumber(const Descriptor* descriptor) {
  const FieldDescriptor** fields =
    new const FieldDescriptor*[descriptor->field_count()];
  for (int i = 0; i < descriptor->field_count(); i++) {
    fields[i] = descriptor->field(i);
  }
  sort(fields, fields + descriptor->field_count(),
       FieldOrderingByNumber());
  return fields;
}

}  // namespace

// ===================================================================

MessageGenerator::MessageGenerator(const Descriptor* descriptor, const Params& params)
  : params_(params),
    descriptor_(descriptor),
    field_generators_(descriptor, params) {
}

MessageGenerator::~MessageGenerator() {}

void MessageGenerator::GenerateStaticVariables(io::Printer* printer) {
  // Generate static members for all nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // TODO(kenton):  Reuse MessageGenerator objects?
    if (IsMapEntry(descriptor_->nested_type(i))) continue;
    MessageGenerator(descriptor_->nested_type(i), params_)
      .GenerateStaticVariables(printer);
  }
}

void MessageGenerator::GenerateStaticVariableInitializers(
    io::Printer* printer) {
  // Generate static member initializers for all nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
   // TODO(kenton):  Reuse MessageGenerator objects?
    if (IsMapEntry(descriptor_->nested_type(i))) continue;
    MessageGenerator(descriptor_->nested_type(i), params_)
      .GenerateStaticVariableInitializers(printer);
  }
}

void MessageGenerator::Generate(io::Printer* printer) {
  if (!params_.store_unknown_fields() &&
      (descriptor_->extension_count() != 0 || descriptor_->extension_range_count() != 0)) {
    GOOGLE_LOG(FATAL) << "Extensions are only supported in NANO_RUNTIME if the "
        "'store_unknown_fields' generator option is 'true'\n";
  }

  const string& file_name = descriptor_->file()->name();
  bool is_own_file =
    params_.java_multiple_files(file_name)
      && descriptor_->containing_type() == NULL;

  if (is_own_file) {
    // Note: constants (from enums and fields requiring stored defaults, emitted in the loop below)
    // may have the same names as constants in the nested classes. This causes Java warnings, but
    // is not fatal, so we suppress those warnings here in the top-most class declaration.
    printer->Print(
      "\n"
      "@SuppressWarnings(\"hiding\")\n"
      "public final class $classname$ extends\n",
      "classname", descriptor_->name());
  } else {
    printer->Print(
      "\n"
      "public static final class $classname$ extends\n",
      "classname", descriptor_->name());
  }
  if (params_.store_unknown_fields() && params_.parcelable_messages()) {
    printer->Print(
      "    com.google.protobuf.nano.android.ParcelableExtendableMessageNano<$classname$>",
      "classname", descriptor_->name());
  } else if (params_.store_unknown_fields()) {
    printer->Print(
      "    com.google.protobuf.nano.ExtendableMessageNano<$classname$>",
      "classname", descriptor_->name());
  } else if (params_.parcelable_messages()) {
    printer->Print(
      "    com.google.protobuf.nano.android.ParcelableMessageNano");
  } else {
    printer->Print(
      "    com.google.protobuf.nano.MessageNano");
  }
  if (params_.generate_clone()) {
    printer->Print(" implements java.lang.Cloneable {\n");
  } else {
    printer->Print(" {\n");
  }
  printer->Indent();

  if (params_.parcelable_messages()) {
    printer->Print(
      "\n"
      "// Used by Parcelable\n"
      "@SuppressWarnings({\"unused\"})\n"
      "public static final android.os.Parcelable.Creator<$classname$> CREATOR =\n"
      "    new com.google.protobuf.nano.android.ParcelableMessageNanoCreator<\n"
      "        $classname$>($classname$.class);\n",
      "classname", descriptor_->name());
  }

  // Nested types and extensions
  for (int i = 0; i < descriptor_->extension_count(); i++) {
    ExtensionGenerator(descriptor_->extension(i), params_).Generate(printer);
  }

  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    EnumGenerator(descriptor_->enum_type(i), params_).Generate(printer);
  }

  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    if (IsMapEntry(descriptor_->nested_type(i))) continue;
    MessageGenerator(descriptor_->nested_type(i), params_).Generate(printer);
  }

  // oneof
  map<string, string> vars;
  vars["message_name"] = descriptor_->name();
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    const OneofDescriptor* oneof_desc = descriptor_->oneof_decl(i);
    vars["oneof_name"] = UnderscoresToCamelCase(oneof_desc);
    vars["oneof_capitalized_name"] =
        UnderscoresToCapitalizedCamelCase(oneof_desc);
    vars["oneof_index"] = SimpleItoa(oneof_desc->index());
    // Oneof Constants
    for (int j = 0; j < oneof_desc->field_count(); j++) {
      const FieldDescriptor* field = oneof_desc->field(j);
      vars["number"] = SimpleItoa(field->number());
      vars["cap_field_name"] = ToUpper(field->name());
      printer->Print(vars,
        "public static final int $cap_field_name$_FIELD_NUMBER = $number$;\n");
    }
    // oneofCase_ and oneof_
    printer->Print(vars,
      "private int $oneof_name$Case_ = 0;\n"
      "private java.lang.Object $oneof_name$_;\n");
    printer->Print(vars,
      "public int get$oneof_capitalized_name$Case() {\n"
      "  return this.$oneof_name$Case_;\n"
      "}\n");
    // Oneof clear
    printer->Print(vars,
      "public $message_name$ clear$oneof_capitalized_name$() {\n"
      "  this.$oneof_name$Case_ = 0;\n"
      "  this.$oneof_name$_ = null;\n"
      "  return this;\n"
      "}\n");
  }

  // Lazy initialization of otherwise static final fields can help prevent the
  // class initializer from being generated. We want to prevent it because it
  // stops ProGuard from inlining any methods in this class into call sites and
  // therefore reducing the method count. However, extensions are best kept as
  // public static final fields with initializers, so with their existence we
  // won't bother with lazy initialization.
  bool lazy_init = descriptor_->extension_count() == 0;

  // Empty array
  if (lazy_init) {
    printer->Print(
      "\n"
      "private static volatile $classname$[] _emptyArray;\n"
      "public static $classname$[] emptyArray() {\n"
      "  // Lazily initializes the empty array\n"
      "  if (_emptyArray == null) {\n"
      "    synchronized (\n"
      "        com.google.protobuf.nano.InternalNano.LAZY_INIT_LOCK) {\n"
      "      if (_emptyArray == null) {\n"
      "        _emptyArray = new $classname$[0];\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "  return _emptyArray;\n"
      "}\n",
      "classname", descriptor_->name());
  } else {
    printer->Print(
      "\n"
      "private static final $classname$[] EMPTY_ARRAY = {};\n"
      "public static $classname$[] emptyArray() {\n"
      "  return EMPTY_ARRAY;\n"
      "}\n",
      "classname", descriptor_->name());
  }

  // Integers for bit fields
  int totalInts = (field_generators_.total_bits() + 31) / 32;
  if (totalInts > 0) {
    printer->Print("\n");
    for (int i = 0; i < totalInts; i++) {
      printer->Print("private int $bit_field_name$;\n",
        "bit_field_name", GetBitFieldName(i));
    }
  }

  // Fields and maybe their default values
  for (int i = 0; i < descriptor_->field_count(); i++) {
    printer->Print("\n");
    PrintFieldComment(printer, descriptor_->field(i));
    field_generators_.get(descriptor_->field(i)).GenerateMembers(
        printer, lazy_init);
  }

  // Constructor, with lazy init code if needed
  if (lazy_init && field_generators_.saved_defaults_needed()) {
    printer->Print(
      "\n"
      "private static volatile boolean _classInitialized;\n"
      "\n"
      "public $classname$() {\n"
      "  // Lazily initializes the field defaults\n"
      "  if (!_classInitialized) {\n"
      "    synchronized (\n"
      "        com.google.protobuf.nano.InternalNano.LAZY_INIT_LOCK) {\n"
      "      if (!_classInitialized) {\n",
      "classname", descriptor_->name());
    printer->Indent();
    printer->Indent();
    printer->Indent();
    printer->Indent();
    for (int i = 0; i < descriptor_->field_count(); i++) {
      field_generators_.get(descriptor_->field(i))
          .GenerateInitSavedDefaultCode(printer);
    }
    printer->Outdent();
    printer->Outdent();
    printer->Outdent();
    printer->Outdent();
    printer->Print(
      "        _classInitialized = true;\n"
      "      }\n"
      "    }\n"
      "  }\n");
    if (params_.generate_clear()) {
      printer->Print("  clear();\n");
    }
    printer->Print("}\n");
  } else {
    printer->Print(
      "\n"
      "public $classname$() {\n",
      "classname", descriptor_->name());
    if (params_.generate_clear()) {
      printer->Print("  clear();\n");
    } else {
      printer->Indent();
      GenerateFieldInitializers(printer);
      printer->Outdent();
    }
    printer->Print("}\n");
  }

  // Other methods in this class

  GenerateClear(printer);

  if (params_.generate_clone()) {
    GenerateClone(printer);
  }

  if (params_.generate_equals()) {
    GenerateEquals(printer);
    GenerateHashCode(printer);
  }

  GenerateMessageSerializationMethods(printer);
  GenerateMergeFromMethods(printer);
  GenerateParseFromMethods(printer);

  printer->Outdent();
  printer->Print("}\n");
}

// ===================================================================

void MessageGenerator::
GenerateMessageSerializationMethods(io::Printer* printer) {
  // Rely on the parent implementations of writeTo() and getSerializedSize()
  // if there are no fields to serialize in this message.
  if (descriptor_->field_count() == 0) {
    return;
  }

  scoped_array<const FieldDescriptor*> sorted_fields(
    SortFieldsByNumber(descriptor_));

  printer->Print(
    "\n"
    "@Override\n"
    "public void writeTo(com.google.protobuf.nano.CodedOutputByteBufferNano output)\n"
    "    throws java.io.IOException {\n");
  printer->Indent();

  // Output the fields in sorted order
  for (int i = 0; i < descriptor_->field_count(); i++) {
    GenerateSerializeOneField(printer, sorted_fields[i]);
  }

  // The parent implementation will write any unknown fields if necessary.
  printer->Print(
    "super.writeTo(output);\n");

  printer->Outdent();
  printer->Print("}\n");

  // The parent implementation will get the serialized size for unknown
  // fields if necessary.
  printer->Print(
    "\n"
    "@Override\n"
    "protected int computeSerializedSize() {\n"
    "  int size = super.computeSerializedSize();\n");
  printer->Indent();

  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(sorted_fields[i]).GenerateSerializedSizeCode(printer);
  }

  printer->Outdent();
  printer->Print(
    "  return size;\n"
    "}\n");
}

void MessageGenerator::GenerateMergeFromMethods(io::Printer* printer) {
  scoped_array<const FieldDescriptor*> sorted_fields(
    SortFieldsByNumber(descriptor_));

  printer->Print(
    "\n"
    "@Override\n"
    "public $classname$ mergeFrom(\n"
    "        com.google.protobuf.nano.CodedInputByteBufferNano input)\n"
    "    throws java.io.IOException {\n",
    "classname", descriptor_->name());

  printer->Indent();
  if (HasMapField(descriptor_)) {
    printer->Print(
      "com.google.protobuf.nano.MapFactories.MapFactory mapFactory =\n"
      "  com.google.protobuf.nano.MapFactories.getMapFactory();\n");
  }

  printer->Print(
    "while (true) {\n");
  printer->Indent();

  printer->Print(
    "int tag = input.readTag();\n"
    "switch (tag) {\n");
  printer->Indent();

  printer->Print(
    "case 0:\n"          // zero signals EOF / limit reached
    "  return this;\n"
    "default: {\n");

  printer->Indent();
  if (params_.store_unknown_fields()) {
    printer->Print(
        "if (!storeUnknownField(input, tag)) {\n"
        "  return this;\n"
        "}\n");
  } else {
    printer->Print(
        "if (!com.google.protobuf.nano.WireFormatNano.parseUnknownField(input, tag)) {\n"
        "  return this;\n"   // it's an endgroup tag
        "}\n");
  }
  printer->Print("break;\n");
  printer->Outdent();
  printer->Print("}\n");

  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = sorted_fields[i];
    uint32 tag = WireFormatLite::MakeTag(field->number(),
      WireFormat::WireTypeForFieldType(field->type()));

    printer->Print(
      "case $tag$: {\n",
      "tag", SimpleItoa(tag));
    printer->Indent();

    field_generators_.get(field).GenerateMergingCode(printer);

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

      field_generators_.get(field).GenerateMergingCodeFromPacked(printer);

      printer->Outdent();
      printer->Print(
        "  break;\n"
        "}\n");
    }
  }

  printer->Outdent();
  printer->Outdent();
  printer->Outdent();
  printer->Print(
    "    }\n"     // switch (tag)
    "  }\n"       // while (true)
    "}\n");
}

void MessageGenerator::
GenerateParseFromMethods(io::Printer* printer) {
  // Note:  These are separate from GenerateMessageSerializationMethods()
  //   because they need to be generated even for messages that are optimized
  //   for code size.
  printer->Print(
    "\n"
    "public static $classname$ parseFrom(byte[] data)\n"
    "    throws com.google.protobuf.nano.InvalidProtocolBufferNanoException {\n"
    "  return com.google.protobuf.nano.MessageNano.mergeFrom(new $classname$(), data);\n"
    "}\n"
    "\n"
    "public static $classname$ parseFrom(\n"
    "        com.google.protobuf.nano.CodedInputByteBufferNano input)\n"
    "    throws java.io.IOException {\n"
    "  return new $classname$().mergeFrom(input);\n"
    "}\n",
    "classname", descriptor_->name());
}

void MessageGenerator::GenerateSerializeOneField(
    io::Printer* printer, const FieldDescriptor* field) {
  field_generators_.get(field).GenerateSerializationCode(printer);
}

void MessageGenerator::GenerateClear(io::Printer* printer) {
  if (!params_.generate_clear()) {
    return;
  }
  printer->Print(
    "\n"
    "public $classname$ clear() {\n",
    "classname", descriptor_->name());
  printer->Indent();

  GenerateFieldInitializers(printer);

  printer->Outdent();
  printer->Print(
    "  return this;\n"
    "}\n");
}

void MessageGenerator::GenerateFieldInitializers(io::Printer* printer) {
  // Clear bit fields.
  int totalInts = (field_generators_.total_bits() + 31) / 32;
  for (int i = 0; i < totalInts; i++) {
    printer->Print("$bit_field_name$ = 0;\n",
      "bit_field_name", GetBitFieldName(i));
  }

  // Call clear for all of the fields.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    field_generators_.get(field).GenerateClearCode(printer);
  }

  // Clear oneofs.
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
      "clear$oneof_capitalized_name$();\n",
      "oneof_capitalized_name", UnderscoresToCapitalizedCamelCase(
          descriptor_->oneof_decl(i)));
  }

  // Clear unknown fields.
  if (params_.store_unknown_fields()) {
    printer->Print("unknownFieldData = null;\n");
  }
  printer->Print("cachedSize = -1;\n");
}

void MessageGenerator::GenerateClone(io::Printer* printer) {
  printer->Print(
    "@Override\n"
    "public $classname$ clone() {\n",
    "classname", descriptor_->name());
  printer->Indent();

  printer->Print(
    "$classname$ cloned;\n"
    "try {\n"
    "  cloned = ($classname$) super.clone();\n"
    "} catch (java.lang.CloneNotSupportedException e) {\n"
    "  throw new java.lang.AssertionError(e);\n"
    "}\n",
    "classname", descriptor_->name());

  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(descriptor_->field(i)).GenerateFixClonedCode(printer);
  }

  printer->Outdent();
  printer->Print(
    "  return cloned;\n"
    "}\n"
    "\n");
}

void MessageGenerator::GenerateEquals(io::Printer* printer) {
  // Don't override if there are no fields. We could generate an
  // equals method that compares types, but often empty messages
  // are used as namespaces.
  if (descriptor_->field_count() == 0 && !params_.store_unknown_fields()) {
    return;
  }

  printer->Print(
    "\n"
    "@Override\n"
    "public boolean equals(Object o) {\n");
  printer->Indent();
  printer->Print(
    "if (o == this) {\n"
    "  return true;\n"
    "}\n"
    "if (!(o instanceof $classname$)) {\n"
    "  return false;\n"
    "}\n"
    "$classname$ other = ($classname$) o;\n",
    "classname", descriptor_->name());

  // Checking oneof case before checking each oneof field.
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    const OneofDescriptor* oneof_desc = descriptor_->oneof_decl(i);
    printer->Print(
      "if (this.$oneof_name$Case_ != other.$oneof_name$Case_) {\n"
      "  return false;\n"
      "}\n",
      "oneof_name", UnderscoresToCamelCase(oneof_desc));
  }

  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    field_generators_.get(field).GenerateEqualsCode(printer);
  }

  if (params_.store_unknown_fields()) {
    printer->Print(
      "if (unknownFieldData == null || unknownFieldData.isEmpty()) {\n"
      "  return other.unknownFieldData == null || other.unknownFieldData.isEmpty();\n"
      "} else {\n"
      "  return unknownFieldData.equals(other.unknownFieldData);\n"
      "}");
  } else {
    printer->Print(
      "return true;\n");
  }

  printer->Outdent();
  printer->Print("}\n");
}

void MessageGenerator::GenerateHashCode(io::Printer* printer) {
  if (descriptor_->field_count() == 0 && !params_.store_unknown_fields()) {
    return;
  }

  printer->Print(
    "\n"
    "@Override\n"
    "public int hashCode() {\n");
  printer->Indent();

  printer->Print("int result = 17;\n");
  printer->Print("result = 31 * result + getClass().getName().hashCode();\n");
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    field_generators_.get(field).GenerateHashCodeCode(printer);
  }

  if (params_.store_unknown_fields()) {
    printer->Print(
      "result = 31 * result + \n"
      "  (unknownFieldData == null || unknownFieldData.isEmpty() ? 0 : \n"
      "  unknownFieldData.hashCode());\n");
  }

  printer->Print("return result;\n");

  printer->Outdent();
  printer->Print("}\n");
}

// ===================================================================

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
