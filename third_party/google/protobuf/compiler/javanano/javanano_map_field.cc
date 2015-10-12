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

#include <google/protobuf/compiler/javanano/javanano_map_field.h>
#include <google/protobuf/compiler/javanano/javanano_helpers.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

namespace {

string TypeName(const Params& params, const FieldDescriptor* field,
                bool boxed) {
  JavaType java_type = GetJavaType(field);
  switch (java_type) {
    case JAVATYPE_MESSAGE:
      return ClassName(params, field->message_type());
    case JAVATYPE_INT:
    case JAVATYPE_LONG:
    case JAVATYPE_FLOAT:
    case JAVATYPE_DOUBLE:
    case JAVATYPE_BOOLEAN:
    case JAVATYPE_STRING:
    case JAVATYPE_BYTES:
    case JAVATYPE_ENUM:
      if (boxed) {
        return BoxedPrimitiveTypeName(java_type);
      } else {
        return PrimitiveTypeName(java_type);
      }
    // No default because we want the compiler to complain if any new JavaTypes
    // are added..
  }

  GOOGLE_LOG(FATAL) << "should not reach here.";
  return "";
}

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

void SetMapVariables(const Params& params,
    const FieldDescriptor* descriptor, map<string, string>* variables) {
  const FieldDescriptor* key = KeyField(descriptor);
  const FieldDescriptor* value = ValueField(descriptor);
  (*variables)["name"] =
    RenameJavaKeywords(UnderscoresToCamelCase(descriptor));
  (*variables)["number"] = SimpleItoa(descriptor->number());
  (*variables)["key_type"] = TypeName(params, key, false);
  (*variables)["boxed_key_type"] = TypeName(params,key, true);
  (*variables)["key_desc_type"] =
      "TYPE_" + ToUpper(FieldDescriptor::TypeName(key->type()));
  (*variables)["key_tag"] = SimpleItoa(internal::WireFormat::MakeTag(key));
  (*variables)["value_type"] = TypeName(params, value, false);
  (*variables)["boxed_value_type"] = TypeName(params, value, true);
  (*variables)["value_desc_type"] =
      "TYPE_" + ToUpper(FieldDescriptor::TypeName(value->type()));
  (*variables)["value_tag"] = SimpleItoa(internal::WireFormat::MakeTag(value));
  (*variables)["type_parameters"] =
      (*variables)["boxed_key_type"] + ", " + (*variables)["boxed_value_type"];
  (*variables)["value_default"] =
      value->type() == FieldDescriptor::TYPE_MESSAGE
          ? "new " + (*variables)["value_type"] + "()"
          : "null";
}
}  // namespace

// ===================================================================
MapFieldGenerator::MapFieldGenerator(const FieldDescriptor* descriptor,
                                     const Params& params)
    : FieldGenerator(params), descriptor_(descriptor) {
  SetMapVariables(params, descriptor, &variables_);
}

MapFieldGenerator::~MapFieldGenerator() {}

void MapFieldGenerator::
GenerateMembers(io::Printer* printer, bool /* unused lazy_init */) const {
  printer->Print(variables_,
    "public java.util.Map<$type_parameters$> $name$;\n");
}

void MapFieldGenerator::
GenerateClearCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$ = null;\n");
}

void MapFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "this.$name$ = com.google.protobuf.nano.InternalNano.mergeMapEntry(\n"
    "  input, this.$name$, mapFactory,\n"
    "  com.google.protobuf.nano.InternalNano.$key_desc_type$,\n"
    "  com.google.protobuf.nano.InternalNano.$value_desc_type$,\n"
    "  $value_default$,\n"
    "  $key_tag$, $value_tag$);\n"
    "\n");
}

void MapFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null) {\n"
    "  com.google.protobuf.nano.InternalNano.serializeMapField(\n"
    "    output, this.$name$, $number$,\n"
    "  com.google.protobuf.nano.InternalNano.$key_desc_type$,\n"
    "  com.google.protobuf.nano.InternalNano.$value_desc_type$);\n"
    "}\n");
}

void MapFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null) {\n"
    "  size += com.google.protobuf.nano.InternalNano.computeMapFieldSize(\n"
    "    this.$name$, $number$,\n"
    "  com.google.protobuf.nano.InternalNano.$key_desc_type$,\n"
    "  com.google.protobuf.nano.InternalNano.$value_desc_type$);\n"
    "}\n");
}

void MapFieldGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!com.google.protobuf.nano.InternalNano.equals(\n"
    "  this.$name$, other.$name$)) {\n"
    "  return false;\n"
    "}\n");
}

void MapFieldGenerator::
GenerateHashCodeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = 31 * result +\n"
    "    com.google.protobuf.nano.InternalNano.hashCode(this.$name$);\n");
}

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
