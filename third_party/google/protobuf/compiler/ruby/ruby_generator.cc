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

#include <google/protobuf/compiler/ruby/ruby_generator.h>

using google::protobuf::internal::scoped_ptr;

namespace google {
namespace protobuf {
namespace compiler {
namespace ruby {

// Forward decls.
std::string IntToString(int32 value);
std::string StripDotProto(const std::string& proto_file);
std::string LabelForField(google::protobuf::FieldDescriptor* field);
std::string TypeName(google::protobuf::FieldDescriptor* field);
void GenerateMessage(const google::protobuf::Descriptor* message,
                     google::protobuf::io::Printer* printer);
void GenerateEnum(const google::protobuf::EnumDescriptor* en,
                  google::protobuf::io::Printer* printer);
void GenerateMessageAssignment(
    const std::string& prefix,
    const google::protobuf::Descriptor* message,
    google::protobuf::io::Printer* printer);
void GenerateEnumAssignment(
    const std::string& prefix,
    const google::protobuf::EnumDescriptor* en,
    google::protobuf::io::Printer* printer);

std::string IntToString(int32 value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

std::string StripDotProto(const std::string& proto_file) {
  int lastindex = proto_file.find_last_of(".");
  return proto_file.substr(0, lastindex);
}

std::string LabelForField(const google::protobuf::FieldDescriptor* field) {
  switch (field->label()) {
    case FieldDescriptor::LABEL_OPTIONAL: return "optional";
    case FieldDescriptor::LABEL_REQUIRED: return "required";
    case FieldDescriptor::LABEL_REPEATED: return "repeated";
    default: assert(false); return "";
  }
}

std::string TypeName(const google::protobuf::FieldDescriptor* field) {
  switch (field->type()) {
    case FieldDescriptor::TYPE_INT32: return "int32";
    case FieldDescriptor::TYPE_INT64: return "int64";
    case FieldDescriptor::TYPE_UINT32: return "uint32";
    case FieldDescriptor::TYPE_UINT64: return "uint64";
    case FieldDescriptor::TYPE_SINT32: return "sint32";
    case FieldDescriptor::TYPE_SINT64: return "sint64";
    case FieldDescriptor::TYPE_FIXED32: return "fixed32";
    case FieldDescriptor::TYPE_FIXED64: return "fixed64";
    case FieldDescriptor::TYPE_SFIXED32: return "sfixed32";
    case FieldDescriptor::TYPE_SFIXED64: return "sfixed64";
    case FieldDescriptor::TYPE_DOUBLE: return "double";
    case FieldDescriptor::TYPE_FLOAT: return "float";
    case FieldDescriptor::TYPE_BOOL: return "bool";
    case FieldDescriptor::TYPE_ENUM: return "enum";
    case FieldDescriptor::TYPE_STRING: return "string";
    case FieldDescriptor::TYPE_BYTES: return "bytes";
    case FieldDescriptor::TYPE_MESSAGE: return "message";
    case FieldDescriptor::TYPE_GROUP: return "group";
    default: assert(false); return "";
  }
}

void GenerateField(const google::protobuf::FieldDescriptor* field,
                   google::protobuf::io::Printer* printer) {

  if (field->is_map()) {
    const FieldDescriptor* key_field =
        field->message_type()->FindFieldByNumber(1);
    const FieldDescriptor* value_field =
        field->message_type()->FindFieldByNumber(2);

    printer->Print(
      "map :$name$, :$key_type$, :$value_type$, $number$",
      "name", field->name(),
      "key_type", TypeName(key_field),
      "value_type", TypeName(value_field),
      "number", IntToString(field->number()));

    if (value_field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      printer->Print(
        ", \"$subtype$\"\n",
        "subtype", value_field->message_type()->full_name());
    } else if (value_field->cpp_type() == FieldDescriptor::CPPTYPE_ENUM) {
      printer->Print(
        ", \"$subtype$\"\n",
        "subtype", value_field->enum_type()->full_name());
    } else {
      printer->Print("\n");
    }
  } else {

    printer->Print(
      "$label$ :$name$, ",
      "label", LabelForField(field),
      "name", field->name());
    printer->Print(
      ":$type$, $number$",
      "type", TypeName(field),
      "number", IntToString(field->number()));

    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      printer->Print(
        ", \"$subtype$\"\n",
       "subtype", field->message_type()->full_name());
    } else if (field->cpp_type() == FieldDescriptor::CPPTYPE_ENUM) {
      printer->Print(
        ", \"$subtype$\"\n",
        "subtype", field->enum_type()->full_name());
    } else {
      printer->Print("\n");
    }
  }
}

void GenerateOneof(const google::protobuf::OneofDescriptor* oneof,
                   google::protobuf::io::Printer* printer) {
  printer->Print(
      "oneof :$name$ do\n",
      "name", oneof->name());
  printer->Indent();

  for (int i = 0; i < oneof->field_count(); i++) {
    const FieldDescriptor* field = oneof->field(i);
    GenerateField(field, printer);
  }

  printer->Outdent();
  printer->Print("end\n");
}

void GenerateMessage(const google::protobuf::Descriptor* message,
                     google::protobuf::io::Printer* printer) {

  // Don't generate MapEntry messages -- we use the Ruby extension's native
  // support for map fields instead.
  if (message->options().map_entry()) {
    return;
  }

  printer->Print(
    "add_message \"$name$\" do\n",
    "name", message->full_name());
  printer->Indent();

  for (int i = 0; i < message->field_count(); i++) {
    const FieldDescriptor* field = message->field(i);
    if (!field->containing_oneof()) {
      GenerateField(field, printer);
    }
  }

  for (int i = 0; i < message->oneof_decl_count(); i++) {
    const OneofDescriptor* oneof = message->oneof_decl(i);
    GenerateOneof(oneof, printer);
  }

  printer->Outdent();
  printer->Print("end\n");

  for (int i = 0; i < message->nested_type_count(); i++) {
    GenerateMessage(message->nested_type(i), printer);
  }
  for (int i = 0; i < message->enum_type_count(); i++) {
    GenerateEnum(message->enum_type(i), printer);
  }
}

void GenerateEnum(const google::protobuf::EnumDescriptor* en,
                  google::protobuf::io::Printer* printer) {
  printer->Print(
    "add_enum \"$name$\" do\n",
    "name", en->full_name());
  printer->Indent();

  for (int i = 0; i < en->value_count(); i++) {
    const EnumValueDescriptor* value = en->value(i);
    printer->Print(
      "value :$name$, $number$\n",
      "name", value->name(),
      "number", IntToString(value->number()));
  }

  printer->Outdent();
  printer->Print(
    "end\n");
}

// Module names, class names, and enum value names need to be Ruby constants,
// which must start with a capital letter.
std::string RubifyConstant(const std::string& name) {
  std::string ret = name;
  if (!ret.empty()) {
    if (ret[0] >= 'a' && ret[0] <= 'z') {
      // If it starts with a lowercase letter, capitalize it.
      ret[0] = ret[0] - 'a' + 'A';
    } else if (ret[0] < 'A' || ret[0] > 'Z') {
      // Otherwise (e.g. if it begins with an underscore), we need to come up
      // with some prefix that starts with a capital letter. We could be smarter
      // here, e.g. try to strip leading underscores, but this may cause other
      // problems if the user really intended the name. So let's just prepend a
      // well-known suffix.
      ret = "PB_" + ret;
    }
  }
  return ret;
}

void GenerateMessageAssignment(
    const std::string& prefix,
    const google::protobuf::Descriptor* message,
    google::protobuf::io::Printer* printer) {

  // Don't generate MapEntry messages -- we use the Ruby extension's native
  // support for map fields instead.
  if (message->options().map_entry()) {
    return;
  }

  printer->Print(
    "$prefix$$name$ = ",
    "prefix", prefix,
    "name", RubifyConstant(message->name()));
  printer->Print(
    "Google::Protobuf::DescriptorPool.generated_pool."
    "lookup(\"$full_name$\").msgclass\n",
    "full_name", message->full_name());

  std::string nested_prefix = prefix + message->name() + "::";
  for (int i = 0; i < message->nested_type_count(); i++) {
    GenerateMessageAssignment(nested_prefix, message->nested_type(i), printer);
  }
  for (int i = 0; i < message->enum_type_count(); i++) {
    GenerateEnumAssignment(nested_prefix, message->enum_type(i), printer);
  }
}

void GenerateEnumAssignment(
    const std::string& prefix,
    const google::protobuf::EnumDescriptor* en,
    google::protobuf::io::Printer* printer) {
  printer->Print(
    "$prefix$$name$ = ",
    "prefix", prefix,
    "name", RubifyConstant(en->name()));
  printer->Print(
    "Google::Protobuf::DescriptorPool.generated_pool."
    "lookup(\"$full_name$\").enummodule\n",
    "full_name", en->full_name());
}

int GeneratePackageModules(
    std::string package_name,
    google::protobuf::io::Printer* printer) {
  int levels = 0;
  while (!package_name.empty()) {
    size_t dot_index = package_name.find(".");
    string component;
    if (dot_index == string::npos) {
      component = package_name;
      package_name = "";
    } else {
      component = package_name.substr(0, dot_index);
      package_name = package_name.substr(dot_index + 1);
    }
    component = RubifyConstant(component);
    printer->Print(
      "module $name$\n",
      "name", component);
    printer->Indent();
    levels++;
  }
  return levels;
}

void EndPackageModules(
    int levels,
    google::protobuf::io::Printer* printer) {
  while (levels > 0) {
    levels--;
    printer->Outdent();
    printer->Print(
      "end\n");
  }
}

void GenerateFile(const google::protobuf::FileDescriptor* file,
                  google::protobuf::io::Printer* printer) {
  printer->Print(
    "# Generated by the protocol buffer compiler.  DO NOT EDIT!\n"
    "# source: $filename$\n"
    "\n",
    "filename", file->name());

  printer->Print(
    "require 'google/protobuf'\n\n");

  for (int i = 0; i < file->dependency_count(); i++) {
    const std::string& name = file->dependency(i)->name();
    printer->Print(
      "require '$name$'\n", "name", StripDotProto(name));
  }

  printer->Print(
    "Google::Protobuf::DescriptorPool.generated_pool.build do\n");
  printer->Indent();
  for (int i = 0; i < file->message_type_count(); i++) {
    GenerateMessage(file->message_type(i), printer);
  }
  for (int i = 0; i < file->enum_type_count(); i++) {
    GenerateEnum(file->enum_type(i), printer);
  }
  printer->Outdent();
  printer->Print(
    "end\n\n");

  int levels = GeneratePackageModules(file->package(), printer);
  for (int i = 0; i < file->message_type_count(); i++) {
    GenerateMessageAssignment("", file->message_type(i), printer);
  }
  for (int i = 0; i < file->enum_type_count(); i++) {
    GenerateEnumAssignment("", file->enum_type(i), printer);
  }
  EndPackageModules(levels, printer);
}

bool Generator::Generate(
    const FileDescriptor* file,
    const string& parameter,
    GeneratorContext* generator_context,
    string* error) const {

  if (file->syntax() != FileDescriptor::SYNTAX_PROTO3) {
    *error =
        "Can only generate Ruby code for proto3 .proto files.\n"
        "Please add 'syntax = \"proto3\";' to the top of your .proto file.\n";
    return false;
  }

  std::string filename =
      StripDotProto(file->name()) + ".rb";
  scoped_ptr<io::ZeroCopyOutputStream> output(
      generator_context->Open(filename));
  io::Printer printer(output.get(), '$');

  GenerateFile(file, &printer);

  return true;
}

}  // namespace ruby
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
