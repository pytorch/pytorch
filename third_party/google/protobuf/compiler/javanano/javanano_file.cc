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

#include <iostream>

#include <google/protobuf/compiler/javanano/javanano_file.h>
#include <google/protobuf/compiler/javanano/javanano_enum.h>
#include <google/protobuf/compiler/javanano/javanano_extension.h>
#include <google/protobuf/compiler/javanano/javanano_helpers.h>
#include <google/protobuf/compiler/javanano/javanano_message.h>
#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

namespace {

// Recursively searches the given message to see if it contains any extensions.
bool UsesExtensions(const Message& message) {
  const Reflection* reflection = message.GetReflection();

  // We conservatively assume that unknown fields are extensions.
  if (reflection->GetUnknownFields(message).field_count() > 0) return true;

  vector<const FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);

  for (int i = 0; i < fields.size(); i++) {
    if (fields[i]->is_extension()) return true;

    if (fields[i]->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      if (fields[i]->is_repeated()) {
        int size = reflection->FieldSize(message, fields[i]);
        for (int j = 0; j < size; j++) {
          const Message& sub_message =
            reflection->GetRepeatedMessage(message, fields[i], j);
          if (UsesExtensions(sub_message)) return true;
        }
      } else {
        const Message& sub_message = reflection->GetMessage(message, fields[i]);
        if (UsesExtensions(sub_message)) return true;
      }
    }
  }

  return false;
}

}  // namespace

FileGenerator::FileGenerator(const FileDescriptor* file, const Params& params)
  : file_(file),
    params_(params),
    java_package_(FileJavaPackage(params, file)),
    classname_(FileClassName(params, file)) {}

FileGenerator::~FileGenerator() {}

bool FileGenerator::Validate(string* error) {
  // Check for extensions
  FileDescriptorProto file_proto;
  file_->CopyTo(&file_proto);
  if (UsesExtensions(file_proto) && !params_.store_unknown_fields()) {
    error->assign(file_->name());
    error->append(
        ": Java NANO_RUNTIME only supports extensions when the "
        "'store_unknown_fields' generator option is 'true'.");
    return false;
  }

  if (file_->service_count() != 0 && !params_.ignore_services()) {
    error->assign(file_->name());
    error->append(
      ": Java NANO_RUNTIME does not support services\"");
    return false;
  }

  if (!IsOuterClassNeeded(params_, file_)) {
    return true;
  }

  // Check whether legacy javanano generator would omit the outer class.
  if (!params_.has_java_outer_classname(file_->name())
      && file_->message_type_count() == 1
      && file_->enum_type_count() == 0 && file_->extension_count() == 0) {
    cout << "INFO: " << file_->name() << ":" << endl;
    cout << "Javanano generator has changed to align with java generator. "
        "An outer class will be created for this file and the single message "
        "in the file will become a nested class. Use java_multiple_files to "
        "skip generating the outer class, or set an explicit "
        "java_outer_classname to suppress this message." << endl;
  }

  // Check that no class name matches the file's class name.  This is a common
  // problem that leads to Java compile errors that can be hard to understand.
  // It's especially bad when using the java_multiple_files, since we would
  // end up overwriting the outer class with one of the inner ones.
  bool found_conflict = false;
  for (int i = 0; !found_conflict && i < file_->message_type_count(); i++) {
    if (file_->message_type(i)->name() == classname_) {
      found_conflict = true;
    }
  }
  if (params_.java_enum_style()) {
    for (int i = 0; !found_conflict && i < file_->enum_type_count(); i++) {
      if (file_->enum_type(i)->name() == classname_) {
        found_conflict = true;
      }
    }
  }
  if (found_conflict) {
    error->assign(file_->name());
    error->append(
      ": Cannot generate Java output because the file's outer class name, \"");
    error->append(classname_);
    error->append(
      "\", matches the name of one of the types declared inside it.  "
      "Please either rename the type or use the java_outer_classname "
      "option to specify a different outer class name for the .proto file.");
    return false;
  }
  return true;
}

void FileGenerator::Generate(io::Printer* printer) {
  // We don't import anything because we refer to all classes by their
  // fully-qualified names in the generated source.
  printer->Print(
    "// Generated by the protocol buffer compiler.  DO NOT EDIT!\n");
  if (!java_package_.empty()) {
    printer->Print(
      "\n"
      "package $package$;\n",
      "package", java_package_);
  }

  // Note: constants (from enums, emitted in the loop below) may have the same names as constants
  // in the nested classes. This causes Java warnings, but is not fatal, so we suppress those
  // warnings here in the top-most class declaration.
  printer->Print(
    "\n"
    "@SuppressWarnings(\"hiding\")\n"
    "public interface $classname$ {\n",
    "classname", classname_);
  printer->Indent();

  // -----------------------------------------------------------------

  // Extensions.
  for (int i = 0; i < file_->extension_count(); i++) {
    ExtensionGenerator(file_->extension(i), params_).Generate(printer);
  }

  // Enums.
  for (int i = 0; i < file_->enum_type_count(); i++) {
    EnumGenerator(file_->enum_type(i), params_).Generate(printer);
  }

  // Messages.
  if (!params_.java_multiple_files(file_->name())) {
    for (int i = 0; i < file_->message_type_count(); i++) {
      MessageGenerator(file_->message_type(i), params_).Generate(printer);
    }
  }

  // Static variables.
  for (int i = 0; i < file_->message_type_count(); i++) {
    // TODO(kenton):  Reuse MessageGenerator objects?
    MessageGenerator(file_->message_type(i), params_).GenerateStaticVariables(printer);
  }

  printer->Outdent();
  printer->Print(
    "}\n");
}

template<typename GeneratorClass, typename DescriptorClass>
static void GenerateSibling(const string& package_dir,
                            const string& java_package,
                            const DescriptorClass* descriptor,
                            GeneratorContext* output_directory,
                            vector<string>* file_list,
                            const Params& params) {
  string filename = package_dir + descriptor->name() + ".java";
  file_list->push_back(filename);

  scoped_ptr<io::ZeroCopyOutputStream> output(
    output_directory->Open(filename));
  io::Printer printer(output.get(), '$');

  printer.Print(
    "// Generated by the protocol buffer compiler.  DO NOT EDIT!\n");
  if (!java_package.empty()) {
    printer.Print(
      "\n"
      "package $package$;\n",
      "package", java_package);
  }

  GeneratorClass(descriptor, params).Generate(&printer);
}

void FileGenerator::GenerateSiblings(const string& package_dir,
                                     GeneratorContext* output_directory,
                                     vector<string>* file_list) {
  if (params_.java_multiple_files(file_->name())) {
    for (int i = 0; i < file_->message_type_count(); i++) {
      GenerateSibling<MessageGenerator>(package_dir, java_package_,
                                        file_->message_type(i),
                                        output_directory, file_list, params_);
    }

    if (params_.java_enum_style()) {
      for (int i = 0; i < file_->enum_type_count(); i++) {
        GenerateSibling<EnumGenerator>(package_dir, java_package_,
                                       file_->enum_type(i),
                                       output_directory, file_list, params_);
      }
    }
  }
}

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
