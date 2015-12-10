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

#include <google/protobuf/compiler/cpp/cpp_file.h>
#include <map>
#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <set>

#include <google/protobuf/compiler/cpp/cpp_enum.h>
#include <google/protobuf/compiler/cpp/cpp_service.h>
#include <google/protobuf/compiler/cpp/cpp_extension.h>
#include <google/protobuf/compiler/cpp/cpp_helpers.h>
#include <google/protobuf/compiler/cpp/cpp_message.h>
#include <google/protobuf/compiler/cpp/cpp_field.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace cpp {

// ===================================================================

FileGenerator::FileGenerator(const FileDescriptor* file, const Options& options)
    : file_(file),
      message_generators_(
          new google::protobuf::scoped_ptr<MessageGenerator>[file->message_type_count()]),
      enum_generators_(
          new google::protobuf::scoped_ptr<EnumGenerator>[file->enum_type_count()]),
      service_generators_(
          new google::protobuf::scoped_ptr<ServiceGenerator>[file->service_count()]),
      extension_generators_(
          new google::protobuf::scoped_ptr<ExtensionGenerator>[file->extension_count()]),
      options_(options) {

  for (int i = 0; i < file->message_type_count(); i++) {
    message_generators_[i].reset(
      new MessageGenerator(file->message_type(i), options));
  }

  for (int i = 0; i < file->enum_type_count(); i++) {
    enum_generators_[i].reset(
      new EnumGenerator(file->enum_type(i), options));
  }

  for (int i = 0; i < file->service_count(); i++) {
    service_generators_[i].reset(
      new ServiceGenerator(file->service(i), options));
  }

  for (int i = 0; i < file->extension_count(); i++) {
    extension_generators_[i].reset(
      new ExtensionGenerator(file->extension(i), options));
  }

  SplitStringUsing(file_->package(), ".", &package_parts_);
}

FileGenerator::~FileGenerator() {}

void FileGenerator::GenerateProtoHeader(io::Printer* printer) {
  if (!options_.proto_h) {
    return;
  }

  string filename_identifier = FilenameIdentifier(file_->name());
  GenerateTopHeaderGuard(printer, filename_identifier);


  GenerateLibraryIncludes(printer);

  for (int i = 0; i < file_->public_dependency_count(); i++) {
    const FileDescriptor* dep = file_->public_dependency(i);
    const char* extension = ".proto.h";
    string dependency = StripProto(dep->name()) + extension;
    printer->Print(
      "#include \"$dependency$\"  // IWYU pragma: export\n",
      "dependency", dependency);
  }

  printer->Print(
    "// @@protoc_insertion_point(includes)\n");


  GenerateForwardDeclarations(printer);

  // Open namespace.
  GenerateNamespaceOpeners(printer);

  GenerateGlobalStateFunctionDeclarations(printer);

  printer->Print("\n");

  GenerateEnumDefinitions(printer);

  printer->Print(kThickSeparator);
  printer->Print("\n");

  GenerateMessageDefinitions(printer);

  printer->Print("\n");
  printer->Print(kThickSeparator);
  printer->Print("\n");

  GenerateServiceDefinitions(printer);

  GenerateExtensionIdentifiers(printer);

  printer->Print("\n");
  printer->Print(kThickSeparator);
  printer->Print("\n");

  GenerateInlineFunctionDefinitions(printer);

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(namespace_scope)\n"
    "\n");

  // Close up namespace.
  GenerateNamespaceClosers(printer);

  // We need to specialize some templates in the ::google::protobuf namespace:
  GenerateProto2NamespaceEnumSpecializations(printer);

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(global_scope)\n"
    "\n");

  GenerateBottomHeaderGuard(printer, filename_identifier);
}

void FileGenerator::GeneratePBHeader(io::Printer* printer) {
  string filename_identifier =
      FilenameIdentifier(file_->name() + (options_.proto_h ? ".pb.h" : ""));
  GenerateTopHeaderGuard(printer, filename_identifier);

  if (options_.proto_h) {
    printer->Print("#include \"$basename$.proto.h\"  // IWYU pragma: export\n",
                   "basename", StripProto(file_->name()));
  } else {
    GenerateLibraryIncludes(printer);
  }
  GenerateDependencyIncludes(printer);

  printer->Print(
    "// @@protoc_insertion_point(includes)\n");



  // Open namespace.
  GenerateNamespaceOpeners(printer);

  if (!options_.proto_h) {
    GenerateGlobalStateFunctionDeclarations(printer);
    GenerateMessageForwardDeclarations(printer);

    printer->Print("\n");

    GenerateEnumDefinitions(printer);

    printer->Print(kThickSeparator);
    printer->Print("\n");

    GenerateMessageDefinitions(printer);

    printer->Print("\n");
    printer->Print(kThickSeparator);
    printer->Print("\n");

    GenerateServiceDefinitions(printer);

    GenerateExtensionIdentifiers(printer);

    printer->Print("\n");
    printer->Print(kThickSeparator);
    printer->Print("\n");

    GenerateInlineFunctionDefinitions(printer);
  }

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(namespace_scope)\n");

  // Close up namespace.
  GenerateNamespaceClosers(printer);

  if (!options_.proto_h) {
    // We need to specialize some templates in the ::google::protobuf namespace:
    GenerateProto2NamespaceEnumSpecializations(printer);
  }

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(global_scope)\n"
    "\n");

  GenerateBottomHeaderGuard(printer, filename_identifier);
}

void FileGenerator::GenerateSource(io::Printer* printer) {
  bool well_known = IsWellKnownMessage(file_);
  string header =
      StripProto(file_->name()) + (options_.proto_h ? ".proto.h" : ".pb.h");
  printer->Print(
    "// Generated by the protocol buffer compiler.  DO NOT EDIT!\n"
    "// source: $filename$\n"
    "\n"
    // The generated code calls accessors that might be deprecated. We don't
    // want the compiler to warn in generated code.
    "#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION\n"
    "#include $left$$header$$right$\n"
    "\n"
    "#include <algorithm>\n"    // for swap()
    "\n"
    "#include <google/protobuf/stubs/common.h>\n"
    "#include <google/protobuf/stubs/once.h>\n"
    "#include <google/protobuf/io/coded_stream.h>\n"
    "#include <google/protobuf/wire_format_lite_inl.h>\n",
    "filename", file_->name(),
    "header", header,
    "left", well_known ? "<" : "\"",
    "right", well_known ? ">" : "\"");

  // Unknown fields implementation in lite mode uses StringOutputStream
  if (!UseUnknownFieldSet(file_) && file_->message_type_count() > 0) {
    printer->Print(
      "#include <google/protobuf/io/zero_copy_stream_impl_lite.h>\n");
  }

  if (HasDescriptorMethods(file_)) {
    printer->Print(
      "#include <google/protobuf/descriptor.h>\n"
      "#include <google/protobuf/generated_message_reflection.h>\n"
      "#include <google/protobuf/reflection_ops.h>\n"
      "#include <google/protobuf/wire_format.h>\n");
  }

  if (options_.proto_h) {
    // Use the smaller .proto.h files.
    for (int i = 0; i < file_->dependency_count(); i++) {
      const FileDescriptor* dep = file_->dependency(i);
      const char* extension = ".proto.h";
      string dependency = StripProto(dep->name()) + extension;
      printer->Print(
          "#include \"$dependency$\"\n",
          "dependency", dependency);
    }
  }

  printer->Print(
    "// @@protoc_insertion_point(includes)\n");

  GenerateNamespaceOpeners(printer);

  if (HasDescriptorMethods(file_)) {
    printer->Print(
      "\n"
      "namespace {\n"
      "\n");
    for (int i = 0; i < file_->message_type_count(); i++) {
      message_generators_[i]->GenerateDescriptorDeclarations(printer);
    }
    for (int i = 0; i < file_->enum_type_count(); i++) {
      printer->Print(
        "const ::google::protobuf::EnumDescriptor* $name$_descriptor_ = NULL;\n",
        "name", ClassName(file_->enum_type(i), false));
    }

    if (HasGenericServices(file_)) {
      for (int i = 0; i < file_->service_count(); i++) {
        printer->Print(
          "const ::google::protobuf::ServiceDescriptor* $name$_descriptor_ = NULL;\n",
          "name", file_->service(i)->name());
      }
    }

    printer->Print(
      "\n"
      "}  // namespace\n"
      "\n");
  }

  // Define our externally-visible BuildDescriptors() function.  (For the lite
  // library, all this does is initialize default instances.)
  GenerateBuildDescriptors(printer);

  // Generate enums.
  for (int i = 0; i < file_->enum_type_count(); i++) {
    enum_generators_[i]->GenerateMethods(printer);
  }

  // Generate classes.
  for (int i = 0; i < file_->message_type_count(); i++) {
    if (i == 0 && HasGeneratedMethods(file_)) {
      printer->Print(
          "\n"
          "namespace {\n"
          "\n"
          "static void MergeFromFail(int line) GOOGLE_ATTRIBUTE_COLD;\n"
          "static void MergeFromFail(int line) {\n"
          "  GOOGLE_CHECK(false) << __FILE__ << \":\" << line;\n"
          "}\n"
          "\n"
          "}  // namespace\n"
          "\n");
    }
    printer->Print("\n");
    printer->Print(kThickSeparator);
    printer->Print("\n");
    message_generators_[i]->GenerateClassMethods(printer);

    printer->Print("#if PROTOBUF_INLINE_NOT_IN_HEADERS\n");
    // Generate class inline methods.
    message_generators_[i]->GenerateInlineMethods(printer,
                                                  /* is_inline = */ false);
    printer->Print("#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS\n");
  }

  if (HasGenericServices(file_)) {
    // Generate services.
    for (int i = 0; i < file_->service_count(); i++) {
      if (i == 0) printer->Print("\n");
      printer->Print(kThickSeparator);
      printer->Print("\n");
      service_generators_[i]->GenerateImplementation(printer);
    }
  }

  // Define extensions.
  for (int i = 0; i < file_->extension_count(); i++) {
    extension_generators_[i]->GenerateDefinition(printer);
  }

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(namespace_scope)\n");

  GenerateNamespaceClosers(printer);

  printer->Print(
    "\n"
    "// @@protoc_insertion_point(global_scope)\n");
}

class FileGenerator::ForwardDeclarations {
 public:
  ~ForwardDeclarations() {
    for (map<string, ForwardDeclarations *>::iterator it = namespaces_.begin(),
                                                      end = namespaces_.end();
         it != end; ++it) {
      delete it->second;
    }
    namespaces_.clear();
  }

  ForwardDeclarations* AddOrGetNamespace(const string& ns_name) {
    ForwardDeclarations*& ns = namespaces_[ns_name];
    if (ns == NULL) {
      ns = new ForwardDeclarations;
    }
    return ns;
  }

  set<string>& classes() { return classes_; }
  set<string>& enums() { return enums_; }

  void Print(io::Printer* printer) const {
    for (set<string>::const_iterator it = enums_.begin(), end = enums_.end();
         it != end; ++it) {
      printer->Print("enum $enumname$ : int;\n"
                     "bool $enumname$_IsValid(int value);\n",
                     "enumname", it->c_str());
    }
    for (set<string>::const_iterator it = classes_.begin(),
                                     end = classes_.end();
         it != end; ++it) {
      printer->Print("class $classname$;\n", "classname", it->c_str());
    }
    for (map<string, ForwardDeclarations *>::const_iterator
             it = namespaces_.begin(),
             end = namespaces_.end();
         it != end; ++it) {
      printer->Print("namespace $nsname$ {\n",
                     "nsname", it->first);
      it->second->Print(printer);
      printer->Print("}  // namespace $nsname$\n",
                     "nsname", it->first);
    }
  }


 private:
  map<string, ForwardDeclarations*> namespaces_;
  set<string> classes_;
  set<string> enums_;
};

void FileGenerator::GenerateBuildDescriptors(io::Printer* printer) {
  // AddDescriptors() is a file-level procedure which adds the encoded
  // FileDescriptorProto for this .proto file to the global DescriptorPool for
  // generated files (DescriptorPool::generated_pool()). It either runs at
  // static initialization time (by default) or when default_instance() is
  // called for the first time (in LITE_RUNTIME mode with
  // GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER flag enabled). This procedure also
  // constructs default instances and registers extensions.
  //
  // Its sibling, AssignDescriptors(), actually pulls the compiled
  // FileDescriptor from the DescriptorPool and uses it to populate all of
  // the global variables which store pointers to the descriptor objects.
  // It also constructs the reflection objects.  It is called the first time
  // anyone calls descriptor() or GetReflection() on one of the types defined
  // in the file.

  // In optimize_for = LITE_RUNTIME mode, we don't generate AssignDescriptors()
  // and we only use AddDescriptors() to allocate default instances.
  if (HasDescriptorMethods(file_)) {
    printer->Print(
      "\n"
      "void $assigndescriptorsname$() {\n",
      "assigndescriptorsname", GlobalAssignDescriptorsName(file_->name()));
    printer->Indent();

    // Make sure the file has found its way into the pool.  If a descriptor
    // is requested *during* static init then AddDescriptors() may not have
    // been called yet, so we call it manually.  Note that it's fine if
    // AddDescriptors() is called multiple times.
    printer->Print(
      "$adddescriptorsname$();\n",
      "adddescriptorsname", GlobalAddDescriptorsName(file_->name()));

    // Get the file's descriptor from the pool.
    printer->Print(
      "const ::google::protobuf::FileDescriptor* file =\n"
      "  ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(\n"
      "    \"$filename$\");\n"
      // Note that this GOOGLE_CHECK is necessary to prevent a warning about "file"
      // being unused when compiling an empty .proto file.
      "GOOGLE_CHECK(file != NULL);\n",
      "filename", file_->name());

    // Go through all the stuff defined in this file and generated code to
    // assign the global descriptor pointers based on the file descriptor.
    for (int i = 0; i < file_->message_type_count(); i++) {
      message_generators_[i]->GenerateDescriptorInitializer(printer, i);
    }
    for (int i = 0; i < file_->enum_type_count(); i++) {
      enum_generators_[i]->GenerateDescriptorInitializer(printer, i);
    }
    if (HasGenericServices(file_)) {
      for (int i = 0; i < file_->service_count(); i++) {
        service_generators_[i]->GenerateDescriptorInitializer(printer, i);
      }
    }

    printer->Outdent();
    printer->Print(
      "}\n"
      "\n");

    // ---------------------------------------------------------------

    // protobuf_AssignDescriptorsOnce():  The first time it is called, calls
    // AssignDescriptors().  All later times, waits for the first call to
    // complete and then returns.
    printer->Print(
      "namespace {\n"
      "\n"
      "GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);\n"
      "inline void protobuf_AssignDescriptorsOnce() {\n"
      "  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,\n"
      "                 &$assigndescriptorsname$);\n"
      "}\n"
      "\n",
      "assigndescriptorsname", GlobalAssignDescriptorsName(file_->name()));

    // protobuf_RegisterTypes():  Calls
    // MessageFactory::InternalRegisterGeneratedType() for each message type.
    printer->Print(
      "void protobuf_RegisterTypes(const ::std::string&) {\n"
      "  protobuf_AssignDescriptorsOnce();\n");
    printer->Indent();

    for (int i = 0; i < file_->message_type_count(); i++) {
      message_generators_[i]->GenerateTypeRegistrations(printer);
    }

    printer->Outdent();
    printer->Print(
      "}\n"
      "\n"
      "}  // namespace\n");
  }

  // -----------------------------------------------------------------

  // ShutdownFile():  Deletes descriptors, default instances, etc. on shutdown.
  printer->Print(
    "\n"
    "void $shutdownfilename$() {\n",
    "shutdownfilename", GlobalShutdownFileName(file_->name()));
  printer->Indent();

  for (int i = 0; i < file_->message_type_count(); i++) {
    message_generators_[i]->GenerateShutdownCode(printer);
  }

  printer->Outdent();
  printer->Print(
    "}\n\n");

  // -----------------------------------------------------------------

  // Now generate the AddDescriptors() function.
  PrintHandlingOptionalStaticInitializers(
    file_, printer,
    // With static initializers.
    // Note that we don't need any special synchronization in the following code
    // because it is called at static init time before any threads exist.
    "void $adddescriptorsname$() {\n"
    "  static bool already_here = false;\n"
    "  if (already_here) return;\n"
    "  already_here = true;\n"
    "  GOOGLE_PROTOBUF_VERIFY_VERSION;\n"
    "\n",
    // Without.
    "void $adddescriptorsname$_impl() {\n"
    "  GOOGLE_PROTOBUF_VERIFY_VERSION;\n"
    "\n",
    // Vars.
    "adddescriptorsname", GlobalAddDescriptorsName(file_->name()));

  printer->Indent();

  // Call the AddDescriptors() methods for all of our dependencies, to make
  // sure they get added first.
  for (int i = 0; i < file_->dependency_count(); i++) {
    const FileDescriptor* dependency = file_->dependency(i);
    // Print the namespace prefix for the dependency.
    string add_desc_name = QualifiedFileLevelSymbol(
        dependency->package(), GlobalAddDescriptorsName(dependency->name()));
    // Call its AddDescriptors function.
    printer->Print(
      "$name$();\n",
      "name", add_desc_name);
  }

  if (HasDescriptorMethods(file_)) {
    // Embed the descriptor.  We simply serialize the entire FileDescriptorProto
    // and embed it as a string literal, which is parsed and built into real
    // descriptors at initialization time.
    FileDescriptorProto file_proto;
    file_->CopyTo(&file_proto);
    string file_data;
    file_proto.SerializeToString(&file_data);

#ifdef _MSC_VER
    bool breakdown_large_file = true;
#else
    bool breakdown_large_file = false;
#endif
    // Workaround for MSVC: "Error C1091: compiler limit: string exceeds 65535
    // bytes in length". Declare a static array of characters rather than use a
    // string literal.
    if (breakdown_large_file && file_data.size() > 65535) {
      // This has to be explicitly marked as a signed char because the generated
      // code puts negative values in the array, and sometimes plain char is
      // unsigned. That implicit narrowing conversion is not allowed in C++11.
      // <http://stackoverflow.com/questions/4434140/narrowing-conversions-in-c0x-is-it-just-me-or-does-this-sound-like-a-breakin>
      // has details on why.
      printer->Print(
          "static const signed char descriptor[] = {\n");
      printer->Indent();

      // Only write 25 bytes per line.
      static const int kBytesPerLine = 25;
      for (int i = 0; i < file_data.size();) {
          for (int j = 0; j < kBytesPerLine && i < file_data.size(); ++i, ++j) {
            printer->Print(
                "$char$, ",
                "char", SimpleItoa(file_data[i]));
          }
          printer->Print(
              "\n");
      }

      printer->Outdent();
      printer->Print(
          "};\n");

      printer->Print(
          "::google::protobuf::DescriptorPool::InternalAddGeneratedFile(descriptor, $size$);\n",
          "size", SimpleItoa(file_data.size()));

    } else {
      printer->Print(
        "::google::protobuf::DescriptorPool::InternalAddGeneratedFile(");

      // Only write 40 bytes per line.
      static const int kBytesPerLine = 40;
      for (int i = 0; i < file_data.size(); i += kBytesPerLine) {
        printer->Print("\n  \"$data$\"",
                       "data",
                       EscapeTrigraphs(
                           CEscape(file_data.substr(i, kBytesPerLine))));
    }
    printer->Print(
        ", $size$);\n",
        "size", SimpleItoa(file_data.size()));
    }

    // Call MessageFactory::InternalRegisterGeneratedFile().
    printer->Print(
      "::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(\n"
      "  \"$filename$\", &protobuf_RegisterTypes);\n",
      "filename", file_->name());
  }

  // Allocate and initialize default instances.  This can't be done lazily
  // since default instances are returned by simple accessors and are used with
  // extensions.  Speaking of which, we also register extensions at this time.
  for (int i = 0; i < file_->message_type_count(); i++) {
    message_generators_[i]->GenerateDefaultInstanceAllocator(printer);
  }
  for (int i = 0; i < file_->extension_count(); i++) {
    extension_generators_[i]->GenerateRegistration(printer);
  }
  for (int i = 0; i < file_->message_type_count(); i++) {
    message_generators_[i]->GenerateDefaultInstanceInitializer(printer);
  }

  printer->Print(
    "::google::protobuf::internal::OnShutdown(&$shutdownfilename$);\n",
    "shutdownfilename", GlobalShutdownFileName(file_->name()));

  printer->Outdent();
  printer->Print(
    "}\n"
    "\n");

  PrintHandlingOptionalStaticInitializers(
    file_, printer,
    // With static initializers.
    "// Force AddDescriptors() to be called at static initialization time.\n"
    "struct StaticDescriptorInitializer_$filename$ {\n"
    "  StaticDescriptorInitializer_$filename$() {\n"
    "    $adddescriptorsname$();\n"
    "  }\n"
    "} static_descriptor_initializer_$filename$_;\n",
    // Without.
    "GOOGLE_PROTOBUF_DECLARE_ONCE($adddescriptorsname$_once_);\n"
    "void $adddescriptorsname$() {\n"
    "  ::google::protobuf::GoogleOnceInit(&$adddescriptorsname$_once_,\n"
    "                 &$adddescriptorsname$_impl);\n"
    "}\n",
    // Vars.
    "adddescriptorsname", GlobalAddDescriptorsName(file_->name()),
    "filename", FilenameIdentifier(file_->name()));
}

void FileGenerator::GenerateNamespaceOpeners(io::Printer* printer) {
  if (package_parts_.size() > 0) printer->Print("\n");

  for (int i = 0; i < package_parts_.size(); i++) {
    printer->Print("namespace $part$ {\n",
                   "part", package_parts_[i]);
  }
}

void FileGenerator::GenerateNamespaceClosers(io::Printer* printer) {
  if (package_parts_.size() > 0) printer->Print("\n");

  for (int i = package_parts_.size() - 1; i >= 0; i--) {
    printer->Print("}  // namespace $part$\n",
                   "part", package_parts_[i]);
  }
}

void FileGenerator::GenerateForwardDeclarations(io::Printer* printer) {
  ForwardDeclarations decls;
  for (int i = 0; i < file_->dependency_count(); i++) {
    FileGenerator dependency(file_->dependency(i), options_);
    dependency.FillForwardDeclarations(&decls);
  }
  FillForwardDeclarations(&decls);
  decls.Print(printer);
}

void FileGenerator::FillForwardDeclarations(ForwardDeclarations* decls) {
  for (int i = 0; i < file_->public_dependency_count(); i++) {
    FileGenerator dependency(file_->public_dependency(i), options_);
    dependency.FillForwardDeclarations(decls);
  }
  for (int i = 0; i < package_parts_.size(); i++) {
    decls = decls->AddOrGetNamespace(package_parts_[i]);
  }
  // Generate enum definitions.
  for (int i = 0; i < file_->message_type_count(); i++) {
    message_generators_[i]->FillEnumForwardDeclarations(&decls->enums());
  }
  for (int i = 0; i < file_->enum_type_count(); i++) {
    enum_generators_[i]->FillForwardDeclaration(&decls->enums());
  }
  // Generate forward declarations of classes.
  for (int i = 0; i < file_->message_type_count(); i++) {
    message_generators_[i]->FillMessageForwardDeclarations(
        &decls->classes());
  }
}

void FileGenerator::GenerateTopHeaderGuard(io::Printer* printer,
                                           const string& filename_identifier) {
  // Generate top of header.
  printer->Print(
    "// Generated by the protocol buffer compiler.  DO NOT EDIT!\n"
    "// source: $filename$\n"
    "\n"
    "#ifndef PROTOBUF_$filename_identifier$__INCLUDED\n"
    "#define PROTOBUF_$filename_identifier$__INCLUDED\n"
    "\n"
    "#include <string>\n"
    "\n",
    "filename", file_->name(),
    "filename_identifier", filename_identifier);
}

void FileGenerator::GenerateBottomHeaderGuard(
    io::Printer* printer, const string& filename_identifier) {
  printer->Print(
    "#endif  // PROTOBUF_$filename_identifier$__INCLUDED\n",
    "filename_identifier", filename_identifier);
}

void FileGenerator::GenerateLibraryIncludes(io::Printer* printer) {

  printer->Print(
    "#include <google/protobuf/stubs/common.h>\n"
    "\n");

  // Verify the protobuf library header version is compatible with the protoc
  // version before going any further.
  printer->Print(
    "#if GOOGLE_PROTOBUF_VERSION < $min_header_version$\n"
    "#error This file was generated by a newer version of protoc which is\n"
    "#error incompatible with your Protocol Buffer headers.  Please update\n"
    "#error your headers.\n"
    "#endif\n"
    "#if $protoc_version$ < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION\n"
    "#error This file was generated by an older version of protoc which is\n"
    "#error incompatible with your Protocol Buffer headers.  Please\n"
    "#error regenerate this file with a newer version of protoc.\n"
    "#endif\n"
    "\n",
    "min_header_version",
      SimpleItoa(protobuf::internal::kMinHeaderVersionForProtoc),
    "protoc_version", SimpleItoa(GOOGLE_PROTOBUF_VERSION));

  // OK, it's now safe to #include other files.
  printer->Print(
    "#include <google/protobuf/arena.h>\n"
    "#include <google/protobuf/arenastring.h>\n"
    "#include <google/protobuf/generated_message_util.h>\n");
  if (UseUnknownFieldSet(file_)) {
    printer->Print(
      "#include <google/protobuf/metadata.h>\n");
  }
  if (file_->message_type_count() > 0) {
    if (HasDescriptorMethods(file_)) {
      printer->Print(
        "#include <google/protobuf/message.h>\n");
    } else {
      printer->Print(
        "#include <google/protobuf/message_lite.h>\n");
    }
  }
  printer->Print(
    "#include <google/protobuf/repeated_field.h>\n"
    "#include <google/protobuf/extension_set.h>\n");
  if (HasMapFields(file_)) {
    printer->Print(
        "#include <google/protobuf/map.h>\n");
    if (HasDescriptorMethods(file_)) {
      printer->Print(
          "#include <google/protobuf/map_field_inl.h>\n");
    } else {
      printer->Print(
          "#include <google/protobuf/map_field_lite.h>\n");
    }
  }

  if (HasEnumDefinitions(file_)) {
    if (HasDescriptorMethods(file_)) {
      printer->Print(
          "#include <google/protobuf/generated_enum_reflection.h>\n");
    } else {
      printer->Print(
          "#include <google/protobuf/generated_enum_util.h>\n");
    }
  }

  if (HasGenericServices(file_)) {
    printer->Print(
      "#include <google/protobuf/service.h>\n");
  }

  if (UseUnknownFieldSet(file_) && file_->message_type_count() > 0) {
    printer->Print(
      "#include <google/protobuf/unknown_field_set.h>\n");
  }


  if (IsAnyMessage(file_)) {
    printer->Print(
      "#include <google/protobuf/any.h>\n");
  }
}

void FileGenerator::GenerateDependencyIncludes(io::Printer* printer) {
  set<string> public_import_names;
  for (int i = 0; i < file_->public_dependency_count(); i++) {
    public_import_names.insert(file_->public_dependency(i)->name());
  }

  for (int i = 0; i < file_->dependency_count(); i++) {
    bool well_known = IsWellKnownMessage(file_->dependency(i));
    const string& name = file_->dependency(i)->name();
    bool public_import = (public_import_names.count(name) != 0);

    printer->Print(
      "#include $left$$dependency$.pb.h$right$$iwyu$\n",
      "dependency", StripProto(name),
      "iwyu", (public_import) ? "  // IWYU pragma: export" : "",
      "left", well_known ? "<" : "\"",
      "right", well_known ? ">" : "\"");
  }
}

void FileGenerator::GenerateGlobalStateFunctionDeclarations(
    io::Printer* printer) {
  // Forward-declare the AddDescriptors, AssignDescriptors, and ShutdownFile
  // functions, so that we can declare them to be friends of each class.
  printer->Print(
    "\n"
    "// Internal implementation detail -- do not call these.\n"
    "void $dllexport_decl$$adddescriptorsname$();\n",
    "adddescriptorsname", GlobalAddDescriptorsName(file_->name()),
    "dllexport_decl",
    options_.dllexport_decl.empty() ? "" : options_.dllexport_decl + " ");

  printer->Print(
    // Note that we don't put dllexport_decl on these because they are only
    // called by the .pb.cc file in which they are defined.
    "void $assigndescriptorsname$();\n"
    "void $shutdownfilename$();\n"
    "\n",
    "assigndescriptorsname", GlobalAssignDescriptorsName(file_->name()),
    "shutdownfilename", GlobalShutdownFileName(file_->name()));
}

void FileGenerator::GenerateMessageForwardDeclarations(io::Printer* printer) {
  set<string> classes;
  for (int i = 0; i < file_->message_type_count(); i++) {
    message_generators_[i]->FillMessageForwardDeclarations(&classes);
  }
  for (set<string>::const_iterator it = classes.begin(), end = classes.end();
       it != end; ++it) {
    printer->Print("class $classname$;\n", "classname", it->c_str());
  }
}

void FileGenerator::GenerateMessageDefinitions(io::Printer* printer) {
  // Generate class definitions.
  for (int i = 0; i < file_->message_type_count(); i++) {
    if (i > 0) {
      printer->Print("\n");
      printer->Print(kThinSeparator);
      printer->Print("\n");
    }
    message_generators_[i]->GenerateClassDefinition(printer);
  }
}

void FileGenerator::GenerateEnumDefinitions(io::Printer* printer) {
  // Generate enum definitions.
  for (int i = 0; i < file_->message_type_count(); i++) {
    message_generators_[i]->GenerateEnumDefinitions(printer);
  }
  for (int i = 0; i < file_->enum_type_count(); i++) {
    enum_generators_[i]->GenerateDefinition(printer);
  }
}

void FileGenerator::GenerateServiceDefinitions(io::Printer* printer) {
  if (HasGenericServices(file_)) {
    // Generate service definitions.
    for (int i = 0; i < file_->service_count(); i++) {
      if (i > 0) {
        printer->Print("\n");
        printer->Print(kThinSeparator);
        printer->Print("\n");
      }
      service_generators_[i]->GenerateDeclarations(printer);
    }

    printer->Print("\n");
    printer->Print(kThickSeparator);
    printer->Print("\n");
  }
}

void FileGenerator::GenerateExtensionIdentifiers(io::Printer* printer) {
  // Declare extension identifiers.
  for (int i = 0; i < file_->extension_count(); i++) {
    extension_generators_[i]->GenerateDeclaration(printer);
  }
}

void FileGenerator::GenerateInlineFunctionDefinitions(io::Printer* printer) {
  // An aside about inline functions in .proto.h mode:
  //
  // The PROTOBUF_INLINE_NOT_IN_HEADERS symbol controls conditionally
  // moving much of the inline functions to the .pb.cc file, which can be a
  // significant performance benefit for compilation time, at the expense
  // of non-inline function calls.
  //
  // However, in .proto.h mode, the definition of the internal dependent
  // base class must remain in the header, and can never be out-lined. The
  // dependent base class also needs access to has-bit manipuation
  // functions, so the has-bit functions must be unconditionally inlined in
  // proto_h mode.
  //
  // This gives us three flavors of functions:
  //
  //  1. Functions on the message not used by the internal dependent base
  //     class: in .proto.h mode, only some functions are defined on the
  //     message class; others are defined on the dependent base class.
  //     These are guarded and can be out-lined. These are generated by
  //     GenerateInlineMethods, and include has_* bit functions in
  //     non-proto_h mode.
  //
  //  2. Functions on the internal dependent base class: these functions
  //     are dependent on a template parameter, so they always need to
  //     remain in the header.
  //
  //  3. Functions on the message that are used by the dependent base: the
  //     dependent base class down casts itself to the message
  //     implementation class to access these functions (the has_* bit
  //     manipulation functions). Unlike #1, these functions must
  //     unconditionally remain in the header. These are emitted by
  //     GenerateDependentInlineMethods, even though they are not actually
  //     dependent.

  printer->Print("#if !PROTOBUF_INLINE_NOT_IN_HEADERS\n");
  // Generate class inline methods.
  for (int i = 0; i < file_->message_type_count(); i++) {
    if (i > 0) {
      printer->Print(kThinSeparator);
      printer->Print("\n");
    }
    message_generators_[i]->GenerateInlineMethods(printer,
                                                  /* is_inline = */ true);
  }
  printer->Print("#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS\n");

  for (int i = 0; i < file_->message_type_count(); i++) {
    if (i > 0) {
      printer->Print(kThinSeparator);
      printer->Print("\n");
    }
    // Methods of the dependent base class must always be inline in the header.
    message_generators_[i]->GenerateDependentInlineMethods(printer);
  }
}

void FileGenerator::GenerateProto2NamespaceEnumSpecializations(
    io::Printer* printer) {
  // Emit GetEnumDescriptor specializations into google::protobuf namespace:
  if (HasEnumDefinitions(file_)) {
    // The SWIG conditional is to avoid a null-pointer dereference
    // (bug 1984964) in swig-1.3.21 resulting from the following syntax:
    //   namespace X { void Y<Z::W>(); }
    // which appears in GetEnumDescriptor() specializations.
    printer->Print(
        "\n"
        "#ifndef SWIG\n"
        "namespace google {\nnamespace protobuf {\n"
        "\n");
    for (int i = 0; i < file_->message_type_count(); i++) {
      message_generators_[i]->GenerateGetEnumDescriptorSpecializations(printer);
    }
    for (int i = 0; i < file_->enum_type_count(); i++) {
      enum_generators_[i]->GenerateGetEnumDescriptorSpecializations(printer);
    }
    printer->Print(
        "\n"
        "}  // namespace protobuf\n}  // namespace google\n"
        "#endif  // SWIG\n");
  }
}

}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
