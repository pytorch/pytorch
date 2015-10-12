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

#include <google/protobuf/compiler/java/java_name_resolver.h>

#include <map>
#include <string>

#include <google/protobuf/compiler/java/java_helpers.h>
#include <google/protobuf/stubs/substitute.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace java {

namespace {
// A suffix that will be appended to the file's outer class name if the name
// conflicts with some other types defined in the file.
const char* kOuterClassNameSuffix = "OuterClass";

// Strip package name from a descriptor's full name.
// For example:
//   Full name   : foo.Bar.Baz
//   Package name: foo
//   After strip : Bar.Baz
string StripPackageName(const string& full_name,
                        const FileDescriptor* file) {
  if (file->package().empty()) {
    return full_name;
  } else {
    // Strip package name
    return full_name.substr(file->package().size() + 1);
  }
}

// Get the name of a message's Java class without package name prefix.
string ClassNameWithoutPackage(const Descriptor* descriptor,
                               bool immutable) {
  return StripPackageName(descriptor->full_name(),
                          descriptor->file());
}

// Get the name of an enum's Java class without package name prefix.
string ClassNameWithoutPackage(const EnumDescriptor* descriptor,
                               bool immutable) {
  // Doesn't append "Mutable" for enum type's name.
  const Descriptor* message_descriptor = descriptor->containing_type();
  if (message_descriptor == NULL) {
    return descriptor->name();
  } else {
    return ClassNameWithoutPackage(message_descriptor, immutable) +
           "." + descriptor->name();
  }
}

// Get the name of a service's Java class without package name prefix.
string ClassNameWithoutPackage(const ServiceDescriptor* descriptor,
                               bool immutable) {
  string full_name = StripPackageName(descriptor->full_name(),
                                      descriptor->file());
  // We don't allow nested service definitions.
  GOOGLE_CHECK(full_name.find('.') == string::npos);
  return full_name;
}

// Check whether a given message or its nested types has the given class name.
bool MessageHasConflictingClassName(const Descriptor* message,
                                    const string& classname) {
  if (message->name() == classname) return true;
  for (int i = 0; i < message->nested_type_count(); ++i) {
    if (MessageHasConflictingClassName(message->nested_type(i), classname)) {
      return true;
    }
  }
  for (int i = 0; i < message->enum_type_count(); ++i) {
    if (message->enum_type(i)->name() == classname) {
      return true;
    }
  }
  return false;
}

}  // namespace

ClassNameResolver::ClassNameResolver() {
}

ClassNameResolver::~ClassNameResolver() {
}

string ClassNameResolver::GetFileDefaultImmutableClassName(
    const FileDescriptor* file) {
  string basename;
  string::size_type last_slash = file->name().find_last_of('/');
  if (last_slash == string::npos) {
    basename = file->name();
  } else {
    basename = file->name().substr(last_slash + 1);
  }
  return UnderscoresToCamelCase(StripProto(basename), true);
}

string ClassNameResolver::GetFileImmutableClassName(
    const FileDescriptor* file) {
  string& class_name = file_immutable_outer_class_names_[file];
  if (class_name.empty()) {
    if (file->options().has_java_outer_classname()) {
      class_name = file->options().java_outer_classname();
    } else {
      class_name = GetFileDefaultImmutableClassName(file);
      if (HasConflictingClassName(file, class_name)) {
        class_name += kOuterClassNameSuffix;
      }
    }
  }
  return class_name;
}

string ClassNameResolver::GetFileClassName(const FileDescriptor* file,
                                           bool immutable) {
  if (immutable) {
    return GetFileImmutableClassName(file);
  } else {
    return "Mutable" + GetFileImmutableClassName(file);
  }
}

// Check whether there is any type defined in the proto file that has
// the given class name.
bool ClassNameResolver::HasConflictingClassName(
    const FileDescriptor* file, const string& classname) {
  for (int i = 0; i < file->enum_type_count(); i++) {
    if (file->enum_type(i)->name() == classname) {
      return true;
    }
  }
  for (int i = 0; i < file->service_count(); i++) {
    if (file->service(i)->name() == classname) {
      return true;
    }
  }
  for (int i = 0; i < file->message_type_count(); i++) {
    if (MessageHasConflictingClassName(file->message_type(i), classname)) {
      return true;
    }
  }
  return false;
}

string ClassNameResolver::GetDescriptorClassName(
    const FileDescriptor* descriptor) {
  return GetFileImmutableClassName(descriptor);
}

string ClassNameResolver::GetClassName(const FileDescriptor* descriptor,
                                       bool immutable) {
  string result = FileJavaPackage(descriptor, immutable);
  if (!result.empty()) result += '.';
  result += GetFileClassName(descriptor, immutable);
  return result;
}

// Get the full name of a Java class by prepending the Java package name
// or outer class name.
string ClassNameResolver::GetClassFullName(const string& name_without_package,
                                           const FileDescriptor* file,
                                           bool immutable,
                                           bool multiple_files) {
  string result;
  if (multiple_files) {
    result = FileJavaPackage(file, immutable);
  } else {
    result = GetClassName(file, immutable);
  }
  if (!result.empty()) {
    result += '.';
  }
  result += name_without_package;
  return result;
}

string ClassNameResolver::GetClassName(const Descriptor* descriptor,
                                       bool immutable) {
  return GetClassFullName(ClassNameWithoutPackage(descriptor, immutable),
                          descriptor->file(), immutable,
                          MultipleJavaFiles(descriptor->file(), immutable));
}

string ClassNameResolver::GetClassName(const EnumDescriptor* descriptor,
                                       bool immutable) {
  return GetClassFullName(ClassNameWithoutPackage(descriptor, immutable),
                          descriptor->file(), immutable,
                          MultipleJavaFiles(descriptor->file(), immutable));
}

string ClassNameResolver::GetClassName(const ServiceDescriptor* descriptor,
                                       bool immutable) {
  return GetClassFullName(ClassNameWithoutPackage(descriptor, immutable),
                          descriptor->file(), immutable,
                          MultipleJavaFiles(descriptor->file(), immutable));
}

// Get the Java Class style full name of a message.
string ClassNameResolver::GetJavaClassFullName(
    const string& name_without_package,
    const FileDescriptor* file,
    bool immutable) {
  string result;
  if (MultipleJavaFiles(file, immutable)) {
    result = FileJavaPackage(file, immutable);
    if (!result.empty()) result += '.';
  } else {
    result = GetClassName(file, immutable);
    if (!result.empty()) result += '$';
  }
  result += StringReplace(name_without_package, ".", "$", true);
  return result;
}

string ClassNameResolver::GetExtensionIdentifierName(
    const FieldDescriptor* descriptor, bool immutable) {
  return GetClassName(descriptor->containing_type(), immutable) + "." +
         descriptor->name();
}


string ClassNameResolver::GetJavaImmutableClassName(
    const Descriptor* descriptor) {
  return GetJavaClassFullName(
      ClassNameWithoutPackage(descriptor, true),
      descriptor->file(), true);
}


}  // namespace java
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
