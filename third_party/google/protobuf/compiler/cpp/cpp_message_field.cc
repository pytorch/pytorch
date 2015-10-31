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

#include <google/protobuf/compiler/cpp/cpp_message_field.h>
#include <google/protobuf/compiler/cpp/cpp_helpers.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace cpp {

namespace {

void SetMessageVariables(const FieldDescriptor* descriptor,
                         map<string, string>* variables,
                         const Options& options) {
  SetCommonFieldVariables(descriptor, variables, options);
  (*variables)["type"] = FieldMessageTypeName(descriptor);
  if (descriptor->options().weak() || !descriptor->containing_oneof()) {
    (*variables)["non_null_ptr_to_name"] =
        StrCat("this->", (*variables)["name"], "_");
  }
  (*variables)["stream_writer"] = (*variables)["declared_type"] +
      (HasFastArraySerialization(descriptor->message_type()->file()) ?
       "MaybeToArray" :
       "");
  // NOTE: Escaped here to unblock proto1->proto2 migration.
  // TODO(liujisi): Extend this to apply for other conflicting methods.
  (*variables)["release_name"] =
      SafeFunctionName(descriptor->containing_type(),
                       descriptor, "release_");
  (*variables)["full_name"] = descriptor->full_name();
  if (options.proto_h && IsFieldDependent(descriptor)) {
    (*variables)["dependent_type"] = "T::" + DependentTypeName(descriptor);
    (*variables)["dependent_typename"] =
        "typename T::" + DependentTypeName(descriptor);
  } else {
    (*variables)["dependent_type"] = FieldMessageTypeName(descriptor);
    (*variables)["dependent_typename"] = FieldMessageTypeName(descriptor);
  }
}

}  // namespace

// ===================================================================

MessageFieldGenerator::
MessageFieldGenerator(const FieldDescriptor* descriptor,
                      const Options& options)
  : descriptor_(descriptor),
    dependent_field_(options.proto_h && IsFieldDependent(descriptor)) {
  SetMessageVariables(descriptor, &variables_, options);
}

MessageFieldGenerator::~MessageFieldGenerator() {}

void MessageFieldGenerator::
GeneratePrivateMembers(io::Printer* printer) const {
  printer->Print(variables_, "$type$* $name$_;\n");
}

void MessageFieldGenerator::
GenerateGetterDeclaration(io::Printer* printer) const {
  printer->Print(variables_,
      "const $type$& $name$() const$deprecation$;\n");
}

void MessageFieldGenerator::
GenerateDependentAccessorDeclarations(io::Printer* printer) const {
  if (!dependent_field_) {
    return;
  }
  // Arena manipulation code is out-of-line in the derived message class.
  printer->Print(variables_,
    "$type$* mutable_$name$()$deprecation$;\n"
    "$type$* $release_name$()$deprecation$;\n"
    "void set_allocated_$name$($type$* $name$)$deprecation$;\n");
}

void MessageFieldGenerator::
GenerateAccessorDeclarations(io::Printer* printer) const {
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables_,
       "private:\n"
       "void _slow_mutable_$name$()$deprecation$;\n");
    if (SupportsArenas(descriptor_->message_type())) {
      printer->Print(variables_,
       "void _slow_set_allocated_$name$(\n"
       "    ::google::protobuf::Arena* message_arena, $type$** $name$)$deprecation$;\n");
    }
    printer->Print(variables_,
       "$type$* _slow_$release_name$()$deprecation$;\n"
       "public:\n");
  }
  GenerateGetterDeclaration(printer);
  if (!dependent_field_) {
    printer->Print(variables_,
      "$type$* mutable_$name$()$deprecation$;\n"
      "$type$* $release_name$()$deprecation$;\n"
      "void set_allocated_$name$($type$* $name$)$deprecation$;\n");
  }
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables_,
      "$type$* unsafe_arena_release_$name$()$deprecation$;\n"
      "void unsafe_arena_set_allocated_$name$(\n"
      "    $type$* $name$)$deprecation$;\n");
  }
}

void MessageFieldGenerator::GenerateNonInlineAccessorDefinitions(
    io::Printer* printer) const {
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables_,
      "void $classname$::_slow_mutable_$name$() {\n");
      if (SupportsArenas(descriptor_->message_type())) {
        printer->Print(variables_,
          "  $name$_ = ::google::protobuf::Arena::CreateMessage< $type$ >(\n"
          "      GetArenaNoVirtual());\n");
      } else {
        printer->Print(variables_,
          "  $name$_ = ::google::protobuf::Arena::Create< $type$ >(\n"
          "      GetArenaNoVirtual());\n");
      }
    printer->Print(variables_,
      "}\n"
      "$type$* $classname$::_slow_$release_name$() {\n"
      "  if ($name$_ == NULL) {\n"
      "    return NULL;\n"
      "  } else {\n"
      "    $type$* temp = new $type$;\n"
      "    temp->MergeFrom(*$name$_);\n"
      "    $name$_ = NULL;\n"
      "    return temp;\n"
      "  }\n"
      "}\n"
      "$type$* $classname$::unsafe_arena_release_$name$() {\n"
      "  $clear_hasbit$\n"
      "  $type$* temp = $name$_;\n"
      "  $name$_ = NULL;\n"
      "  return temp;\n"
      "}\n");
    if (SupportsArenas(descriptor_->message_type())) {
      // NOTE: the same logic is mirrored in weak_message_field.cc. Any
      // arena-related semantics changes should be made in both places.
      printer->Print(variables_,
          "void $classname$::_slow_set_allocated_$name$(\n"
          "    ::google::protobuf::Arena* message_arena, $type$** $name$) {\n"
          "    if (message_arena != NULL && \n"
          "        ::google::protobuf::Arena::GetArena(*$name$) == NULL) {\n"
          "      message_arena->Own(*$name$);\n"
          "    } else if (message_arena !=\n"
          "               ::google::protobuf::Arena::GetArena(*$name$)) {\n"
          "      $type$* new_$name$ = \n"
          "            ::google::protobuf::Arena::CreateMessage< $type$ >(\n"
          "            message_arena);\n"
          "      new_$name$->CopyFrom(**$name$);\n"
          "      *$name$ = new_$name$;\n"
          "    }\n"
          "}\n");
    }
    printer->Print(variables_,
      "void $classname$::unsafe_arena_set_allocated_$name$(\n"
      "    $type$* $name$) {\n"
      // If we're not on an arena, free whatever we were holding before.
      // (If we are on arena, we can just forget the earlier pointer.)
      "  if (GetArenaNoVirtual() == NULL) {\n"
      "    delete $name$_;\n"
      "  }\n"
      "  $name$_ = $name$;\n"
      "  if ($name$) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_unsafe_arena_set_allocated"
      ":$full_name$)\n"
      "}\n");
  }
}

void MessageFieldGenerator::
GenerateDependentInlineAccessorDefinitions(io::Printer* printer) const {
  if (!dependent_field_) {
    return;
  }

  map<string, string> variables(variables_);
  // For the CRTP base class, all mutation methods are dependent, and so
  // they must be in the header.
  variables["dependent_classname"] =
      DependentBaseClassTemplateName(descriptor_->containing_type()) + "<T>";
  variables["this_message"] = DependentBaseDownCast();
  if (!variables["set_hasbit"].empty()) {
    variables["set_hasbit"] =
        variables["this_message"] + variables["set_hasbit"];
  }
  if (!variables["clear_hasbit"].empty()) {
    variables["clear_hasbit"] =
        variables["this_message"] + variables["clear_hasbit"];
  }

  if (SupportsArenas(descriptor_)) {
    printer->Print(variables,
      "template <class T>\n"
      "inline $type$* $dependent_classname$::mutable_$name$() {\n"
      "  $set_hasbit$\n"
      "  $dependent_typename$*& $name$_ = $this_message$$name$_;\n"
      "  if ($name$_ == NULL) {\n"
      "    $this_message$_slow_mutable_$name$();\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_;\n"
      "}\n"
      "template <class T>\n"
      "inline $type$* $dependent_classname$::$release_name$() {\n"
      "  $dependent_typename$*& $name$_ = $this_message$$name$_;\n"
      "  $clear_hasbit$\n"
      "  if ($this_message$GetArenaNoVirtual() != NULL) {\n"
      "    return $this_message$_slow_$release_name$();\n"
      "  } else {\n"
      "    $dependent_typename$* temp = $name$_;\n"
      "    $name$_ = NULL;\n"
      "    return temp;\n"
      "  }\n"
      "}\n"
      "template <class T>\n"
      "inline void $dependent_classname$::"
      "set_allocated_$name$($type$* $name$) {\n"
      "  ::google::protobuf::Arena* message_arena = $this_message$GetArenaNoVirtual();\n"
      "  $dependent_typename$*& $name$_ = $this_message$$name$_;\n"
      "  if (message_arena == NULL) {\n"
      "    delete $name$_;\n"
      "  }\n"
      "  if ($name$ != NULL) {\n");
    if (SupportsArenas(descriptor_->message_type())) {
      // If we're on an arena and the incoming message is not, simply Own() it
      // rather than copy to the arena -- either way we need a heap dealloc,
      // so we might as well defer it. Otherwise, if incoming message is on a
      // different ownership domain (specific arena, or the heap) than we are,
      // copy to our arena (or heap, as the case may be).
      printer->Print(variables,
        "    $this_message$_slow_set_allocated_$name$(message_arena, "
        "&$name$);\n");
    } else {
      printer->Print(variables,
        "    if (message_arena != NULL) {\n"
        "      message_arena->Own($name$);\n"
        "    }\n");
    }
    printer->Print(variables,
      "  }\n"
      "  $name$_ = $name$;\n"
      "  if ($name$) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      // TODO(dlj): move insertion points to message class.
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  } else {
    printer->Print(variables,
      "template <class T>\n"
      "inline $type$* $dependent_classname$::mutable_$name$() {\n"
      "  $set_hasbit$\n"
      "  $dependent_typename$*& $name$_ = $this_message$$name$_;\n"
      "  if ($name$_ == NULL) {\n"
      "    $name$_ = new $dependent_typename$;\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_;\n"
      "}\n"
      "template <class T>\n"
      "inline $type$* $dependent_classname$::$release_name$() {\n"
      "  $clear_hasbit$\n"
      "  $dependent_typename$*& $name$_ = $this_message$$name$_;\n"
      "  $dependent_typename$* temp = $name$_;\n"
      "  $name$_ = NULL;\n"
      "  return temp;\n"
      "}\n"
      "template <class T>\n"
      "inline void $dependent_classname$::"
      "set_allocated_$name$($type$* $name$) {\n"
      "  $dependent_typename$*& $name$_ = $this_message$$name$_;\n"
      "  delete $name$_;\n");

    if (SupportsArenas(descriptor_->message_type())) {
      printer->Print(variables,
      "  if ($name$ != NULL && static_cast< $dependent_typename$* >($name$)"
      "->GetArena() != NULL) {\n"
      "    $dependent_typename$* new_$name$ = new $dependent_typename$;\n"
      "    new_$name$->CopyFrom(*$name$);\n"
      "    $name$ = new_$name$;\n"
      "  }\n");
    }

    printer->Print(variables,
      "  $name$_ = $name$;\n"
      "  if ($name$) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  }
}

void MessageFieldGenerator::
GenerateInlineAccessorDefinitions(io::Printer* printer,
                                  bool is_inline) const {
  map<string, string> variables(variables_);
  variables["inline"] = is_inline ? "inline " : "";
  printer->Print(variables,
    "$inline$const $type$& $classname$::$name$() const {\n"
    "  // @@protoc_insertion_point(field_get:$full_name$)\n");

  PrintHandlingOptionalStaticInitializers(
    variables, descriptor_->file(), printer,
    // With static initializers.
    "  return $name$_ != NULL ? *$name$_ : *default_instance_->$name$_;\n",
    // Without.
    "  return $name$_ != NULL ? *$name$_ : *default_instance().$name$_;\n");
  printer->Print(variables, "}\n");

  if (dependent_field_) {
    return;
  }

  if (SupportsArenas(descriptor_)) {
    printer->Print(variables,
      "$inline$"
      "$type$* $classname$::mutable_$name$() {\n"
      "  $set_hasbit$\n"
      "  if ($name$_ == NULL) {\n"
      "    _slow_mutable_$name$();\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_;\n"
      "}\n"
      "$inline$"
      "$type$* $classname$::$release_name$() {\n"
      "  $clear_hasbit$\n"
      "  if (GetArenaNoVirtual() != NULL) {\n"
      "    return _slow_$release_name$();\n"
      "  } else {\n"
      "    $type$* temp = $name$_;\n"
      "    $name$_ = NULL;\n"
      "    return temp;\n"
      "  }\n"
      "}\n"
      "$inline$ "
      "void $classname$::set_allocated_$name$($type$* $name$) {\n"
      "  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();\n"
      "  if (message_arena == NULL) {\n"
      "    delete $name$_;\n"
      "  }\n"
      "  if ($name$ != NULL) {\n");
    if (SupportsArenas(descriptor_->message_type())) {
      // If we're on an arena and the incoming message is not, simply Own() it
      // rather than copy to the arena -- either way we need a heap dealloc,
      // so we might as well defer it. Otherwise, if incoming message is on a
      // different ownership domain (specific arena, or the heap) than we are,
      // copy to our arena (or heap, as the case may be).
      printer->Print(variables,
        "    _slow_set_allocated_$name$(message_arena, &$name$);\n");
    } else {
      printer->Print(variables,
        "    if (message_arena != NULL) {\n"
        "      message_arena->Own($name$);\n"
        "    }\n");
    }
    printer->Print(variables,
      "  }\n"
      "  $name$_ = $name$;\n"
      "  if ($name$) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  } else {
    printer->Print(variables,
      "$inline$"
      "$type$* $classname$::mutable_$name$() {\n"
      "  $set_hasbit$\n"
      "  if ($name$_ == NULL) {\n"
      "    $name$_ = new $type$;\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_;\n"
      "}\n"
      "$inline$"
      "$type$* $classname$::$release_name$() {\n"
      "  $clear_hasbit$\n"
      "  $type$* temp = $name$_;\n"
      "  $name$_ = NULL;\n"
      "  return temp;\n"
      "}\n"
      "$inline$"
      "void $classname$::set_allocated_$name$($type$* $name$) {\n"
      "  delete $name$_;\n");

    if (SupportsArenas(descriptor_->message_type())) {
      printer->Print(variables,
      "  if ($name$ != NULL && $name$->GetArena() != NULL) {\n"
      "    $type$* new_$name$ = new $type$;\n"
      "    new_$name$->CopyFrom(*$name$);\n"
      "    $name$ = new_$name$;\n"
      "  }\n");
    }

    printer->Print(variables,
      "  $name$_ = $name$;\n"
      "  if ($name$) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  }
}

void MessageFieldGenerator::
GenerateClearingCode(io::Printer* printer) const {
  map<string, string> variables(variables_);
  variables["this_message"] = dependent_field_ ? DependentBaseDownCast() : "";
  if (!HasFieldPresence(descriptor_->file())) {
    // If we don't have has-bits, message presence is indicated only by ptr !=
    // NULL. Thus on clear, we need to delete the object.
    printer->Print(variables,
      "if ($this_message$GetArenaNoVirtual() == NULL && "
      "$this_message$$name$_ != NULL) delete $this_message$$name$_;\n"
      "$this_message$$name$_ = NULL;\n");
  } else {
    printer->Print(variables,
      "if ($this_message$$name$_ != NULL) $this_message$$name$_->"
      "$dependent_type$::Clear();\n");
  }
}

void MessageFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "mutable_$name$()->$type$::MergeFrom(from.$name$());\n");
}

void MessageFieldGenerator::
GenerateSwappingCode(io::Printer* printer) const {
  printer->Print(variables_, "std::swap($name$_, other->$name$_);\n");
}

void MessageFieldGenerator::
GenerateConstructorCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_ = NULL;\n");
}

void MessageFieldGenerator::
GenerateMergeFromCodedStream(io::Printer* printer) const {
  if (descriptor_->type() == FieldDescriptor::TYPE_MESSAGE) {
    printer->Print(variables_,
      "DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(\n"
      "     input, mutable_$name$()));\n");
  } else {
    printer->Print(variables_,
      "DO_(::google::protobuf::internal::WireFormatLite::ReadGroupNoVirtual(\n"
      "      $number$, input, mutable_$name$()));\n");
  }
}

void MessageFieldGenerator::
GenerateSerializeWithCachedSizes(io::Printer* printer) const {
  printer->Print(variables_,
    "::google::protobuf::internal::WireFormatLite::Write$stream_writer$(\n"
    "  $number$, *$non_null_ptr_to_name$, output);\n");
}

void MessageFieldGenerator::
GenerateSerializeWithCachedSizesToArray(io::Printer* printer) const {
  printer->Print(variables_,
    "target = ::google::protobuf::internal::WireFormatLite::\n"
    "  Write$declared_type$NoVirtualToArray(\n"
    "    $number$, *$non_null_ptr_to_name$, target);\n");
}

void MessageFieldGenerator::
GenerateByteSize(io::Printer* printer) const {
  printer->Print(variables_,
    "total_size += $tag_size$ +\n"
    "  ::google::protobuf::internal::WireFormatLite::$declared_type$SizeNoVirtual(\n"
    "    *$non_null_ptr_to_name$);\n");
}

// ===================================================================

MessageOneofFieldGenerator::
MessageOneofFieldGenerator(const FieldDescriptor* descriptor,
                           const Options& options)
  : MessageFieldGenerator(descriptor, options),
    dependent_base_(options.proto_h) {
  SetCommonOneofFieldVariables(descriptor, &variables_);
}

MessageOneofFieldGenerator::~MessageOneofFieldGenerator() {}


void MessageOneofFieldGenerator::
GenerateDependentAccessorDeclarations(io::Printer* printer) const {
  // Oneof field getters must be dependent as they call default_instance().
  // Otherwise, the logic is the same as MessageFields.
  if (!dependent_field_) {
    return;
  }
  printer->Print(variables_,
      "const $type$& $name$() const$deprecation$;\n");
  MessageFieldGenerator::GenerateDependentAccessorDeclarations(printer);
}

void MessageOneofFieldGenerator::
GenerateGetterDeclaration(io::Printer* printer) const {
  // Oneof field getters must be dependent as they call default_instance().
  // Unlike MessageField, this means there is no (non-dependent) getter to
  // generate.
  if (dependent_field_) {
    return;
  }
  printer->Print(variables_,
      "const $type$& $name$() const$deprecation$;\n");
}

void MessageOneofFieldGenerator::
GenerateDependentInlineAccessorDefinitions(io::Printer* printer) const {
  // For the CRTP base class, all mutation methods are dependent, and so
  // they must be in the header.
  if (!dependent_base_) {
    return;
  }
  map<string, string> variables(variables_);
  variables["inline"] = "inline ";
  variables["dependent_classname"] =
      DependentBaseClassTemplateName(descriptor_->containing_type()) + "<T>";
  variables["this_message"] = "reinterpret_cast<T*>(this)->";
  // Const message access is needed for the dependent getter.
  variables["this_const_message"] = "reinterpret_cast<const T*>(this)->";
  variables["tmpl"] = "template <class T>\n";
  variables["field_member"] = variables["this_message"] +
                              variables["oneof_prefix"] + variables["name"] +
                              "_";
  InternalGenerateInlineAccessorDefinitions(variables, printer);
}

void MessageOneofFieldGenerator::
GenerateInlineAccessorDefinitions(io::Printer* printer,
                                  bool is_inline) const {
  if (dependent_base_) {
    return;
  }
  map<string, string> variables(variables_);
  variables["inline"] = is_inline ? "inline " : "";
  variables["dependent_classname"] = variables["classname"];
  variables["this_message"] = "";
  variables["this_const_message"] = "";
  variables["tmpl"] = "";
  variables["field_member"] =
      variables["oneof_prefix"] + variables["name"] + "_";
  variables["dependent_type"] = variables["type"];
  InternalGenerateInlineAccessorDefinitions(variables, printer);
}

void MessageOneofFieldGenerator::
GenerateNonInlineAccessorDefinitions(io::Printer* printer) const {
  map<string, string> variables(variables_);
  variables["field_member"] =
      variables["oneof_prefix"] + variables["name"] + "_";

  //printer->Print(variables,
}

void MessageOneofFieldGenerator::
InternalGenerateInlineAccessorDefinitions(const map<string, string>& variables,
                                          io::Printer* printer) const {
  printer->Print(variables,
    "$tmpl$"
    "$inline$ "
    "const $type$& $dependent_classname$::$name$() const {\n"
    "  // @@protoc_insertion_point(field_get:$full_name$)\n"
    "  return $this_const_message$has_$name$()\n"
    "      ? *$this_const_message$$oneof_prefix$$name$_\n"
    "      : $dependent_type$::default_instance();\n"
    "}\n");

  if (SupportsArenas(descriptor_)) {
    printer->Print(variables,
      "$tmpl$"
      "$inline$"
      "$type$* $dependent_classname$::mutable_$name$() {\n"
      "  if (!$this_message$has_$name$()) {\n"
      "    $this_message$clear_$oneof_name$();\n"
      "    $this_message$set_has_$name$();\n");
    if (SupportsArenas(descriptor_->message_type())) {
      printer->Print(variables,
         "    $field_member$ = \n"
         "      ::google::protobuf::Arena::CreateMessage< $dependent_typename$ >(\n"
         "      $this_message$GetArenaNoVirtual());\n");
    } else {
      printer->Print(variables,
         "    $this_message$$oneof_prefix$$name$_ = \n"
         "      ::google::protobuf::Arena::Create< $dependent_typename$ >(\n"
         "      $this_message$GetArenaNoVirtual());\n");
    }
    printer->Print(variables,
      "  }\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $field_member$;\n"
      "}\n"
      "$tmpl$"
      "$inline$"
      "$type$* $dependent_classname$::$release_name$() {\n"
      "  if ($this_message$has_$name$()) {\n"
      "    $this_message$clear_has_$oneof_name$();\n"
      "    if ($this_message$GetArenaNoVirtual() != NULL) {\n"
      // N.B.: safe to use the underlying field pointer here because we are sure
      // that it is non-NULL (because has_$name$() returned true).
      "      $dependent_typename$* temp = new $dependent_typename$;\n"
      "      temp->MergeFrom(*$field_member$);\n"
      "      $field_member$ = NULL;\n"
      "      return temp;\n"
      "    } else {\n"
      "      $dependent_typename$* temp = $field_member$;\n"
      "      $field_member$ = NULL;\n"
      "      return temp;\n"
      "    }\n"
      "  } else {\n"
      "    return NULL;\n"
      "  }\n"
      "}\n"
      "$tmpl$"
      "$inline$"
      "void $dependent_classname$::"
      "set_allocated_$name$($type$* $name$) {\n"
      "  $this_message$clear_$oneof_name$();\n"
      "  if ($name$) {\n");

    if (SupportsArenas(descriptor_->message_type())) {
      printer->Print(variables,
        // If incoming message is on the heap and we are on an arena, just Own()
        // it (see above). If it's on a different arena than we are or one of us
        // is on the heap, we make a copy to our arena/heap.
        "    if ($this_message$GetArenaNoVirtual() != NULL &&\n"
        "        ::google::protobuf::Arena::GetArena($name$) == NULL) {\n"
        "      $this_message$GetArenaNoVirtual()->Own($name$);\n"
        "    } else if ($this_message$GetArenaNoVirtual() !=\n"
        "               ::google::protobuf::Arena::GetArena($name$)) {\n"
        "      $dependent_typename$* new_$name$ = \n"
        "          ::google::protobuf::Arena::CreateMessage< $dependent_typename$ >(\n"
        "          $this_message$GetArenaNoVirtual());\n"
        "      new_$name$->CopyFrom(*$name$);\n"
        "      $name$ = new_$name$;\n"
        "    }\n");
    } else {
      printer->Print(variables,
        "    if ($this_message$GetArenaNoVirtual() != NULL) {\n"
        "      $this_message$GetArenaNoVirtual()->Own($name$);\n"
        "    }\n");
    }

    printer->Print(variables,
      "    $this_message$set_has_$name$();\n"
      "    $field_member$ = $name$;\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n"
      "$inline$ $type$* $classname$::unsafe_arena_release_$name$() {\n"
      "  if (has_$name$()) {\n"
      "    clear_has_$oneof_name$();\n"
      "    $type$* temp = $oneof_prefix$$name$_;\n"
      "    $oneof_prefix$$name$_ = NULL;\n"
      "    return temp;\n"
      "  } else {\n"
      "    return NULL;\n"
      "  }\n"
      "}\n"
      "$inline$ void $classname$::unsafe_arena_set_allocated_$name$"
      "($type$* $name$) {\n"
      // We rely on the oneof clear method to free the earlier contents of this
      // oneof. We can directly use the pointer we're given to set the new
      // value.
      "  clear_$oneof_name$();\n"
      "  if ($name$) {\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_ = $name$;\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:"
      "$full_name$)\n"
      "}\n");
  } else {
    printer->Print(variables,
      "$tmpl$"
      "$inline$"
      "$type$* $dependent_classname$::mutable_$name$() {\n"
      "  if (!$this_message$has_$name$()) {\n"
      "    $this_message$clear_$oneof_name$();\n"
      "    $this_message$set_has_$name$();\n"
      "    $field_member$ = new $dependent_typename$;\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $field_member$;\n"
      "}\n"
      "$tmpl$"
      "$inline$"
      "$type$* $dependent_classname$::$release_name$() {\n"
      "  if ($this_message$has_$name$()) {\n"
      "    $this_message$clear_has_$oneof_name$();\n"
      "    $dependent_typename$* temp = $field_member$;\n"
      "    $field_member$ = NULL;\n"
      "    return temp;\n"
      "  } else {\n"
      "    return NULL;\n"
      "  }\n"
      "}\n"
      "$tmpl$"
      "$inline$"
      "void $dependent_classname$::"
      "set_allocated_$name$($type$* $name$) {\n"
      "  $this_message$clear_$oneof_name$();\n"
      "  if ($name$) {\n");
    if (SupportsArenas(descriptor_->message_type())) {
      printer->Print(variables,
        "    if (static_cast< $dependent_typename$*>($name$)->"
        "GetArena() != NULL) {\n"
        "      $dependent_typename$* new_$name$ = new $dependent_typename$;\n"
        "      new_$name$->CopyFrom(*$name$);\n"
        "      $name$ = new_$name$;\n"
        "    }\n");
    }
    printer->Print(variables,
      "    $this_message$set_has_$name$();\n"
      "    $field_member$ = $name$;\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  }
}

void MessageOneofFieldGenerator::
GenerateClearingCode(io::Printer* printer) const {
  map<string, string> variables(variables_);
  variables["this_message"] = dependent_field_ ? DependentBaseDownCast() : "";
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables,
      "if ($this_message$GetArenaNoVirtual() == NULL) {\n"
      "  delete $this_message$$oneof_prefix$$name$_;\n"
      "}\n");
  } else {
    printer->Print(variables,
      "delete $this_message$$oneof_prefix$$name$_;\n");
  }
}

void MessageOneofFieldGenerator::
GenerateSwappingCode(io::Printer* printer) const {
  // Don't print any swapping code. Swapping the union will swap this field.
}

void MessageOneofFieldGenerator::
GenerateConstructorCode(io::Printer* printer) const {
  // Don't print any constructor code. The field is in a union. We allocate
  // space only when this field is used.
}

// ===================================================================

RepeatedMessageFieldGenerator::
RepeatedMessageFieldGenerator(const FieldDescriptor* descriptor,
                              const Options& options)
  : descriptor_(descriptor),
    dependent_field_(options.proto_h && IsFieldDependent(descriptor)),
    dependent_getter_(dependent_field_ && options.safe_boundary_check) {
  SetMessageVariables(descriptor, &variables_, options);
}

RepeatedMessageFieldGenerator::~RepeatedMessageFieldGenerator() {}

void RepeatedMessageFieldGenerator::
GeneratePrivateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "::google::protobuf::RepeatedPtrField< $type$ > $name$_;\n");
}

void RepeatedMessageFieldGenerator::
InternalGenerateTypeDependentAccessorDeclarations(io::Printer* printer) const {
  printer->Print(variables_,
    "$type$* mutable_$name$(int index)$deprecation$;\n"
    "$type$* add_$name$()$deprecation$;\n");
  if (dependent_getter_) {
    printer->Print(variables_,
      "const ::google::protobuf::RepeatedPtrField< $type$ >&\n"
      "    $name$() const$deprecation$;\n");
  }
  printer->Print(variables_,
    "::google::protobuf::RepeatedPtrField< $type$ >*\n"
    "    mutable_$name$()$deprecation$;\n");
}

void RepeatedMessageFieldGenerator::
GenerateDependentAccessorDeclarations(io::Printer* printer) const {
  if (dependent_getter_) {
    printer->Print(variables_,
      "const $type$& $name$(int index) const$deprecation$;\n");
  }
  if (dependent_field_) {
    InternalGenerateTypeDependentAccessorDeclarations(printer);
  }
}

void RepeatedMessageFieldGenerator::
GenerateAccessorDeclarations(io::Printer* printer) const {
  if (!dependent_getter_) {
    printer->Print(variables_,
      "const $type$& $name$(int index) const$deprecation$;\n");
  }
  if (!dependent_field_) {
    InternalGenerateTypeDependentAccessorDeclarations(printer);
  }
  if (!dependent_getter_) {
    printer->Print(variables_,
      "const ::google::protobuf::RepeatedPtrField< $type$ >&\n"
      "    $name$() const$deprecation$;\n");
  }
}

void RepeatedMessageFieldGenerator::
GenerateDependentInlineAccessorDefinitions(io::Printer* printer) const {
  if (!dependent_field_) {
    return;
  }
  map<string, string> variables(variables_);
  // For the CRTP base class, all mutation methods are dependent, and so
  // they must be in the header.
  variables["dependent_classname"] =
      DependentBaseClassTemplateName(descriptor_->containing_type()) + "<T>";
  variables["this_message"] = DependentBaseDownCast();
  variables["this_const_message"] = DependentBaseConstDownCast();

  if (dependent_getter_) {
    printer->Print(variables,
      "template <class T>\n"
      "inline const $type$& $dependent_classname$::$name$(int index) const {\n"
      "  // @@protoc_insertion_point(field_get:$full_name$)\n"
      "  return $this_const_message$$name$_.$cppget$(index);\n"
      "}\n");
  }

  // Generate per-element accessors:
  printer->Print(variables,
    "template <class T>\n"
    "inline $type$* $dependent_classname$::mutable_$name$(int index) {\n"
    // TODO(dlj): move insertion points
    "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
    "  return $this_message$$name$_.Mutable(index);\n"
    "}\n"
    "template <class T>\n"
    "inline $type$* $dependent_classname$::add_$name$() {\n"
    "  // @@protoc_insertion_point(field_add:$full_name$)\n"
    "  return $this_message$$name$_.Add();\n"
    "}\n");


  if (dependent_getter_) {
    printer->Print(variables,
      "template <class T>\n"
      "inline const ::google::protobuf::RepeatedPtrField< $type$ >&\n"
      "$dependent_classname$::$name$() const {\n"
      "  // @@protoc_insertion_point(field_list:$full_name$)\n"
      "  return $this_const_message$$name$_;\n"
      "}\n");
  }

  // Generate mutable access to the entire list:
  printer->Print(variables,
    "template <class T>\n"
    "inline ::google::protobuf::RepeatedPtrField< $type$ >*\n"
    "$dependent_classname$::mutable_$name$() {\n"
    "  // @@protoc_insertion_point(field_mutable_list:$full_name$)\n"
    "  return &$this_message$$name$_;\n"
    "}\n");
}

void RepeatedMessageFieldGenerator::
GenerateInlineAccessorDefinitions(io::Printer* printer,
                                  bool is_inline) const {
  map<string, string> variables(variables_);
  variables["inline"] = is_inline ? "inline " : "";

  if (!dependent_getter_) {
    printer->Print(variables,
      "$inline$"
      "const $type$& $classname$::$name$(int index) const {\n"
      "  // @@protoc_insertion_point(field_get:$full_name$)\n"
      "  return $name$_.$cppget$(index);\n"
      "}\n");
  }

  if (!dependent_field_) {
    printer->Print(variables,
      "$inline$"
      "$type$* $classname$::mutable_$name$(int index) {\n"
      // TODO(dlj): move insertion points
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_.Mutable(index);\n"
      "}\n"
      "$inline$"
      "$type$* $classname$::add_$name$() {\n"
      "  // @@protoc_insertion_point(field_add:$full_name$)\n"
      "  return $name$_.Add();\n"
      "}\n");
  }


  if (!dependent_field_) {
    printer->Print(variables,
      "$inline$"
      "::google::protobuf::RepeatedPtrField< $type$ >*\n"
      "$classname$::mutable_$name$() {\n"
      "  // @@protoc_insertion_point(field_mutable_list:$full_name$)\n"
      "  return &$name$_;\n"
      "}\n");
  }
  if (!dependent_getter_) {
    printer->Print(variables,
      "$inline$"
      "const ::google::protobuf::RepeatedPtrField< $type$ >&\n"
      "$classname$::$name$() const {\n"
      "  // @@protoc_insertion_point(field_list:$full_name$)\n"
      "  return $name$_;\n"
      "}\n");
  }
}

void RepeatedMessageFieldGenerator::
GenerateClearingCode(io::Printer* printer) const {
  map<string, string> variables(variables_);
  variables["this_message"] = dependent_field_ ? DependentBaseDownCast() : "";
  printer->Print(variables, "$this_message$$name$_.Clear();\n");
}

void RepeatedMessageFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.MergeFrom(from.$name$_);\n");
}

void RepeatedMessageFieldGenerator::
GenerateSwappingCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.UnsafeArenaSwap(&other->$name$_);\n");
}

void RepeatedMessageFieldGenerator::
GenerateConstructorCode(io::Printer* printer) const {
  // Not needed for repeated fields.
}

void RepeatedMessageFieldGenerator::
GenerateMergeFromCodedStream(io::Printer* printer) const {
  if (descriptor_->type() == FieldDescriptor::TYPE_MESSAGE) {
    printer->Print(variables_,
      "DO_(::google::protobuf::internal::WireFormatLite::"
      "ReadMessageNoVirtualNoRecursionDepth(\n"
      "      input, add_$name$()));\n");
  } else {
    printer->Print(variables_,
      "DO_(::google::protobuf::internal::WireFormatLite::"
      "ReadGroupNoVirtualNoRecursionDepth(\n"
      "      $number$, input, add_$name$()));\n");
  }
}

void RepeatedMessageFieldGenerator::
GenerateSerializeWithCachedSizes(io::Printer* printer) const {
  printer->Print(variables_,
    "for (unsigned int i = 0, n = this->$name$_size(); i < n; i++) {\n"
    "  ::google::protobuf::internal::WireFormatLite::Write$stream_writer$(\n"
    "    $number$, this->$name$(i), output);\n"
    "}\n");
}

void RepeatedMessageFieldGenerator::
GenerateSerializeWithCachedSizesToArray(io::Printer* printer) const {
  printer->Print(variables_,
    "for (unsigned int i = 0, n = this->$name$_size(); i < n; i++) {\n"
    "  target = ::google::protobuf::internal::WireFormatLite::\n"
    "    Write$declared_type$NoVirtualToArray(\n"
    "      $number$, this->$name$(i), target);\n"
    "}\n");
}

void RepeatedMessageFieldGenerator::
GenerateByteSize(io::Printer* printer) const {
  printer->Print(variables_,
    "total_size += $tag_size$ * this->$name$_size();\n"
    "for (int i = 0; i < this->$name$_size(); i++) {\n"
    "  total_size +=\n"
    "    ::google::protobuf::internal::WireFormatLite::$declared_type$SizeNoVirtual(\n"
    "      this->$name$(i));\n"
    "}\n");
}

}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
