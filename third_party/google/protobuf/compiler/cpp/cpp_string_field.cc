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

#include <google/protobuf/compiler/cpp/cpp_string_field.h>
#include <google/protobuf/compiler/cpp/cpp_helpers.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace cpp {

namespace {

void SetStringVariables(const FieldDescriptor* descriptor,
                        map<string, string>* variables,
                        const Options& options) {
  SetCommonFieldVariables(descriptor, variables, options);
  (*variables)["default"] = DefaultValue(descriptor);
  (*variables)["default_length"] =
      SimpleItoa(descriptor->default_value_string().length());
  string default_variable_string =
      descriptor->default_value_string().empty()
          ? "&::google::protobuf::internal::GetEmptyStringAlreadyInited()"
          : "_default_" + FieldName(descriptor) + "_";
  (*variables)["default_variable"] = default_variable_string;
  (*variables)["default_value_init"] =
      descriptor->default_value_string().empty()
      ? "" : "*" + default_variable_string;
  (*variables)["pointer_type"] =
      descriptor->type() == FieldDescriptor::TYPE_BYTES ? "void" : "char";
  // NOTE: Escaped here to unblock proto1->proto2 migration.
  // TODO(liujisi): Extend this to apply for other conflicting methods.
  (*variables)["release_name"] =
      SafeFunctionName(descriptor->containing_type(),
                       descriptor, "release_");
  (*variables)["full_name"] = descriptor->full_name();

  (*variables)["string_piece"] = "::std::string";
}

}  // namespace

// ===================================================================

StringFieldGenerator::
StringFieldGenerator(const FieldDescriptor* descriptor,
                     const Options& options)
  : descriptor_(descriptor) {
  SetStringVariables(descriptor, &variables_, options);
}

StringFieldGenerator::~StringFieldGenerator() {}

void StringFieldGenerator::
GeneratePrivateMembers(io::Printer* printer) const {
  // N.B. that we continue to use |ArenaStringPtr| instead of |string*| for
  // string fields, even when SupportArenas(descriptor_) == false. Why?
  // The simple answer is to avoid unmaintainable complexity. The reflection
  // code assumes ArenaStringPtrs. These are *almost* in-memory-compatible with
  // string*, except for the pointer tags and related ownership semantics. We
  // could modify the runtime code to use string* for the not-supporting-arenas
  // case, but this would require a way to detect which type of class was
  // generated (adding overhead and complexity to GeneratedMessageReflection)
  // and littering the runtime code paths with conditionals. It's simpler to
  // stick with this but use lightweight accessors that assume arena == NULL.
  // There should be very little overhead anyway because it's just a tagged
  // pointer in-memory.
  printer->Print(variables_, "::google::protobuf::internal::ArenaStringPtr $name$_;\n");
}

void StringFieldGenerator::
GenerateStaticMembers(io::Printer* printer) const {
  if (!descriptor_->default_value_string().empty()) {
    printer->Print(variables_, "static ::std::string* $default_variable$;\n");
  }
}

void StringFieldGenerator::
GenerateAccessorDeclarations(io::Printer* printer) const {
  // If we're using StringFieldGenerator for a field with a ctype, it's
  // because that ctype isn't actually implemented.  In particular, this is
  // true of ctype=CORD and ctype=STRING_PIECE in the open source release.
  // We aren't releasing Cord because it has too many Google-specific
  // dependencies and we aren't releasing StringPiece because it's hardly
  // useful outside of Google and because it would get confusing to have
  // multiple instances of the StringPiece class in different libraries (PCRE
  // already includes it for their C++ bindings, which came from Google).
  //
  // In any case, we make all the accessors private while still actually
  // using a string to represent the field internally.  This way, we can
  // guarantee that if we do ever implement the ctype, it won't break any
  // existing users who might be -- for whatever reason -- already using .proto
  // files that applied the ctype.  The field can still be accessed via the
  // reflection interface since the reflection interface is independent of
  // the string's underlying representation.

  bool unknown_ctype =
      descriptor_->options().ctype() != EffectiveStringCType(descriptor_);

  if (unknown_ctype) {
    printer->Outdent();
    printer->Print(
      " private:\n"
      "  // Hidden due to unknown ctype option.\n");
    printer->Indent();
  }

  printer->Print(variables_,
    "const ::std::string& $name$() const$deprecation$;\n"
    "void set_$name$(const ::std::string& value)$deprecation$;\n"
    "void set_$name$(const char* value)$deprecation$;\n"
    "void set_$name$(const $pointer_type$* value, size_t size)"
                 "$deprecation$;\n"
    "::std::string* mutable_$name$()$deprecation$;\n"
    "::std::string* $release_name$()$deprecation$;\n"
    "void set_allocated_$name$(::std::string* $name$)$deprecation$;\n");
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables_,
      "::std::string* unsafe_arena_release_$name$()$deprecation$;\n"
      "void unsafe_arena_set_allocated_$name$(\n"
      "    ::std::string* $name$)$deprecation$;\n");
  }


  if (unknown_ctype) {
    printer->Outdent();
    printer->Print(" public:\n");
    printer->Indent();
  }
}

void StringFieldGenerator::
GenerateInlineAccessorDefinitions(io::Printer* printer,
                                  bool is_inline) const {
  map<string, string> variables(variables_);
  variables["inline"] = is_inline ? "inline" : "";
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables,
      "$inline$ const ::std::string& $classname$::$name$() const {\n"
      "  // @@protoc_insertion_point(field_get:$full_name$)\n"
      "  return $name$_.Get($default_variable$);\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const ::std::string& value) {\n"
      "  $set_hasbit$\n"
      "  $name$_.Set($default_variable$, value, GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set:$full_name$)\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const char* value) {\n"
      "  $set_hasbit$\n"
      "  $name$_.Set($default_variable$, $string_piece$(value),\n"
      "              GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
      "}\n"
      "$inline$ "
      "void $classname$::set_$name$(const $pointer_type$* value,\n"
      "    size_t size) {\n"
      "  $set_hasbit$\n"
      "  $name$_.Set($default_variable$, $string_piece$(\n"
      "      reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
      "}\n"
      "$inline$ ::std::string* $classname$::mutable_$name$() {\n"
      "  $set_hasbit$\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_.Mutable($default_variable$, GetArenaNoVirtual());\n"
      "}\n"
      "$inline$ ::std::string* $classname$::$release_name$() {\n"
      "  $clear_hasbit$\n"
      "  return $name$_.Release($default_variable$, GetArenaNoVirtual());\n"
      "}\n"
      "$inline$ ::std::string* $classname$::unsafe_arena_release_$name$() {\n"
      "  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);\n"
      "  $clear_hasbit$\n"
      "  return $name$_.UnsafeArenaRelease($default_variable$,\n"
      "      GetArenaNoVirtual());\n"
      "}\n"
      "$inline$ void $classname$::set_allocated_$name$(::std::string* $name$) {\n"
      "  if ($name$ != NULL) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      "  $name$_.SetAllocated($default_variable$, $name$,\n"
      "      GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n"
      "$inline$ void $classname$::unsafe_arena_set_allocated_$name$(\n"
      "    ::std::string* $name$) {\n"
      "  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);\n"
      "  if ($name$ != NULL) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      "  $name$_.UnsafeArenaSetAllocated($default_variable$,\n"
      "      $name$, GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  } else {
    // No-arena case.
    printer->Print(variables,
      "$inline$ const ::std::string& $classname$::$name$() const {\n"
      "  // @@protoc_insertion_point(field_get:$full_name$)\n"
      "  return $name$_.GetNoArena($default_variable$);\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const ::std::string& value) {\n"
      "  $set_hasbit$\n"
      "  $name$_.SetNoArena($default_variable$, value);\n"
      "  // @@protoc_insertion_point(field_set:$full_name$)\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const char* value) {\n"
      "  $set_hasbit$\n"
      "  $name$_.SetNoArena($default_variable$, $string_piece$(value));\n"
      "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
      "}\n"
      "$inline$ "
      "void $classname$::set_$name$(const $pointer_type$* value, "
      "size_t size) {\n"
      "  $set_hasbit$\n"
      "  $name$_.SetNoArena($default_variable$,\n"
      "      $string_piece$(reinterpret_cast<const char*>(value), size));\n"
      "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
      "}\n"
      "$inline$ ::std::string* $classname$::mutable_$name$() {\n"
      "  $set_hasbit$\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $name$_.MutableNoArena($default_variable$);\n"
      "}\n"
      "$inline$ ::std::string* $classname$::$release_name$() {\n"
      "  $clear_hasbit$\n"
      "  return $name$_.ReleaseNoArena($default_variable$);\n"
      "}\n"
      "$inline$ void $classname$::set_allocated_$name$(::std::string* $name$) {\n"
      "  if ($name$ != NULL) {\n"
      "    $set_hasbit$\n"
      "  } else {\n"
      "    $clear_hasbit$\n"
      "  }\n"
      "  $name$_.SetAllocatedNoArena($default_variable$, $name$);\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  }
}

void StringFieldGenerator::
GenerateNonInlineAccessorDefinitions(io::Printer* printer) const {
  if (!descriptor_->default_value_string().empty()) {
    // Initialized in GenerateDefaultInstanceAllocator.
    printer->Print(variables_,
      "::std::string* $classname$::$default_variable$ = NULL;\n");
  }
}

void StringFieldGenerator::
GenerateClearingCode(io::Printer* printer) const {
  // Two-dimension specialization here: supporting arenas or not, and default
  // value is the empty string or not. Complexity here ensures the minimal
  // number of branches / amount of extraneous code at runtime (given that the
  // below methods are inlined one-liners)!
  if (SupportsArenas(descriptor_)) {
    if (descriptor_->default_value_string().empty()) {
      printer->Print(variables_,
        "$name$_.ClearToEmpty($default_variable$, GetArenaNoVirtual());\n");
    } else {
      printer->Print(variables_,
        "$name$_.ClearToDefault($default_variable$, GetArenaNoVirtual());\n");
    }
  } else {
    if (descriptor_->default_value_string().empty()) {
      printer->Print(variables_,
        "$name$_.ClearToEmptyNoArena($default_variable$);\n");
    } else {
      printer->Print(variables_,
        "$name$_.ClearToDefaultNoArena($default_variable$);\n");
    }
  }
}

void StringFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  if (SupportsArenas(descriptor_) || descriptor_->containing_oneof() != NULL) {
    // TODO(gpike): improve this
    printer->Print(variables_, "set_$name$(from.$name$());\n");
  } else {
    printer->Print(variables_,
      "$set_hasbit$\n"
      "$name$_.AssignWithDefault($default_variable$, from.$name$_);\n");
  }
}

void StringFieldGenerator::
GenerateSwappingCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.Swap(&other->$name$_);\n");
}

void StringFieldGenerator::
GenerateConstructorCode(io::Printer* printer) const {
  printer->Print(variables_,
      "$name$_.UnsafeSetDefault($default_variable$);\n");
}

void StringFieldGenerator::
GenerateDestructorCode(io::Printer* printer) const {
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables_,
      "$name$_.Destroy($default_variable$, GetArenaNoVirtual());\n");
  } else {
    printer->Print(variables_,
      "$name$_.DestroyNoArena($default_variable$);\n");
  }
}

void StringFieldGenerator::
GenerateDefaultInstanceAllocator(io::Printer* printer) const {
  if (!descriptor_->default_value_string().empty()) {
    printer->Print(variables_,
      "$classname$::$default_variable$ =\n"
      "    new ::std::string($default$, $default_length$);\n");
  }
}

void StringFieldGenerator::
GenerateShutdownCode(io::Printer* printer) const {
  if (!descriptor_->default_value_string().empty()) {
    printer->Print(variables_,
      "delete $classname$::$default_variable$;\n");
  }
}

void StringFieldGenerator::
GenerateMergeFromCodedStream(io::Printer* printer) const {
  printer->Print(variables_,
    "DO_(::google::protobuf::internal::WireFormatLite::Read$declared_type$(\n"
    "      input, this->mutable_$name$()));\n");

  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, true, variables_,
        "this->$name$().data(), this->$name$().length(),\n", printer);
  }
}

void StringFieldGenerator::
GenerateSerializeWithCachedSizes(io::Printer* printer) const {
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, false, variables_,
        "this->$name$().data(), this->$name$().length(),\n", printer);
  }
  printer->Print(variables_,
    "::google::protobuf::internal::WireFormatLite::Write$declared_type$MaybeAliased(\n"
    "  $number$, this->$name$(), output);\n");
}

void StringFieldGenerator::
GenerateSerializeWithCachedSizesToArray(io::Printer* printer) const {
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, false, variables_,
        "this->$name$().data(), this->$name$().length(),\n", printer);
  }
  printer->Print(variables_,
    "target =\n"
    "  ::google::protobuf::internal::WireFormatLite::Write$declared_type$ToArray(\n"
    "    $number$, this->$name$(), target);\n");
}

void StringFieldGenerator::
GenerateByteSize(io::Printer* printer) const {
  printer->Print(variables_,
    "total_size += $tag_size$ +\n"
    "  ::google::protobuf::internal::WireFormatLite::$declared_type$Size(\n"
    "    this->$name$());\n");
}

// ===================================================================

StringOneofFieldGenerator::
StringOneofFieldGenerator(const FieldDescriptor* descriptor,
                          const Options& options)
    : StringFieldGenerator(descriptor, options),
      dependent_field_(options.proto_h) {
  SetCommonOneofFieldVariables(descriptor, &variables_);
}

StringOneofFieldGenerator::~StringOneofFieldGenerator() {}

void StringOneofFieldGenerator::
GenerateInlineAccessorDefinitions(io::Printer* printer,
                                  bool is_inline) const {
  map<string, string> variables(variables_);
  variables["inline"] = is_inline ? "inline" : "";
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables,
      "$inline$ const ::std::string& $classname$::$name$() const {\n"
      "  // @@protoc_insertion_point(field_get:$full_name$)\n"
      "  if (has_$name$()) {\n"
      "    return $oneof_prefix$$name$_.Get($default_variable$);\n"
      "  }\n"
      "  return *$default_variable$;\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const ::std::string& value) {\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  $oneof_prefix$$name$_.Set($default_variable$, value,\n"
      "      GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set:$full_name$)\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const char* value) {\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  $oneof_prefix$$name$_.Set($default_variable$,\n"
      "      $string_piece$(value), GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
      "}\n"
      "$inline$ "
      "void $classname$::set_$name$(const $pointer_type$* value,\n"
      "                             size_t size) {\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  $oneof_prefix$$name$_.Set($default_variable$, $string_piece$(\n"
      "      reinterpret_cast<const char*>(value), size),\n"
      "      GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
      "}\n"
      "$inline$ ::std::string* $classname$::mutable_$name$() {\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  return $oneof_prefix$$name$_.Mutable($default_variable$,\n"
      "      GetArenaNoVirtual());\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "}\n"
      "$inline$ ::std::string* $classname$::$release_name$() {\n"
      "  if (has_$name$()) {\n"
      "    clear_has_$oneof_name$();\n"
      "    return $oneof_prefix$$name$_.Release($default_variable$,\n"
      "        GetArenaNoVirtual());\n"
      "  } else {\n"
      "    return NULL;\n"
      "  }\n"
      "}\n"
      "$inline$ ::std::string* $classname$::unsafe_arena_release_$name$() {\n"
      "  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);\n"
      "  if (has_$name$()) {\n"
      "    clear_has_$oneof_name$();\n"
      "    return $oneof_prefix$$name$_.UnsafeArenaRelease(\n"
      "        $default_variable$, GetArenaNoVirtual());\n"
      "  } else {\n"
      "    return NULL;\n"
      "  }\n"
      "}\n"
      "$inline$ void $classname$::set_allocated_$name$(::std::string* $name$) {\n"
      "  if (!has_$name$()) {\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  clear_$oneof_name$();\n"
      "  if ($name$ != NULL) {\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.SetAllocated($default_variable$, $name$,\n"
      "        GetArenaNoVirtual());\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n"
      "$inline$ void $classname$::unsafe_arena_set_allocated_$name$("
      "::std::string* $name$) {\n"
      "  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);\n"
      "  if (!has_$name$()) {\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  clear_$oneof_name$();\n"
      "  if ($name$) {\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeArenaSetAllocated($default_variable$, "
      "$name$, GetArenaNoVirtual());\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  } else {
    // No-arena case.
    printer->Print(variables,
      "$inline$ const ::std::string& $classname$::$name$() const {\n"
      "  // @@protoc_insertion_point(field_get:$full_name$)\n"
      "  if (has_$name$()) {\n"
      "    return $oneof_prefix$$name$_.GetNoArena($default_variable$);\n"
      "  }\n"
      "  return *$default_variable$;\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const ::std::string& value) {\n"
      "  // @@protoc_insertion_point(field_set:$full_name$)\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  $oneof_prefix$$name$_.SetNoArena($default_variable$, value);\n"
      "  // @@protoc_insertion_point(field_set:$full_name$)\n"
      "}\n"
      "$inline$ void $classname$::set_$name$(const char* value) {\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  $oneof_prefix$$name$_.SetNoArena($default_variable$,\n"
      "      $string_piece$(value));\n"
      "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
      "}\n"
      "$inline$ "
      "void $classname$::set_$name$(const $pointer_type$* value, size_t size) {\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  $oneof_prefix$$name$_.SetNoArena($default_variable$, $string_piece$(\n"
      "      reinterpret_cast<const char*>(value), size));\n"
      "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
      "}\n"
      "$inline$ ::std::string* $classname$::mutable_$name$() {\n"
      "  if (!has_$name$()) {\n"
      "    clear_$oneof_name$();\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
      "  return $oneof_prefix$$name$_.MutableNoArena($default_variable$);\n"
      "}\n"
      "$inline$ ::std::string* $classname$::$release_name$() {\n"
      "  if (has_$name$()) {\n"
      "    clear_has_$oneof_name$();\n"
      "    return $oneof_prefix$$name$_.ReleaseNoArena($default_variable$);\n"
      "  } else {\n"
      "    return NULL;\n"
      "  }\n"
      "}\n"
      "$inline$ void $classname$::set_allocated_$name$(::std::string* $name$) {\n"
      "  if (!has_$name$()) {\n"
      "    $oneof_prefix$$name$_.UnsafeSetDefault($default_variable$);\n"
      "  }\n"
      "  clear_$oneof_name$();\n"
      "  if ($name$ != NULL) {\n"
      "    set_has_$name$();\n"
      "    $oneof_prefix$$name$_.SetAllocatedNoArena($default_variable$,\n"
      "        $name$);\n"
      "  }\n"
      "  // @@protoc_insertion_point(field_set_allocated:$full_name$)\n"
      "}\n");
  }
}

void StringOneofFieldGenerator::
GenerateClearingCode(io::Printer* printer) const {
  map<string, string> variables(variables_);
  if (dependent_field_) {
    variables["this_message"] = DependentBaseDownCast();
    // This clearing code may be in the dependent base class. If the default
    // value is an empty string, then the $default_variable$ is a global
    // singleton. If the default is not empty, we need to down-cast to get the
    // default value's global singleton instance. See SetStringVariables() for
    // possible values of default_variable.
    if (!descriptor_->default_value_string().empty()) {
      variables["default_variable"] =
          DependentBaseDownCast() + variables["default_variable"];
    }
  } else {
    variables["this_message"] = "";
  }
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables,
      "$this_message$$oneof_prefix$$name$_.Destroy($default_variable$,\n"
      "    $this_message$GetArenaNoVirtual());\n");
  } else {
    printer->Print(variables,
      "$this_message$$oneof_prefix$$name$_."
      "DestroyNoArena($default_variable$);\n");
  }
}

void StringOneofFieldGenerator::
GenerateSwappingCode(io::Printer* printer) const {
  // Don't print any swapping code. Swapping the union will swap this field.
}

void StringOneofFieldGenerator::
GenerateConstructorCode(io::Printer* printer) const {
  printer->Print(variables_,
    "  $classname$_default_oneof_instance_->$name$_.UnsafeSetDefault("
    "$default_variable$);\n");
}

void StringOneofFieldGenerator::
GenerateDestructorCode(io::Printer* printer) const {
  if (SupportsArenas(descriptor_)) {
    printer->Print(variables_,
      "if (has_$name$()) {\n"
      "  $oneof_prefix$$name$_.Destroy($default_variable$,\n"
      "      GetArenaNoVirtual());\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "if (has_$name$()) {\n"
      "  $oneof_prefix$$name$_.DestroyNoArena($default_variable$);\n"
      "}\n");
  }
}

void StringOneofFieldGenerator::
GenerateMergeFromCodedStream(io::Printer* printer) const {
    printer->Print(variables_,
      "DO_(::google::protobuf::internal::WireFormatLite::Read$declared_type$(\n"
      "      input, this->mutable_$name$()));\n");

  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, true, variables_,
        "this->$name$().data(), this->$name$().length(),\n", printer);
  }
}


// ===================================================================

RepeatedStringFieldGenerator::
RepeatedStringFieldGenerator(const FieldDescriptor* descriptor,
                             const Options& options)
    : descriptor_(descriptor) {
  SetStringVariables(descriptor, &variables_, options);
}

RepeatedStringFieldGenerator::~RepeatedStringFieldGenerator() {}

void RepeatedStringFieldGenerator::
GeneratePrivateMembers(io::Printer* printer) const {
  printer->Print(variables_,
    "::google::protobuf::RepeatedPtrField< ::std::string> $name$_;\n");
}

void RepeatedStringFieldGenerator::
GenerateAccessorDeclarations(io::Printer* printer) const {
  // See comment above about unknown ctypes.
  bool unknown_ctype =
      descriptor_->options().ctype() != EffectiveStringCType(descriptor_);

  if (unknown_ctype) {
    printer->Outdent();
    printer->Print(
      " private:\n"
      "  // Hidden due to unknown ctype option.\n");
    printer->Indent();
  }

  printer->Print(variables_,
    "const ::std::string& $name$(int index) const$deprecation$;\n"
    "::std::string* mutable_$name$(int index)$deprecation$;\n"
    "void set_$name$(int index, const ::std::string& value)$deprecation$;\n"
    "void set_$name$(int index, const char* value)$deprecation$;\n"
    ""
    "void set_$name$(int index, const $pointer_type$* value, size_t size)"
                 "$deprecation$;\n"
    "::std::string* add_$name$()$deprecation$;\n"
    "void add_$name$(const ::std::string& value)$deprecation$;\n"
    "void add_$name$(const char* value)$deprecation$;\n"
    "void add_$name$(const $pointer_type$* value, size_t size)"
                 "$deprecation$;\n");

  printer->Print(variables_,
    "const ::google::protobuf::RepeatedPtrField< ::std::string>& $name$() const"
                 "$deprecation$;\n"
    "::google::protobuf::RepeatedPtrField< ::std::string>* mutable_$name$()"
                 "$deprecation$;\n");

  if (unknown_ctype) {
    printer->Outdent();
    printer->Print(" public:\n");
    printer->Indent();
  }
}

void RepeatedStringFieldGenerator::
GenerateInlineAccessorDefinitions(io::Printer* printer,
                                  bool is_inline) const {
  map<string, string> variables(variables_);
  variables["inline"] = is_inline ? "inline" : "";
  printer->Print(variables,
    "$inline$ const ::std::string& $classname$::$name$(int index) const {\n"
    "  // @@protoc_insertion_point(field_get:$full_name$)\n"
    "  return $name$_.$cppget$(index);\n"
    "}\n"
    "$inline$ ::std::string* $classname$::mutable_$name$(int index) {\n"
    "  // @@protoc_insertion_point(field_mutable:$full_name$)\n"
    "  return $name$_.Mutable(index);\n"
    "}\n"
    "$inline$ void $classname$::set_$name$(int index, const ::std::string& value) {\n"
    "  // @@protoc_insertion_point(field_set:$full_name$)\n"
    "  $name$_.Mutable(index)->assign(value);\n"
    "}\n"
    "$inline$ void $classname$::set_$name$(int index, const char* value) {\n"
    "  $name$_.Mutable(index)->assign(value);\n"
    "  // @@protoc_insertion_point(field_set_char:$full_name$)\n"
    "}\n"
    "$inline$ void "
    "$classname$::set_$name$"
    "(int index, const $pointer_type$* value, size_t size) {\n"
    "  $name$_.Mutable(index)->assign(\n"
    "    reinterpret_cast<const char*>(value), size);\n"
    "  // @@protoc_insertion_point(field_set_pointer:$full_name$)\n"
    "}\n"
    "$inline$ ::std::string* $classname$::add_$name$() {\n"
    "  return $name$_.Add();\n"
    "}\n"
    "$inline$ void $classname$::add_$name$(const ::std::string& value) {\n"
    "  $name$_.Add()->assign(value);\n"
    "  // @@protoc_insertion_point(field_add:$full_name$)\n"
    "}\n"
    "$inline$ void $classname$::add_$name$(const char* value) {\n"
    "  $name$_.Add()->assign(value);\n"
    "  // @@protoc_insertion_point(field_add_char:$full_name$)\n"
    "}\n"
    "$inline$ void "
    "$classname$::add_$name$(const $pointer_type$* value, size_t size) {\n"
    "  $name$_.Add()->assign(reinterpret_cast<const char*>(value), size);\n"
    "  // @@protoc_insertion_point(field_add_pointer:$full_name$)\n"
    "}\n");
  printer->Print(variables,
    "$inline$ const ::google::protobuf::RepeatedPtrField< ::std::string>&\n"
    "$classname$::$name$() const {\n"
    "  // @@protoc_insertion_point(field_list:$full_name$)\n"
    "  return $name$_;\n"
    "}\n"
    "$inline$ ::google::protobuf::RepeatedPtrField< ::std::string>*\n"
    "$classname$::mutable_$name$() {\n"
    "  // @@protoc_insertion_point(field_mutable_list:$full_name$)\n"
    "  return &$name$_;\n"
    "}\n");
}

void RepeatedStringFieldGenerator::
GenerateClearingCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.Clear();\n");
}

void RepeatedStringFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.MergeFrom(from.$name$_);\n");
}

void RepeatedStringFieldGenerator::
GenerateSwappingCode(io::Printer* printer) const {
  printer->Print(variables_, "$name$_.UnsafeArenaSwap(&other->$name$_);\n");
}

void RepeatedStringFieldGenerator::
GenerateConstructorCode(io::Printer* printer) const {
  // Not needed for repeated fields.
}

void RepeatedStringFieldGenerator::
GenerateMergeFromCodedStream(io::Printer* printer) const {
  printer->Print(variables_,
    "DO_(::google::protobuf::internal::WireFormatLite::Read$declared_type$(\n"
    "      input, this->add_$name$()));\n");
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, true, variables_,
        "this->$name$(this->$name$_size() - 1).data(),\n"
        "this->$name$(this->$name$_size() - 1).length(),\n",
        printer);
  }
}

void RepeatedStringFieldGenerator::
GenerateSerializeWithCachedSizes(io::Printer* printer) const {
  printer->Print(variables_,
    "for (int i = 0; i < this->$name$_size(); i++) {\n");
  printer->Indent();
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, false, variables_,
        "this->$name$(i).data(), this->$name$(i).length(),\n", printer);
  }
  printer->Outdent();
  printer->Print(variables_,
    "  ::google::protobuf::internal::WireFormatLite::Write$declared_type$(\n"
    "    $number$, this->$name$(i), output);\n"
    "}\n");
}

void RepeatedStringFieldGenerator::
GenerateSerializeWithCachedSizesToArray(io::Printer* printer) const {
  printer->Print(variables_,
    "for (int i = 0; i < this->$name$_size(); i++) {\n");
  printer->Indent();
  if (descriptor_->type() == FieldDescriptor::TYPE_STRING) {
    GenerateUtf8CheckCodeForString(
        descriptor_, false, variables_,
        "this->$name$(i).data(), this->$name$(i).length(),\n", printer);
  }
  printer->Outdent();
  printer->Print(variables_,
    "  target = ::google::protobuf::internal::WireFormatLite::\n"
    "    Write$declared_type$ToArray($number$, this->$name$(i), target);\n"
    "}\n");
}

void RepeatedStringFieldGenerator::
GenerateByteSize(io::Printer* printer) const {
  printer->Print(variables_,
    "total_size += $tag_size$ * this->$name$_size();\n"
    "for (int i = 0; i < this->$name$_size(); i++) {\n"
    "  total_size += ::google::protobuf::internal::WireFormatLite::$declared_type$Size(\n"
    "    this->$name$(i));\n"
    "}\n");
}

}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
