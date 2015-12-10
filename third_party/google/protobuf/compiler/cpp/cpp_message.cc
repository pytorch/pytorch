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

#include <algorithm>
#include <google/protobuf/stubs/hash.h>
#include <map>
#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <utility>
#include <vector>
#include <google/protobuf/compiler/cpp/cpp_message.h>
#include <google/protobuf/compiler/cpp/cpp_field.h>
#include <google/protobuf/compiler/cpp/cpp_enum.h>
#include <google/protobuf/compiler/cpp/cpp_extension.h>
#include <google/protobuf/compiler/cpp/cpp_helpers.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/descriptor.pb.h>


namespace google {
namespace protobuf {
namespace compiler {
namespace cpp {

using internal::WireFormat;
using internal::WireFormatLite;

namespace {

template <class T>
void PrintFieldComment(io::Printer* printer, const T* field) {
  // Print the field's (or oneof's) proto-syntax definition as a comment.
  // We don't want to print group bodies so we cut off after the first
  // line.
  DebugStringOptions options;
  options.elide_group_body = true;
  options.elide_oneof_body = true;
  string def = field->DebugStringWithOptions(options);
  printer->Print("// $def$\n",
    "def", def.substr(0, def.find_first_of('\n')));
}

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
  std::sort(fields, fields + descriptor->field_count(),
            FieldOrderingByNumber());
  return fields;
}

// Functor for sorting extension ranges by their "start" field number.
struct ExtensionRangeSorter {
  bool operator()(const Descriptor::ExtensionRange* left,
                  const Descriptor::ExtensionRange* right) const {
    return left->start < right->start;
  }
};

// Returns true if the "required" restriction check should be ignored for the
// given field.
inline static bool ShouldIgnoreRequiredFieldCheck(
    const FieldDescriptor* field) {
  return false;
}

// Returns true if the message type has any required fields.  If it doesn't,
// we can optimize out calls to its IsInitialized() method.
//
// already_seen is used to avoid checking the same type multiple times
// (and also to protect against recursion).
static bool HasRequiredFields(
    const Descriptor* type,
    hash_set<const Descriptor*>* already_seen) {
  if (already_seen->count(type) > 0) {
    // Since the first occurrence of a required field causes the whole
    // function to return true, we can assume that if the type is already
    // in the cache it didn't have any required fields.
    return false;
  }
  already_seen->insert(type);

  // If the type has extensions, an extension with message type could contain
  // required fields, so we have to be conservative and assume such an
  // extension exists.
  if (type->extension_range_count() > 0) return true;

  for (int i = 0; i < type->field_count(); i++) {
    const FieldDescriptor* field = type->field(i);
    if (field->is_required()) {
      return true;
    }
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE &&
        !ShouldIgnoreRequiredFieldCheck(field)) {
      if (HasRequiredFields(field->message_type(), already_seen)) {
        return true;
      }
    }
  }

  return false;
}

static bool HasRequiredFields(const Descriptor* type) {
  hash_set<const Descriptor*> already_seen;
  return HasRequiredFields(type, &already_seen);
}

// This returns an estimate of the compiler's alignment for the field.  This
// can't guarantee to be correct because the generated code could be compiled on
// different systems with different alignment rules.  The estimates below assume
// 64-bit pointers.
int EstimateAlignmentSize(const FieldDescriptor* field) {
  if (field == NULL) return 0;
  if (field->is_repeated()) return 8;
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_BOOL:
      return 1;

    case FieldDescriptor::CPPTYPE_INT32:
    case FieldDescriptor::CPPTYPE_UINT32:
    case FieldDescriptor::CPPTYPE_ENUM:
    case FieldDescriptor::CPPTYPE_FLOAT:
      return 4;

    case FieldDescriptor::CPPTYPE_INT64:
    case FieldDescriptor::CPPTYPE_UINT64:
    case FieldDescriptor::CPPTYPE_DOUBLE:
    case FieldDescriptor::CPPTYPE_STRING:
    case FieldDescriptor::CPPTYPE_MESSAGE:
      return 8;
  }
  GOOGLE_LOG(FATAL) << "Can't get here.";
  return -1;  // Make compiler happy.
}

// FieldGroup is just a helper for OptimizePadding below.  It holds a vector of
// fields that are grouped together because they have compatible alignment, and
// a preferred location in the final field ordering.
class FieldGroup {
 public:
  FieldGroup()
      : preferred_location_(0) {}

  // A group with a single field.
  FieldGroup(float preferred_location, const FieldDescriptor* field)
      : preferred_location_(preferred_location),
        fields_(1, field) {}

  // Append the fields in 'other' to this group.
  void Append(const FieldGroup& other) {
    if (other.fields_.empty()) {
      return;
    }
    // Preferred location is the average among all the fields, so we weight by
    // the number of fields on each FieldGroup object.
    preferred_location_ =
        (preferred_location_ * fields_.size() +
         (other.preferred_location_ * other.fields_.size())) /
        (fields_.size() + other.fields_.size());
    fields_.insert(fields_.end(), other.fields_.begin(), other.fields_.end());
  }

  void SetPreferredLocation(float location) { preferred_location_ = location; }
  const vector<const FieldDescriptor*>& fields() const { return fields_; }

  // FieldGroup objects sort by their preferred location.
  bool operator<(const FieldGroup& other) const {
    return preferred_location_ < other.preferred_location_;
  }

 private:
  // "preferred_location_" is an estimate of where this group should go in the
  // final list of fields.  We compute this by taking the average index of each
  // field in this group in the original ordering of fields.  This is very
  // approximate, but should put this group close to where its member fields
  // originally went.
  float preferred_location_;
  vector<const FieldDescriptor*> fields_;
  // We rely on the default copy constructor and operator= so this type can be
  // used in a vector.
};

// Reorder 'fields' so that if the fields are output into a c++ class in the new
// order, the alignment padding is minimized.  We try to do this while keeping
// each field as close as possible to its original position so that we don't
// reduce cache locality much for function that access each field in order.
void OptimizePadding(vector<const FieldDescriptor*>* fields) {
  // First divide fields into those that align to 1 byte, 4 bytes or 8 bytes.
  vector<FieldGroup> aligned_to_1, aligned_to_4, aligned_to_8;
  for (int i = 0; i < fields->size(); ++i) {
    switch (EstimateAlignmentSize((*fields)[i])) {
      case 1: aligned_to_1.push_back(FieldGroup(i, (*fields)[i])); break;
      case 4: aligned_to_4.push_back(FieldGroup(i, (*fields)[i])); break;
      case 8: aligned_to_8.push_back(FieldGroup(i, (*fields)[i])); break;
      default:
        GOOGLE_LOG(FATAL) << "Unknown alignment size.";
    }
  }

  // Now group fields aligned to 1 byte into sets of 4, and treat those like a
  // single field aligned to 4 bytes.
  for (int i = 0; i < aligned_to_1.size(); i += 4) {
    FieldGroup field_group;
    for (int j = i; j < aligned_to_1.size() && j < i + 4; ++j) {
      field_group.Append(aligned_to_1[j]);
    }
    aligned_to_4.push_back(field_group);
  }
  // Sort by preferred location to keep fields as close to their original
  // location as possible.  Using stable_sort ensures that the output is
  // consistent across runs.
  std::stable_sort(aligned_to_4.begin(), aligned_to_4.end());

  // Now group fields aligned to 4 bytes (or the 4-field groups created above)
  // into pairs, and treat those like a single field aligned to 8 bytes.
  for (int i = 0; i < aligned_to_4.size(); i += 2) {
    FieldGroup field_group;
    for (int j = i; j < aligned_to_4.size() && j < i + 2; ++j) {
      field_group.Append(aligned_to_4[j]);
    }
    if (i == aligned_to_4.size() - 1) {
      // Move incomplete 4-byte block to the end.
      field_group.SetPreferredLocation(fields->size() + 1);
    }
    aligned_to_8.push_back(field_group);
  }
  // Sort by preferred location.
  std::stable_sort(aligned_to_8.begin(), aligned_to_8.end());

  // Now pull out all the FieldDescriptors in order.
  fields->clear();
  for (int i = 0; i < aligned_to_8.size(); ++i) {
    fields->insert(fields->end(),
                   aligned_to_8[i].fields().begin(),
                   aligned_to_8[i].fields().end());
  }
}

string MessageTypeProtoName(const FieldDescriptor* field) {
  return field->message_type()->full_name();
}

// Emits an if-statement with a condition that evaluates to true if |field| is
// considered non-default (will be sent over the wire), for message types
// without true field presence. Should only be called if
// !HasFieldPresence(message_descriptor).
bool EmitFieldNonDefaultCondition(io::Printer* printer,
                                  const string& prefix,
                                  const FieldDescriptor* field) {
  // Merge and serialize semantics: primitive fields are merged/serialized only
  // if non-zero (numeric) or non-empty (string).
  if (!field->is_repeated() && !field->containing_oneof()) {
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_STRING) {
      printer->Print(
          "if ($prefix$$name$().size() > 0) {\n",
          "prefix", prefix,
          "name", FieldName(field));
    } else if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      // Message fields still have has_$name$() methods.
      printer->Print(
          "if ($prefix$has_$name$()) {\n",
          "prefix", prefix,
          "name", FieldName(field));
    } else {
      printer->Print(
          "if ($prefix$$name$() != 0) {\n",
          "prefix", prefix,
          "name", FieldName(field));
    }
    printer->Indent();
    return true;
  } else if (field->containing_oneof()) {
    printer->Print(
        "if (has_$name$()) {\n",
        "name", FieldName(field));
    printer->Indent();
    return true;
  }
  return false;
}

// Does the given field have a has_$name$() method?
bool HasHasMethod(const FieldDescriptor* field) {
  if (HasFieldPresence(field->file())) {
    // In proto1/proto2, every field has a has_$name$() method.
    return true;
  }
  // For message types without true field presence, only fields with a message
  // type have a has_$name$() method.
  return field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE;
}

// Collects map entry message type information.
void CollectMapInfo(const Descriptor* descriptor,
                    map<string, string>* variables) {
  GOOGLE_CHECK(IsMapEntryMessage(descriptor));
  const FieldDescriptor* key = descriptor->FindFieldByName("key");
  const FieldDescriptor* val = descriptor->FindFieldByName("value");
  (*variables)["key"] = PrimitiveTypeName(key->cpp_type());
  switch (val->cpp_type()) {
    case FieldDescriptor::CPPTYPE_MESSAGE:
      (*variables)["val"] = FieldMessageTypeName(val);
      break;
    case FieldDescriptor::CPPTYPE_ENUM:
      (*variables)["val"] = ClassName(val->enum_type(), true);
      break;
    default:
      (*variables)["val"] = PrimitiveTypeName(val->cpp_type());
  }
  (*variables)["key_wire_type"] =
      "::google::protobuf::internal::WireFormatLite::TYPE_" +
      ToUpper(DeclaredTypeMethodName(key->type()));
  (*variables)["val_wire_type"] =
      "::google::protobuf::internal::WireFormatLite::TYPE_" +
      ToUpper(DeclaredTypeMethodName(val->type()));
}

// Does the given field have a private (internal helper only) has_$name$()
// method?
bool HasPrivateHasMethod(const FieldDescriptor* field) {
  // Only for oneofs in message types with no field presence. has_$name$(),
  // based on the oneof case, is still useful internally for generated code.
  return (!HasFieldPresence(field->file()) &&
          field->containing_oneof() != NULL);
}

}  // anonymous namespace

// ===================================================================

MessageGenerator::MessageGenerator(const Descriptor* descriptor,
                                   const Options& options)
    : descriptor_(descriptor),
      classname_(ClassName(descriptor, false)),
      options_(options),
      field_generators_(descriptor, options),
      nested_generators_(new google::protobuf::scoped_ptr<
          MessageGenerator>[descriptor->nested_type_count()]),
      enum_generators_(
          new google::protobuf::scoped_ptr<EnumGenerator>[descriptor->enum_type_count()]),
      extension_generators_(new google::protobuf::scoped_ptr<
          ExtensionGenerator>[descriptor->extension_count()]),
      use_dependent_base_(false) {

  for (int i = 0; i < descriptor->nested_type_count(); i++) {
    nested_generators_[i].reset(
      new MessageGenerator(descriptor->nested_type(i), options));
  }

  for (int i = 0; i < descriptor->enum_type_count(); i++) {
    enum_generators_[i].reset(
      new EnumGenerator(descriptor->enum_type(i), options));
  }

  for (int i = 0; i < descriptor->extension_count(); i++) {
    extension_generators_[i].reset(
      new ExtensionGenerator(descriptor->extension(i), options));
  }

  num_required_fields_ = 0;
  for (int i = 0; i < descriptor->field_count(); i++) {
    if (descriptor->field(i)->is_required()) {
      ++num_required_fields_;
    }
    if (options.proto_h && IsFieldDependent(descriptor->field(i))) {
      use_dependent_base_ = true;
    }
  }
  if (options.proto_h && descriptor->oneof_decl_count() > 0) {
    // Always make oneofs dependent.
    use_dependent_base_ = true;
  }
}

MessageGenerator::~MessageGenerator() {}

void MessageGenerator::
FillMessageForwardDeclarations(set<string>* class_names) {
  class_names->insert(classname_);

  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // map entry message doesn't need forward declaration. Since map entry
    // message cannot be a top level class, we just need to avoid calling
    // GenerateForwardDeclaration here.
    if (IsMapEntryMessage(descriptor_->nested_type(i))) continue;
    nested_generators_[i]->FillMessageForwardDeclarations(class_names);
  }
}

void MessageGenerator::
FillEnumForwardDeclarations(set<string>* enum_names) {
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    nested_generators_[i]->FillEnumForwardDeclarations(enum_names);
  }
  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    enum_generators_[i]->FillForwardDeclaration(enum_names);
  }
}

void MessageGenerator::
GenerateEnumDefinitions(io::Printer* printer) {
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    nested_generators_[i]->GenerateEnumDefinitions(printer);
  }

  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    enum_generators_[i]->GenerateDefinition(printer);
  }
}

void MessageGenerator::
GenerateGetEnumDescriptorSpecializations(io::Printer* printer) {
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    nested_generators_[i]->GenerateGetEnumDescriptorSpecializations(printer);
  }
  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    enum_generators_[i]->GenerateGetEnumDescriptorSpecializations(printer);
  }
}

void MessageGenerator::
GenerateDependentFieldAccessorDeclarations(io::Printer* printer) {
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    PrintFieldComment(printer, field);

    map<string, string> vars;
    SetCommonFieldVariables(field, &vars, options_);

    if (use_dependent_base_ && IsFieldDependent(field)) {
      // If the message is dependent, the inline clear_*() method will need
      // to delete the message type, so it must be in the dependent base
      // class. (See also GenerateFieldAccessorDeclarations.)
      printer->Print(vars, "void clear_$name$()$deprecation$;\n");
    }
    // Generate type-specific accessor declarations.
    field_generators_.get(field).GenerateDependentAccessorDeclarations(printer);
    printer->Print("\n");
  }
}

void MessageGenerator::
GenerateFieldAccessorDeclarations(io::Printer* printer) {
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    PrintFieldComment(printer, field);

    map<string, string> vars;
    SetCommonFieldVariables(field, &vars, options_);
    vars["constant_name"] = FieldConstantName(field);

    bool dependent_field = use_dependent_base_ && IsFieldDependent(field);
    if (dependent_field &&
        field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE &&
        !field->is_map()) {
      // If this field is dependent, the dependent base class determines
      // the message type from the derived class (which is a template
      // parameter). This typedef is for that:
      printer->Print(
          "private:\n"
          "typedef $field_type$ $dependent_type$;\n"
          "public:\n",
          "field_type", FieldMessageTypeName(field),
          "dependent_type", DependentTypeName(field));
    }

    if (field->is_repeated()) {
      printer->Print(vars, "int $name$_size() const$deprecation$;\n");
    } else if (HasHasMethod(field)) {
      printer->Print(vars, "bool has_$name$() const$deprecation$;\n");
    } else if (HasPrivateHasMethod(field)) {
      printer->Print(vars,
          "private:\n"
          "bool has_$name$() const$deprecation$;\n"
          "public:\n");
    }

    if (!dependent_field) {
      // If this field is dependent, then its clear_() method is in the
      // depenent base class. (See also GenerateDependentAccessorDeclarations.)
      printer->Print(vars, "void clear_$name$()$deprecation$;\n");
    }
    printer->Print(vars, "static const int $constant_name$ = $number$;\n");

    // Generate type-specific accessor declarations.
    field_generators_.get(field).GenerateAccessorDeclarations(printer);

    printer->Print("\n");
  }

  if (descriptor_->extension_range_count() > 0) {
    // Generate accessors for extensions.  We just call a macro located in
    // extension_set.h since the accessors about 80 lines of static code.
    printer->Print(
      "GOOGLE_PROTOBUF_EXTENSION_ACCESSORS($classname$)\n",
      "classname", classname_);
  }

  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "$camel_oneof_name$Case $oneof_name$_case() const;\n",
        "camel_oneof_name",
        UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), true),
        "oneof_name", descriptor_->oneof_decl(i)->name());
  }
}

void MessageGenerator::
GenerateDependentFieldAccessorDefinitions(io::Printer* printer) {
  if (!use_dependent_base_) return;

  printer->Print("// $classname$\n\n", "classname",
                 DependentBaseClassTemplateName(descriptor_));

  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    PrintFieldComment(printer, field);

    // These functions are not really dependent: they are part of the
    // (non-dependent) derived class. However, they need to live outside
    // any #ifdef guards, so we treat them as if they were dependent.
    //
    // See the comment in FileGenerator::GenerateInlineFunctionDefinitions
    // for a more complete explanation.
    if (use_dependent_base_ && IsFieldDependent(field)) {
      map<string, string> vars;
      SetCommonFieldVariables(field, &vars, options_);
      vars["inline"] = "inline ";
      if (field->containing_oneof()) {
        vars["field_name"] = UnderscoresToCamelCase(field->name(), true);
        vars["oneof_name"] = field->containing_oneof()->name();
        vars["oneof_index"] = SimpleItoa(field->containing_oneof()->index());
        GenerateOneofMemberHasBits(field, vars, printer);
      } else if (!field->is_repeated()) {
        // There will be no header guard, so this always has to be inline.
        GenerateSingularFieldHasBits(field, vars, printer);
      }
      // vars needed for clear_(), which is in the dependent base:
      // (See also GenerateDependentFieldAccessorDeclarations.)
      vars["tmpl"] = "template<class T>\n";
      vars["dependent_classname"] =
          DependentBaseClassTemplateName(descriptor_) + "<T>";
      vars["this_message"] = DependentBaseDownCast();
      vars["this_const_message"] = DependentBaseConstDownCast();
      GenerateFieldClear(field, vars, printer);
    }

    // Generate type-specific accessors.
    field_generators_.get(field)
        .GenerateDependentInlineAccessorDefinitions(printer);

    printer->Print("\n");
  }

  // Generate has_$name$() and clear_has_$name$() functions for oneofs
  // Similar to other has-bits, these must always be in the header if we
  // are using a dependent base class.
  GenerateOneofHasBits(printer, true /* is_inline */);
}

void MessageGenerator::
GenerateSingularFieldHasBits(const FieldDescriptor* field,
                             map<string, string> vars,
                             io::Printer* printer) {
  if (HasFieldPresence(descriptor_->file())) {
    // N.B.: without field presence, we do not use has-bits or generate
    // has_$name$() methods.
    vars["has_array_index"] = SimpleItoa(field->index() / 32);
    vars["has_mask"] = StrCat(strings::Hex(1u << (field->index() % 32),
                                           strings::ZERO_PAD_8));
    printer->Print(vars,
      "$inline$"
      "bool $classname$::has_$name$() const {\n"
      "  return (_has_bits_[$has_array_index$] & 0x$has_mask$u) != 0;\n"
      "}\n"
      "$inline$"
      "void $classname$::set_has_$name$() {\n"
      "  _has_bits_[$has_array_index$] |= 0x$has_mask$u;\n"
      "}\n"
      "$inline$"
      "void $classname$::clear_has_$name$() {\n"
      "  _has_bits_[$has_array_index$] &= ~0x$has_mask$u;\n"
      "}\n");
  } else {
    // Message fields have a has_$name$() method.
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      bool is_lazy = false;
      if (is_lazy) {
        printer->Print(vars,
          "$inline$"
          "bool $classname$::has_$name$() const {\n"
          "  return !$name$_.IsCleared();\n"
          "}\n");
      } else {
        printer->Print(vars,
          "$inline$"
          "bool $classname$::has_$name$() const {\n"
          "  return !_is_default_instance_ && $name$_ != NULL;\n"
          "}\n");
      }
    }
  }
}

void MessageGenerator::
GenerateOneofHasBits(io::Printer* printer, bool is_inline) {
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    map<string, string> vars;
    vars["oneof_name"] = descriptor_->oneof_decl(i)->name();
    vars["oneof_index"] = SimpleItoa(descriptor_->oneof_decl(i)->index());
    vars["cap_oneof_name"] =
        ToUpper(descriptor_->oneof_decl(i)->name());
    vars["classname"] = classname_;
    vars["inline"] = (is_inline ? "inline " : "");
    printer->Print(
        vars,
        "$inline$"
        "bool $classname$::has_$oneof_name$() const {\n"
        "  return $oneof_name$_case() != $cap_oneof_name$_NOT_SET;\n"
        "}\n"
        "$inline$"
        "void $classname$::clear_has_$oneof_name$() {\n"
        "  _oneof_case_[$oneof_index$] = $cap_oneof_name$_NOT_SET;\n"
        "}\n");
  }
}

void MessageGenerator::
GenerateOneofMemberHasBits(const FieldDescriptor* field,
                           const map<string, string>& vars,
                           io::Printer* printer) {
  // Singular field in a oneof
  // N.B.: Without field presence, we do not use has-bits or generate
  // has_$name$() methods, but oneofs still have set_has_$name$().
  // Oneofs also have has_$name$() but only as a private helper
  // method, so that generated code is slightly cleaner (vs.  comparing
  // _oneof_case_[index] against a constant everywhere).
  printer->Print(vars,
    "$inline$"
    "bool $classname$::has_$name$() const {\n"
    "  return $oneof_name$_case() == k$field_name$;\n"
    "}\n");
  printer->Print(vars,
    "$inline$"
    "void $classname$::set_has_$name$() {\n"
    "  _oneof_case_[$oneof_index$] = k$field_name$;\n"
    "}\n");
}

void MessageGenerator::
GenerateFieldClear(const FieldDescriptor* field,
                   const map<string, string>& vars,
                   io::Printer* printer) {
  // Generate clear_$name$() (See GenerateFieldAccessorDeclarations and
  // GenerateDependentFieldAccessorDeclarations, $dependent_classname$ is
  // set by the Generate*Definitions functions.)
  printer->Print(vars,
    "$tmpl$"
    "$inline$"
    "void $dependent_classname$::clear_$name$() {\n");

  printer->Indent();

  if (field->containing_oneof()) {
    // Clear this field only if it is the active field in this oneof,
    // otherwise ignore
    printer->Print(vars,
      "if ($this_message$has_$name$()) {\n");
    printer->Indent();
    field_generators_.get(field)
        .GenerateClearingCode(printer);
    printer->Print(vars,
      "$this_message$clear_has_$oneof_name$();\n");
    printer->Outdent();
    printer->Print("}\n");
  } else {
    field_generators_.get(field)
        .GenerateClearingCode(printer);
    if (HasFieldPresence(descriptor_->file())) {
      if (!field->is_repeated()) {
        printer->Print(vars,
                       "$this_message$clear_has_$name$();\n");
      }
    }
  }

  printer->Outdent();
  printer->Print("}\n");
}

void MessageGenerator::
GenerateFieldAccessorDefinitions(io::Printer* printer, bool is_inline) {
  printer->Print("// $classname$\n\n", "classname", classname_);

  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    PrintFieldComment(printer, field);

    map<string, string> vars;
    SetCommonFieldVariables(field, &vars, options_);
    vars["inline"] = is_inline ? "inline " : "";
    if (use_dependent_base_ && IsFieldDependent(field)) {
      vars["tmpl"] = "template<class T>\n";
      vars["dependent_classname"] =
          DependentBaseClassTemplateName(descriptor_) + "<T>";
      vars["this_message"] = "reinterpret_cast<T*>(this)->";
      vars["this_const_message"] = "reinterpret_cast<const T*>(this)->";
    } else {
      vars["tmpl"] = "";
      vars["dependent_classname"] = vars["classname"];
      vars["this_message"] = "";
      vars["this_const_message"] = "";
    }

    // Generate has_$name$() or $name$_size().
    if (field->is_repeated()) {
      printer->Print(vars,
        "$inline$"
        "int $classname$::$name$_size() const {\n"
        "  return $name$_.size();\n"
        "}\n");
    } else if (field->containing_oneof()) {
      vars["field_name"] = UnderscoresToCamelCase(field->name(), true);
      vars["oneof_name"] = field->containing_oneof()->name();
      vars["oneof_index"] = SimpleItoa(field->containing_oneof()->index());
      if (!use_dependent_base_ || !IsFieldDependent(field)) {
        GenerateOneofMemberHasBits(field, vars, printer);
      }
    } else {
      // Singular field.
      if (!use_dependent_base_ || !IsFieldDependent(field)) {
        GenerateSingularFieldHasBits(field, vars, printer);
      }
    }

    if (!use_dependent_base_ || !IsFieldDependent(field)) {
      GenerateFieldClear(field, vars, printer);
    }

    // Generate type-specific accessors.
    field_generators_.get(field).GenerateInlineAccessorDefinitions(printer,
                                                                   is_inline);

    printer->Print("\n");
  }

  if (!use_dependent_base_) {
    // Generate has_$name$() and clear_has_$name$() functions for oneofs
    // If we aren't using a dependent base, they can be with the other functions
    // that are #ifdef-guarded.
    GenerateOneofHasBits(printer, is_inline);
  }
}

// Helper for the code that emits the Clear() method.
static bool CanClearByZeroing(const FieldDescriptor* field) {
  if (field->is_repeated() || field->is_extension()) return false;
  switch (field->cpp_type()) {
    case internal::WireFormatLite::CPPTYPE_ENUM:
      return field->default_value_enum()->number() == 0;
    case internal::WireFormatLite::CPPTYPE_INT32:
      return field->default_value_int32() == 0;
    case internal::WireFormatLite::CPPTYPE_INT64:
      return field->default_value_int64() == 0;
    case internal::WireFormatLite::CPPTYPE_UINT32:
      return field->default_value_uint32() == 0;
    case internal::WireFormatLite::CPPTYPE_UINT64:
      return field->default_value_uint64() == 0;
    case internal::WireFormatLite::CPPTYPE_FLOAT:
      return field->default_value_float() == 0;
    case internal::WireFormatLite::CPPTYPE_DOUBLE:
      return field->default_value_double() == 0;
    case internal::WireFormatLite::CPPTYPE_BOOL:
      return field->default_value_bool() == false;
    default:
      return false;
  }
}

void MessageGenerator::
GenerateDependentBaseClassDefinition(io::Printer* printer) {
  if (!use_dependent_base_) {
    return;
  }

  map<string, string> vars;
  vars["classname"] = DependentBaseClassTemplateName(descriptor_);
  vars["superclass"] = SuperClassName(descriptor_);

  printer->Print(vars,
    "template <class T>\n"
    "class $classname$ : public $superclass$ {\n"
    " public:\n");
  printer->Indent();

  printer->Print(vars,
    "$classname$() {}\n"
    "virtual ~$classname$() {}\n"
    "\n");

  // Generate dependent accessor methods for all fields.
  GenerateDependentFieldAccessorDeclarations(printer);

  printer->Outdent();
  printer->Print("};\n");
}

void MessageGenerator::
GenerateClassDefinition(io::Printer* printer) {
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // map entry message doesn't need class definition. Since map entry message
    // cannot be a top level class, we just need to avoid calling
    // GenerateClassDefinition here.
    if (IsMapEntryMessage(descriptor_->nested_type(i))) continue;
    nested_generators_[i]->GenerateClassDefinition(printer);
    printer->Print("\n");
    printer->Print(kThinSeparator);
    printer->Print("\n");
  }

  if (use_dependent_base_) {
    GenerateDependentBaseClassDefinition(printer);
      printer->Print("\n");
  }

  map<string, string> vars;
  vars["classname"] = classname_;
  vars["field_count"] = SimpleItoa(descriptor_->field_count());
  vars["oneof_decl_count"] = SimpleItoa(descriptor_->oneof_decl_count());
  if (options_.dllexport_decl.empty()) {
    vars["dllexport"] = "";
  } else {
    vars["dllexport"] = options_.dllexport_decl + " ";
  }
  if (use_dependent_base_) {
    vars["superclass"] =
        DependentBaseClassTemplateName(descriptor_) + "<" + classname_ + ">";
  } else {
    vars["superclass"] = SuperClassName(descriptor_);
  }
  printer->Print(vars,
    "class $dllexport$$classname$ : public $superclass$ {\n");
  if (use_dependent_base_) {
    printer->Print(vars, "  friend class $superclass$;\n");
  }
  printer->Print(" public:\n");
  printer->Indent();

  printer->Print(vars,
    "$classname$();\n"
    "virtual ~$classname$();\n"
    "\n"
    "$classname$(const $classname$& from);\n"
    "\n"
    "inline $classname$& operator=(const $classname$& from) {\n"
    "  CopyFrom(from);\n"
    "  return *this;\n"
    "}\n"
    "\n");

  if (PreserveUnknownFields(descriptor_)) {
    if (UseUnknownFieldSet(descriptor_->file())) {
      printer->Print(
        "inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {\n"
        "  return _internal_metadata_.unknown_fields();\n"
        "}\n"
        "\n"
        "inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {\n"
        "  return _internal_metadata_.mutable_unknown_fields();\n"
        "}\n"
        "\n");
    } else {
      if (SupportsArenas(descriptor_)) {
        printer->Print(
          "inline const ::std::string& unknown_fields() const {\n"
          "  return _unknown_fields_.Get(\n"
          "      &::google::protobuf::internal::GetEmptyStringAlreadyInited());\n"
          "}\n"
          "\n"
          "inline ::std::string* mutable_unknown_fields() {\n"
          "  return _unknown_fields_.Mutable(\n"
          "      &::google::protobuf::internal::GetEmptyStringAlreadyInited(),\n"
          "      GetArenaNoVirtual());\n"
          "}\n"
          "\n");
      } else {
        printer->Print(
          "inline const ::std::string& unknown_fields() const {\n"
          "  return _unknown_fields_.GetNoArena(\n"
          "      &::google::protobuf::internal::GetEmptyStringAlreadyInited());\n"
          "}\n"
          "\n"
          "inline ::std::string* mutable_unknown_fields() {\n"
          "  return _unknown_fields_.MutableNoArena(\n"
          "      &::google::protobuf::internal::GetEmptyStringAlreadyInited());\n"
          "}\n"
          "\n");
      }
    }
  }

  // N.B.: We exclude GetArena() when arena support is disabled, falling back on
  // MessageLite's implementation which returns NULL rather than generating our
  // own method which returns NULL, in order to reduce code size.
  if (SupportsArenas(descriptor_)) {
    // virtual method version of GetArenaNoVirtual(), required for generic dispatch given a
    // MessageLite* (e.g., in RepeatedField::AddAllocated()).
    printer->Print(
        "inline ::google::protobuf::Arena* GetArena() const { return GetArenaNoVirtual(); }\n"
        "inline void* GetMaybeArenaPointer() const {\n"
        "  return MaybeArenaPtr();\n"
        "}\n");
  }

  // Only generate this member if it's not disabled.
  if (HasDescriptorMethods(descriptor_->file()) &&
      !descriptor_->options().no_standard_descriptor_accessor()) {
    printer->Print(vars,
      "static const ::google::protobuf::Descriptor* descriptor();\n");
  }

  printer->Print(vars,
    "static const $classname$& default_instance();\n"
    "\n");

  // Generate enum values for every field in oneofs. One list is generated for
  // each oneof with an additional *_NOT_SET value.
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "enum $camel_oneof_name$Case {\n",
        "camel_oneof_name",
        UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), true));
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      printer->Print(
          "k$field_name$ = $field_number$,\n",
          "field_name",
          UnderscoresToCamelCase(
              descriptor_->oneof_decl(i)->field(j)->name(), true),
          "field_number",
          SimpleItoa(descriptor_->oneof_decl(i)->field(j)->number()));
    }
    printer->Print(
        "$cap_oneof_name$_NOT_SET = 0,\n",
        "cap_oneof_name",
        ToUpper(descriptor_->oneof_decl(i)->name()));
    printer->Outdent();
    printer->Print(
        "};\n"
        "\n");
  }

  if (!StaticInitializersForced(descriptor_->file())) {
    printer->Print(vars,
      "#ifdef GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER\n"
      "// Returns the internal default instance pointer. This function can\n"
      "// return NULL thus should not be used by the user. This is intended\n"
      "// for Protobuf internal code. Please use default_instance() declared\n"
      "// above instead.\n"
      "static inline const $classname$* internal_default_instance() {\n"
      "  return default_instance_;\n"
      "}\n"
      "#endif\n"
      "\n");
  }


  if (SupportsArenas(descriptor_)) {
    printer->Print(vars,
      "void UnsafeArenaSwap($classname$* other);\n");
  }

  if (IsAnyMessage(descriptor_)) {
    printer->Print(vars,
      "// implements Any -----------------------------------------------\n"
      "\n"
      "void PackFrom(const ::google::protobuf::Message& message);\n"
      "bool UnpackTo(::google::protobuf::Message* message) const;\n"
      "template<typename T> bool Is() const {\n"
      "  return _any_metadata_.Is<T>();\n"
      "}\n"
      "\n");
  }

  printer->Print(vars,
    "void Swap($classname$* other);\n"
    "\n"
    "// implements Message ----------------------------------------------\n"
    "\n"
    "inline $classname$* New() const { return New(NULL); }\n"
    "\n"
    "$classname$* New(::google::protobuf::Arena* arena) const;\n");

  if (HasGeneratedMethods(descriptor_->file())) {
    if (HasDescriptorMethods(descriptor_->file())) {
      printer->Print(vars,
        "void CopyFrom(const ::google::protobuf::Message& from);\n"
        "void MergeFrom(const ::google::protobuf::Message& from);\n");
    } else {
      printer->Print(vars,
        "void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from);\n");
    }

    printer->Print(vars,
      "void CopyFrom(const $classname$& from);\n"
      "void MergeFrom(const $classname$& from);\n"
      "void Clear();\n"
      "bool IsInitialized() const;\n"
      "\n"
      "int ByteSize() const;\n"
      "bool MergePartialFromCodedStream(\n"
      "    ::google::protobuf::io::CodedInputStream* input);\n"
      "void SerializeWithCachedSizes(\n"
      "    ::google::protobuf::io::CodedOutputStream* output) const;\n");
    // DiscardUnknownFields() is implemented in message.cc using reflections. We
    // need to implement this function in generated code for messages.
    if (!UseUnknownFieldSet(descriptor_->file())) {
      printer->Print(
        "void DiscardUnknownFields();\n");
    }
    if (HasFastArraySerialization(descriptor_->file())) {
      printer->Print(
        "::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;\n");
    }
  }

  // Check all FieldDescriptors including those in oneofs to estimate
  // whether ::std::string is likely to be used, and depending on that
  // estimate, set uses_string_ to true or false.  That contols
  // whether to force initialization of empty_string_ in SharedCtor().
  // It's often advantageous to do so to keep "is empty_string_
  // inited?" code from appearing all over the place.
  vector<const FieldDescriptor*> descriptors;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    descriptors.push_back(descriptor_->field(i));
  }
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      descriptors.push_back(descriptor_->oneof_decl(i)->field(j));
    }
  }
  uses_string_ = false;
  if (PreserveUnknownFields(descriptor_) &&
      !UseUnknownFieldSet(descriptor_->file())) {
    uses_string_ = true;
  }
  for (int i = 0; i < descriptors.size(); i++) {
    const FieldDescriptor* field = descriptors[i];
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_STRING) {
      switch (field->options().ctype()) {
        default: uses_string_ = true; break;
      }
    }
  }

  printer->Print(
    "int GetCachedSize() const { return _cached_size_; }\n"
    "private:\n"
    "void SharedCtor();\n"
    "void SharedDtor();\n"
    "void SetCachedSize(int size) const;\n"
    "void InternalSwap($classname$* other);\n",
    "classname", classname_);
  if (SupportsArenas(descriptor_)) {
    printer->Print(
      "protected:\n"
      "explicit $classname$(::google::protobuf::Arena* arena);\n"
      "private:\n"
      "static void ArenaDtor(void* object);\n"
      "inline void RegisterArenaDtor(::google::protobuf::Arena* arena);\n",
      "classname", classname_);
  }

  if (UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(
      "private:\n"
      "inline ::google::protobuf::Arena* GetArenaNoVirtual() const {\n"
      "  return _internal_metadata_.arena();\n"
      "}\n"
      "inline void* MaybeArenaPtr() const {\n"
      "  return _internal_metadata_.raw_arena_ptr();\n"
      "}\n"
      "public:\n"
      "\n");
  } else {
    printer->Print(
      "private:\n"
      "inline ::google::protobuf::Arena* GetArenaNoVirtual() const {\n"
      "  return _arena_ptr_;\n"
      "}\n"
      "inline ::google::protobuf::Arena* MaybeArenaPtr() const {\n"
      "  return _arena_ptr_;\n"
      "}\n"
      "public:\n"
      "\n");
  }

  if (HasDescriptorMethods(descriptor_->file())) {
    printer->Print(
      "::google::protobuf::Metadata GetMetadata() const;\n"
      "\n");
  } else {
    printer->Print(
      "::std::string GetTypeName() const;\n"
      "\n");
  }

  printer->Print(
    "// nested types ----------------------------------------------------\n"
    "\n");

  // Import all nested message classes into this class's scope with typedefs.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    const Descriptor* nested_type = descriptor_->nested_type(i);
    if (!IsMapEntryMessage(nested_type)) {
      printer->Print("typedef $nested_full_name$ $nested_name$;\n",
                     "nested_name", nested_type->name(),
                     "nested_full_name", ClassName(nested_type, false));
    }
  }

  if (descriptor_->nested_type_count() > 0) {
    printer->Print("\n");
  }

  // Import all nested enums and their values into this class's scope with
  // typedefs and constants.
  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    enum_generators_[i]->GenerateSymbolImports(printer);
    printer->Print("\n");
  }

  printer->Print(
    "// accessors -------------------------------------------------------\n"
    "\n");

  // Generate accessor methods for all fields.
  GenerateFieldAccessorDeclarations(printer);

  // Declare extension identifiers.
  for (int i = 0; i < descriptor_->extension_count(); i++) {
    extension_generators_[i]->GenerateDeclaration(printer);
  }


  printer->Print(
    "// @@protoc_insertion_point(class_scope:$full_name$)\n",
    "full_name", descriptor_->full_name());

  // Generate private members.
  printer->Outdent();
  printer->Print(" private:\n");
  printer->Indent();


  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->is_repeated()) {
      // set_has_***() generated in all proto1/2 code and in oneofs (only) for
      // messages without true field presence.
      if (HasFieldPresence(descriptor_->file()) ||
          descriptor_->field(i)->containing_oneof()) {
        printer->Print(
          "inline void set_has_$name$();\n",
          "name", FieldName(descriptor_->field(i)));
      }
      // clear_has_***() generated only for non-oneof fields
      // in proto1/2.
      if (!descriptor_->field(i)->containing_oneof() &&
          HasFieldPresence(descriptor_->file())) {
        printer->Print(
          "inline void clear_has_$name$();\n",
          "name", FieldName(descriptor_->field(i)));
      }
    }
  }
  printer->Print("\n");

  // Generate oneof function declarations
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "inline bool has_$oneof_name$() const;\n"
        "void clear_$oneof_name$();\n"
        "inline void clear_has_$oneof_name$();\n\n",
        "oneof_name", descriptor_->oneof_decl(i)->name());
  }

  if (HasGeneratedMethods(descriptor_->file()) &&
      !descriptor_->options().message_set_wire_format() &&
      num_required_fields_ > 1) {
    printer->Print(
        "// helper for ByteSize()\n"
        "int RequiredFieldsByteSizeFallback() const;\n\n");
  }

  // Prepare decls for _cached_size_ and _has_bits_.  Their position in the
  // output will be determined later.

  bool need_to_emit_cached_size = true;
  // TODO(kenton):  Make _cached_size_ an atomic<int> when C++ supports it.
  const string cached_size_decl = "mutable int _cached_size_;\n";

  // TODO(jieluo) - Optimize _has_bits_ for repeated and oneof fields.
  size_t sizeof_has_bits = (descriptor_->field_count() + 31) / 32 * 4;
  if (descriptor_->field_count() == 0) {
    // Zero-size arrays aren't technically allowed, and MSVC in particular
    // doesn't like them.  We still need to declare these arrays to make
    // other code compile.  Since this is an uncommon case, we'll just declare
    // them with size 1 and waste some space.  Oh well.
    sizeof_has_bits = 4;
  }
  const string has_bits_decl = sizeof_has_bits == 0 ? "" :
      "::google::protobuf::uint32 _has_bits_[" + SimpleItoa(sizeof_has_bits / 4) + "];\n";


  // To minimize padding, data members are divided into three sections:
  // (1) members assumed to align to 8 bytes
  // (2) members corresponding to message fields, re-ordered to optimize
  //     alignment.
  // (3) members assumed to align to 4 bytes.

  // Members assumed to align to 8 bytes:

  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "::google::protobuf::internal::ExtensionSet _extensions_;\n"
      "\n");
  }

  if (UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(
      "::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;\n");
  } else {
    printer->Print(
      "::google::protobuf::internal::ArenaStringPtr _unknown_fields_;\n"
      "::google::protobuf::Arena* _arena_ptr_;\n"
      "\n");
  }

  if (SupportsArenas(descriptor_)) {
    printer->Print(
      "friend class ::google::protobuf::Arena;\n"
      "typedef void InternalArenaConstructable_;\n"
      "typedef void DestructorSkippable_;\n");
  }

  if (HasFieldPresence(descriptor_->file())) {
    // _has_bits_ is frequently accessed, so to reduce code size and improve
    // speed, it should be close to the start of the object.  But, try not to
    // waste space:_has_bits_ by itself always makes sense if its size is a
    // multiple of 8, but, otherwise, maybe _has_bits_ and cached_size_ together
    // will work well.
    printer->Print(has_bits_decl.c_str());
    if ((sizeof_has_bits % 8) != 0) {
      printer->Print(cached_size_decl.c_str());
      need_to_emit_cached_size = false;
    }
  } else {
    // Without field presence, we need another way to disambiguate the default
    // instance, because the default instance's submessage fields (if any) store
    // pointers to the default instances of the submessages even when they
    // aren't present. Alternatives to this approach might be to (i) use a
    // tagged pointer on all message fields, setting a tag bit for "not really
    // present, just default instance"; or (ii) comparing |this| against the
    // return value from GeneratedMessageFactory::GetPrototype() in all
    // has_$field$() calls. However, both of these options are much more
    // expensive (in code size and CPU overhead) than just checking a field in
    // the message. Long-term, the best solution would be to rearchitect the
    // default instance design not to store pointers to submessage default
    // instances, and have reflection get those some other way; but that change
    // would have too much impact on proto2.
    printer->Print(
      "bool _is_default_instance_;\n");
  }

  // Field members:

  // List fields which doesn't belong to any oneof
  vector<const FieldDescriptor*> fields;
  hash_map<string, int> fieldname_to_chunk;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {
      const FieldDescriptor* field = descriptor_->field(i);
      fields.push_back(field);
      fieldname_to_chunk[FieldName(field)] = i / 8;
    }
  }
  OptimizePadding(&fields);
  // Emit some private and static members
  runs_of_fields_ = vector< vector<string> >(1);
  for (int i = 0; i < fields.size(); ++i) {
    const FieldDescriptor* field = fields[i];
    const FieldGenerator& generator = field_generators_.get(field);
    generator.GenerateStaticMembers(printer);
    generator.GeneratePrivateMembers(printer);
    if (CanClearByZeroing(field)) {
      const string& fieldname = FieldName(field);
      if (!runs_of_fields_.back().empty() &&
          (fieldname_to_chunk[runs_of_fields_.back().back()] !=
           fieldname_to_chunk[fieldname])) {
        runs_of_fields_.push_back(vector<string>());
      }
      runs_of_fields_.back().push_back(fieldname);
    } else if (!runs_of_fields_.back().empty()) {
      runs_of_fields_.push_back(vector<string>());
    }
  }

  // For each oneof generate a union
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "union $camel_oneof_name$Union {\n"
        // explicit empty constructor is needed when union contains
        // ArenaStringPtr members for string fields.
        "  $camel_oneof_name$Union() {}\n",
        "camel_oneof_name",
        UnderscoresToCamelCase(descriptor_->oneof_decl(i)->name(), true));
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      field_generators_.get(descriptor_->oneof_decl(i)->
                            field(j)).GeneratePrivateMembers(printer);
    }
    printer->Outdent();
    printer->Print(
        "} $oneof_name$_;\n",
        "oneof_name", descriptor_->oneof_decl(i)->name());
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      field_generators_.get(descriptor_->oneof_decl(i)->
                            field(j)).GenerateStaticMembers(printer);
    }
  }

  // Members assumed to align to 4 bytes:

  if (need_to_emit_cached_size) {
    printer->Print(cached_size_decl.c_str());
    need_to_emit_cached_size = false;
  }

  // Generate _oneof_case_.
  if (descriptor_->oneof_decl_count() > 0) {
    printer->Print(vars,
      "::google::protobuf::uint32 _oneof_case_[$oneof_decl_count$];\n"
      "\n");
  }

  // Generate _any_metadata_ for the Any type.
  if (IsAnyMessage(descriptor_)) {
    printer->Print(vars,
      "::google::protobuf::internal::AnyMetadata _any_metadata_;\n");
  }

  // Declare AddDescriptors(), BuildDescriptors(), and ShutdownFile() as
  // friends so that they can access private static variables like
  // default_instance_ and reflection_.
  PrintHandlingOptionalStaticInitializers(
    descriptor_->file(), printer,
    // With static initializers.
    "friend void $dllexport_decl$ $adddescriptorsname$();\n",
    // Without.
    "friend void $dllexport_decl$ $adddescriptorsname$_impl();\n",
    // Vars.
    "dllexport_decl", options_.dllexport_decl,
    "adddescriptorsname",
    GlobalAddDescriptorsName(descriptor_->file()->name()));

  printer->Print(
    "friend void $assigndescriptorsname$();\n"
    "friend void $shutdownfilename$();\n"
    "\n",
    "assigndescriptorsname",
      GlobalAssignDescriptorsName(descriptor_->file()->name()),
    "shutdownfilename", GlobalShutdownFileName(descriptor_->file()->name()));

  printer->Print(
    "void InitAsDefaultInstance();\n"
    "static $classname$* default_instance_;\n",
    "classname", classname_);

  printer->Outdent();
  printer->Print(vars, "};");
  GOOGLE_DCHECK(!need_to_emit_cached_size);
}

void MessageGenerator::
GenerateDependentInlineMethods(io::Printer* printer) {
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // map entry message doesn't need inline methods. Since map entry message
    // cannot be a top level class, we just need to avoid calling
    // GenerateInlineMethods here.
    if (IsMapEntryMessage(descriptor_->nested_type(i))) continue;
    nested_generators_[i]->GenerateDependentInlineMethods(printer);
    printer->Print(kThinSeparator);
    printer->Print("\n");
  }

  GenerateDependentFieldAccessorDefinitions(printer);
}

void MessageGenerator::
GenerateInlineMethods(io::Printer* printer, bool is_inline) {
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // map entry message doesn't need inline methods. Since map entry message
    // cannot be a top level class, we just need to avoid calling
    // GenerateInlineMethods here.
    if (IsMapEntryMessage(descriptor_->nested_type(i))) continue;
    nested_generators_[i]->GenerateInlineMethods(printer, is_inline);
    printer->Print(kThinSeparator);
    printer->Print("\n");
  }

  GenerateFieldAccessorDefinitions(printer, is_inline);

  // Generate oneof_case() functions.
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    map<string, string> vars;
    vars["class_name"] = classname_;
    vars["camel_oneof_name"] = UnderscoresToCamelCase(
        descriptor_->oneof_decl(i)->name(), true);
    vars["oneof_name"] = descriptor_->oneof_decl(i)->name();
    vars["oneof_index"] = SimpleItoa(descriptor_->oneof_decl(i)->index());
    vars["inline"] = is_inline ? "inline " : "";
    printer->Print(
        vars,
        "$inline$"
        "$class_name$::$camel_oneof_name$Case $class_name$::"
        "$oneof_name$_case() const {\n"
        "  return $class_name$::$camel_oneof_name$Case("
        "_oneof_case_[$oneof_index$]);\n"
        "}\n");
  }
}

void MessageGenerator::
GenerateDescriptorDeclarations(io::Printer* printer) {
  if (!IsMapEntryMessage(descriptor_)) {
    printer->Print(
      "const ::google::protobuf::Descriptor* $name$_descriptor_ = NULL;\n"
      "const ::google::protobuf::internal::GeneratedMessageReflection*\n"
      "  $name$_reflection_ = NULL;\n",
      "name", classname_);
  } else {
    printer->Print(
      "const ::google::protobuf::Descriptor* $name$_descriptor_ = NULL;\n",
      "name", classname_);
  }

  // Generate oneof default instance for reflection usage.
  if (descriptor_->oneof_decl_count() > 0) {
    printer->Print("struct $name$OneofInstance {\n",
                   "name", classname_);
    for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
      for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
        const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
        printer->Print("  ");
        if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE ||
            (field->cpp_type() == FieldDescriptor::CPPTYPE_STRING &&
             EffectiveStringCType(field) != FieldOptions::STRING)) {
          printer->Print("const ");
        }
        field_generators_.get(field).GeneratePrivateMembers(printer);
      }
    }

    printer->Print("}* $name$_default_oneof_instance_ = NULL;\n",
                   "name", classname_);
  }

  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    nested_generators_[i]->GenerateDescriptorDeclarations(printer);
  }

  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    printer->Print(
      "const ::google::protobuf::EnumDescriptor* $name$_descriptor_ = NULL;\n",
      "name", ClassName(descriptor_->enum_type(i), false));
  }
}

void MessageGenerator::
GenerateDescriptorInitializer(io::Printer* printer, int index) {
  // TODO(kenton):  Passing the index to this method is redundant; just use
  //   descriptor_->index() instead.
  map<string, string> vars;
  vars["classname"] = classname_;
  vars["index"] = SimpleItoa(index);

  // Obtain the descriptor from the parent's descriptor.
  if (descriptor_->containing_type() == NULL) {
    printer->Print(vars,
      "$classname$_descriptor_ = file->message_type($index$);\n");
  } else {
    vars["parent"] = ClassName(descriptor_->containing_type(), false);
    printer->Print(vars,
      "$classname$_descriptor_ = "
        "$parent$_descriptor_->nested_type($index$);\n");
  }

  if (IsMapEntryMessage(descriptor_)) return;

  // Generate the offsets.
  GenerateOffsets(printer);

  const bool pass_pool_and_factory = false;
  vars["fn"] = pass_pool_and_factory ?
      "new ::google::protobuf::internal::GeneratedMessageReflection" :
      "::google::protobuf::internal::GeneratedMessageReflection"
      "::NewGeneratedMessageReflection";
  // Construct the reflection object.
  printer->Print(vars,
    "$classname$_reflection_ =\n"
    "  $fn$(\n"
    "    $classname$_descriptor_,\n"
    "    $classname$::default_instance_,\n"
    "    $classname$_offsets_,\n");
  if (!HasFieldPresence(descriptor_->file())) {
    // If we don't have field presence, then _has_bits_ does not exist.
    printer->Print(vars,
    "    -1,\n");
  } else {
    printer->Print(vars,
    "    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET($classname$, _has_bits_[0]),\n");
  }

  // Unknown field offset: either points to the unknown field set if embedded
  // directly, or indicates that the unknown field set is stored as part of the
  // internal metadata if not.
  if (UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(vars,
    "    -1,\n");
  } else {
    printer->Print(vars,
    "    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET("
      "$classname$, _unknown_fields_),\n");
  }

  if (descriptor_->extension_range_count() > 0) {
    printer->Print(vars,
      "    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET("
        "$classname$, _extensions_),\n");
  } else {
    // No extensions.
    printer->Print(vars,
      "    -1,\n");
  }

  if (descriptor_->oneof_decl_count() > 0) {
    printer->Print(vars,
    "    $classname$_default_oneof_instance_,\n"
    "    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET("
      "$classname$, _oneof_case_[0]),\n");
  }

  if (pass_pool_and_factory) {
    printer->Print(
        "    ::google::protobuf::DescriptorPool::generated_pool(),\n");
      printer->Print(vars,
                     "    ::google::protobuf::MessageFactory::generated_factory(),\n");
  }

  printer->Print(vars,
    "    sizeof($classname$),\n");

  // Arena offset: either an offset to the metadata struct that contains the
  // arena pointer and unknown field set (in a space-efficient way) if we use
  // that implementation strategy, or an offset directly to the arena pointer if
  // not (because e.g. we don't have an unknown field set).
  if (UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(vars,
    "    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET("
    "$classname$, _internal_metadata_),\n");
  } else {
    printer->Print(vars,
    "    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET("
    "$classname$, _arena_),\n");
  }

  // is_default_instance_ offset.
  if (HasFieldPresence(descriptor_->file())) {
    printer->Print(vars,
    "    -1);\n");
  } else {
    printer->Print(vars,
    "    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET("
    "$classname$, _is_default_instance_));\n");
  }

  // Handle nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    nested_generators_[i]->GenerateDescriptorInitializer(printer, i);
  }

  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    enum_generators_[i]->GenerateDescriptorInitializer(printer, i);
  }
}

void MessageGenerator::
GenerateTypeRegistrations(io::Printer* printer) {
  // Register this message type with the message factory.
  if (!IsMapEntryMessage(descriptor_)) {
    printer->Print(
      "::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(\n"
      "    $classname$_descriptor_, &$classname$::default_instance());\n",
      "classname", classname_);
  }
  else {
    map<string, string> vars;
    CollectMapInfo(descriptor_, &vars);
    vars["classname"] = classname_;

    const FieldDescriptor* val = descriptor_->FindFieldByName("value");
    if (descriptor_->file()->syntax() == FileDescriptor::SYNTAX_PROTO2 &&
        val->type() == FieldDescriptor::TYPE_ENUM) {
      const EnumValueDescriptor* default_value = val->default_value_enum();
      vars["default_enum_value"] = Int32ToString(default_value->number());
    } else {
      vars["default_enum_value"] = "0";
    }

    printer->Print(vars,
      "::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(\n"
      "      $classname$_descriptor_,\n"
      "      ::google::protobuf::internal::MapEntry<\n"
      "          $key$,\n"
      "          $val$,\n"
      "          $key_wire_type$,\n"
      "          $val_wire_type$,\n"
      "          $default_enum_value$>::CreateDefaultInstance(\n"
      "              $classname$_descriptor_));\n");
  }

  // Handle nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    nested_generators_[i]->GenerateTypeRegistrations(printer);
  }
}

void MessageGenerator::
GenerateDefaultInstanceAllocator(io::Printer* printer) {
  // Construct the default instances of all fields, as they will be used
  // when creating the default instance of the entire message.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(descriptor_->field(i))
                     .GenerateDefaultInstanceAllocator(printer);
  }

  if (IsMapEntryMessage(descriptor_)) return;

  // Construct the default instance.  We can't call InitAsDefaultInstance() yet
  // because we need to make sure all default instances that this one might
  // depend on are constructed first.
  printer->Print(
    "$classname$::default_instance_ = new $classname$();\n",
    "classname", classname_);

  if ((descriptor_->oneof_decl_count() > 0) &&
      HasDescriptorMethods(descriptor_->file())) {
    printer->Print(
    "$classname$_default_oneof_instance_ = new $classname$OneofInstance();\n",
    "classname", classname_);
  }

  // Handle nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    nested_generators_[i]->GenerateDefaultInstanceAllocator(printer);
  }

}

void MessageGenerator::
GenerateDefaultInstanceInitializer(io::Printer* printer) {
  printer->Print(
    "$classname$::default_instance_->InitAsDefaultInstance();\n",
    "classname", classname_);

  // Register extensions.
  for (int i = 0; i < descriptor_->extension_count(); i++) {
    extension_generators_[i]->GenerateRegistration(printer);
  }

  // Handle nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // map entry message doesn't need to initialize default instance manually.
    // Since map entry message cannot be a top level class, we just need to
    // avoid calling DefaultInstanceInitializer here.
    if (IsMapEntryMessage(descriptor_->nested_type(i))) continue;
    nested_generators_[i]->GenerateDefaultInstanceInitializer(printer);
  }
}

void MessageGenerator::
GenerateShutdownCode(io::Printer* printer) {
  printer->Print(
    "delete $classname$::default_instance_;\n",
    "classname", classname_);

  if (HasDescriptorMethods(descriptor_->file())) {
    if (descriptor_->oneof_decl_count() > 0) {
      printer->Print(
        "delete $classname$_default_oneof_instance_;\n",
        "classname", classname_);
    }
    printer->Print(
      "delete $classname$_reflection_;\n",
      "classname", classname_);
  }

  // Handle default instances of fields.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(descriptor_->field(i))
                     .GenerateShutdownCode(printer);
  }

  // Handle nested types.
  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    if (IsMapEntryMessage(descriptor_->nested_type(i))) continue;
    nested_generators_[i]->GenerateShutdownCode(printer);
  }
}

void MessageGenerator::
GenerateClassMethods(io::Printer* printer) {
  if (IsAnyMessage(descriptor_)) {
    printer->Print(
      "void $classname$::PackFrom(const ::google::protobuf::Message& message) {\n"
      "  _any_metadata_.PackFrom(message);\n"
      "}\n"
      "\n"
      "bool $classname$::UnpackTo(::google::protobuf::Message* message) const {\n"
      "  return _any_metadata_.UnpackTo(message);\n"
      "}\n"
      "\n",
      "classname", classname_);
  }

  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    enum_generators_[i]->GenerateMethods(printer);
  }

  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    // map entry message doesn't need class methods. Since map entry message
    // cannot be a top level class, we just need to avoid calling
    // GenerateClassMethods here.
    if (IsMapEntryMessage(descriptor_->nested_type(i))) continue;
    nested_generators_[i]->GenerateClassMethods(printer);
    printer->Print("\n");
    printer->Print(kThinSeparator);
    printer->Print("\n");
  }

  // Generate non-inline field definitions.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(descriptor_->field(i))
                     .GenerateNonInlineAccessorDefinitions(printer);
  }

  // Generate field number constants.
  printer->Print("#if !defined(_MSC_VER) || _MSC_VER >= 1900\n");
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor *field = descriptor_->field(i);
    printer->Print(
      "const int $classname$::$constant_name$;\n",
      "classname", ClassName(FieldScope(field), false),
      "constant_name", FieldConstantName(field));
  }
  printer->Print(
    "#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900\n"
    "\n");

  // Define extension identifiers.
  for (int i = 0; i < descriptor_->extension_count(); i++) {
    extension_generators_[i]->GenerateDefinition(printer);
  }

  GenerateStructors(printer);
  printer->Print("\n");

  if (descriptor_->oneof_decl_count() > 0) {
    GenerateOneofClear(printer);
    printer->Print("\n");
  }

  if (HasGeneratedMethods(descriptor_->file())) {
    GenerateClear(printer);
    printer->Print("\n");

    GenerateMergeFromCodedStream(printer);
    printer->Print("\n");

    GenerateSerializeWithCachedSizes(printer);
    printer->Print("\n");

    if (HasFastArraySerialization(descriptor_->file())) {
      GenerateSerializeWithCachedSizesToArray(printer);
      printer->Print("\n");
    }

    GenerateByteSize(printer);
    printer->Print("\n");

    GenerateMergeFrom(printer);
    printer->Print("\n");

    GenerateCopyFrom(printer);
    printer->Print("\n");

    GenerateIsInitialized(printer);
    printer->Print("\n");
  }

  GenerateSwap(printer);
  printer->Print("\n");

  if (HasDescriptorMethods(descriptor_->file())) {
    printer->Print(
      "::google::protobuf::Metadata $classname$::GetMetadata() const {\n"
      "  protobuf_AssignDescriptorsOnce();\n"
      "  ::google::protobuf::Metadata metadata;\n"
      "  metadata.descriptor = $classname$_descriptor_;\n"
      "  metadata.reflection = $classname$_reflection_;\n"
      "  return metadata;\n"
      "}\n"
      "\n",
      "classname", classname_);
  } else {
    printer->Print(
      "::std::string $classname$::GetTypeName() const {\n"
      "  return \"$type_name$\";\n"
      "}\n"
      "\n",
      "classname", classname_,
      "type_name", descriptor_->full_name());
  }

}

void MessageGenerator::
GenerateOffsets(io::Printer* printer) {
  printer->Print(
    "static const int $classname$_offsets_[$field_count$] = {\n",
    "classname", classname_,
    "field_count", SimpleItoa(max(
        1, descriptor_->field_count() + descriptor_->oneof_decl_count())));
  printer->Indent();

  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (field->containing_oneof()) {
      printer->Print(
          "PROTO2_GENERATED_DEFAULT_ONEOF_FIELD_OFFSET("
          "$classname$_default_oneof_instance_, $name$_),\n",
          "classname", classname_,
          "name", FieldName(field));
    } else {
      printer->Print(
          "GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET($classname$, "
                                                 "$name$_),\n",
          "classname", classname_,
          "name", FieldName(field));
    }
  }

  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    const OneofDescriptor* oneof = descriptor_->oneof_decl(i);
    printer->Print(
      "GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET($classname$, $name$_),\n",
      "classname", classname_,
      "name", oneof->name());
  }

  printer->Outdent();
  printer->Print("};\n");
}

void MessageGenerator::
GenerateSharedConstructorCode(io::Printer* printer) {
  printer->Print(
    "void $classname$::SharedCtor() {\n",
    "classname", classname_);
  printer->Indent();

  if (!HasFieldPresence(descriptor_->file())) {
    printer->Print(
      "  _is_default_instance_ = false;\n");
  }

  printer->Print(StrCat(
      uses_string_ ? "::google::protobuf::internal::GetEmptyString();\n" : "",
      "_cached_size_ = 0;\n").c_str());

  if (PreserveUnknownFields(descriptor_) &&
      !UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(
        "_unknown_fields_.UnsafeSetDefault(\n"
        "    &::google::protobuf::internal::GetEmptyStringAlreadyInited());\n");
  }

  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {
      field_generators_.get(descriptor_->field(i))
          .GenerateConstructorCode(printer);
    }
  }

  if (HasFieldPresence(descriptor_->file())) {
    printer->Print(
      "::memset(_has_bits_, 0, sizeof(_has_bits_));\n");
  }

  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "clear_has_$oneof_name$();\n",
        "oneof_name", descriptor_->oneof_decl(i)->name());
  }

  printer->Outdent();
  printer->Print("}\n\n");
}

void MessageGenerator::
GenerateSharedDestructorCode(io::Printer* printer) {
  printer->Print(
    "void $classname$::SharedDtor() {\n",
    "classname", classname_);
  printer->Indent();
  if (SupportsArenas(descriptor_)) {
    // Do nothing when the message is allocated in an arena.
    printer->Print(
      "if (GetArenaNoVirtual() != NULL) {\n"
      "  return;\n"
      "}\n"
      "\n");
  }

  // Write the desctructor for _unknown_fields_ in lite runtime.
  if (PreserveUnknownFields(descriptor_) &&
      !UseUnknownFieldSet(descriptor_->file())) {
    if (SupportsArenas(descriptor_)) {
      printer->Print(
          "_unknown_fields_.Destroy(\n"
          "    &::google::protobuf::internal::GetEmptyStringAlreadyInited(),\n"
          "    GetArenaNoVirtual());\n");
    } else {
      printer->Print(
          "_unknown_fields_.DestroyNoArena(\n"
          "    &::google::protobuf::internal::GetEmptyStringAlreadyInited());\n");
    }
  }

  // Write the destructors for each field except oneof members.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (!descriptor_->field(i)->containing_oneof()) {
      field_generators_.get(descriptor_->field(i))
                       .GenerateDestructorCode(printer);
    }
  }

  // Generate code to destruct oneofs. Clearing should do the work.
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "if (has_$oneof_name$()) {\n"
        "  clear_$oneof_name$();\n"
        "}\n",
        "oneof_name", descriptor_->oneof_decl(i)->name());
  }

  PrintHandlingOptionalStaticInitializers(
    descriptor_->file(), printer,
    // With static initializers.
    "if (this != default_instance_) {\n",
    // Without.
    "if (this != &default_instance()) {\n");

  // We need to delete all embedded messages.
  // TODO(kenton):  If we make unset messages point at default instances
  //   instead of NULL, then it would make sense to move this code into
  //   MessageFieldGenerator::GenerateDestructorCode().
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    if (!field->is_repeated() &&
        field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      // Skip oneof members
      if (!field->containing_oneof()) {
        printer->Print(
            "  delete $name$_;\n",
            "name", FieldName(field));
      }
    }
  }

  printer->Outdent();
  printer->Print(
    "  }\n"
    "}\n"
    "\n");
}

void MessageGenerator::
GenerateArenaDestructorCode(io::Printer* printer) {
  // Generate the ArenaDtor() method. Track whether any fields actually produced
  // code that needs to be called.
  printer->Print(
      "void $classname$::ArenaDtor(void* object) {\n",
      "classname", classname_);
  printer->Indent();

  // This code is placed inside a static method, rather than an ordinary one,
  // since that simplifies Arena's destructor list (ordinary function pointers
  // rather than member function pointers). _this is the object being
  // destructed.
  printer->Print(
      "$classname$* _this = reinterpret_cast< $classname$* >(object);\n"
      // avoid an "unused variable" warning in case no fields have dtor code.
      "(void)_this;\n",
      "classname", classname_);

  bool need_registration = false;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (field_generators_.get(descriptor_->field(i))
                         .GenerateArenaDestructorCode(printer)) {
      need_registration = true;
    }
  }
  printer->Outdent();
  printer->Print(
      "}\n");

  if (need_registration) {
    printer->Print(
        "inline void $classname$::RegisterArenaDtor(::google::protobuf::Arena* arena) {\n"
        "  if (arena != NULL) {\n"
        "    arena->OwnCustomDestructor(this, &$classname$::ArenaDtor);\n"
        "  }\n"
        "}\n",
        "classname", classname_);
  } else {
    printer->Print(
        "void $classname$::RegisterArenaDtor(::google::protobuf::Arena* arena) {\n"
        "}\n",
        "classname", classname_);
  }
}

void MessageGenerator::
GenerateStructors(io::Printer* printer) {
  string superclass;
  if (use_dependent_base_) {
    superclass =
        DependentBaseClassTemplateName(descriptor_) + "<" + classname_ + ">";
  } else {
    superclass = SuperClassName(descriptor_);
  }
  string initializer_with_arena = superclass + "()";

  if (descriptor_->extension_range_count() > 0) {
    initializer_with_arena += ",\n  _extensions_(arena)";
  }

  if (UseUnknownFieldSet(descriptor_->file())) {
    initializer_with_arena += ",\n  _internal_metadata_(arena)";
  } else {
    initializer_with_arena += ",\n  _arena_ptr_(arena)";
  }

  // Initialize member variables with arena constructor.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    bool has_arena_constructor = descriptor_->field(i)->is_repeated();
    if (has_arena_constructor) {
      initializer_with_arena += string(",\n  ") +
          FieldName(descriptor_->field(i)) + string("_(arena)");
    }
  }

  if (IsAnyMessage(descriptor_)) {
    initializer_with_arena += ",\n  _any_metadata_(&type_url, &value_)";
  }

  string initializer_null;
  initializer_null = (UseUnknownFieldSet(descriptor_->file()) ?
    ", _internal_metadata_(NULL)" : ", _arena_ptr_(NULL)");
  if (IsAnyMessage(descriptor_)) {
    initializer_null += ", _any_metadata_(&type_url_, &value_)";
  }

  printer->Print(
      "$classname$::$classname$()\n"
      "  : $superclass$()$initializer$ {\n"
      "  SharedCtor();\n"
      "  // @@protoc_insertion_point(constructor:$full_name$)\n"
      "}\n",
      "classname", classname_,
      "superclass", superclass,
      "full_name", descriptor_->full_name(),
      "initializer", initializer_null);

  if (SupportsArenas(descriptor_)) {
    printer->Print(
        "\n"
        "$classname$::$classname$(::google::protobuf::Arena* arena)\n"
        "  : $initializer$ {\n"
        "  SharedCtor();\n"
        "  RegisterArenaDtor(arena);\n"
        "  // @@protoc_insertion_point(arena_constructor:$full_name$)\n"
        "}\n",
        "initializer", initializer_with_arena,
        "classname", classname_,
        "superclass", superclass,
        "full_name", descriptor_->full_name());
  }

  printer->Print(
    "\n"
    "void $classname$::InitAsDefaultInstance() {\n",
    "classname", classname_);

  if (!HasFieldPresence(descriptor_->file())) {
    printer->Print(
      "  _is_default_instance_ = true;\n");
  }

  // The default instance needs all of its embedded message pointers
  // cross-linked to other default instances.  We can't do this initialization
  // in the constructor because some other default instances may not have been
  // constructed yet at that time.
  // TODO(kenton):  Maybe all message fields (even for non-default messages)
  //   should be initialized to point at default instances rather than NULL?
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    if (!field->is_repeated() &&
        field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE &&
        (field->containing_oneof() == NULL ||
         HasDescriptorMethods(descriptor_->file()))) {
      string name;
      if (field->containing_oneof()) {
        name = classname_ + "_default_oneof_instance_->";
      }
      name += FieldName(field);
      PrintHandlingOptionalStaticInitializers(
        descriptor_->file(), printer,
        // With static initializers.
        "  $name$_ = const_cast< $type$*>(&$type$::default_instance());\n",
        // Without.
        "  $name$_ = const_cast< $type$*>(\n"
        "      $type$::internal_default_instance());\n",
        // Vars.
        "name", name,
        "type", FieldMessageTypeName(field));
    } else if (field->containing_oneof() &&
               HasDescriptorMethods(descriptor_->file())) {
      field_generators_.get(descriptor_->field(i))
          .GenerateConstructorCode(printer);
    }
  }
  printer->Print(
    "}\n"
    "\n");

  // Generate the copy constructor.
  printer->Print(
    "$classname$::$classname$(const $classname$& from)\n"
    "  : $superclass$()",
    "classname", classname_,
    "superclass", superclass,
    "full_name", descriptor_->full_name());
  if (UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(
        ",\n    _internal_metadata_(NULL)");
  } else if (!UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(",\n    _arena_ptr_(NULL)");
  }
  if (IsAnyMessage(descriptor_)) {
    printer->Print(",\n    _any_metadata_(&type_url_, &value_)");
  }
  printer->Print(" {\n");
  printer->Print(
    "  SharedCtor();\n"
    "  MergeFrom(from);\n"
    "  // @@protoc_insertion_point(copy_constructor:$full_name$)\n"
    "}\n"
    "\n",
    "classname", classname_,
    "superclass", superclass,
    "full_name", descriptor_->full_name());

  // Generate the shared constructor code.
  GenerateSharedConstructorCode(printer);

  // Generate the destructor.
  printer->Print(
    "$classname$::~$classname$() {\n"
    "  // @@protoc_insertion_point(destructor:$full_name$)\n"
    "  SharedDtor();\n"
    "}\n"
    "\n",
    "classname", classname_,
    "full_name", descriptor_->full_name());

  // Generate the shared destructor code.
  GenerateSharedDestructorCode(printer);

  // Generate the arena-specific destructor code.
  if (SupportsArenas(descriptor_)) {
    GenerateArenaDestructorCode(printer);
  }

  // Generate SetCachedSize.
  printer->Print(
    "void $classname$::SetCachedSize(int size) const {\n"
    "  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();\n"
    "  _cached_size_ = size;\n"
    "  GOOGLE_SAFE_CONCURRENT_WRITES_END();\n"
    "}\n",
    "classname", classname_);

  // Only generate this member if it's not disabled.
  if (HasDescriptorMethods(descriptor_->file()) &&
      !descriptor_->options().no_standard_descriptor_accessor()) {
    printer->Print(
      "const ::google::protobuf::Descriptor* $classname$::descriptor() {\n"
      "  protobuf_AssignDescriptorsOnce();\n"
      "  return $classname$_descriptor_;\n"
      "}\n"
      "\n",
      "classname", classname_,
      "adddescriptorsname",
      GlobalAddDescriptorsName(descriptor_->file()->name()));
  }

  printer->Print(
    "const $classname$& $classname$::default_instance() {\n",
    "classname", classname_);

  PrintHandlingOptionalStaticInitializers(
    descriptor_->file(), printer,
    // With static initializers.
    "  if (default_instance_ == NULL) $adddescriptorsname$();\n",
    // Without.
    "  $adddescriptorsname$();\n",
    // Vars.
    "adddescriptorsname",
    GlobalAddDescriptorsName(descriptor_->file()->name()));

  printer->Print(
    "  return *default_instance_;\n"
    "}\n"
    "\n"
    "$classname$* $classname$::default_instance_ = NULL;\n"
    "\n",
    "classname", classname_);

  if (SupportsArenas(descriptor_)) {
    printer->Print(
      "$classname$* $classname$::New(::google::protobuf::Arena* arena) const {\n"
      "  return ::google::protobuf::Arena::CreateMessage<$classname$>(arena);\n"
      "}\n",
      "classname", classname_);
  } else {
    printer->Print(
      "$classname$* $classname$::New(::google::protobuf::Arena* arena) const {\n"
      "  $classname$* n = new $classname$;\n"
      "  if (arena != NULL) {\n"
      "    arena->Own(n);\n"
      "  }\n"
      "  return n;\n"
      "}\n",
      "classname", classname_);
  }

}

// Return the number of bits set in n, a non-negative integer.
static int popcnt(uint32 n) {
  int result = 0;
  while (n != 0) {
    result += (n & 1);
    n = n / 2;
  }
  return result;
}

void MessageGenerator::
GenerateClear(io::Printer* printer) {
  printer->Print("void $classname$::Clear() {\n",
                 "classname", classname_);
  printer->Indent();

  // Step 1: Extensions
  if (descriptor_->extension_range_count() > 0) {
    printer->Print("_extensions_.Clear();\n");
  }

  // Step 2: Everything but extensions, repeateds, unions.
  // These are handled in chunks of 8.  The first chunk is
  // the non-extensions-non-repeateds-non-unions in
  //  descriptor_->field(0), descriptor_->field(1), ... descriptor_->field(7),
  // and the second chunk is the same for
  //  descriptor_->field(8), descriptor_->field(9), ... descriptor_->field(15),
  // etc.
  set<int> step2_indices;
  hash_map<string, int> fieldname_to_chunk;
  hash_map<int, string> memsets_for_chunk;
  hash_map<int, int> memset_field_count_for_chunk;
  hash_set<string> handled;  // fields that appear anywhere in memsets_for_chunk
  hash_map<int, uint32> fields_mask_for_chunk;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (!field->is_repeated() && !field->containing_oneof()) {
      step2_indices.insert(i);
      int chunk = i / 8;
      fieldname_to_chunk[FieldName(field)] = chunk;
      fields_mask_for_chunk[chunk] |= static_cast<uint32>(1) << (i % 32);
    }
  }

  // Step 2a: Greedily seek runs of fields that can be cleared by memset-to-0.
  // The generated code uses two macros to help it clear runs of fields:
  // ZR_HELPER_(f1) - ZR_HELPER_(f0) computes the difference, in bytes, of the
  // positions of two fields in the Message.
  // ZR_ zeroes a non-empty range of fields via memset.
  const char* macros =
      "#define ZR_HELPER_(f) reinterpret_cast<char*>(\\\n"
      "  &reinterpret_cast<$classname$*>(16)->f)\n\n"
      "#define ZR_(first, last) do {\\\n"
      "  ::memset(&first, 0,\\\n"
      "           ZR_HELPER_(last) - ZR_HELPER_(first) + sizeof(last));\\\n"
      "} while (0)\n\n";
  for (int i = 0; i < runs_of_fields_.size(); i++) {
    const vector<string>& run = runs_of_fields_[i];
    if (run.size() < 2) continue;
    const string& first_field_name = run[0];
    const string& last_field_name = run.back();
    int chunk = fieldname_to_chunk[run[0]];
    memsets_for_chunk[chunk].append(
      "ZR_(" + first_field_name + "_, " + last_field_name + "_);\n");
    for (int j = 0; j < run.size(); j++) {
      GOOGLE_DCHECK_EQ(chunk, fieldname_to_chunk[run[j]]);
      handled.insert(run[j]);
    }
    memset_field_count_for_chunk[chunk] += run.size();
  }
  const bool macros_are_needed = handled.size() > 0;
  if (macros_are_needed) {
    printer->Outdent();
    printer->Print(macros,
                   "classname", classname_);
    printer->Indent();
  }
  // Step 2b: Finish step 2, ignoring fields handled in step 2a.
  int last_index = -1;
  bool chunk_block_in_progress = false;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    if (step2_indices.count(i) == 0) continue;
    const FieldDescriptor* field = descriptor_->field(i);
    const string fieldname = FieldName(field);
    if (i / 8 != last_index / 8 || last_index < 0) {
      // End previous chunk, if there was one.
      if (chunk_block_in_progress) {
        printer->Outdent();
        printer->Print("}\n");
        chunk_block_in_progress = false;
      }
      // Start chunk.
      const string& memsets = memsets_for_chunk[i / 8];
      uint32 mask = fields_mask_for_chunk[i / 8];
      int count = popcnt(mask);
      GOOGLE_DCHECK_GE(count, 1);
      if (count == 1 ||
          (count <= 4 && count == memset_field_count_for_chunk[i / 8])) {
        // No "if" here because the chunk is trivial.
      } else {
        if (HasFieldPresence(descriptor_->file())) {
          printer->Print(
            "if (_has_bits_[$index$ / 32] & $mask$u) {\n",
            "index", SimpleItoa(i / 8 * 8),
            "mask", SimpleItoa(mask));
          printer->Indent();
          chunk_block_in_progress = true;
        }
      }
      printer->Print(memsets.c_str());
    }
    last_index = i;
    if (handled.count(fieldname) > 0) continue;

    // It's faster to just overwrite primitive types, but we should
    // only clear strings and messages if they were set.
    // TODO(kenton):  Let the CppFieldGenerator decide this somehow.
    bool should_check_bit =
      field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE ||
      field->cpp_type() == FieldDescriptor::CPPTYPE_STRING;

    bool have_enclosing_if = false;
    if (should_check_bit &&
        // If no field presence, then always clear strings/messages as well.
        HasFieldPresence(descriptor_->file())) {
      printer->Print("if (has_$name$()) {\n", "name", fieldname);
      printer->Indent();
      have_enclosing_if = true;
    }

    if (use_dependent_base_ && IsFieldDependent(field)) {
      printer->Print("clear_$name$();\n", "name", fieldname);
    } else {
      field_generators_.get(field).GenerateClearingCode(printer);
    }

    if (have_enclosing_if) {
      printer->Outdent();
      printer->Print("}\n");
    }
  }

  if (chunk_block_in_progress) {
    printer->Outdent();
    printer->Print("}\n");
  }
  if (macros_are_needed) {
    printer->Outdent();
    printer->Print("\n#undef ZR_HELPER_\n#undef ZR_\n\n");
    printer->Indent();
  }

  // Step 3: Repeated fields don't use _has_bits_; emit code to clear them here.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    if (field->is_repeated()) {
      if (use_dependent_base_ && IsFieldDependent(field)) {
        printer->Print("clear_$name$();\n", "name", FieldName(field));
      } else {
        field_generators_.get(field).GenerateClearingCode(printer);
      }
    }
  }

  // Step 4: Unions.
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "clear_$oneof_name$();\n",
        "oneof_name", descriptor_->oneof_decl(i)->name());
  }

  if (HasFieldPresence(descriptor_->file())) {
    // Step 5: Everything else.
    printer->Print(
      "::memset(_has_bits_, 0, sizeof(_has_bits_));\n");
  }

  if (PreserveUnknownFields(descriptor_)) {
    if (UseUnknownFieldSet(descriptor_->file())) {
      printer->Print(
        "if (_internal_metadata_.have_unknown_fields()) {\n"
        "  mutable_unknown_fields()->Clear();\n"
        "}\n");
    } else {
      if (SupportsArenas(descriptor_)) {
        printer->Print(
          "_unknown_fields_.ClearToEmpty(\n"
          "    &::google::protobuf::internal::GetEmptyStringAlreadyInited(),\n"
          "    GetArenaNoVirtual());\n");
      } else {
        printer->Print(
          "_unknown_fields_.ClearToEmptyNoArena(\n"
          "    &::google::protobuf::internal::GetEmptyStringAlreadyInited());\n");
      }
    }
  }

  printer->Outdent();
  printer->Print("}\n");
}

void MessageGenerator::
GenerateOneofClear(io::Printer* printer) {
  // Generated function clears the active field and union case (e.g. foo_case_).
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    map<string, string> oneof_vars;
    oneof_vars["classname"] = classname_;
    oneof_vars["oneofname"] = descriptor_->oneof_decl(i)->name();
    string message_class;

    printer->Print(oneof_vars,
        "void $classname$::clear_$oneofname$() {\n");
    printer->Indent();
    printer->Print(oneof_vars,
        "switch($oneofname$_case()) {\n");
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      printer->Print(
          "case k$field_name$: {\n",
          "field_name", UnderscoresToCamelCase(field->name(), true));
      printer->Indent();
      // We clear only allocated objects in oneofs
      if (!IsStringOrMessage(field)) {
        printer->Print(
            "// No need to clear\n");
      } else {
        field_generators_.get(field).GenerateClearingCode(printer);
      }
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
        ToUpper(descriptor_->oneof_decl(i)->name()));
    printer->Outdent();
    printer->Print(
        "}\n"
        "_oneof_case_[$oneof_index$] = $cap_oneof_name$_NOT_SET;\n",
        "oneof_index", SimpleItoa(i),
        "cap_oneof_name",
        ToUpper(descriptor_->oneof_decl(i)->name()));
    printer->Outdent();
    printer->Print(
        "}\n"
        "\n");
  }
}

void MessageGenerator::
GenerateSwap(io::Printer* printer) {
  if (SupportsArenas(descriptor_)) {
    // Generate the Swap member function. This is a lightweight wrapper around
    // UnsafeArenaSwap() / MergeFrom() with temporaries, depending on the memory
    // ownership situation: swapping across arenas or between an arena and a
    // heap requires copying.
    printer->Print(
        "void $classname$::Swap($classname$* other) {\n"
        "  if (other == this) return;\n"
        "  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {\n"
        "    InternalSwap(other);\n"
        "  } else {\n"
        "    $classname$ temp;\n"
        "    temp.MergeFrom(*this);\n"
        "    CopyFrom(*other);\n"
        "    other->CopyFrom(temp);\n"
        "  }\n"
        "}\n"
        "void $classname$::UnsafeArenaSwap($classname$* other) {\n"
        "  if (other == this) return;\n"
        "  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());\n"
        "  InternalSwap(other);\n"
        "}\n",
        "classname", classname_);
  } else {
    printer->Print(
        "void $classname$::Swap($classname$* other) {\n"
        "  if (other == this) return;\n"
        "  InternalSwap(other);\n"
        "}\n",
        "classname", classname_);
  }

  // Generate the UnsafeArenaSwap member function.
  printer->Print("void $classname$::InternalSwap($classname$* other) {\n",
                 "classname", classname_);
  printer->Indent();

  if (HasGeneratedMethods(descriptor_->file())) {
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = descriptor_->field(i);
      field_generators_.get(field).GenerateSwappingCode(printer);
    }

    for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
      printer->Print(
        "std::swap($oneof_name$_, other->$oneof_name$_);\n"
        "std::swap(_oneof_case_[$i$], other->_oneof_case_[$i$]);\n",
        "oneof_name", descriptor_->oneof_decl(i)->name(),
        "i", SimpleItoa(i));
    }

    if (HasFieldPresence(descriptor_->file())) {
      for (int i = 0; i < (descriptor_->field_count() + 31) / 32; ++i) {
        printer->Print("std::swap(_has_bits_[$i$], other->_has_bits_[$i$]);\n",
                       "i", SimpleItoa(i));
      }
    }

    if (PreserveUnknownFields(descriptor_)) {
      if (UseUnknownFieldSet(descriptor_->file())) {
        printer->Print(
          "_internal_metadata_.Swap(&other->_internal_metadata_);\n");
      } else {
        printer->Print("_unknown_fields_.Swap(&other->_unknown_fields_);\n");
      }
    } else {
      // Still swap internal_metadata as it may contain more than just
      // unknown fields.
      printer->Print(
        "_internal_metadata_.Swap(&other->_internal_metadata_);\n");
    }
    printer->Print("std::swap(_cached_size_, other->_cached_size_);\n");
    if (descriptor_->extension_range_count() > 0) {
      printer->Print("_extensions_.Swap(&other->_extensions_);\n");
    }
  } else {
    printer->Print("GetReflection()->Swap(this, other);");
  }

  printer->Outdent();
  printer->Print("}\n");
}

void MessageGenerator::
GenerateMergeFrom(io::Printer* printer) {
  if (HasDescriptorMethods(descriptor_->file())) {
    // Generate the generalized MergeFrom (aka that which takes in the Message
    // base class as a parameter).
    printer->Print(
      "void $classname$::MergeFrom(const ::google::protobuf::Message& from) {\n"
      "  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);\n",
      "classname", classname_);
    printer->Indent();

    // Cast the message to the proper type. If we find that the message is
    // *not* of the proper type, we can still call Merge via the reflection
    // system, as the GOOGLE_CHECK above ensured that we have the same descriptor
    // for each message.
    printer->Print(
      "const $classname$* source = \n"
      "    ::google::protobuf::internal::DynamicCastToGenerated<const $classname$>(\n"
      "        &from);\n"
      "if (source == NULL) {\n"
      "  ::google::protobuf::internal::ReflectionOps::Merge(from, this);\n"
      "} else {\n"
      "  MergeFrom(*source);\n"
      "}\n",
      "classname", classname_);

    printer->Outdent();
    printer->Print("}\n\n");
  } else {
    // Generate CheckTypeAndMergeFrom().
    printer->Print(
      "void $classname$::CheckTypeAndMergeFrom(\n"
      "    const ::google::protobuf::MessageLite& from) {\n"
      "  MergeFrom(*::google::protobuf::down_cast<const $classname$*>(&from));\n"
      "}\n"
      "\n",
      "classname", classname_);
  }

  // Generate the class-specific MergeFrom, which avoids the GOOGLE_CHECK and cast.
  printer->Print(
    "void $classname$::MergeFrom(const $classname$& from) {\n"
    "  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);\n",
    "classname", classname_);
  printer->Indent();

  // Merge Repeated fields. These fields do not require a
  // check as we can simply iterate over them.
  for (int i = 0; i < descriptor_->field_count(); ++i) {
    const FieldDescriptor* field = descriptor_->field(i);

    if (field->is_repeated()) {
      field_generators_.get(field).GenerateMergingCode(printer);
    }
  }

  // Merge oneof fields. Oneof field requires oneof case check.
  for (int i = 0; i < descriptor_->oneof_decl_count(); ++i) {
    printer->Print(
        "switch (from.$oneofname$_case()) {\n",
        "oneofname", descriptor_->oneof_decl(i)->name());
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      printer->Print(
          "case k$field_name$: {\n",
          "field_name", UnderscoresToCamelCase(field->name(), true));
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
        ToUpper(descriptor_->oneof_decl(i)->name()));
    printer->Outdent();
    printer->Print(
        "}\n");
  }

  // Merge Optional and Required fields (after a _has_bit check).
  int last_index = -1;

  for (int i = 0; i < descriptor_->field_count(); ++i) {
    const FieldDescriptor* field = descriptor_->field(i);

    if (!field->is_repeated() && !field->containing_oneof()) {
      if (HasFieldPresence(descriptor_->file())) {
        // See above in GenerateClear for an explanation of this.
        if (i / 8 != last_index / 8 || last_index < 0) {
          if (last_index >= 0) {
            printer->Outdent();
            printer->Print("}\n");
          }
          printer->Print(
            "if (from._has_bits_[$index$ / 32] & "
            "(0xffu << ($index$ % 32))) {\n",
            "index", SimpleItoa(field->index()));
          printer->Indent();
        }
      }

      last_index = i;

      bool have_enclosing_if = false;
      if (HasFieldPresence(descriptor_->file())) {
        printer->Print(
          "if (from.has_$name$()) {\n",
          "name", FieldName(field));
        printer->Indent();
        have_enclosing_if = true;
      } else {
        // Merge semantics without true field presence: primitive fields are
        // merged only if non-zero (numeric) or non-empty (string).
        have_enclosing_if = EmitFieldNonDefaultCondition(
            printer, "from.", field);
      }

      field_generators_.get(field).GenerateMergingCode(printer);

      if (have_enclosing_if) {
        printer->Outdent();
        printer->Print("}\n");
      }
    }
  }

  if (HasFieldPresence(descriptor_->file()) &&
      last_index >= 0) {
    printer->Outdent();
    printer->Print("}\n");
  }

  if (descriptor_->extension_range_count() > 0) {
    printer->Print("_extensions_.MergeFrom(from._extensions_);\n");
  }

  if (PreserveUnknownFields(descriptor_)) {
    if (UseUnknownFieldSet(descriptor_->file())) {
      printer->Print(
        "if (from._internal_metadata_.have_unknown_fields()) {\n"
        "  mutable_unknown_fields()->MergeFrom(from.unknown_fields());\n"
        "}\n");
    } else {
      printer->Print(
        "mutable_unknown_fields()->append(from.unknown_fields());\n");
    }
  }

  printer->Outdent();
  printer->Print("}\n");
}

void MessageGenerator::
GenerateCopyFrom(io::Printer* printer) {
  if (HasDescriptorMethods(descriptor_->file())) {
    // Generate the generalized CopyFrom (aka that which takes in the Message
    // base class as a parameter).
    printer->Print(
      "void $classname$::CopyFrom(const ::google::protobuf::Message& from) {\n",
      "classname", classname_);
    printer->Indent();

    printer->Print(
      "if (&from == this) return;\n"
      "Clear();\n"
      "MergeFrom(from);\n");

    printer->Outdent();
    printer->Print("}\n\n");
  }

  // Generate the class-specific CopyFrom.
  printer->Print(
    "void $classname$::CopyFrom(const $classname$& from) {\n",
    "classname", classname_);
  printer->Indent();

  printer->Print(
    "if (&from == this) return;\n"
    "Clear();\n"
    "MergeFrom(from);\n");

  printer->Outdent();
  printer->Print("}\n");
}

void MessageGenerator::
GenerateMergeFromCodedStream(io::Printer* printer) {
  if (descriptor_->options().message_set_wire_format()) {
    // Special-case MessageSet.
    printer->Print(
      "bool $classname$::MergePartialFromCodedStream(\n"
      "    ::google::protobuf::io::CodedInputStream* input) {\n",
      "classname", classname_);

    PrintHandlingOptionalStaticInitializers(
      descriptor_->file(), printer,
      // With static initializers.
      "  return _extensions_.ParseMessageSet(input, default_instance_,\n"
      "                                      mutable_unknown_fields());\n",
      // Without.
      "  return _extensions_.ParseMessageSet(input, &default_instance(),\n"
      "                                      mutable_unknown_fields());\n",
      // Vars.
      "classname", classname_);

    printer->Print(
      "}\n");
    return;
  }

  printer->Print(
    "bool $classname$::MergePartialFromCodedStream(\n"
    "    ::google::protobuf::io::CodedInputStream* input) {\n"
    "#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure\n"
    "  ::google::protobuf::uint32 tag;\n",
    "classname", classname_);

  if (!UseUnknownFieldSet(descriptor_->file())) {
    printer->Print(
      "  ::google::protobuf::io::StringOutputStream unknown_fields_string(\n"
      "      mutable_unknown_fields());\n"
      "  ::google::protobuf::io::CodedOutputStream unknown_fields_stream(\n"
      "      &unknown_fields_string);\n");
  }

  printer->Print(
    "  // @@protoc_insertion_point(parse_start:$full_name$)\n",
    "full_name", descriptor_->full_name());

  printer->Indent();
  printer->Print("for (;;) {\n");
  printer->Indent();

  google::protobuf::scoped_array<const FieldDescriptor * > ordered_fields(
      SortFieldsByNumber(descriptor_));
  uint32 maxtag = descriptor_->field_count() == 0 ? 0 :
      WireFormat::MakeTag(ordered_fields[descriptor_->field_count() - 1]);
  const int kCutoff0 = 127;               // fits in 1-byte varint
  const int kCutoff1 = (127 << 7) + 127;  // fits in 2-byte varint
  printer->Print("::std::pair< ::google::protobuf::uint32, bool> p = "
                 "input->ReadTagWithCutoff($max$);\n"
                 "tag = p.first;\n"
                 "if (!p.second) goto handle_unusual;\n",
                 "max", SimpleItoa(maxtag <= kCutoff0 ? kCutoff0 :
                                   (maxtag <= kCutoff1 ? kCutoff1 :
                                    maxtag)));
  if (descriptor_->field_count() > 0) {
    // We don't even want to print the switch() if we have no fields because
    // MSVC dislikes switch() statements that contain only a default value.

    // Note:  If we just switched on the tag rather than the field number, we
    // could avoid the need for the if() to check the wire type at the beginning
    // of each case.  However, this is actually a bit slower in practice as it
    // creates a jump table that is 8x larger and sparser, and meanwhile the
    // if()s are highly predictable.
    printer->Print("switch (::google::protobuf::internal::WireFormatLite::"
                   "GetTagFieldNumber(tag)) {\n");

    printer->Indent();

    // Find repeated messages and groups now, to simplify what follows.
    hash_set<int> fields_with_parse_loop;
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = ordered_fields[i];
      if (field->is_repeated() &&
          (field->type() == FieldDescriptor::TYPE_MESSAGE ||
           field->type() == FieldDescriptor::TYPE_GROUP)) {
        fields_with_parse_loop.insert(i);
      }
    }

    // need_label is true if we generated "goto parse_$name$" while handling the
    // previous field.
    bool need_label = false;
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = ordered_fields[i];
      const bool loops = fields_with_parse_loop.count(i) > 0;
      const bool next_field_loops = fields_with_parse_loop.count(i + 1) > 0;

      PrintFieldComment(printer, field);

      printer->Print(
        "case $number$: {\n",
        "number", SimpleItoa(field->number()));
      printer->Indent();
      const FieldGenerator& field_generator = field_generators_.get(field);

      // Emit code to parse the common, expected case.
      printer->Print("if (tag == $commontag$) {\n",
                     "commontag", SimpleItoa(WireFormat::MakeTag(field)));

      if (need_label ||
          (field->is_repeated() && !field->is_packed() && !loops)) {
        printer->Print(
            " parse_$name$:\n",
            "name", field->name());
      }
      if (loops) {
        printer->Print(
          "  DO_(input->IncrementRecursionDepth());\n"
          " parse_loop_$name$:\n",
          "name", field->name());
      }

      printer->Indent();
      if (field->is_packed()) {
        field_generator.GenerateMergeFromCodedStreamWithPacking(printer);
      } else {
        field_generator.GenerateMergeFromCodedStream(printer);
      }
      printer->Outdent();

      // Emit code to parse unexpectedly packed or unpacked values.
      if (field->is_packed()) {
        internal::WireFormatLite::WireType wiretype =
            WireFormat::WireTypeForFieldType(field->type());
        printer->Print("} else if (tag == $uncommontag$) {\n",
                       "uncommontag", SimpleItoa(
                           internal::WireFormatLite::MakeTag(
                               field->number(), wiretype)));
        printer->Indent();
        field_generator.GenerateMergeFromCodedStream(printer);
        printer->Outdent();
      } else if (field->is_packable() && !field->is_packed()) {
        internal::WireFormatLite::WireType wiretype =
            internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED;
        printer->Print("} else if (tag == $uncommontag$) {\n",
                       "uncommontag", SimpleItoa(
                           internal::WireFormatLite::MakeTag(
                               field->number(), wiretype)));
        printer->Indent();
        field_generator.GenerateMergeFromCodedStreamWithPacking(printer);
        printer->Outdent();
      }

      printer->Print(
        "} else {\n"
        "  goto handle_unusual;\n"
        "}\n");

      // switch() is slow since it can't be predicted well.  Insert some if()s
      // here that attempt to predict the next tag.
      // For non-packed repeated fields, expect the same tag again.
      if (loops) {
        printer->Print(
          "if (input->ExpectTag($tag$)) goto parse_loop_$name$;\n",
          "tag", SimpleItoa(WireFormat::MakeTag(field)),
          "name", field->name());
      } else if (field->is_repeated() && !field->is_packed()) {
        printer->Print(
          "if (input->ExpectTag($tag$)) goto parse_$name$;\n",
          "tag", SimpleItoa(WireFormat::MakeTag(field)),
          "name", field->name());
      }

      // Have we emitted "if (input->ExpectTag($next_tag$)) ..." yet?
      bool emitted_goto_next_tag = false;

      // For repeated messages/groups, we need to decrement recursion depth,
      // unless the next tag is also for a repeated message/group.
      if (loops) {
        if (next_field_loops) {
          const FieldDescriptor* next_field = ordered_fields[i + 1];
          printer->Print(
            "if (input->ExpectTag($next_tag$)) goto parse_loop_$next_name$;\n",
            "next_tag", SimpleItoa(WireFormat::MakeTag(next_field)),
            "next_name", next_field->name());
          emitted_goto_next_tag = true;
        }
        printer->Print(
          "input->UnsafeDecrementRecursionDepth();\n");
      }

      // If there are more fields, expect the next one.
      need_label = false;
      if (!emitted_goto_next_tag) {
        if (i + 1 == descriptor_->field_count()) {
          // Expect EOF.
          // TODO(kenton):  Expect group end-tag?
          printer->Print(
            "if (input->ExpectAtEnd()) goto success;\n");
        } else {
          const FieldDescriptor* next_field = ordered_fields[i + 1];
          printer->Print(
            "if (input->ExpectTag($next_tag$)) goto parse_$next_name$;\n",
            "next_tag", SimpleItoa(WireFormat::MakeTag(next_field)),
            "next_name", next_field->name());
          need_label = true;
        }
      }

      printer->Print(
        "break;\n");

      printer->Outdent();
      printer->Print("}\n\n");
    }

    printer->Print("default: {\n");
    printer->Indent();
  }

  printer->Outdent();
  printer->Print("handle_unusual:\n");
  printer->Indent();
  // If tag is 0 or an end-group tag then this must be the end of the message.
  printer->Print(
    "if (tag == 0 ||\n"
    "    ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==\n"
    "    ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {\n"
    "  goto success;\n"
    "}\n");

  // Handle extension ranges.
  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "if (");
    for (int i = 0; i < descriptor_->extension_range_count(); i++) {
      const Descriptor::ExtensionRange* range =
        descriptor_->extension_range(i);
      if (i > 0) printer->Print(" ||\n    ");

      uint32 start_tag = WireFormatLite::MakeTag(
        range->start, static_cast<WireFormatLite::WireType>(0));
      uint32 end_tag = WireFormatLite::MakeTag(
        range->end, static_cast<WireFormatLite::WireType>(0));

      if (range->end > FieldDescriptor::kMaxNumber) {
        printer->Print(
          "($start$u <= tag)",
          "start", SimpleItoa(start_tag));
      } else {
        printer->Print(
          "($start$u <= tag && tag < $end$u)",
          "start", SimpleItoa(start_tag),
          "end", SimpleItoa(end_tag));
      }
    }
    printer->Print(") {\n");
    if (PreserveUnknownFields(descriptor_)) {
      if (UseUnknownFieldSet(descriptor_->file())) {
        PrintHandlingOptionalStaticInitializers(
          descriptor_->file(), printer,
          // With static initializers.
          "  DO_(_extensions_.ParseField(tag, input, default_instance_,\n"
          "                              mutable_unknown_fields()));\n",
          // Without.
          "  DO_(_extensions_.ParseField(tag, input, &default_instance(),\n"
          "                              mutable_unknown_fields()));\n");
      } else {
        PrintHandlingOptionalStaticInitializers(
          descriptor_->file(), printer,
          // With static initializers.
          "  DO_(_extensions_.ParseField(tag, input, default_instance_,\n"
          "                              &unknown_fields_stream));\n",
          // Without.
          "  DO_(_extensions_.ParseField(tag, input, &default_instance(),\n"
          "                              &unknown_fields_stream));\n");
      }
    } else {
      PrintHandlingOptionalStaticInitializers(
        descriptor_->file(), printer,
        // With static initializers.
        "  DO_(_extensions_.ParseField(tag, input, default_instance_);\n",
        // Without.
        "  DO_(_extensions_.ParseField(tag, input, &default_instance());\n");
    }
    printer->Print(
      "  continue;\n"
      "}\n");
  }

  // We really don't recognize this tag.  Skip it.
  if (PreserveUnknownFields(descriptor_)) {
    if (UseUnknownFieldSet(descriptor_->file())) {
      printer->Print(
        "DO_(::google::protobuf::internal::WireFormat::SkipField(\n"
        "      input, tag, mutable_unknown_fields()));\n");
    } else {
      printer->Print(
        "DO_(::google::protobuf::internal::WireFormatLite::SkipField(\n"
        "    input, tag, &unknown_fields_stream));\n");
    }
  } else {
    printer->Print(
      "DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));\n");
  }

  if (descriptor_->field_count() > 0) {
    printer->Print("break;\n");
    printer->Outdent();
    printer->Print("}\n");    // default:
    printer->Outdent();
    printer->Print("}\n");    // switch
  }

  printer->Outdent();
  printer->Outdent();
  printer->Print(
    "  }\n"                   // for (;;)
    "success:\n"
    "  // @@protoc_insertion_point(parse_success:$full_name$)\n"
    "  return true;\n"
    "failure:\n"
    "  // @@protoc_insertion_point(parse_failure:$full_name$)\n"
    "  return false;\n"
    "#undef DO_\n"
    "}\n", "full_name", descriptor_->full_name());
}

void MessageGenerator::GenerateSerializeOneField(
    io::Printer* printer, const FieldDescriptor* field, bool to_array) {
  PrintFieldComment(printer, field);

  bool have_enclosing_if = false;
  if (!field->is_repeated() && HasFieldPresence(descriptor_->file())) {
    printer->Print(
      "if (has_$name$()) {\n",
      "name", FieldName(field));
    printer->Indent();
    have_enclosing_if = true;
  } else if (!HasFieldPresence(descriptor_->file())) {
    have_enclosing_if = EmitFieldNonDefaultCondition(printer, "this->", field);
  }

  if (to_array) {
    field_generators_.get(field).GenerateSerializeWithCachedSizesToArray(
        printer);
  } else {
    field_generators_.get(field).GenerateSerializeWithCachedSizes(printer);
  }

  if (have_enclosing_if) {
    printer->Outdent();
    printer->Print("}\n");
  }
  printer->Print("\n");
}

void MessageGenerator::GenerateSerializeOneExtensionRange(
    io::Printer* printer, const Descriptor::ExtensionRange* range,
    bool to_array) {
  map<string, string> vars;
  vars["start"] = SimpleItoa(range->start);
  vars["end"] = SimpleItoa(range->end);
  printer->Print(vars,
    "// Extension range [$start$, $end$)\n");
  if (to_array) {
    printer->Print(vars,
      "target = _extensions_.SerializeWithCachedSizesToArray(\n"
      "    $start$, $end$, target);\n\n");
  } else {
    printer->Print(vars,
      "_extensions_.SerializeWithCachedSizes(\n"
      "    $start$, $end$, output);\n\n");
  }
}

void MessageGenerator::
GenerateSerializeWithCachedSizes(io::Printer* printer) {
  if (descriptor_->options().message_set_wire_format()) {
    // Special-case MessageSet.
    printer->Print(
      "void $classname$::SerializeWithCachedSizes(\n"
      "    ::google::protobuf::io::CodedOutputStream* output) const {\n"
      "  _extensions_.SerializeMessageSetWithCachedSizes(output);\n",
      "classname", classname_);
    GOOGLE_CHECK(UseUnknownFieldSet(descriptor_->file()));
    printer->Print(
      "  ::google::protobuf::internal::WireFormat::SerializeUnknownMessageSetItems(\n"
      "      unknown_fields(), output);\n");
    printer->Print(
      "}\n");
    return;
  }

  printer->Print(
    "void $classname$::SerializeWithCachedSizes(\n"
    "    ::google::protobuf::io::CodedOutputStream* output) const {\n",
    "classname", classname_);
  printer->Indent();

  printer->Print(
    "// @@protoc_insertion_point(serialize_start:$full_name$)\n",
    "full_name", descriptor_->full_name());

  GenerateSerializeWithCachedSizesBody(printer, false);

  printer->Print(
    "// @@protoc_insertion_point(serialize_end:$full_name$)\n",
    "full_name", descriptor_->full_name());

  printer->Outdent();
  printer->Print(
    "}\n");
}

void MessageGenerator::
GenerateSerializeWithCachedSizesToArray(io::Printer* printer) {
  if (descriptor_->options().message_set_wire_format()) {
    // Special-case MessageSet.
    printer->Print(
      "::google::protobuf::uint8* $classname$::SerializeWithCachedSizesToArray(\n"
      "    ::google::protobuf::uint8* target) const {\n"
      "  target =\n"
      "      _extensions_.SerializeMessageSetWithCachedSizesToArray(target);\n",
      "classname", classname_);
    GOOGLE_CHECK(UseUnknownFieldSet(descriptor_->file()));
    printer->Print(
      "  target = ::google::protobuf::internal::WireFormat::\n"
      "             SerializeUnknownMessageSetItemsToArray(\n"
      "               unknown_fields(), target);\n");
    printer->Print(
      "  return target;\n"
      "}\n");
    return;
  }

  printer->Print(
    "::google::protobuf::uint8* $classname$::SerializeWithCachedSizesToArray(\n"
    "    ::google::protobuf::uint8* target) const {\n",
    "classname", classname_);
  printer->Indent();

  printer->Print(
    "// @@protoc_insertion_point(serialize_to_array_start:$full_name$)\n",
    "full_name", descriptor_->full_name());

  GenerateSerializeWithCachedSizesBody(printer, true);

  printer->Print(
    "// @@protoc_insertion_point(serialize_to_array_end:$full_name$)\n",
    "full_name", descriptor_->full_name());

  printer->Outdent();
  printer->Print(
    "  return target;\n"
    "}\n");
}

void MessageGenerator::
GenerateSerializeWithCachedSizesBody(io::Printer* printer, bool to_array) {
  google::protobuf::scoped_array<const FieldDescriptor * > ordered_fields(
      SortFieldsByNumber(descriptor_));

  vector<const Descriptor::ExtensionRange*> sorted_extensions;
  for (int i = 0; i < descriptor_->extension_range_count(); ++i) {
    sorted_extensions.push_back(descriptor_->extension_range(i));
  }
  std::sort(sorted_extensions.begin(), sorted_extensions.end(),
            ExtensionRangeSorter());

  // Merge the fields and the extension ranges, both sorted by field number.
  int i, j;
  for (i = 0, j = 0;
       i < descriptor_->field_count() || j < sorted_extensions.size();
       ) {
    if (i == descriptor_->field_count()) {
      GenerateSerializeOneExtensionRange(printer,
                                         sorted_extensions[j++],
                                         to_array);
    } else if (j == sorted_extensions.size()) {
      GenerateSerializeOneField(printer, ordered_fields[i++], to_array);
    } else if (ordered_fields[i]->number() < sorted_extensions[j]->start) {
      GenerateSerializeOneField(printer, ordered_fields[i++], to_array);
    } else {
      GenerateSerializeOneExtensionRange(printer,
                                         sorted_extensions[j++],
                                         to_array);
    }
  }

  if (PreserveUnknownFields(descriptor_)) {
    if (UseUnknownFieldSet(descriptor_->file())) {
      printer->Print("if (_internal_metadata_.have_unknown_fields()) {\n");
      printer->Indent();
      if (to_array) {
        printer->Print(
          "target = "
              "::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(\n"
          "    unknown_fields(), target);\n");
      } else {
        printer->Print(
          "::google::protobuf::internal::WireFormat::SerializeUnknownFields(\n"
          "    unknown_fields(), output);\n");
      }
      printer->Outdent();

      printer->Print(
        "}\n");
    } else {
      printer->Print(
        "output->WriteRaw(unknown_fields().data(),\n"
        "                 static_cast<int>(unknown_fields().size()));\n");
    }
  }
}

static vector<uint32> RequiredFieldsBitMask(const Descriptor* desc) {
  vector<uint32> result;
  uint32 mask = 0;
  for (int i = 0; i < desc->field_count(); i++) {
    if (i > 0 && i % 32 == 0) {
      result.push_back(mask);
      mask = 0;
    }
    if (desc->field(i)->is_required()) {
      mask |= (1 << (i & 31));
    }
  }
  if (mask != 0) {
    result.push_back(mask);
  }
  return result;
}

// Create an expression that evaluates to
//  "for all i, (_has_bits_[i] & masks[i]) == masks[i]"
// masks is allowed to be shorter than _has_bits_, but at least one element of
// masks must be non-zero.
static string ConditionalToCheckBitmasks(const vector<uint32>& masks) {
  vector<string> parts;
  for (int i = 0; i < masks.size(); i++) {
    if (masks[i] == 0) continue;
    string m = StrCat("0x", strings::Hex(masks[i], strings::ZERO_PAD_8));
    // Each xor evaluates to 0 if the expected bits are present.
    parts.push_back(StrCat("((_has_bits_[", i, "] & ", m, ") ^ ", m, ")"));
  }
  GOOGLE_CHECK(!parts.empty());
  // If we have multiple parts, each expected to be 0, then bitwise-or them.
  string result = parts.size() == 1 ? parts[0] :
      StrCat("(", Join(parts, "\n       | "), ")");
  return result + " == 0";
}

void MessageGenerator::
GenerateByteSize(io::Printer* printer) {
  if (descriptor_->options().message_set_wire_format()) {
    // Special-case MessageSet.
    printer->Print(
      "int $classname$::ByteSize() const {\n"
      "  int total_size = _extensions_.MessageSetByteSize();\n",
      "classname", classname_);
    GOOGLE_CHECK(UseUnknownFieldSet(descriptor_->file()));
    printer->Print(
      "if (_internal_metadata_.have_unknown_fields()) {\n"
      "  total_size += ::google::protobuf::internal::WireFormat::\n"
      "      ComputeUnknownMessageSetItemsSize(unknown_fields());\n"
      "}\n");
    printer->Print(
      "  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();\n"
      "  _cached_size_ = total_size;\n"
      "  GOOGLE_SAFE_CONCURRENT_WRITES_END();\n"
      "  return total_size;\n"
      "}\n");
    return;
  }

  if (num_required_fields_ > 1 && HasFieldPresence(descriptor_->file())) {
    // Emit a function (rarely used, we hope) that handles the required fields
    // by checking for each one individually.
    printer->Print(
        "int $classname$::RequiredFieldsByteSizeFallback() const {\n",
        "classname", classname_);
    printer->Indent();
    printer->Print("int total_size = 0;\n");
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = descriptor_->field(i);
      if (field->is_required()) {
        printer->Print("\n"
                       "if (has_$name$()) {\n",
                       "name", FieldName(field));
        printer->Indent();
        PrintFieldComment(printer, field);
        field_generators_.get(field).GenerateByteSize(printer);
        printer->Outdent();
        printer->Print("}\n");
      }
    }
    printer->Print("\n"
                   "return total_size;\n");
    printer->Outdent();
    printer->Print("}\n");
  }

  printer->Print(
    "int $classname$::ByteSize() const {\n",
    "classname", classname_);
  printer->Indent();
  printer->Print(
    "int total_size = 0;\n"
    "\n");

  // Handle required fields (if any).  We expect all of them to be
  // present, so emit one conditional that checks for that.  If they are all
  // present then the fast path executes; otherwise the slow path executes.
  if (num_required_fields_ > 1 && HasFieldPresence(descriptor_->file())) {
    // The fast path works if all required fields are present.
    vector<uint32> masks_for_has_bits = RequiredFieldsBitMask(descriptor_);
    printer->Print((string("if (") +
                    ConditionalToCheckBitmasks(masks_for_has_bits) +
                    ") {  // All required fields are present.\n").c_str());
    printer->Indent();
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = descriptor_->field(i);
      if (!field->is_required()) continue;
      PrintFieldComment(printer, field);
      field_generators_.get(field).GenerateByteSize(printer);
      printer->Print("\n");
    }
    printer->Outdent();
    printer->Print("} else {\n"  // the slow path
                   "  total_size += RequiredFieldsByteSizeFallback();\n"
                   "}\n");
  } else {
    // num_required_fields_ <= 1: no need to be tricky
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* field = descriptor_->field(i);
      if (!field->is_required()) continue;
      PrintFieldComment(printer, field);
      printer->Print("if (has_$name$()) {\n",
                     "name", FieldName(field));
      printer->Indent();
      field_generators_.get(field).GenerateByteSize(printer);
      printer->Outdent();
      printer->Print("}\n");
    }
  }

  // Handle optional fields (worry below about repeateds, oneofs, etc.).
  // These are handled in chunks of 8.  The first chunk is
  // the non-requireds-non-repeateds-non-unions-non-extensions in
  //  descriptor_->field(0), descriptor_->field(1), ... descriptor_->field(7),
  // and the second chunk is the same for
  //  descriptor_->field(8), descriptor_->field(9), ... descriptor_->field(15),
  // etc.
  hash_map<int, uint32> fields_mask_for_chunk;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (!field->is_required() && !field->is_repeated() &&
        !field->containing_oneof()) {
      fields_mask_for_chunk[i / 8] |= static_cast<uint32>(1) << (i % 32);
    }
  }

  int last_index = -1;
  bool chunk_block_in_progress = false;
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (!field->is_required() && !field->is_repeated() &&
        !field->containing_oneof()) {
      // See above in GenerateClear for an explanation of this.
      // TODO(kenton):  Share code?  Unclear how to do so without
      //   over-engineering.
      if (i / 8 != last_index / 8 || last_index < 0) {
        // End previous chunk, if there was one.
        if (chunk_block_in_progress) {
          printer->Outdent();
          printer->Print("}\n");
          chunk_block_in_progress = false;
        }
        // Start chunk.
        uint32 mask = fields_mask_for_chunk[i / 8];
        int count = popcnt(mask);
        GOOGLE_DCHECK_GE(count, 1);
        if (count == 1) {
          // No "if" here because the chunk is trivial.
        } else {
          if (HasFieldPresence(descriptor_->file())) {
            printer->Print(
              "if (_has_bits_[$index$ / 32] & $mask$u) {\n",
              "index", SimpleItoa(i),
              "mask", SimpleItoa(mask));
            printer->Indent();
            chunk_block_in_progress = true;
          }
        }
      }
      last_index = i;

      PrintFieldComment(printer, field);

      bool have_enclosing_if = false;
      if (HasFieldPresence(descriptor_->file())) {
        printer->Print(
          "if (has_$name$()) {\n",
          "name", FieldName(field));
        printer->Indent();
        have_enclosing_if = true;
      } else {
        // Without field presence: field is serialized only if it has a
        // non-default value.
        have_enclosing_if = EmitFieldNonDefaultCondition(
            printer, "this->", field);
      }

      field_generators_.get(field).GenerateByteSize(printer);

      if (have_enclosing_if) {
        printer->Outdent();
        printer->Print(
          "}\n"
          "\n");
      }
    }
  }

  if (chunk_block_in_progress) {
    printer->Outdent();
    printer->Print("}\n");
  }

  // Repeated fields don't use _has_bits_ so we count them in a separate
  // pass.
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);

    if (field->is_repeated()) {
      PrintFieldComment(printer, field);
      field_generators_.get(field).GenerateByteSize(printer);
      printer->Print("\n");
    }
  }

  // Fields inside a oneof don't use _has_bits_ so we count them in a separate
  // pass.
  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    printer->Print(
        "switch ($oneofname$_case()) {\n",
        "oneofname", descriptor_->oneof_decl(i)->name());
    printer->Indent();
    for (int j = 0; j < descriptor_->oneof_decl(i)->field_count(); j++) {
      const FieldDescriptor* field = descriptor_->oneof_decl(i)->field(j);
      PrintFieldComment(printer, field);
      printer->Print(
          "case k$field_name$: {\n",
          "field_name", UnderscoresToCamelCase(field->name(), true));
      printer->Indent();
      field_generators_.get(field).GenerateByteSize(printer);
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
        ToUpper(descriptor_->oneof_decl(i)->name()));
    printer->Outdent();
    printer->Print(
        "}\n");
  }

  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "total_size += _extensions_.ByteSize();\n"
      "\n");
  }

  if (PreserveUnknownFields(descriptor_)) {
    if (UseUnknownFieldSet(descriptor_->file())) {
      printer->Print(
        "if (_internal_metadata_.have_unknown_fields()) {\n"
        "  total_size +=\n"
        "    ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(\n"
        "      unknown_fields());\n"
        "}\n");
    } else {
      printer->Print(
        "total_size += unknown_fields().size();\n"
        "\n");
    }
  }

  // We update _cached_size_ even though this is a const method.  In theory,
  // this is not thread-compatible, because concurrent writes have undefined
  // results.  In practice, since any concurrent writes will be writing the
  // exact same value, it works on all common processors.  In a future version
  // of C++, _cached_size_ should be made into an atomic<int>.
  printer->Print(
    "GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();\n"
    "_cached_size_ = total_size;\n"
    "GOOGLE_SAFE_CONCURRENT_WRITES_END();\n"
    "return total_size;\n");

  printer->Outdent();
  printer->Print("}\n");
}

void MessageGenerator::
GenerateIsInitialized(io::Printer* printer) {
  printer->Print(
    "bool $classname$::IsInitialized() const {\n",
    "classname", classname_);
  printer->Indent();

  if (HasFieldPresence(descriptor_->file())) {
    // Check that all required fields in this message are set.  We can do this
    // most efficiently by checking 32 "has bits" at a time.
    int has_bits_array_size = (descriptor_->field_count() + 31) / 32;
    for (int i = 0; i < has_bits_array_size; i++) {
      uint32 mask = 0;
      for (int bit = 0; bit < 32; bit++) {
        int index = i * 32 + bit;
        if (index >= descriptor_->field_count()) break;
        const FieldDescriptor* field = descriptor_->field(index);

        if (field->is_required()) {
          mask |= 1 << bit;
        }
      }

      if (mask != 0) {
        printer->Print(
          "if ((_has_bits_[$i$] & 0x$mask$) != 0x$mask$) return false;\n",
          "i", SimpleItoa(i),
          "mask", StrCat(strings::Hex(mask, strings::ZERO_PAD_8)));
      }
    }
  }

  // Now check that all embedded messages are initialized.
  printer->Print("\n");
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE &&
        !ShouldIgnoreRequiredFieldCheck(field) &&
        HasRequiredFields(field->message_type())) {
      if (field->is_repeated()) {
        printer->Print(
          "if (!::google::protobuf::internal::AllAreInitialized(this->$name$()))"
          " return false;\n",
          "name", FieldName(field));
      } else {
        if (field->options().weak() || !field->containing_oneof()) {
          // For weak fields, use the data member (::google::protobuf::Message*) instead
          // of the getter to avoid a link dependency on the weak message type
          // which is only forward declared.
          printer->Print(
              "if (has_$name$()) {\n"
              "  if (!this->$name$_->IsInitialized()) return false;\n"
              "}\n",
            "name", FieldName(field));
        } else {
          printer->Print(
            "if (has_$name$()) {\n"
            "  if (!this->$name$().IsInitialized()) return false;\n"
            "}\n",
            "name", FieldName(field));
        }
      }
    }
  }

  if (descriptor_->extension_range_count() > 0) {
    printer->Print(
      "\n"
      "if (!_extensions_.IsInitialized()) return false;");
  }

  printer->Outdent();
  printer->Print(
    "  return true;\n"
    "}\n");
}


}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
