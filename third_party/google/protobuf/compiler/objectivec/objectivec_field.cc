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

#include <google/protobuf/compiler/objectivec/objectivec_field.h>
#include <google/protobuf/compiler/objectivec/objectivec_helpers.h>
#include <google/protobuf/compiler/objectivec/objectivec_enum_field.h>
#include <google/protobuf/compiler/objectivec/objectivec_map_field.h>
#include <google/protobuf/compiler/objectivec/objectivec_message_field.h>
#include <google/protobuf/compiler/objectivec/objectivec_primitive_field.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace objectivec {

namespace {
void SetCommonFieldVariables(const FieldDescriptor* descriptor,
                             map<string, string>* variables) {
  string camel_case_name = FieldName(descriptor);
  string raw_field_name;
  if (descriptor->type() == FieldDescriptor::TYPE_GROUP) {
    raw_field_name = descriptor->message_type()->name();
  } else {
    raw_field_name = descriptor->name();
  }
  // The logic here has to match -[GGPBFieldDescriptor textFormatName].
  const string un_camel_case_name(
      UnCamelCaseFieldName(camel_case_name, descriptor));
  const bool needs_custom_name = (raw_field_name != un_camel_case_name);

  SourceLocation location;
  if (descriptor->GetSourceLocation(&location)) {
    (*variables)["comments"] = BuildCommentsString(location);
  } else {
    (*variables)["comments"] = "\n";
  }
  const string& classname = ClassName(descriptor->containing_type());
  (*variables)["classname"] = classname;
  (*variables)["name"] = camel_case_name;
  const string& capitalized_name = FieldNameCapitalized(descriptor);
  (*variables)["capitalized_name"] = capitalized_name;
  (*variables)["raw_field_name"] = raw_field_name;
  (*variables)["field_number_name"] =
      classname + "_FieldNumber_" + capitalized_name;
  (*variables)["field_number"] = SimpleItoa(descriptor->number());
  (*variables)["has_index"] = SimpleItoa(descriptor->index());
  (*variables)["field_type"] = GetCapitalizedType(descriptor);
  std::vector<string> field_flags;
  if (descriptor->is_repeated()) field_flags.push_back("GPBFieldRepeated");
  if (descriptor->is_required()) field_flags.push_back("GPBFieldRequired");
  if (descriptor->is_optional()) field_flags.push_back("GPBFieldOptional");
  if (descriptor->is_packed()) field_flags.push_back("GPBFieldPacked");

  // ObjC custom flags.
  if (descriptor->has_default_value())
    field_flags.push_back("GPBFieldHasDefaultValue");
  if (needs_custom_name) field_flags.push_back("GPBFieldTextFormatNameCustom");
  if (descriptor->type() == FieldDescriptor::TYPE_ENUM) {
    field_flags.push_back("GPBFieldHasEnumDescriptor");
  }

  (*variables)["fieldflags"] = BuildFlagsString(field_flags);

  (*variables)["default"] = DefaultValue(descriptor);
  (*variables)["default_name"] = GPBGenericValueFieldName(descriptor);

  (*variables)["dataTypeSpecific_name"] = "className";
  (*variables)["dataTypeSpecific_value"] = "NULL";

  string field_options = descriptor->options().SerializeAsString();
  // Must convert to a standard byte order for packing length into
  // a cstring.
  uint32 length = ghtonl(field_options.length());
  if (length > 0) {
    string bytes((const char*)&length, sizeof(length));
    bytes.append(field_options);
    string options_str = "\"" + CEscape(bytes) + "\"";
    (*variables)["fieldoptions"] = "\"" + CEscape(bytes) + "\"";
  } else {
    (*variables)["fieldoptions"] = "";
  }

  // Clear some common things so they can be set just when needed.
  (*variables)["storage_attribute"] = "";
}

}  // namespace

FieldGenerator* FieldGenerator::Make(const FieldDescriptor* field) {
  FieldGenerator* result = NULL;
  if (field->is_repeated()) {
    switch (GetObjectiveCType(field)) {
      case OBJECTIVECTYPE_MESSAGE: {
        if (field->is_map()) {
          result = new MapFieldGenerator(field);
        } else {
          result = new RepeatedMessageFieldGenerator(field);
        }
        break;
      }
      case OBJECTIVECTYPE_ENUM:
        result = new RepeatedEnumFieldGenerator(field);
        break;
      default:
        result = new RepeatedPrimitiveFieldGenerator(field);
        break;
    }
  } else {
    switch (GetObjectiveCType(field)) {
      case OBJECTIVECTYPE_MESSAGE: {
        result = new MessageFieldGenerator(field);
        break;
      }
      case OBJECTIVECTYPE_ENUM:
        result = new EnumFieldGenerator(field);
        break;
      default:
        if (IsReferenceType(field)) {
          result = new PrimitiveObjFieldGenerator(field);
        } else {
          result = new PrimitiveFieldGenerator(field);
        }
        break;
    }
  }
  result->FinishInitialization();
  return result;
}


FieldGenerator::FieldGenerator(const FieldDescriptor* descriptor)
    : descriptor_(descriptor) {
  SetCommonFieldVariables(descriptor, &variables_);
}

FieldGenerator::~FieldGenerator() {}

void FieldGenerator::GenerateFieldNumberConstant(io::Printer* printer) const {
  printer->Print(
      variables_,
      "$field_number_name$ = $field_number$,\n");
}

void FieldGenerator::GenerateCFunctionDeclarations(
    io::Printer* printer) const {
  // Nothing
}

void FieldGenerator::GenerateCFunctionImplementations(
    io::Printer* printer) const {
  // Nothing
}

void FieldGenerator::DetermineForwardDeclarations(
    set<string>* fwd_decls) const {
  // Nothing
}

void FieldGenerator::GenerateFieldDescription(
    io::Printer* printer) const {
  printer->Print(
      variables_,
      "{\n"
      "  .name = \"$name$\",\n"
      "  .number = $field_number_name$,\n"
      "  .hasIndex = $has_index$,\n"
      "  .flags = $fieldflags$,\n"
      "  .dataType = GPBDataType$field_type$,\n"
      "  .offset = offsetof($classname$__storage_, $name$),\n"
      "  .defaultValue.$default_name$ = $default$,\n");

  // TODO(thomasvl): It might be useful to add a CPP wrapper to support
  // compiling away the EnumDescriptors.  To do that, we'd need a #if here
  // to control setting the descriptor vs. the validator, and above in
  // SetCommonFieldVariables() we'd want to wrap how we add
  // GPBFieldHasDefaultValue to the flags.

  // "  .dataTypeSpecific.value* = [something],"
  GenerateFieldDescriptionTypeSpecific(printer);

  const string& field_options(variables_.find("fieldoptions")->second);
  if (field_options.empty()) {
    printer->Print("  .fieldOptions = NULL,\n");
  } else {
    // Can't use PrintRaw() here to get the #if/#else/#endif lines completely
    // outdented because the need for indent captured on the previous
    // printing of a \n and there is no way to get the current indent level
    // to call the right number of Outdent()/Indents() to maintain state.
    printer->Print(
        variables_,
        "#if GPBOBJC_INCLUDE_FIELD_OPTIONS\n"
        "  .fieldOptions = $fieldoptions$,\n"
        "#else\n"
        "  .fieldOptions = NULL,\n"
        "#endif  // GPBOBJC_INCLUDE_FIELD_OPTIONS\n");
  }

  printer->Print("},\n");
}

void FieldGenerator::GenerateFieldDescriptionTypeSpecific(
    io::Printer* printer) const {
  printer->Print(
      variables_,
      "  .dataTypeSpecific.$dataTypeSpecific_name$ = $dataTypeSpecific_value$,\n");
}

void FieldGenerator::SetOneofIndexBase(int index_base) {
  if (descriptor_->containing_oneof() != NULL) {
    int index = descriptor_->containing_oneof()->index() + index_base;
    // Flip the sign to mark it as a oneof.
    variables_["has_index"] = SimpleItoa(-index);
  }
}

void FieldGenerator::FinishInitialization(void) {
  // If "property_type" wasn't set, make it "storage_type".
  if ((variables_.find("property_type") == variables_.end()) &&
      (variables_.find("storage_type") != variables_.end())) {
    variables_["property_type"] = variable("storage_type");
  }
}

SingleFieldGenerator::SingleFieldGenerator(
    const FieldDescriptor* descriptor)
    : FieldGenerator(descriptor) {
  // Nothing
}

SingleFieldGenerator::~SingleFieldGenerator() {}

void SingleFieldGenerator::GenerateFieldStorageDeclaration(
    io::Printer* printer) const {
  printer->Print(variables_, "$storage_type$ $name$;\n");
}

void SingleFieldGenerator::GeneratePropertyDeclaration(
    io::Printer* printer) const {
  printer->Print(variables_, "$comments$");
  if (WantsHasProperty()) {
    printer->Print(
        variables_,
        "@property(nonatomic, readwrite) BOOL has$capitalized_name$;\n");
  }
  printer->Print(
      variables_,
      "@property(nonatomic, readwrite) $property_type$ $name$;\n"
      "\n");
}

void SingleFieldGenerator::GeneratePropertyImplementation(
    io::Printer* printer) const {
  if (WantsHasProperty()) {
    printer->Print(variables_, "@dynamic has$capitalized_name$, $name$;\n");
  } else {
    printer->Print(variables_, "@dynamic $name$;\n");
  }
}

bool SingleFieldGenerator::WantsHasProperty(void) const {
  if (descriptor_->containing_oneof() != NULL) {
    // If in a oneof, it uses the oneofcase instead of a has bit.
    return false;
  }
  if (HasFieldPresence(descriptor_->file())) {
    // In proto1/proto2, every field has a has_$name$() method.
    return true;
  }
  return false;
}

ObjCObjFieldGenerator::ObjCObjFieldGenerator(
    const FieldDescriptor* descriptor)
    : SingleFieldGenerator(descriptor) {
  variables_["property_storage_attribute"] = "strong";
  if (IsRetainedName(variables_["name"])) {
    variables_["storage_attribute"] = " NS_RETURNS_NOT_RETAINED";
  }
}

ObjCObjFieldGenerator::~ObjCObjFieldGenerator() {}

void ObjCObjFieldGenerator::GenerateFieldStorageDeclaration(
    io::Printer* printer) const {
  printer->Print(variables_, "$storage_type$ *$name$;\n");
}

void ObjCObjFieldGenerator::GeneratePropertyDeclaration(
    io::Printer* printer) const {

  // Differs from SingleFieldGenerator::GeneratePropertyDeclaration() in that
  // it uses pointers and deals with Objective C's rules around storage name
  // conventions (init*, new*, etc.)

  printer->Print(variables_, "$comments$");
  if (WantsHasProperty()) {
    printer->Print(
        variables_,
        "@property(nonatomic, readwrite) BOOL has$capitalized_name$;\n");
  }
  printer->Print(
      variables_,
      "@property(nonatomic, readwrite, $property_storage_attribute$, null_resettable) $property_type$ *$name$$storage_attribute$;\n");
  if (IsInitName(variables_.find("name")->second)) {
    // If property name starts with init we need to annotate it to get past ARC.
    // http://stackoverflow.com/questions/18723226/how-do-i-annotate-an-objective-c-property-with-an-objc-method-family/18723227#18723227
    printer->Print(variables_,
                   "- ($property_type$ *)$name$ GPB_METHOD_FAMILY_NONE;\n");
  }
  printer->Print("\n");
}

RepeatedFieldGenerator::RepeatedFieldGenerator(
    const FieldDescriptor* descriptor)
    : ObjCObjFieldGenerator(descriptor) {
  // Repeated fields don't use the has index.
  variables_["has_index"] = "GPBNoHasBit";
}

RepeatedFieldGenerator::~RepeatedFieldGenerator() {}

void RepeatedFieldGenerator::FinishInitialization(void) {
  FieldGenerator::FinishInitialization();
  variables_["array_comment"] =
      "// |" + variables_["name"] + "| contains |" + variables_["storage_type"] + "|\n";
}

void RepeatedFieldGenerator::GenerateFieldStorageDeclaration(
    io::Printer* printer) const {
  printer->Print(variables_, "$array_storage_type$ *$name$;\n");
}

void RepeatedFieldGenerator::GeneratePropertyImplementation(
    io::Printer* printer) const {
  printer->Print(variables_, "@dynamic $name$, $name$_Count;\n");
}

void RepeatedFieldGenerator::GeneratePropertyDeclaration(
    io::Printer* printer) const {

  // Repeated fields don't need the has* properties, but they do expose a
  // *Count (to check without autocreation).  So for the field property we need
  // the same logic as ObjCObjFieldGenerator::GeneratePropertyDeclaration() for
  // dealing with needing Objective C's rules around storage name conventions
  // (init*, new*, etc.)

  printer->Print(
      variables_,
      "$comments$"
      "$array_comment$"
      "@property(nonatomic, readwrite, strong, null_resettable) $array_storage_type$ *$name$$storage_attribute$;\n"
      "@property(nonatomic, readonly) NSUInteger $name$_Count;\n");
  if (IsInitName(variables_.find("name")->second)) {
    // If property name starts with init we need to annotate it to get past ARC.
    // http://stackoverflow.com/questions/18723226/how-do-i-annotate-an-objective-c-property-with-an-objc-method-family/18723227#18723227
    printer->Print(variables_,
                   "- ($array_storage_type$ *)$name$ GPB_METHOD_FAMILY_NONE;\n");
  }
  printer->Print("\n");
}

bool RepeatedFieldGenerator::WantsHasProperty(void) const {
  // Consumer check the array size/existance rather than a has bit.
  return false;
}

FieldGeneratorMap::FieldGeneratorMap(const Descriptor* descriptor)
    : descriptor_(descriptor),
      field_generators_(
          new scoped_ptr<FieldGenerator>[descriptor->field_count()]),
      extension_generators_(
          new scoped_ptr<FieldGenerator>[descriptor->extension_count()]) {
  // Construct all the FieldGenerators.
  for (int i = 0; i < descriptor->field_count(); i++) {
    field_generators_[i].reset(FieldGenerator::Make(descriptor->field(i)));
  }
  for (int i = 0; i < descriptor->extension_count(); i++) {
    extension_generators_[i].reset(FieldGenerator::Make(descriptor->extension(i)));
  }
}

FieldGeneratorMap::~FieldGeneratorMap() {}

const FieldGenerator& FieldGeneratorMap::get(
    const FieldDescriptor* field) const {
  GOOGLE_CHECK_EQ(field->containing_type(), descriptor_);
  return *field_generators_[field->index()];
}

const FieldGenerator& FieldGeneratorMap::get_extension(int index) const {
  return *extension_generators_[index];
}

void FieldGeneratorMap::SetOneofIndexBase(int index_base) {
  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_[i]->SetOneofIndexBase(index_base);
  }
}

}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
