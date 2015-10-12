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

#include <google/protobuf/compiler/javanano/javanano_field.h>
#include <google/protobuf/compiler/javanano/javanano_helpers.h>
#include <google/protobuf/compiler/javanano/javanano_primitive_field.h>
#include <google/protobuf/compiler/javanano/javanano_enum_field.h>
#include <google/protobuf/compiler/javanano/javanano_map_field.h>
#include <google/protobuf/compiler/javanano/javanano_message_field.h>
#include <google/protobuf/stubs/common.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

FieldGenerator::~FieldGenerator() {}

bool FieldGenerator::SavedDefaultNeeded() const {
  // No saved default for this field by default.
  // Subclasses whose instances may need saved defaults will override this
  // and return the appropriate value.
  return false;
}

void FieldGenerator::GenerateInitSavedDefaultCode(io::Printer* printer) const {
  // No saved default for this field by default.
  // Subclasses whose instances may need saved defaults will override this
  // and generate the appropriate init code to the printer.
}

void FieldGenerator::GenerateMergingCodeFromPacked(io::Printer* printer) const {
  // Reaching here indicates a bug. Cases are:
  //   - This FieldGenerator should support packing, but this method should be
  //     overridden.
  //   - This FieldGenerator doesn't support packing, and this method should
  //     never have been called.
  GOOGLE_LOG(FATAL) << "GenerateParsingCodeFromPacked() "
             << "called on field generator that does not support packing.";
}

// =============================================

FieldGeneratorMap::FieldGeneratorMap(
    const Descriptor* descriptor, const Params &params)
  : descriptor_(descriptor),
    field_generators_(
      new scoped_ptr<FieldGenerator>[descriptor->field_count()]) {

  int next_has_bit_index = 0;
  bool saved_defaults_needed = false;
  // Construct all the FieldGenerators.
  for (int i = 0; i < descriptor->field_count(); i++) {
    FieldGenerator* field_generator = MakeGenerator(
        descriptor->field(i), params, &next_has_bit_index);
    saved_defaults_needed = saved_defaults_needed
        || field_generator->SavedDefaultNeeded();
    field_generators_[i].reset(field_generator);
  }
  total_bits_ = next_has_bit_index;
  saved_defaults_needed_ = saved_defaults_needed;
}

FieldGenerator* FieldGeneratorMap::MakeGenerator(const FieldDescriptor* field,
    const Params &params, int* next_has_bit_index) {
  JavaType java_type = GetJavaType(field);
  if (field->is_repeated()) {
    switch (java_type) {
      case JAVATYPE_MESSAGE:
        if (IsMapEntry(field->message_type())) {
          return new MapFieldGenerator(field, params);
        } else {
          return new RepeatedMessageFieldGenerator(field, params);
        }
      case JAVATYPE_ENUM:
        return new RepeatedEnumFieldGenerator(field, params);
      default:
        return new RepeatedPrimitiveFieldGenerator(field, params);
    }
  } else if (field->containing_oneof()) {
    switch (java_type) {
      case JAVATYPE_MESSAGE:
        return new MessageOneofFieldGenerator(field, params);
      case JAVATYPE_ENUM:
      default:
        return new PrimitiveOneofFieldGenerator(field, params);
    }
  } else if (params.optional_field_accessors() && field->is_optional()
      && java_type != JAVATYPE_MESSAGE) {
    // We need a has-bit for each primitive/enum field because their default
    // values could be same as explicitly set values. But we don't need it
    // for a message field because they have no defaults and Nano uses 'null'
    // for unset messages, which cannot be set explicitly.
    switch (java_type) {
      case JAVATYPE_ENUM:
        return new AccessorEnumFieldGenerator(
            field, params, (*next_has_bit_index)++);
      default:
        return new AccessorPrimitiveFieldGenerator(
            field, params, (*next_has_bit_index)++);
    }
  } else {
    switch (java_type) {
      case JAVATYPE_MESSAGE:
        return new MessageFieldGenerator(field, params);
      case JAVATYPE_ENUM:
        return new EnumFieldGenerator(field, params);
      default:
        return new PrimitiveFieldGenerator(field, params);
    }
  }
}

FieldGeneratorMap::~FieldGeneratorMap() {}

const FieldGenerator& FieldGeneratorMap::get(
    const FieldDescriptor* field) const {
  GOOGLE_CHECK_EQ(field->containing_type(), descriptor_);
  return *field_generators_[field->index()];
}

void SetCommonOneofVariables(const FieldDescriptor* descriptor,
                             map<string, string>* variables) {
  (*variables)["oneof_name"] =
      UnderscoresToCamelCase(descriptor->containing_oneof());
  (*variables)["oneof_capitalized_name"] =
      UnderscoresToCapitalizedCamelCase(descriptor->containing_oneof());
  (*variables)["oneof_index"] =
      SimpleItoa(descriptor->containing_oneof()->index());
  (*variables)["set_oneof_case"] =
      "this." + (*variables)["oneof_name"] +
      "Case_ = " + SimpleItoa(descriptor->number());
  (*variables)["clear_oneof_case"] =
      "this." + (*variables)["oneof_name"] + "Case_ = 0";
  (*variables)["has_oneof_case"] =
      "this." + (*variables)["oneof_name"] + "Case_ == " +
      SimpleItoa(descriptor->number());
}

void GenerateOneofFieldEquals(const FieldDescriptor* descriptor,
                              const map<string, string>& variables,
                              io::Printer* printer) {
  if (GetJavaType(descriptor) == JAVATYPE_BYTES) {
    printer->Print(variables,
      "if (this.has$capitalized_name$()) {\n"
      "  if (!java.util.Arrays.equals((byte[]) this.$oneof_name$_,\n"
      "                               (byte[]) other.$oneof_name$_)) {\n"
      "    return false;\n"
      "  }\n"
      "}\n");
  } else {
    printer->Print(variables,
      "if (this.has$capitalized_name$()) {\n"
      "  if (!this.$oneof_name$_.equals(other.$oneof_name$_)) {\n"
      "    return false;\n"
      "  }\n"
      "}\n");
  }
}

void GenerateOneofFieldHashCode(const FieldDescriptor* descriptor,
                                const map<string, string>& variables,
                                io::Printer* printer) {
  if (GetJavaType(descriptor) == JAVATYPE_BYTES) {
    printer->Print(variables,
      "result = 31 * result + ($has_oneof_case$\n"
      "   ? java.util.Arrays.hashCode((byte[]) this.$oneof_name$_) : 0);\n");
  } else {
    printer->Print(variables,
      "result = 31 * result +\n"
      "  ($has_oneof_case$ ? this.$oneof_name$_.hashCode() : 0);\n");
  }
}

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
