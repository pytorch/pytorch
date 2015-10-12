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

#include <google/protobuf/compiler/cpp/cpp_field.h>
#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif

#include <google/protobuf/compiler/cpp/cpp_helpers.h>
#include <google/protobuf/compiler/cpp/cpp_primitive_field.h>
#include <google/protobuf/compiler/cpp/cpp_string_field.h>
#include <google/protobuf/compiler/cpp/cpp_enum_field.h>
#include <google/protobuf/compiler/cpp/cpp_map_field.h>
#include <google/protobuf/compiler/cpp/cpp_message_field.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace cpp {

using internal::WireFormat;

void SetCommonFieldVariables(const FieldDescriptor* descriptor,
                             map<string, string>* variables,
                             const Options& options) {
  (*variables)["name"] = FieldName(descriptor);
  (*variables)["index"] = SimpleItoa(descriptor->index());
  (*variables)["number"] = SimpleItoa(descriptor->number());
  (*variables)["classname"] = ClassName(FieldScope(descriptor), false);
  (*variables)["declared_type"] = DeclaredTypeMethodName(descriptor->type());

  // non_null_ptr_to_name is usable only if has_$name$ is true.  It yields a
  // pointer that will not be NULL.  Subclasses of FieldGenerator may set
  // (*variables)["non_null_ptr_to_name"] differently.
  (*variables)["non_null_ptr_to_name"] =
      StrCat("&this->", FieldName(descriptor), "()");

  (*variables)["tag_size"] = SimpleItoa(
    WireFormat::TagSize(descriptor->number(), descriptor->type()));
  (*variables)["deprecation"] = descriptor->options().deprecated()
      ? " PROTOBUF_DEPRECATED" : "";

  (*variables)["cppget"] = "Get";

  if (HasFieldPresence(descriptor->file())) {
    (*variables)["set_hasbit"] =
        "set_has_" + FieldName(descriptor) + "();";
    (*variables)["clear_hasbit"] =
        "clear_has_" + FieldName(descriptor) + "();";
  } else {
    (*variables)["set_hasbit"] = "";
    (*variables)["clear_hasbit"] = "";
  }

  // By default, empty string, so that generic code used for both oneofs and
  // singular fields can be written.
  (*variables)["oneof_prefix"] = "";
}

void SetCommonOneofFieldVariables(const FieldDescriptor* descriptor,
                                  map<string, string>* variables) {
  const string prefix = descriptor->containing_oneof()->name() + "_.";
  (*variables)["oneof_prefix"] = prefix;
  (*variables)["oneof_name"] = descriptor->containing_oneof()->name();
  (*variables)["non_null_ptr_to_name"] =
      StrCat(prefix, (*variables)["name"], "_");
}

FieldGenerator::~FieldGenerator() {}

void FieldGenerator::
GenerateMergeFromCodedStreamWithPacking(io::Printer* printer) const {
  // Reaching here indicates a bug. Cases are:
  //   - This FieldGenerator should support packing, but this method should be
  //     overridden.
  //   - This FieldGenerator doesn't support packing, and this method should
  //     never have been called.
  GOOGLE_LOG(FATAL) << "GenerateMergeFromCodedStreamWithPacking() "
             << "called on field generator that does not support packing.";

}

FieldGeneratorMap::FieldGeneratorMap(const Descriptor* descriptor,
                                     const Options& options)
    : descriptor_(descriptor),
      field_generators_(
          new google::protobuf::scoped_ptr<FieldGenerator>[descriptor->field_count()]) {
  // Construct all the FieldGenerators.
  for (int i = 0; i < descriptor->field_count(); i++) {
    field_generators_[i].reset(MakeGenerator(descriptor->field(i), options));
  }
}

FieldGenerator* FieldGeneratorMap::MakeGenerator(const FieldDescriptor* field,
                                                 const Options& options) {
  if (field->is_repeated()) {
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_MESSAGE:
        if (field->is_map()) {
          return new MapFieldGenerator(field, options);
        } else {
          return new RepeatedMessageFieldGenerator(field, options);
        }
      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default:  // RepeatedStringFieldGenerator handles unknown ctypes.
          case FieldOptions::STRING:
            return new RepeatedStringFieldGenerator(field, options);
        }
      case FieldDescriptor::CPPTYPE_ENUM:
        return new RepeatedEnumFieldGenerator(field, options);
      default:
        return new RepeatedPrimitiveFieldGenerator(field, options);
    }
  } else if (field->containing_oneof()) {
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_MESSAGE:
        return new MessageOneofFieldGenerator(field, options);
      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default:  // StringOneofFieldGenerator handles unknown ctypes.
          case FieldOptions::STRING:
            return new StringOneofFieldGenerator(field, options);
        }
      case FieldDescriptor::CPPTYPE_ENUM:
        return new EnumOneofFieldGenerator(field, options);
      default:
        return new PrimitiveOneofFieldGenerator(field, options);
    }
  } else {
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_MESSAGE:
        return new MessageFieldGenerator(field, options);
      case FieldDescriptor::CPPTYPE_STRING:
        switch (field->options().ctype()) {
          default:  // StringFieldGenerator handles unknown ctypes.
          case FieldOptions::STRING:
            return new StringFieldGenerator(field, options);
        }
      case FieldDescriptor::CPPTYPE_ENUM:
        return new EnumFieldGenerator(field, options);
      default:
        return new PrimitiveFieldGenerator(field, options);
    }
  }
}

FieldGeneratorMap::~FieldGeneratorMap() {}

const FieldGenerator& FieldGeneratorMap::get(
    const FieldDescriptor* field) const {
  GOOGLE_CHECK_EQ(field->containing_type(), descriptor_);
  return *field_generators_[field->index()];
}


}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
