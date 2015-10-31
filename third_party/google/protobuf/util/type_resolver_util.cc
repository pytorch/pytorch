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

#include <google/protobuf/util/type_resolver_util.h>

#include <google/protobuf/type.pb.h>
#include <google/protobuf/wrappers.pb.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/util/internal/utility.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/status.h>

namespace google {
namespace protobuf {
namespace util {
namespace {
using google::protobuf::BoolValue;
using google::protobuf::Enum;
using google::protobuf::EnumValue;
using google::protobuf::Field;
using google::protobuf::Option;
using google::protobuf::Type;

using util::Status;
using util::error::INVALID_ARGUMENT;
using util::error::NOT_FOUND;

bool SplitTypeUrl(const string& type_url, string* url_prefix,
                  string* message_name) {
  size_t pos = type_url.find_last_of("/");
  if (pos == string::npos) {
    return false;
  }
  *url_prefix = type_url.substr(0, pos);
  *message_name = type_url.substr(pos + 1);
  return true;
}

class DescriptorPoolTypeResolver : public TypeResolver {
 public:
  DescriptorPoolTypeResolver(const string& url_prefix,
                             const DescriptorPool* pool)
      : url_prefix_(url_prefix), pool_(pool) {}

  Status ResolveMessageType(const string& type_url, Type* type) {
    string url_prefix, message_name;
    if (!SplitTypeUrl(type_url, &url_prefix, &message_name) ||
        url_prefix != url_prefix_) {
      return Status(INVALID_ARGUMENT,
                    StrCat("Invalid type URL, type URLs must be of the form '",
                           url_prefix_, "/<typename>', got: ", type_url));
    }
    if (url_prefix != url_prefix_) {
      return Status(INVALID_ARGUMENT,
                    "Cannot resolve types from URL: " + url_prefix);
    }
    const Descriptor* descriptor = pool_->FindMessageTypeByName(message_name);
    if (descriptor == NULL) {
      return Status(NOT_FOUND,
                    "Invalid type URL, unknown type: " + message_name);
    }
    ConvertDescriptor(descriptor, type);
    return Status();
  }

  Status ResolveEnumType(const string& type_url, Enum* enum_type) {
    string url_prefix, type_name;
    if (!SplitTypeUrl(type_url, &url_prefix, &type_name) ||
        url_prefix != url_prefix_) {
      return Status(INVALID_ARGUMENT,
                    StrCat("Invalid type URL, type URLs must be of the form '",
                           url_prefix_, "/<typename>', got: ", type_url));
    }
    if (url_prefix != url_prefix_) {
      return Status(INVALID_ARGUMENT,
                    "Cannot resolve types from URL: " + url_prefix);
    }
    const EnumDescriptor* descriptor = pool_->FindEnumTypeByName(type_name);
    if (descriptor == NULL) {
      return Status(NOT_FOUND, "Invalid type URL, unknown type: " + type_name);
    }
    ConvertEnumDescriptor(descriptor, enum_type);
    return Status();
  }

 private:
  void ConvertDescriptor(const Descriptor* descriptor, Type* type) {
    type->Clear();
    type->set_name(descriptor->full_name());
    for (int i = 0; i < descriptor->field_count(); ++i) {
      const FieldDescriptor* field = descriptor->field(i);
      if (field->type() == FieldDescriptor::TYPE_GROUP) {
        // Group fields cannot be represented with Type. We discard them.
        continue;
      }
      ConvertFieldDescriptor(descriptor->field(i), type->add_fields());
    }
    for (int i = 0; i < descriptor->oneof_decl_count(); ++i) {
      type->add_oneofs(descriptor->oneof_decl(i)->name());
    }
    type->mutable_source_context()->set_file_name(descriptor->file()->name());
    ConvertMessageOptions(descriptor->options(), type->mutable_options());
  }

  void ConvertMessageOptions(const MessageOptions& options,
                             RepeatedPtrField<Option>* output) {
    if (options.map_entry()) {
      Option* option = output->Add();
      option->set_name("map_entry");
      BoolValue value;
      value.set_value(true);
      option->mutable_value()->PackFrom(value);
    }

    // TODO(xiaofeng): Set other "options"?
  }

  void ConvertFieldDescriptor(const FieldDescriptor* descriptor, Field* field) {
    field->set_kind(static_cast<Field::Kind>(descriptor->type()));
    switch (descriptor->label()) {
      case FieldDescriptor::LABEL_OPTIONAL:
        field->set_cardinality(Field::CARDINALITY_OPTIONAL);
        break;
      case FieldDescriptor::LABEL_REPEATED:
        field->set_cardinality(Field::CARDINALITY_REPEATED);
        break;
      case FieldDescriptor::LABEL_REQUIRED:
        field->set_cardinality(Field::CARDINALITY_REQUIRED);
        break;
    }
    field->set_number(descriptor->number());
    field->set_name(descriptor->name());
    field->set_json_name(converter::ToCamelCase(descriptor->name()));
    if (descriptor->type() == FieldDescriptor::TYPE_MESSAGE) {
      field->set_type_url(GetTypeUrl(descriptor->message_type()));
    } else if (descriptor->type() == FieldDescriptor::TYPE_ENUM) {
      field->set_type_url(GetTypeUrl(descriptor->enum_type()));
    }
    if (descriptor->containing_oneof() != NULL) {
      field->set_oneof_index(descriptor->containing_oneof()->index() + 1);
    }
    if (descriptor->is_packed()) {
      field->set_packed(true);
    }

    // TODO(xiaofeng): Set other field "options"?
  }

  void ConvertEnumDescriptor(const EnumDescriptor* descriptor,
                             Enum* enum_type) {
    enum_type->Clear();
    enum_type->set_name(descriptor->full_name());
    enum_type->mutable_source_context()->set_file_name(
        descriptor->file()->name());
    for (int i = 0; i < descriptor->value_count(); ++i) {
      const EnumValueDescriptor* value_descriptor = descriptor->value(i);
      EnumValue* value = enum_type->mutable_enumvalue()->Add();
      value->set_name(value_descriptor->name());
      value->set_number(value_descriptor->number());

      // TODO(xiaofeng): Set EnumValue options.
    }
    // TODO(xiaofeng): Set Enum "options".
  }

  string GetTypeUrl(const Descriptor* descriptor) {
    return url_prefix_ + "/" + descriptor->full_name();
  }

  string GetTypeUrl(const EnumDescriptor* descriptor) {
    return url_prefix_ + "/" + descriptor->full_name();
  }

  string url_prefix_;
  const DescriptorPool* pool_;
};

}  // namespace

TypeResolver* NewTypeResolverForDescriptorPool(const string& url_prefix,
                                               const DescriptorPool* pool) {
  return new DescriptorPoolTypeResolver(url_prefix, pool);
}

}  // namespace util
}  // namespace protobuf
}  // namespace google
