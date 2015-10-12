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

#include <google/protobuf/util/internal/default_value_objectwriter.h>

#include <google/protobuf/stubs/hash.h>

#include <google/protobuf/util/internal/constants.h>
#include <google/protobuf/stubs/map_util.h>

namespace google {
namespace protobuf {
namespace util {
using util::Status;
using util::StatusOr;
namespace converter {

DefaultValueObjectWriter::DefaultValueObjectWriter(
    TypeResolver* type_resolver, const google::protobuf::Type& type,
    ObjectWriter* ow)
    : typeinfo_(TypeInfo::NewTypeInfo(type_resolver)),
      own_typeinfo_(true),
      type_(type),
      disable_normalize_(false),
      current_(NULL),
      root_(NULL),
      ow_(ow) {}

DefaultValueObjectWriter::~DefaultValueObjectWriter() {
  for (int i = 0; i < string_values_.size(); ++i) {
    delete string_values_[i];
  }
  if (own_typeinfo_) {
    delete typeinfo_;
  }
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderBool(StringPiece name,
                                                               bool value) {
  if (current_ == NULL) {
    ow_->RenderBool(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderInt32(
    StringPiece name, int32 value) {
  if (current_ == NULL) {
    ow_->RenderInt32(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderUint32(
    StringPiece name, uint32 value) {
  if (current_ == NULL) {
    ow_->RenderUint32(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderInt64(
    StringPiece name, int64 value) {
  if (current_ == NULL) {
    ow_->RenderInt64(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderUint64(
    StringPiece name, uint64 value) {
  if (current_ == NULL) {
    ow_->RenderUint64(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderDouble(
    StringPiece name, double value) {
  if (current_ == NULL) {
    ow_->RenderDouble(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderFloat(
    StringPiece name, float value) {
  if (current_ == NULL) {
    ow_->RenderBool(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderString(
    StringPiece name, StringPiece value) {
  if (current_ == NULL) {
    ow_->RenderString(name, value);
  } else {
    // Since StringPiece is essentially a pointer, takes a copy of "value" to
    // avoid ownership issues.
    string_values_.push_back(new string(value.ToString()));
    RenderDataPiece(name, DataPiece(*string_values_.back()));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderBytes(
    StringPiece name, StringPiece value) {
  if (current_ == NULL) {
    ow_->RenderBytes(name, value);
  } else {
    RenderDataPiece(name, DataPiece(value));
  }
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::RenderNull(
    StringPiece name) {
  if (current_ == NULL) {
    ow_->RenderNull(name);
  } else {
    RenderDataPiece(name, DataPiece::NullData());
  }
  return this;
}

DefaultValueObjectWriter*
DefaultValueObjectWriter::DisableCaseNormalizationForNextKey() {
  disable_normalize_ = true;
  return this;
}

DefaultValueObjectWriter::Node::Node(const string& name,
                                     const google::protobuf::Type* type,
                                     NodeKind kind, const DataPiece& data,
                                     bool is_placeholder)
    : name_(name),
      type_(type),
      kind_(kind),
      disable_normalize_(false),
      is_any_(false),
      data_(data),
      is_placeholder_(is_placeholder) {}

DefaultValueObjectWriter::Node* DefaultValueObjectWriter::Node::FindChild(
    StringPiece name) {
  if (name.empty() || kind_ != OBJECT) {
    return NULL;
  }
  for (int i = 0; i < children_.size(); ++i) {
    Node* child = children_[i];
    if (child->name() == name) {
      return child;
    }
  }
  return NULL;
}

void DefaultValueObjectWriter::Node::WriteTo(ObjectWriter* ow) {
  if (disable_normalize_) {
    ow->DisableCaseNormalizationForNextKey();
  }

  if (kind_ == PRIMITIVE) {
    ObjectWriter::RenderDataPieceTo(data_, name_, ow);
    return;
  }

  // Render maps. Empty maps are rendered as "{}".
  if (kind_ == MAP) {
    ow->StartObject(name_);
    WriteChildren(ow);
    ow->EndObject();
    return;
  }

  // Write out lists. If we didn't have any list in response, write out empty
  // list.
  if (kind_ == LIST) {
    ow->StartList(name_);
    WriteChildren(ow);
    ow->EndList();
    return;
  }

  // If is_placeholder_ = true, we didn't see this node in the response, so
  // skip output.
  if (is_placeholder_) return;

  ow->StartObject(name_);
  WriteChildren(ow);
  ow->EndObject();
}

void DefaultValueObjectWriter::Node::WriteChildren(ObjectWriter* ow) {
  for (int i = 0; i < children_.size(); ++i) {
    Node* child = children_[i];
    child->WriteTo(ow);
  }
}

const google::protobuf::Type* DefaultValueObjectWriter::Node::GetMapValueType(
    const google::protobuf::Type& found_type, const TypeInfo* typeinfo) {
  // If this field is a map, we should use the type of its "Value" as
  // the type of the child node.
  for (int i = 0; i < found_type.fields_size(); ++i) {
    const google::protobuf::Field& sub_field = found_type.fields(i);
    if (sub_field.number() != 2) {
      continue;
    }
    if (sub_field.kind() != google::protobuf::Field_Kind_TYPE_MESSAGE) {
      // This map's value type is not a message type. We don't need to
      // get the field_type in this case.
      break;
    }
    util::StatusOr<const google::protobuf::Type*> sub_type =
        typeinfo->ResolveTypeUrl(sub_field.type_url());
    if (!sub_type.ok()) {
      GOOGLE_LOG(WARNING) << "Cannot resolve type '" << sub_field.type_url() << "'.";
    } else {
      return sub_type.ValueOrDie();
    }
    break;
  }
  return NULL;
}

void DefaultValueObjectWriter::Node::PopulateChildren(
    const TypeInfo* typeinfo) {
  // Ignores well known types that don't require automatically populating their
  // primitive children. For type "Any", we only populate its children when the
  // "@type" field is set.
  // TODO(tsun): remove "kStructValueType" from the list. It's being checked
  //     now because of a bug in the tool-chain that causes the "oneof_index"
  //     of kStructValueType to not be set correctly.
  if (type_ == NULL || type_->name() == kAnyType ||
      type_->name() == kStructType || type_->name() == kTimestampType ||
      type_->name() == kDurationType || type_->name() == kStructValueType) {
    return;
  }
  std::vector<Node*> new_children;
  hash_map<string, int> orig_children_map;

  // Creates a map of child nodes to speed up lookup.
  for (int i = 0; i < children_.size(); ++i) {
    InsertIfNotPresent(&orig_children_map, children_[i]->name_, i);
  }

  for (int i = 0; i < type_->fields_size(); ++i) {
    const google::protobuf::Field& field = type_->fields(i);
    hash_map<string, int>::iterator found =
        orig_children_map.find(field.name());
    // If the child field has already been set, we just add it to the new list
    // of children.
    if (found != orig_children_map.end()) {
      new_children.push_back(children_[found->second]);
      children_[found->second] = NULL;
      continue;
    }

    const google::protobuf::Type* field_type = NULL;
    bool is_map = false;
    NodeKind kind = PRIMITIVE;

    if (field.kind() == google::protobuf::Field_Kind_TYPE_MESSAGE) {
      kind = OBJECT;
      util::StatusOr<const google::protobuf::Type*> found_result =
          typeinfo->ResolveTypeUrl(field.type_url());
      if (!found_result.ok()) {
        // "field" is of an unknown type.
        GOOGLE_LOG(WARNING) << "Cannot resolve type '" << field.type_url() << "'.";
      } else {
        const google::protobuf::Type* found_type = found_result.ValueOrDie();
        is_map = IsMap(field, *found_type);

        if (!is_map) {
          field_type = found_type;
        } else {
          // If this field is a map, we should use the type of its "Value" as
          // the type of the child node.
          field_type = GetMapValueType(*found_type, typeinfo);
          kind = MAP;
        }
      }
    }
    if (!is_map &&
        field.cardinality() ==
            google::protobuf::Field_Cardinality_CARDINALITY_REPEATED) {
      kind = LIST;
    }

    // If oneof_index() != 0, the child field is part of a "oneof", which means
    // the child field is optional and we shouldn't populate its default value.
    if (field.oneof_index() != 0) continue;

    // If the child field is of primitive type, sets its data to the default
    // value of its type.
    google::protobuf::scoped_ptr<Node> child(
        new Node(field.json_name(), field_type, kind,
                 kind == PRIMITIVE ? CreateDefaultDataPieceForField(field)
                                   : DataPiece::NullData(),
                 true));
    new_children.push_back(child.release());
  }
  // Adds all leftover nodes in children_ to the beginning of new_child.
  for (int i = 0; i < children_.size(); ++i) {
    if (children_[i] == NULL) {
      continue;
    }
    new_children.insert(new_children.begin(), children_[i]);
    children_[i] = NULL;
  }
  children_.swap(new_children);
}

void DefaultValueObjectWriter::MaybePopulateChildrenOfAny(Node* node) {
  // If this is an "Any" node with "@type" already given and no other children
  // have been added, populates its children.
  if (node != NULL && node->is_any() && node->type() != NULL &&
      node->type()->name() != kAnyType && node->number_of_children() == 1) {
    node->PopulateChildren(typeinfo_);
  }
}

DataPiece DefaultValueObjectWriter::CreateDefaultDataPieceForField(
    const google::protobuf::Field& field) {
  switch (field.kind()) {
    case google::protobuf::Field_Kind_TYPE_DOUBLE: {
      return DataPiece(static_cast<double>(0));
    }
    case google::protobuf::Field_Kind_TYPE_FLOAT: {
      return DataPiece(static_cast<float>(0));
    }
    case google::protobuf::Field_Kind_TYPE_INT64:
    case google::protobuf::Field_Kind_TYPE_SINT64:
    case google::protobuf::Field_Kind_TYPE_SFIXED64: {
      return DataPiece(static_cast<int64>(0));
    }
    case google::protobuf::Field_Kind_TYPE_UINT64:
    case google::protobuf::Field_Kind_TYPE_FIXED64: {
      return DataPiece(static_cast<uint64>(0));
    }
    case google::protobuf::Field_Kind_TYPE_INT32:
    case google::protobuf::Field_Kind_TYPE_SINT32:
    case google::protobuf::Field_Kind_TYPE_SFIXED32: {
      return DataPiece(static_cast<int32>(0));
    }
    case google::protobuf::Field_Kind_TYPE_BOOL: {
      return DataPiece(false);
    }
    case google::protobuf::Field_Kind_TYPE_STRING: {
      return DataPiece(string());
    }
    case google::protobuf::Field_Kind_TYPE_BYTES: {
      return DataPiece("", false);
    }
    case google::protobuf::Field_Kind_TYPE_UINT32:
    case google::protobuf::Field_Kind_TYPE_FIXED32: {
      return DataPiece(static_cast<uint32>(0));
    }
    default: { return DataPiece::NullData(); }
  }
}

DefaultValueObjectWriter* DefaultValueObjectWriter::StartObject(
    StringPiece name) {
  if (current_ == NULL) {
    root_.reset(new Node(name.ToString(), &type_, OBJECT, DataPiece::NullData(),
                         false));
    root_->set_disable_normalize(GetAndResetDisableNormalize());
    root_->PopulateChildren(typeinfo_);
    current_ = root_.get();
    return this;
  }
  MaybePopulateChildrenOfAny(current_);
  Node* child = current_->FindChild(name);
  if (current_->kind() == LIST || current_->kind() == MAP || child == NULL) {
    // If current_ is a list or a map node, we should create a new child and use
    // the type of current_ as the type of the new child.
    google::protobuf::scoped_ptr<Node> node(new Node(
        name.ToString(), ((current_->kind() == LIST || current_->kind() == MAP)
                              ? current_->type()
                              : NULL),
        OBJECT, DataPiece::NullData(), false));
    child = node.get();
    current_->AddChild(node.release());
  }

  child->set_is_placeholder(false);
  child->set_disable_normalize(GetAndResetDisableNormalize());
  if (child->kind() == OBJECT && child->number_of_children() == 0) {
    child->PopulateChildren(typeinfo_);
  }

  stack_.push(current_);
  current_ = child;
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::EndObject() {
  if (stack_.empty()) {
    // The root object ends here. Writes out the tree.
    WriteRoot();
    return this;
  }
  current_ = stack_.top();
  stack_.pop();
  return this;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::StartList(
    StringPiece name) {
  if (current_ == NULL) {
    root_.reset(
        new Node(name.ToString(), &type_, LIST, DataPiece::NullData(), false));
    root_->set_disable_normalize(GetAndResetDisableNormalize());
    current_ = root_.get();
    return this;
  }
  MaybePopulateChildrenOfAny(current_);
  Node* child = current_->FindChild(name);
  if (child == NULL || child->kind() != LIST) {
    GOOGLE_LOG(WARNING) << "Cannot find field '" << name << "'.";
    google::protobuf::scoped_ptr<Node> node(
        new Node(name.ToString(), NULL, LIST, DataPiece::NullData(), false));
    child = node.get();
    current_->AddChild(node.release());
  }
  child->set_is_placeholder(false);
  child->set_disable_normalize(GetAndResetDisableNormalize());

  stack_.push(current_);
  current_ = child;
  return this;
}

void DefaultValueObjectWriter::WriteRoot() {
  root_->WriteTo(ow_);
  root_.reset(NULL);
  current_ = NULL;
}

DefaultValueObjectWriter* DefaultValueObjectWriter::EndList() {
  if (stack_.empty()) {
    WriteRoot();
    return this;
  }
  current_ = stack_.top();
  stack_.pop();
  return this;
}

void DefaultValueObjectWriter::RenderDataPiece(StringPiece name,
                                               const DataPiece& data) {
  MaybePopulateChildrenOfAny(current_);
  util::StatusOr<string> data_string = data.ToString();
  if (current_->type() != NULL && current_->type()->name() == kAnyType &&
      name == "@type" && data_string.ok()) {
    const string& string_value = data_string.ValueOrDie();
    // If the type of current_ is "Any" and its "@type" field is being set here,
    // sets the type of current_ to be the type specified by the "@type".
    util::StatusOr<const google::protobuf::Type*> found_type =
        typeinfo_->ResolveTypeUrl(string_value);
    if (!found_type.ok()) {
      GOOGLE_LOG(WARNING) << "Failed to resolve type '" << string_value << "'.";
    } else {
      current_->set_type(found_type.ValueOrDie());
    }
    current_->set_is_any(true);
    // If the "@type" field is placed after other fields, we should populate
    // other children of primitive type now. Otherwise, we should wait until the
    // first value field is rendered before we populate the children, because
    // the "value" field of a Any message could be omitted.
    if (current_->number_of_children() > 1 && current_->type() != NULL) {
      current_->PopulateChildren(typeinfo_);
    }
  }
  Node* child = current_->FindChild(name);
  if (child == NULL || child->kind() != PRIMITIVE) {
    // No children are found, creates a new child.
    google::protobuf::scoped_ptr<Node> node(
        new Node(name.ToString(), NULL, PRIMITIVE, data, false));
    child = node.get();
    current_->AddChild(node.release());
  } else {
    child->set_data(data);
  }
  child->set_disable_normalize(GetAndResetDisableNormalize());
}

}  // namespace converter
}  // namespace util
}  // namespace protobuf
}  // namespace google
