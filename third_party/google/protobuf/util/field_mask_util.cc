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

#include <google/protobuf/util/field_mask_util.h>

#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/map_util.h>

namespace google {
namespace protobuf {
namespace util {

using google::protobuf::FieldMask;

string FieldMaskUtil::ToString(const FieldMask& mask) {
  return Join(mask.paths(), ",");
}

void FieldMaskUtil::FromString(StringPiece str, FieldMask* out) {
  out->Clear();
  vector<string> paths = Split(str, ",");
  for (int i = 0; i < paths.size(); ++i) {
    if (paths[i].empty()) continue;
    out->add_paths(paths[i]);
  }
}

bool FieldMaskUtil::InternalIsValidPath(const Descriptor* descriptor,
                                        StringPiece path) {
  vector<string> parts = Split(path, ".");
  for (int i = 0; i < parts.size(); ++i) {
    const string& field_name = parts[i];
    if (descriptor == NULL) {
      return false;
    }
    const FieldDescriptor* field = descriptor->FindFieldByName(field_name);
    if (field == NULL) {
      return false;
    }
    if (!field->is_repeated() &&
        field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      descriptor = field->message_type();
    } else {
      descriptor = NULL;
    }
  }
  return true;
}

void FieldMaskUtil::InternalGetFieldMaskForAllFields(
    const Descriptor* descriptor, FieldMask* out) {
  for (int i = 0; i < descriptor->field_count(); ++i) {
    out->add_paths(descriptor->field(i)->name());
  }
}

namespace {
// A FieldMaskTree represents a FieldMask in a tree structure. For example,
// given a FieldMask "foo.bar,foo.baz,bar.baz", the FieldMaskTree will be:
//
//   [root] -+- foo -+- bar
//           |       |
//           |       +- baz
//           |
//           +- bar --- baz
//
// In the tree, each leaf node represents a field path.
class FieldMaskTree {
 public:
  FieldMaskTree();
  ~FieldMaskTree();

  void MergeFromFieldMask(const FieldMask& mask);
  void MergeToFieldMask(FieldMask* mask);

  // Add a field path into the tree. In a FieldMask, each field path matches
  // the specified field and also all its sub-fields. If the field path to
  // add is a sub-path of an existing field path in the tree (i.e., a leaf
  // node), it means the tree already matchesthe the given path so nothing will
  // be added to the tree. If the path matches an existing non-leaf node in the
  // tree, that non-leaf node will be turned into a leaf node with all its
  // children removed because the path matches all the node's children.
  void AddPath(const string& path);

  // Calculate the intersection part of a field path with this tree and add
  // the intersection field path into out.
  void IntersectPath(const string& path, FieldMaskTree* out);

  // Merge all fields specified by this tree from one message to another.
  void MergeMessage(const Message& source,
                    const FieldMaskUtil::MergeOptions& options,
                    Message* destination) {
    // Do nothing if the tree is empty.
    if (root_.children.empty()) {
      return;
    }
    MergeMessage(&root_, source, options, destination);
  }

 private:
  struct Node {
    Node() {}

    ~Node() { ClearChildren(); }

    void ClearChildren() {
      for (map<string, Node*>::iterator it = children.begin();
           it != children.end(); ++it) {
        delete it->second;
      }
      children.clear();
    }

    map<string, Node*> children;

   private:
    GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Node);
  };

  // Merge a sub-tree to mask. This method adds the field paths represented
  // by all leaf nodes descended from "node" to mask.
  void MergeToFieldMask(const string& prefix, const Node* node, FieldMask* out);

  // Merge all leaf nodes of a sub-tree to another tree.
  void MergeLeafNodesToTree(const string& prefix, const Node* node,
                            FieldMaskTree* out);

  // Merge all fields specified by a sub-tree from one message to another.
  void MergeMessage(const Node* node, const Message& source,
                    const FieldMaskUtil::MergeOptions& options,
                    Message* destination);

  Node root_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FieldMaskTree);
};

FieldMaskTree::FieldMaskTree() {}

FieldMaskTree::~FieldMaskTree() {}

void FieldMaskTree::MergeFromFieldMask(const FieldMask& mask) {
  for (int i = 0; i < mask.paths_size(); ++i) {
    AddPath(mask.paths(i));
  }
}

void FieldMaskTree::MergeToFieldMask(FieldMask* mask) {
  MergeToFieldMask("", &root_, mask);
}

void FieldMaskTree::MergeToFieldMask(const string& prefix, const Node* node,
                                     FieldMask* out) {
  if (node->children.empty()) {
    if (prefix.empty()) {
      // This is the root node.
      return;
    }
    out->add_paths(prefix);
    return;
  }
  for (map<string, Node*>::const_iterator it = node->children.begin();
       it != node->children.end(); ++it) {
    string current_path = prefix.empty() ? it->first : prefix + "." + it->first;
    MergeToFieldMask(current_path, it->second, out);
  }
}

void FieldMaskTree::AddPath(const string& path) {
  vector<string> parts = Split(path, ".");
  if (parts.empty()) {
    return;
  }
  bool new_branch = false;
  Node* node = &root_;
  for (int i = 0; i < parts.size(); ++i) {
    if (!new_branch && node != &root_ && node->children.empty()) {
      // Path matches an existing leaf node. This means the path is already
      // coverred by this tree (for example, adding "foo.bar.baz" to a tree
      // which already contains "foo.bar").
      return;
    }
    const string& node_name = parts[i];
    Node*& child = node->children[node_name];
    if (child == NULL) {
      new_branch = true;
      child = new Node();
    }
    node = child;
  }
  if (!node->children.empty()) {
    node->ClearChildren();
  }
}

void FieldMaskTree::IntersectPath(const string& path, FieldMaskTree* out) {
  vector<string> parts = Split(path, ".");
  if (parts.empty()) {
    return;
  }
  const Node* node = &root_;
  for (int i = 0; i < parts.size(); ++i) {
    if (node->children.empty()) {
      if (node != &root_) {
        out->AddPath(path);
      }
      return;
    }
    const string& node_name = parts[i];
    const Node* result = FindPtrOrNull(node->children, node_name);
    if (result == NULL) {
      // No intersection found.
      return;
    }
    node = result;
  }
  // Now we found a matching node with the given path. Add all leaf nodes
  // to out.
  MergeLeafNodesToTree(path, node, out);
}

void FieldMaskTree::MergeLeafNodesToTree(const string& prefix, const Node* node,
                                         FieldMaskTree* out) {
  if (node->children.empty()) {
    out->AddPath(prefix);
  }
  for (map<string, Node*>::const_iterator it = node->children.begin();
       it != node->children.end(); ++it) {
    string current_path = prefix.empty() ? it->first : prefix + "." + it->first;
    MergeLeafNodesToTree(current_path, it->second, out);
  }
}

void FieldMaskTree::MergeMessage(const Node* node, const Message& source,
                                 const FieldMaskUtil::MergeOptions& options,
                                 Message* destination) {
  GOOGLE_DCHECK(!node->children.empty());
  const Reflection* source_reflection = source.GetReflection();
  const Reflection* destination_reflection = destination->GetReflection();
  const Descriptor* descriptor = source.GetDescriptor();
  for (map<string, Node*>::const_iterator it = node->children.begin();
       it != node->children.end(); ++it) {
    const string& field_name = it->first;
    const Node* child = it->second;
    const FieldDescriptor* field = descriptor->FindFieldByName(field_name);
    if (field == NULL) {
      GOOGLE_LOG(ERROR) << "Cannot find field \"" << field_name << "\" in message "
                 << descriptor->full_name();
      continue;
    }
    if (!child->children.empty()) {
      // Sub-paths are only allowed for singular message fields.
      if (field->is_repeated() ||
          field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
        GOOGLE_LOG(ERROR) << "Field \"" << field_name << "\" in message "
                   << descriptor->full_name()
                   << " is not a singular message field and cannot "
                   << "have sub-fields.";
        continue;
      }
      MergeMessage(child, source_reflection->GetMessage(source, field), options,
                   destination_reflection->MutableMessage(destination, field));
      continue;
    }
    if (!field->is_repeated()) {
      switch (field->cpp_type()) {
#define COPY_VALUE(TYPE, Name)                                            \
  case FieldDescriptor::CPPTYPE_##TYPE: {                                 \
    destination_reflection->Set##Name(                                    \
        destination, field, source_reflection->Get##Name(source, field)); \
    break;                                                                \
  }
        COPY_VALUE(BOOL, Bool)
        COPY_VALUE(INT32, Int32)
        COPY_VALUE(INT64, Int64)
        COPY_VALUE(UINT32, UInt32)
        COPY_VALUE(UINT64, UInt64)
        COPY_VALUE(FLOAT, Float)
        COPY_VALUE(DOUBLE, Double)
        COPY_VALUE(ENUM, Enum)
        COPY_VALUE(STRING, String)
#undef COPY_VALUE
        case FieldDescriptor::CPPTYPE_MESSAGE: {
          if (options.replace_message_fields()) {
            destination_reflection->ClearField(destination, field);
          }
          if (source_reflection->HasField(source, field)) {
            destination_reflection->MutableMessage(destination, field)
                ->MergeFrom(source_reflection->GetMessage(source, field));
          }
          break;
        }
      }
    } else {
      if (options.replace_repeated_fields()) {
        destination_reflection->ClearField(destination, field);
      }
      switch (field->cpp_type()) {
#define COPY_REPEATED_VALUE(TYPE, Name)                            \
  case FieldDescriptor::CPPTYPE_##TYPE: {                          \
    int size = source_reflection->FieldSize(source, field);        \
    for (int i = 0; i < size; ++i) {                               \
      destination_reflection->Add##Name(                           \
          destination, field,                                      \
          source_reflection->GetRepeated##Name(source, field, i)); \
    }                                                              \
    break;                                                         \
  }
        COPY_REPEATED_VALUE(BOOL, Bool)
        COPY_REPEATED_VALUE(INT32, Int32)
        COPY_REPEATED_VALUE(INT64, Int64)
        COPY_REPEATED_VALUE(UINT32, UInt32)
        COPY_REPEATED_VALUE(UINT64, UInt64)
        COPY_REPEATED_VALUE(FLOAT, Float)
        COPY_REPEATED_VALUE(DOUBLE, Double)
        COPY_REPEATED_VALUE(ENUM, Enum)
        COPY_REPEATED_VALUE(STRING, String)
#undef COPY_REPEATED_VALUE
        case FieldDescriptor::CPPTYPE_MESSAGE: {
          int size = source_reflection->FieldSize(source, field);
          for (int i = 0; i < size; ++i) {
            destination_reflection->AddMessage(destination, field)
                ->MergeFrom(
                    source_reflection->GetRepeatedMessage(source, field, i));
          }
          break;
        }
      }
    }
  }
}

}  // namespace

void FieldMaskUtil::ToCanonicalForm(const FieldMask& mask, FieldMask* out) {
  FieldMaskTree tree;
  tree.MergeFromFieldMask(mask);
  out->Clear();
  tree.MergeToFieldMask(out);
}

void FieldMaskUtil::Union(const FieldMask& mask1, const FieldMask& mask2,
                          FieldMask* out) {
  FieldMaskTree tree;
  tree.MergeFromFieldMask(mask1);
  tree.MergeFromFieldMask(mask2);
  out->Clear();
  tree.MergeToFieldMask(out);
}

void FieldMaskUtil::Intersect(const FieldMask& mask1, const FieldMask& mask2,
                              FieldMask* out) {
  FieldMaskTree tree, intersection;
  tree.MergeFromFieldMask(mask1);
  for (int i = 0; i < mask2.paths_size(); ++i) {
    tree.IntersectPath(mask2.paths(i), &intersection);
  }
  out->Clear();
  intersection.MergeToFieldMask(out);
}

bool FieldMaskUtil::IsPathInFieldMask(StringPiece path, const FieldMask& mask) {
  for (int i = 0; i < mask.paths_size(); ++i) {
    const string& mask_path = mask.paths(i);
    if (path == mask_path) {
      return true;
    } else if (mask_path.length() < path.length()) {
      // Also check whether mask.paths(i) is a prefix of path.
      if (path.substr(0, mask_path.length() + 1).compare(mask_path + ".") ==
          0) {
        return true;
      }
    }
  }
  return false;
}

void FieldMaskUtil::MergeMessageTo(const Message& source, const FieldMask& mask,
                                   const MergeOptions& options,
                                   Message* destination) {
  GOOGLE_CHECK(source.GetDescriptor() == destination->GetDescriptor());
  // Build a FieldMaskTree and walk through the tree to merge all specified
  // fields.
  FieldMaskTree tree;
  tree.MergeFromFieldMask(mask);
  tree.MergeMessage(source, options, destination);
}

}  // namespace util
}  // namespace protobuf
}  // namespace google
