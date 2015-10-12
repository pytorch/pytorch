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

#ifndef GOOGLE_PROTOBUF_UTIL_CONVERTER_DEFAULT_VALUE_OBJECTWRITER_H__
#define GOOGLE_PROTOBUF_UTIL_CONVERTER_DEFAULT_VALUE_OBJECTWRITER_H__

#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <stack>
#include <vector>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/util/internal/type_info.h>
#include <google/protobuf/util/internal/datapiece.h>
#include <google/protobuf/util/internal/object_writer.h>
#include <google/protobuf/util/internal/utility.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/stubs/stringpiece.h>

namespace google {
namespace protobuf {
namespace util {
namespace converter {

// An ObjectWriter that renders non-repeated primitive fields of proto messages
// with their default values. DefaultValueObjectWriter holds objects, lists and
// fields it receives in a tree structure and writes them out to another
// ObjectWriter when EndObject() is called on the root object. It also writes
// out all non-repeated primitive fields that haven't been explicitly rendered
// with their default values (0 for numbers, "" for strings, etc).
class LIBPROTOBUF_EXPORT DefaultValueObjectWriter : public ObjectWriter {
 public:
  DefaultValueObjectWriter(TypeResolver* type_resolver,
                           const google::protobuf::Type& type,
                           ObjectWriter* ow);

  virtual ~DefaultValueObjectWriter();

  // ObjectWriter methods.
  virtual DefaultValueObjectWriter* StartObject(StringPiece name);

  virtual DefaultValueObjectWriter* EndObject();

  virtual DefaultValueObjectWriter* StartList(StringPiece name);

  virtual DefaultValueObjectWriter* EndList();

  virtual DefaultValueObjectWriter* RenderBool(StringPiece name, bool value);

  virtual DefaultValueObjectWriter* RenderInt32(StringPiece name, int32 value);

  virtual DefaultValueObjectWriter* RenderUint32(StringPiece name,
                                                 uint32 value);

  virtual DefaultValueObjectWriter* RenderInt64(StringPiece name, int64 value);

  virtual DefaultValueObjectWriter* RenderUint64(StringPiece name,
                                                 uint64 value);

  virtual DefaultValueObjectWriter* RenderDouble(StringPiece name,
                                                 double value);

  virtual DefaultValueObjectWriter* RenderFloat(StringPiece name, float value);

  virtual DefaultValueObjectWriter* RenderString(StringPiece name,
                                                 StringPiece value);
  virtual DefaultValueObjectWriter* RenderBytes(StringPiece name,
                                                StringPiece value);

  virtual DefaultValueObjectWriter* RenderNull(StringPiece name);

  virtual DefaultValueObjectWriter* DisableCaseNormalizationForNextKey();

 private:
  enum NodeKind {
    PRIMITIVE = 0,
    OBJECT = 1,
    LIST = 2,
    MAP = 3,
  };

  // "Node" represents a node in the tree that holds the input of
  // DefaultValueObjectWriter.
  class LIBPROTOBUF_EXPORT Node {
   public:
    Node(const string& name, const google::protobuf::Type* type, NodeKind kind,
         const DataPiece& data, bool is_placeholder);
    virtual ~Node() {
      for (int i = 0; i < children_.size(); ++i) {
        delete children_[i];
      }
    }

    // Adds a child to this node. Takes ownership of this child.
    void AddChild(Node* child) { children_.push_back(child); }

    // Finds the child given its name.
    Node* FindChild(StringPiece name);

    // Populates children of this Node based on its type. If there are already
    // children created, they will be merged to the result. Caller should pass
    // in TypeInfo for looking up types of the children.
    void PopulateChildren(const TypeInfo* typeinfo);

    // If this node is a leaf (has data), writes the current node to the
    // ObjectWriter; if not, then recursively writes the children to the
    // ObjectWriter.
    void WriteTo(ObjectWriter* ow);

    // Accessors
    const string& name() const { return name_; }

    const google::protobuf::Type* type() { return type_; }

    void set_type(const google::protobuf::Type* type) { type_ = type; }

    NodeKind kind() { return kind_; }

    int number_of_children() { return children_.size(); }

    void set_data(const DataPiece& data) { data_ = data; }

    void set_disable_normalize(bool disable_normalize) {
      disable_normalize_ = disable_normalize;
    }

    bool is_any() { return is_any_; }

    void set_is_any(bool is_any) { is_any_ = is_any; }

    void set_is_placeholder(bool is_placeholder) {
      is_placeholder_ = is_placeholder;
    }

   private:
    // Returns the Value Type of a map given the Type of the map entry and a
    // TypeInfo instance.
    const google::protobuf::Type* GetMapValueType(
        const google::protobuf::Type& entry_type, const TypeInfo* typeinfo);

    // Calls WriteTo() on every child in children_.
    void WriteChildren(ObjectWriter* ow);

    // The name of this node.
    string name_;
    // google::protobuf::Type of this node. Owned by TypeInfo.
    const google::protobuf::Type* type_;
    // The kind of this node.
    NodeKind kind_;
    // Whether to disable case normalization of the name.
    bool disable_normalize_;
    // Whether this is a node for "Any".
    bool is_any_;
    // The data of this node when it is a leaf node.
    DataPiece data_;
    // Children of this node.
    std::vector<Node*> children_;
    // Whether this node is a placeholder for an object or list automatically
    // generated when creating the parent node. Should be set to false after
    // the parent node's StartObject()/StartList() method is called with this
    // node's name.
    bool is_placeholder_;
  };

  // Populates children of "node" if it is an "any" Node and its real type has
  // been given.
  void MaybePopulateChildrenOfAny(Node* node);

  // Writes the root_ node to ow_ and resets the root_ and current_ pointer to
  // NULL.
  void WriteRoot();

  // Creates a DataPiece containing the default value of the type of the field.
  static DataPiece CreateDefaultDataPieceForField(
      const google::protobuf::Field& field);

  // Returns disable_normalize_ and reset it to false.
  bool GetAndResetDisableNormalize() {
    return disable_normalize_ ? (disable_normalize_ = false, true) : false;
  }

  // Adds or replaces the data_ of a primitive child node.
  void RenderDataPiece(StringPiece name, const DataPiece& data);

  // Type information for all the types used in the descriptor. Used to find
  // google::protobuf::Type of nested messages/enums.
  const TypeInfo* typeinfo_;
  // Whether the TypeInfo object is owned by this class.
  bool own_typeinfo_;
  // google::protobuf::Type of the root message type.
  const google::protobuf::Type& type_;
  // Holds copies of strings passed to RenderString.
  vector<string*> string_values_;

  // Whether to disable case normalization of the next node.
  bool disable_normalize_;
  // The current Node. Owned by its parents.
  Node* current_;
  // The root Node.
  google::protobuf::scoped_ptr<Node> root_;
  // The stack to hold the path of Nodes from current_ to root_;
  std::stack<Node*> stack_;

  ObjectWriter* ow_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(DefaultValueObjectWriter);
};

}  // namespace converter
}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_CONVERTER_DEFAULT_VALUE_OBJECTWRITER_H__
