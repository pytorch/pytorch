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

#ifndef GOOGLE_PROTOBUF_UTIL_CONVERTER_PROTOSTREAM_OBJECTWRITER_H__
#define GOOGLE_PROTOBUF_UTIL_CONVERTER_PROTOSTREAM_OBJECTWRITER_H__

#include <deque>
#include <google/protobuf/stubs/hash.h>
#include <string>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/util/internal/type_info.h>
#include <google/protobuf/util/internal/datapiece.h>
#include <google/protobuf/util/internal/error_listener.h>
#include <google/protobuf/util/internal/structured_objectwriter.h>
#include <google/protobuf/util/type_resolver.h>
#include <google/protobuf/stubs/bytestream.h>

namespace google {
namespace protobuf {
namespace io {
class CodedOutputStream;
}  // namespace io
}  // namespace protobuf


namespace protobuf {
class Type;
class Field;
}  // namespace protobuf


namespace protobuf {
namespace util {
namespace converter {

class ObjectLocationTracker;

// An ObjectWriter that can write protobuf bytes directly from writer events.
//
// It also supports streaming.
class LIBPROTOBUF_EXPORT ProtoStreamObjectWriter : public StructuredObjectWriter {
 public:
// Constructor. Does not take ownership of any parameter passed in.
  ProtoStreamObjectWriter(TypeResolver* type_resolver,
                          const google::protobuf::Type& type,
                          strings::ByteSink* output, ErrorListener* listener);
  virtual ~ProtoStreamObjectWriter();

  // ObjectWriter methods.
  virtual ProtoStreamObjectWriter* StartObject(StringPiece name);
  virtual ProtoStreamObjectWriter* EndObject();
  virtual ProtoStreamObjectWriter* StartList(StringPiece name);
  virtual ProtoStreamObjectWriter* EndList();
  virtual ProtoStreamObjectWriter* RenderBool(StringPiece name, bool value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderInt32(StringPiece name, int32 value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderUint32(StringPiece name,
                                                uint32 value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderInt64(StringPiece name, int64 value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderUint64(StringPiece name,
                                                uint64 value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderDouble(StringPiece name,
                                                double value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderFloat(StringPiece name, float value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderString(StringPiece name,
                                                StringPiece value) {
    return RenderDataPiece(name, DataPiece(value));
  }
  virtual ProtoStreamObjectWriter* RenderBytes(StringPiece name,
                                               StringPiece value) {
    return RenderDataPiece(name, DataPiece(value, false));
  }
  virtual ProtoStreamObjectWriter* RenderNull(StringPiece name) {
    return RenderDataPiece(name, DataPiece::NullData());
  }

  // Renders a DataPiece 'value' into a field whose wire type is determined
  // from the given field 'name'.
  ProtoStreamObjectWriter* RenderDataPiece(StringPiece name,
                                           const DataPiece& value);

  // Returns the location tracker to use for tracking locations for errors.
  const LocationTrackerInterface& location() {
    return element_ != NULL ? *element_ : *tracker_;
  }

  // When true, we finished writing to output a complete message.
  bool done() const { return done_; }

 private:
  // Function that renders a well known type with modified behavior.
  typedef util::Status (*TypeRenderer)(ProtoStreamObjectWriter*,
                                         const DataPiece&);

  // Handles writing Anys out using nested object writers and the like.
  class LIBPROTOBUF_EXPORT AnyWriter {
   public:
    explicit AnyWriter(ProtoStreamObjectWriter* parent);
    ~AnyWriter();

    // Passes a StartObject call through to the Any writer.
    void StartObject(StringPiece name);

    // Passes an EndObject call through to the Any. Returns true if the any
    // handled the EndObject call, false if the Any is now all done and is no
    // longer needed.
    bool EndObject();

    // Passes a StartList call through to the Any writer.
    void StartList(StringPiece name);

    // Passes an EndList call through to the Any writer.
    void EndList();

    // Renders a data piece on the any.
    void RenderDataPiece(StringPiece name, const DataPiece& value);

   private:
    // Handles starting up the any once we have a type.
    void StartAny(const DataPiece& value);

    // Writes the Any out to the parent writer in its serialized form.
    void WriteAny();

    // The parent of this writer, needed for various bits such as type info and
    // the listeners.
    ProtoStreamObjectWriter* parent_;

    // The nested object writer, used to write events.
    google::protobuf::scoped_ptr<ProtoStreamObjectWriter> ow_;

    // The type_url_ that this Any represents.
    string type_url_;

    // Whether this any is invalid. This allows us to only report an invalid
    // Any message a single time rather than every time we get a nested field.
    bool invalid_;

    // The output data and wrapping ByteSink.
    string data_;
    strings::StringByteSink output_;

    // The depth within the Any, so we can track when we're done.
    int depth_;

    // True if the message type contained in Any has a special "value" message
    // injected. This is true for well-known message types like Any or Struct.
    bool has_injected_value_message_;
  };

  class LIBPROTOBUF_EXPORT ProtoElement : public BaseElement, public LocationTrackerInterface {
   public:
    // Indicates the type of element. Special types like LIST, MAP, MAP_ENTRY,
    // STRUCT etc. are used to deduce other information based on their position
    // on the stack of elements.
    enum ElementType {
      MESSAGE,       // Simple message
      LIST,          // List/repeated element
      MAP,           // Proto3 map type
      MAP_ENTRY,     // Proto3 map message type, with 'key' and 'value' fields
      ANY,           // Proto3 Any type
      STRUCT,        // Proto3 struct type
      STRUCT_VALUE,  // Struct's Value message type
      STRUCT_LIST,   // List type indicator within a struct
      STRUCT_LIST_VALUE,  // Struct Value's ListValue message type
      STRUCT_MAP,         // Struct within a struct type
      STRUCT_MAP_ENTRY    // Struct map's entry type with 'key' and 'value'
                          // fields
    };

    // Constructor for the root element. No parent nor field.
    ProtoElement(const TypeInfo* typeinfo, const google::protobuf::Type& type,
                 ProtoStreamObjectWriter* enclosing);

    // Constructor for a field of an element.
    ProtoElement(ProtoElement* parent, const google::protobuf::Field* field,
                 const google::protobuf::Type& type, ElementType element_type);

    virtual ~ProtoElement() {}

    // Called just before the destructor for clean up:
    //   - reports any missing required fields
    //   - computes the space needed by the size field, and augment the
    //     length of all parent messages by this additional space.
    //   - releases and returns the parent pointer.
    ProtoElement* pop();

    // Accessors
    const google::protobuf::Field* field() const { return field_; }
    const google::protobuf::Type& type() const { return type_; }

    // These functions return true if the element type is corresponding to the
    // type in function name.
    bool IsMap() { return element_type_ == MAP; }
    bool IsStructMap() { return element_type_ == STRUCT_MAP; }
    bool IsStructMapEntry() { return element_type_ == STRUCT_MAP_ENTRY; }
    bool IsStructList() { return element_type_ == STRUCT_LIST; }
    bool IsAny() { return element_type_ == ANY; }

    ElementType element_type() { return element_type_; }

    void RegisterField(const google::protobuf::Field* field);
    virtual string ToString() const;

    AnyWriter* any() const { return any_.get(); }

    virtual ProtoElement* parent() const {
      return static_cast<ProtoElement*>(BaseElement::parent());
    }

    // Returns true if the index is already taken by a preceeding oneof input.
    bool OneofIndexTaken(int32 index);

    // Marks the oneof 'index' as taken. Future inputs to this oneof will
    // generate an error.
    void TakeOneofIndex(int32 index);

    // Inserts map key into hash set if and only if the key did NOT already
    // exist in hash set.
    // The hash set (map_keys_) is ONLY used to keep track of map keys.
    // Return true if insert successfully; returns false if the map key was
    // already present.
    bool InsertMapKeyIfNotPresent(StringPiece map_key);

   private:
    // Used for access to variables of the enclosing instance of
    // ProtoStreamObjectWriter.
    ProtoStreamObjectWriter* ow_;

    // A writer for Any objects, handles all Any-related nonsense.
    google::protobuf::scoped_ptr<AnyWriter> any_;

    // Describes the element as a field in the parent message.
    // field_ is NULL if and only if this element is the root element.
    const google::protobuf::Field* field_;

    // TypeInfo to lookup types.
    const TypeInfo* typeinfo_;

    // Additional variables if this element is a message:
    // (Root element is always a message).
    // descriptor_     : describes allowed fields in the message.
    // required_fields_: set of required fields.
    // is_repeated_type_ : true if the element is of type list or map.
    // size_index_     : index into ProtoStreamObjectWriter::size_insert_
    //                   for later insertion of serialized message length.
    const google::protobuf::Type& type_;
    std::set<const google::protobuf::Field*> required_fields_;
    const bool is_repeated_type_;
    const int size_index_;

    // Tracks position in repeated fields, needed for LocationTrackerInterface.
    int array_index_;

    // The type of this element, see enum for permissible types.
    ElementType element_type_;

    // Set of oneof indices already seen for the type_. Used to validate
    // incoming messages so no more than one oneof is set.
    hash_set<int32> oneof_indices_;

    // Set of map keys already seen for the type_. Used to validate incoming
    // messages so no map key appears more than once.
    hash_set<StringPiece> map_keys_;

    GOOGLE_DISALLOW_IMPLICIT_CONSTRUCTORS(ProtoElement);
  };

  // Container for inserting 'size' information at the 'pos' position.
  struct SizeInfo {
    const int pos;
    int size;
  };

  ProtoStreamObjectWriter(const TypeInfo* typeinfo,
                          const google::protobuf::Type& type,
                          strings::ByteSink* output, ErrorListener* listener);

  ProtoElement* element() { return element_.get(); }

  // Helper methods for calling ErrorListener. See error_listener.h.
  void InvalidName(StringPiece unknown_name, StringPiece message);
  void InvalidValue(StringPiece type_name, StringPiece value);
  void MissingField(StringPiece missing_name);

  // Common code for BeginObject() and BeginList() that does invalid_depth_
  // bookkeeping associated with name lookup.
  const google::protobuf::Field* BeginNamed(StringPiece name, bool is_list);

  // Lookup the field in the current element. Looks in the base descriptor
  // and in any extension. This will report an error if the field cannot be
  // found or if multiple matching extensions are found.
  const google::protobuf::Field* Lookup(StringPiece name);

  // Lookup the field type in the type descriptor. Returns NULL if the type
  // is not known.
  const google::protobuf::Type* LookupType(
      const google::protobuf::Field* field);

  // Looks up the oneof struct Value field depending on the type.
  // On failure to find, it returns an appropriate error.
  util::StatusOr<const google::protobuf::Field*> LookupStructField(
      DataPiece::Type type);

  // Starts an entry in map. This will be called after placing map element at
  // the top of the stack. Uses this information to write map entries.
  const google::protobuf::Field* StartMapEntry(StringPiece name);

  // Starts a google.protobuf.Struct.
  // 'field' is of type google.protobuf.Struct.
  // If field is NULL, it indicates that the top-level message is a struct
  // type.
  void StartStruct(const google::protobuf::Field* field);

  // Starts another struct within a struct.
  // 'field' is of type google.protobuf.Value (see struct.proto).
  const google::protobuf::Field* StartStructValueInStruct(
      const google::protobuf::Field* field);

  // Starts a list within a struct.
  // 'field' is of type google.protobuf.ListValue (see struct.proto).
  const google::protobuf::Field* StartListValueInStruct(
      const google::protobuf::Field* field);

  // Starts the repeated "values" field in struct.proto's
  // google.protobuf.ListValue type. 'field' should be of type
  // google.protobuf.ListValue.
  const google::protobuf::Field* StartRepeatedValuesInListValue(
      const google::protobuf::Field* field);

  // Pops sentinel elements off the stack.
  void SkipElements();

  // Write serialized output to the final output ByteSink, inserting all
  // the size information for nested messages that are missing from the
  // intermediate Cord buffer.
  void WriteRootMessage();

  // Returns true if the field is a map.
  bool IsMap(const google::protobuf::Field& field);

  // Returns true if the field is an any.
  bool IsAny(const google::protobuf::Field& field);

  // Helper method to write proto tags based on the given field.
  void WriteTag(const google::protobuf::Field& field);


  // Helper function to render primitive data types in DataPiece.
  void RenderSimpleDataPiece(const google::protobuf::Field& field,
                             const google::protobuf::Type& type,
                             const DataPiece& data);

  // Renders google.protobuf.Value in struct.proto. It picks the right oneof
  // type based on value's type.
  static util::Status RenderStructValue(ProtoStreamObjectWriter* ow,
                                          const DataPiece& value);

  // Renders google.protobuf.Timestamp value.
  static util::Status RenderTimestamp(ProtoStreamObjectWriter* ow,
                                        const DataPiece& value);

  // Renders google.protobuf.FieldMask value.
  static util::Status RenderFieldMask(ProtoStreamObjectWriter* ow,
                                        const DataPiece& value);

  // Renders google.protobuf.Duration value.
  static util::Status RenderDuration(ProtoStreamObjectWriter* ow,
                                       const DataPiece& value);

  // Renders wrapper message types for primitive types in
  // google/protobuf/wrappers.proto.
  static util::Status RenderWrapperType(ProtoStreamObjectWriter* ow,
                                          const DataPiece& value);

  // Helper functions to create the map and find functions responsible for
  // rendering well known types, keyed by type URL.
  static hash_map<string, TypeRenderer>* renderers_;
  static void InitRendererMap();
  static void DeleteRendererMap();
  static TypeRenderer* FindTypeRenderer(const string& type_url);

  // Returns the ProtoElement::ElementType for the given Type.
  static ProtoElement::ElementType GetElementType(
      const google::protobuf::Type& type);

  // Returns true if the field for type_ can be set as a oneof. If field is not
  // a oneof type, this function does nothing and returns true.
  // If another field for this oneof is already set, this function returns
  // false. It also calls the appropriate error callback.
  // unnormalized_name is used for error string.
  bool ValidOneof(const google::protobuf::Field& field,
                  StringPiece unnormalized_name);

  // Returns true if the map key for type_ is not duplicated key.
  // If map key is duplicated key, this function returns false.
  // Note that caller should make sure that the current proto element (element_)
  // is of element type MAP or STRUCT_MAP.
  // It also calls the appropriate error callback and unnormalzied_name is used
  // for error string.
  bool ValidMapKey(StringPiece unnormalized_name);

  // Variables for describing the structure of the input tree:
  // master_type_: descriptor for the whole protobuf message.
  // typeinfo_ : the TypeInfo object to lookup types.
  const google::protobuf::Type& master_type_;
  const TypeInfo* typeinfo_;
  // Whether we own the typeinfo_ object.
  bool own_typeinfo_;

  // Indicates whether we finished writing root message completely.
  bool done_;

  // Variable for internal state processing:
  // element_    : the current element.
  // size_insert_: sizes of nested messages.
  //               pos  - position to insert the size field.
  //               size - size value to be inserted.
  google::protobuf::scoped_ptr<ProtoElement> element_;
  std::deque<SizeInfo> size_insert_;

  // Variables for output generation:
  // output_  : pointer to an external ByteSink for final user-visible output.
  // buffer_  : buffer holding partial message before being ready for output_.
  // adapter_ : internal adapter between CodedOutputStream and Cord buffer_.
  // stream_  : wrapper for writing tags and other encodings in wire format.
  strings::ByteSink* output_;
  string buffer_;
  google::protobuf::io::StringOutputStream adapter_;
  google::protobuf::scoped_ptr<google::protobuf::io::CodedOutputStream> stream_;

  // Variables for error tracking and reporting:
  // listener_     : a place to report any errors found.
  // invalid_depth_: number of enclosing invalid nested messages.
  // tracker_      : the root location tracker interface.
  ErrorListener* listener_;
  int invalid_depth_;
  google::protobuf::scoped_ptr<LocationTrackerInterface> tracker_;

  GOOGLE_DISALLOW_IMPLICIT_CONSTRUCTORS(ProtoStreamObjectWriter);
};

}  // namespace converter
}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_CONVERTER_PROTOSTREAM_OBJECTWRITER_H__
