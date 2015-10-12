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

#ifndef GOOGLE_PROTOBUF_UTIL_FIELD_MASK_UTIL_H__
#define GOOGLE_PROTOBUF_UTIL_FIELD_MASK_UTIL_H__

#include <string>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/field_mask.pb.h>
#include <google/protobuf/stubs/stringpiece.h>

namespace google {
namespace protobuf {
namespace util {

class LIBPROTOBUF_EXPORT FieldMaskUtil {
  typedef google::protobuf::FieldMask FieldMask;

 public:
  // Converts FieldMask to/from string, formatted according to proto3 JSON
  // spec for FieldMask (e.g., "foo,bar,baz.quz").
  static string ToString(const FieldMask& mask);
  static void FromString(StringPiece str, FieldMask* out);

  // Checks whether the given path is valid for type T.
  template <typename T>
  static bool IsValidPath(StringPiece path) {
    return InternalIsValidPath(T::descriptor(), path);
  }

  // Checks whether the given FieldMask is valid for type T.
  template <typename T>
  static bool IsValidFieldMask(const FieldMask& mask) {
    for (int i = 0; i < mask.paths_size(); ++i) {
      if (!InternalIsValidPath(T::descriptor(), mask.paths(i))) return false;
    }
    return true;
  }

  // Adds a path to FieldMask after checking whether the given path is valid.
  // This method check-fails if the path is not a valid path for type T.
  template <typename T>
  static void AddPathToFieldMask(StringPiece path, FieldMask* mask) {
    GOOGLE_CHECK(IsValidPath<T>(path));
    mask->add_paths(path);
  }

  // Creates a FieldMask with all fields of type T. This FieldMask only
  // contains fields of T but not any sub-message fields.
  template <typename T>
  static void GetFieldMaskForAllFields(FieldMask* out) {
    InternalGetFieldMaskForAllFields(T::descriptor(), out);
  }

  // Converts a FieldMask to the canonical form. It will:
  //   1. Remove paths that are covered by another path. For example,
  //      "foo.bar" is covered by "foo" and will be removed if "foo"
  //      is also in the FieldMask.
  //   2. Sort all paths in alphabetical order.
  static void ToCanonicalForm(const FieldMask& mask, FieldMask* out);

  // Creates an union of two FieldMasks.
  static void Union(const FieldMask& mask1, const FieldMask& mask2,
                    FieldMask* out);

  // Creates an intersection of two FieldMasks.
  static void Intersect(const FieldMask& mask1, const FieldMask& mask2,
                        FieldMask* out);

  // Returns true if path is covered by the given FieldMask. Note that path
  // "foo.bar" covers all paths like "foo.bar.baz", "foo.bar.quz.x", etc.
  static bool IsPathInFieldMask(StringPiece path, const FieldMask& mask);

  class MergeOptions;
  // Merges fields specified in a FieldMask into another message.
  static void MergeMessageTo(const Message& source, const FieldMask& mask,
                             const MergeOptions& options, Message* destination);

 private:
  static bool InternalIsValidPath(const Descriptor* descriptor,
                                  StringPiece path);

  static void InternalGetFieldMaskForAllFields(const Descriptor* descriptor,
                                               FieldMask* out);
};

class LIBPROTOBUF_EXPORT FieldMaskUtil::MergeOptions {
 public:
  MergeOptions()
      : replace_message_fields_(false), replace_repeated_fields_(false) {}
  // When merging message fields, the default behavior is to merge the
  // content of two message fields together. If you instead want to use
  // the field from the source message to replace the corresponding field
  // in the destination message, set this flag to true. When this flag is set,
  // specified submessage fields that are missing in source will be cleared in
  // destination.
  void set_replace_message_fields(bool value) {
    replace_message_fields_ = value;
  }
  bool replace_message_fields() const { return replace_message_fields_; }
  // The default merging behavior will append entries from the source
  // repeated field to the destination repeated field. If you only want
  // to keep the entries from the source repeated field, set this flag
  // to true.
  void set_replace_repeated_fields(bool value) {
    replace_repeated_fields_ = value;
  }
  bool replace_repeated_fields() const { return replace_repeated_fields_; }

 private:
  bool replace_message_fields_;
  bool replace_repeated_fields_;
};

}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_FIELD_MASK_UTIL_H__
