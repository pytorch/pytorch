// Protocol Buffers - Google's data interchange format
// Copyright 2010 Google Inc.  All rights reserved.
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

// Author: wink@google.com (Wink Saville)

#ifndef PROTOBUF_COMPILER_JAVANANO_JAVANANO_PARAMS_H_
#define PROTOBUF_COMPILER_JAVANANO_JAVANANO_PARAMS_H_

#include <map>
#include <set>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

enum eMultipleFiles { JAVANANO_MUL_UNSET, JAVANANO_MUL_FALSE, JAVANANO_MUL_TRUE };

// Parameters for used by the generators
class Params {
 public:
  typedef map<string, string> NameMap;
  typedef set<string> NameSet;
 private:
  string empty_;
  string base_name_;
  eMultipleFiles override_java_multiple_files_;
  bool store_unknown_fields_;
  NameMap java_packages_;
  NameMap java_outer_classnames_;
  NameSet java_multiple_files_;
  bool generate_has_;
  bool java_enum_style_;
  bool optional_field_accessors_;
  bool use_reference_types_for_primitives_;
  bool generate_equals_;
  bool ignore_services_;
  bool parcelable_messages_;
  bool reftypes_primitive_enums_;
  bool generate_clear_;
  bool generate_clone_;
  bool generate_intdefs_;

 public:
  Params(const string & base_name) :
    empty_(""),
    base_name_(base_name),
    override_java_multiple_files_(JAVANANO_MUL_UNSET),
    store_unknown_fields_(false),
    generate_has_(false),
    java_enum_style_(false),
    optional_field_accessors_(false),
    use_reference_types_for_primitives_(false),
    generate_equals_(false),
    ignore_services_(false),
    parcelable_messages_(false),
    reftypes_primitive_enums_(false),
    generate_clear_(true),
    generate_clone_(false),
    generate_intdefs_(false) {
  }

  const string& base_name() const {
    return base_name_;
  }

  bool has_java_package(const string& file_name) const {
    return java_packages_.find(file_name)
                        != java_packages_.end();
  }
  void set_java_package(const string& file_name,
      const string& java_package) {
    java_packages_[file_name] = java_package;
  }
  const string& java_package(const string& file_name) const {
    NameMap::const_iterator itr;

    itr = java_packages_.find(file_name);
    if  (itr == java_packages_.end()) {
      return empty_;
    } else {
      return itr->second;
    }
  }
  const NameMap& java_packages() {
    return java_packages_;
  }

  bool has_java_outer_classname(const string& file_name) const {
    return java_outer_classnames_.find(file_name)
                        != java_outer_classnames_.end();
  }
  void set_java_outer_classname(const string& file_name,
      const string& java_outer_classname) {
    java_outer_classnames_[file_name] = java_outer_classname;
  }
  const string& java_outer_classname(const string& file_name) const {
    NameMap::const_iterator itr;

    itr = java_outer_classnames_.find(file_name);
    if  (itr == java_outer_classnames_.end()) {
      return empty_;
    } else {
      return itr->second;
    }
  }
  const NameMap& java_outer_classnames() {
    return java_outer_classnames_;
  }

  void set_override_java_multiple_files(bool java_multiple_files) {
    if (java_multiple_files) {
      override_java_multiple_files_ = JAVANANO_MUL_TRUE;
    } else {
      override_java_multiple_files_ = JAVANANO_MUL_FALSE;
    }
  }
  void clear_override_java_multiple_files() {
    override_java_multiple_files_ = JAVANANO_MUL_UNSET;
  }

  void set_java_multiple_files(const string& file_name, bool value) {
    if (value) {
      java_multiple_files_.insert(file_name);
    } else {
      java_multiple_files_.erase(file_name);
    }
  }
  bool java_multiple_files(const string& file_name) const {
    switch (override_java_multiple_files_) {
      case JAVANANO_MUL_FALSE:
        return false;
      case JAVANANO_MUL_TRUE:
        return true;
      default:
        return java_multiple_files_.find(file_name)
                != java_multiple_files_.end();
    }
  }

  void set_store_unknown_fields(bool value) {
    store_unknown_fields_ = value;
  }
  bool store_unknown_fields() const {
    return store_unknown_fields_;
  }

  void set_generate_has(bool value) {
    generate_has_ = value;
  }
  bool generate_has() const {
    return generate_has_;
  }

  void set_java_enum_style(bool value) {
    java_enum_style_ = value;
  }
  bool java_enum_style() const {
    return java_enum_style_;
  }

  void set_optional_field_accessors(bool value) {
    optional_field_accessors_ = value;
  }
  bool optional_field_accessors() const {
    return optional_field_accessors_;
  }

  void set_use_reference_types_for_primitives(bool value) {
    use_reference_types_for_primitives_ = value;
  }
  bool use_reference_types_for_primitives() const {
    return use_reference_types_for_primitives_;
  }

  void set_generate_equals(bool value) {
    generate_equals_ = value;
  }
  bool generate_equals() const {
    return generate_equals_;
  }

  void set_ignore_services(bool value) {
    ignore_services_ = value;
  }
  bool ignore_services() const {
    return ignore_services_;
  }

  void set_parcelable_messages(bool value) {
    parcelable_messages_ = value;
  }
  bool parcelable_messages() const {
    return parcelable_messages_;
  }

  void set_reftypes_primitive_enums(bool value) {
    reftypes_primitive_enums_ = value;
  }
  bool reftypes_primitive_enums() const {
    return reftypes_primitive_enums_;
  }

  void set_generate_clear(bool value) {
    generate_clear_ = value;
  }
  bool generate_clear() const {
    return generate_clear_;
  }

  void set_generate_clone(bool value) {
    generate_clone_ = value;
  }
  bool generate_clone() const {
    return generate_clone_;
  }

  void set_generate_intdefs(bool value) {
    generate_intdefs_ = value;
  }
  bool generate_intdefs() const {
    return generate_intdefs_;
  }
};

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
#endif  // PROTOBUF_COMPILER_JAVANANO_JAVANANO_PARAMS_H_
