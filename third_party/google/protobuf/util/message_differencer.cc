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

// Author: jschorr@google.com (Joseph Schorr)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.
//
// This file defines static methods and classes for comparing Protocol
// Messages (see //google/protobuf/util/message_differencer.h for more
// information).

#include <google/protobuf/util/message_differencer.h>

#include <algorithm>
#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <utility>

#include <google/protobuf/stubs/callback.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/stringprintf.h>
#include <google/protobuf/any.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/field_comparator.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {

namespace util {

// When comparing a repeated field as map, MultipleFieldMapKeyComparator can
// be used to specify multiple fields as key for key comparison.
// Two elements of a repeated field will be regarded as having the same key
// iff they have the same value for every specified key field.
// Note that you can also specify only one field as key.
class MessageDifferencer::MultipleFieldsMapKeyComparator
    : public MessageDifferencer::MapKeyComparator {
 public:
  MultipleFieldsMapKeyComparator(
      MessageDifferencer* message_differencer,
      const vector<vector<const FieldDescriptor*> >& key_field_paths)
        : message_differencer_(message_differencer),
          key_field_paths_(key_field_paths) {
    GOOGLE_CHECK(!key_field_paths_.empty());
    for (int i = 0; i < key_field_paths_.size(); ++i) {
      GOOGLE_CHECK(!key_field_paths_[i].empty());
    }
  }
  MultipleFieldsMapKeyComparator(
      MessageDifferencer* message_differencer,
      const FieldDescriptor* key)
        : message_differencer_(message_differencer) {
    vector<const FieldDescriptor*> key_field_path;
    key_field_path.push_back(key);
    key_field_paths_.push_back(key_field_path);
  }
  virtual bool IsMatch(
      const Message& message1,
      const Message& message2,
      const vector<SpecificField>& parent_fields) const {
    for (int i = 0; i < key_field_paths_.size(); ++i) {
      if (!IsMatchInternal(message1, message2, parent_fields,
                           key_field_paths_[i], 0)) {
        return false;
      }
    }
    return true;
  }
 private:
  bool IsMatchInternal(
      const Message& message1,
      const Message& message2,
      const vector<SpecificField>& parent_fields,
      const vector<const FieldDescriptor*>& key_field_path,
      int path_index) const {
    const FieldDescriptor* field = key_field_path[path_index];
    vector<SpecificField> current_parent_fields(parent_fields);
    if (path_index == key_field_path.size() - 1) {
      if (field->is_repeated()) {
        if (!message_differencer_->CompareRepeatedField(
            message1, message2, field, &current_parent_fields)) {
          return false;
        }
      } else {
        if (!message_differencer_->CompareFieldValueUsingParentFields(
            message1, message2, field, -1, -1, &current_parent_fields)) {
          return false;
        }
      }
      return true;
    } else {
      const Reflection* reflection1 = message1.GetReflection();
      const Reflection* reflection2 = message2.GetReflection();
      bool has_field1 = reflection1->HasField(message1, field);
      bool has_field2 = reflection2->HasField(message2, field);
      if (!has_field1 && !has_field2) {
        return true;
      }
      if (has_field1 != has_field2) {
        return false;
      }
      SpecificField specific_field;
      specific_field.field = field;
      current_parent_fields.push_back(specific_field);
      return IsMatchInternal(
          reflection1->GetMessage(message1, field),
          reflection2->GetMessage(message2, field),
          current_parent_fields,
          key_field_path,
          path_index + 1);
    }
  }
  MessageDifferencer* message_differencer_;
  vector<vector<const FieldDescriptor*> > key_field_paths_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(MultipleFieldsMapKeyComparator);
};

bool MessageDifferencer::Equals(const Message& message1,
                                const Message& message2) {
  MessageDifferencer differencer;

  return differencer.Compare(message1, message2);
}

bool MessageDifferencer::Equivalent(const Message& message1,
                                    const Message& message2) {
  MessageDifferencer differencer;
  differencer.set_message_field_comparison(MessageDifferencer::EQUIVALENT);

  return differencer.Compare(message1, message2);
}

bool MessageDifferencer::ApproximatelyEquals(const Message& message1,
                                             const Message& message2) {
  MessageDifferencer differencer;
  differencer.set_float_comparison(
      MessageDifferencer::APPROXIMATE);

  return differencer.Compare(message1, message2);
}

bool MessageDifferencer::ApproximatelyEquivalent(const Message& message1,
                                                 const Message& message2) {
  MessageDifferencer differencer;
  differencer.set_message_field_comparison(MessageDifferencer::EQUIVALENT);
  differencer.set_float_comparison(MessageDifferencer::APPROXIMATE);

  return differencer.Compare(message1, message2);
}

// ===========================================================================

MessageDifferencer::MessageDifferencer()
    : reporter_(NULL),
      field_comparator_(NULL),
      message_field_comparison_(EQUAL),
      scope_(FULL),
      repeated_field_comparison_(AS_LIST),
      report_matches_(false),
      output_string_(NULL) { }

MessageDifferencer::~MessageDifferencer() {
  for (int i = 0; i < owned_key_comparators_.size(); ++i) {
    delete owned_key_comparators_[i];
  }
  for (int i = 0; i < ignore_criteria_.size(); ++i) {
    delete ignore_criteria_[i];
  }
}

void MessageDifferencer::set_field_comparator(FieldComparator* comparator) {
  GOOGLE_CHECK(comparator) << "Field comparator can't be NULL.";
  field_comparator_ = comparator;
}

void MessageDifferencer::set_message_field_comparison(
    MessageFieldComparison comparison) {
  message_field_comparison_ = comparison;
}

void MessageDifferencer::set_scope(Scope scope) {
  scope_ = scope;
}

MessageDifferencer::Scope MessageDifferencer::scope() {
  return scope_;
}

void MessageDifferencer::set_float_comparison(FloatComparison comparison) {
  default_field_comparator_.set_float_comparison(
      comparison == EXACT ?
      DefaultFieldComparator::EXACT : DefaultFieldComparator::APPROXIMATE);
}

void MessageDifferencer::set_repeated_field_comparison(
    RepeatedFieldComparison comparison) {
  repeated_field_comparison_ = comparison;
}

void MessageDifferencer::TreatAsSet(const FieldDescriptor* field) {
  GOOGLE_CHECK(field->is_repeated()) << "Field must be repeated: "
                               << field->full_name();
  const MapKeyComparator* key_comparator = GetMapKeyComparator(field);
  GOOGLE_CHECK(key_comparator == NULL)
      << "Cannot treat this repeated field as both Map and Set for"
      << " comparison.  Field name is: " << field->full_name();
  set_fields_.insert(field);
}

void MessageDifferencer::TreatAsMap(const FieldDescriptor* field,
                                    const FieldDescriptor* key) {
  GOOGLE_CHECK(field->is_repeated()) << "Field must be repeated: "
                               << field->full_name();
  GOOGLE_CHECK_EQ(FieldDescriptor::CPPTYPE_MESSAGE, field->cpp_type())
      << "Field has to be message type.  Field name is: "
      << field->full_name();
  GOOGLE_CHECK(key->containing_type() == field->message_type())
      << key->full_name()
      << " must be a direct subfield within the repeated field "
      << field->full_name() << ", not " << key->containing_type()->full_name();
  GOOGLE_CHECK(set_fields_.find(field) == set_fields_.end())
      << "Cannot treat this repeated field as both Map and Set for "
      << "comparison.";
  MapKeyComparator* key_comparator =
      new MultipleFieldsMapKeyComparator(this, key);
  owned_key_comparators_.push_back(key_comparator);
  map_field_key_comparator_[field] = key_comparator;
}

void MessageDifferencer::TreatAsMapWithMultipleFieldsAsKey(
    const FieldDescriptor* field,
    const vector<const FieldDescriptor*>& key_fields) {
  vector<vector<const FieldDescriptor*> > key_field_paths;
  for (int i = 0; i < key_fields.size(); ++i) {
    vector<const FieldDescriptor*> key_field_path;
    key_field_path.push_back(key_fields[i]);
    key_field_paths.push_back(key_field_path);
  }
  TreatAsMapWithMultipleFieldPathsAsKey(field, key_field_paths);
}

void MessageDifferencer::TreatAsMapWithMultipleFieldPathsAsKey(
    const FieldDescriptor* field,
    const vector<vector<const FieldDescriptor*> >& key_field_paths) {
  GOOGLE_CHECK(field->is_repeated()) << "Field must be repeated: "
                              << field->full_name();
  GOOGLE_CHECK_EQ(FieldDescriptor::CPPTYPE_MESSAGE, field->cpp_type())
      << "Field has to be message type.  Field name is: "
      << field->full_name();
  for (int i = 0; i < key_field_paths.size(); ++i) {
    const vector<const FieldDescriptor*>& key_field_path = key_field_paths[i];
    for (int j = 0; j < key_field_path.size(); ++j) {
      const FieldDescriptor* parent_field =
          j == 0 ? field : key_field_path[j - 1];
      const FieldDescriptor* child_field = key_field_path[j];
      GOOGLE_CHECK(child_field->containing_type() == parent_field->message_type())
          << child_field->full_name()
          << " must be a direct subfield within the field: "
          << parent_field->full_name();
      if (j != 0) {
        GOOGLE_CHECK_EQ(FieldDescriptor::CPPTYPE_MESSAGE, parent_field->cpp_type())
            << parent_field->full_name() << " has to be of type message.";
        GOOGLE_CHECK(!parent_field->is_repeated())
            << parent_field->full_name() << " cannot be a repeated field.";
      }
    }
  }
  GOOGLE_CHECK(set_fields_.find(field) == set_fields_.end())
      << "Cannot treat this repeated field as both Map and Set for "
      << "comparison.";
  MapKeyComparator* key_comparator =
      new MultipleFieldsMapKeyComparator(this, key_field_paths);
  owned_key_comparators_.push_back(key_comparator);
  map_field_key_comparator_[field] = key_comparator;
}

void MessageDifferencer::TreatAsMapUsingKeyComparator(
    const FieldDescriptor* field,
    const MapKeyComparator* key_comparator) {
  GOOGLE_CHECK(field->is_repeated()) << "Field must be repeated: "
                               << field->full_name();
  GOOGLE_CHECK_EQ(FieldDescriptor::CPPTYPE_MESSAGE, field->cpp_type())
      << "Field has to be message type.  Field name is: "
      << field->full_name();
  GOOGLE_CHECK(set_fields_.find(field) == set_fields_.end())
      << "Cannot treat this repeated field as both Map and Set for "
      << "comparison.";
  map_field_key_comparator_[field] = key_comparator;
}

void MessageDifferencer::AddIgnoreCriteria(IgnoreCriteria* ignore_criteria) {
  ignore_criteria_.push_back(ignore_criteria);
}

void MessageDifferencer::IgnoreField(const FieldDescriptor* field) {
  ignored_fields_.insert(field);
}

void MessageDifferencer::SetFractionAndMargin(const FieldDescriptor* field,
                                              double fraction, double margin) {
  default_field_comparator_.SetFractionAndMargin(field, fraction, margin);
}

void MessageDifferencer::ReportDifferencesToString(string* output) {
  GOOGLE_DCHECK(output) << "Specified output string was NULL";

  output_string_ = output;
  output_string_->clear();
}

void MessageDifferencer::ReportDifferencesTo(Reporter* reporter) {
  // If an output string is set, clear it to prevent
  // it superceding the specified reporter.
  if (output_string_) {
    output_string_ = NULL;
  }

  reporter_ = reporter;
}

bool MessageDifferencer::FieldBefore(const FieldDescriptor* field1,
                                     const FieldDescriptor* field2) {
  // Handle sentinel values (i.e. make sure NULLs are always ordered
  // at the end of the list).
  if (field1 == NULL) {
    return false;
  }

  if (field2 == NULL) {
    return true;
  }

  // Always order fields by their tag number
  return (field1->number() < field2->number());
}

bool MessageDifferencer::Compare(const Message& message1,
                                 const Message& message2) {
  vector<SpecificField> parent_fields;

  bool result = false;

  // Setup the internal reporter if need be.
  if (output_string_) {
    io::StringOutputStream output_stream(output_string_);
    StreamReporter reporter(&output_stream);
    reporter_ = &reporter;
    result = Compare(message1, message2, &parent_fields);
    reporter_ = NULL;
  } else {
    result = Compare(message1, message2, &parent_fields);
  }

  return result;
}

bool MessageDifferencer::CompareWithFields(
    const Message& message1,
    const Message& message2,
    const vector<const FieldDescriptor*>& message1_fields_arg,
    const vector<const FieldDescriptor*>& message2_fields_arg) {
  if (message1.GetDescriptor() != message2.GetDescriptor()) {
    GOOGLE_LOG(DFATAL) << "Comparison between two messages with different "
                << "descriptors.";
    return false;
  }

  vector<SpecificField> parent_fields;

  bool result = false;

  vector<const FieldDescriptor*> message1_fields(message1_fields_arg);
  vector<const FieldDescriptor*> message2_fields(message2_fields_arg);

  std::sort(message1_fields.begin(), message1_fields.end(), FieldBefore);
  std::sort(message2_fields.begin(), message2_fields.end(), FieldBefore);
  // Append NULL sentinel values.
  message1_fields.push_back(NULL);
  message2_fields.push_back(NULL);

  // Setup the internal reporter if need be.
  if (output_string_) {
    io::StringOutputStream output_stream(output_string_);
    StreamReporter reporter(&output_stream);
    reporter_ = &reporter;
    result = CompareRequestedFieldsUsingSettings(
        message1, message2, message1_fields, message2_fields, &parent_fields);
    reporter_ = NULL;
  } else {
    result = CompareRequestedFieldsUsingSettings(
        message1, message2, message1_fields, message2_fields, &parent_fields);
  }

  return result;
}

bool MessageDifferencer::Compare(
    const Message& message1,
    const Message& message2,
    vector<SpecificField>* parent_fields) {
  const Descriptor* descriptor1 = message1.GetDescriptor();
  const Descriptor* descriptor2 = message2.GetDescriptor();
  if (descriptor1 != descriptor2) {
    GOOGLE_LOG(DFATAL) << "Comparison between two messages with different "
                << "descriptors.";
    return false;
  }
  // Expand google.protobuf.Any payload if possible.
  if (descriptor1->full_name() == internal::kAnyFullTypeName) {
    google::protobuf::scoped_ptr<Message> data1;
    google::protobuf::scoped_ptr<Message> data2;
    if (UnpackAny(message1, &data1) && UnpackAny(message2, &data2)) {
      return Compare(*data1, *data2, parent_fields);
    }
  }
  const Reflection* reflection1 = message1.GetReflection();
  const Reflection* reflection2 = message2.GetReflection();

  // Retrieve all the set fields, including extensions.
  vector<const FieldDescriptor*> message1_fields;
  vector<const FieldDescriptor*> message2_fields;

  reflection1->ListFields(message1, &message1_fields);
  reflection2->ListFields(message2, &message2_fields);

  // Add sentinel values to deal with the
  // case where the number of the fields in
  // each list are different.
  message1_fields.push_back(NULL);
  message2_fields.push_back(NULL);

  bool unknown_compare_result = true;
  // Ignore unknown fields in EQUIVALENT mode
  if (message_field_comparison_ != EQUIVALENT) {
    const google::protobuf::UnknownFieldSet* unknown_field_set1 =
        &reflection1->GetUnknownFields(message1);
    const google::protobuf::UnknownFieldSet* unknown_field_set2 =
        &reflection2->GetUnknownFields(message2);
    if (!CompareUnknownFields(message1, message2,
                              *unknown_field_set1, *unknown_field_set2,
                              parent_fields)) {
      if (reporter_ == NULL) {
        return false;
      };
      unknown_compare_result = false;
    }
  }

  return CompareRequestedFieldsUsingSettings(
      message1, message2,
      message1_fields, message2_fields,
      parent_fields) && unknown_compare_result;
}

bool MessageDifferencer::CompareRequestedFieldsUsingSettings(
    const Message& message1,
    const Message& message2,
    const vector<const FieldDescriptor*>& message1_fields,
    const vector<const FieldDescriptor*>& message2_fields,
    vector<SpecificField>* parent_fields) {
  if (scope_ == FULL) {
    if (message_field_comparison_ == EQUIVALENT) {
      // We need to merge the field lists of both messages (i.e.
      // we are merely checking for a difference in field values,
      // rather than the addition or deletion of fields).
      vector<const FieldDescriptor*> fields_union;
      CombineFields(message1_fields, FULL, message2_fields, FULL,
                    &fields_union);
      return CompareWithFieldsInternal(message1, message2, fields_union,
                                       fields_union, parent_fields);
    } else {
      // Simple equality comparison, use the unaltered field lists.
      return CompareWithFieldsInternal(message1, message2, message1_fields,
                                       message2_fields, parent_fields);
    }
  } else {
    if (message_field_comparison_ == EQUIVALENT) {
      // We use the list of fields for message1 for both messages when
      // comparing.  This way, extra fields in message2 are ignored,
      // and missing fields in message2 use their default value.
      return CompareWithFieldsInternal(message1, message2, message1_fields,
                                       message1_fields, parent_fields);
    } else {
      // We need to consider the full list of fields for message1
      // but only the intersection for message2.  This way, any fields
      // only present in message2 will be ignored, but any fields only
      // present in message1 will be marked as a difference.
      vector<const FieldDescriptor*> fields_intersection;
      CombineFields(message1_fields, PARTIAL, message2_fields, PARTIAL,
                    &fields_intersection);
      return CompareWithFieldsInternal(message1, message2, message1_fields,
                                       fields_intersection, parent_fields);
    }
  }
}

void MessageDifferencer::CombineFields(
    const vector<const FieldDescriptor*>& fields1,
    Scope fields1_scope,
    const vector<const FieldDescriptor*>& fields2,
    Scope fields2_scope,
    vector<const FieldDescriptor*>* combined_fields) {

  int index1 = 0;
  int index2 = 0;

  while (index1 < fields1.size() && index2 < fields2.size()) {
    const FieldDescriptor* field1 = fields1[index1];
    const FieldDescriptor* field2 = fields2[index2];

    if (FieldBefore(field1, field2)) {
      if (fields1_scope == FULL) {
        combined_fields->push_back(fields1[index1]);
      }
      ++index1;
    } else if (FieldBefore(field2, field1)) {
      if (fields2_scope == FULL) {
        combined_fields->push_back(fields2[index2]);
      }
      ++index2;
    } else {
      combined_fields->push_back(fields1[index1]);
      ++index1;
      ++index2;
    }
  }
}

bool MessageDifferencer::CompareWithFieldsInternal(
    const Message& message1,
    const Message& message2,
    const vector<const FieldDescriptor*>& message1_fields,
    const vector<const FieldDescriptor*>& message2_fields,
    vector<SpecificField>* parent_fields) {
  bool isDifferent = false;
  int field_index1 = 0;
  int field_index2 = 0;

  const Reflection* reflection1 = message1.GetReflection();
  const Reflection* reflection2 = message2.GetReflection();

  while (true) {
    const FieldDescriptor* field1 = message1_fields[field_index1];
    const FieldDescriptor* field2 = message2_fields[field_index2];

    // Once we have reached sentinel values, we are done the comparison.
    if (field1 == NULL && field2 == NULL) {
      break;
    }

    // Check for differences in the field itself.
    if (FieldBefore(field1, field2)) {
      // Field 1 is not in the field list for message 2.
      if (IsIgnored(message1, message2, field1, *parent_fields)) {
        // We are ignoring field1. Report the ignore and move on to
        // the next field in message1_fields.
        if (reporter_ != NULL) {
          SpecificField specific_field;
          specific_field.field = field1;

          parent_fields->push_back(specific_field);
          reporter_->ReportIgnored(message1, message2, *parent_fields);
          parent_fields->pop_back();
        }
        ++field_index1;
        continue;
      }

      if (reporter_ != NULL) {
        int count = field1->is_repeated() ?
            reflection1->FieldSize(message1, field1) : 1;

        for (int i = 0; i < count; ++i) {
          SpecificField specific_field;
          specific_field.field = field1;
          specific_field.index = field1->is_repeated() ? i : -1;

          parent_fields->push_back(specific_field);
          reporter_->ReportDeleted(message1, message2, *parent_fields);
          parent_fields->pop_back();
        }

        isDifferent = true;
      } else {
        return false;
      }

      ++field_index1;
      continue;
    } else if (FieldBefore(field2, field1)) {
      // Field 2 is not in the field list for message 1.
      if (IsIgnored(message1, message2, field2, *parent_fields)) {
        // We are ignoring field2. Report the ignore and move on to
        // the next field in message2_fields.
        if (reporter_ != NULL) {
          SpecificField specific_field;
          specific_field.field = field2;

          parent_fields->push_back(specific_field);
          reporter_->ReportIgnored(message1, message2, *parent_fields);
          parent_fields->pop_back();
        }
        ++field_index2;
        continue;
      }

      if (reporter_ != NULL) {
        int count = field2->is_repeated() ?
            reflection2->FieldSize(message2, field2) : 1;

        for (int i = 0; i < count; ++i) {
          SpecificField specific_field;
          specific_field.field = field2;
          specific_field.index = field2->is_repeated() ? i : -1;
          specific_field.new_index = specific_field.index;

          parent_fields->push_back(specific_field);
          reporter_->ReportAdded(message1, message2, *parent_fields);
          parent_fields->pop_back();
        }

        isDifferent = true;
      } else {
        return false;
      }

      ++field_index2;
      continue;
    }

    // By this point, field1 and field2 are guarenteed to point to the same
    // field, so we can now compare the values.
    if (IsIgnored(message1, message2, field1, *parent_fields)) {
      // Ignore this field. Report and move on.
      if (reporter_ != NULL) {
        SpecificField specific_field;
        specific_field.field = field1;

        parent_fields->push_back(specific_field);
        reporter_->ReportIgnored(message1, message2, *parent_fields);
        parent_fields->pop_back();
      }

      ++field_index1;
      ++field_index2;
      continue;
    }

    bool fieldDifferent = false;
    if (field1->is_repeated()) {
      fieldDifferent = !CompareRepeatedField(message1, message2, field1,
                                             parent_fields);
      if (fieldDifferent) {
        if (reporter_ == NULL) return false;
        isDifferent = true;
      }
    } else {
      fieldDifferent = !CompareFieldValueUsingParentFields(
          message1, message2, field1, -1, -1, parent_fields);

      // If we have found differences, either report them or terminate if
      // no reporter is present.
      if (fieldDifferent && reporter_ == NULL) {
        return false;
      }

      if (reporter_ != NULL) {
        SpecificField specific_field;
        specific_field.field = field1;
        parent_fields->push_back(specific_field);
        if (fieldDifferent) {
          reporter_->ReportModified(message1, message2, *parent_fields);
          isDifferent = true;
        } else if (report_matches_) {
          reporter_->ReportMatched(message1, message2, *parent_fields);
        }
        parent_fields->pop_back();
      }
    }
    // Increment the field indicies.
    ++field_index1;
    ++field_index2;
  }

  return !isDifferent;
}

bool MessageDifferencer::IsMatch(const FieldDescriptor* repeated_field,
                                 const MapKeyComparator* key_comparator,
                                 const Message* message1,
                                 const Message* message2,
                                 const vector<SpecificField>& parent_fields,
                                 int index1, int index2) {
  vector<SpecificField> current_parent_fields(parent_fields);
  if (repeated_field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
    return CompareFieldValueUsingParentFields(
        *message1, *message2, repeated_field, index1, index2,
        &current_parent_fields);
  }
  // Back up the Reporter and output_string_.  They will be reset in the
  // following code.
  Reporter* backup_reporter = reporter_;
  string* output_string = output_string_;
  reporter_ = NULL;
  output_string_ = NULL;
  bool match;

  if (key_comparator == NULL) {
    match = CompareFieldValueUsingParentFields(
        *message1, *message2, repeated_field, index1, index2,
        &current_parent_fields);
  } else {
    const Reflection* reflection1 = message1->GetReflection();
    const Reflection* reflection2 = message2->GetReflection();
    const Message& m1 =
        reflection1->GetRepeatedMessage(*message1, repeated_field, index1);
    const Message& m2 =
        reflection2->GetRepeatedMessage(*message2, repeated_field, index2);
    SpecificField specific_field;
    specific_field.field = repeated_field;
    current_parent_fields.push_back(specific_field);
    match = key_comparator->IsMatch(m1, m2, current_parent_fields);
  }

  reporter_ = backup_reporter;
  output_string_ = output_string;
  return match;
}

bool MessageDifferencer::CompareRepeatedField(
    const Message& message1,
    const Message& message2,
    const FieldDescriptor* repeated_field,
    vector<SpecificField>* parent_fields) {
  // the input FieldDescriptor is guaranteed to be repeated field.
  const Reflection* reflection1 = message1.GetReflection();
  const Reflection* reflection2 = message2.GetReflection();
  const int count1 = reflection1->FieldSize(message1, repeated_field);
  const int count2 = reflection2->FieldSize(message2, repeated_field);
  const bool treated_as_subset = IsTreatedAsSubset(repeated_field);

  // If the field is not treated as subset and no detailed reports is needed,
  // we do a quick check on the number of the elements to avoid unnecessary
  // comparison.
  if (count1 != count2 && reporter_ == NULL && !treated_as_subset) {
    return false;
  }
  // A match can never be found if message1 has more items than message2.
  if (count1 > count2 && reporter_ == NULL) {
    return false;
  }

  // These two list are used for store the index of the correspondent
  // element in peer repeated field.
  vector<int> match_list1;
  vector<int> match_list2;

  // Try to match indices of the repeated fields. Return false if match fails
  // and there's no detailed report needed.
  if (!MatchRepeatedFieldIndices(message1, message2, repeated_field,
                                 *parent_fields, &match_list1, &match_list2) &&
      reporter_ == NULL) {
    return false;
  }

  bool fieldDifferent = false;
  SpecificField specific_field;
  specific_field.field = repeated_field;

  // At this point, we have already matched pairs of fields (with the reporting
  // to be done later). Now to check if the paired elements are different.
  for (int i = 0; i < count1; i++) {
    if (match_list1[i] == -1) continue;
    specific_field.index = i;
    specific_field.new_index = match_list1[i];

    const bool result = CompareFieldValueUsingParentFields(
        message1, message2, repeated_field, i, specific_field.new_index,
        parent_fields);

    // If we have found differences, either report them or terminate if
    // no reporter is present. Note that ReportModified, ReportMoved, and
    // ReportMatched are all mutually exclusive.
    if (!result) {
      if (reporter_ == NULL) return false;
      parent_fields->push_back(specific_field);
      reporter_->ReportModified(message1, message2, *parent_fields);
      parent_fields->pop_back();
      fieldDifferent = true;
    } else if (reporter_ != NULL &&
               specific_field.index != specific_field.new_index) {
      parent_fields->push_back(specific_field);
      reporter_->ReportMoved(message1, message2, *parent_fields);
      parent_fields->pop_back();
    } else if (report_matches_ && reporter_ != NULL) {
      parent_fields->push_back(specific_field);
      reporter_->ReportMatched(message1, message2, *parent_fields);
      parent_fields->pop_back();
    }
  }

  // Report any remaining additions or deletions.
  for (int i = 0; i < count2; ++i) {
    if (match_list2[i] != -1) continue;
    if (!treated_as_subset) {
      fieldDifferent = true;
    }

    if (reporter_ == NULL) continue;
    specific_field.index = i;
    specific_field.new_index = i;
    parent_fields->push_back(specific_field);
    reporter_->ReportAdded(message1, message2, *parent_fields);
    parent_fields->pop_back();
  }

  for (int i = 0; i < count1; ++i) {
    if (match_list1[i] != -1) continue;
    specific_field.index = i;
    parent_fields->push_back(specific_field);
    reporter_->ReportDeleted(message1, message2, *parent_fields);
    parent_fields->pop_back();
    fieldDifferent = true;
  }
  return !fieldDifferent;
}

bool MessageDifferencer::CompareFieldValue(const Message& message1,
                                           const Message& message2,
                                           const FieldDescriptor* field,
                                           int index1,
                                           int index2) {
  return CompareFieldValueUsingParentFields(message1, message2, field, index1,
                                            index2, NULL);
}

bool MessageDifferencer::CompareFieldValueUsingParentFields(
    const Message& message1, const Message& message2,
    const FieldDescriptor* field, int index1, int index2,
    vector<SpecificField>* parent_fields) {
  FieldContext field_context(parent_fields);
  FieldComparator::ComparisonResult result = GetFieldComparisonResult(
      message1, message2, field, index1, index2, &field_context);

  if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE &&
      result == FieldComparator::RECURSE) {
    // Get the nested messages and compare them using one of the Compare
    // methods.
    const Reflection* reflection1 = message1.GetReflection();
    const Reflection* reflection2 = message2.GetReflection();
    const Message& m1 = field->is_repeated() ?
        reflection1->GetRepeatedMessage(message1, field, index1) :
        reflection1->GetMessage(message1, field);
    const Message& m2 = field->is_repeated() ?
        reflection2->GetRepeatedMessage(message2, field, index2) :
        reflection2->GetMessage(message2, field);

    // parent_fields is used in calls to Reporter methods.
    if (parent_fields != NULL) {
      // Append currently compared field to the end of parent_fields.
      SpecificField specific_field;
      specific_field.field = field;
      specific_field.index = index1;
      specific_field.new_index = index2;
      parent_fields->push_back(specific_field);
      const bool compare_result = Compare(m1, m2, parent_fields);
      parent_fields->pop_back();
      return compare_result;
    } else {
      // Recreates parent_fields as if m1 and m2 had no parents.
      return Compare(m1, m2);
    }
  } else {
    return (result == FieldComparator::SAME);
  }
}

bool MessageDifferencer::CheckPathChanged(
    const vector<SpecificField>& field_path) {
  for (int i = 0; i < field_path.size(); ++i) {
    if (field_path[i].index != field_path[i].new_index) return true;
  }
  return false;
}

bool MessageDifferencer::IsTreatedAsSet(const FieldDescriptor* field) {
  if (!field->is_repeated()) return false;
  if (field->is_map()) return true;
  if (repeated_field_comparison_ == AS_SET) return true;
  return (set_fields_.find(field) != set_fields_.end());
}

bool MessageDifferencer::IsTreatedAsSubset(const FieldDescriptor* field) {
  return scope_ == PARTIAL &&
      (IsTreatedAsSet(field) || GetMapKeyComparator(field) != NULL);
}

bool MessageDifferencer::IsIgnored(
    const Message& message1,
    const Message& message2,
    const FieldDescriptor* field,
    const vector<SpecificField>& parent_fields) {
  if (ignored_fields_.find(field) != ignored_fields_.end()) {
    return true;
  }
  for (int i = 0; i < ignore_criteria_.size(); ++i) {
    if (ignore_criteria_[i]->IsIgnored(message1, message2, field,
                                       parent_fields)) {
      return true;
    }
  }
  return false;
}

bool MessageDifferencer::IsUnknownFieldIgnored(
    const Message& message1, const Message& message2,
    const SpecificField& field, const vector<SpecificField>& parent_fields) {
  for (int i = 0; i < ignore_criteria_.size(); ++i) {
    if (ignore_criteria_[i]->IsUnknownFieldIgnored(message1, message2, field,
                                                   parent_fields)) {
      return true;
    }
  }
  return false;
}

const MessageDifferencer::MapKeyComparator* MessageDifferencer
    ::GetMapKeyComparator(const FieldDescriptor* field) {
  if (!field->is_repeated()) return NULL;
  if (map_field_key_comparator_.find(field) !=
      map_field_key_comparator_.end()) {
    return map_field_key_comparator_[field];
  }
  return NULL;
}

namespace {

typedef pair<int, const UnknownField*> IndexUnknownFieldPair;

struct UnknownFieldOrdering {
  inline bool operator()(const IndexUnknownFieldPair& a,
                         const IndexUnknownFieldPair& b) const {
    if (a.second->number() < b.second->number()) return true;
    if (a.second->number() > b.second->number()) return false;
    return a.second->type() < b.second->type();
  }
};

}  // namespace

bool MessageDifferencer::UnpackAny(const Message& any,
                                   google::protobuf::scoped_ptr<Message>* data) {
  const Reflection* reflection = any.GetReflection();
  const FieldDescriptor* type_url_field;
  const FieldDescriptor* value_field;
  if (!internal::GetAnyFieldDescriptors(any, &type_url_field, &value_field)) {
    return false;
  }
  const string& type_url = reflection->GetString(any, type_url_field);
  string full_type_name;
  if (!internal::ParseAnyTypeUrl(type_url, &full_type_name)) {
    return false;
  }

  const google::protobuf::Descriptor* desc =
      any.GetDescriptor()->file()->pool()->FindMessageTypeByName(
          full_type_name);
  if (desc == NULL) {
    GOOGLE_LOG(ERROR) << "Proto type '" << full_type_name << "' not found";
    return false;
  }

  if (dynamic_message_factory_ == NULL) {
    dynamic_message_factory_.reset(new DynamicMessageFactory());
  }
  data->reset(dynamic_message_factory_->GetPrototype(desc)->New());
  string serialized_value = reflection->GetString(any, value_field);
  if (!(*data)->ParseFromString(serialized_value)) {
    GOOGLE_LOG(ERROR) << "Failed to parse value for " << full_type_name;
    return false;
  }
  return true;
}

bool MessageDifferencer::CompareUnknownFields(
    const Message& message1, const Message& message2,
    const google::protobuf::UnknownFieldSet& unknown_field_set1,
    const google::protobuf::UnknownFieldSet& unknown_field_set2,
    vector<SpecificField>* parent_field) {
  // Ignore unknown fields in EQUIVALENT mode.
  if (message_field_comparison_ == EQUIVALENT) return true;

  if (unknown_field_set1.empty() && unknown_field_set2.empty()) {
    return true;
  }

  bool is_different = false;

  // We first sort the unknown fields by field number and type (in other words,
  // in tag order), making sure to preserve ordering of values with the same
  // tag.  This allows us to report only meaningful differences between the
  // two sets -- that is, differing values for the same tag.  We use
  // IndexUnknownFieldPairs to keep track of the field's original index for
  // reporting purposes.
  vector<IndexUnknownFieldPair> fields1;  // unknown_field_set1, sorted
  vector<IndexUnknownFieldPair> fields2;  // unknown_field_set2, sorted
  fields1.reserve(unknown_field_set1.field_count());
  fields2.reserve(unknown_field_set2.field_count());

  for (int i = 0; i < unknown_field_set1.field_count(); i++) {
    fields1.push_back(std::make_pair(i, &unknown_field_set1.field(i)));
  }
  for (int i = 0; i < unknown_field_set2.field_count(); i++) {
    fields2.push_back(std::make_pair(i, &unknown_field_set2.field(i)));
  }

  UnknownFieldOrdering is_before;
  std::stable_sort(fields1.begin(), fields1.end(), is_before);
  std::stable_sort(fields2.begin(), fields2.end(), is_before);

  // In order to fill in SpecificField::index, we have to keep track of how
  // many values we've seen with the same field number and type.
  // current_repeated points at the first field in this range, and
  // current_repeated_start{1,2} are the indexes of the first field in the
  // range within fields1 and fields2.
  const UnknownField* current_repeated = NULL;
  int current_repeated_start1 = 0;
  int current_repeated_start2 = 0;

  // Now that we have two sorted lists, we can detect fields which appear only
  // in one list or the other by traversing them simultaneously.
  int index1 = 0;
  int index2 = 0;
  while (index1 < fields1.size() || index2 < fields2.size()) {
    enum { ADDITION, DELETION, MODIFICATION, COMPARE_GROUPS,
      NO_CHANGE } change_type;

    // focus_field is the field we're currently reporting on.  (In the case
    // of a modification, it's the field on the left side.)
    const UnknownField* focus_field;
    bool match = false;

    if (index2 == fields2.size() ||
        (index1 < fields1.size() &&
          is_before(fields1[index1], fields2[index2]))) {
      // fields1[index1] is not present in fields2.
      change_type = DELETION;
      focus_field = fields1[index1].second;
    } else if (index1 == fields1.size() ||
               is_before(fields2[index2], fields1[index1])) {
      // fields2[index2] is not present in fields1.
      if (scope_ == PARTIAL) {
        // Ignore.
        ++index2;
        continue;
      }
      change_type = ADDITION;
      focus_field = fields2[index2].second;
    } else {
      // Field type and number are the same.  See if the values differ.
      change_type = MODIFICATION;
      focus_field = fields1[index1].second;

      switch (focus_field->type()) {
        case UnknownField::TYPE_VARINT:
          match = fields1[index1].second->varint() ==
                  fields2[index2].second->varint();
          break;
        case UnknownField::TYPE_FIXED32:
          match = fields1[index1].second->fixed32() ==
                  fields2[index2].second->fixed32();
          break;
        case UnknownField::TYPE_FIXED64:
          match = fields1[index1].second->fixed64() ==
                  fields2[index2].second->fixed64();
          break;
        case UnknownField::TYPE_LENGTH_DELIMITED:
          match = fields1[index1].second->length_delimited() ==
                  fields2[index2].second->length_delimited();
          break;
        case UnknownField::TYPE_GROUP:
          // We must deal with this later, after building the SpecificField.
          change_type = COMPARE_GROUPS;
          break;
      }
      if (match && change_type != COMPARE_GROUPS) {
        change_type = NO_CHANGE;
      }
    }

    if (current_repeated == NULL ||
        focus_field->number() != current_repeated->number() ||
        focus_field->type() != current_repeated->type()) {
      // We've started a new repeated field.
      current_repeated = focus_field;
      current_repeated_start1 = index1;
      current_repeated_start2 = index2;
    }

    if (change_type == NO_CHANGE && reporter_ == NULL) {
      // Fields were already compared and matched and we have no reporter.
      ++index1;
      ++index2;
      continue;
    }

    // Build the SpecificField.  This is slightly complicated.
    SpecificField specific_field;
    specific_field.unknown_field_number = focus_field->number();
    specific_field.unknown_field_type = focus_field->type();

    specific_field.unknown_field_set1 = &unknown_field_set1;
    specific_field.unknown_field_set2 = &unknown_field_set2;

    if (change_type != ADDITION) {
      specific_field.unknown_field_index1 = fields1[index1].first;
    }
    if (change_type != DELETION) {
      specific_field.unknown_field_index2 = fields2[index2].first;
    }

    // Calculate the field index.
    if (change_type == ADDITION) {
      specific_field.index = index2 - current_repeated_start2;
      specific_field.new_index = index2 - current_repeated_start2;
    } else {
      specific_field.index = index1 - current_repeated_start1;
      specific_field.new_index = index2 - current_repeated_start2;
    }

    if (IsUnknownFieldIgnored(message1, message2, specific_field,
                              *parent_field)) {
      if (reporter_ != NULL) {
        parent_field->push_back(specific_field);
        reporter_->ReportUnknownFieldIgnored(message1, message2, *parent_field);
        parent_field->pop_back();
      }
      return true;
    }

    if (change_type == ADDITION || change_type == DELETION ||
        change_type == MODIFICATION) {
      if (reporter_ == NULL) {
        // We found a difference and we have no reproter.
        return false;
      }
      is_different = true;
    }

    parent_field->push_back(specific_field);

    switch (change_type) {
      case ADDITION:
        reporter_->ReportAdded(message1, message2, *parent_field);
        ++index2;
        break;
      case DELETION:
        reporter_->ReportDeleted(message1, message2, *parent_field);
        ++index1;
        break;
      case MODIFICATION:
        reporter_->ReportModified(message1, message2, *parent_field);
        ++index1;
        ++index2;
        break;
      case COMPARE_GROUPS:
        if (!CompareUnknownFields(message1, message2,
                                  fields1[index1].second->group(),
                                  fields2[index2].second->group(),
                                  parent_field)) {
          if (reporter_ == NULL) return false;
          is_different = true;
          reporter_->ReportModified(message1, message2, *parent_field);
        }
        ++index1;
        ++index2;
        break;
      case NO_CHANGE:
        ++index1;
        ++index2;
        if (report_matches_) {
          reporter_->ReportMatched(message1, message2, *parent_field);
        }
    }

    parent_field->pop_back();
  }

  return !is_different;
}

namespace {

// Find maximum bipartite matching using the argumenting path algorithm.
class MaximumMatcher {
 public:
  typedef ResultCallback2<bool, int, int> NodeMatchCallback;
  // MaximumMatcher takes ownership of the passed in callback and uses it to
  // determine whether a node on the left side of the bipartial graph matches
  // a node on the right side. count1 is the number of nodes on the left side
  // of the graph and count2 to is the number of nodes on the right side.
  // Every node is referred to using 0-based indices.
  // If a maximum match is found, the result will be stored in match_list1 and
  // match_list2. match_list1[i] == j means the i-th node on the left side is
  // matched to the j-th node on the right side and match_list2[x] == y means
  // the x-th node on the right side is matched to y-th node on the left side.
  // match_list1[i] == -1 means the node is not matched. Same with match_list2.
  MaximumMatcher(int count1, int count2, NodeMatchCallback* callback,
                 vector<int>* match_list1, vector<int>* match_list2);
  // Find a maximum match and return the number of matched node pairs.
  // If early_return is true, this method will return 0 immediately when it
  // finds that not all nodes on the left side can be matched.
  int FindMaximumMatch(bool early_return);
 private:
  // Determines whether the node on the left side of the bipartial graph
  // matches the one on the right side.
  bool Match(int left, int right);
  // Find an argumenting path starting from the node v on the left side. If a
  // path can be found, update match_list2_ to reflect the path and return
  // true.
  bool FindArgumentPathDFS(int v, vector<bool>* visited);

  int count1_;
  int count2_;
  google::protobuf::scoped_ptr<NodeMatchCallback> match_callback_;
  map<pair<int, int>, bool> cached_match_results_;
  vector<int>* match_list1_;
  vector<int>* match_list2_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(MaximumMatcher);
};

MaximumMatcher::MaximumMatcher(int count1, int count2,
                               NodeMatchCallback* callback,
                               vector<int>* match_list1,
                               vector<int>* match_list2)
    : count1_(count1), count2_(count2), match_callback_(callback),
      match_list1_(match_list1), match_list2_(match_list2) {
  match_list1_->assign(count1, -1);
  match_list2_->assign(count2, -1);
}

int MaximumMatcher::FindMaximumMatch(bool early_return) {
  int result = 0;
  for (int i = 0; i < count1_; ++i) {
    vector<bool> visited(count1_);
    if (FindArgumentPathDFS(i, &visited)) {
      ++result;
    } else if (early_return) {
      return 0;
    }
  }
  // Backfill match_list1_ as we only filled match_list2_ when finding
  // argumenting pathes.
  for (int i = 0; i < count2_; ++i) {
    if ((*match_list2_)[i] != -1) {
      (*match_list1_)[(*match_list2_)[i]] = i;
    }
  }
  return result;
}

bool MaximumMatcher::Match(int left, int right) {
  pair<int, int> p(left, right);
  map<pair<int, int>, bool>::iterator it = cached_match_results_.find(p);
  if (it != cached_match_results_.end()) {
    return it->second;
  }
  cached_match_results_[p] = match_callback_->Run(left, right);
  return cached_match_results_[p];
}

bool MaximumMatcher::FindArgumentPathDFS(int v, vector<bool>* visited) {
  (*visited)[v] = true;
  // We try to match those un-matched nodes on the right side first. This is
  // the step that the navie greedy matching algorithm uses. In the best cases
  // where the greedy algorithm can find a maximum matching, we will always
  // find a match in this step and the performance will be identical to the
  // greedy algorithm.
  for (int i = 0; i < count2_; ++i) {
    int matched = (*match_list2_)[i];
    if (matched == -1 && Match(v, i)) {
      (*match_list2_)[i] = v;
      return true;
    }
  }
  // Then we try those already matched nodes and see if we can find an
  // alternaive match for the node matched to them.
  // The greedy algorithm will stop before this and fail to produce the
  // correct result.
  for (int i = 0; i < count2_; ++i) {
    int matched = (*match_list2_)[i];
    if (matched != -1 && Match(v, i)) {
      if (!(*visited)[matched] && FindArgumentPathDFS(matched, visited)) {
        (*match_list2_)[i] = v;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

bool MessageDifferencer::MatchRepeatedFieldIndices(
    const Message& message1,
    const Message& message2,
    const FieldDescriptor* repeated_field,
    const vector<SpecificField>& parent_fields,
    vector<int>* match_list1,
    vector<int>* match_list2) {
  const int count1 =
      message1.GetReflection()->FieldSize(message1, repeated_field);
  const int count2 =
      message2.GetReflection()->FieldSize(message2, repeated_field);
  const MapKeyComparator* key_comparator = GetMapKeyComparator(repeated_field);

  match_list1->assign(count1, -1);
  match_list2->assign(count2, -1);

  SpecificField specific_field;
  specific_field.field = repeated_field;

  bool success = true;
  // Find potential match if this is a special repeated field.
  if (key_comparator != NULL || IsTreatedAsSet(repeated_field)) {
    if (scope_ == PARTIAL) {
      // When partial matching is enabled, Compare(a, b) && Compare(a, c)
      // doesn't neccessarily imply Compare(b, c). Therefore a naive greedy
      // algorithm will fail to find a maximum matching.
      // Here we use the argumenting path algorithm.
      MaximumMatcher::NodeMatchCallback* callback =
          google::protobuf::internal::NewPermanentCallback(
              this, &MessageDifferencer::IsMatch,
              repeated_field, key_comparator,
              &message1, &message2, parent_fields);
      MaximumMatcher matcher(count1, count2, callback, match_list1,
                             match_list2);
      // If diff info is not needed, we should end the matching process as
      // soon as possible if not all items can be matched.
      bool early_return = (reporter_ == NULL);
      int match_count = matcher.FindMaximumMatch(early_return);
      if (match_count != count1 && reporter_ == NULL) return false;
      success = success && (match_count == count1);
    } else {
      for (int i = 0; i < count1; ++i) {
        // Indicates any matched elements for this repeated field.
        bool match = false;

        specific_field.index = i;
        specific_field.new_index = i;

        for (int j = 0; j < count2; j++) {
          if (match_list2->at(j) != -1) continue;
          specific_field.index = i;
          specific_field.new_index = j;

          match = IsMatch(repeated_field, key_comparator,
                          &message1, &message2, parent_fields, i, j);

          if (match) {
            match_list1->at(specific_field.index) = specific_field.new_index;
            match_list2->at(specific_field.new_index) = specific_field.index;
            break;
          }
        }
        if (!match && reporter_ == NULL) return false;
        success = success && match;
      }
    }
  } else {
    // If this field should be treated as list, just label the match_list.
    for (int i = 0; i < count1 && i < count2; i++) {
      match_list1->at(i) = i;
      match_list2->at(i) = i;
    }
  }

  return success;
}

FieldComparator::ComparisonResult MessageDifferencer::GetFieldComparisonResult(
    const Message& message1, const Message& message2,
    const FieldDescriptor* field, int index1, int index2,
    const FieldContext* field_context) {
  FieldComparator* comparator = field_comparator_ != NULL ?
      field_comparator_ : &default_field_comparator_;
  return comparator->Compare(message1, message2, field,
                             index1, index2, field_context);
}

// ===========================================================================

MessageDifferencer::Reporter::Reporter() { }
MessageDifferencer::Reporter::~Reporter() {}

// ===========================================================================

MessageDifferencer::MapKeyComparator::MapKeyComparator() {}
MessageDifferencer::MapKeyComparator::~MapKeyComparator() {}

// ===========================================================================

MessageDifferencer::IgnoreCriteria::IgnoreCriteria() {}
MessageDifferencer::IgnoreCriteria::~IgnoreCriteria() {}

// ===========================================================================

// Note that the printer's delimiter is not used, because if we are given a
// printer, we don't know its delimiter.
MessageDifferencer::StreamReporter::StreamReporter(
    io::ZeroCopyOutputStream* output) : printer_(new io::Printer(output, '$')),
                                        delete_printer_(true),
                                        report_modified_aggregates_(false) { }

MessageDifferencer::StreamReporter::StreamReporter(
    io::Printer* printer) : printer_(printer),
                            delete_printer_(false),
                            report_modified_aggregates_(false) { }

MessageDifferencer::StreamReporter::~StreamReporter() {
  if (delete_printer_) delete printer_;
}

void MessageDifferencer::StreamReporter::PrintPath(
    const vector<SpecificField>& field_path, bool left_side) {
  for (int i = 0; i < field_path.size(); ++i) {
    if (i > 0) {
      printer_->Print(".");
    }

    SpecificField specific_field = field_path[i];

    if (specific_field.field != NULL) {
      if (specific_field.field->is_extension()) {
        printer_->Print("($name$)", "name",
                        specific_field.field->full_name());
      } else {
        printer_->PrintRaw(specific_field.field->name());
      }
    } else {
      printer_->PrintRaw(SimpleItoa(specific_field.unknown_field_number));
    }
    if (left_side && specific_field.index >= 0) {
      printer_->Print("[$name$]", "name", SimpleItoa(specific_field.index));
    }
    if (!left_side && specific_field.new_index >= 0) {
      printer_->Print("[$name$]", "name", SimpleItoa(specific_field.new_index));
    }
  }
}

void MessageDifferencer::
StreamReporter::PrintValue(const Message& message,
                           const vector<SpecificField>& field_path,
                           bool left_side) {
  const SpecificField& specific_field = field_path.back();
  const FieldDescriptor* field = specific_field.field;
  if (field != NULL) {
    string output;
    int index = left_side ? specific_field.index : specific_field.new_index;
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      const Reflection* reflection = message.GetReflection();
      const Message& field_message = field->is_repeated() ?
          reflection->GetRepeatedMessage(message, field, index) :
          reflection->GetMessage(message, field);
      output = field_message.ShortDebugString();
      if (output.empty()) {
        printer_->Print("{ }");
      } else {
        printer_->Print("{ $name$ }", "name", output);
      }
    } else {
      TextFormat::PrintFieldValueToString(message, field, index, &output);
      printer_->PrintRaw(output);
    }
  } else {
    const UnknownFieldSet* unknown_fields =
        (left_side ?
         specific_field.unknown_field_set1 :
         specific_field.unknown_field_set2);
    const UnknownField* unknown_field = &unknown_fields->field(
        left_side ?
        specific_field.unknown_field_index1 :
        specific_field.unknown_field_index2);
    PrintUnknownFieldValue(unknown_field);
  }
}

void MessageDifferencer::
StreamReporter::PrintUnknownFieldValue(const UnknownField* unknown_field) {
  GOOGLE_CHECK(unknown_field != NULL) << " Cannot print NULL unknown_field.";

  string output;
  switch (unknown_field->type()) {
    case UnknownField::TYPE_VARINT:
      output = SimpleItoa(unknown_field->varint());
      break;
    case UnknownField::TYPE_FIXED32:
      output = StrCat("0x", strings::Hex(unknown_field->fixed32(),
                                         strings::ZERO_PAD_8));
      break;
    case UnknownField::TYPE_FIXED64:
      output = StrCat("0x", strings::Hex(unknown_field->fixed64(),
                                         strings::ZERO_PAD_16));
      break;
    case UnknownField::TYPE_LENGTH_DELIMITED:
      output = StringPrintf("\"%s\"",
          CEscape(unknown_field->length_delimited()).c_str());
      break;
    case UnknownField::TYPE_GROUP:
      // TODO(kenton):  Print the contents of the group like we do for
      //   messages.  Requires an equivalent of ShortDebugString() for
      //   UnknownFieldSet.
      output = "{ ... }";
      break;
  }
  printer_->PrintRaw(output);
}

void MessageDifferencer::StreamReporter::Print(const string& str) {
  printer_->Print(str.c_str());
}

void MessageDifferencer::StreamReporter::ReportAdded(
    const Message& message1,
    const Message& message2,
    const vector<SpecificField>& field_path) {
  printer_->Print("added: ");
  PrintPath(field_path, false);
  printer_->Print(": ");
  PrintValue(message2, field_path, false);
  printer_->Print("\n");  // Print for newlines.
}

void MessageDifferencer::StreamReporter::ReportDeleted(
    const Message& message1,
    const Message& message2,
    const vector<SpecificField>& field_path) {
  printer_->Print("deleted: ");
  PrintPath(field_path, true);
  printer_->Print(": ");
  PrintValue(message1, field_path, true);
  printer_->Print("\n");  // Print for newlines
}

void MessageDifferencer::StreamReporter::ReportModified(
    const Message& message1,
    const Message& message2,
    const vector<SpecificField>& field_path) {
  if (!report_modified_aggregates_ && field_path.back().field == NULL) {
    if (field_path.back().unknown_field_type == UnknownField::TYPE_GROUP) {
      // Any changes to the subfields have already been printed.
      return;
    }
  } else if (!report_modified_aggregates_) {
    if (field_path.back().field->cpp_type() ==
        FieldDescriptor::CPPTYPE_MESSAGE) {
      // Any changes to the subfields have already been printed.
      return;
    }
  }

  printer_->Print("modified: ");
  PrintPath(field_path, true);
  if (CheckPathChanged(field_path)) {
    printer_->Print(" -> ");
    PrintPath(field_path, false);
  }
  printer_->Print(": ");
  PrintValue(message1, field_path, true);
  printer_->Print(" -> ");
  PrintValue(message2, field_path, false);
  printer_->Print("\n");  // Print for newlines.
}

void MessageDifferencer::StreamReporter::ReportMoved(
    const Message& message1,
    const Message& message2,
    const vector<SpecificField>& field_path) {
  printer_->Print("moved: ");
  PrintPath(field_path, true);
  printer_->Print(" -> ");
  PrintPath(field_path, false);
  printer_->Print(" : ");
  PrintValue(message1, field_path, true);
  printer_->Print("\n");  // Print for newlines.
}

void MessageDifferencer::StreamReporter::ReportMatched(
    const Message& message1,
    const Message& message2,
    const vector<SpecificField>& field_path) {
  printer_->Print("matched: ");
  PrintPath(field_path, true);
  if (CheckPathChanged(field_path)) {
    printer_->Print(" -> ");
    PrintPath(field_path, false);
  }
  printer_->Print(" : ");
  PrintValue(message1, field_path, true);
  printer_->Print("\n");  // Print for newlines.
}

void MessageDifferencer::StreamReporter::ReportIgnored(
    const Message& message1,
    const Message& message2,
    const vector<SpecificField>& field_path) {
  printer_->Print("ignored: ");
  PrintPath(field_path, true);
  if (CheckPathChanged(field_path)) {
    printer_->Print(" -> ");
    PrintPath(field_path, false);
  }
  printer_->Print("\n");  // Print for newlines.
}

void MessageDifferencer::StreamReporter::ReportUnknownFieldIgnored(
    const Message& message1, const Message& message2,
    const vector<SpecificField>& field_path) {
  printer_->Print("ignored: ");
  PrintPath(field_path, true);
  if (CheckPathChanged(field_path)) {
    printer_->Print(" -> ");
    PrintPath(field_path, false);
  }
  printer_->Print("\n");  // Print for newlines.
}

}  // namespace util
}  // namespace protobuf
}  // namespace google
