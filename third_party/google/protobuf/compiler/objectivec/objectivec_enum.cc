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

#include <map>
#include <string>

#include <google/protobuf/compiler/objectivec/objectivec_enum.h>
#include <google/protobuf/compiler/objectivec/objectivec_helpers.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/stubs/strutil.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace objectivec {

EnumGenerator::EnumGenerator(const EnumDescriptor* descriptor)
    : descriptor_(descriptor),
      name_(EnumName(descriptor_)) {
  for (int i = 0; i < descriptor_->value_count(); i++) {
    const EnumValueDescriptor* value = descriptor_->value(i);
    const EnumValueDescriptor* canonical_value =
        descriptor_->FindValueByNumber(value->number());

    if (value == canonical_value) {
      base_values_.push_back(value);
    }
    all_values_.push_back(value);
  }
}

EnumGenerator::~EnumGenerator() {}

void EnumGenerator::GenerateHeader(io::Printer* printer) {
  string enum_comments;
  SourceLocation location;
  if (descriptor_->GetSourceLocation(&location)) {
    enum_comments = BuildCommentsString(location);
  } else {
    enum_comments = "";
  }

  printer->Print(
      "#pragma mark - Enum $name$\n"
      "\n",
      "name", name_);

  printer->Print("$comments$typedef GPB_ENUM($name$) {\n",
                 "comments", enum_comments,
                 "name", name_);
  printer->Indent();

  if (HasPreservingUnknownEnumSemantics(descriptor_->file())) {
    // Include the unknown value.
    printer->Print(
      "$name$_GPBUnrecognizedEnumeratorValue = kGPBUnrecognizedEnumeratorValue,\n",
      "name", name_);
  }

  for (int i = 0; i < all_values_.size(); i++) {
    SourceLocation location;
    if (all_values_[i]->GetSourceLocation(&location)) {
      string comments = BuildCommentsString(location).c_str();
      if (comments.length() > 0) {
        if (i > 0) {
          printer->Print("\n");
        }
        printer->Print(comments.c_str());
      }
    }

    printer->Print(
        "$name$ = $value$,\n",
        "name", EnumValueName(all_values_[i]),
        "value", SimpleItoa(all_values_[i]->number()));
  }
  printer->Outdent();
  printer->Print(
      "};\n"
      "\n"
      "GPBEnumDescriptor *$name$_EnumDescriptor(void);\n"
      "\n"
      "BOOL $name$_IsValidValue(int32_t value);\n"
      "\n",
      "name", name_);
}

void EnumGenerator::GenerateSource(io::Printer* printer) {
  printer->Print(
      "#pragma mark - Enum $name$\n"
      "\n",
      "name", name_);

  printer->Print(
      "GPBEnumDescriptor *$name$_EnumDescriptor(void) {\n"
      "  static GPBEnumDescriptor *descriptor = NULL;\n"
      "  if (!descriptor) {\n"
      "    static GPBMessageEnumValueDescription values[] = {\n",
      "name", name_);
  printer->Indent();
  printer->Indent();
  printer->Indent();

  // Note: For the TextFormat decode info, we can't use the enum value as
  // the key because protocol buffer enums have 'allow_alias', which lets
  // a value be used more than once. Instead, the index into the list of
  // enum value descriptions is used. Note: start with -1 so the first one
  // will be zero.
  TextFormatDecodeData text_format_decode_data;
  int enum_value_description_key = -1;

  for (int i = 0; i < all_values_.size(); i++) {
    ++enum_value_description_key;
    string short_name(EnumValueShortName(all_values_[i]));
    printer->Print("{ .name = \"$short_name$\", .number = $name$ },\n",
                   "short_name", short_name,
                   "name", EnumValueName(all_values_[i]));
    if (UnCamelCaseEnumShortName(short_name) != all_values_[i]->name()) {
      text_format_decode_data.AddString(enum_value_description_key, short_name,
                                        all_values_[i]->name());
    }
  }
  printer->Outdent();
  printer->Outdent();
  printer->Outdent();
  printer->Print("    };\n");
  if (text_format_decode_data.num_entries() == 0) {
    printer->Print(
        "    descriptor = [GPBEnumDescriptor allocDescriptorForName:GPBNSStringifySymbol($name$)\n"
        "                                                   values:values\n"
        "                                               valueCount:sizeof(values) / sizeof(GPBMessageEnumValueDescription)\n"
        "                                             enumVerifier:$name$_IsValidValue];\n",
        "name", name_);
    } else {
      printer->Print(
        "    static const char *extraTextFormatInfo = \"$extraTextFormatInfo$\";\n"
        "    descriptor = [GPBEnumDescriptor allocDescriptorForName:GPBNSStringifySymbol($name$)\n"
        "                                                   values:values\n"
        "                                               valueCount:sizeof(values) / sizeof(GPBMessageEnumValueDescription)\n"
        "                                             enumVerifier:$name$_IsValidValue\n"
        "                                      extraTextFormatInfo:extraTextFormatInfo];\n",
        "name", name_,
        "extraTextFormatInfo", CEscape(text_format_decode_data.Data()));
    }
    printer->Print(
      "  }\n"
      "  return descriptor;\n"
      "}\n\n");

  printer->Print(
      "BOOL $name$_IsValidValue(int32_t value__) {\n"
      "  switch (value__) {\n",
      "name", name_);

  for (int i = 0; i < base_values_.size(); i++) {
    printer->Print(
        "    case $name$:\n",
        "name", EnumValueName(base_values_[i]));
  }

  printer->Print(
      "      return YES;\n"
      "    default:\n"
      "      return NO;\n"
      "  }\n"
      "}\n\n");
}
}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
