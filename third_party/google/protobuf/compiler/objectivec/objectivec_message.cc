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

#include <algorithm>
#include <iostream>
#include <sstream>

#include <google/protobuf/stubs/hash.h>
#include <google/protobuf/compiler/objectivec/objectivec_message.h>
#include <google/protobuf/compiler/objectivec/objectivec_enum.h>
#include <google/protobuf/compiler/objectivec/objectivec_extension.h>
#include <google/protobuf/compiler/objectivec/objectivec_helpers.h>
#include <google/protobuf/stubs/stl_util.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.pb.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace objectivec {

using internal::WireFormat;
using internal::WireFormatLite;

namespace {
struct FieldOrderingByNumber {
  inline bool operator()(const FieldDescriptor* a,
                         const FieldDescriptor* b) const {
    return a->number() < b->number();
  }
};

int OrderGroupForFieldDescriptor(const FieldDescriptor* descriptor) {
  // The first item in the object structure is our uint32[] for has bits.
  // We then want to order things to make the instances as small as
  // possible. So we follow the has bits with:
  //   1. Bools (1 byte)
  //   2. Anything always 4 bytes - float, *32, enums
  //   3. Anything that is always a pointer (they will be 8 bytes on 64 bit
  //      builds and 4 bytes on 32bit builds.
  //   4. Anything always 8 bytes - double, *64
  //
  // Why? Using 64bit builds as an example, this means worse case, we have
  // enough bools that we overflow 1 byte from 4 byte alignment, so 3 bytes
  // are wasted before the 4 byte values. Then if we have an odd number of
  // those 4 byte values, the 8 byte values will be pushed down by 32bits to
  // keep them aligned. But the structure will end 8 byte aligned, so no
  // waste on the end. If you did the reverse order, you could waste 4 bytes
  // before the first 8 byte value (after the has array), then a single
  // bool on the end would need 7 bytes of padding to make the overall
  // structure 8 byte aligned; so 11 bytes, wasted total.

  // Anything repeated is a GPB*Array/NSArray, so pointer.
  if (descriptor->is_repeated()) {
    return 3;
  }

  switch (descriptor->type()) {
    // All always 8 bytes.
    case FieldDescriptor::TYPE_DOUBLE:
    case FieldDescriptor::TYPE_INT64:
    case FieldDescriptor::TYPE_SINT64:
    case FieldDescriptor::TYPE_UINT64:
    case FieldDescriptor::TYPE_SFIXED64:
    case FieldDescriptor::TYPE_FIXED64:
      return 4;

    // Pointers (string and bytes are NSString and NSData); 8 or 4 bytes
    // depending on the build architecture.
    case FieldDescriptor::TYPE_GROUP:
    case FieldDescriptor::TYPE_MESSAGE:
    case FieldDescriptor::TYPE_STRING:
    case FieldDescriptor::TYPE_BYTES:
      return 3;

    // All always 4 bytes (enums are int32s).
    case FieldDescriptor::TYPE_FLOAT:
    case FieldDescriptor::TYPE_INT32:
    case FieldDescriptor::TYPE_SINT32:
    case FieldDescriptor::TYPE_UINT32:
    case FieldDescriptor::TYPE_SFIXED32:
    case FieldDescriptor::TYPE_FIXED32:
    case FieldDescriptor::TYPE_ENUM:
      return 2;

    // 1 byte.
    case FieldDescriptor::TYPE_BOOL:
      return 1;
  }

  // Some compilers report reaching end of function even though all cases of
  // the enum are handed in the switch.
  GOOGLE_LOG(FATAL) << "Can't get here.";
  return 0;
}

struct FieldOrderingByStorageSize {
  inline bool operator()(const FieldDescriptor* a,
                         const FieldDescriptor* b) const {
    // Order by grouping.
    const int order_group_a = OrderGroupForFieldDescriptor(a);
    const int order_group_b = OrderGroupForFieldDescriptor(b);
    if (order_group_a != order_group_b) {
      return order_group_a < order_group_b;
    }
    // Within the group, order by field number (provides stable ordering).
    return a->number() < b->number();
  }
};

struct ExtensionRangeOrdering {
  bool operator()(const Descriptor::ExtensionRange* a,
                  const Descriptor::ExtensionRange* b) const {
    return a->start < b->start;
  }
};

// Sort the fields of the given Descriptor by number into a new[]'d array
// and return it.
const FieldDescriptor** SortFieldsByNumber(const Descriptor* descriptor) {
  const FieldDescriptor** fields =
      new const FieldDescriptor* [descriptor->field_count()];
  for (int i = 0; i < descriptor->field_count(); i++) {
    fields[i] = descriptor->field(i);
  }
  sort(fields, fields + descriptor->field_count(), FieldOrderingByNumber());
  return fields;
}

// Sort the fields of the given Descriptor by storage size into a new[]'d
// array and return it.
const FieldDescriptor** SortFieldsByStorageSize(const Descriptor* descriptor) {
  const FieldDescriptor** fields =
      new const FieldDescriptor* [descriptor->field_count()];
  for (int i = 0; i < descriptor->field_count(); i++) {
    fields[i] = descriptor->field(i);
  }
  sort(fields, fields + descriptor->field_count(),
       FieldOrderingByStorageSize());
  return fields;
}
}  // namespace

MessageGenerator::MessageGenerator(const string& root_classname,
                                   const Descriptor* descriptor)
    : root_classname_(root_classname),
      descriptor_(descriptor),
      field_generators_(descriptor),
      class_name_(ClassName(descriptor_)) {
  for (int i = 0; i < descriptor_->extension_count(); i++) {
    extension_generators_.push_back(
        new ExtensionGenerator(class_name_, descriptor_->extension(i)));
  }

  for (int i = 0; i < descriptor_->oneof_decl_count(); i++) {
    OneofGenerator* generator = new OneofGenerator(descriptor_->oneof_decl(i));
    oneof_generators_.push_back(generator);
  }

  for (int i = 0; i < descriptor_->enum_type_count(); i++) {
    EnumGenerator* generator = new EnumGenerator(descriptor_->enum_type(i));
    enum_generators_.push_back(generator);
  }

  for (int i = 0; i < descriptor_->nested_type_count(); i++) {
    MessageGenerator* generator =
        new MessageGenerator(root_classname_, descriptor_->nested_type(i));
    nested_message_generators_.push_back(generator);
  }
}

MessageGenerator::~MessageGenerator() {
  STLDeleteContainerPointers(extension_generators_.begin(),
                             extension_generators_.end());
  STLDeleteContainerPointers(enum_generators_.begin(), enum_generators_.end());
  STLDeleteContainerPointers(nested_message_generators_.begin(),
                             nested_message_generators_.end());
  STLDeleteContainerPointers(oneof_generators_.begin(),
                             oneof_generators_.end());
}

void MessageGenerator::GenerateStaticVariablesInitialization(
    io::Printer* printer) {
  for (vector<ExtensionGenerator*>::iterator iter =
           extension_generators_.begin();
       iter != extension_generators_.end(); ++iter) {
    (*iter)->GenerateStaticVariablesInitialization(printer);
  }

  for (vector<MessageGenerator*>::iterator iter =
           nested_message_generators_.begin();
       iter != nested_message_generators_.end(); ++iter) {
    (*iter)->GenerateStaticVariablesInitialization(printer);
  }
}

void MessageGenerator::DetermineForwardDeclarations(set<string>* fwd_decls) {
  if (!IsMapEntryMessage(descriptor_)) {
    for (int i = 0; i < descriptor_->field_count(); i++) {
      const FieldDescriptor* fieldDescriptor = descriptor_->field(i);
      // If it is a the field is repeated, the type will be and *Array, and we
      // don't need any forward decl.
      if (fieldDescriptor->is_repeated()) {
        continue;
      }
      field_generators_.get(fieldDescriptor)
          .DetermineForwardDeclarations(fwd_decls);
    }
  }

  for (vector<MessageGenerator*>::iterator iter =
           nested_message_generators_.begin();
       iter != nested_message_generators_.end(); ++iter) {
    (*iter)->DetermineForwardDeclarations(fwd_decls);
  }
}

void MessageGenerator::GenerateEnumHeader(io::Printer* printer) {
  for (vector<EnumGenerator*>::iterator iter = enum_generators_.begin();
       iter != enum_generators_.end(); ++iter) {
    (*iter)->GenerateHeader(printer);
  }

  for (vector<MessageGenerator*>::iterator iter =
           nested_message_generators_.begin();
       iter != nested_message_generators_.end(); ++iter) {
    (*iter)->GenerateEnumHeader(printer);
  }
}

void MessageGenerator::GenerateExtensionRegistrationSource(
    io::Printer* printer) {
  for (vector<ExtensionGenerator*>::iterator iter =
           extension_generators_.begin();
       iter != extension_generators_.end(); ++iter) {
    (*iter)->GenerateRegistrationSource(printer);
  }

  for (vector<MessageGenerator*>::iterator iter =
           nested_message_generators_.begin();
       iter != nested_message_generators_.end(); ++iter) {
    (*iter)->GenerateExtensionRegistrationSource(printer);
  }
}

void MessageGenerator::GenerateMessageHeader(io::Printer* printer) {
  // This a a map entry message, just recurse and do nothing directly.
  if (IsMapEntryMessage(descriptor_)) {
    for (vector<MessageGenerator*>::iterator iter =
             nested_message_generators_.begin();
         iter != nested_message_generators_.end(); ++iter) {
      (*iter)->GenerateMessageHeader(printer);
    }
    return;
  }

  printer->Print(
      "#pragma mark - $classname$\n"
      "\n",
      "classname", class_name_);

  if (descriptor_->field_count()) {
    scoped_array<const FieldDescriptor*> sorted_fields(
        SortFieldsByNumber(descriptor_));

    printer->Print("typedef GPB_ENUM($classname$_FieldNumber) {\n",
                   "classname", class_name_);
    printer->Indent();

    for (int i = 0; i < descriptor_->field_count(); i++) {
      field_generators_.get(sorted_fields[i])
          .GenerateFieldNumberConstant(printer);
    }

    printer->Outdent();
    printer->Print("};\n\n");
  }

  for (vector<OneofGenerator*>::iterator iter = oneof_generators_.begin();
       iter != oneof_generators_.end(); ++iter) {
    (*iter)->GenerateCaseEnum(printer);
  }

  string message_comments;
  SourceLocation location;
  if (descriptor_->GetSourceLocation(&location)) {
    message_comments = BuildCommentsString(location);
  } else {
    message_comments = "";
  }

  printer->Print(
      "$comments$@interface $classname$ : GPBMessage\n\n",
      "classname", class_name_,
      "comments", message_comments);

  vector<char> seen_oneofs(descriptor_->oneof_decl_count(), 0);
  for (int i = 0; i < descriptor_->field_count(); i++) {
    const FieldDescriptor* field = descriptor_->field(i);
    if (field->containing_oneof() != NULL) {
      const int oneof_index = field->containing_oneof()->index();
      if (!seen_oneofs[oneof_index]) {
        seen_oneofs[oneof_index] = 1;
        oneof_generators_[oneof_index]->GeneratePublicCasePropertyDeclaration(
            printer);
      }
    }
    field_generators_.get(field).GeneratePropertyDeclaration(printer);
  }

  printer->Print("@end\n\n");

  for (int i = 0; i < descriptor_->field_count(); i++) {
    field_generators_.get(descriptor_->field(i))
        .GenerateCFunctionDeclarations(printer);
  }

  if (!oneof_generators_.empty()) {
    for (vector<OneofGenerator*>::iterator iter = oneof_generators_.begin();
         iter != oneof_generators_.end(); ++iter) {
      (*iter)->GenerateClearFunctionDeclaration(printer);
    }
    printer->Print("\n");
  }

  if (descriptor_->extension_count() > 0) {
    printer->Print("@interface $classname$ (DynamicMethods)\n\n",
                   "classname", class_name_);
    for (vector<ExtensionGenerator*>::iterator iter =
             extension_generators_.begin();
         iter != extension_generators_.end(); ++iter) {
      (*iter)->GenerateMembersHeader(printer);
    }
    printer->Print("@end\n\n");
  }

  for (vector<MessageGenerator*>::iterator iter =
           nested_message_generators_.begin();
       iter != nested_message_generators_.end(); ++iter) {
    (*iter)->GenerateMessageHeader(printer);
  }
}

void MessageGenerator::GenerateSource(io::Printer* printer) {
  if (!IsMapEntryMessage(descriptor_)) {
    printer->Print(
        "#pragma mark - $classname$\n"
        "\n",
        "classname", class_name_);

    printer->Print("@implementation $classname$\n\n",
                   "classname", class_name_);

    for (vector<OneofGenerator*>::iterator iter = oneof_generators_.begin();
         iter != oneof_generators_.end(); ++iter) {
      (*iter)->GeneratePropertyImplementation(printer);
    }

    for (int i = 0; i < descriptor_->field_count(); i++) {
      field_generators_.get(descriptor_->field(i))
          .GeneratePropertyImplementation(printer);
    }

    scoped_array<const FieldDescriptor*> sorted_fields(
        SortFieldsByNumber(descriptor_));
    scoped_array<const FieldDescriptor*> size_order_fields(
        SortFieldsByStorageSize(descriptor_));

    vector<const Descriptor::ExtensionRange*> sorted_extensions;
    for (int i = 0; i < descriptor_->extension_range_count(); ++i) {
      sorted_extensions.push_back(descriptor_->extension_range(i));
    }

    sort(sorted_extensions.begin(), sorted_extensions.end(),
         ExtensionRangeOrdering());

    // TODO(thomasvl): Finish optimizing has bit. The current behavior is as
    // follows:
    // 1. objectivec_field.cc's SetCommonFieldVariables() defaults the has_index
    //    to the field's index in the list of fields.
    // 2. RepeatedFieldGenerator::RepeatedFieldGenerator() sets has_index to
    //    GPBNoHasBit because repeated fields & map<> fields don't use the has
    //    bit.
    // 3. FieldGenerator::SetOneofIndexBase() overrides has_bit with a negative
    //    index that groups all the elements on of the oneof.
    // So in has_storage, we need enough bits for the single fields that aren't
    // in any oneof, and then one int32 for each oneof (to store the field
    // number).  So we could save a little space by not using the field's index
    // and instead make a second pass only assigning indexes for the fields
    // that would need it.  The only savings would come when messages have over
    // a multiple of 32 fields with some number being repeated or in oneofs to
    // drop the count below that 32 multiple; so it hasn't seemed worth doing
    // at the moment.
    size_t num_has_bits = descriptor_->field_count();
    size_t sizeof_has_storage = (num_has_bits + 31) / 32;
    // Tell all the fields the oneof base.
    for (vector<OneofGenerator*>::iterator iter = oneof_generators_.begin();
         iter != oneof_generators_.end(); ++iter) {
      (*iter)->SetOneofIndexBase(sizeof_has_storage);
    }
    field_generators_.SetOneofIndexBase(sizeof_has_storage);
    // Add an int32 for each oneof to store which is set.
    sizeof_has_storage += descriptor_->oneof_decl_count();

    printer->Print(
        "\n"
        "typedef struct $classname$__storage_ {\n"
        "  uint32_t _has_storage_[$sizeof_has_storage$];\n",
        "classname", class_name_,
        "sizeof_has_storage", SimpleItoa(sizeof_has_storage));
    printer->Indent();

    for (int i = 0; i < descriptor_->field_count(); i++) {
      field_generators_.get(size_order_fields[i])
          .GenerateFieldStorageDeclaration(printer);
    }
    printer->Outdent();

    printer->Print("} $classname$__storage_;\n\n", "classname", class_name_);


    printer->Print(
        "// This method is threadsafe because it is initially called\n"
        "// in +initialize for each subclass.\n"
        "+ (GPBDescriptor *)descriptor {\n"
        "  static GPBDescriptor *descriptor = nil;\n"
        "  if (!descriptor) {\n");

    bool has_oneofs = oneof_generators_.size();
    if (has_oneofs) {
      printer->Print(
          "    static GPBMessageOneofDescription oneofs[] = {\n");
      printer->Indent();
      printer->Indent();
      printer->Indent();
      for (vector<OneofGenerator*>::iterator iter = oneof_generators_.begin();
           iter != oneof_generators_.end(); ++iter) {
        (*iter)->GenerateDescription(printer);
      }
      printer->Outdent();
      printer->Outdent();
      printer->Outdent();
      printer->Print(
          "    };\n");
    }

    TextFormatDecodeData text_format_decode_data;
    bool has_fields = descriptor_->field_count() > 0;
    if (has_fields) {
      // TODO(thomasvl): The plugin's FieldGenerator::GenerateFieldDescription()
      // wraps the fieldOptions's value of this structure in an CPP gate so
      // they can be compiled away; but that still results in a const char* in
      // the structure for a NULL pointer for every message field.  If the
      // fieldOptions are moved to a separate payload like the TextFormat extra
      // data is, then it would shrink that static data shrinking the binaries
      // a little more.
      // TODO(thomasvl): proto3 syntax doens't need a defaultValue in the
      // structure because primitive types are always zero.  If we add a second
      // structure and a different initializer, we can avoid the wasted static
      // storage for every field in a proto3 message.
      printer->Print(
          "    static GPBMessageFieldDescription fields[] = {\n");
      printer->Indent();
      printer->Indent();
      printer->Indent();
      for (int i = 0; i < descriptor_->field_count(); ++i) {
        const FieldGenerator& field_generator =
            field_generators_.get(sorted_fields[i]);
        field_generator.GenerateFieldDescription(printer);
        if (field_generator.needs_textformat_name_support()) {
          text_format_decode_data.AddString(sorted_fields[i]->number(),
                                            field_generator.generated_objc_name(),
                                            field_generator.raw_field_name());
        }
      }
      printer->Outdent();
      printer->Outdent();
      printer->Outdent();
      printer->Print(
          "    };\n");
    }

    bool has_enums = enum_generators_.size();
    if (has_enums) {
      printer->Print(
          "    static GPBMessageEnumDescription enums[] = {\n");
      printer->Indent();
      printer->Indent();
      printer->Indent();
      for (vector<EnumGenerator*>::iterator iter = enum_generators_.begin();
           iter != enum_generators_.end(); ++iter) {
        printer->Print("{ .enumDescriptorFunc = $name$_EnumDescriptor },\n",
                       "name", (*iter)->name());
      }
      printer->Outdent();
      printer->Outdent();
      printer->Outdent();
      printer->Print(
          "    };\n");
    }

    bool has_extensions = sorted_extensions.size();
    if (has_extensions) {
      printer->Print(
          "    static GPBExtensionRange ranges[] = {\n");
      printer->Indent();
      printer->Indent();
      printer->Indent();
      for (int i = 0; i < sorted_extensions.size(); i++) {
        printer->Print("{ .start = $start$, .end = $end$ },\n",
                       "start", SimpleItoa(sorted_extensions[i]->start),
                       "end", SimpleItoa(sorted_extensions[i]->end));
      }
      printer->Outdent();
      printer->Outdent();
      printer->Outdent();
      printer->Print(
          "    };\n");
    }

    map<string, string> vars;
    vars["classname"] = class_name_;
    vars["rootclassname"] = root_classname_;
    vars["fields"] = has_fields ? "fields" : "NULL";
    vars["fields_count"] =
        has_fields ? "sizeof(fields) / sizeof(GPBMessageFieldDescription)" : "0";
    vars["oneofs"] = has_oneofs ? "oneofs" : "NULL";
    vars["oneof_count"] =
        has_oneofs ? "sizeof(oneofs) / sizeof(GPBMessageOneofDescription)" : "0";
    vars["enums"] = has_enums ? "enums" : "NULL";
    vars["enum_count"] =
        has_enums ? "sizeof(enums) / sizeof(GPBMessageEnumDescription)" : "0";
    vars["ranges"] = has_extensions ? "ranges" : "NULL";
    vars["range_count"] =
        has_extensions ? "sizeof(ranges) / sizeof(GPBExtensionRange)" : "0";
    vars["wireformat"] =
        descriptor_->options().message_set_wire_format() ? "YES" : "NO";

    if (text_format_decode_data.num_entries() == 0) {
      printer->Print(
          vars,
          "    GPBDescriptor *localDescriptor =\n"
          "        [GPBDescriptor allocDescriptorForClass:[$classname$ class]\n"
          "                                     rootClass:[$rootclassname$ class]\n"
          "                                          file:$rootclassname$_FileDescriptor()\n"
          "                                        fields:$fields$\n"
          "                                    fieldCount:$fields_count$\n"
          "                                        oneofs:$oneofs$\n"
          "                                    oneofCount:$oneof_count$\n"
          "                                         enums:$enums$\n"
          "                                     enumCount:$enum_count$\n"
          "                                        ranges:$ranges$\n"
          "                                    rangeCount:$range_count$\n"
          "                                   storageSize:sizeof($classname$__storage_)\n"
          "                                    wireFormat:$wireformat$];\n");
    } else {
      vars["extraTextFormatInfo"] = CEscape(text_format_decode_data.Data());
      printer->Print(
          vars,
          "#if GPBOBJC_SKIP_MESSAGE_TEXTFORMAT_EXTRAS\n"
          "    const char *extraTextFormatInfo = NULL;\n"
          "#else\n"
          "    static const char *extraTextFormatInfo = \"$extraTextFormatInfo$\";\n"
          "#endif  // GPBOBJC_SKIP_MESSAGE_TEXTFORMAT_EXTRAS\n"
          "    GPBDescriptor *localDescriptor =\n"
          "        [GPBDescriptor allocDescriptorForClass:[$classname$ class]\n"
          "                                     rootClass:[$rootclassname$ class]\n"
          "                                          file:$rootclassname$_FileDescriptor()\n"
          "                                        fields:$fields$\n"
          "                                    fieldCount:$fields_count$\n"
          "                                        oneofs:$oneofs$\n"
          "                                    oneofCount:$oneof_count$\n"
          "                                         enums:$enums$\n"
          "                                     enumCount:$enum_count$\n"
          "                                        ranges:$ranges$\n"
          "                                    rangeCount:$range_count$\n"
          "                                   storageSize:sizeof($classname$__storage_)\n"
          "                                    wireFormat:$wireformat$\n"
          "                           extraTextFormatInfo:extraTextFormatInfo];\n");
      }
      printer->Print(
          "    NSAssert(descriptor == nil, @\"Startup recursed!\");\n"
          "    descriptor = localDescriptor;\n"
          "  }\n"
          "  return descriptor;\n"
          "}\n\n"
          "@end\n\n");

    for (int i = 0; i < descriptor_->field_count(); i++) {
      field_generators_.get(descriptor_->field(i))
          .GenerateCFunctionImplementations(printer);
    }

    for (vector<OneofGenerator*>::iterator iter = oneof_generators_.begin();
         iter != oneof_generators_.end(); ++iter) {
      (*iter)->GenerateClearFunctionImplementation(printer);
    }
  }

  for (vector<EnumGenerator*>::iterator iter = enum_generators_.begin();
       iter != enum_generators_.end(); ++iter) {
    (*iter)->GenerateSource(printer);
  }

  for (vector<MessageGenerator*>::iterator iter =
           nested_message_generators_.begin();
       iter != nested_message_generators_.end(); ++iter) {
    (*iter)->GenerateSource(printer);
  }
}

}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
