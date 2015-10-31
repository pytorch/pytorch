// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
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

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <map>
#include <math.h>
#include <string>

#include <google/protobuf/compiler/javanano/javanano_primitive_field.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/compiler/javanano/javanano_helpers.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/wire_format.h>
#include <google/protobuf/stubs/strutil.h>
#include <google/protobuf/stubs/substitute.h>

namespace google {
namespace protobuf {
namespace compiler {
namespace javanano {

using internal::WireFormat;
using internal::WireFormatLite;

namespace {

bool IsReferenceType(JavaType type) {
  switch (type) {
    case JAVATYPE_INT    : return false;
    case JAVATYPE_LONG   : return false;
    case JAVATYPE_FLOAT  : return false;
    case JAVATYPE_DOUBLE : return false;
    case JAVATYPE_BOOLEAN: return false;
    case JAVATYPE_STRING : return true;
    case JAVATYPE_BYTES  : return true;
    case JAVATYPE_ENUM   : return false;
    case JAVATYPE_MESSAGE: return true;

    // No default because we want the compiler to complain if any new
    // JavaTypes are added.
  }

  GOOGLE_LOG(FATAL) << "Can't get here.";
  return false;
}

bool IsArrayType(JavaType type) {
  switch (type) {
    case JAVATYPE_INT    : return false;
    case JAVATYPE_LONG   : return false;
    case JAVATYPE_FLOAT  : return false;
    case JAVATYPE_DOUBLE : return false;
    case JAVATYPE_BOOLEAN: return false;
    case JAVATYPE_STRING : return false;
    case JAVATYPE_BYTES  : return true;
    case JAVATYPE_ENUM   : return false;
    case JAVATYPE_MESSAGE: return false;

    // No default because we want the compiler to complain if any new
    // JavaTypes are added.
  }

  GOOGLE_LOG(FATAL) << "Can't get here.";
  return false;
}

const char* GetCapitalizedType(const FieldDescriptor* field) {
  switch (field->type()) {
    case FieldDescriptor::TYPE_INT32   : return "Int32"   ;
    case FieldDescriptor::TYPE_UINT32  : return "UInt32"  ;
    case FieldDescriptor::TYPE_SINT32  : return "SInt32"  ;
    case FieldDescriptor::TYPE_FIXED32 : return "Fixed32" ;
    case FieldDescriptor::TYPE_SFIXED32: return "SFixed32";
    case FieldDescriptor::TYPE_INT64   : return "Int64"   ;
    case FieldDescriptor::TYPE_UINT64  : return "UInt64"  ;
    case FieldDescriptor::TYPE_SINT64  : return "SInt64"  ;
    case FieldDescriptor::TYPE_FIXED64 : return "Fixed64" ;
    case FieldDescriptor::TYPE_SFIXED64: return "SFixed64";
    case FieldDescriptor::TYPE_FLOAT   : return "Float"   ;
    case FieldDescriptor::TYPE_DOUBLE  : return "Double"  ;
    case FieldDescriptor::TYPE_BOOL    : return "Bool"    ;
    case FieldDescriptor::TYPE_STRING  : return "String"  ;
    case FieldDescriptor::TYPE_BYTES   : return "Bytes"   ;
    case FieldDescriptor::TYPE_ENUM    : return "Enum"    ;
    case FieldDescriptor::TYPE_GROUP   : return "Group"   ;
    case FieldDescriptor::TYPE_MESSAGE : return "Message" ;

    // No default because we want the compiler to complain if any new
    // types are added.
  }

  GOOGLE_LOG(FATAL) << "Can't get here.";
  return NULL;
}

// For encodings with fixed sizes, returns that size in bytes.  Otherwise
// returns -1.
int FixedSize(FieldDescriptor::Type type) {
  switch (type) {
    case FieldDescriptor::TYPE_INT32   : return -1;
    case FieldDescriptor::TYPE_INT64   : return -1;
    case FieldDescriptor::TYPE_UINT32  : return -1;
    case FieldDescriptor::TYPE_UINT64  : return -1;
    case FieldDescriptor::TYPE_SINT32  : return -1;
    case FieldDescriptor::TYPE_SINT64  : return -1;
    case FieldDescriptor::TYPE_FIXED32 : return WireFormatLite::kFixed32Size;
    case FieldDescriptor::TYPE_FIXED64 : return WireFormatLite::kFixed64Size;
    case FieldDescriptor::TYPE_SFIXED32: return WireFormatLite::kSFixed32Size;
    case FieldDescriptor::TYPE_SFIXED64: return WireFormatLite::kSFixed64Size;
    case FieldDescriptor::TYPE_FLOAT   : return WireFormatLite::kFloatSize;
    case FieldDescriptor::TYPE_DOUBLE  : return WireFormatLite::kDoubleSize;

    case FieldDescriptor::TYPE_BOOL    : return WireFormatLite::kBoolSize;
    case FieldDescriptor::TYPE_ENUM    : return -1;

    case FieldDescriptor::TYPE_STRING  : return -1;
    case FieldDescriptor::TYPE_BYTES   : return -1;
    case FieldDescriptor::TYPE_GROUP   : return -1;
    case FieldDescriptor::TYPE_MESSAGE : return -1;

    // No default because we want the compiler to complain if any new
    // types are added.
  }
  GOOGLE_LOG(FATAL) << "Can't get here.";
  return -1;
}

bool AllAscii(const string& text) {
  for (int i = 0; i < text.size(); i++) {
    if ((text[i] & 0x80) != 0) {
      return false;
    }
  }
  return true;
}


void SetPrimitiveVariables(const FieldDescriptor* descriptor, const Params params,
                           map<string, string>* variables) {
  (*variables)["name"] =
    RenameJavaKeywords(UnderscoresToCamelCase(descriptor));
  (*variables)["capitalized_name"] =
    RenameJavaKeywords(UnderscoresToCapitalizedCamelCase(descriptor));
  (*variables)["number"] = SimpleItoa(descriptor->number());
  if (params.use_reference_types_for_primitives()
      && !descriptor->is_repeated()) {
    (*variables)["type"] = BoxedPrimitiveTypeName(GetJavaType(descriptor));
  } else {
    (*variables)["type"] = PrimitiveTypeName(GetJavaType(descriptor));
  }
  // Deals with defaults. For C++-string types (string and bytes),
  // we might need to have the generated code do the unicode decoding
  // (see comments in InternalNano.java for gory details.). We would
  // like to do this once into a static field and re-use that from
  // then on.
  if (descriptor->cpp_type() == FieldDescriptor::CPPTYPE_STRING &&
      !descriptor->default_value_string().empty() &&
      !params.use_reference_types_for_primitives()) {
    if (descriptor->type() == FieldDescriptor::TYPE_BYTES) {
      (*variables)["default"] = DefaultValue(params, descriptor);
      (*variables)["default_constant"] = FieldDefaultConstantName(descriptor);
      (*variables)["default_constant_value"] = strings::Substitute(
          "com.google.protobuf.nano.InternalNano.bytesDefaultValue(\"$0\")",
          CEscape(descriptor->default_value_string()));
      (*variables)["default_copy_if_needed"] =
          (*variables)["default"] + ".clone()";
    } else if (AllAscii(descriptor->default_value_string())) {
      // All chars are ASCII.  In this case directly referencing a
      // CEscape()'d string literal works fine.
      (*variables)["default"] =
          "\"" + CEscape(descriptor->default_value_string()) + "\"";
      (*variables)["default_copy_if_needed"] = (*variables)["default"];
    } else {
      // Strings where some chars are non-ASCII. We need to save the
      // default value.
      (*variables)["default"] = DefaultValue(params, descriptor);
      (*variables)["default_constant"] = FieldDefaultConstantName(descriptor);
      (*variables)["default_constant_value"] = strings::Substitute(
          "com.google.protobuf.nano.InternalNano.stringDefaultValue(\"$0\")",
          CEscape(descriptor->default_value_string()));
      (*variables)["default_copy_if_needed"] = (*variables)["default"];
    }
  } else {
    // Non-string, non-bytes field. Defaults are literals.
    (*variables)["default"] = DefaultValue(params, descriptor);
    (*variables)["default_copy_if_needed"] = (*variables)["default"];
  }
  (*variables)["boxed_type"] = BoxedPrimitiveTypeName(GetJavaType(descriptor));
  (*variables)["capitalized_type"] = GetCapitalizedType(descriptor);
  (*variables)["tag"] = SimpleItoa(WireFormat::MakeTag(descriptor));
  (*variables)["tag_size"] = SimpleItoa(
      WireFormat::TagSize(descriptor->number(), descriptor->type()));
  (*variables)["non_packed_tag"] = SimpleItoa(
      internal::WireFormatLite::MakeTag(descriptor->number(),
          internal::WireFormat::WireTypeForFieldType(descriptor->type())));
  int fixed_size = FixedSize(descriptor->type());
  if (fixed_size != -1) {
    (*variables)["fixed_size"] = SimpleItoa(fixed_size);
  }
  (*variables)["message_name"] = descriptor->containing_type()->name();
  (*variables)["empty_array_name"] = EmptyArrayName(params, descriptor);
}
}  // namespace

// ===================================================================

PrimitiveFieldGenerator::
PrimitiveFieldGenerator(const FieldDescriptor* descriptor, const Params& params)
  : FieldGenerator(params), descriptor_(descriptor) {
  SetPrimitiveVariables(descriptor, params, &variables_);
}

PrimitiveFieldGenerator::~PrimitiveFieldGenerator() {}

bool PrimitiveFieldGenerator::SavedDefaultNeeded() const {
  return variables_.find("default_constant") != variables_.end();
}

void PrimitiveFieldGenerator::GenerateInitSavedDefaultCode(io::Printer* printer) const {
  if (variables_.find("default_constant") != variables_.end()) {
    printer->Print(variables_,
      "$default_constant$ = $default_constant_value$;\n");
  }
}

void PrimitiveFieldGenerator::
GenerateMembers(io::Printer* printer, bool lazy_init) const {
  if (variables_.find("default_constant") != variables_.end()) {
    // Those primitive types that need a saved default.
    if (lazy_init) {
      printer->Print(variables_,
        "private static $type$ $default_constant$;\n");
    } else {
      printer->Print(variables_,
        "private static final $type$ $default_constant$ =\n"
        "    $default_constant_value$;\n");
    }
  }

  printer->Print(variables_,
    "public $type$ $name$;\n");

  if (params_.generate_has()) {
    printer->Print(variables_,
      "public boolean has$capitalized_name$;\n");
  }
}

void PrimitiveFieldGenerator::
GenerateClearCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$ = $default_copy_if_needed$;\n");

  if (params_.generate_has()) {
    printer->Print(variables_,
      "has$capitalized_name$ = false;\n");
  }
}

void PrimitiveFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "this.$name$ = input.read$capitalized_type$();\n");

  if (params_.generate_has()) {
    printer->Print(variables_,
      "has$capitalized_name$ = true;\n");
  }
}

void PrimitiveFieldGenerator::
GenerateSerializationConditional(io::Printer* printer) const {
  if (params_.use_reference_types_for_primitives()) {
    // For reference type mode, serialize based on equality
    // to null.
    printer->Print(variables_,
      "if (this.$name$ != null) {\n");
    return;
  }
  if (params_.generate_has()) {
    printer->Print(variables_,
      "if (has$capitalized_name$ || ");
  } else {
    printer->Print(variables_,
      "if (");
  }
  JavaType java_type = GetJavaType(descriptor_);
  if (IsArrayType(java_type)) {
    printer->Print(variables_,
      "!java.util.Arrays.equals(this.$name$, $default$)) {\n");
  } else if (IsReferenceType(java_type)) {
    printer->Print(variables_,
      "!this.$name$.equals($default$)) {\n");
  } else if (java_type == JAVATYPE_FLOAT) {
    printer->Print(variables_,
      "java.lang.Float.floatToIntBits(this.$name$)\n"
      "    != java.lang.Float.floatToIntBits($default$)) {\n");
  } else if (java_type == JAVATYPE_DOUBLE) {
    printer->Print(variables_,
      "java.lang.Double.doubleToLongBits(this.$name$)\n"
      "    != java.lang.Double.doubleToLongBits($default$)) {\n");
  } else {
    printer->Print(variables_,
      "this.$name$ != $default$) {\n");
  }
}

void PrimitiveFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  if (descriptor_->is_required() && !params_.generate_has()) {
    // Always serialize a required field if we don't have the 'has' signal.
    printer->Print(variables_,
      "output.write$capitalized_type$($number$, this.$name$);\n");
  } else {
    GenerateSerializationConditional(printer);
    printer->Print(variables_,
      "  output.write$capitalized_type$($number$, this.$name$);\n"
      "}\n");
  }
}

void PrimitiveFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  if (descriptor_->is_required() && !params_.generate_has()) {
    printer->Print(variables_,
      "size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
      "    .compute$capitalized_type$Size($number$, this.$name$);\n");
  } else {
    GenerateSerializationConditional(printer);
    printer->Print(variables_,
      "  size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
      "      .compute$capitalized_type$Size($number$, this.$name$);\n"
      "}\n");
  }
}

void RepeatedPrimitiveFieldGenerator::
GenerateFixClonedCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null && this.$name$.length > 0) {\n"
    "  cloned.$name$ = this.$name$.clone();\n"
    "}\n");
}

void PrimitiveFieldGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  // We define equality as serialized form equality. If generate_has(),
  // then if the field value equals the default value in both messages,
  // but one's 'has' field is set and the other's is not, the serialized
  // forms are different and we should return false.
  JavaType java_type = GetJavaType(descriptor_);
  if (java_type == JAVATYPE_BYTES) {
    printer->Print(variables_,
      "if (!java.util.Arrays.equals(this.$name$, other.$name$)");
    if (params_.generate_has()) {
      printer->Print(variables_,
        "\n"
        "    || (java.util.Arrays.equals(this.$name$, $default$)\n"
        "        && this.has$capitalized_name$ != other.has$capitalized_name$)");
    }
    printer->Print(") {\n"
      "  return false;\n"
      "}\n");
  } else if (java_type == JAVATYPE_STRING
      || params_.use_reference_types_for_primitives()) {
    printer->Print(variables_,
      "if (this.$name$ == null) {\n"
      "  if (other.$name$ != null) {\n"
      "    return false;\n"
      "  }\n"
      "} else if (!this.$name$.equals(other.$name$)");
    if (params_.generate_has()) {
      printer->Print(variables_,
        "\n"
        "    || (this.$name$.equals($default$)\n"
        "        && this.has$capitalized_name$ != other.has$capitalized_name$)");
    }
    printer->Print(") {\n"
      "  return false;\n"
      "}\n");
  } else if (java_type == JAVATYPE_FLOAT) {
    printer->Print(variables_,
      "{\n"
      "  int bits = java.lang.Float.floatToIntBits(this.$name$);\n"
      "  if (bits != java.lang.Float.floatToIntBits(other.$name$)");
    if (params_.generate_has()) {
      printer->Print(variables_,
        "\n"
        "      || (bits == java.lang.Float.floatToIntBits($default$)\n"
        "          && this.has$capitalized_name$ != other.has$capitalized_name$)");
    }
    printer->Print(") {\n"
      "    return false;\n"
      "  }\n"
      "}\n");
  } else if (java_type == JAVATYPE_DOUBLE) {
    printer->Print(variables_,
      "{\n"
      "  long bits = java.lang.Double.doubleToLongBits(this.$name$);\n"
      "  if (bits != java.lang.Double.doubleToLongBits(other.$name$)");
    if (params_.generate_has()) {
      printer->Print(variables_,
        "\n"
        "      || (bits == java.lang.Double.doubleToLongBits($default$)\n"
        "          && this.has$capitalized_name$ != other.has$capitalized_name$)");
    }
    printer->Print(") {\n"
      "    return false;\n"
      "  }\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "if (this.$name$ != other.$name$");
    if (params_.generate_has()) {
      printer->Print(variables_,
        "\n"
        "    || (this.$name$ == $default$\n"
        "        && this.has$capitalized_name$ != other.has$capitalized_name$)");
    }
    printer->Print(") {\n"
      "  return false;\n"
      "}\n");
  }
}

void PrimitiveFieldGenerator::
GenerateHashCodeCode(io::Printer* printer) const {
  JavaType java_type = GetJavaType(descriptor_);
  if (java_type == JAVATYPE_BYTES) {
    printer->Print(variables_,
      "result = 31 * result + java.util.Arrays.hashCode(this.$name$);\n");
  } else if (java_type == JAVATYPE_STRING
      || params_.use_reference_types_for_primitives()) {
    printer->Print(variables_,
      "result = 31 * result\n"
      "    + (this.$name$ == null ? 0 : this.$name$.hashCode());\n");
  } else {
    switch (java_type) {
      // For all Java primitive types below, the hash codes match the
      // results of BoxedType.valueOf(primitiveValue).hashCode().
      case JAVATYPE_INT:
        printer->Print(variables_,
          "result = 31 * result + this.$name$;\n");
        break;
      case JAVATYPE_LONG:
        printer->Print(variables_,
          "result = 31 * result\n"
          "    + (int) (this.$name$ ^ (this.$name$ >>> 32));\n");
        break;
      case JAVATYPE_FLOAT:
        printer->Print(variables_,
          "result = 31 * result\n"
          "    + java.lang.Float.floatToIntBits(this.$name$);\n");
        break;
      case JAVATYPE_DOUBLE:
        printer->Print(variables_,
          "{\n"
          "  long v = java.lang.Double.doubleToLongBits(this.$name$);\n"
          "  result = 31 * result + (int) (v ^ (v >>> 32));\n"
          "}\n");
        break;
      case JAVATYPE_BOOLEAN:
        printer->Print(variables_,
          "result = 31 * result + (this.$name$ ? 1231 : 1237);\n");
        break;
      default:
        GOOGLE_LOG(ERROR) << "unknown java type for primitive field";
        break;
    }
  }
}

// ===================================================================

AccessorPrimitiveFieldGenerator::
AccessorPrimitiveFieldGenerator(const FieldDescriptor* descriptor,
     const Params& params, int has_bit_index)
  : FieldGenerator(params), descriptor_(descriptor) {
  SetPrimitiveVariables(descriptor, params, &variables_);
  SetBitOperationVariables("has", has_bit_index, &variables_);
}

AccessorPrimitiveFieldGenerator::~AccessorPrimitiveFieldGenerator() {}

bool AccessorPrimitiveFieldGenerator::SavedDefaultNeeded() const {
  return variables_.find("default_constant") != variables_.end();
}

void AccessorPrimitiveFieldGenerator::
GenerateInitSavedDefaultCode(io::Printer* printer) const {
  if (variables_.find("default_constant") != variables_.end()) {
    printer->Print(variables_,
      "$default_constant$ = $default_constant_value$;\n");
  }
}

void AccessorPrimitiveFieldGenerator::
GenerateMembers(io::Printer* printer, bool lazy_init) const {
  if (variables_.find("default_constant") != variables_.end()) {
    // Those primitive types that need a saved default.
    if (lazy_init) {
      printer->Print(variables_,
        "private static $type$ $default_constant$;\n");
    } else {
      printer->Print(variables_,
        "private static final $type$ $default_constant$ =\n"
        "    $default_constant_value$;\n");
    }
  }
  printer->Print(variables_,
    "private $type$ $name$_;\n"
    "public $type$ get$capitalized_name$() {\n"
    "  return $name$_;\n"
    "}\n"
    "public $message_name$ set$capitalized_name$($type$ value) {\n");
  if (IsReferenceType(GetJavaType(descriptor_))) {
    printer->Print(variables_,
      "  if (value == null) {\n"
      "    throw new java.lang.NullPointerException();\n"
      "  }\n");
  }
  printer->Print(variables_,
    "  $name$_ = value;\n"
    "  $set_has$;\n"
    "  return this;\n"
    "}\n"
    "public boolean has$capitalized_name$() {\n"
    "  return $get_has$;\n"
    "}\n"
    "public $message_name$ clear$capitalized_name$() {\n"
    "  $name$_ = $default_copy_if_needed$;\n"
    "  $clear_has$;\n"
    "  return this;\n"
    "}\n");
}

void AccessorPrimitiveFieldGenerator::
GenerateClearCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_ = $default_copy_if_needed$;\n");
}

void AccessorPrimitiveFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$_ = input.read$capitalized_type$();\n"
    "$set_has$;\n");
}

void AccessorPrimitiveFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($get_has$) {\n"
    "  output.write$capitalized_type$($number$, $name$_);\n"
    "}\n");
}

void AccessorPrimitiveFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if ($get_has$) {\n"
    "  size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
    "      .compute$capitalized_type$Size($number$, $name$_);\n"
    "}\n");
}

void AccessorPrimitiveFieldGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  switch (GetJavaType(descriptor_)) {
    // For all Java primitive types below, the equality checks match the
    // results of BoxedType.valueOf(primitiveValue).equals(otherValue).
    case JAVATYPE_FLOAT:
      printer->Print(variables_,
        "if ($different_has$\n"
        "    || java.lang.Float.floatToIntBits($name$_)\n"
        "        != java.lang.Float.floatToIntBits(other.$name$_)) {\n"
        "  return false;\n"
        "}\n");
      break;
    case JAVATYPE_DOUBLE:
      printer->Print(variables_,
        "if ($different_has$\n"
        "    || java.lang.Double.doubleToLongBits($name$_)\n"
        "        != java.lang.Double.doubleToLongBits(other.$name$_)) {\n"
        "  return false;\n"
        "}\n");
      break;
    case JAVATYPE_INT:
    case JAVATYPE_LONG:
    case JAVATYPE_BOOLEAN:
      printer->Print(variables_,
        "if ($different_has$\n"
        "    || $name$_ != other.$name$_) {\n"
        "  return false;\n"
        "}\n");
      break;
    case JAVATYPE_STRING:
      // Accessor style would guarantee $name$_ non-null
      printer->Print(variables_,
        "if ($different_has$\n"
        "    || !$name$_.equals(other.$name$_)) {\n"
        "  return false;\n"
        "}\n");
      break;
    case JAVATYPE_BYTES:
      // Accessor style would guarantee $name$_ non-null
      printer->Print(variables_,
        "if ($different_has$\n"
        "    || !java.util.Arrays.equals($name$_, other.$name$_)) {\n"
        "  return false;\n"
        "}\n");
      break;
    default:
      GOOGLE_LOG(ERROR) << "unknown java type for primitive field";
      break;
  }
}

void AccessorPrimitiveFieldGenerator::
GenerateHashCodeCode(io::Printer* printer) const {
  switch (GetJavaType(descriptor_)) {
    // For all Java primitive types below, the hash codes match the
    // results of BoxedType.valueOf(primitiveValue).hashCode().
    case JAVATYPE_INT:
      printer->Print(variables_,
        "result = 31 * result + $name$_;\n");
      break;
    case JAVATYPE_LONG:
      printer->Print(variables_,
        "result = 31 * result + (int) ($name$_ ^ ($name$_ >>> 32));\n");
      break;
    case JAVATYPE_FLOAT:
      printer->Print(variables_,
        "result = 31 * result +\n"
        "    java.lang.Float.floatToIntBits($name$_);\n");
      break;
    case JAVATYPE_DOUBLE:
      printer->Print(variables_,
        "{\n"
        "  long v = java.lang.Double.doubleToLongBits($name$_);\n"
        "  result = 31 * result + (int) (v ^ (v >>> 32));\n"
        "}\n");
      break;
    case JAVATYPE_BOOLEAN:
      printer->Print(variables_,
        "result = 31 * result + ($name$_ ? 1231 : 1237);\n");
      break;
    case JAVATYPE_STRING:
      // Accessor style would guarantee $name$_ non-null
      printer->Print(variables_,
        "result = 31 * result + $name$_.hashCode();\n");
      break;
    case JAVATYPE_BYTES:
      // Accessor style would guarantee $name$_ non-null
      printer->Print(variables_,
        "result = 31 * result + java.util.Arrays.hashCode($name$_);\n");
      break;
    default:
      GOOGLE_LOG(ERROR) << "unknown java type for primitive field";
      break;
  }
}

// ===================================================================

PrimitiveOneofFieldGenerator::PrimitiveOneofFieldGenerator(
    const FieldDescriptor* descriptor, const Params& params)
  : FieldGenerator(params), descriptor_(descriptor) {
    SetPrimitiveVariables(descriptor, params, &variables_);
    SetCommonOneofVariables(descriptor, &variables_);
}

PrimitiveOneofFieldGenerator::~PrimitiveOneofFieldGenerator() {}

void PrimitiveOneofFieldGenerator::GenerateMembers(
    io::Printer* printer, bool /*unused lazy_init*/) const {
  printer->Print(variables_,
    "public boolean has$capitalized_name$() {\n"
    "  return $has_oneof_case$;\n"
    "}\n"
    "public $type$ get$capitalized_name$() {\n"
    "  if ($has_oneof_case$) {\n"
    "    return ($type$) ($boxed_type$) this.$oneof_name$_;\n"
    "  }\n"
    "  return $default$;\n"
    "}\n"
    "public $message_name$ set$capitalized_name$($type$ value) {\n"
    "  $set_oneof_case$;\n"
    "  this.$oneof_name$_ = value;\n"
    "  return this;\n"
    "}\n");
}

void PrimitiveOneofFieldGenerator::GenerateClearCode(
    io::Printer* printer) const {
  // No clear method for oneof fields.
}

void PrimitiveOneofFieldGenerator::GenerateMergingCode(
    io::Printer* printer) const {
  printer->Print(variables_,
    "this.$oneof_name$_ = input.read$capitalized_type$();\n"
    "$set_oneof_case$;\n");
}

void PrimitiveOneofFieldGenerator::GenerateSerializationCode(
    io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case$) {\n"
    "  output.write$capitalized_type$(\n"
    "      $number$, ($boxed_type$) this.$oneof_name$_);\n"
    "}\n");
}

void PrimitiveOneofFieldGenerator::GenerateSerializedSizeCode(
    io::Printer* printer) const {
  printer->Print(variables_,
    "if ($has_oneof_case$) {\n"
    "  size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
    "      .compute$capitalized_type$Size(\n"
    "          $number$, ($boxed_type$) this.$oneof_name$_);\n"
    "}\n");
}

void PrimitiveOneofFieldGenerator::GenerateEqualsCode(
    io::Printer* printer) const {
  GenerateOneofFieldEquals(descriptor_, variables_, printer);
}

void PrimitiveOneofFieldGenerator::GenerateHashCodeCode(
    io::Printer* printer) const {
  GenerateOneofFieldHashCode(descriptor_, variables_, printer);
}

// ===================================================================

RepeatedPrimitiveFieldGenerator::RepeatedPrimitiveFieldGenerator(
    const FieldDescriptor* descriptor, const Params& params)
  : FieldGenerator(params), descriptor_(descriptor) {
  SetPrimitiveVariables(descriptor, params, &variables_);
}

RepeatedPrimitiveFieldGenerator::~RepeatedPrimitiveFieldGenerator() {}

void RepeatedPrimitiveFieldGenerator::
GenerateMembers(io::Printer* printer, bool /*unused init_defaults*/) const {
  printer->Print(variables_,
    "public $type$[] $name$;\n");
}

void RepeatedPrimitiveFieldGenerator::
GenerateClearCode(io::Printer* printer) const {
  printer->Print(variables_,
    "$name$ = $default$;\n");
}

void RepeatedPrimitiveFieldGenerator::
GenerateMergingCode(io::Printer* printer) const {
  // First, figure out the length of the array, then parse.
  printer->Print(variables_,
    "int arrayLength = com.google.protobuf.nano.WireFormatNano\n"
    "    .getRepeatedFieldArrayLength(input, $non_packed_tag$);\n"
    "int i = this.$name$ == null ? 0 : this.$name$.length;\n");

  if (GetJavaType(descriptor_) == JAVATYPE_BYTES) {
    printer->Print(variables_,
      "byte[][] newArray = new byte[i + arrayLength][];\n");
  } else {
    printer->Print(variables_,
      "$type$[] newArray = new $type$[i + arrayLength];\n");
  }
  printer->Print(variables_,
    "if (i != 0) {\n"
    "  java.lang.System.arraycopy(this.$name$, 0, newArray, 0, i);\n"
    "}\n"
    "for (; i < newArray.length - 1; i++) {\n"
    "  newArray[i] = input.read$capitalized_type$();\n"
    "  input.readTag();\n"
    "}\n"
    "// Last one without readTag.\n"
    "newArray[i] = input.read$capitalized_type$();\n"
    "this.$name$ = newArray;\n");
}

void RepeatedPrimitiveFieldGenerator::
GenerateMergingCodeFromPacked(io::Printer* printer) const {
  printer->Print(
    "int length = input.readRawVarint32();\n"
    "int limit = input.pushLimit(length);\n");

  // If we know the elements will all be of the same size, the arrayLength
  // can be calculated much more easily. However, FixedSize() returns 1 for
  // repeated bool fields, which are guaranteed to have the fixed size of
  // 1 byte per value only if we control the output. On the wire they can
  // legally appear as variable-size integers, so we need to use the slow
  // way for repeated bool fields.
  if (descriptor_->type() == FieldDescriptor::TYPE_BOOL
      || FixedSize(descriptor_->type()) == -1) {
    printer->Print(variables_,
      "// First pass to compute array length.\n"
      "int arrayLength = 0;\n"
      "int startPos = input.getPosition();\n"
      "while (input.getBytesUntilLimit() > 0) {\n"
      "  input.read$capitalized_type$();\n"
      "  arrayLength++;\n"
      "}\n"
      "input.rewindToPosition(startPos);\n");
  } else {
    printer->Print(variables_,
      "int arrayLength = length / $fixed_size$;\n");
  }

  printer->Print(variables_,
    "int i = this.$name$ == null ? 0 : this.$name$.length;\n"
    "$type$[] newArray = new $type$[i + arrayLength];\n"
    "if (i != 0) {\n"
    "  java.lang.System.arraycopy(this.$name$, 0, newArray, 0, i);\n"
    "}\n"
    "for (; i < newArray.length; i++) {\n"
    "  newArray[i] = input.read$capitalized_type$();\n"
    "}\n"
    "this.$name$ = newArray;\n"
    "input.popLimit(limit);\n");
}

void RepeatedPrimitiveFieldGenerator::
GenerateRepeatedDataSizeCode(io::Printer* printer) const {
  // Creates a variable dataSize and puts the serialized size in there.
  // If the element type is a Java reference type, also generates
  // dataCount which stores the number of non-null elements in the field.
  if (IsReferenceType(GetJavaType(descriptor_))) {
    printer->Print(variables_,
      "int dataCount = 0;\n"
      "int dataSize = 0;\n"
      "for (int i = 0; i < this.$name$.length; i++) {\n"
      "  $type$ element = this.$name$[i];\n"
      "  if (element != null) {\n"
      "    dataCount++;\n"
      "    dataSize += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
      "        .compute$capitalized_type$SizeNoTag(element);\n"
      "  }\n"
      "}\n");
  } else if (FixedSize(descriptor_->type()) == -1) {
    printer->Print(variables_,
      "int dataSize = 0;\n"
      "for (int i = 0; i < this.$name$.length; i++) {\n"
      "  $type$ element = this.$name$[i];\n"
      "  dataSize += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
      "      .compute$capitalized_type$SizeNoTag(element);\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "int dataSize = $fixed_size$ * this.$name$.length;\n");
  }
}

void RepeatedPrimitiveFieldGenerator::
GenerateSerializationCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null && this.$name$.length > 0) {\n");
  printer->Indent();

  if (descriptor_->is_packable() && descriptor_->options().packed()) {
    GenerateRepeatedDataSizeCode(printer);
    printer->Print(variables_,
      "output.writeRawVarint32($tag$);\n"
      "output.writeRawVarint32(dataSize);\n"
      "for (int i = 0; i < this.$name$.length; i++) {\n"
      "  output.write$capitalized_type$NoTag(this.$name$[i]);\n"
      "}\n");
  } else if (IsReferenceType(GetJavaType(descriptor_))) {
    printer->Print(variables_,
      "for (int i = 0; i < this.$name$.length; i++) {\n"
      "  $type$ element = this.$name$[i];\n"
      "  if (element != null) {\n"
      "    output.write$capitalized_type$($number$, element);\n"
      "  }\n"
      "}\n");
  } else {
    printer->Print(variables_,
      "for (int i = 0; i < this.$name$.length; i++) {\n"
      "  output.write$capitalized_type$($number$, this.$name$[i]);\n"
      "}\n");
  }

  printer->Outdent();
  printer->Print("}\n");
}

void RepeatedPrimitiveFieldGenerator::
GenerateSerializedSizeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (this.$name$ != null && this.$name$.length > 0) {\n");
  printer->Indent();

  GenerateRepeatedDataSizeCode(printer);

  printer->Print(
    "size += dataSize;\n");
  if (descriptor_->is_packable() && descriptor_->options().packed()) {
    printer->Print(variables_,
      "size += $tag_size$;\n"
      "size += com.google.protobuf.nano.CodedOutputByteBufferNano\n"
      "    .computeRawVarint32Size(dataSize);\n");
  } else if (IsReferenceType(GetJavaType(descriptor_))) {
    printer->Print(variables_,
      "size += $tag_size$ * dataCount;\n");
  } else {
    printer->Print(variables_,
      "size += $tag_size$ * this.$name$.length;\n");
  }

  printer->Outdent();

  printer->Print(
    "}\n");
}

void RepeatedPrimitiveFieldGenerator::
GenerateEqualsCode(io::Printer* printer) const {
  printer->Print(variables_,
    "if (!com.google.protobuf.nano.InternalNano.equals(\n"
    "    this.$name$, other.$name$)) {\n"
    "  return false;\n"
    "}\n");
}

void RepeatedPrimitiveFieldGenerator::
GenerateHashCodeCode(io::Printer* printer) const {
  printer->Print(variables_,
    "result = 31 * result\n"
    "    + com.google.protobuf.nano.InternalNano.hashCode(this.$name$);\n");
}

}  // namespace javanano
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
