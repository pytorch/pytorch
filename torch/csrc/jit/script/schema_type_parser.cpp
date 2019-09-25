#include <ATen/core/alias_info.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/script/lexer.h>
#include <torch/csrc/jit/script/parse_string_literal.h>
#include <torch/csrc/jit/script/schema_type_parser.h>
#include <string>

using c10::AliasInfo;
using c10::BoolType;
using c10::CapsuleType;
using c10::DeviceObjType;
using c10::DictType;
using c10::FloatType;
using c10::FutureType;
using c10::GeneratorType;
using c10::IntType;
using c10::ListType;
using c10::NoneType;
using c10::NumberType;
using c10::OptionalType;
using c10::StringType;
using c10::Symbol;
using c10::TensorType;
using c10::TupleType;
using c10::VarType;

namespace torch {
namespace jit {
namespace script {

TypeAndAlias SchemaTypeParser::parseBaseType() {
  static std::unordered_map<std::string, TypePtr> type_map = {
      {"Generator", GeneratorType::get()},
      {"ScalarType", IntType::get()},
      {"Layout", IntType::get()},
      {"MemoryFormat", IntType::get()},
      {"QScheme", IntType::get()},
      {"Device", DeviceObjType::get()},
      {"Scalar", NumberType::get()},
      {"str", StringType::get()},
      {"float", FloatType::get()},
      {"int", IntType::get()},
      {"bool", BoolType::get()},
      {"None", NoneType::get()},
      {"Capsule", CapsuleType::get()},
  };
  auto tok = L.cur();
  if (!L.nextIf(TK_NONE)) {
    L.expect(TK_IDENT);
  }
  std::string text = tok.text();

  auto it = type_map.find(text);
  if (it == type_map.end()) {
    if (text.size() > 0 && islower(text[0])) {
      // lower case identifiers that are not otherwise valid types
      // are treated as type variables
      return TypeAndAlias(VarType::create(text), parseAliasAnnotation());
    }
    throw ErrorReport(tok.range) << "unknown type specifier";
  }
  return TypeAndAlias(it->second, c10::nullopt);
}

// Examples:
// Tensor(a) // Tensor is in set a
// Tensor(a!) // it is also written to
// Tensor!  // shorthand for Tensor(fresh_identifier!)
// Tensor(a! -> a|b) // Tensor is in set a, written to,
//                      and after the write is in set a AND b.
c10::optional<AliasInfo> SchemaTypeParser::parseAliasAnnotation() {
  std::set<Symbol> sets;
  AliasInfo alias_info;
  if (L.nextIf('(')) {
    // optional 'alias set annotation'
    parseList(TK_NOTHING, '|', TK_NOTHING, [&] {
      if (L.nextIf('*')) {
        alias_info.addBeforeSet(AliasInfo::wildcardSet());

        // If we found a wildcard, ignore all subsequent annotations
      } else if (!alias_info.isWildcardBefore()) {
        alias_info.addBeforeSet(
            Symbol::fromQualString("alias::" + L.expect(TK_IDENT).text()));
      }
    });
    if (L.nextIf('!')) {
      alias_info.setIsWrite(true);
    }
    if (L.nextIf(TK_ARROW)) {
      // optional 'alias set annotation'
      parseList(TK_NOTHING, '|', TK_NOTHING, [&] {
        if (L.nextIf('*')) {
          alias_info.addAfterSet(AliasInfo::wildcardSet());

          // If we found a wildcard, ignore all subsequent annotations
        } else if (!alias_info.isWildcardAfter()) {
          alias_info.addAfterSet(
              Symbol::fromQualString("alias::" + L.expect(TK_IDENT).text()));
        }
      });
    } else {
      // We didn't encounter an ->, so assume the "after set" is identical
      // to the "before set"
      AT_ASSERT(alias_info.afterSets().empty());
      for (const auto& set : alias_info.beforeSets()) {
        alias_info.addAfterSet(set);
      }
    }
    L.expect(')');
  } else if (L.nextIf('!')) {
    alias_info.addBeforeSet(
        Symbol::fromQualString("alias::$" + c10::guts::to_string(next_id++)));
    alias_info.setIsWrite(true);
  } else {
    return c10::nullopt;
  }

  return alias_info;
}

c10::optional<at::ScalarType> SchemaTypeParser::parseTensorDType(
    const std::string& dtype) {
#define DEFINE_SCALAR_TYPE(_1, n) {#n, at::ScalarType::n},

  static std::unordered_map<std::string, at::ScalarType> type_map = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

  auto type = type_map.find(dtype);
  if (type != type_map.end()) {
    return type->second;
  }
  return c10::nullopt;
}

TypePtr SchemaTypeParser::parseRefinedTensor() {
  auto maybe_dtype = parseTensorDType(L.expect(TK_IDENT).text());
  AT_ASSERT(maybe_dtype);
  at::ScalarType dtype = *maybe_dtype;
  TypePtr ptr;
  L.expect('(');
  TypePtr tensor_type;
  if (L.cur().kind == '*') {
    size_t num_dims = 0;
    parseList(TK_NOTHING, ',', ')', [&] {
      L.expect('*');
      num_dims++;
    });
    ptr = at::TensorType::create(
        dtype,
        at::DeviceType::CPU,
        c10::VaryingShape(num_dims),
        c10::VaryingShape(num_dims),
        c10::nullopt);
  } else {
    std::vector<int64_t> dims;
    parseList(TK_NOTHING, ',', ')', [&] {
      const std::string& num = L.expect(TK_NUMBER).text();
      std::string::size_type num_len;
      size_t dim = c10::stoi(num, &num_len);
      AT_ASSERTM(
          num_len == num.size(),
          "Bad tensor dimension size. Strides not yet supported in parsing",
          num);
      dims.push_back(dim);
    });
    at::IntArrayRef dims_ref(dims);
    ptr = at::TensorType::create(dtype, at::DeviceType::CPU, dims_ref, false);
  }
  return ptr;
}

std::pair<TypePtr, c10::optional<AliasInfo>> SchemaTypeParser::parseType() {
  TypePtr value;
  c10::optional<AliasInfo> alias_info;
  // Tuple type
  if (L.cur().kind == '(') {
    std::vector<TypePtr> types;
    parseList('(', ',', ')', [&] {
      auto r = parseType();
      types.push_back(std::move(r.first));
      if (alias_info && r.second) {
        alias_info->addContainedType(std::move(*r.second));
      }
    });
    value = TupleType::create(std::move(types));
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Future") {
    L.next(); // Future
    L.expect('(');
    auto p = parseType();
    auto subtype = std::move(p.first);
    auto subalias = std::move(p.second);
    L.expect(')');
    value = FutureType::create(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Tensor") {
    L.next();
    value = TensorType::get();
    alias_info = parseAliasAnnotation();
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Dict") {
    L.next();
    L.expect('(');
    auto key_type = parseType().first;
    L.expect(',');
    auto value_type = parseType().first;
    L.expect(')');
    alias_info = parseAliasAnnotation();
    value = DictType::create(key_type, value_type);
  } else if (
      complete_tensor_types && L.cur().kind == TK_IDENT &&
      parseTensorDType(L.cur().text())) {
    value = parseRefinedTensor();
    alias_info = parseAliasAnnotation();
  } else {
    auto value_alias = parseBaseType();
    value = value_alias.first;
    alias_info = value_alias.second;
  }
  while (true) {
    if (L.cur().kind == '[' && L.lookahead().kind == ']') {
      L.next(); // [
      L.next(); // ]
      value = ListType::create(value);
      auto container = parseAliasAnnotation();
      if (container && alias_info) {
        container->addContainedType(std::move(*alias_info));
      }
      alias_info = std::move(container);
    } else if (L.nextIf('?')) {
      value = OptionalType::create(value);
    } else {
      break;
    }
  }
  return std::make_pair(std::move(value), std::move(alias_info));
}

void SchemaTypeParser::parseList(
    int begin,
    int sep,
    int end,
    const std::function<void()>& callback) {
  auto r = L.cur().range;
  if (begin != TK_NOTHING)
    L.expect(begin);
  if (L.cur().kind != end) {
    do {
      callback();
    } while (L.nextIf(sep));
  }
  if (end != TK_NOTHING)
    L.expect(end);
}
} // namespace script
} // namespace jit
} // namespace torch
