#include <torch/csrc/jit/frontend/schema_type_parser.h>

#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <ATen/core/type_factory.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/custom_class.h>
#include <string>

using c10::AliasInfo;
using c10::AwaitType;
using c10::BoolType;
using c10::CapsuleType;
using c10::ComplexType;
using c10::DeviceObjType;
using c10::DictType;
using c10::FloatType;
using c10::FutureType;
using c10::GeneratorType;
using c10::IntType;
using c10::LayoutType;
using c10::ListType;
using c10::MemoryFormatType;
using c10::NoneType;
using c10::NumberType;
using c10::QSchemeType;
using c10::QuantizerType;
using c10::RRefType;
using c10::ScalarTypeType;
using c10::StorageType;
using c10::StreamObjType;
using c10::StringType;
using c10::Symbol;
using c10::SymIntType;
using c10::TensorType;
using c10::TupleType;
using c10::UnionType;
using c10::VarType;

namespace torch::jit {

TypePtr SchemaTypeParser::parseBaseType() {
  static std::unordered_map<std::string, TypePtr> type_map = {
      {"Generator", c10::TypeFactory::get<GeneratorType>()},
      {"Dimname", c10::TypeFactory::get<StringType>()},
      {"ScalarType", c10::TypeFactory::get<ScalarTypeType>()},
      {"Layout", c10::TypeFactory::get<LayoutType>()},
      {"MemoryFormat", c10::TypeFactory::get<MemoryFormatType>()},
      {"Storage", c10::TypeFactory::get<StorageType>()},
      {"QScheme", c10::TypeFactory::get<QSchemeType>()},
      {"Quantizer", c10::TypeFactory::get<QuantizerType>()},
      {"ConstQuantizerPtr",
       c10::TypeFactory::get<IntType>()}, // TODO This type should be removed
                                          // from the schema parser, it should
                                          // use the custom class mechanism
                                          // instead. @jerryzh
      {"Device", c10::TypeFactory::get<DeviceObjType>()},
      {"DeviceIndex", c10::TypeFactory::get<IntType>()},
      {"Stream", c10::TypeFactory::get<StreamObjType>()},
      {"Scalar", c10::TypeFactory::get<NumberType>()},
      {"str", c10::TypeFactory::get<StringType>()},
      {"float", c10::TypeFactory::get<FloatType>()},
      {"complex", c10::TypeFactory::get<ComplexType>()},
      {"int", c10::TypeFactory::get<IntType>()},
      {"SymInt", c10::TypeFactory::get<SymIntType>()},
      {"bool", c10::TypeFactory::get<BoolType>()},
      {"None", c10::TypeFactory::get<NoneType>()},
      {"NoneType", c10::TypeFactory::get<NoneType>()},
      {"Capsule", c10::TypeFactory::get<CapsuleType>()},
      {"Any", c10::TypeFactory::get<c10::AnyType>()},
      {"AnyClassType", c10::TypeFactory::get<c10::AnyClassType>()},
      {"AnyEnumType", c10::TypeFactory::get<c10::AnyEnumType>()},
  };
  auto tok = L.cur();
  if (!L.nextIf(TK_NONE) && !L.nextIf(TK_NONE_TYPE)) {
    L.expect(TK_IDENT);
  }
  std::string text = tok.text();

  auto it = type_map.find(text);
  if (it == type_map.end()) {
    if (allow_typevars_ && !text.empty() && islower(text[0])) {
      // lower case identifiers that are not otherwise valid types
      // are treated as type variables
      return c10::TypeFactory::createNamed<VarType>(text);
    }
    if (text == "double") {
      throw ErrorReport(tok.range)
          << "Use `float` instead of `double` in an operator's schema string. "
             "`float` in schema corresponds to the double type in C++";
    }
    if (text == "int64_t") {
      throw ErrorReport(tok.range)
          << "Use `SymInt` or `int` instead of `int64_t` in an operator's schema string. "
             "`SymInt` corresponds to c10::SymInt in C++ while `int` in schema corresponds "
             "to the int64_t type in C++.";
    }
    throw ErrorReport(tok.range)
        << "unknown type specifier. Common valid schema types include "
           "Tensor, SymInt, int, float, bool, Scalar; "
           "for a full list, please see "
           "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func ";
  }
  return it->second;
}

// Examples:
// Tensor(a) // Tensor is in set a
// Tensor(a!) // it is also written to
// Tensor!  // shorthand for Tensor(fresh_identifier!)
// Tensor(a! -> a|b) // Tensor is in set a, written to,
//                      and after the write is in set a AND b.
std::optional<AliasInfo> SchemaTypeParser::parseAliasAnnotation() {
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
        Symbol::fromQualString("alias::$" + std::to_string(next_id++)));
    alias_info.setIsWrite(true);
  } else {
    return c10::nullopt;
  }

  return alias_info;
}

std::optional<at::ScalarType> SchemaTypeParser::parseTensorDType(
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

std::optional<c10::Device> SchemaTypeParser::tryToParseDeviceType() {
  L.expect('=');
  const std::string& dev = L.expect(TK_IDENT).text();

  if (dev == "cpu") {
    return c10::Device(at::kCPU);
  }

  if (dev == "cuda" || dev == "hpu") {
    c10::DeviceIndex device_idx = -1;
    if (L.cur().kind == ':') {
      L.expect(':');
      const std::string& num = L.expect(TK_NUMBER).text();
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      std::string::size_type num_len;
      try {
        device_idx = std::stoi(num, &num_len);
      } catch (const std::invalid_argument& e) {
        throw ErrorReport(L.cur())
            << "Device index cannot be converted to integer";
      } catch (const std::out_of_range& e) {
        throw ErrorReport(L.cur()) << "Device index is too long";
      }
    }
    if (dev == "cuda") {
      return c10::Device(at::kCUDA, device_idx);
    } else {
      return c10::Device(at::kHPU, device_idx);
    }
  }

  throw ErrorReport(L.cur()) << "cannot parse device type '" << dev << "'\n";
}

std::optional<bool> SchemaTypeParser::tryToParseRequiresGrad() {
  L.expect('=');
  const std::string& num = L.expect(TK_NUMBER).text();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::string::size_type num_len;

  try {
    return (bool)std::stoi(num, &num_len);
  } catch (const std::invalid_argument& e) {
    throw ErrorReport(L.cur())
        << "Field requires_grad cannot be converted to integer";
  } catch (const std::out_of_range& e) {
    throw ErrorReport(L.cur()) << "Field requires_grad is too long";
  }
}

TypePtr SchemaTypeParser::parseRefinedTensor() {
  auto maybe_dtype = parseTensorDType(L.expect(TK_IDENT).text());
  AT_ASSERT(maybe_dtype);
  at::ScalarType dtype = *maybe_dtype;
  TypePtr ptr;
  L.expect('(');
  TypePtr tensor_type;
  std::optional<c10::Device> device;
  std::optional<bool> requires_grad;
  // Parse a type with either no ranks, known ranks with sizes, ranks with
  // unknown sizes, a mix of ranks with known and unknown sizes, or ranks with
  // known sizes and strides. The type might also have requires_grad and/or
  // device option. Examples of types we're handling here:
  //   Long(10, 8, 6, strides=[48, 6, 1], requires_grad=0, device=cuda:1)
  //   Float(10, *, 20, device=cuda:1)
  //   Float(requires_grad=1)
  std::vector<std::optional<int64_t>> dims;
  bool seen_strides = false;
  std::vector<int64_t> strides;
  parseList(TK_NOTHING, ',', ')', [&] {
    // Extra handling for options like 'device' and 'requires_grad'
    if (L.cur().kind == TK_IDENT && L.cur().text() != "SS") {
      const std::string& field = L.expect(TK_IDENT).text();
      if (field == "device") {
        auto parsed_device = tryToParseDeviceType();
        if (parsed_device.has_value()) {
          if (device.has_value()) {
            throw ErrorReport(L.cur()) << "'device' is specified twice";
          }
          device = parsed_device;
        }
        return;
      }
      if (field == "requires_grad") {
        auto parsed_requires_grad = tryToParseRequiresGrad();
        if (parsed_requires_grad.has_value()) {
          if (requires_grad.has_value()) {
            throw ErrorReport(L.cur()) << "'requires_grad' is specified twice";
          }
          requires_grad = parsed_requires_grad;
        }
        return;
      }
      if (field == "strides") {
        seen_strides = true;
        L.expect('=');
        parseList('[', ',', ']', [&] {
          const std::string& num = L.expect(TK_NUMBER).text();
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          std::string::size_type num_len;
          try {
            auto stride = std::stoll(num, &num_len);
            strides.push_back(stride);
          } catch (const std::invalid_argument& e) {
            throw ErrorReport(L.cur())
                << "The stride value cannot be converted to int";
          } catch (const std::out_of_range& e) {
            throw ErrorReport(L.cur()) << "The stride is too big";
          }
        });
        return;
      }
      throw ErrorReport(L.cur()) << "Unexpected specifier '" << field << "'";
    }
    if (device.has_value() || requires_grad.has_value()) {
      throw ErrorReport(L.cur())
          << "'device' and 'requires_grad' should come after dimensions in the type specification";
    }

    // Parsing ranks, supports mix of sized and unsized ranks, or, just strided
    // ranks
    if (L.cur().kind == '*') {
      dims.emplace_back(c10::nullopt);
      L.next();
      if (L.cur().kind == ':') {
        throw ErrorReport(L.cur()) << "Strides for unsized ranks not supported";
      }
      return;
    }
    bool shape_symbol = false;
    if (L.cur().kind == TK_IDENT && L.cur().text() == "SS") {
      L.next();
      L.expect('(');
      L.expect('-');
      shape_symbol = true;
    }
    const std::string& num = L.expect(TK_NUMBER).text();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::string::size_type num_len;
    int64_t dim = 0;
    try {
      dim = std::stoll(num, &num_len);
    } catch (const std::invalid_argument& e) {
      throw ErrorReport(L.cur()) << "The number can't be converted to int";
    } catch (const std::out_of_range& e) {
      throw ErrorReport(L.cur()) << "Number is too big";
    }
    if (shape_symbol) {
      L.expect(')');
      dim = -dim;
    }
    dims.emplace_back(dim);
  });
  if (seen_strides) {
    at::IntArrayRef strides_ref(strides);
    if (strides.size() != dims.size()) {
      // note: mixing unsized ranks and ranks with strides will always trigger
      // this
      throw ErrorReport(L.cur())
          << "Strides info is specified for some but not for all dimensions";
    }
    ptr = at::TensorType::create(
        dtype,
        device,
        c10::VaryingShape<int64_t>(dims),
        c10::VaryingShape<int64_t>(strides),
        requires_grad);
  } else {
    ptr = at::TensorType::create(
        dtype,
        device,
        c10::VaryingShape<int64_t>(dims),
        c10::VaryingShape<int64_t>(dims.size()),
        requires_grad);
  }
  return ptr;
}

std::pair<TypePtr, std::optional<AliasInfo>> SchemaTypeParser::parseType() {
  auto r = parseFakeAndRealType();
  return std::make_pair(std::move(std::get<0>(r)), std::move(std::get<2>(r)));
}

std::tuple</*fake*/ TypePtr, /*real*/ TypePtr, std::optional<AliasInfo>>
SchemaTypeParser::parseFakeAndRealType() {
  TypePtr fake_value;
  TypePtr real_value;
  std::optional<AliasInfo> alias_info;
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
    fake_value = real_value =
        c10::TypeFactory::create<TupleType>(std::move(types));
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Future") {
    L.next(); // Future
    L.expect('(');
    auto p = parseType();
    auto subtype = std::move(p.first);
    auto subalias = std::move(p.second);
    L.expect(')');
    fake_value = real_value = c10::TypeFactory::create<FutureType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Await") {
    L.next(); // Await
    L.expect('(');
    auto p = parseType();
    auto subtype = std::move(p.first);
    auto subalias = std::move(p.second);
    L.expect(')');
    fake_value = real_value = c10::TypeFactory::create<AwaitType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "RRef") {
    L.next(); // RRef
    L.expect('(');
    auto p = parseType();
    auto subtype = std::move(p.first);
    auto subalias = std::move(p.second);
    L.expect(')');
    fake_value = real_value = c10::TypeFactory::create<RRefType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Tensor") {
    L.next();
    fake_value = real_value = c10::TypeFactory::get<TensorType>();
    alias_info = parseAliasAnnotation();
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Dict") {
    L.next();
    L.expect('(');
    auto key_type = parseType().first;
    L.expect(',');
    auto value_type = parseType().first;
    L.expect(')');
    alias_info = parseAliasAnnotation();
    fake_value = real_value =
        c10::TypeFactory::create<DictType>(key_type, value_type);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Union") {
    L.next();
    L.expect('(');
    std::vector<TypePtr> types;
    types.emplace_back(parseType().first);
    while (L.cur().kind != ')') {
      L.expect(',');
      types.emplace_back(parseType().first);
    }
    L.expect(')');
    alias_info = parseAliasAnnotation();
    fake_value = real_value =
        c10::TypeFactory::create<c10::UnionType>(std::move(types));
  } else if (
      complete_tensor_types && L.cur().kind == TK_IDENT &&
      parseTensorDType(L.cur().text())) {
    fake_value = real_value = parseRefinedTensor();
    alias_info = parseAliasAnnotation();
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "__torch__") {
    L.next();
    L.expect('.');
    auto torch_tok = L.expect(TK_IDENT);
    if (torch_tok.text() != "torch") {
      throw ErrorReport(torch_tok.range)
          << "Expected classes namespace but got " << torch_tok.text();
    }
    L.expect('.');
    auto classes_tok = L.expect(TK_IDENT);
    if (classes_tok.text() != "classes") {
      throw ErrorReport(classes_tok.range)
          << "Expected classes namespace but got " << classes_tok.text();
    }
    L.expect('.');
    auto ns_tok = L.expect(TK_IDENT);
    L.expect('.');
    auto class_tok = L.expect(TK_IDENT);
    fake_value = real_value = getCustomClass(
        std::string("__torch__.torch.classes.") + ns_tok.text() + "." +
        class_tok.text());
    if (!fake_value) {
      throw ErrorReport(class_tok.range)
          << "Unknown custom class type "
          << ns_tok.text() + "." + class_tok.text()
          << ". Please ensure it is registered.";
    }
  } else {
    real_value = parseBaseType();
    if (real_value->kind() == ScalarTypeType::Kind ||
        real_value->kind() == MemoryFormatType::Kind ||
        real_value->kind() == LayoutType::Kind ||
        real_value->kind() == SymIntType::Kind) {
      fake_value = c10::TypeFactory::get<IntType>();
    } else {
      fake_value = real_value;
    }
    alias_info = parseAliasAnnotation();
  }
  while (true) {
    if (L.cur().kind == '[' && L.lookahead().kind == ']') {
      L.next(); // [
      L.next(); // ]
      fake_value = c10::TypeFactory::create<ListType>(fake_value);
      real_value = c10::TypeFactory::create<ListType>(real_value);
      auto container = parseAliasAnnotation();
      if (alias_info) {
        if (!container) {
          container = std::optional<AliasInfo>(AliasInfo());
          container->setIsWrite(alias_info->isWrite());
        }
        container->addContainedType(std::move(*alias_info));
      }
      alias_info = std::move(container);
    } else if (L.nextIf('?')) {
      fake_value = c10::OptionalType::get(fake_value);
      real_value = c10::OptionalType::get(real_value);
    } else {
      break;
    }
  }
  return std::make_tuple(
      std::move(fake_value), std::move(real_value), std::move(alias_info));
}

void SchemaTypeParser::parseList(
    int begin,
    int sep,
    int end,
    c10::function_ref<void()> callback) {
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

} // namespace torch::jit
