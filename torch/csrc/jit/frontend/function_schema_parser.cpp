#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <ATen/core/Reduction.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>
#include <fmt/format.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/schema_type_parser.h>
#include <optional>

#include <memory>
#include <vector>

using at::TypeKind;
using c10::Argument;
using c10::FunctionSchema;
using c10::IValue;
using c10::ListType;
using c10::OperatorName;

namespace torch::jit {

namespace {
struct SchemaParser {
  explicit SchemaParser(const std::string& str, bool allow_typevars)
      : L(std::make_shared<Source>(
            std::string_view(str),
            std::nullopt,
            0,
            nullptr,
            Source::DONT_COPY)),
        type_parser(L, /*parse_complete_tensor_types*/ false, allow_typevars) {}

  std::variant<OperatorName, FunctionSchema> parseDeclaration() {
    OperatorName name = parseName();

    // If there is no parentheses coming, then this is just the operator name
    // without an argument list
    if (L.cur().kind != '(') {
      return OperatorName(std::move(name));
    }

    std::vector<Argument> arguments;
    std::vector<Argument> returns;
    bool kwarg_only = false;
    bool is_vararg = false;
    bool is_varret = false;
    size_t idx = 0;
    parseList('(', ',', ')', [&] {
      if (is_vararg)
        throw(
            ErrorReport(L.cur())
            << "... must be the last element of the argument list");
      if (L.nextIf('*')) {
        kwarg_only = true;
      } else if (L.nextIf(TK_DOTS)) {
        is_vararg = true;
      } else {
        arguments.push_back(parseArgument(
            idx++, /*is_return=*/false, /*kwarg_only=*/kwarg_only));
      }
    });

    // check if all arguments are not-default for vararg schemas
    if (is_vararg) {
      for (const auto& arg : arguments) {
        if (arg.default_value().has_value()) {
          throw(
              ErrorReport(L.cur())
              << "schemas with vararg (...) can't have default value args");
        }
      }
    }

    idx = 0;
    L.expect(TK_ARROW);
    if (L.nextIf(TK_DOTS)) {
      is_varret = true;
    } else if (L.cur().kind == '(') {
      parseList('(', ',', ')', [&] {
        if (is_varret) {
          throw(
              ErrorReport(L.cur())
              << "... must be the last element of the return list");
        }
        if (L.nextIf(TK_DOTS)) {
          is_varret = true;
        } else {
          returns.push_back(
              parseArgument(idx++, /*is_return=*/true, /*kwarg_only=*/false));
        }
      });
    } else {
      returns.push_back(
          parseArgument(0, /*is_return=*/true, /*kwarg_only=*/false));
    }

    return FunctionSchema(
        std::move(name.name),
        std::move(name.overload_name),
        std::move(arguments),
        std::move(returns),
        is_vararg,
        is_varret);
  }

  c10::OperatorName parseName() {
    std::string name = L.expect(TK_IDENT).text();
    if (L.nextIf(':')) {
      L.expect(':');
      name = fmt::format("{}::{}", name, L.expect(TK_IDENT).text_view());
    }
    std::string overload_name = "";
    if (L.nextIf('.')) {
      overload_name = L.expect(TK_IDENT).text();
    }
    // default is used as an attribute on the `OpOverloadPacket`
    // (obtained using `torch.ops.aten.foo`) to get the operator
    // overload with overload name as an empty string
    // and so shouldn't be used as an overload name
    // also disallow dunder attribute names to be overload names
    bool is_a_valid_overload_name =
        !((overload_name == "default") || (overload_name.rfind("__", 0) == 0));
    TORCH_CHECK(
        is_a_valid_overload_name,
        overload_name,
        " is not a legal overload name for aten operators");
    return {std::move(name), std::move(overload_name)};
  }

  std::vector<std::variant<OperatorName, FunctionSchema>> parseDeclarations() {
    std::vector<std::variant<OperatorName, FunctionSchema>> results;
    do {
      results.emplace_back(parseDeclaration());
    } while (L.nextIf(TK_NEWLINE));
    L.expect(TK_EOF);
    return results;
  }

  std::variant<OperatorName, FunctionSchema> parseExactlyOneDeclaration() {
    auto result = parseDeclaration();
    L.nextIf(TK_NEWLINE);
    L.expect(TK_EOF);
    return result;
  }

  Argument parseArgument(size_t /*idx*/, bool is_return, bool kwarg_only) {
    // fake and real type coincide except for Layout/MemoryFormat/ScalarType
    // the fake type for these is Int instead
    auto p = type_parser.parseFakeAndRealType();
    auto fake_type = std::move(std::get<0>(p));
    auto real_type = std::move(std::get<1>(p));
    auto alias_info = std::move(std::get<2>(p));
    std::optional<int32_t> N;
    std::optional<IValue> default_value;
    std::optional<std::string> alias_set;
    std::string name;
    if (L.nextIf('[')) {
      // note: an array with a size hint can only occur at the Argument level
      fake_type = c10::TypeFactory::create<ListType>(std::move(fake_type));
      real_type = c10::TypeFactory::create<ListType>(std::move(real_type));
      N = std::stoll(L.expect(TK_NUMBER).text());
      L.expect(']');
      auto container = type_parser.parseAliasAnnotation();
      if (alias_info) {
        if (!container) {
          container = std::optional<at::AliasInfo>(at::AliasInfo());
          container->setIsWrite(alias_info->isWrite());
        }
        container->addContainedType(std::move(*alias_info));
      }
      alias_info = std::move(container);
      if (L.nextIf('?')) {
        fake_type =
            c10::TypeFactory::create<c10::OptionalType>(std::move(fake_type));
        real_type =
            c10::TypeFactory::create<c10::OptionalType>(std::move(real_type));
      }
    }
    if (is_return) {
      // optionally field names in return values
      if (L.cur().kind == TK_IDENT) {
        name = L.next().text();
      } else {
        name = "";
      }
    } else {
      name = L.expect(TK_IDENT).text();
      if (L.nextIf('=')) {
        // NB: this means we have to unswizzle default too
        default_value =
            parseDefaultValue(*fake_type, fake_type->kind(), *real_type, N);
      }
    }
    return Argument(
        std::move(name),
        std::move(fake_type),
        std::move(real_type),
        N,
        std::move(default_value),
        !is_return && kwarg_only,
        std::move(alias_info));
  }

  bool isPossiblyOptionalScalarType(const c10::Type& type) {
    if (type.kind() == at::ScalarTypeType::Kind) {
      return true;
    }
    if (type.kind() == at::OptionalType::Kind) {
      for (const auto& inner : type.containedTypes()) {
        if (isPossiblyOptionalScalarType(*inner))
          return true;
      }
    }
    return false;
  }

  IValue parseSingleConstant(
      const c10::Type& type,
      TypeKind kind,
      const c10::Type& real_type) {
    if (kind == c10::TypeKind::DynamicType) {
      return parseSingleConstant(
          type, type.expectRef<c10::DynamicType>().dynamicKind(), real_type);
    }
    const auto& str2dtype = c10::getStringToDtypeMap();
    switch (L.cur().kind) {
      case TK_TRUE:
        L.next();
        return true;
      case TK_FALSE:
        L.next();
        return false;
      case TK_NONE:
        L.next();
        return IValue();
      case TK_STRINGLITERAL: {
        auto token = L.next();
        return parseStringLiteral(token.range, token.text());
      }
      case TK_IDENT: {
        auto tok = L.next();
        auto text_view = tok.text_view();
        // NB: float/complex/long are here for BC purposes. Other dtypes
        // are handled via str2dtype.
        // Please don't add more cases to this if-else block.
        if ("float" == text_view) {
          return static_cast<int64_t>(at::kFloat);
        } else if ("complex" == text_view) {
          return static_cast<int64_t>(at::kComplexFloat);
        } else if ("long" == text_view) {
          return static_cast<int64_t>(at::kLong);
        } else if ("strided" == text_view) {
          return static_cast<int64_t>(at::kStrided);
        } else if ("Mean" == text_view) {
          return static_cast<int64_t>(at::Reduction::Mean);
        } else if ("contiguous_format" == text_view) {
          return static_cast<int64_t>(c10::MemoryFormat::Contiguous);
        } else {
          auto text = tok.text();
          if (isPossiblyOptionalScalarType(real_type) &&
              str2dtype.count(text) > 0) {
            return static_cast<int64_t>(str2dtype.at(text));
          } else {
            throw(
                ErrorReport(L.cur().range) << "invalid numeric default value");
          }
        }
      }
      default:
        std::string n;
        if (L.nextIf('-'))
          n = "-" + L.expect(TK_NUMBER).text();
        else
          n = L.expect(TK_NUMBER).text();

        if (kind == TypeKind::ComplexType || n.find('j') != std::string::npos) {
          auto imag = std::stod(n.substr(0, n.size() - 1));
          return c10::complex<double>(0, imag);
        } else if (
            kind == TypeKind::FloatType || n.find('.') != std::string::npos ||
            n.find('e') != std::string::npos) {
          return std::stod(n);
        } else {
          int64_t v = std::stoll(n);
          return v;
        }
    }
  }
  IValue convertToList(
      const c10::Type& type,
      TypeKind kind,
      const SourceRange& range,
      const std::vector<IValue>& vs) {
    switch (kind) {
      case TypeKind::ComplexType:
        return fmap(vs, [](const IValue& v) { return v.toComplexDouble(); });
      case TypeKind::FloatType:
        return fmap(vs, [](const IValue& v) { return v.toDouble(); });
      case TypeKind::IntType:
        return fmap(vs, [](const IValue& v) { return v.toInt(); });
      case TypeKind::BoolType:
        return fmap(vs, [](const IValue& v) { return v.toBool(); });
      case TypeKind::DynamicType:
        return convertToList(
            type, type.expectRef<c10::DynamicType>().dynamicKind(), range, vs);
      default:
        throw(
            ErrorReport(range)
            << "lists are only supported for float, int and complex types");
    }
  }
  IValue parseConstantList(
      const c10::Type& type,
      TypeKind kind,
      const c10::Type& real_type) {
    auto tok = L.expect('[');
    std::vector<IValue> vs;
    if (L.cur().kind != ']') {
      do {
        vs.push_back(parseSingleConstant(type, kind, real_type));
      } while (L.nextIf(','));
    }
    L.expect(']');
    return convertToList(type, kind, tok.range, vs);
  }

  IValue parseTensorDefault(const SourceRange& /*range*/) {
    L.expect(TK_NONE);
    return IValue();
  }
  IValue parseDefaultValue(
      const c10::Type& arg_type,
      TypeKind kind,
      const c10::Type& real_type,
      std::optional<int32_t> arg_N) {
    auto range = L.cur().range;
    switch (kind) {
      case TypeKind::TensorType:
      case TypeKind::GeneratorType:
      case TypeKind::QuantizerType: {
        return parseTensorDefault(range);
      } break;
      case TypeKind::StringType:
      case TypeKind::OptionalType:
      case TypeKind::NumberType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::ComplexType:
        return parseSingleConstant(arg_type, kind, real_type);
        break;
      case TypeKind::DeviceObjType: {
        auto device_text =
            parseStringLiteral(range, L.expect(TK_STRINGLITERAL).text());
        return c10::Device(device_text);
        break;
      }
      case TypeKind::ListType: {
        auto elem_type = arg_type.containedType(0);
        auto real_elem_type = real_type.containedType(0);
        if (L.cur().kind == TK_IDENT) {
          return parseTensorDefault(range);
        } else if (arg_N && L.cur().kind != '[') {
          IValue v = parseSingleConstant(
              *elem_type, elem_type->kind(), *real_elem_type);
          std::vector<IValue> repeated(*arg_N, v);
          return convertToList(*elem_type, elem_type->kind(), range, repeated);
        } else {
          return parseConstantList(
              *elem_type, elem_type->kind(), *real_elem_type);
        }
      } break;
      case TypeKind::DynamicType:
        return parseDefaultValue(
            arg_type,
            arg_type.expectRef<c10::DynamicType>().dynamicKind(),
            real_type,
            arg_N);
      default:
        throw(ErrorReport(range) << "unexpected type, file a bug report");
    }
    return IValue(); // silence warnings
  }

  void parseList(
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
  Lexer L;
  SchemaTypeParser type_parser;
};
} // namespace

std::variant<OperatorName, FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName,
    bool allow_typevars) {
  // We're ignoring aten and prim for BC reasons
  if (schemaOrName.rfind("aten::", 0) == 0 ||
      schemaOrName.rfind("prim::", 0) == 0) {
    allow_typevars = true;
  }
  return SchemaParser(schemaOrName, allow_typevars)
      .parseExactlyOneDeclaration();
}

FunctionSchema parseSchema(const std::string& schema, bool allow_typevars) {
  auto parsed = parseSchemaOrName(schema, allow_typevars);
  TORCH_CHECK(
      std::holds_alternative<FunctionSchema>(parsed),
      "Tried to parse a function schema but only the operator name was given");
  return std::get<FunctionSchema>(std::move(parsed));
}

OperatorName parseName(const std::string& name) {
  auto parsed = parseSchemaOrName(name);
  TORCH_CHECK(
      std::holds_alternative<OperatorName>(parsed),
      "Tried to parse an operator name but function schema was given");
  return std::get<OperatorName>(std::move(parsed));
}

} // namespace torch::jit
