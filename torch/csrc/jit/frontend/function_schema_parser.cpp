#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <ATen/core/Reduction.h>
#include <ATen/core/type_factory.h>
#include <c10/util/Optional.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/schema_type_parser.h>

#include <functional>
#include <memory>
#include <vector>

using at::TypeKind;
using c10::Argument;
using c10::either;
using c10::FunctionSchema;
using c10::IValue;
using c10::ListType;
using c10::make_left;
using c10::make_right;
using c10::OperatorName;
using c10::OptionalType;

namespace torch {
namespace jit {

namespace {
struct SchemaParser {
  explicit SchemaParser(const std::string& str)
      : L(std::make_shared<Source>(
            c10::string_view(str),
            c10::nullopt,
            0,
            nullptr,
            Source::DONT_COPY)),
        type_parser(L, /*parse_complete_tensor_types*/ false) {}

  either<OperatorName, FunctionSchema> parseDeclaration() {
    OperatorName name = parseName();

    // If there is no parentheses coming, then this is just the operator name
    // without an argument list
    if (L.cur().kind != '(') {
      return make_left<OperatorName, FunctionSchema>(std::move(name));
    }

    std::vector<Argument> arguments;
    std::vector<Argument> returns;
    bool kwarg_only = false;
    bool is_vararg = false;
    bool is_varret = false;
    size_t idx = 0;
    parseList('(', ',', ')', [&] {
      if (is_vararg)
        throw ErrorReport(L.cur())
            << "... must be the last element of the argument list";
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
          throw ErrorReport(L.cur())
              << "schemas with vararg (...) can't have default value args";
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
          throw ErrorReport(L.cur())
              << "... must be the last element of the return list";
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

    return make_right<OperatorName, FunctionSchema>(
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
      name = name + "::" + L.expect(TK_IDENT).text();
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
    return {name, overload_name};
  }

  std::vector<either<OperatorName, FunctionSchema>> parseDeclarations() {
    std::vector<either<OperatorName, FunctionSchema>> results;
    do {
      results.push_back(parseDeclaration());
    } while (L.nextIf(TK_NEWLINE));
    L.expect(TK_EOF);
    return results;
  }

  either<OperatorName, FunctionSchema> parseExactlyOneDeclaration() {
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
    c10::optional<int32_t> N;
    c10::optional<IValue> default_value;
    c10::optional<std::string> alias_set;
    std::string name;
    if (L.nextIf('[')) {
      // note: an array with a size hint can only occur at the Argument level
      fake_type = ListType::create(std::move(fake_type));
      real_type = ListType::create(std::move(real_type));
      N = c10::stoll(L.expect(TK_NUMBER).text());
      L.expect(']');
      auto container = type_parser.parseAliasAnnotation();
      if (alias_info) {
        if (!container) {
          container = c10::optional<at::AliasInfo>(at::AliasInfo());
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
        default_value = parseDefaultValue(*fake_type, fake_type->kind(), N);
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
  IValue parseSingleConstant(const c10::Type& type, TypeKind kind) {
    if (kind == c10::TypeKind::DynamicType) {
      return parseSingleConstant(
          type, type.expectRef<c10::DynamicType>().dynamicKind());
    }
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
        auto text = tok.text();
        if ("float" == text) {
          return static_cast<int64_t>(at::kFloat);
        } else if ("complex" == text) {
          return static_cast<int64_t>(at::kComplexFloat);
        } else if ("long" == text) {
          return static_cast<int64_t>(at::kLong);
        } else if ("strided" == text) {
          return static_cast<int64_t>(at::kStrided);
        } else if ("Mean" == text) {
          return static_cast<int64_t>(at::Reduction::Mean);
        } else if ("contiguous_format" == text) {
          return static_cast<int64_t>(c10::MemoryFormat::Contiguous);
        } else {
          throw ErrorReport(L.cur().range) << "invalid numeric default value";
        }
      }
      default:
        std::string n;
        if (L.nextIf('-'))
          n = "-" + L.expect(TK_NUMBER).text();
        else
          n = L.expect(TK_NUMBER).text();

        if (kind == TypeKind::ComplexType || n.find('j') != std::string::npos) {
          auto imag = c10::stod(n.substr(0, n.size() - 1));
          return c10::complex<double>(0, imag);
        } else if (
            kind == TypeKind::FloatType || n.find('.') != std::string::npos ||
            n.find('e') != std::string::npos) {
          return c10::stod(n);
        } else {
          int64_t v = c10::stoll(n);
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
        throw ErrorReport(range)
            << "lists are only supported for float, int and complex types";
    }
  }
  IValue parseConstantList(const c10::Type& type, TypeKind kind) {
    auto tok = L.expect('[');
    std::vector<IValue> vs;
    if (L.cur().kind != ']') {
      do {
        vs.push_back(parseSingleConstant(type, kind));
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
      c10::optional<int32_t> arg_N) {
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
        return parseSingleConstant(arg_type, kind);
        break;
      case TypeKind::DeviceObjType: {
        auto device_text =
            parseStringLiteral(range, L.expect(TK_STRINGLITERAL).text());
        return c10::Device(device_text);
        break;
      }
      case TypeKind::ListType: {
        auto elem_type = arg_type.containedType(0);
        if (L.cur().kind == TK_IDENT) {
          return parseTensorDefault(range);
        } else if (arg_N && L.cur().kind != '[') {
          IValue v = parseSingleConstant(*elem_type, elem_type->kind());
          std::vector<IValue> repeated(*arg_N, v);
          return convertToList(*elem_type, elem_type->kind(), range, repeated);
        } else {
          return parseConstantList(*elem_type, elem_type->kind());
        }
      } break;
      case TypeKind::DynamicType:
        return parseDefaultValue(
            arg_type,
            arg_type.expectRef<c10::DynamicType>().dynamicKind(),
            arg_N);
      default:
        throw ErrorReport(range) << "unexpected type, file a bug report";
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

either<OperatorName, FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName) {
  return SchemaParser(schemaOrName).parseExactlyOneDeclaration();
}

FunctionSchema parseSchema(const std::string& schema) {
  auto parsed = parseSchemaOrName(schema);
  TORCH_CHECK(
      parsed.is_right(),
      "Tried to parse a function schema but only the operator name was given");
  return std::move(parsed.right());
}

OperatorName parseName(const std::string& name) {
  auto parsed = parseSchemaOrName(name);
  TORCH_CHECK(
      parsed.is_left(),
      "Tried to parse an operator name but function schema was given");
  return std::move(parsed.left());
}

} // namespace jit
} // namespace torch
