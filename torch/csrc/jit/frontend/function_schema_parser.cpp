#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <ATen/core/Reduction.h>
#include <c10/util/string_utils.h>
#include <c10/util/string_view.h>
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
  SchemaParser(const std::string& str)
      : L(str),
        strView(str),
        type_parser(L, /*parse_complete_tensor_types*/ false) {
    type_parser.setStringView(strView);
  }
  static std::shared_ptr<Source> newSource(c10::string_view v) {
    return std::make_shared<Source>(std::string(v.begin(), v.end()));
  }

  Token withSource(Token t) {
    auto result = std::move(t);
    result.range = SourceRange(
        newSource(strView), result.range.start(), result.range.end());
    return result;
  }

  // We can't use Token::text() because we've caused the lexer to not
  // set up Source for our tokens
  c10::string_view textForToken(const Token& t) {
    return strView.substr(t.range.start(), t.range.end() - t.range.start());
  }

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
        throw ErrorReport(withSource(L.cur()))
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
          throw ErrorReport(withSource(L.cur()))
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
          throw ErrorReport(withSource(L.cur()))
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
    auto name_view = textForToken(L.expect(TK_IDENT));
    std::string name(name_view.begin(), name_view.end());
    if (L.nextIf(':')) {
      L.expect(':');
      const auto ident = textForToken(L.expect(TK_IDENT));
      name.reserve(name.size() + 2 + ident.size());
      name.push_back(':');
      name.push_back(':');
      name.append(ident.begin(), ident.end());
    }
    std::string overload_name = "";
    if (L.nextIf('.')) {
      const auto text = textForToken(L.expect(TK_IDENT));
      overload_name.append(text.begin(), text.end());
    }
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

  Argument parseArgument(size_t idx, bool is_return, bool kwarg_only) {
    auto p = type_parser.parseType();
    auto type = std::move(p.first);
    auto alias_info = std::move(p.second);
    c10::optional<int32_t> N;
    c10::optional<IValue> default_value;
    c10::optional<std::string> alias_set;
    std::string name;
    if (L.nextIf('[')) {
      // note: an array with a size hint can only occur at the Argument level
      type = ListType::create(type);
      const auto num_view = textForToken(L.expect(TK_NUMBER));
      N = c10::stoll(std::string(num_view.begin(), num_view.end()));
      L.expect(']');
      auto container = type_parser.parseAliasAnnotation();
      if (container && alias_info) {
        container->addContainedType(std::move(*alias_info));
      }
      alias_info = std::move(container);
      if (L.nextIf('?')) {
        type = OptionalType::create(type);
      }
    }
    if (is_return) {
      // optionally field names in return values
      if (L.cur().kind == TK_IDENT) {
        const auto tok = textForToken(L.next());
        name.append(tok.begin(), tok.end());
      } else {
        name = "";
      }
    } else {
      const auto tok = textForToken(L.expect(TK_IDENT));
      name.append(tok.begin(), tok.end());
      if (L.nextIf('=')) {
        default_value = parseDefaultValue(type, N);
      }
    }
    return Argument(
        std::move(name),
        std::move(type),
        N,
        std::move(default_value),
        !is_return && kwarg_only,
        std::move(alias_info));
  }
  IValue parseSingleConstant(TypeKind kind) {
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
        return parseStringLiteral(token.range, textForToken(token));
      }
      case TK_IDENT: {
        auto tok = L.next();
        auto text = textForToken(tok);
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
          throw ErrorReport(withSource(L.cur()).range)
              << "invalid numeric default value";
        }
      }
      default:
        std::string n;
        if (L.nextIf('-')) {
          const auto num = textForToken(L.expect(TK_NUMBER));
          n.reserve(1 + num.size());
          n.push_back('-');
          n.append(num.begin(), num.end());
        } else {
          const auto num = textForToken(L.expect(TK_NUMBER));
          n.append(num.begin(), num.end());
        }

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
      default:
        // throw ErrorReport(SourceRange(newSource(strView), range.start(),
        // range.end()))
        throw ErrorReport(range)
            << "lists are only supported for float, int and complex types";
    }
  }
  IValue parseConstantList(TypeKind kind) {
    auto tok = L.expect('[');
    std::vector<IValue> vs;
    if (L.cur().kind != ']') {
      do {
        vs.push_back(parseSingleConstant(kind));
      } while (L.nextIf(','));
    }
    L.expect(']');
    return convertToList(kind, tok.range, vs);
  }

  IValue parseTensorDefault(const SourceRange& range) {
    L.expect(TK_NONE);
    return IValue();
  }
  IValue parseDefaultValue(
      const TypePtr& arg_type,
      c10::optional<int32_t> arg_N) {
    auto range = L.cur().range;
    switch (arg_type->kind()) {
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
        return parseSingleConstant(arg_type->kind());
        break;
      case TypeKind::DeviceObjType: {
        auto device_text =
            parseStringLiteral(range, textForToken(L.expect(TK_STRINGLITERAL)));
        return c10::Device(device_text);
        break;
      }
      case TypeKind::ListType: {
        auto elem_kind = arg_type->castRaw<ListType>()->getElementType();
        if (L.cur().kind == TK_IDENT) {
          return parseTensorDefault(range);
        } else if (arg_N && L.cur().kind != '[') {
          IValue v = parseSingleConstant(elem_kind->kind());
          std::vector<IValue> repeated(*arg_N, v);
          return convertToList(elem_kind->kind(), range, repeated);
        } else {
          return parseConstantList(elem_kind->kind());
        }
      } break;
      default:
        // throw ErrorReport(SourceRange(newSource(strView), range.start(),
        // range.end()))
        throw ErrorReport(range) << "unexpected type, file a bug report";
    }
    return IValue(); // silence warnings
  }

  void parseList(
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
  Lexer L;
  c10::string_view strView;
  SchemaTypeParser type_parser;
};
} // namespace

C10_EXPORT either<OperatorName, FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName) {
  return SchemaParser(schemaOrName).parseExactlyOneDeclaration();
}

C10_EXPORT FunctionSchema parseSchema(const std::string& schema) {
  auto parsed = parseSchemaOrName(schema);
  TORCH_CHECK(
      parsed.is_right(),
      "Tried to parse a function schema but only the operator name was given");
  return parsed.right();
}

C10_EXPORT OperatorName parseName(const std::string& name) {
  auto parsed = parseSchemaOrName(name);
  TORCH_CHECK(
      parsed.is_left(),
      "Tried to parse an operator name but function schema was given");
  return parsed.left();
}

} // namespace jit
} // namespace torch
