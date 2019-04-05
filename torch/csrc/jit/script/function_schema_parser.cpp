#include <torch/csrc/jit/script/function_schema_parser.h>
#include <torch/csrc/jit/script/lexer.h>
#include <torch/csrc/jit/script/parse_string_literal.h>
#include <torch/csrc/jit/script/schema_type_parser.h>

#include <functional>
#include <memory>
#include <vector>

using c10::FunctionSchema;
using c10::Argument;
using c10::IValue;
using c10::ListType;
using at::TypeKind;

namespace torch {
namespace jit {

namespace script {
namespace {
struct SchemaParser {
  SchemaParser(const std::string& str)
      : L(str), type_parser(L, /*parse_complete_tensor_types*/ false) {}

  FunctionSchema parseDeclaration() {
    std::string name = L.expect(TK_IDENT).text();
    if (L.nextIf(':')) {
      L.expect(':');
      name = name + "::" + L.expect(TK_IDENT).text();
    }
    std::string overload_name = "";
    if (L.nextIf('.')) {
      overload_name = L.expect(TK_IDENT).text();
    }
    std::vector<Argument> arguments;
    std::vector<Argument> returns;
    bool kwarg_only = false;
    bool is_vararg = false;
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
    idx = 0;
    L.expect(TK_ARROW);
    if (L.cur().kind == '(') {
      parseList('(', ',', ')', [&] {
        returns.push_back(
            parseArgument(idx++, /*is_return=*/true, /*kwarg_only=*/false));
      });
    } else {
      returns.push_back(
          parseArgument(0, /*is_return=*/true, /*kwarg_only=*/false));
    }
    return FunctionSchema{
        std::move(name), std::move(overload_name), std::move(arguments), std::move(returns), is_vararg, false};
  }

  std::vector<FunctionSchema> parseDeclarations() {
    std::vector<FunctionSchema> results;
    do {
      results.push_back(parseDeclaration());
    } while (L.nextIf(TK_NEWLINE));
    L.expect(TK_EOF);
    return results;
  }

  Argument parseArgument(size_t idx, bool is_return, bool kwarg_only) {
    Argument result;
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
      N = std::stoll(L.expect(TK_NUMBER).text());
      L.expect(']');
      auto container = type_parser.parseAliasAnnotation();
      if (container && alias_info) {
        container->addContainedType(std::move(*alias_info));
      }
      alias_info = std::move(container);
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
        return parseStringLiteral(token.range, token.text());
      }
      case TK_IDENT: {
        auto tok = L.next();
        auto text = tok.text();
        if ("float" == text) {
          return static_cast<int64_t>(at::kFloat);
        } else if ("long" == text) {
          return static_cast<int64_t>(at::kLong);
        } else if ("strided" == text) {
          return static_cast<int64_t>(at::kStrided);
        } else if ("Mean" == text) {
          return static_cast<int64_t>(Reduction::Mean);
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
        if (kind == TypeKind::FloatType || n.find('.') != std::string::npos ||
            n.find('e') != std::string::npos) {
          return std::stod(n);
        } else {
          int64_t v = std::stoll(n);
          return v;
        }
    }
  }
  IValue convertToList(
      TypeKind kind,
      const SourceRange& range,
      std::vector<IValue> vs) {
    switch (kind) {
      case TypeKind::FloatType:
        return fmap(vs, [](IValue v) { return v.toDouble(); });
      case TypeKind::IntType:
        return fmap(vs, [](IValue v) { return v.toInt(); });
      case TypeKind::BoolType:
        return fmap(vs, [](IValue v) { return v.toBool(); });
      default:
        throw ErrorReport(range)
            << "lists are only supported for float or int types.";
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
    return convertToList(kind, tok.range, std::move(vs));
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
      case TypeKind::GeneratorType: {
        return parseTensorDefault(range);
      } break;
      case TypeKind::StringType:
      case TypeKind::OptionalType:
      case TypeKind::NumberType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
        return parseSingleConstant(arg_type->kind());
        break;
      case TypeKind::DeviceObjType: {
        auto device_text =
            parseStringLiteral(range, L.expect(TK_STRINGLITERAL).text());
        return c10::Device(device_text);
        break;
      }
      case TypeKind::ListType: {
        auto elem_kind = arg_type->cast<ListType>()->getElementType();
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
  SchemaTypeParser type_parser;
};
} // namespace
} // namespace script

FunctionSchema parseSchema(const std::string& schema) {
  return script::SchemaParser(schema).parseDeclarations().at(0);
}

} // namespace jit
} // namespace torch
