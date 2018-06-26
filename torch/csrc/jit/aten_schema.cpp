#include "ATen/ATen.h"
#include "torch/csrc/jit/script/lexer.h"
#include "torch/csrc/jit/script/tree.h"
#include "torch/csrc/jit/aten_schema.h"
#include "torch/csrc/jit/tensor_conversions.h"
#include "torch/csrc/jit/script/error_report.h"

namespace torch { namespace jit {

namespace script {
struct SchemaParser {
  SchemaParser(const std::string& str)
  : L(str) {}

  FunctionSchema parseDeclaration() {
    auto name = L.expect(TK_IDENT).text();
    std::vector<Argument> arguments;
    std::vector<Argument> returns;
    kwarg_only = false;
    parseList('(', ',', ')', arguments, &SchemaParser::parseArgument);
    L.expect(TK_ARROW);
    if(L.cur().kind == '(') {
      parseList('(', ',', ')', returns, &SchemaParser::parseReturn);
    } else {
      parseReturn(returns);
    }
    return FunctionSchema { name, arguments, returns };
  }

  std::vector<FunctionSchema> parseDeclarations() {
    std::vector<FunctionSchema> results;
    do {
      results.push_back(parseDeclaration());
    } while(L.nextIf(TK_NEWLINE));
    L.expect(TK_EOF);
    return results;
  }

  TreeRef parseIdent() {
    return String::create(L.expect(TK_IDENT).text());
  }
  TypePtr parseBaseType() {
    static std::unordered_map<std::string, TypePtr> type_map = {
      {"Tensor", DynamicType::get() },
      {"Generator", DynamicType::get() },
      {"ScalarType", IntType::get() },
      {"Layout", IntType::get() },
      {"Device", ListType::ofInts() },
      {"Scalar", NumberType::get() },
    };
    switch(L.cur().kind) {
      case TK_FLOAT:
        L.next();
        return FloatType::get();
      case TK_INT:
      case TK_BOOL: // TODO: add separate bool type
        L.next();
        return IntType::get();
      default:
        auto tok = L.expect(TK_IDENT);
        auto text = tok.text();
        auto it = type_map.find(text);
        if(it == type_map.end())
          throw ErrorReport(tok.range) << "unknown type specifier";
        return it->second;
    }
  }
  void parseType(Argument& arg) {
    arg.type = parseBaseType();
    if(L.nextIf('[')) {
      arg.type = std::make_shared<ListType>(arg.type);
      if(L.cur().kind == TK_NUMBER) {
        arg.N = std::stoll(L.next().text());
      }
      L.expect(']');
    }
  }

  void parseArgument(std::vector<Argument>& arguments) {
    // varargs
    if(L.nextIf('*')) {
      kwarg_only = true;
      return;
    }
    Argument arg;
    parseType(arg);

    // nullability is ignored for now, since the JIT never cares about it
    L.nextIf('?');
    arg.name = L.expect(TK_IDENT).text();
    if(L.nextIf('=')) {
      parseDefaultValue(arg);
    }
    arg.kwarg_only = kwarg_only;
    arguments.push_back(std::move(arg));
  }
  void parseReturn(std::vector<Argument>& args) {
    Argument arg("ret" + std::to_string(args.size()));
    parseType(arg);
    args.push_back(std::move(arg));
  }
  at::Tensor parseSingleConstant(TypeKind kind) {
    switch(L.cur().kind) {
      case TK_TRUE:
        L.next();
        return one;
      case TK_FALSE:
        L.next();
        return zero;
      case TK_FLOAT:
        L.next();
        return as_tensor(static_cast<int64_t>(at::kFloat));
      case TK_IDENT: {
        auto tok = L.next();
        auto text = tok.text();
        if("cpu" == text) {
          return as_tensor(static_cast<int64_t>(at::Device::Type::CPU));
        } else if("strided" == text) {
          return as_tensor(static_cast<int64_t>(at::kStrided));
        } else {
          throw ErrorReport() << "invalid numeric default value";
        }
      } default:
        std::string n;
        if(L.nextIf('-'))
          n = "-" + L.expect(TK_NUMBER).text();
        else
          n = L.expect(TK_NUMBER).text();
        if(kind == TypeKind::FloatType || n.find(".") != std::string::npos || n.find("e") != std::string::npos) {
          return at::full({}, std::stod(n), at::kDouble); // float?
        } else {
          int64_t v = std::stoll(n);
          return at::full({}, v, at::kLong);
        }
    }
  }
  at::Tensor parseConstantList(TypeKind kind) {
    auto tok = L.expect('[');
    std::vector<at::Tensor> vs;
    if(L.cur().kind != ']') {
      do {
        vs.push_back(parseSingleConstant(kind));
      } while(L.nextIf(','));
    }
    L.expect(']');
    if(vs.size() == 0) {
      switch(kind) {
        case TypeKind::FloatType:
          return at::empty({}, at::kFloat);
        case TypeKind::IntType:
          return at::empty({}, at::kLong);
        default:
          throw ErrorReport(tok) << "empty lists are only supported for float or int types.";
      }
    }
    return at::stack(vs);
  }
  at::Tensor parseTensorDefault(const SourceRange& range) {
    if("None" == L.expect(TK_IDENT).text()) {
      return at::Tensor();
    } else {
      throw ErrorReport(range) << "invalid tensor default value";
    }
  }
  void parseDefaultValue(Argument& arg) {
    auto range = L.cur().range;
    switch(arg.type->kind()) {
      case TypeKind::DynamicType: {
        arg.default_value = parseTensorDefault(range);
      }  break;
      case TypeKind::NumberType:
      case TypeKind::IntType:
      case TypeKind::FloatType:
        arg.default_value = parseSingleConstant(arg.type->kind());
        break;
      case TypeKind::ListType: {
        auto elem_kind = arg.type->cast<ListType>()->getElementType();
        if(L.cur().kind == TK_IDENT) {
          arg.default_value = parseTensorDefault(range);
        } else if(arg.N && L.cur().kind != '[') {
          arg.default_value = parseSingleConstant(elem_kind->kind()).expand({*arg.N});
        } else {
          arg.default_value = parseConstantList(elem_kind->kind());
        }
      } break;
      default:
        throw ErrorReport(range) << "unexpected type, file a bug report";
    }
  }

  template<typename T>
  void parseList(int begin, int sep, int end, std::vector<T>& result, void (SchemaParser::*parse)(std::vector<T>&)) {
    auto r = L.cur().range;
    if (begin != TK_NOTHING)
      L.expect(begin);
    if (L.cur().kind != end) {
      do {
        (this->*parse)(result);
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING)
      L.expect(end);
  }
  Lexer L;
  bool kwarg_only;
  at::Tensor one = at::full({}, 1, at::kLong);
  at::Tensor zero = at::full({}, 0, at::kLong);
};
}

using SchemaMap = std::unordered_map<std::string, std::vector<FunctionSchema>>;

// defined in aten_schema_declarations.cpp
extern const char * schema_declarations;

std::vector<FunctionSchema> createOperatorSchemas() {
  return script::SchemaParser(schema_declarations).parseDeclarations();
}

std::vector<FunctionSchema> & getOperatorSchemas() {
  static std::vector<FunctionSchema> schema = createOperatorSchemas();
  return schema;
}

static SchemaMap createSchemaMap() {
  auto& schemas = getOperatorSchemas();
  SchemaMap result;
  for(auto & schema : schemas) {
    auto it = result.find(schema.name);
    if(it == result.end()) {
      it = result.insert({schema.name, {}}).first;
    }
    it->second.push_back(std::move(schema));
  }
  return result;
}

const std::vector<FunctionSchema>& getOperatorSchema(const std::string& name) {
  static SchemaMap map = createSchemaMap();
  static std::vector<FunctionSchema> empty;
  auto it = map.find(name);
  if(it != map.end())
    return it->second;
  return empty;
}

FunctionSchema parseSchema(const std::string& schema) {
  return script::SchemaParser(schema).parseDeclarations().at(0);
}

}}
