#include "ATen/ATen.h"
#include "torch/csrc/jit/script/lexer.h"
#include "torch/csrc/jit/script/tree.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/tensor_conversions.h"
#include "torch/csrc/jit/script/error_report.h"

namespace torch { namespace jit {

namespace script {
struct SchemaParser {
  SchemaParser(const std::string& str)
  : L(str) {}

  FunctionSchema parseDeclaration() {
    auto name = L.expect(TK_IDENT).text();
    if(L.nextIf(':')) {
      L.expect(':');
      name = name + "::" + L.expect(TK_IDENT).text();
    }
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
        return one();
      case TK_FALSE:
        L.next();
        return zero();
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
        } else if("ElementwiseMean" == text) {
          return as_tensor(static_cast<int64_t>(Reduction::ElementwiseMean));
        } else {
          throw ErrorReport(L.cur().range) << "invalid numeric default value";
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
  static at::Tensor one() {
    static at::Tensor v = at::full({}, 1, at::kLong);
    return v;
  }
  static at::Tensor zero() {
    static at::Tensor v = at::full({}, 0, at::kLong);
    return v;
  }
};
}


namespace {

using OperatorMap = std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>>;
struct OperatorRegistry  {
  OperatorMap operators;
  std::mutex lock;
  void registerOperator(Operator&& op){
    std::lock_guard<std::mutex> guard(lock);
    Symbol sym = Symbol::fromQualString(op.schema.name);
    operators[sym].push_back(std::make_shared<Operator>(std::move(op)));
  }
  const std::vector<std::shared_ptr<Operator>>& getOperators(Symbol name) {
    std::lock_guard<std::mutex> guard(lock);
    static std::vector<std::shared_ptr<Operator>> empty;
    auto it = operators.find(name);
    if(it != operators.end())
      return it->second;
    return empty;
  }
};

OperatorRegistry& getRegsitry() {
  static OperatorRegistry r;
  return r;
}

}

void registerOperator(Operator&& op) {
  getRegsitry().registerOperator(std::move(op));
}

const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name) {
  return getRegsitry().getOperators(name);
}

FunctionSchema parseSchema(const std::string& schema) {
  return script::SchemaParser(schema).parseDeclarations().at(0);
}

at::optional<AttributeKind> attributeKindOf(TypePtr type) {
  switch(type->kind()) {
    case TypeKind::IntType: return AttributeKind::i;
    case TypeKind::FloatType: return AttributeKind::f;
    case TypeKind::NumberType: return AttributeKind::t;
    case TypeKind::ListType:
      if(type->isSubtypeOf(*ListType::ofInts()))
        return AttributeKind::is;
      else
        return at::nullopt;
    default:
      return at::nullopt;
  }
}

bool typeMatches(TypePtr actual, TypePtr formal) {
  if(actual->isSubtypeOf(*formal))
    return true;

  // XXX - this is here because we allow tensors to be used in place of numbers
  // or lists of numbers in the script because of the restriction that all inputs to script must be tensors.
  // Once numbers are always treated as seperate types from Tensors, this line
  // should be removed, since it opens up the possibility of ambigous declarations
  // dispatching to the wrong implementation.
  if ((formal->isSubtypeOf(*NumberType::get()) ||
       formal->isSubtypeOf(*ListType::ofInts())) &&
      actual->isSubtypeOf(*DynamicType::get()))
    return true;

  return false;
}

bool Operator::matchesNode(Node* node) const {
  size_t attributes_size = node->numAttributes();
  size_t attributes_seen = 0;
  auto inputs_size = node->inputs().size();
  size_t input_i = 0;
  for(size_t arg_i = 0; arg_i < schema.arguments.size(); ++arg_i) {
    at::optional<AttributeKind> attribute_kind;
    const Argument& arg = schema.arguments[arg_i];
    if(attributes_size > 0 && (attribute_kind = attributeKindOf(arg.type))) {
      auto name = Symbol::fromQualString("attr::" + arg.name);
      if(!node->hasAttribute(name) || node->kindOf(name) != *attribute_kind) {
        // std::cout << "missing attribute: " << name << "\n";
        return false;
      }
      attributes_seen++;
    } else if(*arg.type == *ListType::ofTensors()) {
      // Tensor[] is handled as varargs, consume inputs until the remaining required arguments
      // XXX - there can only be a single Tensor[] in a declaration
      size_t remaining_required = 0;
      for(size_t j = arg_i + 1; j < schema.arguments.size(); ++j){
        // remaining arguments are only those that won't be consumed from attributes
        if(attributes_size == 0 || !attributeKindOf(schema.arguments[j].type))
          remaining_required++;
      }
      while(inputs_size - input_i > remaining_required) {
        auto input = node->inputs()[input_i++];
        if(!typeMatches(input->type(), DynamicType::get())) {
          // std::cout << "vararg argument is not Dynamic\n";
          return false;
        }
      }
    } else {
      if(input_i == inputs_size) {
        // std::cout << "not enough inputs\n";
        return false;
      }
      auto input = node->inputs()[input_i++];
      if(!typeMatches(input->type(), arg.type)) {
        // std::cout << "argument " << arg_i << " has the wrong type\n";
        return false;
      }
    }
  }

  if(!schema.is_vararg && input_i != inputs_size) {
    // std::cout << "not all inputs used\n" << input_i << " " << inputs_size << "\n";
    return false;
  }
  if(!schema.is_vararg && attributes_seen != attributes_size) {
    // std::cout << "not all attributes used\n" << attributes_seen << " " << attributes_size << "\n";
    return false;
  }
  return true;
}

std::shared_ptr<Operator> findOperatorFor(Node* node) {
  const auto& candidates = getAllOperatorsFor(node->kind());
  for(const auto& candidate : candidates) {
    if(candidate->matchesNode(node)) {
      return candidate;
    }
  }
  return nullptr;
}

const Operator& getOperatorFor(Node* node) {
  auto op = findOperatorFor(node);
  if(op)
    return *op;

  auto er = script::ErrorReport(node->getSourceLocation());
  er << "Schema not found for node. File a bug report.\n";
  er << "Node: " << *node << "\n";
  er << "Input types:";
  for(size_t i = 0; i < node->inputs().size(); ++i) {
    if(i > 0)
      er << ", ";
    er << *node->inputs()[i]->type();
  }
  er << "\ncandidates were:\n";
  const auto& candidates = getAllOperatorsFor(node->kind());
  for(auto & candidate : candidates) {
    er << "  " << candidate->schema << "\n";
  }
  throw er;
}

}}
