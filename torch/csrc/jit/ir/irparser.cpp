#include <torch/csrc/jit/ir/irparser.h>

#include <ATen/EmptyTensor.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/schema_type_parser.h>
#include <torch/csrc/jit/ir/ir.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif

#include <string>
#include <vector>

namespace torch::jit {

struct VarWithType;
struct ParsedLiteral;

class IRParser {
  friend void parseIR(
      const std::string& str,
      torch::jit::Graph* graph,
      std::unordered_map<std::string, Value*>& vmap,
      bool parse_tensor_constants);
  IRParser(
      const std::string& str,
      torch::jit::Graph* graph,
      std::unordered_map<std::string, Value*>& vmap,
      bool parse_tensor_constants)
      : L(std::make_shared<Source>(str)),
        g(graph),
        vmap(vmap),
        type_parser(
            L,
            /*parse_complete_tensor_types*/ true,
            /*allow_type_vars*/ true),
        parse_tensor_constants_(parse_tensor_constants) {}

  std::string parseVar();
  VarWithType parseVarWithType(bool allow_optional = false);
  ParsedLiteral parseScalarLiteral(Node* n);

  void parse();
  void parseGraphInputs();
  void parseReturnOperator();

  void parseBlocks(Node* parentNode);
  void parseBlock(Node* parentNode);
  void parseBlockInputs(Block* b);
  void parseBlockOutputs(Block* b);

  void parseOperatorsList(Block* b);
  void parseOperator(Block* b);
  void parseOperatorOutputs(std::vector<VarWithType>* outs);
  std::string parseOperatorName();
  void parseOperatorInputs(Node* n);
  void parseAttrs(Node* n);
  void parseAttr(Node* n);

  void parseList(
      int begin,
      int sep,
      int end,
      const std::function<void()>& callback);

  void bypassTypeAnnotationList();

  Value* findValueInVMap(const std::string& name);

  torch::jit::Lexer L;
  torch::jit::Graph* g = nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::unordered_map<std::string, Value*>& vmap;
  SchemaTypeParser type_parser;
  bool parse_tensor_constants_;
  std::vector<Node*> deferred_tensor_value_initializations_;
  std::vector<Node*> deferred_empty_container_initializations_;
};

struct ParsedLiteral {
  ParsedLiteral() = default;

  AttributeKind k = AttributeKind::t;

  int64_t i = 0;
  std::string s = "";
  double f = 0.0;
  c10::complex<double> c = c10::complex<double>(0, 0);
  TypePtr ty;
  std::vector<int64_t> is;
  std::vector<std::string> ss;
  std::vector<double> fs;
  std::vector<c10::complex<double>> cs;
  std::vector<TypePtr> tys;
};

struct VarWithType {
  VarWithType() = default;
  std::string name;
  TypePtr type;
};

void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    std::unordered_map<std::string, Value*>& vmap,
    bool parse_tensor_constants) {
  torch::jit::IRParser p(str, graph, vmap, parse_tensor_constants);
  p.parse();
}

void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    bool parse_tensor_constants) {
  std::unordered_map<std::string, Value*> vmap;
  parseIR(str, graph, vmap, parse_tensor_constants);
}

VarWithType IRParser::parseVarWithType(bool allow_optional) {
  VarWithType r;
  r.name = parseVar();
  if (allow_optional) {
    r.type = nullptr;
  } else {
    r.type = TensorType::get();
  }
  if (L.nextIf(':')) {
    auto type_alias = type_parser.parseType();
    TORCH_INTERNAL_ASSERT(
        !type_alias.second, "Parsing IR with Alias Info not handled");
    r.type = type_alias.first;
  }
  return r;
}

std::string IRParser::parseVar() {
  L.expect('%');
  std::string name;
  bool continue_parsing = false;
  do {
    if (L.cur().kind == TK_IDENT) {
      name += L.expect(TK_IDENT).text();
    } else {
      name += L.expect(TK_NUMBER).text();
    }
    continue_parsing = false;
    if (L.nextIf('.')) {
      continue_parsing = true;
      name += '.';
    } else if (L.cur().kind == TK_NUMBER && L.cur().text()[0] == '.') {
      continue_parsing = true;
    }
  } while (continue_parsing);
  return name;
}

void IRParser::parseOperatorOutputs(std::vector<VarWithType>* outs) {
  if (L.cur().kind != '%') {
    return;
  }
  parseList(TK_NOTHING, ',', TK_NOTHING, [&] {
    outs->push_back(parseVarWithType(true));
  });
  L.expect('=');
}

// Parse string or numeric literal and return it along with its type.
ParsedLiteral IRParser::parseScalarLiteral(Node* n) {
  auto token = L.cur();
  std::string str;
  std::pair<TypePtr, std::optional<c10::AliasInfo>> type_alias;
  ParsedLiteral r;
  switch (token.kind) {
    case TK_STRINGLITERAL:
      r.k = AttributeKind::s;
      r.s = parseStringLiteral(token.range, token.text());
      L.next();
      return r;
    case '-':
      str = "-";
      L.next();
      if (L.cur().kind != TK_NUMBER) {
        throw(
            ErrorReport(token.range)
            << "Expected a number after '-' but got:" << token.text());
      }
      [[fallthrough]];
    case TK_NUMBER:
      str += L.cur().text();
      if (str.find('j') != std::string::npos) {
        r.k = AttributeKind::c;
        double imag = 0.0f;
        try {
          imag = std::stod(str.substr(0, str.size() - 1));
        } catch (const std::invalid_argument& e) {
          throw(
              ErrorReport(token.range)
              << "Number cannot be converted to double");
        } catch (const std::out_of_range& e) {
          throw(
              ErrorReport(token.range)
              << "Number is too long to be represented in type double");
        }
        r.c = c10::complex<double>(0, imag);
      } else if (
          str.find('.') != std::string::npos ||
          str.find('e') != std::string::npos) {
        r.k = AttributeKind::f;
        try {
          r.f = std::stod(str);
        } catch (const std::invalid_argument& e) {
          throw(
              ErrorReport(token.range)
              << "Number cannot be converted to double");
        } catch (const std::out_of_range& e) {
          throw(
              ErrorReport(token.range)
              << "Number is too long to be represented in type double");
        }
      } else {
        r.k = AttributeKind::i;
        try {
          r.i = std::stoll(str);
        } catch (const std::invalid_argument& e) {
          throw(
              ErrorReport(token.range)
              << "Number cannot be converted to integer");
        } catch (const std::out_of_range& e) {
          throw(ErrorReport(token.range) << "Number is too big");
        }
      }
      L.next();
      return r;
    case TK_IDENT:
      // Type literal
      r.k = AttributeKind::ty;
      type_alias = type_parser.parseType();
      TORCH_INTERNAL_ASSERT(
          !type_alias.second, "Parsing IR with Alias Info not handled");
      r.ty = type_alias.first;
      return r;
    case '<': {
      L.next();
      auto text = L.expect(TK_IDENT);
      if (text.text() != "Tensor") {
        throw(
            ErrorReport(token.range)
            << "Could not parse literal" << token.text());
      }
      if (!parse_tensor_constants_) {
        throw(
            ErrorReport(token.range)
            << "Tensor constant encountered but `parse_tensor_constants` set to false"
            << token.text());
      }
      L.expect('>');
      // these values will be set with randomly initialized data in
      // a post processing pass;
      deferred_tensor_value_initializations_.push_back(n);
      r.k = AttributeKind::t;
      return r;
    }
    case '{': {
      L.next();
      if (L.cur().kind == '-') {
        L.next();
      }
      if (!parse_tensor_constants_) {
        auto text = L.expect(TK_NUMBER);
        throw(
            ErrorReport(token.range)
            << "Single-element tensor constant encountered but "
            << "`parse_tensor_constants` is set to false " << token.text());
      }
      if (L.cur().kind != TK_NUMBER) {
        auto text = L.expect(TK_NUMBER);
        throw(
            ErrorReport(token.range)
            << "Expected single-element tensor constant to contain a number"
            << token.text());
      }
      auto number = parseScalarLiteral(n);
      switch (number.k) {
        case AttributeKind::i:
          n->ival_(attr::value, c10::Scalar(number.i));
          break;
        case AttributeKind::f:
          n->ival_(attr::value, c10::Scalar(number.f));
          break;
        case AttributeKind::c:
          n->ival_(attr::value, c10::Scalar(number.c));
          break;
        default:
          throw(
              ErrorReport(token.range)
              << "Expected single-element tensor constant to contain a number"
              << token.text());
      }
      deferred_tensor_value_initializations_.push_back(n);
      L.expect('}');
      r.k = AttributeKind::t;
      return r;
    }
    default:
      throw(
          ErrorReport(token.range)
          << "Could not parse literal" << token.text());
  }
}

void IRParser::bypassTypeAnnotationList() {
  int depth = 0;
  bool bypassed_list = false;
  while (depth != 0 || !bypassed_list) {
    if (L.cur().kind == '[') {
      bypassed_list = true;
      depth++;
    } else if (L.cur().kind == ']') {
      depth--;
    }
    L.next();
  }
}

/** \brief Parse attribute and add it to the node N.
 *
 * The function determines the attribute type (string, int, float, complex, list
 * of strings, list of ints, list of floats, list of complex, and a list of
 * tensors (currently only for empty lists)). An attribute looks like the
 * following: AttrName=AttrValue Where AttrValue can be a list or a scalar
 * literal, e.g.: size = 27 name = "Bob" coefs = [1.2, 3.4, 0.6]
 */
void IRParser::parseAttr(Node* n) {
  std::string attrname = L.expect(TK_IDENT).text();
  L.expect('=');
  if (L.cur().kind == '[') {
    // list
    AttributeKind k = AttributeKind::ts;
    c10::List<int64_t> is;
    c10::List<std::string> ss;
    c10::List<double> fs;
    c10::List<c10::complex<double>> cs;
    std::vector<TypePtr> tys;
    int elem_num = 0;
    parseList('[', ',', ']', [&] {
      ParsedLiteral r = parseScalarLiteral(n);
      switch (r.k) {
        case AttributeKind::s:
          ss.push_back(r.s);
          AT_ASSERT(!elem_num++ || k == AttributeKind::ss);
          k = AttributeKind::ss;
          break;
        case AttributeKind::i:
          is.push_back(r.i);
          AT_ASSERT(!elem_num++ || k == AttributeKind::is);
          k = AttributeKind::is;
          break;
        case AttributeKind::f:
          fs.push_back(r.f);
          AT_ASSERT(!elem_num++ || k == AttributeKind::fs);
          k = AttributeKind::fs;
          break;
        case AttributeKind::c:
          cs.push_back(r.c);
          AT_ASSERT(!elem_num++ || k == AttributeKind::cs);
          k = AttributeKind::cs;
          break;
        case AttributeKind::ty:
          tys.push_back(r.ty);
          AT_ASSERT(!elem_num++ || k == AttributeKind::tys);
          k = AttributeKind::tys;
          break;
        default:
          throw(ErrorReport(L.cur().range) << "Unexpected attr type");
      }
    });
    switch (k) {
      case AttributeKind::ts:
        n->ival_(Symbol::attr(attrname), IValue());
        break;
      case AttributeKind::ss:
        n->ival_(Symbol::attr(attrname), IValue(ss));
        break;
      case AttributeKind::fs:
        n->ival_(Symbol::attr(attrname), IValue(fs));
        break;
      case AttributeKind::cs:
        n->ival_(Symbol::attr(attrname), IValue(cs));
        break;
      case AttributeKind::is:
        n->ival_(Symbol::attr(attrname), IValue(is));
        break;
      case AttributeKind::tys:
        n->tys_(Symbol::attr(attrname), tys);
        break;
      default:
        throw(ErrorReport(L.cur().range) << "Unexpected attr type");
    }
  } else if (L.cur().text() == "annotate") {
    L.next();
    L.expect('(');
    auto type = L.cur().text();
    if (type != "List" && type != "Dict") {
      throw(
          ErrorReport(L.cur().range)
          << "Unexpected annotation (only List and Dict can be parsed)");
    }
    L.next();
    // ignore the annotations on the IValue constants, and instead recover
    // type from the Node output
    // Note: we could also use script_type_parser
    bypassTypeAnnotationList();
    L.expect(',');
    // expect an empty definition (note - this isn't always true)
    if (type == "Dict") {
      L.expect('{');
      L.expect('}');
    } else if (type == "List") {
      L.expect('[');
      L.expect(']');
    }
    L.expect(')');
    deferred_empty_container_initializations_.push_back(n);
  } else if (L.cur().text() == "torch") {
    L.next();
    L.expect('.');
    auto function = L.cur().text();
    if (function == "Generator") {
      L.next();
      L.expect('(');
      std::optional<uint64_t> seed;
      std::string device = "cpu";
      while (!L.nextIf(')')) {
        auto arg = L.expect(TK_IDENT).text();
        L.expect('=');
        if (arg == "device") {
          ParsedLiteral r = parseScalarLiteral(n);
          if (r.k != AttributeKind::s) {
            throw(
                ErrorReport(L.cur().range)
                << "Expected string literal for device argument");
          }
          if (r.s != "cpu") {
            throw(
                ErrorReport(L.cur().range)
                << "Only cpu device is supported for Generator at this time.");
          }
          device = r.s;
        } else if (arg == "seed") {
          ParsedLiteral r = parseScalarLiteral(n);
          if (r.k != AttributeKind::i) {
            throw(
                ErrorReport(L.cur().range)
                << "Expected int literal for seed argument");
          }
          if (r.i < 0) {
            throw(
                ErrorReport(L.cur().range)
                << "Seed must be a non-negative integer");
          }
          seed = r.i;
        } else {
          throw(
              ErrorReport(L.cur().range)
              << "Generator only supports the following arguments:\n"
              << "- device\n"
              << "- seed\n"
              << "Got: " << arg);
        }
        L.nextIf(',');
      }
      if (device == "cpu") {
        if (seed.has_value()) {
          n->ival_(
              Symbol::attr(attrname), at::detail::createCPUGenerator(*seed));
        } else {
          n->ival_(Symbol::attr(attrname), at::detail::createCPUGenerator());
        }
      }
    } else {
      throw(
          ErrorReport(L.cur().range)
          << "Expected one of the following torch functions:\n"
          << "- Generator\n"
          << "Got: " << function);
    }
  } else {
    // scalar
    ParsedLiteral r = parseScalarLiteral(n);
    switch (r.k) {
      case AttributeKind::s:
        n->s_(Symbol::attr(attrname), r.s);
        break;
      case AttributeKind::i:
        n->i_(Symbol::attr(attrname), r.i);
        break;
      case AttributeKind::f:
        n->f_(Symbol::attr(attrname), r.f);
        break;
      case AttributeKind::c:
        n->c_(Symbol::attr(attrname), r.c);
        break;
      case AttributeKind::ty:
        n->ty_(Symbol::attr(attrname), r.ty);
        break;
      case AttributeKind::t:
        // initialized with random data later
        break;
      default:
        throw(ErrorReport(L.cur().range) << "Unexpected attr type");
    }
    return;
  }
}

void IRParser::parseAttrs(Node* n) {
  parseList('[', ',', ']', [&] { parseAttr(n); });
}

void IRParser::parseOperatorInputs(Node* n) {
  if (L.cur().kind == '[') {
    parseAttrs(n);
  }
  parseList('(', ',', ')', [&] {
    std::string var_name = parseVar();
    n->addInput(findValueInVMap(var_name));
  });
}

void IRParser::parseBlocks(Node* parentNode) {
  L.expect(TK_INDENT);
  while (L.cur().kind != TK_DEDENT) {
    parseBlock(parentNode);
  }
  L.expect(TK_DEDENT);
}

void IRParser::parseBlockInputs(Block* b) {
  parseList('(', ',', ')', [&] {
    VarWithType v = parseVarWithType();
    // If the name isn't valid, don't use it
    std::string uniq_name = Value::isValidName(v.name) ? v.name : "";
    vmap[v.name] = b->addInput(uniq_name);
    vmap[v.name]->setType(v.type);
  });
}

void IRParser::parseBlockOutputs(Block* b) {
  L.expect(TK_ARROW);
  parseList('(', ',', ')', [&] {
    std::string var_name = parseVar();
    b->registerOutput(findValueInVMap(var_name));
  });
  L.expect(TK_NEWLINE);
  L.expect(TK_DEDENT);
}

/** \brief Parse a block.
 *
 * It should look like the following:
 * blockName(input1, input2, input3, ...):
 *   op1
 *   op2
 *   ...
 *   opN
 *   -> (output1, output2, output3, ...)
 */
void IRParser::parseBlock(Node* parentNode) {
  Block* b = parentNode->addBlock();
  L.expect(TK_IDENT).text(); // Block name is not used anywhere.
  parseBlockInputs(b);
  L.expect(':');
  parseOperatorsList(b);
  parseBlockOutputs(b);
}

/** \brief Parse a list of statements.
 *
 * It is expected to be delimited by TK_NEWLINE and end with TK_RETURN or
 * TK_ARROW.
 */
void IRParser::parseOperatorsList(Block* b) {
  L.expect(TK_INDENT);
  while (L.cur().kind != TK_ARROW && L.cur().kind != TK_RETURN) {
    parseOperator(b);
  }
}

std::string IRParser::parseOperatorName() {
  std::string name = L.expect(TK_IDENT).text();
  L.expect(':');
  L.expect(':');
  name += "::" + L.expect(TK_IDENT).text();
  return name;
}

/** \brief Parse a statement.
 *
 * It should look like the following:
 *   <outputs> = NodeName[<attributes>](<inputs>)
 *     <blocks>
 * Outputs, blocks and attributes are optional.
 */
void IRParser::parseOperator(Block* b) {
  // Parse lefthand side.
  std::vector<VarWithType> outs;
  parseOperatorOutputs(&outs);

  // Parse the name and create the corresponding node in the graph.
  auto source_range = L.cur().range;
  std::string name = parseOperatorName();
  Node* n = g->create(Symbol::fromQualString(name), {}, outs.size())
                ->setSourceRange(source_range);

  // Parse attributes and inputs.
  parseOperatorInputs(n);

  const FunctionSchema* schema = n->maybeSchema();

  // Register outputs.
  unsigned idx = 0;
  for (const VarWithType& v : outs) {
    vmap[v.name] = n->outputs()[idx];
    if (schema && !schema->is_varret()) {
      TORCH_CHECK(
          schema->returns().size() > idx,
          "Operator parsing error: out of bounds access at ",
          idx,
          " to schema->returns() which size is ",
          schema->returns().size(),
          " in size");
      auto schema_return_type = schema->returns().at(idx).type();
      if (!v.type) {
        vmap[v.name]->setType(schema_return_type);
      } else {
        // Don't currently support checking against type variables
        // TODO: support?
        if (!schema_return_type->hasFreeVariables() &&
            !v.type->isSubtypeOf(*schema_return_type)) {
          throw(
              ErrorReport(source_range)
              << "Annotated type " << v.type->repr_str()
              << " does not match schema type "
              << schema_return_type->repr_str() << " for operator " << *schema);
        }
        vmap[v.name]->setType(v.type);
      }
    } else {
      vmap[v.name]->setType(v.type ? v.type : TensorType::get());
    }
    idx++;
  }

  // Insert the new node into block B.
  b->appendNode(n);

  // If the statement has nested blocks, parse them:
  if (L.cur().kind == TK_INDENT) {
    parseBlocks(n);
  }
  L.nextIf(TK_NEWLINE);
}

void IRParser::parseGraphInputs() {
  parseList('(', ',', ')', [&] {
    VarWithType v = parseVarWithType();
    // If the name isn't valid, don't use it
    std::string uniq_name = Value::isValidName(v.name) ? v.name : "";
    vmap[v.name] = g->addInput(uniq_name);
    vmap[v.name]->setType(v.type);
  });
}

/** \brief Parse return statement.
 *
 * It should look like the following:
 *   return (x : TypeX, y : TypeY, z, ...)
 */
void IRParser::parseReturnOperator() {
  L.expect(TK_RETURN);

  // Parse output names and types
  parseList('(', ',', ')', [&] {
    std::string var_name = parseVar();
    g->registerOutput(findValueInVMap(var_name));
  });

  // Consume ending tokens
  if (L.cur().kind != TK_EOF) {
    L.expect(TK_NEWLINE);
    L.expect(TK_DEDENT);
  }
}

/** \brief Parse entire graph.
 *
 * It should look like the following:
 *   graphName (input1, input2, ... inputN):
 *     op1
 *     op2
 *     ...
 *     opN
 *     return (output1, output2, ... outputN)
 */
void IRParser::parse() {
  // Parse graph definition, it should look like the following:
  // graphName (input1, input2, ... inputN):
  L.expect(TK_IDENT);
  parseGraphInputs();
  L.expect(':');

  // After the definition we should have a list of statements, parse it:
  parseOperatorsList(g->block());

  // The last statement should be return, which specifies graph outputs
  parseReturnOperator();

  for (Node* n : deferred_tensor_value_initializations_) {
    auto type = n->output()->type()->expect<TensorType>();
    auto tt = n->output()->type()->cast<TensorType>();
    TORCH_INTERNAL_ASSERT(tt, "expected tensor output ", *n);
    auto sizes = tt->sizes().concrete_sizes();
    TORCH_INTERNAL_ASSERT(sizes);
    auto strides = tt->strides().concrete_sizes();
    TORCH_INTERNAL_ASSERT(strides);
    auto device = tt->device();
    TORCH_INTERNAL_ASSERT(device);
    auto dtype = tt->scalarType();
    TORCH_INTERNAL_ASSERT(dtype);
    auto options = at::TensorOptions(*device).dtype(dtype);

    auto e = at::empty_strided(*sizes, *strides, options);
    if (n->hasAttribute(attr::value)) {
      auto value = n->ival(attr::value);
      e.fill_(value.toScalar());
    }

    auto t = n->t_(attr::value, e);
    (void)t;
  }

  for (Node* n : deferred_empty_container_initializations_) {
    auto type = n->output()->type();
    IValue val;
    if (type->kind() == TypeKind::ListType) {
      val = c10::impl::GenericList(type->containedType(0));
    } else if (type->kind() == TypeKind::DictType) {
      val = c10::impl::GenericDict(
          type->containedType(0), type->containedType(1));
    }
    n->ival_(attr::value, val);
  }
}

void IRParser::parseList(
    int begin,
    int sep,
    int end,
    const std::function<void()>& callback) {
  if (begin != TK_NOTHING) {
    L.expect(begin);
  }
  if (L.cur().kind != end) {
    do {
      callback();
    } while (L.nextIf(sep));
  }
  if (end != TK_NOTHING) {
    L.expect(end);
  }
}

Value* IRParser::findValueInVMap(const std::string& name) {
  if (!vmap.count(name)) {
    throw(
        ErrorReport(L.cur().range)
        << "Cannot find a variable with name '" << name << "'");
  }
  return vmap.at(name);
}

} // namespace torch::jit
