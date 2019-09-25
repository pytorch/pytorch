#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/lexer.h>
#include <torch/csrc/jit/script/parse_string_literal.h>
#include <torch/csrc/jit/script/schema_type_parser.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace script {

struct VarWithType;
struct ParsedLiteral;

class IRParser {
  friend void parseIR(
      const std::string& str,
      torch::jit::Graph* graph,
      std::unordered_map<std::string, Value*>& vmap);
  IRParser(
      const std::string& str,
      torch::jit::Graph* graph,
      std::unordered_map<std::string, Value*>& vmap)
      : L(std::make_shared<Source>(str)),
        g(graph),
        vmap(vmap),
        type_parser(L, /*parse_complete_tensor_types*/ true) {}

  std::string parseVar();
  VarWithType parseVarWithType();
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

  Value* findValueInVMap(const std::string& name);

  torch::jit::script::Lexer L;
  torch::jit::Graph* g = nullptr;
  std::unordered_map<std::string, Value*>& vmap;
  SchemaTypeParser type_parser;
};

struct ParsedLiteral {
  ParsedLiteral() = default;

  AttributeKind k = AttributeKind::t;

  int64_t i = 0;
  std::string s = "";
  double f = 0.0;
  std::vector<int64_t> is;
  std::vector<std::string> ss;
  std::vector<double> fs;
};

struct VarWithType {
  VarWithType() = default;
  std::string name;
  TypePtr type;
};

void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    std::unordered_map<std::string, Value*>& vmap) {
  torch::jit::script::IRParser p(str, graph, vmap);
  p.parse();
}

void parseIR(const std::string& str, torch::jit::Graph* graph) {
  std::unordered_map<std::string, Value*> vmap;
  parseIR(str, graph, vmap);
}

VarWithType IRParser::parseVarWithType() {
  VarWithType r;
  r.name = parseVar();
  r.type = TensorType::get();
  if (L.nextIf(':')) {
    auto type_alias = type_parser.parseType();
    AT_ASSERTM(!type_alias.second, "Parsing IR with Alias Info not handled");
    r.type = type_alias.first;
  }
  return r;
}

std::string IRParser::parseVar() {
  L.expect('%');
  if (L.cur().kind == TK_IDENT) {
    auto name = L.expect(TK_IDENT).text();
    if (L.cur().kind == TK_NUMBER) {
      auto suffix = L.expect(TK_NUMBER).text();
      AT_ASSERT(suffix[0] == '.');
      name += suffix;
    }
    return name;
  } else {
    return L.expect(TK_NUMBER).text();
  }
}

void IRParser::parseOperatorOutputs(std::vector<VarWithType>* outs) {
  if (L.cur().kind != '%') {
    return;
  }
  parseList(TK_NOTHING, ',', TK_NOTHING, [&] {
    outs->push_back(parseVarWithType());
  });
  L.expect('=');
}

// Parse string or numeric literal and return it along with its type.
ParsedLiteral IRParser::parseScalarLiteral(Node* n) {
  auto token = L.cur();
  std::string str;
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
      L.expect(TK_NUMBER);
      // Fallthrough
    case TK_NUMBER:
      str += L.cur().text();

      if (str.find('.') != std::string::npos ||
          str.find('e') != std::string::npos) {
        r.k = AttributeKind::f;
        r.f = std::stod(str);
      } else {
        r.k = AttributeKind::i;
        r.i = std::stoll(str);
      }
      L.next();
      return r;
    default:
      throw ErrorReport(token.range)
          << "Could not parse literal" << token.text();
  }
}

/** \brief Parse attribute and add it to the node N.
 *
 * The function determines the attribute type (string, int, float, list of
 * strings, list of ints, list of floats, and a list of tensors (currently only
 * for empty lists)).
 * An attribute looks like the following:
 *   AttrName=AttrValue
 *  Where AttrValue can be a list or a scalar literal, e.g.:
 *   size = 27
 *   name = "Bob"
 *   coefs = [1.2, 3.4, 0.6]
 */
void IRParser::parseAttr(Node* n) {
  std::string attrname = L.expect(TK_IDENT).text();
  L.expect('=');
  if (L.cur().kind == '[') {
    // list
    AttributeKind k = AttributeKind::ts;
    std::vector<int64_t> is;
    std::vector<std::string> ss;
    std::vector<double> fs;
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
        default:
          throw ErrorReport(L.cur().range) << "Unexpected attr type";
      }
    });
    switch (k) {
      case AttributeKind::ts:
        n->ts_(Symbol::attr(attrname), {});
        break;
      case AttributeKind::ss:
        n->ss_(Symbol::attr(attrname), ss);
        break;
      case AttributeKind::fs:
        n->fs_(Symbol::attr(attrname), fs);
        break;
      case AttributeKind::is:
        n->is_(Symbol::attr(attrname), is);
        break;
      default:
        throw ErrorReport(L.cur().range) << "Unexpected attr type";
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
      default:
        throw ErrorReport(L.cur().range) << "Unexpected attr type";
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
  std::string name = parseOperatorName();
  Node* n = g->create(Symbol::fromQualString(name), {}, outs.size());

  // Parse attributes and inputs.
  parseOperatorInputs(n);

  // Register outputs.
  int idx = 0;
  for (const VarWithType& v : outs) {
    vmap[v.name] = n->outputs()[idx++];
    vmap[v.name]->setType(v.type);
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
  std::string graphName = L.expect(TK_IDENT).text();
  parseGraphInputs();
  L.expect(':');

  // After the definition we should have a list of statements, parse it:
  parseOperatorsList(g->block());

  // The last statement should be return, which specifies graph outputs
  parseReturnOperator();
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
    throw ErrorReport(L.cur().range)
        << "Cannot find a variable with name '" << name << "'";
  }
  return vmap.at(name);
}

} // namespace script
} // namespace jit
} // namespace torch
