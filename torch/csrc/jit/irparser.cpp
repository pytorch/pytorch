#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/lexer.h>
#include <torch/csrc/jit/script/parse_string_literal.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace script {

struct ParsedValue;
struct ParsedLiteral;

class IRParser {
  friend void parseIR(const std::string& str, torch::jit::Graph* graph);
  IRParser(const std::string& str, torch::jit::Graph* graph)
      : L(str), g(graph) {}

  ParsedValue parseParameter();
  ParsedLiteral parseScalarLiteral(Node* n);

  void parse();
  void parseGraphInputs();
  void parseReturnStmt();

  void parseBlocks(Node* parentNode);
  void parseBlock(Node* parentNode);
  void parseBlockInputs(Block* b);
  void parseBlockOutputs(Block* b);

  void parseStmtsList(Block* b);
  void parseStmt(Block* b);
  void parseStmtOutputs(std::vector<ParsedValue>* outs);
  std::string parseStmtName();
  void parseStmtInputs(Node* n);
  void parseAttrs(Node* n);
  void parseAttr(Node* n);

  void parseList(
      int begin,
      int sep,
      int end,
      const std::function<void()>& callback);

  torch::jit::script::Lexer L;
  torch::jit::Graph* g = nullptr;
  std::unordered_map<std::string, Value*> vmap;
};

struct ParsedLiteral {
  AttributeKind k = AttributeKind::t;

  long i;
  std::string s;
  double f;
  std::vector<long> is;
  std::vector<std::string> ss;
  std::vector<double> fs;
};

struct ParsedValue {
  ParsedValue(){};
  std::string name;
  std::string type;
};

void parseIR(const std::string& str, torch::jit::Graph* graph) {
  torch::jit::script::IRParser p(str, graph);
  p.parse();
}

TypePtr parseType(const std::string& s) {
  if (s == "Tensor" || s == "NoType") {
    return TensorType::get();
  }
  if (s == "int") {
    return IntType::get();
  }
  if (s == "float") {
    return FloatType::get();
  }
  if (s == "string") {
    return StringType::get();
  }
  // TODO: Support other types.
  std::cerr << "ERROR: type '" << s << "' not supported by parser" << std::endl;
  abort();
}

ParsedValue IRParser::parseParameter() {
  L.expect('%');
  ParsedValue r;
  if (L.cur().kind == TK_IDENT) {
    r.name = L.expect(TK_IDENT).text();
  } else {
    r.name = L.expect(TK_NUMBER).text();
  }
  r.type = "NoType";
  if (L.nextIf(':')) {
    r.type = L.expect(TK_IDENT).text();
  }
  return r;
}

void IRParser::parseStmtOutputs(std::vector<ParsedValue>* outs) {
  if (L.cur().kind != '%') {
    return;
  }
  parseList(
      TK_NOTHING, ',', TK_NOTHING, [&] { outs->push_back(parseParameter()); });
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
      std::cout << "Could not parse literal " << token.text() << std::endl;
      abort();
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
  std::string attrname = "attr::" + L.expect(TK_IDENT).text();
  L.expect('=');
  if (L.cur().kind == '[') {
    // list
    AttributeKind k = AttributeKind::ts;
    std::vector<long> is;
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
          std::cout << "Unexpected attr type.\n";
          abort();
      }
    });
    switch (k) {
      case AttributeKind::ts:
        n->ts_(Symbol::fromQualString(attrname), {});
        break;
      case AttributeKind::ss:
        n->ss_(Symbol::fromQualString(attrname), ss);
        break;
      case AttributeKind::fs:
        n->fs_(Symbol::fromQualString(attrname), fs);
        break;
      case AttributeKind::is:
        n->is_(Symbol::fromQualString(attrname), is);
        break;
      default:
        std::cout << "Unexpected type of list.\n";
        abort();
    }
  } else {
    // scalar
    ParsedLiteral r = parseScalarLiteral(n);
    switch (r.k) {
      case AttributeKind::s:
        n->s_(Symbol::fromQualString(attrname), r.s);
        break;
      case AttributeKind::i:
        n->i_(Symbol::fromQualString(attrname), r.i);
        break;
      case AttributeKind::f:
        n->f_(Symbol::fromQualString(attrname), r.f);
        break;
      default:
        std::cout << "Unexpected attr type.\n";
        abort();
    }
    return;
  }
}

void IRParser::parseAttrs(Node* n) {
  parseList('[', ',', ']', [&] { parseAttr(n); });
}

void IRParser::parseStmtInputs(Node* n) {
  if (L.cur().kind == '[') {
    parseAttrs(n);
  }
  parseList('(', ',', ')', [&] {
    ParsedValue i = parseParameter();
    if (!vmap.count(i.name)) {
      vmap[i.name] = g->addInput(); // XXX: Or should we fail here?
      vmap[i.name]->setType(parseType(i.type));
    }
    n->addInput(vmap[i.name]);
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
  L.expect('(');
  // TODO: Actually parse inputs.
  L.expect(')');
}

void IRParser::parseBlockOutputs(Block* b) {
  L.expect(TK_ARROW);
  parseList('(', ',', ')', [&] {
    ParsedValue o = parseParameter();
    b->registerOutput(vmap[o.name]);
  });
  L.expect(TK_NEWLINE);
  L.expect(TK_DEDENT);
}

/** \brief Parse a block.
 *
 * It should look like the following:
 * blockName(input1, input2, input3, ...):
 *   stmt1
 *   stmt2
 *   ...
 *   stmtN
 *   -> (output1, output2, output3, ...)
 */
void IRParser::parseBlock(Node* parentNode) {
  Block* b = parentNode->addBlock();
  L.expect(TK_IDENT).text(); // TODO: Do we need the block name anywhere?
  parseBlockInputs(b);
  L.expect(':');
  parseStmtsList(b);
  parseBlockOutputs(b);
}

/** \brief Parse a list of statements.
 *
 * It is expected to be delimited by TK_NEWLINE and end with TK_RETURN or
 * TK_ARROW.
 */
void IRParser::parseStmtsList(Block* b = nullptr) {
  L.expect(TK_INDENT);
  while (L.cur().kind != TK_ARROW && L.cur().kind != TK_RETURN) {
    parseStmt(b);
  }
}

std::string IRParser::parseStmtName() {
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
void IRParser::parseStmt(Block* b) {
  // Parse lefthand side.
  std::vector<ParsedValue> outs;
  parseStmtOutputs(&outs);

  // Parse the name and create the corresponding node in the graph.
  std::string name = parseStmtName();
  Node* n = g->create(Symbol::fromQualString(name), {}, outs.size());

  // Parse attributes and inputs.
  parseStmtInputs(n);

  // Register outputs.
  int idx = 0;
  for (const ParsedValue& o : outs) {
    vmap[o.name] = n->outputs()[idx++];
    vmap[o.name]->setType(parseType(o.type));
  }

  // Insert the new node into block B if we're inside it or into the graph
  // directly if we are not inside a block.
  if (b) {
    b->appendNode(n);
  } else {
    g->insertNode(n);
  }

  // If the statement has nested blocks, parse them:
  if (L.cur().kind == TK_INDENT) {
    parseBlocks(n);
  }
  L.nextIf(TK_NEWLINE);
}

void IRParser::parseGraphInputs() {
  parseList('(', ',', ')', [&] {
    ParsedValue v = parseParameter();
    // TODO: Do we need to do anything with the type here?
    vmap[v.name] = g->addInput();
  });
}

/** \brief Parse return statement.
 *
 * It should look like the following:
 *   return (x : TypeX, y : TypeY, z, ...)
 */
void IRParser::parseReturnStmt() {
  L.expect(TK_RETURN);

  // Parse output names and types
  parseList('(', ',', ')', [&] {
    ParsedValue o = parseParameter();
    // Outputs should already be in VMAP, otherwise we're trying to return
    // undefined value.
    AT_ASSERT(vmap.count(o.name));
    g->registerOutput(vmap.at(o.name));
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
 *     stmt1
 *     stmt2
 *     ...
 *     stmtN
 *     return (output1, output2, ... outputN)
 */
void IRParser::parse() {
  // Parse graph definition, it should look like the following:
  // graphName (input1, input2, ... inputN):
  std::string graphName = L.expect(TK_IDENT).text();
  parseGraphInputs();
  L.expect(':');

  // After the definition we should have a list of statements, parse it:
  parseStmtsList();

  // The last statement should be return, which specifies graph outputs
  parseReturnStmt();
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

} // namespace script
} // namespace jit
} // namespace torch
