#include <torch/csrc/jit/passes/python_print.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/attributes.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

using c10::QualifiedName;

namespace torch {
namespace jit {

// unix isprint but insensitive to locale
static bool isPrint(char s) {
  return s > 0x1f && s < 0x7f;
}

void printQuotedString(std::ostream& stmt, const std::string& str) {
  stmt << "\"";
  for (auto s : str) {
    switch (s) {
      case '\\':
        stmt << "\\\\";
        break;
      case '\'':
        stmt << "\\'";
        break;
      case '\"':
        stmt << "\\\"";
        break;
      case '\a':
        stmt << "\\a";
        break;
      case '\b':
        stmt << "\\b";
        break;
      case '\f':
        stmt << "\\f";
        break;
      case '\n':
        stmt << "\\n";
        break;
      case '\r':
        stmt << "\\r";
        break;
      case '\t':
        stmt << "\\t";
        break;
      case '\v':
        stmt << "\\v";
        break;
      default:
        if (isPrint(s)) {
          stmt << s;
        } else {
          // C++ io has stateful formatting settings. Messing with
          // them is probably worse than doing this manually.
          char buf[4] = "000";
          buf[2] += s % 8;
          s /= 8;
          buf[1] += s % 8;
          s /= 8;
          buf[0] += s;
          stmt << "\\" << buf;
        }
        break;
    }
  }
  stmt << "\"";
}

static bool isValidIdentifierChar(char c, size_t pos) {
  return islower(c) || isupper(c) || c == '_' || (pos > 0 && isdigit(c));
}

static bool isValidIdentifier(const std::string& name) {
  if (name.size() == 0)
    return false;
  for (size_t i = 0; i < name.size(); ++i) {
    if (!isValidIdentifierChar(name[i], i))
      return false;
  }
  return true;
}

static void emitQualifiedName(std::ostream& out, const QualifiedName& name) {
  const auto& name_ = name.name();
  const auto& prefix_ = name.prefix();
  if (isValidIdentifier(name_)) {
    if (!prefix_.empty()) {
      emitQualifiedName(out, QualifiedName(prefix_));
      out << ".";
    }
    out << name_;
  } else {
    AT_ASSERT(!prefix_.empty());
    out << "getattr(";
    emitQualifiedName(out, QualifiedName(prefix_));
    out << ", ";
    printQuotedString(out, name_);
    out << ")";
  }
}

// Get a stringified version of the qualified name.
// if a field is not a valid Python identifier, then it will print as, e.g.
// getattr(self, "0").b
static std::string getValidQualifiedName(const QualifiedName& name) {
  std::stringstream ss;
  emitQualifiedName(ss, name);
  return ss.str();
}

// some names are valid identifiers but off limits because
// they are keywords or namespaces used in the output
const static std::unordered_set<std::string> reserved_names = {
    // identifiers in the environment while parsing
    "_", // avoid the confusing unnamed _
    "aten",
    "attribute",
    "CONSTANTS",
    "fork",
    "getattr",
    "inf",
    "nan",
    "ops",
    "__torch__",
    // the python keywords
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
};

struct PythonPrintPass {
  std::ostringstream body_;

  // constants are written to this table, and given then named CONSTANTS.cN
  // where N is the index into this table.
  std::vector<at::Tensor>& tensor_table_;

  // Any classes used are written to this table, to be later written out as
  // dependencies.
  std::vector<ClassTypePtr>& class_table_;
  std::vector<ClassTypePtr> class_deps_;
  // Helper to avoid duplicating class types
  void addToClassTable(const ClassTypePtr& classType) {
    // we serialize module classes separately.
    // Including them in the class table as well will cause the code
    // to get imported twice.
    if (classType->is_module()) {
      return;
    }
    if (std::find(class_table_.cbegin(), class_table_.cend(), classType) ==
        class_table_.cend()) {
      class_table_.push_back(classType);
    }
    if (std::find(class_deps_.cbegin(), class_deps_.cend(), classType) ==
        class_deps_.cend()) {
      class_deps_.push_back(classType);
    }
  }

  // When printing this node, is it safe to write it inline (i.e. without
  // assigning a temporary variable
  std::unordered_set<Node*> output_inline_;

  // when we print this, should we error if the resulting output would
  // not be able to be reparsed?
  bool enforce_importable_;

  // are funcitons being printed considered methods
  // either of a class or some module?
  // If true, this will surpress type annotation on their
  // first (self) argument. And forked functions will
  // be emitted as method calls (self.__fork...) rather
  // than as method calls
  bool is_method_;

  // what valid identifiers are in use for the current function
  std::unordered_set<std::string> used_names_;

  // used method names
  std::unordered_set<std::string> used_method_names_;

  // scanValue, scanNode, scanBlock:
  // decide if it is safe to omit the output of a temporary variable,
  // and inline the expression into its use
  // we only do this if
  // (1) it is a constant, or
  // (2) the temporary is unnamed, is single output, is used once,
  //     and would appear in the same order when the expression tree is
  //     reparsed.
  // The last case can be checked
  // becuase when we emit a expresion tree in the parser,
  // we do a left-to-right postorder traversal of the expression tree (emit
  // children, then emit op). The reverse of this is a right-to-left preorder
  // traversal of the tree. By doing a right-to-left preorder traversal of the
  // inputs of a node, while also scanning the list of emitted nodes backward,
  // we can see if they line up with what would happen when parsed the node as
  // an expression. While they line up we collapse them into an inline
  // expression.

  // The inductive step is that the right-most input should be produced by the
  // node immediatly before the current node if it is in tree order.

  bool canInline(Value* v) {
    Node* n = v->node();
    // there must be only 1 values, otherwise we need an assignment to handle
    // the multiple outout values
    if (n->outputs().size() != 1)
      return false;
    // if it is used more than once, then we need a variable
    if (v->uses().size() != 1)
      return false;
    auto use = v->uses().at(0);
    // if it has a name set, then it was written as a variable so preserve that
    // unless it is being fed directly to the end of the block.
    // in which case it is not as useful to give it a name just to return it
    if (v->hasDebugName() && use.user->kind() != prim::Return)
      return false;
    // don't try to inline control blocks
    if (n->blocks().size() != 0)
      return false;
    // if it is a loop-carried input, we need a variable
    // otherwise the condition or trip count may be emitted in the wrong order
    // w.r.t. to it
    if (use.user->kind() == prim::Loop && use.offset >= 2)
      return false;

    // subgraph may use this more than once, so disable inlining
    if (use.user->kind() == prim::fork)
      return false;

    return true;
  }

  // block_point is the current node in the reverse linear scan of the emitted
  // nodes v is the current value in the tree traversal that may match with
  // block_point's output.
  Node* scanValue(Node* block_point, Value* v) {
    Node* n = v->node();
    AT_ASSERT(n->kind() == prim::Constant || output_inline_.count(n) == 0);

    if (n == block_point &&
        canInline(v)) { // the node must be at the expected point of the typical
                        // tree traversal
      // recursively see if we can inline the inputs to this input
      block_point = scanNode(block_point);
      output_inline_.insert(n);
    } else if (n->kind() == prim::Constant) {
      // constant nodes can always be inlined, we will de-dup them on parsing
      // and put them at the top of the function regardless
      output_inline_.insert(n);
    }
    return block_point;
  }
  Node* previousNonConstant(Node* n) {
    do {
      n = n->prev();
    } while (n->kind() == prim::Constant);
    return n;
  }

  Node* scanNode(Node* n) {
    // don't bother to scan nodes we have already determined to be inline
    if (output_inline_.count(n)) {
      return n;
    }
    for (auto b : n->blocks()) {
      scanBlock(b);
    }
    Node* block_point = previousNonConstant(n);
    for (auto it = n->inputs().rbegin(), end = n->inputs().rend(); it != end;
         ++it) {
      block_point = scanValue(block_point, *it);
    }
    return block_point;
  }

  void scanBlock(Block* b) {
    scanNode(b->return_node());
    for (auto node : b->nodes().reverse()) {
      scanNode(node);
    }
  }

  size_t getOrAddTensorConstant(at::Tensor t) {
    // XXX - N^2 warning. This code does the exact same thing as
    // ConstantPool, which is also N^2 in the size of the constants,
    // because it doesn't hash any information about the tensors.
    // We will probably need to optimize this at some point using hashing.
    for (size_t i = 0; i < tensor_table_.size(); ++i) {
      if (t.type() == tensor_table_[i].type() && t.equal(tensor_table_[i])) {
        return i;
      }
    }
    AT_ASSERT(t.is_variable());
    tensor_table_.emplace_back(std::move(t));
    return tensor_table_.size() - 1;
  }

  std::unordered_set<Node*> seen_constants;
  void buildConstantList(Node* n, std::vector<Node*>& constants) {
    for (auto input : n->inputs()) {
      if (input->node()->kind() == prim::Constant &&
          seen_constants.count(input->node()) == 0) {
        constants.push_back(input->node());
        seen_constants.insert(input->node());
      }
    }
    for (auto b : n->blocks()) {
      buildConstantList(b, constants);
    }
  }
  void buildConstantList(Block* b, std::vector<Node*>& constants) {
    for (auto n : b->nodes())
      buildConstantList(n, constants);
    buildConstantList(b->return_node(), constants);
  }

  // get a new name unique across calls to debugName() and
  // anything we have used.
  std::unordered_map<std::string, size_t> next_id;

  std::string genNameImpl(
      const std::string& candidate,
      std::unordered_set<std::string>& used) {
    std::string name = candidate;
    while (used.count(name) || reserved_names.count(name)) {
      name = candidate + std::to_string(next_id[name]++);
    }
    used.insert(name);
    return name;
  }
  std::string genName(const std::string& candidate) {
    return genNameImpl(candidate, used_names_);
  }

  // methods self.foo are in a different namespace than
  // global identifiers, so they have a different procedure for finding a
  // uniquename
  std::string genMethodName(const std::string& candidate) {
    return genNameImpl(candidate, used_method_names_);
  }

  // unique names might not be valid identifiers,
  // force them to be by rewriting them
  static std::string makeValidIdentifier(const std::string& candidate) {
    std::stringstream ss;
    if (candidate.size() == 0 || isdigit(candidate[0]))
      ss << "_";
    for (char c : candidate) {
      if (isupper(c) || islower(c) || isdigit(c) || c == '_')
        ss << c;
      else
        ss << '_';
    }
    return ss.str();
  }
  // if we have to assign 'v' a name, what should it be?
  // use the debugName if it was set, otherwise generate a name.
  std::string genUniqueNameFor(Value* v) {
    return genName(
        v->hasDebugName() ? makeValidIdentifier(v->debugNameBase()) : "_");
  }

  // map from Value to how it should be printed at each use
  std::unordered_map<Value*, std::string> value_names_;

  std::string useOf(Value* v) const {
    return value_names_.at(v);
  }
  void assignValue(Value* v, const std::string& s) {
    value_names_[v] = s;
  }
  void assignValue(Value* v, Value* w) {
    assignValue(v, useOf(w));
  }
  void assignValuesToTheirUniqueNames(at::ArrayRef<Value*> values) {
    for (auto v : values) {
      assignValue(v, genUniqueNameFor(v));
    }
  }

  size_t level = 0;
  // indent to the current indent level
  std::ostream& indent() {
    for (size_t i = 0; i < level; ++i) {
      body_ << "  ";
    }
    return body_;
  }

  ResourceGuard WithIndented() {
    level++;
    return ResourceGuard([this] { level--; });
  }

  template <class T0, class T1, class F>
  void zipWith(at::ArrayRef<T0> list_a, at::ArrayRef<T1> list_b, F action)
      const {
    auto it_a = list_a.begin();
    auto it_b = list_b.begin();

    if (list_a.size() != list_b.size()) {
      AT_ERROR("Python printer expected 2 lists of same size");
    }

    for (; it_a != list_a.end(); ++it_a, ++it_b) {
      action(*it_a, *it_b);
    }
  }

  void printValueList(
      std::ostream& stmt,
      at::ArrayRef<Value*> list,
      const char* begin = "",
      const char* end = "") {
    stmt << begin;
    auto delimiter = "";
    for (auto* value : list) {
      stmt << delimiter;
      stmt << useOf(value);
      delimiter = ", ";
    }
    stmt << end;
  }

  void printDict(
      std::ostream& stmt,
      at::ArrayRef<Value*> key_value_pairs,
      const char* begin = "{",
      const char* end = "}") {
    stmt << begin;
    auto delimiter = "";
    for (size_t i = 0; i < key_value_pairs.size(); i += 2) {
      stmt << delimiter;
      auto key = key_value_pairs[i];
      auto value = key_value_pairs[i + 1];

      stmt << useOf(key) << ": " << useOf(value);

      delimiter = ", ";
    }
    stmt << end;
  }

  void printAssignment(at::ArrayRef<Value*> lhs, at::ArrayRef<Value*> rhs) {
    if (lhs.size() > 0) {
      indent();
      printValueList(body_, lhs);
      body_ << " = ";
      printValueList(body_, rhs);
      body_ << "\n";
    }
  }

  void printIf(IfView stmt) {
    assignValuesToTheirUniqueNames(stmt.outputs());
    indent() << "if " << useOf(stmt.cond()) << ":\n";
    {
      auto guard = WithIndented();
      // Print node contents
      printBlock(stmt.thenBlock(), stmt.outputs().size() > 0);
      printAssignment(stmt.outputs(), stmt.thenOutputs());
    }
    indent() << "else:\n";
    {
      auto guard = WithIndented();
      printBlock(stmt.elseBlock(), stmt.outputs().size() > 0);
      printAssignment(stmt.outputs(), stmt.elseOutputs());
    }
  }

  // our way of encoding loops makes them difficult to turn back into python
  // syntax. we have to check properties of the condition and trip count inputs
  // to figure out which one it initially was
  static bool shouldEmitAsForLoop(LoopView stmt) {
    auto trip_count = toIValue(stmt.maxTripCount());
    auto cond_input = toIValue(stmt.inputCond());
    auto cond_next = toIValue(stmt.nextCond());

    bool condition_is_always_true =
        cond_input && cond_input->toBool() && cond_next && cond_next->toBool();
    bool trip_count_is_specified = !trip_count || // trip is not a constant
        trip_count->toInt() !=
            std::numeric_limits<int64_t>::max() || // it is a constant but not
                                                   // the default one
        stmt.currentTripCount()->uses().size() >
            0; // it is actually being used in the body.

    if (condition_is_always_true) {
      // if the trip count was not specified this was a user-written while True:
      return trip_count_is_specified;
    } else {
      // this must be a while loop, but check that there isn't _also_ a trip
      // count
      if (trip_count_is_specified) {
        throw script::ErrorReport(stmt.node()->sourceRange())
            << "loop cannot be printed as python "
            << "because it has gone through an optimization "
            << "that combined while and for loops. File a bug.";
      }
      return false;
    }
  }

  void printLoop(LoopView stmt) {
    // Loop carried dependencies are handled by assigning their initial
    // values to the node->outputs() before the loop,
    // and assign node->outputs() to the new values at the end of each trip.

    bool emit_as_for_loop = shouldEmitAsForLoop(stmt);

    assignValuesToTheirUniqueNames(stmt.carriedOutputs());
    // Add aliases for loop-carried dependencies
    zipWith(
        stmt.bodyCarriedInputs(), // Start at 1 to ignore trip count
        stmt.carriedOutputs(),
        [&](Value* block_input, Value* node_output) {
          assignValue(block_input, node_output);
        });

    // Print initial assignments of loop node outputs = loop node inputs
    printAssignment(stmt.carriedOutputs(), stmt.carriedInputs());

    assignValuesToTheirUniqueNames(stmt.currentTripCount());
    // Loop header
    if (emit_as_for_loop) {
      indent();
      body_ << "for " << useOf(stmt.currentTripCount()) << " in range("
            << useOf(stmt.maxTripCount()) << "):\n";
    } else {
      // note: trip_count_in_block is unused because this is a while loop,
      // so we reuse the Value* as a stand-in for the loop condition
      printAssignment(stmt.currentTripCount(), stmt.inputCond());
      indent();
      body_ << "while " << useOf(stmt.currentTripCount()) << ":\n";
    }
    // Loop body
    {
      ResourceGuard indent = WithIndented();
      // Update block outputs to block inputs for next loop iteration
      // skip the assignment to the new condition in for loops because
      // the condition is always True
      size_t offset = emit_as_for_loop ? 1 : 0;
      auto body_block = stmt.bodyBlock();
      ArrayRef<Value*> loop_carried_block_inputs =
          body_block->inputs().slice(offset);
      printBlock(body_block, loop_carried_block_inputs.size() > 0);
      printAssignment(
          loop_carried_block_inputs, body_block->outputs().slice(offset));
    }
  }

  bool isLongLine(const std::string& str) {
    return str.size() + level * 2 >= 40;
  }

  bool isLongInline(Node* node) {
    return output_inline_.count(node) && isLongLine(useOf(node->output()));
  }

  bool isNonConstantInline(Value* input) {
    return input->node()->kind() != prim::Constant &&
        output_inline_.count(input->node());
  }

  // [reordering of inlines]
  // We inline anything that is semantically legal to inline, but sometimes
  // we find that these lines get too long. In that case we break the lines
  /// and it  is important that we un-inline all the inputs preceeding the long
  /// input:
  //   r = foo(x.add_(b), some_long + expression)
  //  wrong!
  //   _0 = some_long + expression
  //   r = foo(x.add_(b), _0) # wrong! _0 runs before mutating add_
  // legal!
  //   _0 = x.add_(b)
  //   _1 = some_long + expression
  //   r = foo(_0, _1)
  void splitLongInlines(at::ArrayRef<Value*> inputs) {
    size_t long_inline_slice = 0;
    // find the last input that is too long
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (isLongInline(inputs[i]->node())) {
        long_inline_slice = i + 1;
      }
    }
    // un-inline everything through the last long line
    // constants are ignored since long constants are never inlined in the
    // first place
    for (size_t i = 0; i < long_inline_slice; ++i) {
      if (isNonConstantInline(inputs[i])) {
        printOutputDefinition(inputs[i]->node(), useOf(inputs[i]));
      }
    }
  }

  void printOutputDefinition(Node* node, const std::string& str) {
    assignValuesToTheirUniqueNames(node->outputs());
    indent();
    // Print outputs
    if (node->outputs().size() > 0) {
      printValueList(body_, node->outputs());
      body_ << " = ";
    }
    body_ << str << "\n";
  }

  // Recursively check contained types for any class dependencies
  void registerClassDependencies(const TypePtr& type) {
    if (const auto classType = type->cast<ClassType>()) {
      addToClassTable(classType);
    }
    for (const auto& containedType : type->containedTypes()) {
      registerClassDependencies(containedType);
    }
  }

  void printNode(Node* node, bool print_const) {
    // Check for class dependencies. If this node inputs or outputs a class
    // type, we need to add it to our table of dependencies.
    for (const auto input : node->inputs()) {
      registerClassDependencies(input->type());
    }
    for (const auto output : node->outputs()) {
      registerClassDependencies(output->type());
    }

    if (!print_const && node->kind() == prim::Constant)
      return;
    splitLongInlines(node->inputs());
    switch (node->kind()) {
      case prim::Return:
        if (enforce_importable_ && node->inputs().size() != 1) {
          throw script::ErrorReport(node->sourceRange())
              << "Exportable methods must have a single return value. "
              << "Normal use of ScriptMethods should enforce this.";
        }
        if (node->inputs().size() > 0) {
          indent();
          body_ << "return ";
          printValueList(body_, node->inputs());
          body_ << "\n";
        }
        break;
      case prim::Loop:
        printLoop(LoopView(node));
        break;
      case prim::If:
        printIf(IfView(node));
        break;
      case prim::TupleUnpack:
      case prim::ListUnpack:
        assignValuesToTheirUniqueNames(node->outputs());
        indent();
        // TupleUnpack(unpacked) turns into an assignment op that forces
        // the unpack to be inserted when parsed back in:
        // a, b, = unpacked
        // a, = unpacked # trailing comma forces an unpack to happen
        if (node->outputs().size() > 0) {
          printValueList(body_, node->outputs(), "", ", = ");
        }
        body_ << useOf(node->input()) << "\n";
        break;
      case prim::SetAttr: {
        const auto obj = node->inputs().at(0);
        const auto newVal = node->inputs().at(1);
        const auto type = obj->type()->expect<ClassType>();
        const auto& attrname = node->s(attr::name);
        indent();
        body_ << useOf(obj) << "." << attrname << " = " << useOf(newVal)
              << "\n";
      } break;
      case prim::fork: {
        // the subgraph gets emitted as another function
        auto name = genName("__forked_function");
        std::shared_ptr<Graph> graph = node->g(attr::Subgraph);
        indent();
        body_ << "def " << name << "():\n";
        for (size_t i = 0; i < node->inputs().size(); ++i) {
          assignValue(graph->inputs().at(i), node->inputs().at(i));
        }
        printBody(graph->block());
        std::stringstream ss;
        ss << "fork(" << name << ")";
        printOutputDefinition(node, ss.str());
      } break;
      case prim::Function: {
        if (enforce_importable_) {
          throw script::ErrorReport(node->sourceRange())
              << "closures are not exportable";
        }
        assignValuesToTheirUniqueNames(node->outputs());
        auto name = useOf(node->output());
        std::shared_ptr<Graph> graph = node->g(attr::Subgraph);
        indent();
        body_ << "def " << name << "(";
        assignValuesToTheirUniqueNames(graph->inputs());
        for (size_t i = 0; i < graph->inputs().size(); ++i) {
          Value* v = graph->inputs().at(i);
          if (i > 0) {
            body_ << ", ";
          }
          body_ << useOf(v) << ": " << v->type()->python_str();
        }
        body_ << "):\n";
        printBody(graph->block());
      } break;
      default:
        std::stringstream ss;
        printRHS(ss, node);

        // we prevent long constants from inlining here.
        // it is not safe to do the same thing for non-constants here
        // because of [reordering of inlines]
        if (output_inline_.count(node) == 0 ||
            (node->kind() == prim::Constant && isLongLine(ss.str()))) {
          printOutputDefinition(node, ss.str());
        } else {
          // this node is safe to inline, so assign the output value
          // to that expression directly
          assignValue(node->output(), ss.str());
        }
    }
  }

  void printMaybeAnnotatedConstantList(
      std::ostream& stmt,
      const char* the_type,
      size_t list_size,
      const IValue& the_list) {
    if (list_size == 0) {
      stmt << "annotate(List[" << the_type << "], [])";
    } else {
      stmt << the_list;
    }
  }

  void printConstant(std::ostream& stmt, const IValue& v) {
    if (v.isTensor()) {
      stmt << "CONSTANTS.c" << getOrAddTensorConstant(v.toTensor());
    } else if (v.isString()) {
      printQuotedString(stmt, v.toStringRef());
    } else if (v.isDevice()) {
      std::stringstream ss;
      ss << v.toDevice();
      stmt << "torch.device(";
      printQuotedString(stmt, ss.str());
      stmt << ")";
    } else if (v.isTensorList()) {
      stmt << "[";
      const char* delim = "";
      for (const at::Tensor& t : v.toTensorListRef()) {
        stmt << delim << "CONSTANTS.c" << getOrAddTensorConstant(t);
        delim = ", ";
      }
      stmt << "]";
    } else if (v.isBoolList()) {
      printMaybeAnnotatedConstantList(
          stmt, "bool", v.toBoolList().size(), v);
    } else if (v.isIntList()) {
      printMaybeAnnotatedConstantList(stmt, "int", v.toIntListRef().size(), v);
    } else if (v.isDoubleList()) {
      printMaybeAnnotatedConstantList(
          stmt, "float", v.toDoubleListRef().size(), v);
    } else {
      stmt << v;
    }
  }

  void printNone(std::ostream& stmt, const Node* node) {
    if (node->output()->type()->isSubtypeOf(NoneType::get())) {
      stmt << "None";
      return;
    }
    // XXX - when None has an Optional[T] type, we must ensure that type
    // can be recovered on parsing. It cannot be recovered if it will be
    // matched to schema with free variables. If it is used only in places
    // where there is schema and the scheme has no free variables, then we
    // can recover it without annotation. Otherwise, we annotate None with
    // the right optional type
    const auto& uses = node->output()->uses();
    bool all_usable_schema =
        std::all_of(uses.begin(), uses.end(), [](const Use& u) {
          if (auto schema = u.user->maybeSchema()) {
            if (u.offset >= schema->arguments().size()) {
              return false;
            }
            return !schema->arguments().at(u.offset).type()->hasFreeVariables();
          }
          return false;
        });

    if (all_usable_schema) {
      stmt << "None";
    } else {
      stmt << "annotate(" << node->output()->type()->python_str() << ", None)";
    }
  }

  // Prints the RHS value of a Node, e.g. `aten.add(x, y)`
  void printRHS(std::ostream& stmt, Node* node) {
    switch (node->kind()) {
      case prim::PythonOp: {
        auto value = static_cast<const PythonOp*>(node);
        if (enforce_importable_ && !value->ignore_on_export) {
          throw script::ErrorReport(node->sourceRange())
              << "Could not export Python function call '" << value->name()
              << "'. Remove calls to Python functions before export. "
              << "Did you forget add @script or @script_method annotation? "
              << "If this is a nn.ModuleList, add it to __constants__";
        }

        if (value->ignore_on_export) {
          stmt << "ops.prim.IgnoredPythonOp";
        } else {
          stmt << "^" << value->name();
          value->writeScalars(stmt);
        }
        printValueList(stmt, node->inputs(), "(", ")");
      } break;
      case prim::Uninitialized: {
        stmt << "uninitialized(" << node->output()->type()->python_str() << ")";
      } break;
      case prim::Constant: {
        if (node->kind() == prim::Constant && !node->mustBeNone()) {
          IValue v = toIValue(node->output()).value();
          printConstant(stmt, v);
        } else {
          printNone(stmt, node);
        }
      } break;
      case prim::ImplicitTensorToNum: {
        stmt << "annotate(" << node->output()->type()->python_str() << ", "
             << useOf(node->input()) << ")";
      } break;
      case prim::Int: {
        printValueList(stmt, node->inputs(), "int(", ")");
      } break;
      case prim::Float: {
        printValueList(stmt, node->inputs(), "float(", ")");
      } break;
      case prim::Bool: {
        printValueList(stmt, node->inputs(), "bool(", ")");
      } break;
      case prim::str: {
        printValueList(stmt, node->inputs(), "str(", ")");
      } break;
      case prim::Print: {
        printValueList(stmt, node->inputs(), "print(", ")");
      } break;
      case prim::TupleConstruct: {
        printValueList(
            stmt, node->inputs(), "(", node->inputs().size() == 1 ? ",)" : ")");
      } break;
      case prim::TupleIndex: {
        stmt << "(" << useOf(node->inputs().at(0)) << ")["
             << useOf(node->inputs().at(1)) << "]";
      } break;
      case prim::TupleSlice: {
        stmt << "(" << useOf(node->input()) << ")[" << node->i(attr::beg) << ":"
             << node->i(attr::end) << "]";
      } break;
      case prim::ListConstruct: {
        // when the list is empty and is not a list of tensors,
        // we need to annotate it, otherwise it won't be possible
        // to infer the type on import
        if (node->inputs().size() == 0 &&
            !node->output()->type()->isSubtypeOf(TensorType::get())) {
          stmt << "annotate(" << node->output()->type()->python_str()
               << ", [])";
        } else {
          printValueList(stmt, node->inputs(), "[", "]");
        }
      } break;
      case prim::DictConstruct: {
        auto dict_type = node->output()->type()->expect<DictType>();
        bool is_default_type =
            dict_type->getKeyType()->isSubtypeOf(StringType::get()) &&
            dict_type->getKeyType()->isSubtypeOf(TensorType::get());
        if (node->inputs().size() == 0 && !is_default_type) {
          stmt << "annotate(" << node->output()->type()->python_str()
               << ", {})";
        } else {
          printDict(stmt, node->inputs());
        }
      } break;
      case prim::DictIndex: {
        stmt << "(" << useOf(node->inputs().at(0)) << ")["
             << useOf(node->inputs().at(1)) << "]";
      } break;
      case prim::CreateObject: {
        const auto classType = node->output()->type()->expect<ClassType>();
        stmt << classType->python_str() << ".__new__("
             << classType->python_str() << ")";
      } break;
      case prim::GetAttr: {
        const auto obj = node->inputs().at(0);
        const auto classType = obj->type()->expect<ClassType>();
        const auto& field = node->s(attr::name);
        if (isValidIdentifier(field)) {
          stmt << useOf(obj) << "." << field;
        } else {
          stmt << "getattr(" << useOf(obj) << ", ";
          printQuotedString(stmt, field);
          stmt << ")";
        }
      } break;
      default: {
        Symbol kind = node->kind();
        if (kind.is_aten()) {
          // special case aten -> torch because we want to rename
          // the aten namespace, but this change will take more time
          // doing it here ensures we do not have fix up archives later
          stmt << "torch." << kind.toUnqualString() << "(";
        } else {
          stmt << "ops." << kind.ns().toUnqualString() << "."
               << kind.toUnqualString() << "(";
        }
        const FunctionSchema& schema = node->schema();
        for (size_t i = 0; i < node->inputs().size(); ++i) {
          if (i > 0) {
            stmt << ", ";
          }
          auto v = useOf(node->inputs().at(i));
          // print the kwarg name if it is a kwarg only argument.
          if (i < schema.arguments().size()) {
            auto arg = schema.arguments().at(i);
            if (arg.kwarg_only()) {
              stmt << arg.name() << "=";
            }
          } else {
            // vararg functions like format can have extra arguments
            AT_ASSERT(schema.is_vararg());
          }
          stmt << v;
        }
        stmt << ")";
      } break;
    }
  }

  std::ostream& printBlock(Block* root, bool block_has_other_statements) {
    // pythons weird 'pass' syntax creates a bunch of places where we have to
    // check if this block would be empty. But not everything in a block is a
    // node. Sometimes if, loop, and return statements will follow this block
    // and block_has_other_statements == true.
    if (!block_has_other_statements &&
        root->nodes().begin() == root->nodes().end()) {
      indent();
      body_ << "pass\n";
    }
    for (auto* node : root->nodes()) {
      printNode(node, /*print_const=*/false);
    }
    return body_;
  }

  void printDefaultValue(
      const TypePtr& typ,
      std::ostream& stmt,
      const IValue& value) {
    // xxx - many weak script modules store default values for broadcasting
    // lists that are not actually the same type as the argument. We can only
    // serialize default values that will implicitly convert to their declared
    // return type since we do not need to serialize these built-in modules with
    // their defaults, we just drop them for now.
    if (typ->kind() == ListType::Kind &&
        (value.isInt() || value.isDouble() || value.isBool())) {
      return;
    }
    stmt << "=";
    printConstant(stmt, value);
  }
  void printBody(Block* body) {
    // we always print constants at the top of the function, in the order
    // in which they are used.
    std::vector<Node*> constants;
    buildConstantList(body, constants);

    // current graph is used to de-dup names within a single graph
    scanBlock(body);
    {
      auto guard = WithIndented();
      // Print initial constant table (most are just inlined into their use,
      // but some like long strings do get emitted)
      for (Node* n : constants) {
        printNode(n, /*print_const=*/true);
      }
      // Print body
      printBlock(body, body->return_node()->inputs().size() > 0);
      printNode(body->return_node(), /*print_const=*/false);
    }
  }

 public:
  void printFunction(const Function& func) {
    const FunctionSchema& schema = func.getSchema();
    Graph& graph = *func.graph();
    used_names_.clear(); // each graph can reuse local names

    indent();
    body_ << "def " << func.name() << "(";
    auto param_it = graph.inputs().begin();
    for (const Argument& arg : schema.arguments()) {
      std::string arg_name = genName(arg.name());
      if (param_it == graph.inputs().begin()) {
        // the first argument may omit its type when it is implied by context
        // the flag is_method_ determines when to do this
        body_ << arg_name;
        if (!is_method_) {
          body_ << ": " << arg.type()->python_str();
        }
      } else {
        body_ << ",\n    " << arg_name << ": " << arg.type()->python_str();
      }
      if (arg.default_value()) {
        printDefaultValue(arg.type(), body_, *arg.default_value());
      }
      assignValue(*param_it++, arg_name);
    }

    body_ << ") -> " << resultType(graph)->python_str() << ":\n";
    printBody(graph.block());
  }

  std::string getImports() {
    std::ostringstream ret;
    std::unordered_set<std::string> already_printed;
    for (const auto& c : class_deps_) {
      if (already_printed.count(c->qualifier())) {
        continue;
      }
      ret << "import " << c->qualifier() << "\n";
      already_printed.insert(c->qualifier());
    }
    return ret.str();
  }

  PythonPrintPass(
      std::vector<at::Tensor>& tensor_table,
      std::vector<ClassTypePtr>& class_table,
      bool enforce_importable,
      bool is_method)
      : tensor_table_(tensor_table),
        class_table_(class_table),
        enforce_importable_(enforce_importable),
        is_method_(is_method) {}

  // TODO: we should consider forcing functions to return a single value
  // instead of handling this tuple logic both in the compiler and the printer
  TypePtr resultType(const Graph& graph) {
    if (graph.outputs().size() == 1) {
      return graph.outputs().at(0)->type();
    } else {
      return TupleType::create(
          fmap(graph.outputs(), [&](const Value* v) { return v->type(); }));
    }
  }

  void printCompilationUnit(const script::CompilationUnit& cu) {
    for (auto& func : cu.get_functions()) {
      printFunction(*func);
    }
  }

  void printClass(const ClassTypePtr& classType) {
    body_ << "class " << classType->basename() << ":\n";
    {
      const auto guard = WithIndented();
      for (auto& method : classType->methods()) {
        printFunction(*method);
      }
    }
    // remove `classType` from the list of deps
    class_deps_.erase(
        std::remove(class_deps_.begin(), class_deps_.end(), classType),
        class_deps_.end());
  }

  void print(std::ostream& out) {
    out << getImports() << body_.str();
  }
};

void PythonPrint(
    std::ostream& out,
    const Function& func,
    bool is_method,
    std::vector<at::Tensor>& tensor_table,
    std::vector<ClassTypePtr>& class_table,
    bool enforce_importable) {
  PythonPrintPass pp(tensor_table, class_table, enforce_importable, is_method);
  pp.printFunction(func);
  pp.print(out);
}

void PythonPrint(
    std::ostream& out,
    const script::CompilationUnit& cu,
    bool is_method,
    std::vector<at::Tensor>& tensor_table,
    std::vector<ClassTypePtr>& class_table,
    bool enforce_importable) {
  PythonPrintPass pp(tensor_table, class_table, enforce_importable, is_method);
  pp.printCompilationUnit(cu);
  pp.print(out);
}

void PythonPrint(
    std::ostream& out,
    const ClassTypePtr& classType,
    std::vector<at::Tensor>& tensor_table,
    std::vector<ClassTypePtr>& class_table,
    bool enforce_importable) {
  PythonPrintPass pp(tensor_table, class_table, enforce_importable, true);
  pp.printClass(classType);
  pp.print(out);
}

bool printerHasSpecialCaseFor(Symbol sym) {
  // WARNING: by adding a value to this set, you are asserting
  // that you have also added special handling of this symbol to
  // the printer above. Not adding handling will cause import and export
  // of modules with this new operator to fail. This is only required
  // for operators without schema. Prefer registering your operator with
  // schema to editing this list here. These cases should only be things
  // that require special handling because they do not fit normal schema
  const static std::unordered_set<Symbol> handled = {
      prim::Constant,
      prim::Uninitialized,
      prim::fork,
      prim::ListConstruct,
      prim::DictConstruct,
      prim::ListUnpack,
      prim::Print,
      prim::PythonOp,
      prim::TupleConstruct,
      prim::TupleIndex,
      prim::DictIndex,
      prim::TupleSlice,
      prim::TupleUnpack,
      prim::CreateObject,
      prim::GetAttr,
      prim::SetAttr,
  };

  // WARNING: by adding a value to this set, you are asserting that your
  // primitive is only ever added during optimization and does not need
  // to be correctly printed for export (a process that happens before
  // optimization passes run)
  const static std::unordered_set<Symbol> unneeded = {
      c10::onnx::Reshape, // only used in onnx
      c10::onnx::Shape, // only used in onnx
      prim::AutogradZero, // temporarily inserted by autograd
      prim::AutogradAnyNonZero, // temporarily inserted by autograd
      prim::AutogradAdd, // temporarily inserted by autograd
      prim::ConstantChunk, // optimization pass adds it
      prim::DifferentiableGraph, // optimization pass adds it
      prim::BroadcastSizes, // optimization pass (fuser) adds it
      prim::ChunkSizes, // optimization pass (fuser) adds it
      prim::Drop, // used in interpreter only
      prim::FusedConcat, // optimization pass adds it
      prim::FusionGroup, // optimization pass adds it
      prim::Load, // used in interpreter only
      prim::MMTreeReduce, // used as an optimization
      prim::MMBatchSide, // used as an optimization
      prim::Store, // used in interpreter only
      prim::profile, // used in interpreter only

  };

  // These namespaces are required to have Python printers unless
  // otherwise noted in unneeded.
  const static std::unordered_set<Symbol> required_namespaces = {
      c10::namespaces::prim,
      c10::namespaces::aten,
      c10::namespaces::onnx,
  };

  return handled.count(sym) || unneeded.count(sym) ||
      !required_namespaces.count(sym.ns());
}
} // namespace jit
} // namespace torch
