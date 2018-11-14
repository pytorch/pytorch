#include "torch/csrc/jit/passes/python_print.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/generic_if.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/script/error_report.h"

namespace torch {
namespace jit {


class PythonPrintPass {
  std::ostream& out;

  // constants are written to this table, and given then named CONSTANTS.cN
  // where N is the index into this table.

  std::vector<at::Tensor> tensor_constants;
  // When printing this node, is it safe to write it inline (i.e. without
  // assigning a temporary variable
  std::unordered_set<const Node*> output_inline_;

  // when we print this, should we error if the resulting output would
  // not be able to be reparsed?
  bool enforce_importable_;

  // what valid identifiers are in use for the current function
  std::unordered_set<std::string> used_names_;

  // for fork,
  // subgraphs get added to the worklist, and will be printed later
  std::vector<std::pair<const Graph*, std::string>> worklist;

  // scanValue, scanNode, scanBlock:
  // decide if it is safe to omit the output of a temporary variable,
  // and inline the expression into its use
  // we only do this if
  // (1) it is a constant, or
  // (2) the temporary is unnamed, is single output, is used once,
  //     and would appear in the same order when the expression tree is reparsed.
  // The last case can be checked
  // becuase when we emit a expresion tree in the parser,
  // we do a left-to-right postorder traversal of the expression tree (emit children, then emit op).
  // The reverse of this is a right-to-left preorder traversal of the tree.
  // By doing a right-to-left preorder traversal of the inputs of a node,
  // while also scanning the list of emitted nodes backward, we can see if
  // they line up with what would happen when parsed the node as an expression. While they line
  // up we collapse them into an inline expression.

  // The inductive step is that the right-most input should be produced by the node
  // immediatly before the current node if it is in tree order.

  // block_point is the current node in the reverse linear scan of the emitted nodes
  // v is the current value in the tree traversal that may match with block_point's output.
  const Node* scanValue(const Node* block_point, const Value* v) {
    const Node* n = v->node();
    JIT_ASSERT(n->kind() == prim::Constant || output_inline_.count(n) == 0);

    if (n == block_point && // the node must be at the expected point of the typical tree traversal
        n->outputs().size() == 1 && // there must be only 1 values, otherwise we need an assignment to handle the multiple outout values
        v->uses().size() == 1 && // if it is used more than once, then we need a variable
        n->blocks().size() == 0 && // don't try to inline control blocks,
        !v->hasUniqueName() && // if it has a name set, then it was written as a variable so preserve that
        (v->uses().at(0).user->kind() != prim::Loop // if it is a loop-carried input, we need a variable
         || v->uses().at(0).offset < 2)) {          // otherwise the condition or trip count may be emitted in the wrong order w.r.t. to it
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

  const Node* scanNode(const Node* n) {
    // don't bother to scan nodes we have already determined to be inline
    if(output_inline_.count(n)) {
      return n;
    }
    for(auto b : n->blocks()) {
      scanBlock(b);
    }
    const Node* block_point = n->prev();
    for(auto it = n->inputs().rbegin(),
             end = n->inputs().rend(); it != end; ++it) {
      block_point = scanValue(block_point, *it);
    }
    return block_point;
  }

  void scanBlock(const Block* b) {
    scanNode(b->return_node());
    for(auto node : b->nodes().reverse()) {
      scanNode(node);
    }
  }

  // get a new name unique across calls to uniqueName() and
  // anything we have used.
  size_t next_id = 0;
  std::string genName(const std::string& candidate) {
    // some names are valid identifiers but off limits because
    // they are keywords or namespaces used in the output
    const static std::unordered_set<std::string> reserved_names = {
      // identifiers in the environment while parsing
      "aten",
      "prim",
      "CONSTANTS",
      "fork",
      "attribute",
      // the python keywords
      "False",
      "None",
      "True",
      "and",
      "as",
      "assert",
      "break",
      "class",
      "continue",
      "def",
      "del",
      "elif",
      "else",
      "except",
      "finally",
      "for",
      "from",
      "global",
      "if",
      "import",
      "in",
      "is",
      "lambda",
      "nonlocal",
      "not",
      "or",
      "pass",
      "raise",
      "return",
      "try",
      "while",
      "with",
      "yield",
    };

    std::string name = candidate;
    while(used_names_.count(name) || reserved_names.count(name)) {
      name = candidate + std::to_string(next_id++);
    }
    used_names_.insert(name);
    return name;
  }

  // unique names might not be valid identifiers,
  // force them to be by rewriting them
  static std::string makeValidIdentifier(const std::string& candidate) {
    std::stringstream ss;
    if (candidate.size() == 0 || isdigit(candidate[0]))
      ss << "_";
    for(char c : candidate) {
      if (isupper(c) || islower(c) || isdigit(c) || c == '_')
        ss << c;
      else
        ss << '_';
    }
    return ss.str();
  }
  // if we have to assign 'v' a name, what should it be?
  // use the uniqueName if it was set, otherwise generate a name.
  std::string genUniqueNameFor(const Value* v) {
    return genName(
        v->hasUniqueName() ? makeValidIdentifier(v->uniqueName()) : "t");
  }

  // map from Value to how it should be printed at each use
  std::unordered_map<const Value*, std::string> value_names_;

  std::string useOf(const Value* v) const {
    return value_names_.at(v);
  }
  void assignValue(const Value* v, const std::string& s) {
    value_names_[v] = s;
  }
  void assignValue(const Value* v, const Value* w) {
    assignValue(v, useOf(w));
  }
  void assignValuesToTheirUniqueNames(at::ArrayRef<const Value*> values) {
    for(auto v : values) {
      assignValue(v, genUniqueNameFor(v));
    }
  }

  size_t level = 0;
  // indent to the current indent level
  std::ostream& indent() {
    for (size_t i = 0; i < level; ++i) {
      out << "  ";
    }
    return out;
  }

  ResourceGuard WithIndented() {
    level++;
    return ResourceGuard([this]{
      level--;
    });
  }

  template <class T0, class T1, class F>
  void zipWith(
      at::ArrayRef<T0> list_a,
      at::ArrayRef<T1> list_b,
      F action) const {
    auto it_a = list_a.begin();
    auto it_b = list_b.begin();

    if (list_a.size() != list_b.size()) {
      AT_ERROR("Pretty printer expected 2 lists of same size");
    }

    for (; it_a != list_a.end(); ++it_a, ++it_b) {
      action(*it_a, *it_b);
    }
  }

  void printValueList(std::ostream& stmt, at::ArrayRef<const Value*> list, const char* begin = "", const char* end = "") {
    stmt << begin;
    auto delimiter = "";
    for (const auto* value : list) {
      stmt << delimiter;
      stmt << useOf(value);
      delimiter = ", ";
    }
    stmt << end;
  }

  void printAssignment(
      at::ArrayRef<const Value*> lhs,
      at::ArrayRef<const Value*> rhs) {
    if(lhs.size() > 0) {
      indent();
      printValueList(out, lhs);
      out << " = ";
      printValueList(out, rhs);
      out << "\n";
    }
  }

  void printIf(
      const Node* node) {
    assignValuesToTheirUniqueNames(node->outputs());
    auto cond = node->inputs()[0];
    const auto if_block = node->blocks()[0];
    const auto else_block = node->blocks()[1];
    indent() << "if " << useOf(cond) << ":\n";
    {
      auto guard = WithIndented();
      // Print node contents
      printBlock(if_block);
      printAssignment(node->outputs(), if_block->outputs());
    }
    indent() << "else:\n";
    {
      auto guard = WithIndented();
      printBlock(else_block);
      printAssignment(node->outputs(), else_block->outputs());
    }
  }

  // our way of encoding loops makes them difficult to turn back into python syntax.
  // we have to check properties of the condition and trip count inputs to
  // figure out which one it initially was
  static bool shouldEmitAsForLoop(const Node* node) {
      const auto body_block = node->blocks()[0];
      auto trip_count = toIValue(node->inputs().at(0));
      auto cond_input = toIValue(node->inputs().at(1));
      auto cond_next = toIValue(body_block->outputs().at(0));

      bool condition_is_always_true = cond_input && cond_input->toBool() && cond_next &&
        cond_next->toBool();
      bool trip_count_is_specified = !trip_count || // trip is not a constant
          trip_count->toInt() != std::numeric_limits<int64_t>::max() || // it is a constant but not the default one
          body_block->inputs().at(0)->uses().size() > 0; // it is actually being used in the body.

      if (condition_is_always_true) {
        // if the trip count was not specified this was a user-written while True:
        return trip_count_is_specified;
      } else {
        // this must be a while loop, but check that there isn't _also_ a trip count
        if (trip_count_is_specified) {
          throw script::ErrorReport(node->getSourceLocation())
              << "loop cannot be printed as python because it has gone through an optimization "
              << "that combined while and for loops. File a bug.";
        }
        return false;
      }
  }

  void printLoop(const Node* node) {

    // Loop carried dependencies are handled by assigning their initial
    // values to the node->outputs() before the loop,
    // and assign node->outputs() to the new values at the end of each trip.


    bool emit_as_for_loop = shouldEmitAsForLoop(node);
    const auto body_block = node->blocks()[0];

    assignValuesToTheirUniqueNames(node->outputs());
    // Add aliases for loop-carried dependencies
    zipWith(
        body_block->inputs().slice(1), // Start at 1 to ignore trip count
        node->outputs(),
        [&](const Value* block_input, const Value* node_output) {
          assignValue(block_input, node_output);
        });

    // Print initial assignments of loop node outputs = loop node inputs
    printAssignment(node->outputs(), node->inputs().slice(2));

    auto trip_count_in_block = body_block->inputs().at(0);
    assignValuesToTheirUniqueNames(trip_count_in_block);
    // Loop header
    if (emit_as_for_loop) {
      indent();
      out << "for " << useOf(trip_count_in_block) << " in range("
          << useOf(node->inputs().at(0)) << "):\n";
    } else {
      // note: trip_count_in_block is unused because this is a while loop,
      // so we reuse the Value* as a stand-in for the loop condition
      printAssignment(trip_count_in_block, node->inputs().at(1));
      indent();
      out << "while " << useOf(trip_count_in_block) << ":\n";
    }
    // Loop body
    {
      ResourceGuard indent = WithIndented();
      printBlock(body_block);
      // Update block outputs to block inputs for next loop iteration
      // skip the assignment to the new condition in for loops because
      // the condition is always True
      size_t offset = emit_as_for_loop ? 1 : 0;
      printAssignment(body_block->inputs().slice(offset), body_block->outputs().slice(offset));
    }
  }

  void printNode(const Node* node) {
    switch (node->kind()) {
      case prim::Return:
        if (node->inputs().size() > 0) {
          indent();
          out << "return ";
          printValueList(out, node->inputs());
          out << "\n";
        }
        break;
      case prim::Loop:
        printLoop(node);
        break;
      case prim::If:
        printIf(node);
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
          printValueList(out, node->outputs(), "", ", = ");
        }
        out << useOf(node->input()) << "\n";
        break;
      default:

        std::stringstream ss;
        printRHS(ss, node);

        // this node is safe to inline, so assign the output value
        // to that expression directly
        // guard against really long lines
        if (output_inline_.count(node) > 0 && ss.str().size() + level * 2 < 40) {
          assignValue(node->output(), ss.str());
          return;
        }
        assignValuesToTheirUniqueNames(node->outputs());
        indent();
        // Print outputs
        if (node->outputs().size() > 0) {
          printValueList(out, node->outputs());
          out << " = ";
        }
        out << ss.str() << "\n";
    }
  }

  size_t addTensorConstant(at::Tensor t) {
    tensor_constants.emplace_back(std::move(t));
    return tensor_constants.size() - 1;
  }

  void printMaybeAnnoatatedConstantList(
      std::ostream& stmt,
      const char* the_type,
      size_t list_size,
      IValue the_list) {
    if(list_size == 0) {
      stmt << "annotate(" << the_type << ", [])";
    } else {
      stmt << the_list;
    }
  }

  // unix isprint but insensitive to locale
  static bool isPrint(char s) {
    return s > 0x1f && s < 0x7f;
  }

  void printQuotedString(std::ostream& stmt, const std::string& str) {
    stmt << "\"";
    for(auto s : str) {
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
            buf[2] += s % 8; s /= 8;
            buf[1] += s % 8; s /= 8;
            buf[0] += s;
            stmt << "\\" << buf;
          }
          break;
      }
    }
    stmt << "\"";
  }

  // Prints the RHS value of a Node, e.g. `aten.add(x, y)`
  void printRHS(std::ostream& stmt, const Node* node) {
    switch(node->kind()) {
      case PythonOp::Kind: {
        auto value = static_cast<const PythonOp*>(node);
        if (enforce_importable_) {
          throw script::ErrorReport(node->getSourceLocation())
              << "could not export python function call " << value->name()
              << ". Remove calls to python functions before export.";
        }

        stmt << "^" << value->name();
        value->writeScalars(stmt);
        printValueList(stmt, node->inputs(), "(", ")");
      } break;
      case prim::Constant: {
        IValue v = toIValue(node->output()).value();
        if(v.isTensor()) {
          stmt << "CONSTANTS.c" << addTensorConstant(std::move(v).toTensor());
        } else if(v.isString()) {
          printQuotedString(stmt, v.toStringRef());
        } else if(v.isTensorList()) {
          auto tl = v.toTensorListRef();
          stmt << "[";
          const char* delim = "";
          for(at::Tensor t : tl) {
            stmt << delim << "CONSTANTS.c" << addTensorConstant(std::move(t));
            delim = ", ";
          }
          stmt << "]";
        } else if(v.isBoolList()) {
          printMaybeAnnoatatedConstantList(stmt, "bool", v.toBoolListRef().size(), v);
        } else if(v.isIntList()) {
          printMaybeAnnoatatedConstantList(stmt, "int", v.toIntListRef().size(), v);
        } else if(v.isDoubleList()) {
          printMaybeAnnoatatedConstantList(stmt, "float", v.toDoubleListRef().size(), v);
        } else {
          stmt << v;
        }
      } break;
      case prim::None:
      case prim::NoneGenerator:
      case prim::Undefined: {
        stmt << "None";
      } break;
      case prim::FloatToInt: {
        printValueList(stmt, node->inputs(), "int(", ")");
      } break;
      case prim::StringToFloat:
      case prim::IntToFloat: {
        printValueList(stmt, node->inputs(), "float(", ")");
      } break;
      case prim::TensorToBool: {
        printValueList(stmt, node->inputs(), "bool(", ")");
      } break;
      case prim::Print: {
        printValueList(stmt, node->inputs(), "print(",")");
      } break;
      case prim::TupleConstruct: {
        printValueList(
            stmt, node->inputs(), "(", node->inputs().size() == 1 ? ",)" : ")");
      } break;
      case prim::TupleIndex: {
        stmt << "(" << useOf(node->input()) << ")[" << node->i(attr::index) << "]";
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
            !node->output()->type()->isSubtypeOf(DynamicType::get())) {
          stmt << "annotate(" << node->output()->type()->python_str() << ", [])";
        } else {
          printValueList(stmt, node->inputs(), "[", "]");
        }
      } break;
      case prim::fork: {
        // the subgraph gets emitted as another function
        auto name = genName("__forked_function");
        std::shared_ptr<Graph> graph = node->g(attr::Subgraph);
        worklist.emplace_back(graph.get(), name);
        // and we put a call to fork which invokes that function.
        stmt << "fork(self." << name;
        for(const Value* v : node->inputs()) {
          stmt << ", " << useOf(v);
        }
        stmt << ")";
      } break;
      default: {
        Symbol kind = node->kind();
        stmt << kind.ns().toUnqualString() << "." << kind.toUnqualString() << "(";
        const FunctionSchema& schema = node->schema();
        for (size_t i = 0; i < schema.arguments().size(); ++i) {
            auto v = useOf(node->inputs().at(i));
            auto arg = schema.arguments().at(i);
            if (i > 0) {
              stmt << ", ";
            }
            if (arg.kwarg_only()) {
              stmt << arg.name() << "=";
            }
            stmt << v;
        }
        stmt << ")";
      } break;
    }
  }

  std::ostream& printBlock(
      const Block* root) {
    for (const auto* node : root->nodes()) {
      printNode(node);
    }
    return out;
  }

  void printOneFunction(const Graph& graph, const std::string& name) {
    used_names_.clear(); // each graph can reuse local names
    // current graph is used to de-dup names within a single graph
    scanBlock(graph.block());
    assignValuesToTheirUniqueNames(graph.inputs());
    out << "def " << name << "(self";
    for(auto input : graph.inputs()) {
      out << ",\n    " << useOf(input) << ": " << input->type()->python_str();
    }
    out << ") -> " << resultType(graph)->python_str() << ":\n";
    {
      auto guard = WithIndented();
      // Print body
      printBlock(graph.block());
      printNode(graph.block()->return_node());
    }
  }

 public:
  PythonPrintPass(
      std::ostream& out_,
      bool enforce_importable = false)
      : out(out_), enforce_importable_(enforce_importable) {}

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

  void printFunction(const Graph& graph, const std::string& name) {
    printOneFunction(graph, name);
    while(!worklist.empty()) {
      out << "\n\n";
      auto work = worklist.back();
      worklist.pop_back();
      printOneFunction(*work.first, work.second);
    }
  }
};

TORCH_API std::ostream& PythonPrint(std::ostream& out, const Graph& graph) {
  PythonPrintPass(out).printFunction(graph, "script");
  return out;
}

TORCH_API bool printerHasSpecialCaseFor(Symbol sym) {
  // WARNING: by adding a value to this set, you are asserting
  // that you have also added special handling of this symbol to
  // the printer above. Not adding handling will cause import and export
  // of modules with this new operator to fail. This is only required
  // for operators without schema. Prefer registering your operator with
  // schema to editing this list here. These cases should only be things
  // that require special handling because they do not fit normal schema
  const static std::unordered_set<Symbol> handled = {
    prim::BoolToTensor,
    prim::Constant,
    prim::TensorToBool,
    prim::FloatToInt,
    prim::fork,
    prim::IntToFloat,
    prim::ListConstruct,
    prim::ListUnpack,
    prim::None,
    prim::NoneGenerator,
    prim::Print,
    prim::PythonOp,
    prim::StringToFloat,
    prim::TupleConstruct,
    prim::TupleIndex,
    prim::TupleSlice,
    prim::TupleUnpack,
    prim::Undefined,
  };

  // WARNING: by adding a value to this set, you are asserting that your
  // primitive is only ever added during optimization and does not need
  // to be correctly printed for export (a process that happens before
  // optimization passes run)
  const static std::unordered_set<Symbol> unneeded = {
    onnx::Reshape, // only used in onnx
    onnx::Shape, // only used in onnx
    prim::AnyDefined, // temporarily inserted by autograd
    prim::AutogradAdd, // temporarily inserted by autograd
    prim::ConstantChunk, // optimization pass adds it
    prim::DifferentiableGraph, // optimization pass adds it
    prim::Drop, // used in interpreter only
    prim::FusedConcat, // optimization pass adds it
    prim::FusionGroup, // optimization pass adds it
    prim::Load, // used in interpreter only
    prim::MMTreeReduce, // used in batched execution only
    prim::Store, // used in interpreter only

  };

  return handled.count(sym) || unneeded.count(sym);
}

} // namespace jit
} // namespace torch
