#include "torch/csrc/jit/passes/pretty_print.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/generic_if.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

static std::ostream& indent(std::ostream& out, size_t level) {
  for (size_t i = 0; i < level; ++i) {
    out << "  ";
  }
  return out;
}

class PrettyPrintPass {
  const Graph& graph_;

  // When printing a name if there is a conflict with an existing name in the
  // graph, record the value -> new generated name mapping
  std::unordered_map<const Value*, const Value*> aliases_;

  // The Graph already tracks unique_names_, this is just for additional ones
  // generated during printing
  std::unordered_map<std::string, const Value*> generated_names_;

  // Cache of value names
  std::unordered_map<const Value*, std::string> value_names_;

  // Nodes that were skipped to be printed later
  std::unordered_set<const Node*> skipped_nodes_;

  template <class T>
  void zipWith(
      at::ArrayRef<T> list_a,
      at::ArrayRef<T> list_b,
      std::function<void(T, T)> action) const {
    auto it_a = list_a.begin();
    auto it_b = list_b.begin();

    if (list_a.size() != list_b.size()) {
      AT_ERROR("Pretty printer expected 2 lists of same size");
    }

    for (; it_a != list_a.end(); ++it_a, ++it_b) {
      action(*it_a, *it_b);
    }
  }

  std::ostream& printValueList(
      std::ostream& out,
      at::ArrayRef<const Value*> list) {
    out << "(";
    auto delimiter = "";
    for (const auto* value : list) {
      out << delimiter;
      printValue(out, value);
      delimiter = ", ";
    }
    out << ")";
    return out;
  }

  void printAssignment(
      std::ostream& out,
      const Value* lhs,
      const Value* rhs,
      const size_t level) {
    indent(out, level);
    printValue(out, lhs);
    out << " = ";
    printValue(out, rhs);
    out << "\n";
  }

  std::ostream& printIf(
      std::ostream& out,
      const Node* node,
      const size_t level) {
    indent(out, level);
    out << "if ";
    const auto if_block = node->blocks()[0];
    const auto else_block = node->blocks()[1];
    printValue(out, node->inputs()[0]);
    out << ":"
        << "\n";

    // Print node contents
    printBlock(out, if_block, level + 1);

    // Print if block output
    zipWith<const Value*>(
        node->outputs(),
        if_block->outputs(),
        [&](const Value* node_output, const Value* return_input) {
          printAssignment(out, node_output, return_input, level + 1);
        });

    indent(out, level);
    out << "else:\n";
    printBlock(out, else_block, level + 1);
    zipWith<const Value*>(
        node->outputs(),
        else_block->outputs(),
        [&](const Value* node_output, const Value* return_input) {
          printAssignment(out, node_output, return_input, level + 1);
        });

    return out;
  }

  std::ostream& printLoop(
      std::ostream& out,
      const Node* node,
      const size_t level) {
    // Prints assignments between the loop node and body block around the
    // loop body itself in the following manner:
    //
    // (silently) alias block input names to node output names
    // assign each node input to the corresponding node output
    // assign condition to loop condition value
    // while ...:
    //    print loop body nodes
    //    assign each block output to the corresponding block input

    const auto body_block = node->blocks()[0];
    // Add aliases for loop-carried dependencies
    zipWith<const Value*>(
        body_block->inputs().slice(1), // Start at 1 to ignore trip count
        node->outputs(),
        [&](const Value* block_input, const Value* node_output) {
          aliases_[block_input] = node_output;
        });

    // Print initial assignments of loop node outputs = loop node inputs
    zipWith<const Value*>(
        node->outputs(),
        node->inputs().slice(2),
        [&](const Value* node_output, const Value* node_input) {
          printAssignment(out, node_output, node_input, level);
        });

    // Print condition initial assignment
    printAssignment(out, body_block->inputs()[0], node->inputs()[1], level);

    // Loop header
    indent(out, level);
    out << "while ";
    printValue(out, body_block->inputs()[0]);
    out << ":\n";

    // Loop body
    printBlock(out, body_block, level + 1);

    // Update block outputs to block inputs for next loop iteration
    zipWith<const Value*>(
        body_block->inputs(),
        body_block->outputs(),
        [&](const Value* block_input, const Value* block_output) {
          printAssignment(out, block_input, block_output, level + 1);
        });
    return out;
  }

  // Returns false if the node has no outputs, or if outputs are used anywhere
  // except in a single prim::Return node
  bool nodeOnlyOutputReturns(const Node* node) {
    if (node->outputs().size() == 0) {
      return false;
    }
    for (const auto* output : node->outputs()) {
      if (output->uses().size() != 1) {
        return false;
      }
      if (output->uses()[0].user->kind() != prim::Return) {
        return false;
      }
    }
    return true;
  }

  std::ostream& printNode(
      std::ostream& out,
      const Node* node,
      const size_t level) {
    switch (node->kind()) {
      case prim::Return:
        break;
      case prim::Constant:
        break;
      case prim::Loop:
        printLoop(out, node, level);
        break;
      case prim::If:
        printIf(out, node, level);
        break;
      default:
        if (nodeOnlyOutputReturns(node)) {
          // This node is assigned to a temp which is then used by prim::Return,
          // so just print this node there
          skipped_nodes_.insert(node);
          return out;
        }

        indent(out, level);
        // Print outputs
        if (node->outputs().size() > 0) {
          auto delim = "";
          for (const auto* output_value : node->outputs()) {
            out << delim;
            printValue(out, output_value);
            delim = ", ";
          }
          out << " = ";
        }

        printRHS(out, node);

        out << "\n";
    }

    return out;
  }

  // Prints the RHS value of a Node, e.g. `aten::add(x, y)`
  std::ostream& printRHS(std::ostream& out, const Node* node) {
    IR_IFM_CONST(node, PythonOp)
    out << "^" << value->name();
    value->writeScalars(out);
    IR_ELSE()
    out << node->kind().toQualString();
    IR_END()

    // Print instruction parameters
    printValueList(out, node->inputs());
    return out;
  }

  std::ostream& printReturn(
      std::ostream& out,
      const Node* node,
      const size_t level) {
    indent(out, level);
    const auto& returns = node->inputs();
    if (returns.size() > 0) {
      out << "return ";
      if (returns.size() > 1) {
        printValueList(out, returns);
      } else {
        printValue(out, returns[0]);
      }
      out << "\n";
    }
    return out;
  }

  std::ostream& printBlock(
      std::ostream& out,
      const Block* root,
      const size_t level) {
    for (const auto* node : root->nodes()) {
      printNode(out, node, level);
    }

    printNode(out, root->return_node(), level);

    return out;
  }

  inline bool isNameUnique(std::string& name, const Value* val) const {
    auto generated_name_value = generated_names_.find(name);
    if (generated_name_value != generated_names_.end() &&
        generated_name_value->second != val) {
      // Found a generated name match, check that it's for a different value
      return false;
    }
    return graph_.uniqueNames().find(name) == graph_.uniqueNames().end();
  }

  std::ostream& printValue(std::ostream& out, const Value* val) {
    auto cached_name = value_names_.find(val);
    if (cached_name != value_names_.end()) {
      // If this value has been seen before, print out cached name
      out << cached_name->second;
      return out;
    }

    const auto node = val->node();
    if (node->kind() == prim::Constant) {
      // printAttributeValue(out, node->attributeNames()[0], node);
      node->printValue(out, node->attributeNames()[0]);
      return out;
    }

    if (skipped_nodes_.count(node) > 0) {
      skipped_nodes_.erase(node);
      // Node was skipped earlier, so print it now
      printRHS(out, node);
      return out;
    }

    auto name_source = val;

    auto aliased_name = aliases_.find(val);
    if (aliased_name != aliases_.end()) {
      name_source = aliased_name->second;
    }

    auto name = name_source->uniqueName();

    bool using_generated_name = false;
    if (isdigit(name.at(0))) {
      std::stringstream ss;
      ss << "t" << name;
      name = ss.str();
      using_generated_name = true;
    } else if (name.find_last_of('.') != std::string::npos) {
      // Make unique name a valid variable name (e.g. a.1 -> a1)
      name.erase(std::remove(name.begin(), name.end(), '.'), name.end());
      using_generated_name = true;
    }

    if (using_generated_name) {
      // Make sure name is unique
      size_t suffix = 0;
      while (!isNameUnique(name, name_source)) {
        std::stringstream ss;
        ss << name << suffix;
        name = ss.str();
        ++suffix;
      }

      // These names aren't in the Graph's list of names but we still need to
      // make sure there are no name conflicts
      generated_names_[name] = name_source;
    }

    value_names_[val] = name;
    out << name;
    return out;
  }

 public:
  PrettyPrintPass(const Graph& graph) : graph_(graph) {}

  std::ostream& run(std::ostream& out) {
    out << "def script";
    const Node* params = graph_.block()->param_node();
    printValueList(out, params->outputs());
    out << ":\n";

    // Print body
    printBlock(out, graph_.block(), 1);

    printReturn(out, graph_.block()->return_node(), 1);

    return out;
  }
};

TORCH_API std::ostream& PrettyPrint(std::ostream& out, const Graph& graph) {
  return PrettyPrintPass(graph).run(out);
}

} // namespace jit
} // namespace torch
