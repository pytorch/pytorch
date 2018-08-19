#include "ir.h"


#include "torch/csrc/jit/operator.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/script/compiler.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stack>
#include <sstream>
#include <algorithm>
#include <string>

namespace torch { namespace jit {

// Sigh, see https://stackoverflow.com/questions/8016780/undefined-reference-to-static-constexpr-char
constexpr Symbol PythonOp::Kind;

constexpr int max_tensor_display_size = 10;

void printValueRef(std::ostream & out, const Value * n) {
  out << "%" << n->uniqueName();
}

template <typename T>
std::ostream& operator<<(std::ostream & out, const std::vector<T> & nodes) {
  out << at::ArrayRef<T>{nodes};
  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream & out, const at::ArrayRef<T> & nodes) {
  size_t i = 0;
  for(auto n : nodes) {
    if(i++ > 0)
      out << ", ";
    printValueRef(out, n);
  }
  return out;
}

struct const_value_list_with_types {
  const ArrayRef<const Value*> values;
  bool use_newlines;
  const_value_list_with_types(ArrayRef<const Value*> values, bool use_newlines = false)
    : values(values), use_newlines(use_newlines) {}
};
std::ostream& operator<<(std::ostream & out, const_value_list_with_types l) {
  size_t i = 0;
  size_t prev_stage = 0;
  for(auto n : l.values) {
    if(i++ > 0) {
      if (l.use_newlines) {
        // TODO: Indent here is hard-coded for "graph(": un-hard-code it
        out << "\n      ";
        if (n->stage() != prev_stage) {
          out << "-------- stage " << n->stage() << " --------\n      ";
          prev_stage = n->stage();
        }
      } else {
        out << ", ";
      }
    }
    printValueRef(out, n);
    out << " : ";
    out << *n->type();
  }
  return out;
}
template<typename T>
void printPrimList(std::ostream & out, const std::vector<T> & items) {
  out << "[";
  int i = 0;
  for(auto & item : items) {
    if(i++ > 0)
      out << ", ";
    out << item;
  }
  out << "]";
}

std::string escapeString(std::string s) {
  std::vector<char> search = {'\n', '\t', '\v'};
  std::vector<std::string> replace = {"\\n", "\\t", "\\v"};
  for (size_t i = 0; i < search.size(); i++) {
    size_t pos = s.find(search[i]);
    while(pos != std::string::npos) {
      s.replace(pos, 1, replace[i]);
      pos = s.find(search[i], pos + 1);
    }
  }
  return s;
}

void printAttributes(std::ostream & out, const Node * n, bool ignore_subgraph=false) {
  out << "[";
  auto names = n->attributeNames();
  int i = 0;
  for(auto name : names) {
    if (ignore_subgraph && name == attr::Subgraph)
      continue;
    if(i++ > 0)
      out << ", ";
    // TODO: debugging mode to see the qualifier.  We definitely
    // don't want to print the qualifier since it should always
    // be attribute, but you might be able to track down a weird
    // bug by printing it out.
    out << name.toUnqualString() <<"=";
    switch(n->kindOf(name)) {
      case AttributeKind::f:
        out << n->f(name);
        break;
      case AttributeKind::fs:
        printPrimList(out,n->fs(name));
        break;
      case AttributeKind::i:
        out << n->i(name);
        break;
      case AttributeKind::is:
        printPrimList(out,n->is(name));
        break;
      case AttributeKind::s:
        out << escapeString(n->s(name));
        break;
      case AttributeKind::ss:
        printPrimList(out,n->ss(name));
        break;
      case AttributeKind::t:
        {
          at::Tensor t = n->t(name);
          // 1-elem tensors are usually boxed scalars, so print them like it
          if (t.numel() == 1) {
            auto scalar = at::Scalar(t.view({})).local();
            out << "{";
            if (scalar.isFloatingPoint()) {
              out << scalar.toDouble();
            } else {
              out << scalar.toLong();
            }
            out << "}";
          } else if (t.numel() <= max_tensor_display_size) {
            // TODO: This is awful code.  Also it doesn't work on Windows.
            std::ostringstream tensor_ss;
            tensor_ss << t;
            std::string tensor_s{tensor_ss.str()};
            // Remove newlines
            std::replace(tensor_s.begin(), tensor_s.end(), '\n', ' ');
            out << tensor_s;
          } else {
            out << "<Tensor>";
          }
          break;
        }
      case AttributeKind::ts:
        out << "[<Tensors>]";
        break;
      case AttributeKind::g:
        out << "<Graph>";
        break;
      case AttributeKind::gs:
        out << "[<Graphs>]";
        break;
    }
  }
  out << "]";
}

static std::ostream & indent(std::ostream & out, size_t level) {
  for(size_t i = 0; i < level; ++i)
    out << "  ";
  return out;
}

std::ostream& printNode(std::ostream & out, size_t level, const Node * n, std::vector<const Node*> * groups) {
  auto outputs = n->outputs();
  indent(out, level) << const_value_list_with_types(outputs);
  out << " = ";
  IR_IFM_CONST(n,PythonOp)
    out << "^" << value->name();
    value->writeScalars(out);
  IR_ELSE()
    if(n->hasAttribute(attr::Subgraph) && groups) {
      out << n->kind().toQualString() << "_" << groups->size();
      if (n->numAttributes() > 1) {
        printAttributes(out, n, /*ignore_subgraph=*/true);
      }
      groups->push_back(n);
    } else {
      out << n->kind().toQualString();
      if(n->hasAttributes()) {
        printAttributes(out,n);
      }
    }
  IR_END()
  out << "(" << n->inputs() << ")";
  std::string scopeName = n->scopeName();
  if (scopeName.empty()) {
    out << "\n";
  }
  else {
    out << ", ";
    out << "scope: " << scopeName << "\n";
  }
  for(size_t i = 0; i < n->blocks().size(); ++i) {
    auto b = n->blocks()[i];
    indent(out, level + 1) << "block" << i << "(" << const_value_list_with_types(b->inputs(), false) << ") {\n";
    for(auto n : b->nodes()) {
      printNode(out, level + 2, n, groups);
    }
    indent(out, level + 2) << "-> (" << b->outputs() << ")\n";
    indent(out, level + 1) << "}\n";
  }
  return out;
}

std::ostream& operator<<(std::ostream & out, const Node & n) {
  return printNode(out, 0, &n, nullptr);
}

std::ostream& operator<<(std::ostream & out, const Graph & g) {
  out << "graph(" << const_value_list_with_types(g.inputs(), true) << ") {\n";
  std::vector<const Node*> groups;
  size_t prev_stage = 0;
  for(auto n : g.nodes()) {
    if (n->stage() != prev_stage) {
      out << "  ---------------- stage " << n->stage() << " ----------------\n";
      prev_stage = n->stage();
    }
    printNode(out, 1, n, &groups);
  }
  out << "  return (" << g.outputs() << ");\n}\n";
  size_t i = 0;
  for(auto fg : groups) {
    out << "with " << fg->kind().toQualString() << "_" <<i++ << " = " << *fg->g(attr::Subgraph);
  }
  /*
  // Uncomment this to debug all_nodes issues
  {
    out << "\n";
    out << "all_nodes:\n";
    for (auto& n : g.all_nodes) {
      printNode(out, const_cast<Node*>(n), nullptr);
    }
  }
  */
  return out;
}

static void checkSameDevice(const Node* node) {
  bool has_device = false;
  int device;
  auto checkValue = [&](const Value* v) {
    if(TensorTypePtr type = v->type()->cast<TensorType>()) {
      if(!has_device) {
        has_device = true;
        device = type->device();
      } else {
        JIT_ASSERT(device == type->device());
      }
    }
  };
  for(auto input : node->inputs()) {
    checkValue(input);
  }
  for(auto output : node->outputs()) {
    checkValue(output);
  }
}

using node_set = std::set<const Node*>;
#define ALL_OF(container) container.begin(), container.end()

// These functions purposely operate on the internal members directly, to force
// you to think about how the invariants change if you change the data
// representation (even if the external API does not change.)

// NB: This assert is written to assume you don't have any unattached
// nodes.  Unattached nodes can occur while manipulations to the
// graph are occurring.
void Node::lint() const {
  // Node invariants
  // - if node should live in list, nodes_iter is consistent
  // - Inputs are all marked as a use by the nodes they refer to
  // - Stage is consistent (stage is >= all input stages)
  // - Owning graph is non-null and consistent
  // - The "Select" invariant, when the node is MultiReturn
  //
  // The handle invariant:
  //    If a node takes a handle as an input, it is always the
  //    LAST input of the node.  There is at most one handle input.

  {
    size_t i = 0;
    for (auto input : inputs_) {
      // WARNING: O(n^2)
      JIT_ASSERT(std::find(ALL_OF(input->uses_), Use(const_cast<Node*>(this), i)) != input->uses_.end());
      JIT_ASSERT(stage_ >= input->stage_);
      JIT_ASSERT(graph_->all_nodes.count(this) == 1);
      i++;
    }
  }

  for(auto o : outputs()) {
    size_t i = 0;
    for (auto use : o->uses()) {
      // Use invariants
      // - Use is consistent with inputs
      // - Every user node is live (checked in Graph)
      JIT_ASSERT(use.user->inputs_[use.offset] == o);
      i++;
    }
  }

  // Node subclass invariants
  // - Return uses is zero
  // - Param inputs is zero
  // - Select inputs is one
  // - Python operator cconv is correct

  IR_IF(this,Constant)
    JIT_ASSERT(inputs_.size() == 0);
  IR_ELSEIF(Return)
    JIT_ASSERT(outputs().size() == 0);
  IR_ELSEIF(Param)
    JIT_ASSERT(inputs_.size() == 0);
  IR_ELSEIFM_CONST(PythonOp)
    size_t n_scalars = 0, n_tensors = 0;
    for (auto c : value->cconv) {
      if (c == 's') {
        n_scalars++;
      } else if (c == 't') {
        n_tensors++;
      } else {
        JIT_ASSERT(0);
      }
      JIT_ASSERT(static_cast<bool>(value->pyobj));
    }
    JIT_ASSERT(n_scalars == value->scalar_args.size());
    JIT_ASSERT(n_tensors == inputs_.size());
  IR_ELSEIF(Eval)
    // TODO: add invariants
  // TODO: It's not good for these ops to be top-level, it makes cases longer.
  IR_ELSEIF(FusionGroup)
    checkSameDevice(value);
    // TODO: Typecheck the parameters
    value->g(attr::Subgraph)->lint();
  IR_END()

}

// TODO: When lint fails, give better indication about which
// instruction triggered the failure.
void Graph::lint() const {
  // Graph invariants

  // Uncomment the following to see the graph
  // std::cout << *const_cast<Graph*>(this);

  // nodes
  // - nodes_ is a valid topological ordering for inputs
  // - No repeated nodes
  // - Params and return do NOT occur in nodes
  // - next_unique_ is greater than all uniques in graph
  // - uniques in all_nodes are unique
  // - every use will occur later in the topsort

  struct LintScope {
    LintScope() = default;
    LintScope(std::unique_ptr<LintScope> parent)
    : parent(std::move(parent)) {}
    bool contains(const Value * v) {
      return values.count(v) > 0 || (parent && parent->contains(v));
    }
    bool contains(const Node * n) {
      return nodes.count(n) > 0 || (parent && parent->contains(n));
    }
    void insert(const Value * v) {
      JIT_ASSERT(!contains(v));
      values.insert(v);
    }
    void insert(const Node * n) {
      JIT_ASSERT(!contains(n));
      nodes.insert(n);
    }
    std::unique_ptr<LintScope> parent;
  private:
    std::unordered_set<const Value*> values;
    std::unordered_set<const Node*> nodes;
  };
  // Struct enables mutual recursion in linting methods.
  // Putting it inside Graph::lint enables access to private Graph members
  struct LintImpl {
    LintImpl(const Graph & g)
    : g(g)
    , scope(new LintScope())
    , all_nodes_set(ALL_OF(g.all_nodes)) {} // NB: all_nodes is *unordered*
    const Graph & g;
    std::unique_ptr<LintScope> scope;
    std::unordered_set<size_t> seen_uniques;
    std::unordered_map<const Node*, int64_t> anticipated_uses;
    node_set all_nodes_set;
    node_set sum_set;

    void check_value(const Value* v) {
      scope->insert(v);
      auto b2 = seen_uniques.insert(v->unique());
      JIT_ASSERT(b2.second);  // insertion took place
      JIT_ASSERT(v->unique() < g.next_unique_);

      for (auto use : v->uses()) {
        JIT_ASSERT(!scope->contains(use.user));
        JIT_ASSERT(g.all_nodes.count(use.user) == 1);
        anticipated_uses[use.user]++;  // int default constructs to 0
      }
    }
    void check_node(const Node* n) {
      for (auto input : n->inputs_) {
        if (!scope->contains(input)) {
          JIT_ASSERTM(0, input->unique(), " not in scope");
        }
      }
      JIT_ASSERT(anticipated_uses[n] == static_cast<int64_t>(n->inputs_.size()));
      anticipated_uses[n] = -1;  // we saw the anticipated user!
      scope->insert(n);
      for(auto block : n->blocks()) {
        std::unique_ptr<LintScope> new_scope(new LintScope(std::move(scope)));
        scope = std::move(new_scope);
        check_block(block);
        scope = std::move(scope->parent);
      }
      size_t i = 0;
      for(auto o : n->outputs()) {
        JIT_ASSERT(o->node() == n);
        JIT_ASSERT(i++ == o->offset_);
        check_value(o);
      }
      n->lint();
    }
    void check_block(const Block *b) {
      for (auto input : b->inputs()) {
        check_value(input);
        JIT_ASSERT(input->node()->kind_ == prim::Param);
      }

      for (auto n : b->nodes()) {
        JIT_ASSERT(n->kind_ != prim::Param);
        JIT_ASSERT(n->kind_ != prim::Return);
        check_node(n);
      }

      JIT_ASSERT(b->output_->kind() == prim::Return);
      check_node(b->output_);

      // all_nodes
      // - inputs_, output_ and nodes_ are all included in all_nodes
      // - all_nodes does not contain dead nodes??? (likely to be temporarily
      // suspended).  Weaker: all_nodes contains all inputs and returns
      // - only one return node???

      node_set nodes_set(ALL_OF(b->nodes()));
      node_set inputs_set {b->input_};
      node_set output_set {b->output_};
      // TODO: Make a more type safe std::includes wrapper which disallows use on
      // non-ordered containers
      JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(nodes_set)));
      JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(inputs_set)));
      JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(output_set)));

      sum_set.insert(ALL_OF(nodes_set));
      sum_set.insert(ALL_OF(inputs_set));
      sum_set.insert(ALL_OF(output_set));
    }
    void check_graph() {
      node_set all_nodes_set(ALL_OF(g.all_nodes)); // NB: all_nodes is *unordered*

      check_block(g.block_);
      for (auto kv : anticipated_uses) {
        JIT_ASSERT(kv.second == -1);
      }
      // graph->stage() should be equal to max(node.stage for node in graph->nodes())
      if (g.nodes().begin() == g.nodes().end()) {
        JIT_ASSERT(g.stage() == 0);
      } else {
        JIT_ASSERT(g.stage() == g.nodes().rbegin()->stage());
      }
      JIT_ASSERT(std::includes(ALL_OF(sum_set), ALL_OF(all_nodes_set)));
    }
  };
  LintImpl(*this).check_graph();
}

void Graph::dump() const {
  std::cout << *this << "\n";
}

void LintGraph(std::shared_ptr<Graph>& graph) {
  graph->lint();
}

void Block::cloneFrom(Block * src, std::function<Value*(Value*)> value_map) {
  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value * v) {
    auto it = local_map.find(v);
    if(it != local_map.end())
      return it->second;
    return value_map(v);
  };

  auto graph = owningGraph();
  for(auto input : src->inputs()) {
    local_map[input] = this->addInput()->copyMetadata(input)->setStage(input->stage());
    graph->setStage(std::max(graph->stage(), input->stage()));
  }
  for(auto node : src->nodes()) {
    auto new_node = this->appendNode(graph->createClone(node, env));
    new_node->setStage(node->stage());
    graph->setStage(std::max(graph->stage(), node->stage()));
    for(size_t i = 0; i < node->outputs().size(); ++i) {
      auto oo = node->outputs()[i];
      auto no = new_node->outputs()[i];
      local_map[oo] = no;
      no->copyMetadata(oo);
      no->setStage(oo->stage());
    }
  }
  for(auto output : src->outputs()) {
    this->registerOutput(env(output));
  }
}

std::shared_ptr<Graph> Graph::copy() {
  auto new_g = std::make_shared<Graph>();
  auto env = [](Value *) -> Value* {
    AT_ERROR("Graph::copy() encountered a use of a value not in scope. Run lint!");
  };
  new_g->block()->cloneFrom(this->block(), env);
  return new_g;
}

Value* Value::setUniqueName(const std::string & name) {
  if (name.size() > 0 && name.find_first_not_of("0123456789") == std::string::npos) {
    throw std::runtime_error("names may not be integers: " + name);
  }

  auto & names = node()->owningGraph()->unique_names_;

  // clear any old name from the map
  if(hasUniqueName()) {
    names.erase(unique_name_);
    unique_name_ = "";
  }

  // allow "" to clear the uniquename
  if(name == "")
    return this;

  // if someone else has this name, then rename the other value
  auto old_owner_of_name = names.find(name);
  if(old_owner_of_name != names.end()) {
    size_t suffix = 1;
    std::string name_base = name;
    auto last_dot_pos = name.find_last_of('.');
    if (last_dot_pos != std::string::npos && last_dot_pos + 1 != name.size()) {
      if (name.find_first_not_of("0123456789", last_dot_pos + 1) == std::string::npos) {
        suffix = std::stoll(name.substr(last_dot_pos + 1));
        name_base = name.substr(0, last_dot_pos);
      }
    }
    std::string replacement_name;
    do {
      std::stringstream ss;
      ss << name_base << "." << suffix++;
      replacement_name = ss.str();
    } while(names.count(replacement_name) > 0);
    old_owner_of_name->second->setUniqueName(replacement_name);
  }

  names[name] = this;
  unique_name_ = name;
  return this;
}

size_t findArgument(const FunctionSchema& the_schema, Symbol name) {
  auto name_str = name.toUnqualString();
  for (size_t i = 0; i < the_schema.arguments.size(); ++i) {
    const Argument* arg = &the_schema.arguments[i];
    if (arg->name == name_str) {
      return i;
    }
  }
  throw std::runtime_error(std::string("Couldn't find an argument called ") + name.toQualString());
}

at::optional<IValue> Node::get(Symbol name) const {
  return toIValue(namedInput(name));
}

Value* Node::namedInput(Symbol name) const {
  return input(findArgument(schema(), name));
}

bool Node::matches(const char *signature_literal, at::ArrayRef<Symbol> const_inputs) {
  if (!sig(signature_literal).matches(this)) return false;
  for (Symbol s : const_inputs) {
    if (!is_constant(s)) return false;
  }
  return true;
}

void Node::dump() const {
  std::cout << *this << "\n";
}

void Node::findSchema() const {
  schema_ = &getOperatorFor(this).schema();
}

inline const SourceRange& fakeRange() {
  static SourceRange range(std::make_shared<std::string>("<internally-created-node>"), 0, 1);
  return range;
}

Value* Graph::insert(Symbol opname, at::ArrayRef<NamedValue> args, at::ArrayRef<NamedValue> kwargs) {
  return script::emitBuiltinCall(fakeRange(), *this, opname, args, kwargs, /*required=*/true);
}

PythonOp* defaultAllocPythonOp(Graph*g) {
  throw std::runtime_error("Trying to allocate a Python object without python bindings loaded");
}
std::atomic<decltype(&defaultAllocPythonOp)> alloc_python_op;

// patched in when python bindings are loaded
PythonOp* allocPythonOp(Graph* g) {
  return alloc_python_op.load()(g);
}
void setAllocPythonOp(PythonOp* (*v)(Graph* g)) {
  alloc_python_op.store(v);
}

}} // namespace torch::jit
