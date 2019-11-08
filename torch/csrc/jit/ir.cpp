#include <torch/csrc/jit/ir.h>

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/script/schema_matching.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace torch {
namespace jit {

// Constants relating to maintaining the topological index of nodes.
//
// Lower and upper bounds of the index. Inclusive range.
static constexpr topo_position_t kLowerBound = INT64_MIN;
static constexpr topo_position_t kUpperBound = INT64_MAX;
static constexpr topo_position_t kMidPoint = 0;

// How far away to space nodes that are appended to the graph.
// should be 2^n, where:
//   - n is the maximum number of repeated insertions without a re-index
//   - 2^(64-n) is the maximum number of appends to the end without reindex
static constexpr topo_position_t kAppendInterval = 1099511627776ULL /* 2^40 */;

static void printValueRef(std::ostream& out, const Value* n) {
  out << "%" << n->debugName();
}

// NB: This overload will become ambiguous with the one Caffe2 provides in its
// logging, if they ever intersect.
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& nodes) {
  out << at::ArrayRef<T>{nodes};
  return out;
}

template <typename T>
static std::ostream& printValueRefs(
    std::ostream& out,
    const at::ArrayRef<T>& nodes) {
  size_t i = 0;
  for (auto n : nodes) {
    if (i++ > 0) {
      out << ", ";
    }
    printValueRef(out, n);
  }
  return out;
}

// Can't make these two overloads directly a template, it'll be ambiguous with
// the global printer for operator<<.

std::ostream& operator<<(
    std::ostream& out,
    const at::ArrayRef<const Value*>& nodes) {
  return printValueRefs(out, nodes);
}

std::ostream& operator<<(std::ostream& out, const at::ArrayRef<Value*>& nodes) {
  return printValueRefs(out, nodes);
}

struct const_value_list_with_types {
  const ArrayRef<const Value*> values;
  std::string delim;
  const_value_list_with_types(
      ArrayRef<const Value*> values,
      std::string delim_ = ", ")
      : values(values), delim(std::move(delim_)) {}
};

std::ostream& operator<<(std::ostream& out, const_value_list_with_types l) {
  size_t i = 0;
  for (auto n : l.values) {
    if (i++ > 0) {
      out << l.delim;
    }
    printValueRef(out, n);
    out << " : ";
    out << *n->type();
  }
  return out;
}

template <typename T>
static void printPrimList(std::ostream& out, const std::vector<T>& items) {
  out << "[";
  int i = 0;
  for (auto& item : items) {
    if (i++ > 0) {
      out << ", ";
    }
    out << item;
  }
  out << "]";
}

static void printStrList(
    std::ostream& out,
    const std::vector<std::string>& items) {
  out << "[";
  int i = 0;
  for (auto& item : items) {
    if (i++ > 0)
      out << ", ";
    c10::printQuotedString(out, item);
  }
  out << "]";
}

static void printTypeList(
    std::ostream& out,
    const std::vector<TypePtr>& items) {
  out << "[";
  int i = 0;
  for (auto& item : items) {
    if (i++ > 0)
      out << ", ";
    out << *item;
  }
  out << "]";
}

void Node::printAttrValue(std::ostream& out, const Symbol& name) const {
  switch (kindOf(name)) {
    case AttributeKind::f:
      out << f(name);
      break;
    case AttributeKind::fs:
      printPrimList(out, fs(name));
      break;
    case AttributeKind::i:
      out << i(name);
      break;
    case AttributeKind::is:
      printPrimList(out, is(name));
      break;
    case AttributeKind::s:
      c10::printQuotedString(out, s(name));
      break;
    case AttributeKind::ss:
      printStrList(out, ss(name));
      break;
    case AttributeKind::t: {
      at::Tensor tensor = t(name);
      // 1-elem tensors are usually boxed scalars, so print them like it
      if (tensor.numel() == 1) {
        auto scalar_tensor = tensor.view({}).item();
        out << "{";
        if (scalar_tensor.isFloatingPoint()) {
          out << scalar_tensor.toDouble();
        } else {
          out << scalar_tensor.toLong();
        }
        out << "}";
      } else if (tensor.numel() <= max_tensor_display_size) {
        // TODO: This is awful code.  Also it doesn't work on Windows.
        std::ostringstream tensor_ss;
        tensor_ss << tensor;
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
    case AttributeKind::ty:
      out << *ty(name);
      break;
    case AttributeKind::tys:
      printTypeList(out, tys(name));
      break;
  }
}

void Node::printAttributes(std::ostream &out,
                           bool ignore_subgraph = false) const {
  out << "[";
  auto names = attributeNames();
  int i = 0;
  for (auto name : names) {
    if (ignore_subgraph && name == attr::Subgraph) {
      continue;
    }
    if (i++ > 0) {
      out << ", ";
    }
    // TODO: debugging mode to see the qualifier.  We definitely
    // don't want to print the qualifier since it should always
    // be attribute, but you might be able to track down a weird
    // bug by printing it out.
    out << name.toUnqualString() << "=";

    printAttrValue(out, name);
  }
  out << "]";
}

SourceRange Node::sourceRange() const {
  if (source_range_) {
    return *source_range_;
  }
  return SourceRange();
}

static std::ostream& indent(std::ostream& out, size_t level) {
  for (size_t i = 0; i < level; ++i) {
    out << "  ";
  }
  return out;
}

std::ostream &Node::print(std::ostream &out, size_t level,
                          std::vector<const Node *> *groups,
                          bool print_source_locations, bool print_attributes,
                          bool print_scopes, bool print_body) const {
  auto outs = outputs();
  indent(out, level) << const_value_list_with_types(outs);
  out << " = ";
  if (kind() == prim::PythonOp) {
    auto* pyOp = static_cast<const ::torch::jit::PythonOp*>(this);
    out << "^" << pyOp->name();
    pyOp->writeScalars(out);
  } else if (hasAttribute(attr::Subgraph) && groups) {
    out << kind().toQualString() << "_" << groups->size();
    if (print_attributes && numAttributes() > 1 &&
        kind() != prim::DifferentiableGraph) {
      printAttributes(out, /*ignore_subgraph=*/true);
    }

    groups->push_back(this);
  } else {
    out << kind().toQualString();
    if (print_attributes && hasAttributes()) {
      printAttributes(out);
    }
  }
  out << "(" << inputs() << ")";

  if (print_scopes) {
    std::string scName = scopeName();
    if (!scName.empty()) {
      out << ", ";
      out << "scope: " << scName;
    }
  }

  // In debug print, append file:line:col as a comment after each node
  if (print_source_locations) {
    SourceRange r = sourceRange();
    if (sourceRange().source()) {
      if (auto orig = sourceRange().source()->findSourceRangeThatGenerated(r)) {
        r = *orig;
      }
    }
    if (auto file_line_col = r.file_line_col()) {
      std::string filename;
      size_t line, col;
      std::tie(filename, line, col) = *file_line_col;
      out << " # " << filename << ":" << line << ":" << col;
    }
  }

  if (!print_body) {
    return out;
  }

  out << "\n";

  for (size_t i = 0; i < blocks().size(); ++i) {
    auto b = blocks()[i];
    indent(out, level + 1) << "block" << i << "("
                           << const_value_list_with_types(b->inputs())
                           << "):\n";
    for (auto nested : b->nodes()) {
      nested->print(out, level + 2, groups);
    }
    indent(out, level + 2) << "-> (" << b->outputs() << ")\n";
  }

  return out;
}

std::ostream& operator<<(std::ostream& out, const Node& n) {
  return n.print(out, 0, nullptr);
}

std::ostream& Graph::print(std::ostream& out, bool print_source_locations)
    const {
  out << "graph(" << const_value_list_with_types(inputs(), ",\n      ")
      << "):\n";
  std::vector<const Node*> groups;
  for (auto n : nodes()) {
    n->print(out, 1, &groups, print_source_locations);
  }
  out << "  return (" << outputs() << ")\n";
  size_t i = 0;
  for (auto fg : groups) {
    out << "with " << fg->kind().toQualString() << "_" << i++ << " = "
        << *fg->g(attr::Subgraph);
  }
  /*
  // Uncomment this to debug all_nodes issues
  {
    out << "\n";
    out << "all_nodes:\n";
    for (auto& n : all_nodes) {
      printNode(out, const_cast<Node*>(n), nullptr);
    }
  }
  */
  return out;
}

std::ostream& operator<<(std::ostream& out, const Graph& g) {
  return g.print(out, true);
}

static void checkSameDevice(const Node* node) {
  bool has_device = false;
  c10::optional<at::Device> device = c10::nullopt;
  auto checkValue = [&](const Value* v) {
    if (TensorTypePtr type = v->type()->cast<TensorType>()) {
      if (type->device() && !has_device) {
        has_device = true;
        device = *type->device();
      } else {
        AT_ASSERT(device == type->device());
      }
    }
  };
  for (auto input : node->inputs()) {
    checkValue(input);
  }
  for (auto output : node->outputs()) {
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
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      AT_ASSERT(
          std::find(ALL_OF(input->uses_), Use(const_cast<Node*>(this), i)) !=
          input->uses_.end());
      AT_ASSERT(graph_->all_nodes.count(this) == 1);
      i++;
    }
  }

  for (auto o : outputs()) {
    size_t i = 0;
    for (auto use : o->uses()) {
      // Use invariants
      // - Use is consistent with inputs
      // - Every user node is live (checked in Graph)
      AT_ASSERT(use.user->inputs_[use.offset] == o);
      i++;
    }
  }

  // Node subclass invariants
  switch (kind()) {
    case prim::Constant:
      AT_ASSERT(inputs_.size() == 0);
      break;
    case prim::Return:
      // Return uses is zero
      AT_ASSERT(outputs().size() == 0);
      break;
    case prim::Param:
      // Param inputs is zero
      AT_ASSERT(inputs_.size() == 0);
      break;
    case prim::PythonOp: {
      // Python operator cconv is correct
      auto* value = static_cast<const PythonOp*>(this);
      value->lint_python();
      break;
    }
    case prim::Eval:
      // TODO: add invariants
      // TODO: It's not good for these ops to be top-level, it makes cases
      // longer.
      break;
    case prim::FusionGroup:
      checkSameDevice(this);
      // TODO: Typecheck the parameters
      g(attr::Subgraph)->lint();
      break;
  }
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
    LintScope(std::unique_ptr<LintScope> parent) : parent(std::move(parent)) {}
    bool contains(const Value* v) {
      return values.count(v) > 0 || (parent && parent->contains(v));
    }
    bool contains(const Node* n) {
      return nodes.count(n) > 0 || (parent && parent->contains(n));
    }
    void insert(const Value* v) {
      AT_ASSERT(!contains(v));
      values.insert(v);
    }
    void insert(const Node* n) {
      AT_ASSERT(!contains(n));
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
    LintImpl(const Graph& g)
        : g(g),
          scope(new LintScope()),
          all_nodes_set(ALL_OF(g.all_nodes)) {} // NB: all_nodes is *unordered*
    const Graph& g;
    std::unique_ptr<LintScope> scope;
    std::unordered_set<size_t> seen_uniques;
    std::unordered_map<const Node*, int64_t> anticipated_uses;
    node_set all_nodes_set;
    node_set sum_set;

    void check_value(const Value* v) {
      scope->insert(v);
      auto b2 = seen_uniques.insert(v->unique());
      AT_ASSERT(b2.second); // insertion took place
      AT_ASSERT(v->unique() < g.next_unique_);

      for (auto use : v->uses()) {
        AT_ASSERT(!scope->contains(use.user));
        AT_ASSERT(g.all_nodes.count(use.user) == 1);
        anticipated_uses[use.user]++; // int default constructs to 0
      }
    }
    void check_node(const Node* n) {
      for (auto input : n->inputs_) {
        if (!scope->contains(input)) {
          AT_ASSERTM(0, input->unique(), " not in scope");
        }
      }
      AT_ASSERT(anticipated_uses[n] == static_cast<int64_t>(n->inputs_.size()));
      anticipated_uses[n] = -1; // we saw the anticipated user!
      scope->insert(n);
      for (auto block : n->blocks()) {
        std::unique_ptr<LintScope> new_scope(new LintScope(std::move(scope)));
        scope = std::move(new_scope);
        check_block(block);
        scope = std::move(scope->parent);
      }
      size_t i = 0;
      for (auto o : n->outputs()) {
        AT_ASSERT(o->node() == n);
        AT_ASSERT(i++ == o->offset_);
        check_value(o);
      }
      n->lint();
    }
    void check_block(const Block* b) {
      // Check topological ordering
      AT_ASSERT(b->param_node()->isBefore(*b->nodes().begin()));
      auto curNode = *b->nodes().begin();
      while (curNode != b->return_node()) {
        AT_ASSERT(curNode->isBefore(curNode->next()));
        curNode = curNode->next();
      }

      for (auto input : b->inputs()) {
        check_value(input);
        AT_ASSERT(input->node()->kind_ == prim::Param);
      }

      for (auto n : b->nodes()) {
        AT_ASSERT(n->kind_ != prim::Param);
        AT_ASSERT(n->kind_ != prim::Return);
        check_node(n);
      }

      AT_ASSERT(b->output_->kind() == prim::Return);
      check_node(b->output_);

      // all_nodes
      // - inputs_, output_ and nodes_ are all included in all_nodes
      // - all_nodes does not contain dead nodes??? (likely to be temporarily
      // suspended).  Weaker: all_nodes contains all inputs and returns
      // - only one return node???

      node_set nodes_set(ALL_OF(b->nodes()));
      node_set inputs_set{b->input_};
      node_set output_set{b->output_};
      // TODO: Make a more type safe std::includes wrapper which disallows use
      // on non-ordered containers
      AT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(nodes_set)));
      AT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(inputs_set)));
      AT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(output_set)));

      sum_set.insert(ALL_OF(nodes_set));
      sum_set.insert(ALL_OF(inputs_set));
      sum_set.insert(ALL_OF(output_set));
    }
    void check_graph() {
      node_set all_nodes_set(
          ALL_OF(g.all_nodes)); // NB: all_nodes is *unordered*

      check_block(g.block_);
      for (auto kv : anticipated_uses) {
        AT_ASSERT(kv.second == -1);
      }
      AT_ASSERT(std::includes(ALL_OF(sum_set), ALL_OF(all_nodes_set)));
    }
  };
  LintImpl(*this).check_graph();
}

void Graph::dump() const {
  std::cout << *this << "\n";
}

void Graph::push_scope(const std::string& scope_name) {
  current_scope_ = current_scope_->push(Symbol::scope(scope_name));
  Node* block_node = insertNode(create(prim::TracedModuleForward, 0));
  block_node->s_(attr::scope, scope_name);
  Block* b = block_node->addBlock();
  setInsertPoint(b);
}
void Graph::pop_scope() {
  current_scope_ = current_scope_->parent();
  if (insertPoint()->owningBlock()->owningNode()->kind() ==
      prim::TracedModuleForward) {
    setInsertPoint(insertPoint()->owningBlock()->owningNode()->next());
  }
}

void LintGraph(const std::shared_ptr<Graph>& graph) {
  graph->lint();
}

Block::Block(Graph* graph_, Node* node_)
    : graph_(graph_),
      output_(graph_->create(prim::Return, 0)),
      input_(graph_->create(prim::Param, 0)),
      owning_node_(node_) {
  input_->next() = output_;
  input_->prev() = output_;
  output_->next() = input_;
  output_->prev() = input_;

  graph_->all_blocks.emplace(this);
  output_->owning_block_ = this;
  output_->topo_position_ = kUpperBound;
  input_->owning_block_ = this;
  input_->topo_position_ = kLowerBound;
}

void Block::reIndexTopology() {
  auto curPos = kLowerBound;
  for (auto node : nodes()) {
    AT_ASSERT(curPos <= (kUpperBound - kAppendInterval));
    curPos += kAppendInterval;
    node->topo_position_ = curPos;
  }
}

void Block::cloneFrom(Block* src, std::function<Value*(Value*)> value_map) {
  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value* v) {
    auto it = local_map.find(v);
    if (it != local_map.end()) {
      return it->second;
    }
    return value_map(v);
  };

  auto graph = owningGraph();
  for (auto input : src->inputs()) {
    local_map[input] = this->addInput()->copyMetadata(input);
  }

  for (auto node : src->nodes()) {
    auto new_node = this->appendNode(graph->createClone(node, env));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      auto oo = node->outputs()[i];
      auto no = new_node->outputs()[i];
      local_map[oo] = no;
      no->copyMetadata(oo);
    }
  }
  for (auto output : src->outputs()) {
    this->registerOutput(env(output));
  }
}

void Block::destroy() {
  // we cannot destroy the output because it is used as the sentinel
  // for the nodes() list and has to remain valid for the loop
  output_->removeAllInputs();
  for (auto it = this->nodes().reverse().begin(),
            end = this->nodes().reverse().end();
       it != end;
       ++it) {
    it.destroyCurrent();
  }
  output_->destroy();
  input_->destroy();
  graph_->freeBlock(this);
}

std::shared_ptr<Graph> Graph::copy() {
  auto new_g = std::make_shared<Graph>();
  auto env = [](Value* v) -> Value* {
    AT_ERROR(
        "Graph::copy() encountered a use of a value " + v->debugName() +
        " not in scope. Run lint!");
  };
  new_g->block()->cloneFrom(this->block(), env);
  return new_g;
}

void Block::remapTypes(const std::function<TypePtr(TypePtr)>& type_map) {
  for (Value* input : inputs()) {
    input->setType(type_map(input->type()));
  }
  for (Node* node : nodes()) {
    for (Value* output : node->outputs()) {
      output->setType(type_map(output->type()));
    }
    for (Block* sub_block : node->blocks()) {
      sub_block->remapTypes(type_map);
    }
    for (Symbol name : node->attributeNames()) {
      if (node->kindOf(name) == AttributeKind::g) {
        node->g(name)->remapTypes(type_map);
      } else if (node->kindOf(name) == AttributeKind::gs) {
        for (const auto& g : node->gs(name)) {
          g->remapTypes(type_map);
        }
      }
    }
  }
}

void Graph::remapTypes(const std::function<TypePtr(TypePtr)>& type_map) {
  block()->remapTypes(type_map);
}

void Value::inferTypeFrom(const at::Tensor& output) {
  setType(TensorType::create(output));
}

bool Value::mustBeNone() const {
  return node_->mustBeNone();
}
bool Value::mustNotBeNone() const {
  return node_->kind() != prim::AutogradAdd && type() != NoneType::get() &&
      !type()->cast<OptionalType>();
}

std::string Value::debugNameBase() const {
  std::string name = debugName();
  std::string name_base = name;
  auto last_dot_pos = name.find_last_of('.');
  if (last_dot_pos != std::string::npos && last_dot_pos + 1 != name.size()) {
    if (name.find_first_not_of("0123456789", last_dot_pos + 1) ==
        std::string::npos) {
      name_base = name.substr(0, last_dot_pos);
    }
  }
  return name_base;
}

bool Value::isValidName(const std::string& name) {
  // Empty strings are legal
  if (!name.size()) {
    return true;
  }

  // Numbers are not legal
  if (name.find_first_not_of("0123456789") == std::string::npos) {
    return false;
  }

  return true;
}

Value* Value::setDebugName(const std::string& name) {
  if (!isValidName(name)) {
    throw std::runtime_error("Invalid name: '" + name + "'");
  }

  auto& names = node()->owningGraph()->unique_names_;

  // clear any old name from the map
  if (hasDebugName()) {
    names.erase(unique_name_);
    unique_name_ = "";
  }

  // allow "" to clear the uniquename
  if (name == "") {
    return this;
  }

  // if someone else has this name, then rename the other value
  auto old_owner_of_name = names.find(name);
  if (old_owner_of_name != names.end()) {
    size_t suffix = 1;
    std::string name_base = name;
    auto last_dot_pos = name.find_last_of('.');
    if (last_dot_pos != std::string::npos && last_dot_pos + 1 != name.size()) {
      if (name.find_first_not_of("0123456789", last_dot_pos + 1) ==
          std::string::npos) {
        suffix = c10::stoll(name.substr(last_dot_pos + 1));
        name_base = name.substr(0, last_dot_pos);
      }
    }
    std::string replacement_name;
    do {
      std::stringstream ss;
      ss << name_base << "." << suffix++;
      replacement_name = ss.str();
    } while (names.count(replacement_name) > 0);
    old_owner_of_name->second->setDebugName(replacement_name);
  }

  names[name] = this;
  unique_name_ = name;
  return this;
}

Value* Value::copyMetadata(Value* from) {
  setType(from->type());
  if (from->hasDebugName()) {
    setDebugName(from->debugName());
  }
  return this;
}

void Value::replaceFirstUseWith(Value* newValue) {
  AT_ASSERT(owningGraph() == newValue->owningGraph());
  auto u = uses()[0];
  u.user->inputs_[u.offset] = newValue;
  newValue->uses_.push_back(u);
  uses_.erase(uses_.begin());
}

void Value::replaceAllUsesWith(Value* newValue) {
  while (!uses().empty()) {
    replaceFirstUseWith(newValue);
  }
}

size_t findArgument(const FunctionSchema& the_schema, Symbol name) {
  auto name_str = name.toUnqualString();
  for (size_t i = 0; i < the_schema.arguments().size(); ++i) {
    const Argument* arg = &the_schema.arguments()[i];
    if (arg->name() == name_str) {
      return i;
    }
  }
  throw std::runtime_error(
      std::string("Couldn't find an argument called ") + name.toQualString());
}

c10::optional<IValue> Node::get(Symbol name) const {
  return toIValue(namedInput(name));
}

Value* Node::namedInput(Symbol name) const {
  return input(findArgument(schema(), name));
}

bool Node::matches(
    const char* signature_literal,
    at::ArrayRef<Symbol> const_inputs) const {
  if (!sig(signature_literal).matches(this)) {
    return false;
  }
  for (Symbol s : const_inputs) {
    if (!is_constant(s)) {
      return false;
    }
  }
  return true;
}

bool Node::mustBeNone() const {
  // We can statically deduce this Node has returning None if:
  return
      // It's an AutogradZero node, or ...
      kind_ == prim::AutogradZero ||
      // It has only one output and that output is NoneType, or ...
      (outputs().size() == 1 && output()->type() == NoneType::get()) ||
      // It's a constant optional with no value in the attributes.
      (kind_ == prim::Constant && !this->hasAttributes() &&
       output()->type()->cast<OptionalType>());
}

void Node::dump() const {
  std::cout << *this << "\n";
}

const FunctionSchema& Node::schema() const {
  if (op_) {
    return op_->schema();
  }
  return getOperatorFor(this).schema();
}

const FunctionSchema* Node::maybeSchema() const {
  if (auto op = maybeOperator()) {
    return &op->schema();
  }
  return nullptr;
}

const Operator& Node::getOperator() const {
  if (!op_) {
    op_ = &getOperatorFor(this);
  }
  return *op_;
}

const Operator* Node::maybeOperator() const {
  if (!op_) {
    if (auto op = findOperatorFor(this)) {
      op_ = op.get();
    }
  }
  return op_;
}

bool Node::isNondeterministic() const {
  static const OperatorSet nondeterministic_ops = {
      "aten::dropout(Tensor input, float p, bool train) -> Tensor",
      "aten::_fused_dropout(Tensor self, float p, Generator? generator) -> (Tensor, Tensor)",
      "aten::_standard_gamma(Tensor self, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
      "aten::multinomial(Tensor self, int num_samples, bool replacement, *, Generator? generator) -> Tensor",
      "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
      "aten::poisson(Tensor self, Generator? generator) -> Tensor",
      "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::rand_like(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
      "aten::rand_like(Tensor self, *, int dtype, int layout, Device device, bool pin_memory, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint_like(Tensor self, int high, *, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint_like(Tensor self, int low, int high, *, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint_like(Tensor self, int high, *, int dtype, int layout, Device device, bool pin_memory, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint_like(Tensor self, int low, int high, *, int dtype, int layout, Device device, bool pin_memory, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randn_like(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randn_like(Tensor self, *, int dtype, int layout, Device device, bool pin_memory, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randperm(int n, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor"};

  if (nondeterministic_ops.find(this) == nullptr) {
    return false;
  }
  // Dropout with train = False is deterministic
  if (matches("aten::dropout(Tensor input, float p, bool train) -> Tensor") &&
      is_constant(attr::train) && !get<bool>(attr::train).value()) {
    return false;
  }
  return true;
}

bool Node::hasSideEffects() const {
  switch (kind_) {
    case prim::PythonOp:
    case prim::IgnoredPythonOp:
    case prim::Print:
    case prim::RaiseException:
    case prim::SetAttr:
    case aten::warn:
    case aten::save:
    case aten::manual_seed:
    case prim::AddStatValue:
    case prim::TimePoint:
    case prim::CallFunction:
    case prim::CallMethod:
    case prim::BailoutTemplate:
    case prim::profile:
    case prim::BailOut:
    case prim::Guard:
      return true;
  }

  auto op = maybeOperator();
  if (!op) {
    TORCH_INTERNAL_ASSERT(
        kind_.is_prim(),
        "Only prim ops are allowed to not have a registered operator but ",
        kind_.toDisplayString(),
        " doesn't have one either. We don't know if this op has side effects.");
    return false;
  }

  if (kind_.is_prim() || kind_.is_aten()) {
    // TODO There is nothing in the system that relies on aten:: and prim::
    // ops using AliasAnalysisKind::FROM_SCHEMA,
    // AliasAnalysisKind::INTERNAL_SPECIAL_CASE, or
    // AliasAnalysisKind::CONSERVATIVE but this is the intended behavior for all
    // current ops and a good error check. We can consider lifting this
    // constraint later if we have a use case for it.
    TORCH_INTERNAL_ASSERT(
        op->aliasAnalysisKind() == AliasAnalysisKind::INTERNAL_SPECIAL_CASE ||
            op->aliasAnalysisKind() == AliasAnalysisKind::FROM_SCHEMA ||
            op->aliasAnalysisKind() == AliasAnalysisKind::CONSERVATIVE,
        "aten:: and prim:: ops should have AliasAnalysisKind::INTERNAL_SPECIAL_CASE"
        ", AliasAnalysisKind::FROM_SCHEMA or AliasAnalysisKind::CONSERVATIVE but ",
        kind_.toDisplayString(),
        " has ",
        toString(op->aliasAnalysisKind()));
  }

  switch (op->aliasAnalysisKind()) {
    case AliasAnalysisKind::PURE_FUNCTION:
      return false;
    case AliasAnalysisKind::FROM_SCHEMA:
      return false;
    case AliasAnalysisKind::INTERNAL_SPECIAL_CASE:
      return false;
    case AliasAnalysisKind::CONSERVATIVE:
      return true;
  }
  TORCH_INTERNAL_ASSERT(false, "Unhandled AliasAnalysisKind case");
  return false; // silence compiler warning
}

// Assign this node a topological position, to facilitate fast isBefore() and
// isAfter() queries. Must be called right after a node is inserted into the
// node list.
//
// The basic scheme is: assign every node a position (uint64_t).  The common
// case (appending to the end of the graph) is made more efficient by advancing
// a fixed interval past the previous node and placing `this` there. Otherwise,
// assign `this` a position at the midpoint between its prev() and next()
// nodes.
//
// If we ever run out of space (by, e.g. inserting too much in place), we
// reindex by spreading out all the nodes again.
void Node::assignTopoPosition() {
  bool is_first = prev() == owningBlock()->param_node();
  bool is_last = next() == owningBlock()->return_node();

  const auto prevPos = prev()->topo_position_;
  const auto nextPos = next()->topo_position_;

  // Append to the end of the graph
  if (is_last) {
    if (is_first) {
      // the node list is empty, assign the first position
      topo_position_ = kMidPoint;
      return;
    }

    if (prevPos >= (kUpperBound - kAppendInterval)) {
      // we're running off the edge
      owningBlock()->reIndexTopology();
      return;
    }

    topo_position_ = prevPos + kAppendInterval;

    // Prepend to the graph
  } else if (is_first) {
    // next() is the first element in the block list
    if (nextPos <= (kLowerBound + kAppendInterval)) {
      // we're running off the edge
      owningBlock()->reIndexTopology();
      return;
    }
    topo_position_ = nextPos - kAppendInterval;

    // insert between two existing nodes
  } else {
    const auto posBetween = prevPos + (nextPos - prevPos) / 2;
    if (posBetween == prevPos) {
      // There was no room
      owningBlock()->reIndexTopology();
      return;
    }
    topo_position_ = posBetween;
  }
}

Node::Node(Graph* graph_, NodeKind kind_)
    : kind_(kind_),
      graph_(graph_),
      owning_block_(nullptr),
      scope_(graph_->current_scope_),
      callstack_(c10::nullopt),
      op_(nullptr),
      topo_position_(0) {
  graph_->all_nodes.emplace(this);
}

void Node::eraseOutput(size_t i) {
  AT_ASSERT(i < outputs_.size());
  AT_ASSERT(outputs_[i]->uses().empty());
  op_ = nullptr;
  Value* n = outputs_[i];
  outputs_.erase(outputs_.begin() + i);
  owningGraph()->freeValue(n);
  for (size_t j = i; j < outputs_.size(); j++) {
    outputs_[j]->offset_--;
  }
}

Block* Node::addBlock() {
  op_ = nullptr;
  blocks_.push_back(new Block(owningGraph(), this));
  return blocks_.back();
}

void Node::eraseBlock(size_t i) {
  AT_ASSERT(i < blocks_.size());
  op_ = nullptr;
  Block* n = blocks_[i];
  blocks_.erase(blocks_.begin() + i);
  n->destroy();
}

void Node::destroy() {
  while (!outputs().empty()) {
    eraseOutput(outputs().size() - 1);
  }
  while (!blocks().empty()) {
    eraseBlock(blocks().size() - 1);
  }
  removeAllInputs();
  if (inBlockList()) {
    removeFromList();
  }
  graph_->freeNode(this);
}

void Node::cloneFrom(Node* s) {
  source_range_ = s->source_range_;
  if (s->scope_ && !s->scope_->isBlank()) {
    scope_ = s->scope_;
  }
  copyAttributes(*s);
  callstack_ = s->callstack_;
}

void Node::replaceAllUsesWith(Node* n) {
  AT_ASSERT(outputs().size() == n->outputs().size());
  size_t nOutputs = outputs().size();
  for (size_t i = 0; i < nOutputs; i++) {
    outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
  }
}

Value* Node::insertInput(size_t i, Value* value) {
  AT_ASSERT(graph_ == value->owningGraph());
  op_ = nullptr;
  // First we update the offsets for all existing inputs that will reside
  // after the one we're inserting. Concretely, these are the inputs at
  // indices [i, # input). Since we're inserting one input before all of
  // these inputs, increment their use offsets for this value by 1
  for (size_t use_itr = i; use_itr < inputs_.size(); ++use_itr) {
    // See Note [User node does not uniquely identify use]
    auto use = findUseForInput(use_itr);
    use->offset += 1;
  }
  // Insert the actual input at the specified index
  inputs_.insert(inputs_.begin() + i, value);
  // Register the new use of the value we're inserted as an input.
  value->uses_.emplace_back(this, i);
  return value;
}

Value* Node::addInput(Value* value) {
  AT_ASSERT(graph_ == value->owningGraph());
  op_ = nullptr;
  value->uses_.emplace_back(this, inputs_.size());
  inputs_.push_back(value);
  return value;
}

Value* Node::replaceInput(size_t i, Value* newValue) {
  AT_ASSERT(newValue->owningGraph() == graph_);
  op_ = nullptr;
  Value* old = dropInput(i);
  inputs_[i] = newValue;
  newValue->uses_.emplace_back(this, i);
  return old;
}

void Node::replaceInputWith(Value* from, Value* to) {
  AT_ASSERT(from->owningGraph() == graph_);
  AT_ASSERT(to->owningGraph() == graph_);
  op_ = nullptr;
  size_t i = 0;
  for (auto input : inputs()) {
    if (input == from) {
      replaceInput(i, to);
    }
    i++;
  }
}

Value* Node::addOutput() {
  outputs_.push_back(new Value(this, outputs_.size()));
  op_ = nullptr;
  return outputs_.back();
}

Value* Node::insertOutput(size_t i) {
  op_ = nullptr;
  outputs_.insert(outputs_.begin() + i, new Value(this, i));
  for (size_t itr = i + 1; itr < outputs_.size(); ++itr) {
    outputs_[itr]->setOffset(outputs_[itr]->offset() + 1);
  }
  return outputs_.at(i);
}

bool Node::isBeforeOrAfter(const Node* n, MoveSide moveSide) const {
  if (this->owningBlock() == n->owningBlock()) {
    if (moveSide == MoveSide::BEFORE) {
      return this->topo_position_ < n->topo_position_;
    }

    if (moveSide == MoveSide::AFTER) {
      return this->topo_position_ > n->topo_position_;
    }

    AT_ASSERT(this == n);
    return false;
  }

  // These nodes don't share a common block. Traverse the blockchains upward
  // until we find the first common block.
  auto lhs = this;
  while (lhs) {
    AT_ASSERT(lhs->owningBlock());

    auto rhs = n;
    while (rhs) {
      if (!rhs->owningBlock()) {
        break;
      }

      if (lhs->owningBlock() == rhs->owningBlock()) {
        return lhs->isBeforeOrAfter(rhs, moveSide);
      }
      rhs = rhs->owningBlock()->owningNode();
    }

    lhs = lhs->owningBlock()->owningNode();
  }
  // should never reach here, since both nodes are ultimately in the same graph
  AT_ASSERT(false);
}

bool Node::isBefore(const Node* n) const {
  return isBeforeOrAfter(n, MoveSide::BEFORE);
}

bool Node::isAfter(const Node* n) const {
  return isBeforeOrAfter(n, MoveSide::AFTER);
}

Node* Node::insertBefore(Node* n) {
  AT_ASSERT(n->inBlockList());
  insertAfter(n->prev());
  return this;
}

Node* Node::insertAfter(Node* n) {
  AT_ASSERT(!inBlockList() && n->inBlockList());
  AT_ASSERT(n->owningBlock());
  AT_ASSERTM(
      n->kind() != prim::Return,
      "Attempting to insert a Node after the Return node or before the Param node");
  this->owning_block_ = n->owningBlock();
  Node* next = n->next();
  n->next() = this;
  this->prev() = n;
  this->next() = next;
  next->prev() = this;
  assignTopoPosition();
  return this;
}

void Node::moveAfter(Node* n) {
  removeFromList();
  insertAfter(n);
}

void Node::moveBefore(Node* n) {
  removeFromList();
  insertBefore(n);
}

void Node::removeInput(size_t i) {
  op_ = nullptr;
  dropInput(i);
  // everything after this input shifts left,
  // so we need to update their use offsets to match
  for (size_t j = i + 1; j < inputs_.size(); j++) {
    auto it = findUseForInput(j);
    it->offset--;
  }
  inputs_.erase(inputs_.begin() + i);
}

void Node::removeAllInputs() {
  op_ = nullptr;
  for (size_t i = 0; i < inputs().size(); ++i) {
    dropInput(i);
  }
  inputs_.clear();
}

void Node::permuteInputs(const std::vector<size_t>& new_order) {
  op_ = nullptr;
  AT_ASSERT(new_order.size() == inputs_.size());
  std::vector<Value*> new_inputs;
  new_inputs.reserve(new_order.size());
  for (size_t i = 0; i < new_order.size(); ++i) {
    AT_ASSERTM(inputs_.at(new_order[i]) != nullptr, "Repeated index");
    new_inputs.push_back(inputs_.at(new_order[i]));
    auto it = findUseForInput(new_order[i]);
    it->offset = i;
    inputs_.at(new_order[i]) = nullptr;
  }
  inputs_ = std::move(new_inputs);
}

void Node::permuteOutputs(const std::vector<size_t>& new_order) {
  op_ = nullptr;
  AT_ASSERT(new_order.size() == outputs_.size());
  std::vector<Value*> new_outputs;
  new_outputs.reserve(new_order.size());
  for (size_t i = 0; i < new_order.size(); ++i) {
    AT_ASSERTM(outputs_.at(new_order[i]) != nullptr, "Repeated index");
    new_outputs.push_back(outputs_.at(new_order[i]));
    outputs_.at(new_order[i])->setOffset(i);
    outputs_.at(new_order[i]) = nullptr;
  }
  outputs_ = std::move(new_outputs);
}

use_list::iterator Node::findUseForInput(size_t i) {
  auto& input_uses = inputs_[i]->uses_;
  // O(N) on the use list, but unless we get nodes with +100 uses
  // vector traversal still is probably faster than linked list
  auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
  AT_ASSERT(use_it != input_uses.end());
  return use_it;
}

Value* Node::dropInput(size_t i) {
  AT_ASSERT(i < inputs_.size());
  auto input_node = inputs_[i];
  auto use_it = findUseForInput(i);
  input_node->uses_.erase(use_it);
  inputs_[i] = nullptr;
  return input_node;
}

void Node::removeFromList() {
  AT_ASSERT(inBlockList());
  this->owning_block_ = nullptr;
  Node* next = this->next();
  Node* prev = this->prev();
  prev->next() = next;
  next->prev() = prev;
  this->next() = nullptr;
  this->prev() = nullptr;
}

Block* Node::findCommonAncestorBlockWith(Node* n) {
  if (n->owningBlock() == owningBlock()) {
    return owningBlock();
  }

  Node* n1 = this;
  Node* n2 = n;

  size_t d_1 = n1->blocksFromGraphBlock();
  size_t d_2 = n2->blocksFromGraphBlock();

  for (; d_1 > d_2; --d_1) {
    n1 = n1->owningBlock()->owningNode();
    // n2 contains n1
  }

  for (; d_2 > d_1; --d_2) {
    n2 = n2->owningBlock()->owningNode();
  }

  // Now they are the same numer of blocks from the graph block,
  // recurse upwards, checking if they are on the same block
  while (true) {
    if (n1->owningBlock() == n2->owningBlock()) {
      return n1->owningBlock();
    }

    n1 = n1->owningBlock()->owningNode();
    n2 = n2->owningBlock()->owningNode();

    AT_ASSERT(n1 != nullptr);
    AT_ASSERT(n2 != nullptr);
  }
}

size_t Node::blocksFromGraphBlock() {
  Node* n = this;
  size_t dist = 0;
  while (n->owningBlock()->owningNode()) {
    n = n->owningBlock()->owningNode();
    ++dist;
  }
  return dist;
}

inline const SourceRange& fakeRange() {
  static SourceRange range(std::make_shared<Source>(""), 0, 1);
  return range;
}

Value* Graph::insert(
    Symbol opname,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<SourceRange>& range) {
  return script::emitBuiltinCall(
      range.value_or(fakeRange()), *this, opname, args, kwargs);
}

Node* Graph::create(NodeKind kind, size_t num_outputs) {
  // NB: Node constructor adds node to all_nodes
  auto n = new Node(this, kind);
  for (size_t i = 0; i < num_outputs; i++) {
    n->addOutput();
  }
  return n;
}

Node* Graph::create(
    NodeKind kind,
    ArrayRef<Value*> inputs,
    size_t num_outputs) {
  auto n = create(kind, num_outputs);
  for (auto i : inputs) {
    n->addInput(i);
  }
  return n;
}

Node* Graph::createAutogradZero() {
  return create(prim::AutogradZero);
}

Node* Graph::createNone() {
  Node* n = create(prim::Constant);
  n->output()->setType(NoneType::get());
  return n;
}

Node* Graph::createUninitialized(TypePtr typ) {
  Node* n = create(prim::Uninitialized);
  n->output()->setType(std::move(typ));
  return n;
}

Node* Graph::createWithSubgraph(Symbol kind) {
  auto n = create(kind, 0);
  n->g_(attr::Subgraph, std::make_shared<Graph>(current_scope()));
  return n;
}

Node* Graph::createTuple(at::ArrayRef<Value*> values, TupleTypePtr tuple_type) {
  TORCH_INTERNAL_ASSERT(
      !tuple_type || tuple_type->schema(),
      "only pass tuple_type when creating a named tuple");
  if (!tuple_type) {
    auto types = fmap(values, [](Value* v) { return v->type(); });
    tuple_type = TupleType::create(std::move(types));
  }
  auto n = create(prim::TupleConstruct, values);
  n->output()->setType(tuple_type);
  return n;
}

Node* Graph::createTupleUnpack(Value* v) {
  TupleTypePtr tt = v->type()->expect<TupleType>();
  auto n = create(prim::TupleUnpack, {v}, 0);
  for (auto& element : tt->elements()) {
    n->addOutput()->setType(element);
  }
  return n;
}

Node* Graph::createTupleIndex(
    Value* tup,
    Value* idx,
    const TypePtr& output_type) {
  auto n = create(prim::TupleIndex, {tup, idx});
  n->output()->setType(output_type);
  return n;
}

Node* Graph::createTupleSlice(Value* tup, int64_t beg, int64_t end) {
  auto n = create(prim::TupleSlice, {tup});
  auto tuple_type = tup->type()->expect<TupleType>();
  n->i_(attr::beg, beg);
  n->i_(attr::end, end);
  std::vector<TypePtr> output_types;
  for (auto i = beg; i < end; ++i) {
    output_types.push_back(tuple_type->elements().at(i));
  }
  auto tt = TupleType::create(std::move(output_types));
  n->output()->setType(tt);
  return n;
}

Node* Graph::createList(const TypePtr& elem_type, at::ArrayRef<Value*> values) {
  auto n = create(prim::ListConstruct, values);
  for (const auto& v : values) {
    TORCH_CHECK(
        v->type()->isSubtypeOf(elem_type),
        "Expected a list element that subtypes '",
        elem_type->python_str(),
        "' but got an element of type '",
        v->type()->python_str(),
        "'");
  }
  n->output()->setType(ListType::create(elem_type));
  return n;
}
Node* Graph::createListUnpack(Value* v, size_t size) {
  ListTypePtr list_type = v->type()->expect<ListType>();
  TypePtr elem_type = list_type->getElementType();
  auto n = create(prim::ListUnpack, {v}, 0);
  for (size_t i = 0; i < size; ++i) {
    n->addOutput()->setType(elem_type);
  }
  return n;
}

Node* Graph::createDict(
    const TypePtr& key_type,
    const TypePtr& value_type,
    at::ArrayRef<Value*> keys,
    at::ArrayRef<Value*> values) {
  AT_ASSERT(keys.size() == values.size());
  auto n = create(prim::DictConstruct, 1);
  for (size_t i = 0; i < keys.size(); ++i) {
    AT_ASSERT(keys[i]->type()->isSubtypeOf(key_type));
    AT_ASSERT(values[i]->type()->isSubtypeOf(value_type));

    n->addInput(keys[i]);
    n->addInput(values[i]);
  }
  n->output()->setType(DictType::create(key_type, value_type));
  return n;
}

Node* Graph::createNumToTensor(Value* value) {
  auto typ = value->type();
  Node* result = create(prim::NumToTensor, {value});
  result->output()->setType(TensorType::fromNumberType(std::move(typ)));
  return result;
}

Node* Graph::createImplicitTensorToNum(const TypePtr& type, Value* value) {
  auto* result = create(prim::ImplicitTensorToNum, {value});
  result->output()->setType(type);
  return result;
}

Node* Graph::createObject(const ClassTypePtr& type) {
  auto result = create(prim::CreateObject);
  result->output()->setType(type);
  return result;
}

Node* Graph::createSetAttr(
    Value* obj,
    const std::string& field,
    Value* newValue) {
  auto n = create(prim::SetAttr, {obj, newValue}, /*num_outputs=*/0);
  n->s_(attr::name, field);
  return n;
}

Node* Graph::createGetAttr(Value* obj, const std::string& field) {
  const auto classType = obj->type()->expect<ClassType>();

  auto n = create(prim::GetAttr, {obj}, /*num_outputs=*/1);
  n->s_(attr::name, field);

  const auto outputType = classType->getAttribute(field);
  n->output()->setType(outputType);
  return n;
}

Node* Graph::createStore(const std::string& name, Value* v) {
  auto n = create(prim::Store, {v}, /*num_outputs*/ 0);
  n->s_(attr::name, name);
  return n;
}

Node* Graph::createLoad(const std::string& name, const TypePtr& type) {
  auto n = create(prim::Load, {}, /*num_outputs*/ 1);
  n->s_(attr::name, name);
  n->output()->setType(type);
  return n;
}

Node* Graph::createIsInstance(
    Value* v,
    at::ArrayRef<TypePtr> types,
    bool is_list,
    bool is_tuple) {
  auto n = create(prim::isinstance, {v}, /*num_outputs*/ 1);
  std::vector<std::string> kinds;
  if (is_list) {
    kinds.push_back("list");
  }
  if (is_tuple) {
    kinds.push_back("tuple");
  }
  n->ss_(attr::kinds, std::move(kinds));
  n->tys_(attr::types, types.vec());
  n->output()->setType(BoolType::get());
  return n;
}
Value* Graph::insertUncheckedCast(Value* v, TypePtr type) {
  Node* n = insertNode(create(prim::unchecked_cast, {v}));
  n->output()->setType(std::move(type));
  return n->output();
}

Value* Graph::insertFunctionCall(
    Function* callee,
    const script::MatchedSchema& matched) {
  std::string func_name = callee->name();
  Value* fn_constant = insertNode(create(prim::Constant))
                           ->s_(attr::name, func_name)
                           ->output()
                           ->setType(FunctionType::create(std::move(callee)));
  std::vector<Value*> inputs = {fn_constant};
  inputs.insert(inputs.end(), matched.inputs.begin(), matched.inputs.end());
  Value* result = insertNode(create(prim::CallFunction, inputs))
                      ->output()
                      ->setType(matched.return_types.at(0));
  return result;
}

Value* Graph::insertMethodCall(
    std::string method_name,
    const script::MatchedSchema& matched) {
  Value* result = insertNode(create(prim::CallMethod, matched.inputs))
                      ->s_(attr::name, std::move(method_name))
                      ->output()
                      ->setType(matched.return_types.at(0));
  return result;
}

Node* Graph::createClone(
    Node* n,
    const std::function<Value*(Value*)>& value_map,
    bool copy_blocks) {
  // n can be from a different graph
  Node* r = n->allocNewInstance(this);
  for (auto o : n->outputs()) {
    r->addOutput()->copyMetadata(o);
  }
  r->cloneFrom(n);
  for (auto i : n->inputs()) {
    r->addInput(value_map(i));
  }
  if (copy_blocks) {
    for (auto b : n->blocks()) {
      r->addBlock()->cloneFrom(b, value_map);
    }
  }
  return r;
}

Value* Graph::insertConstant(
    IValue val,
    c10::optional<SourceRange> loc,
    c10::optional<ScopePtr> scope) {
  return jit::insertConstant(
      *this, std::move(val), std::move(loc), std::move(scope));
}

std::string Graph::toString(bool print_source_locations) const {
  std::ostringstream oss;
  print(oss, print_source_locations);
  return oss.str();
}

Graph::~Graph() {
  for (const Node* n : all_nodes) {
    delete n;
  }
  for (const Value* v : all_values) {
    delete v;
  }
  for (const Block* b : all_blocks) {
    delete b;
  }
}

void Graph::freeNode(Node* n) {
  auto it = all_nodes.find(n);
  AT_ASSERT(it != all_nodes.end());
  delete *it;
  all_nodes.erase(it);
}
void Graph::freeValue(Value* v) {
  v->setDebugName("");
  auto it = all_values.find(v);
  AT_ASSERT(it != all_values.end());
  delete *it;
  all_values.erase(it);
}
void Graph::freeBlock(Block* b) {
  auto it = all_blocks.find(b);
  AT_ASSERT(it != all_blocks.end());
  delete *it;
  all_blocks.erase(it);
}

void Node::setCallStack(InlinedCallStackPtr cs) {
  callstack_ = cs;
}

static InlinedCallStackPtr getOrCreateCallStackEntry(
    InlinedCallStackPtr cs,
    Function* f,
    const SourceRange& sr) {
  return cs ? cs : c10::make_intrusive<InlinedCallStack>(f, sr);
}

void Node::insertCallStackEntry(Function* f, const SourceRange& sr) {
  AT_ASSERT(callstack_);
  callstack_ = (*callstack_)->insertCallStackEntry(f, sr);
}

at::ArrayRef<Value*> createTupleUnpack(Value* v) {
  // small peephole optimization to ensure IntArrayRef attributes can still turn
  // into constants e.g. in x.expand([3, 4])
  if (v->node()->kind() == prim::TupleConstruct) {
    return v->node()->inputs();
  }
  auto& g = *v->owningGraph();
  return g.insertNode(g.createTupleUnpack(v))->outputs();
}

std::vector<Value*> inlineCallTo(Node* to_replace, Function* callee) {
  WithInsertPoint guard(to_replace);
  std::unordered_map<Value*, Value*> value_map;
  auto new_outputs = insertGraph(
      *to_replace->owningGraph(),
      *(callee->optimized_graph()),
      to_replace->inputs(),
      value_map);

  // TODO: We might need to use nodes_map instead of value_map. Otherwise, we
  // are missing nodes without outputs (e.g. prim::Print).
  std::unordered_set<Node*> updated_nodes;
  InlinedCallStackPtr new_callstack_entry;
  for (const auto& kv : value_map) {
    Node* new_node = kv.second->node();
    if (updated_nodes.insert(new_node).second) {
      if (!new_node->callstack()) {
        new_callstack_entry = getOrCreateCallStackEntry(
            new_callstack_entry, callee, to_replace->sourceRange());
        new_node->setCallStack(new_callstack_entry);
      } else {
        new_node->insertCallStackEntry(callee, to_replace->sourceRange());
      }
    }
  }

  const auto& old_outputs = to_replace->outputs();

  AT_ASSERT(new_outputs.size() == old_outputs.size());
  for (size_t i = 0; i < old_outputs.size(); ++i) {
    if (old_outputs[i]->hasDebugName()) {
      new_outputs[i]->setDebugName(old_outputs[i]->debugName());
    }
    old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
  }
  to_replace->destroy();

  return new_outputs;
}

std::vector<Value*> unpackOutputs(const std::vector<Value*>& outputs) {
  std::vector<Value*> new_outputs;
  if (outputs.size() != 1 || outputs.at(0)->type()->kind() != TupleType::Kind) {
    return outputs;
  }

  auto tup = outputs[0];
  for (Value* v : createTupleUnpack(tup)) {
    new_outputs.emplace_back(v);
  }
  // if this was a peephole tuple unpack we can just get rid of
  // the tuple construct here and prevent needing DCE
  if (tup->node()->kind() == prim::TupleConstruct && !tup->node()->hasUses()) {
    tup->node()->destroy();
  }
  return new_outputs;
}

std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs,
    std::unordered_map<Value*, Value*>& value_map) {
  auto value_map_func = [&](Value* v) { return value_map.at(v); };
  AT_ASSERT(callee.inputs().size() == inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    value_map[callee.inputs()[i]] = inputs[i];
  }
  for (auto* node : callee.nodes()) {
    auto* new_node = g.insertNode(g.createClone(node, value_map_func));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_map[node->outputs()[i]] = new_node->outputs()[i];
    }
  }

  std::vector<Value*> outputs;
  for (auto* output : callee.outputs()) {
    outputs.push_back(value_map_func(output));
  }

  return outputs;
}

std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs) {
  std::unordered_map<Value*, Value*> value_map;
  return insertGraph(g, callee, inputs, value_map);
}

void ProfileOp::cloneFrom(Node* other_) {
  Node::cloneFrom(other_);
  auto other = other_->cast<ProfileOp>();
  this->callback_ = other->getCallback();
}
Node* ProfileOp::allocNewInstance(Graph* g) {
  return new ProfileOp(g, {nullptr});
}

TypePtr NamedValue::type() const {
  if (value_) {
    return value_->type();
  } else {
    return ivalue_.type();
  }
}

constexpr Symbol ProfileOp::Kind;
} // namespace jit
} // namespace torch
