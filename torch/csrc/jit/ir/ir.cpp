#include <torch/csrc/jit/ir/ir.h>

#include <ATen/core/builtin_function.h>
#include <ATen/core/function.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/serialization/python_print.h>

#include <algorithm>
#include <iostream>
#include <locale>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace torch::jit {

namespace utils {
std::string getNodesModuleHierarchy(const Node& n) {
  if (!n.callstack().has_value()) {
    return std::string();
  }
  InlinedCallStackPtr callstack_ptr = n.callstack().value();
  std::string module_hierarchy;
  for (auto& entry : callstack_ptr->vec()) {
    const auto& opt_module_info = std::get<kModuleInstanceInfo>(entry);
    if (opt_module_info.has_value()) {
      const auto& module_instance_info = opt_module_info.value();
      if (!module_hierarchy.empty()) {
        module_hierarchy.append(".");
      }
      module_hierarchy.append(utils::get_module_info(module_instance_info));
    } else {
      module_hierarchy += ".UNKNOWN_INSTANCE(UNKNOWN_TYPE)";
    }
  }
  return module_hierarchy;
}
} // namespace utils

namespace {

// Constants relating to maintaining the topological index of nodes.
//
// Lower and upper bounds of the index. Inclusive range.
constexpr topo_position_t kLowerBound = INT64_MIN;
constexpr topo_position_t kUpperBound = INT64_MAX;
constexpr topo_position_t kMidPoint = 0;

// How far away to space nodes that are appended to the graph.
// should be 2^n, where:
//   - n is the maximum number of repeated insertions without a re-index
//   - 2^(64-n) is the maximum number of appends to the end without reindex
constexpr topo_position_t kAppendInterval = 1099511627776ULL /* 2^40 */;

void printValueRef(std::ostream& out, const Value* n) {
  out << "%" << n->debugName();
}

bool isNumber(c10::string_view str) {
  return str.find_first_not_of("0123456789") == std::string::npos;
}

std::string normalizeAttrName(c10::string_view field) {
  if (isNumber(field)) {
    return "_" + std::string{field};
  }
  return std::string{field};
}

void findAllNodes(
    Block& block,
    Symbol kind,
    bool recurse,
    std::vector<Node*>& ret) {
  for (Node* n : block.nodes()) {
    if (n->kind() == kind) {
      ret.push_back(n);
    }
    if (recurse) {
      for (auto b : n->blocks()) {
        findAllNodes(*b, kind, recurse, ret);
      }
    }
  }
}

} // namespace

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
    const at::ArrayRef<T> nodes) {
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

static std::ostream& operator<<(
    std::ostream& out,
    const at::ArrayRef<const Value*> nodes) {
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

static std::ostream& operator<<(
    std::ostream& out,
    const const_value_list_with_types& l) {
  size_t i = 0;
  for (auto n : l.values) {
    if (i++ > 0) {
      out << l.delim;
    }
    printValueRef(out, n);
    if (c10::type_verbosity() >= c10::TypeVerbosity::Type) {
      out << " : ";
      out << *n->type();
    }
  }
  return out;
}

static void printAttribute(std::ostream& out, const at::Tensor& tensor) {
  // 1-elem tensors are usually boxed scalars, so print them like it
  if (tensor.numel() == 1) {
    auto scalar_tensor = tensor.view(std::vector<int64_t>{}).item();
    out << "{";
    if (scalar_tensor.isFloatingPoint()) {
      out << scalar_tensor.toDouble();
    } else if (scalar_tensor.isComplex()) {
      out << scalar_tensor.toComplexDouble();
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
}

static void printAttribute(std::ostream& out, const IValue& ival) {
  const auto customFormatter = [](std::ostream& ss, const IValue& input) {
    if (input.isTensor()) {
      printAttribute(ss, input.toTensor());
      return true;
    } else if (input.isTensorList()) {
      ss << "[<Tensors>]";
      return true;
    } else if (input.isObject() && !input.type()->is_module()) {
      ss << "object(" << &input.toObjectRef() << ")";
      return true;
    }
    return false;
  };
  ival.repr(out, customFormatter);
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
    case AttributeKind::c:
      printAttribute(out, c(name));
      break;
    case AttributeKind::cs:
      // TODO(@anjali411): fix this
      AT_ASSERT(false);
      break;
    case AttributeKind::f:
      printAttribute(out, f(name));
      break;
    case AttributeKind::fs:
      printAttribute(out, fs(name));
      break;
    case AttributeKind::i:
      printAttribute(out, i(name));
      break;
    case AttributeKind::is:
      printAttribute(out, is(name));
      break;
    case AttributeKind::s:
      printAttribute(out, s(name));
      break;
    case AttributeKind::ss:
      printAttribute(out, ss(name));
      break;
    case AttributeKind::t:
      printAttribute(out, t(name));
      break;
    case AttributeKind::ts:
      out << "[<Tensors>]";
      break;
    case AttributeKind::ival:
      printAttribute(out, ival(name));
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

void Node::printAttributes(std::ostream& out, bool ignore_subgraph = false)
    const {
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
  for (const auto i : c10::irange(level)) {
    (void)i; // Suppress unused variable warning
    out << "  ";
  }
  return out;
}

std::ostream& Node::print(
    std::ostream& out,
    size_t level,
    std::vector<const Node*>* groups,
    bool print_source_locations,
    bool print_attributes,
    bool print_scopes,
    bool print_body) const {
  auto outs = outputs();
  indent(out, level) << const_value_list_with_types(outs);
  out << " = ";
  if (kind() == prim::PythonOp) {
    auto* pyOp = static_cast<const ::torch::jit::PythonOp*>(this);
    out << "^" << pyOp->name();
    printAttributes(out, /*ignore_subgraph=*/false);
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
      auto [filename, line, col] = *file_line_col;
      out << " # " << filename << ":" << line << ":" << col;
    }
  }

  if (!print_body) {
    return out;
  }

  out << "\n";

  for (const auto i : c10::irange(blocks().size())) {
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
  out.flush();

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
  std::optional<at::Device> device = std::nullopt;
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
    for (auto use : o->uses()) {
      // Use invariants
      // - Use is consistent with inputs
      // - Every user node is live (checked in Graph)
      AT_ASSERT(use.user->inputs_[use.offset] == o);
    }
  }

  // Node subclass invariants
  switch (kind()) {
    case prim::Constant:
      AT_ASSERT(inputs_.empty());
      break;
    case prim::Return:
      // Return uses is zero
      AT_ASSERT(outputs().empty());
      break;
    case prim::Param:
      // Param inputs is zero
      AT_ASSERT(inputs_.empty());
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
    case prim::CudaFusionGroup:
    case prim::oneDNNFusionGroup:
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
  // - every use will occur later in the toposort

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
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
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
        scope = std::make_unique<LintScope>(std::move(scope));
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

void Graph::cloneFrom(Graph& src) {
  auto env = [](Value* v) -> Value* {
    AT_ERROR(
        "Graph::copy() encountered a use of a value " + v->debugName() +
        " not in scope. Run lint!");
  };
  block()->cloneFrom(src.block(), env);
}

std::shared_ptr<Graph> Graph::copy() {
  auto new_g = std::make_shared<Graph>();
  new_g->cloneFrom(*this);
  return new_g;
}

std::unique_ptr<Graph> Graph::copyUnique() {
  auto new_g = std::make_unique<Graph>();
  new_g->cloneFrom(*this);
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

void Value::inferTypeFrom(
    const c10::intrusive_ptr<c10::ivalue::Object>& output) {
  setType(output->type());
}

bool Value::mustBeNone() const {
  return type()->cast<NoneType>() || node_->mustBeNone();
}
bool Value::mustNotBeNone() const {
  return node_->kind() != prim::AutogradAdd && type() != NoneType::get() &&
      !type()->cast<OptionalType>() &&
      !(type()->cast<UnionType>() &&
        type()->expect<UnionType>()->canHoldType(*NoneType::get()));
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
  if (name.empty()) {
    return true;
  }

  // Numbers are not legal
  if (isNumber(name)) {
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
  if (name.empty()) {
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
        suffix = std::stoll(name.substr(last_dot_pos + 1));
        name_base = name.substr(0, last_dot_pos);
      }
    }

    auto& names_suffixes = node()->owningGraph()->name_base_suffix_;
    auto it = names_suffixes.find(name_base);
    if (it != names_suffixes.end()) {
      suffix = std::max(suffix, it->second + 1);
    }

    // Verify that new name is not used and find next usable name in case
    // suffix is used.
    std::string replacement_name;
    do {
      std::stringstream ss;
#ifndef _WIN32
      // Protect 12345 integer from becoming "1,2345" if some other process sets
      // global locale For more details see
      // https://github.com/pytorch/pytorch/issues/79583#issuecomment-1161260061
      static std::locale c_locale("C");
      ss.imbue(c_locale);
#endif
      ss << name_base << "." << suffix++;
      replacement_name = ss.str();
    } while (names.count(replacement_name) > 0);

    names_suffixes[name_base] = suffix;

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

void Value::replaceAllUsesAfterNodeWith(const Node* node, Value* newValue) {
  std::for_each(uses_.begin(), uses_.end(), [&node, newValue](Use& u) {
    if (u.user->isAfter(node)) {
      u.user->inputs_[u.offset] = newValue;
      newValue->uses_.push_back(u);
    }
  });

  uses_.erase(
      std::remove_if(
          uses_.begin(),
          uses_.end(),
          [&node](const Use& u) { return u.user->isAfter(node); }),
      uses_.end());
}

void Value::replaceAllUsesDominatedByNodeWith(
    const Node* node,
    Value* newValue) {
  std::for_each(uses_.begin(), uses_.end(), [&node, newValue](Use& u) {
    if (u.user->isDominatedBy(node)) {
      u.user->inputs_[u.offset] = newValue;
      newValue->uses_.push_back(u);
    }
  });

  uses_.erase(
      std::remove_if(
          uses_.begin(),
          uses_.end(),
          [&node](const Use& u) { return u.user->isDominatedBy(node); }),
      uses_.end());
}

static size_t findArgument(
    const FunctionSchema& the_schema,
    const std::string& unqualName) {
  for (const auto i : c10::irange(the_schema.arguments().size())) {
    const Argument* arg = &the_schema.arguments()[i];
    if (arg->name() == unqualName) {
      return i;
    }
  }
  throw std::runtime_error(
      std::string("Couldn't find an argument called ") + unqualName);
}

static size_t findArgument(const FunctionSchema& the_schema, Symbol name) {
  const auto unqualName = name.toUnqualString();
  return findArgument(the_schema, unqualName);
}

std::optional<IValue> Node::get(Symbol name) const {
  return toIValue(namedInput(name));
}

bool Node::hasNamedInput(const std::string& name) const {
  for (const auto& argument : schema().arguments()) {
    if (argument.name() == name) {
      return true;
    }
  }
  return false;
}

Value* Node::namedInput(const std::string& unqualName) const {
  return input(findArgument(schema(), unqualName));
}
Value* Node::namedInput(Symbol name) const {
  return input(findArgument(schema(), name));
}

bool Node::matches(const FunctionSchema& schema) const {
  if (isBlockListedSchema(schema)) {
    return false;
  }
  // wrong name
  if (kind().toQualString() != schema.name()) {
    return false;
  }
  at::ArrayRef<const Value*> actuals = inputs();
  const auto& formals = schema.arguments();

  // not enough inputs
  if (actuals.size() < formals.size()) {
    return false;
  }

  TypeEnv type_env;
  for (const auto i : c10::irange(formals.size())) {
    auto formal = formals[i].type();
    const MatchTypeReturn matched_type =
        matchTypeVariables(formal, actuals[i]->type(), type_env);
    if (!matched_type.success()) {
      return false;
    }

    TypePtr resolved = tryEvalTypeVariables(formal, type_env);
    if (resolved) {
      formal = resolved;
    }
    // note: it is possible at this point that type variable matching has
    // not resolved all type variables, e.g. if None was matched to Optional[T]
    // we will not succeed at matching T. However None <: Optional[T] so this
    // check can still succeed.

    if (!actuals[i]->type()->isSubtypeOf(*formal)) {
      return false;
    }
  }

  // too many inputs
  if (!schema.is_vararg() && actuals.size() != formals.size()) {
    return false;
  }

  return true;
}

bool Node::matches(
    const char* signature_literal,
    at::ArrayRef<Symbol> const_inputs) const {
  if (!matches(getOperatorForLiteral(signature_literal)->schema())) {
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
  return getOperator().schema();
}

const FunctionSchema* Node::maybeSchema() const {
  if (auto op = maybeOperator()) {
    return &op->schema();
  }
  return nullptr;
}

const Operator* Node::maybeOperator() const {
  if (!op_) {
    const auto& candidates = getAllOperatorsFor(kind());
    for (const auto& candidate : candidates) {
      if (matches(candidate->schema())) {
        op_ = candidate.get();
        break;
      }
    }
  }
  return op_;
}

const Operator& Node::getOperator() const {
  const Operator* maybe = maybeOperator();
  if (maybe)
    return *maybe;

  auto er = ErrorReport(sourceRange());
  er << "Schema not found for node. File a bug report.\n";
  er << "Node: " << *this << "\n";
  er << "Input types:";
  for (const auto i : c10::irange(inputs().size())) {
    if (i > 0)
      er << ", ";
    er << *inputs()[i]->type();
  }
  const auto& candidates = getAllOperatorsFor(kind());
  if (!candidates.empty()) {
    er << "\ncandidates were:\n";
    for (auto& candidate : candidates) {
      er << "  " << candidate->schema() << "\n";
    }
  } else {
    er << "\nno candidates found\n";
  }
  er << "within the graph:\n";
  er << *owningGraph() << "\n";
  throw er;
}

Operation Node::getOperation() const {
  // note: some operators require the node to produce a runnable operation,
  // which is why 'this' is passed here. getOperator() ensures that 'this'
  // matches the schema of the returned operator.
  return getOperator().getOperation(this);
}

bool Node::isNondeterministic() const {
  const auto schema = maybeSchema();
  if (!kind().is_aten()) {
    return false;
  }
  // All aten ops are expecte to have a schema. However this is left as a
  // warning instead of an assert to ensure that previous use cases do not
  // break.
  if (!schema) {
    TORCH_WARN("aten Schema not found.");
    return false;
  }
  torch::utils::SchemaInfo schema_info(*schema);
  if (hasNamedInput("train")) {
    auto value = constant_as<bool>(namedInput("train"));
    if (value.has_value()) {
      schema_info.addArgumentValue("train", *value);
    }
  }
  return schema_info.is_nondeterministic();
}

bool Node::hasSideEffects() const {
  switch (kind_) {
    case prim::PythonOp:
    case prim::IgnoredPythonOp:
    case prim::Print:
    case prim::RaiseException:
    case aten::warn:
    case aten::save:
    case aten::manual_seed:
    case prim::AddStatValue:
    case prim::TimePoint:
    case prim::CallFunction:
    case prim::CallMethod:
    case prim::BailoutTemplate:
    case prim::BailOut:
    case prim::rpc_async: // It represents RPC message sent.
    case prim::rpc_sync: // It represents RPC message sent.
    case prim::rpc_remote: // It represents RPC message sent.
    case aten::wait: // It can represent RPC message received.
#if !defined(USE_ROCM)
    case cuda::set_stream:
    case cuda::_set_device:
    case cuda::_current_device:
    case cuda::synchronize:
#endif
    case prim::Enter:
    case prim::Exit:
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

  if (kind_.is_prim() || kind_.is_aten() || kind_.is_cuda()) {
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
    case AliasAnalysisKind::FROM_SCHEMA:
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
    int64_t remaining = nextPos - prevPos;
    AT_ASSERT(remaining > 0);
    if (remaining == 1) {
      // There was no room
      owningBlock()->reIndexTopology();
      return;
    }
    int64_t predicted_future_insertions = 0;
    if (next() == graph_->insertPoint()) {
      predicted_future_insertions = graph_->predicted_insert_count_++;
    }
    topo_position_ = prevPos +
        std::max(int64_t(1), remaining / (2 + predicted_future_insertions));
    AT_ASSERT(prevPos < topo_position_ && topo_position_ < nextPos);
  }
}

Node::Node(Graph* graph_, NodeKind kind_)
    : kind_(kind_),
      graph_(graph_),
      owning_block_(nullptr),
      scope_(graph_->current_scope_),
      callstack_(std::nullopt),
      op_(nullptr)
      {
  graph_->all_nodes.emplace(this);
}

void Node::eraseOutput(size_t i) {
  AT_ASSERT(i < outputs_.size());
  AT_ASSERT(outputs_[i]->uses().empty());
  op_ = nullptr;
  Value* n = outputs_[i];
  outputs_.erase(outputs_.begin() + i);
  owningGraph()->freeValue(n);
  for (const auto j : c10::irange(i, outputs_.size())) {
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
  for (const auto i : c10::irange(nOutputs)) {
    outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
  }
}

Node* Node::replaceWithNewSymbol(Symbol new_symbol) {
  WithInsertPoint insert_guard{this};
  bool had_operator = maybeOperator() != nullptr;
  auto graph = owningGraph();
  auto replace_node = graph->insertNode(graph->create(new_symbol, 0));
  for (Value* v : inputs()) {
    replace_node->addInput(v);
  }
  for (Value* v : outputs()) {
    auto new_out = replace_node->addOutput()->copyMetadata(v);
    v->replaceAllUsesWith(new_out);
  }
  replace_node->copyMetadata(this);
  replace_node->copyAttributes(*this);
  TORCH_INTERNAL_ASSERT(
      (replace_node->maybeOperator() != nullptr) == had_operator,
      "invalid symbol replacement:",
      new_symbol,
      kind());
  return replace_node;
}

bool Node::isDominatedBy(const Node* dominator) const {
  const Node* node = this;
  while (node) {
    if (node->owningBlock() == dominator->owningBlock()) {
      return dominator->isBefore(node);
    }
    node = node->owningBlock()->owningNode();
  }
  return false;
}

Value* Node::insertInput(size_t i, Value* value) {
  AT_ASSERT(graph_ == value->owningGraph());
  op_ = nullptr;
  // First we update the offsets for all existing inputs that will reside
  // after the one we're inserting. Concretely, these are the inputs at
  // indices [i, # input). Since we're inserting one input before all of
  // these inputs, increment their use offsets for this value by 1
  for (const auto use_itr : c10::irange(i, inputs_.size())) {
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
      "Attempting to insert a Node after the Return node or before the Param node. Tried to insert",
      *this,
      " after ",
      *n,
      ".");
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
  for (const auto i : c10::irange(inputs().size())) {
    dropInput(i);
  }
  inputs_.clear();
}

void Node::removeAllOutputs() {
  op_ = nullptr;
  size_t init_osize = outputs_.size();
  for (auto i : c10::irange(init_osize)) {
    eraseOutput(init_osize - i - 1);
  }
}

void Node::permuteInputs(const std::vector<size_t>& new_order) {
  op_ = nullptr;
  AT_ASSERT(new_order.size() == inputs_.size());
  std::vector<Value*> new_inputs;
  new_inputs.reserve(new_order.size());
  for (const auto i : c10::irange(new_order.size())) {
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
  for (const auto i : c10::irange(new_order.size())) {
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
  static SourceRange range(std::make_shared<Source>(std::string("")), 0, 1);
  return range;
}

Value* Graph::insert(
    Symbol opname,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const std::optional<SourceRange>& range) {
  return emitBuiltinCall(
      range.value_or(fakeRange()), *this, opname, args, kwargs);
}

Node* Graph::create(NodeKind kind, size_t num_outputs) {
  // NB: Node constructor adds node to all_nodes
  auto n = new Node(this, kind);
  for (const auto i : c10::irange(num_outputs)) {
    (void)i;
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

Node* Graph::createTupleSlice(
    Value* tup,
    int64_t beg,
    int64_t step_size,
    int64_t num_values) {
  std::vector<Value*> new_vals;
  TupleTypePtr tt = tup->type()->expect<TupleType>();
  new_vals.reserve(num_values);

  int64_t i = beg;
  for (const auto j : c10::irange(num_values)) {
    (void)j; // Suppress unused variable warning
    auto idx = insertConstant(IValue(static_cast<int64_t>(i)));
    auto tupleIndex = insertNode(createTupleIndex(tup, idx, tt->elements()[i]));

    new_vals.push_back(tupleIndex->output());
    i += step_size;
  }

  auto n = createTuple(new_vals);
  return n;
}

Node* Graph::createEnumName(Value* e) {
  e->type()->expect<EnumType>();
  assert(e->type()->cast<EnumType>());
  auto n = create(prim::EnumName, {e});
  n->output()->setType(StringType::get());
  return n;
}

Node* Graph::createEnumValue(Value* e) {
  auto enum_type = e->type()->expect<EnumType>();
  auto n = create(prim::EnumValue, {e});
  n->output()->setType(enum_type->getValueType());
  return n;
}

Node* Graph::createList(
    const TypePtr& contained_type,
    at::ArrayRef<Value*> values) {
  auto n = create(prim::ListConstruct, values);
  for (const auto& v : values) {
    TORCH_CHECK(
        v->type()->isSubtypeOf(*contained_type),
        "Expected a list element that subtypes '",
        contained_type->repr_str(),
        "' but got an element of type '",
        v->type()->repr_str(),
        "'");
  }
  n->output()->setType(ListType::create(contained_type));
  return n;
}

Node* Graph::createListUnpack(Value* v, size_t size) {
  ListTypePtr list_type = v->type()->expect<ListType>();
  TypePtr elem_type = list_type->getElementType();
  auto n = create(prim::ListUnpack, {v}, 0);
  for (const auto i : c10::irange(size)) {
    (void)i; // Suppress unused variable warning
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
  for (const auto i : c10::irange(keys.size())) {
    AT_ASSERT(keys[i]->type()->isSubtypeOf(*key_type));
    AT_ASSERT(values[i]->type()->isSubtypeOf(*value_type));

    n->addInput(keys[i]);
    n->addInput(values[i]);
  }
  n->output()->setType(DictType::create(key_type, value_type));
  return n;
}

Node* Graph::createNumToTensor(Value* value) {
  Node* result = create(prim::NumToTensor, {value});
  result->output()->setType(TensorType::fromNumberType(*value->type()));
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
  n->output()->setDebugName(normalizeAttrName(field));
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

Node* Graph::createIsInstance(Value* v, at::ArrayRef<TypePtr> types) {
  auto n = create(prim::isinstance, {v}, /*num_outputs*/ 1);
  n->tys_(attr::types, types.vec());
  n->output()->setType(BoolType::get());
  return n;
}
Value* Graph::insertUncheckedCast(Value* v, TypePtr type) {
  Node* n = insertNode(create(prim::unchecked_cast, {v}));
  n->output()->setType(std::move(type));
  return n->output();
}

Value* Graph::insertToList(Value* v, TypePtr type) {
  int dim = 0;
  TypePtr ptr = type;

  // Unwrap the type to determine the number of dimensions.
  while (auto list_type = ptr->cast<ListType>()) {
    ptr = list_type->getElementType();
    ++dim;
  }

  // Encode the base element type as an integer.
  int elem_ty = 0;
  if (ptr == IntType::get()) {
    elem_ty = 0;
  } else if (ptr == FloatType::get()) {
    elem_ty = 1;
  } else if (ptr == BoolType::get()) {
    elem_ty = 2;
  } else if (ptr == ComplexType::get()) {
    elem_ty = 3;
  } else {
    TORCH_CHECK(
        false,
        ptr->repr_str(),
        " is not one of the supported element types for tolist: int, float, complex, bool");
  }

  // Pass in the number of dimensions and base element type as arguments
  // to the op.
  Value* dim_val = insertConstant(IValue(dim));
  Value* elem_ty_val = insertConstant(IValue(elem_ty));
  Node* n = insertNode(create(prim::tolist, {v, dim_val, elem_ty_val}));
  n->output()->setType(std::move(type));
  return n->output();
}

Value* Graph::insertFunctionCall(
    Function* callee,
    const MatchedSchema& matched) {
  std::string func_name = callee->name();
  Value* fn_constant = insertNode(create(prim::Constant))
                           ->s_(attr::name, func_name)
                           ->output()
                           ->setType(FunctionType::create(callee));
  std::vector<Value*> inputs = {fn_constant};
  inputs.insert(inputs.end(), matched.inputs.begin(), matched.inputs.end());
  Value* result = insertNode(create(prim::CallFunction, inputs))
                      ->output()
                      ->setType(matched.return_types.at(0));
  return result;
}

Value* Graph::insertMethodCall(
    std::string method_name,
    const MatchedSchema& matched) {
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
    const IValue& val,
    std::optional<SourceRange> loc,
    std::optional<ScopePtr> scope) {
  return jit::insertConstant(*this, val, std::move(loc), std::move(scope));
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

at::ArrayRef<Value*> createTupleUnpack(Value* v) {
  // small peephole optimization to ensure IntArrayRef attributes can still turn
  // into constants e.g. in x.expand([3, 4])
  if (v->node()->kind() == prim::TupleConstruct) {
    return v->node()->inputs();
  }
  auto& g = *v->owningGraph();
  return g.insertNode(g.createTupleUnpack(v))->outputs();
}

void inlineCallStackOfNode(
    Node* n,
    std::unordered_map<InlinedCallStack*, InlinedCallStackPtr>& new_cs_entries,
    Function* callee,
    Node* to_replace,
    const std::optional<ModuleInstanceInfo>& m_info);

static void inlineCallStackOfBlock(
    Block* b,
    std::unordered_map<InlinedCallStack*, InlinedCallStackPtr>& new_cs_entries,
    Function* callee,
    Node* to_replace,
    const std::optional<ModuleInstanceInfo>& m_info) {
  for (auto n : b->nodes()) {
    inlineCallStackOfNode(n, new_cs_entries, callee, to_replace, m_info);
  }
}

void inlineCallStackOfNode(
    Node* new_node,
    std::unordered_map<InlinedCallStack*, InlinedCallStackPtr>& new_cs_entries,
    Function* callee,
    Node* to_replace,
    const std::optional<ModuleInstanceInfo>& m_info) {
  auto new_node_cs = new_node->callstack();

  InlinedCallStack* raw_callstack_ptr =
      new_node_cs ? new_node_cs->get() : nullptr;

  if (!new_cs_entries.count(raw_callstack_ptr)) {
    if (new_node_cs) {
      new_cs_entries[raw_callstack_ptr] = c10::make_intrusive<InlinedCallStack>(
          *new_node_cs, callee, to_replace->sourceRange(), m_info);
    } else {
      new_cs_entries[raw_callstack_ptr] = c10::make_intrusive<InlinedCallStack>(
          callee, to_replace->sourceRange(), m_info);
    }
  }
  new_node->setCallStack(new_cs_entries.at(raw_callstack_ptr));
  // We updated the inlined callstack of new_node.
  // Same must be done for the nodes of the blocks of new_node.
  // For example If node's block otherwise is not annotated appropriately.
  for (auto block : new_node->blocks()) {
    inlineCallStackOfBlock(block, new_cs_entries, callee, to_replace, m_info);
  }
}

std::vector<Value*> inlineCallTo(
    Node* to_replace,
    GraphFunction* callee,
    Graph* callee_graph) {
  WithInsertPoint guard(to_replace);
  std::unordered_map<Value*, Value*> value_map;
  std::vector<torch::jit::Value*> new_outputs = insertGraph(
      *to_replace->owningGraph(),
      *callee_graph,
      to_replace->inputs(),
      value_map);

  std::unordered_map<InlinedCallStack*, InlinedCallStackPtr>
      new_callstack_entries;

  std::optional<ModuleInstanceInfo> module_instance_info = std::nullopt;
  if (to_replace->kind() == prim::CallMethod) {
    auto class_type_ptr = to_replace->input(0)->type()->cast<c10::ClassType>();
    if (to_replace->input(0)->node()->kind() == prim::GetAttr) {
      module_instance_info = std::make_optional(ModuleInstanceInfo(
          class_type_ptr, to_replace->input(0)->node()->s(attr::name)));
    } else if (
        !to_replace->owningGraph()->inputs().empty() &&
        to_replace->input(0) == to_replace->owningGraph()->inputs()[0]) {
      // This CallMethod must correspond to method of the same object
      // to which this graph belongs.
      module_instance_info =
          std::make_optional(ModuleInstanceInfo(class_type_ptr, "SELF"));
    } else {
      // Not sure if it is possible to come here ever.
      // TODO: Remove this else. Or add assert
      module_instance_info = std::make_optional(
          ModuleInstanceInfo(class_type_ptr, "INSTANCE_NAME_UNKNOWN"));
    }
  }

  // TODO: We might need to use nodes_map instead of value_map. Otherwise, we
  // are missing nodes without outputs (e.g. prim::Print).
  std::unordered_set<Node*> updated_nodes;
  for (const auto& kv : value_map) {
    /* Skip the old value if it is the graph input.
     * The reason is that, value_map contains values not all for the nodes of
     * the graph but primary inputs as well, and it will create duplicates when
     * the first inlined graph is input to the next one. To avoid this issue,
     * skip the old value when it is one of the
     * callee->optimized_graph()->inputs() or callee->graph()->inputs(), depends
     * on if it is inlined_optimized_graph
     */
    auto is_graph_input = std::find(
        callee_graph->inputs().begin(), callee_graph->inputs().end(), kv.first);
    if (is_graph_input != callee_graph->inputs().end()) {
      continue;
    }

    Node* new_node = kv.second->node();
    if (!updated_nodes.insert(new_node).second) {
      continue;
    }

    inlineCallStackOfNode(
        new_node,
        new_callstack_entries,
        callee,
        to_replace,
        module_instance_info);
  }
  const auto& old_outputs = to_replace->outputs();

  AT_ASSERT(new_outputs.size() == old_outputs.size());
  for (const auto i : c10::irange(old_outputs.size())) {
    if (old_outputs[i]->hasDebugName()) {
      new_outputs[i]->setDebugName(old_outputs[i]->debugName());
    }
    old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
  }
  to_replace->destroy();

  return new_outputs;
}

// inline_optimized_graph argument is used in substitute function call for
// ONNX conversion
std::vector<Value*> inlineCallTo(
    Node* to_replace,
    GraphFunction* callee,
    bool inline_optimized_graph /*=true*/) {
  auto graph =
      inline_optimized_graph ? callee->optimized_graph() : callee->graph();
  return inlineCallTo(to_replace, callee, graph.get());
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

std::vector<Node*> findAllNodes(
    at::ArrayRef<Block*> array,
    Symbol kind,
    bool recurse) {
  std::vector<Node*> ret;
  for (auto block : array) {
    findAllNodes(*block, kind, recurse, ret);
  }
  return ret;
}

std::vector<Node*> findAllNodes(Block& block, Symbol kind, bool recurse) {
  return findAllNodes({&block}, kind, recurse);
}

std::vector<Node*> findAllNodes(Graph& g, Symbol kind, bool recurse) {
  return findAllNodes(*g.block(), kind, recurse);
}

std::vector<Value*> insertGraph(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs,
    std::unordered_map<Value*, Value*>& value_map) {
  auto value_map_func = [&](Value* v) { return value_map.at(v); };
  AT_ASSERT(callee.inputs().size() == inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
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

void ProfileIValueOp::cloneFrom(Node* other_) {
  Node::cloneFrom(other_);
  auto other = other_->cast<ProfileIValueOp>();
  this->callback_ = other->getCallback();
}

Node* ProfileIValueOp::allocNewInstance(Graph* g) {
  return new ProfileIValueOp(g, {nullptr});
}

TypePtr NamedValue::type() const {
  if (value_) {
    return value_->type();
  } else {
    return ivalue_.type();
  }
}

const Symbol ProfileOp::Kind = ::c10::prim::profile;
const Symbol ProfileIValueOp::Kind = ::c10::prim::profile_ivalue;

OperatorSet::OperatorSet(std::initializer_list<const char*> sig_literals) {
  insert(sig_literals);
}

std::vector<std::shared_ptr<Operator>> OperatorSet::getOps() const {
  std::vector<std::shared_ptr<Operator>> result;
  for (const auto& kv : ops) {
    auto ops_for_symbol = kv.second;
    result.insert(result.end(), ops_for_symbol.begin(), ops_for_symbol.end());
  }
  return result;
}

void OperatorSet::insert(std::initializer_list<const char*> sig_literals) {
  for (const char* sig : sig_literals) {
    auto op = getOperatorForLiteral(sig);
    ops[Symbol::fromQualString(op->schema().name())].push_back(op);
  }
}

bool Node::isMemberOf(const OperatorSet& os) const {
  auto it = os.ops.find(kind());
  if (it == os.ops.end()) {
    return false;
  }
  for (auto& op : it->second) {
    if (matches(op->schema())) {
      return true;
    }
  }
  return false;
}

} // namespace torch::jit
