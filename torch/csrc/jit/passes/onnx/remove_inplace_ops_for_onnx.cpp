#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>

#include <c10/util/irange.h>

#include <limits>

namespace torch::jit {

namespace {

const std::set<c10::Symbol> inplace_ops =
    {aten::append, aten::index_put_, aten::pop, aten::insert, aten::Delete};

// InplaceConverter defines a set of functions that together enables the
// conversion from prim::GetAttr, prim::SetAttr, and ATen in-place operators to
// ONNX out-place operators.
struct InplaceConverter {
  InplaceConverter(
      std::shared_ptr<Graph> graph,
      MutationRemover* mr,
      Module* model = nullptr)
      : graph_(std::move(graph)), mr_(mr), module_(model) {}

  void convertMutationForONNX();

 private:
  void gatherAttrNameInitialValueMap(
      Block* block,
      std::unordered_map<std::string, Value*>& attr_name_value_map,
      std::unordered_map<Node*, std::string>& attr_node_fullname_map);
  void replaceAttrWithInplaceOps(
      Block* block,
      const std::unordered_map<std::string, Value*>& attr_name_value_map,
      const std::unordered_map<Node*, std::string>& attr_node_fullname_map);

  void convertInplaceOpsAndTrackAlias();
  void convertInplaceOpsAndTrackAlias(Block* block);

  void correctAliasReferences();
  void correctAliasReferences(Block* block);
  void correctAliasReferences(Node* n);

  void convertGetSetAttrToInplaceOps(Block* block);

  // ValueTracker provides apis to record aliases for a single value,
  // and to retrieve the correct alias of any given value based on the location
  // in the graph it is used.
  struct ValueTracker {
    ValueTracker() : graph_(nullptr) {}

    void init(const std::shared_ptr<Graph>& graph);
    void recordSetValue(Value* old_v, Value* new_v);
    Value* findAliasForValueAtNode(Value* v, const Node* n) const;

    std::string toString() const;

   private:
    std::shared_ptr<Graph> graph_;

    // Map from aliases to root value.
    // A single value can have multiple aliases throughout the graph,
    // created by inplace operators, and preserved through loop carried
    // input/output. For each such value, its first occurrence will be set as
    // root value.
    std::unordered_map<Value*, Value*> alias_to_value_;

    // Sort the alias based on their order in graph.
    // A tie can happen when two distinct aliases belong to different blocks,
    // while having the same ancestor node. The unique id is used as tie
    // breaker, otherwise the two aliases will be considered equal to each
    // other. aliasComp must satisfy strict weak ordering.
    struct aliasComp {
      bool operator()(const Value* a, const Value* b) const {
        auto* n_a = a->node();
        auto* n_b = b->node();
        if (n_a == n_b) {
          return false;
        }
        auto a_b = n_a->isBefore(n_b);
        auto b_a = n_b->isBefore(n_a);
        if (a_b == b_a) {
          return a->unique() < b->unique();
        }
        return a_b;
      }
    };
    // Map from root value to aliases sorted by their order in graph.
    std::unordered_map<Value*, std::set<Value*, aliasComp>>
        value_to_sorted_aliases_;
  };

  std::shared_ptr<Graph> graph_;
  MutationRemover* mr_;
  Module* module_;
  ValueTracker vt_;
};

bool isAncestor(const Block* a, const Block* b) {
  while (b && b->owningNode()) {
    if (a == b) {
      return true;
    }
    b = b->owningNode()->owningBlock();
  }
  return a == b;
}

Node* addDummyClone(
    Graph* graph,
    Value* orig_data,
    bool insertBefore,
    Node* referenceNode) {
  Node* newNode = nullptr;
  if (orig_data->type()->kind() == TypeKind::ListType) {
    newNode = graph->create(aten::list, /*num_outputs =*/1);
    newNode->addInput(orig_data);
    newNode->output()->setType(orig_data->type());
    if (insertBefore)
      newNode->insertBefore(referenceNode);
    else
      referenceNode->owningBlock()->prependNode(newNode);
  } else if (
      orig_data->type()->kind() == TypeKind::TensorType ||
      orig_data->type()->kind() == TypeKind::IntType ||
      orig_data->type()->kind() == TypeKind::FloatType ||
      orig_data->type()->kind() == TypeKind::BoolType) {
    auto* noneNode = graph->create(prim::Constant);
    noneNode->output()->setType(NoneType::get());
    // For scripting mode, aten::clone requires input to be a TensorType
    // Hence if we encounter an IntType, FloatType, or BoolType,
    // we set the input to the appropriate TensorType
    if (orig_data->type()->kind() == TypeKind::IntType &&
        insertBefore == false) {
      orig_data->setType(TensorType::fromNumberType(*IntType::get()));
    } else if (
        orig_data->type()->kind() == TypeKind::FloatType &&
        insertBefore == false) {
      orig_data->setType(TensorType::fromNumberType(*FloatType::get()));
    } else if (
        orig_data->type()->kind() == TypeKind::BoolType &&
        insertBefore == false) {
      orig_data->setType(TensorType::fromBoolType());
    }
    newNode = graph->create(aten::clone, /*num_outputs =*/1);
    newNode->addInput(orig_data);
    newNode->addInput(noneNode->output());
    newNode->output()->setType(orig_data->type());
    if (insertBefore)
      newNode->insertBefore(referenceNode);
    else
      referenceNode->owningBlock()->prependNode(newNode);
    noneNode->insertBefore(newNode);
  }
  return newNode;
}

std::pair<Value*, Value*> PrepareIndexPutForONNX(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == aten::index_put || node->kind() == aten::index_put_);
  auto placeholder_node = EncapsulatePatternIntoSubblock(node).value();
  node->destroy();
  return std::make_pair(placeholder_node->input(0), placeholder_node->output());
}

std::pair<Value*, Value*> PrepareCopyForONNX(Node* node) {
  TORCH_INTERNAL_ASSERT(node->kind() == aten::copy_);
  // aten::copy_ can be viewed as a special case of index_put, where the
  // tensor indices input is empty.
  // Remove aten::copy_, and replace it with index_put.
  // 1. create an empty listConstruct node as indices input for index_put.
  // 2. create index_put node.

  // Tracing aten::copy_ broadcasts the rhs values.
  // 3. Apply broadcasting for scripting.
  WithInsertPoint guard(node);
  auto graph = node->owningGraph();
  auto dummy_list =
      graph->insertNode(graph->createList(OptionalType::ofTensor(), {}))
          ->output();

  auto expanded_value =
      graph->insert(aten::expand_as, {node->input(1), node->input(0)});
  expanded_value->node()->setSourceRange(node->sourceRange());
  expanded_value->copyMetadata(node->input(1));
  expanded_value->node()->copyMetadata(node);

  auto index_put = graph->insert(
      aten::index_put_,
      {node->input(0), dummy_list, expanded_value, node->input(2)});
  index_put->node()->copyMetadata(node);
  index_put->copyMetadata(node->output());
  node->output()->replaceAllUsesWith(index_put);

  node->destroy();

  return PrepareIndexPutForONNX(index_put->node());
}

auto PrepareSetForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::set_);
  auto clone_n = addDummyClone(n->owningGraph(), n->input(1), true, n);
  TORCH_INTERNAL_ASSERT(nullptr != clone_n);
  clone_n->copyMetadata(n);

  auto orig_input = n->input(0);
  n->output()->replaceAllUsesWith(clone_n->output());
  n->destroy();
  return std::make_pair(orig_input, clone_n->output());
}

std::pair<Value*, Value*> PrepareInplaceOpsInBlocksForONNX(Node* node) {
  if (!node->kind().is_aten())
    return {};

  auto name = node->schema().name();
  bool inplace_op = name.at(name.size() - 1) == '_';
  if (!inplace_op)
    return {};

  auto new_schema = name.substr(0, name.size() - 1);

  Node* input_node = node->inputs().at(0)->node();

  auto graph = node->owningGraph();
  auto new_node = graph->create(Symbol::fromQualString(new_schema), 1);
  for (Value* input : node->inputs()) {
    new_node->addInput(input);
  }
  new_node->output()->setType(node->output()->type());
  new_node->insertBefore(node);
  new_node->copyMetadata(node);
  node->replaceAllUsesWith(new_node);
  node->destroy();

  if (input_node->kind() == aten::select || input_node->kind() == aten::slice) {
    // Cases from a[i] = x. Convert to copy_ and eventually index_put_.
    WithInsertPoint guard(new_node);
    auto false_val_ = graph->insertConstant(false);

    auto new_copy = graph->create(aten::copy_, 1);
    new_copy->addInput(new_node->inputs().at(0));
    new_copy->addInput(new_node->output());
    new_copy->addInput(false_val_);
    new_copy->insertAfter(new_node);
    new_copy->copyMetadata(new_node);

    return PrepareCopyForONNX(new_copy);
  } else {
    // Direct aliasing, the node is a standalone inplace op.
    return std::make_pair(new_node->input(0), new_node->output());
  }
}

// aten::pop is inplace. The tensor list input is updated.
// This pass creates an aten::__getitem__ op to return the original output from
// aten::pop. Then it makes the original aten::pop operator return the updated
// tensor list, and replaces all later uses of that tensor list with this new
// output.
static std::pair<Value*, Value*> PrepareListPopForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::pop);
  //   %ten : Tensor = aten::pop(%seq, %pos)
  // Convert to
  //   %ten : Tensor = aten::__getitem__(%seq, %pos)
  //   %new_seq : Tensor[] = aten::pop(%seq, %pos)
  // And replace all uses of %seq afterwards with %new_seq
  Node* getitem_node =
      n->owningGraph()->create(aten::__getitem__, {n->inputs()});
  getitem_node->output()->setType(n->output()->type());
  getitem_node->insertBefore(n);
  getitem_node->copyMetadata(n);
  n->output()->replaceAllUsesWith(getitem_node->output());
  n->output()->setType(n->inputs().at(0)->type());

  return std::make_pair(n->input(0), n->output());
}

static std::pair<Value*, Value*> PrepareListDeleteForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::Delete);
  n->addOutput();
  n->output()->setType(n->inputs().at(0)->type());

  return std::make_pair(n->input(0), n->output());
}

static std::pair<Value*, Value*> PrepareListAppendAndInsertForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::insert || n->kind() == aten::append);
  if (n->outputs().empty()) {
    n->addOutput();
    n->output()->setType(n->inputs().at(0)->type());
  }
  return std::make_pair(n->input(0), n->output());
}

static std::pair<Value*, Value*> PrepareSetItemForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::_set_item);
  // It seems the JIT does not always produce an output for _set_item.
  // In particular it seems to for list but not for dict.
  // So we add one if needed.
  if (n->outputs().empty()) {
    n->addOutput();
    n->output()->setType(n->inputs().at(0)->type());
  }
  return std::make_pair(n->input(0), n->output());
}

// Remove Mutation pass does not handle mutation on block inputs.
// To fix this, insert a clone node following the graph input:
// Example for graph input node %0:
// Before:
// graph(%0 : Tensor):
//   %5 : Tensor = aten::zero_(%0)
//   ...
// After:
// graph(%0 : Tensor):
//   %2 : None = prim::Constant()
//   %3 : Tensor = aten::clone(%0, %2)
//   %5 : Tensor = aten::zero_(%3)
//   ...

static void PrepareForRemoveMutations(MutationRemover& mr, Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareForRemoveMutations(mr, child_block);
    }
  }

  for (auto input : b->inputs()) {
    bool needsRestart = false;
    do {
      needsRestart = false;
      for (auto use : input->uses()) {
        Node* node = use.user;
        if (!mr.inplaceOpVariant(node)) {
          continue;
        }
        auto it =
            std::find(node->inputs().begin(), node->inputs().end(), input);
        if (it != node->inputs().end()) {
          int index = std::distance(node->inputs().begin(), it);
          TORCH_WARN(
              "ONNX Preprocess - Removing mutation from node ",
              node->kind().toQualString(),
              " on block input: '",
              (*it)->debugName(),
              "'. This changes graph semantics.");

          Node* newNode =
              addDummyClone(b->owningGraph(), input, false, b->return_node());
          TORCH_INTERNAL_ASSERT(nullptr != newNode);
          newNode->copyMetadata(node);
          node->replaceInput(index, newNode->output());
          input->replaceAllUsesAfterNodeWith(node, newNode->output());
          needsRestart = true;
          break;
        }
      }
    } while (needsRestart);
  }
}

static void PrepareForRemoveMutations(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  PrepareForRemoveMutations(mr, graph->block());
  GRAPH_DUMP("After PrepareForRemoveMutations: ", graph);
}

// findSubModuleAttr function chases getAttr chains backwards to locate the
// submodules. For example: module M {
//   attributes {
//     A = <SubModule at ...>
//   }
//   ...
//   %A = prim::GetAttr[name="A"](%self)
//   ...
//   %B = prim::GetAttr[name="B"](%A)
//   ...
//   %weight = prim::GetAttr[name="scale"](%B)
//   ...
std::deque<std::string> findSubModuleAttr(
    Value* input,
    std::string& name,
    Module& attrModule,
    const std::shared_ptr<Graph>& graph) {
  Node* node = input->node();
  std::deque<std::string> moduleNames;

  // Loop starts from inner submodule and follows the chain until reaches the
  // top module.

  auto selfNode = graph->nodes().begin();
  auto n = *selfNode;
  while (node->outputs().at(0)->type() != n->output()->type()) {
    if (node->kind() == prim::GetAttr) {
      moduleNames.push_front(node->s(attr::name));
      node = node->inputs()[0]->node();
    } else {
      break;
    }
  }
  // Assign the inner module to attrModule.
  for (auto& moduleName : moduleNames) {
    attrModule = attrModule.attr(moduleName).toModule();
  }
  return moduleNames;
}

Value* findArgumentAsInputParam(
    const std::shared_ptr<Graph>& graph,
    std::string& name,
    IValue& attr) {
  for (auto input : graph->inputs()) {
    if (input->debugName() == name)
      return input;
  }
  throw std::runtime_error(
      "Attribute is not part of model parameters. Cannot handle SetAttr and GetAttr nodes for : " +
      name);
}

void InplaceConverter::ValueTracker::init(const std::shared_ptr<Graph>& graph) {
  alias_to_value_ = {};
  value_to_sorted_aliases_ = {};
  graph_ = graph;
}

std::string InplaceConverter::ValueTracker::toString() const {
  std::stringstream ss;

  // ss << "Current graph: " << graph_->toString() << std::endl;
  ss << "Tracking " << value_to_sorted_aliases_.size() << " individual values."
     << '\n';
  ss << "value_to_sorted_aliases_: " << '\n';
  size_t idx = 0;
  for (const auto& it : value_to_sorted_aliases_) {
    ss << "Value[" << idx << "]: " << it.first->debugName() << '\n';
    ss << "  Mapping to ";
    for (auto v : it.second) {
      ss << v->debugName() << " ";
    }
    ss << '\n';
    idx++;
  }

  ss << "alias_to_value_: " << '\n';
  for (auto it : alias_to_value_) {
    ss << "  Alias " << it.first->debugName();
    ss << " map to " << it.second->debugName() << '\n';
  }

  return ss.str();
}

void InplaceConverter::ValueTracker::recordSetValue(
    Value* old_v,
    Value* new_v) {
  GRAPH_UPDATE(
      "Calling recordSetValue with old_v: ",
      old_v->debugName(),
      " new_v: ",
      new_v->debugName());
  GRAPH_UPDATE(this->toString());
  auto* n = new_v->node();
  auto* owning_block = n->owningBlock();

  if (alias_to_value_.find(old_v) == alias_to_value_.end()) {
    alias_to_value_[old_v] = old_v;
    value_to_sorted_aliases_[old_v] = {old_v};
  }

  auto root_v = alias_to_value_[old_v];
  alias_to_value_[new_v] = root_v;
  auto& sorted_alias = value_to_sorted_aliases_[root_v];
  sorted_alias.insert(new_v);

  // check if new_v is created inside if or loop subblock.
  auto* owning_blocknode = owning_block->owningNode();
  if (nullptr == owning_blocknode) {
    return;
  }
  auto owning_block_nkind = owning_blocknode->kind();
  if (owning_block_nkind != prim::Loop && owning_block_nkind != prim::If) {
    return;
  }

  bool registered = std::any_of(
      owning_block->outputs().begin(),
      owning_block->outputs().end(),
      [&sorted_alias](Value* out) {
        return std::any_of(
            sorted_alias.begin(), sorted_alias.end(), [&out](Value* alias) {
              return alias == out;
            });
      });

  bool from_outer_alias = std::any_of(
      sorted_alias.begin(),
      sorted_alias.end(),
      [&owning_blocknode](Value* alias) {
        return isAncestor(
            alias->node()->owningBlock(), owning_blocknode->owningBlock());
      });

  // The data of this value has been changed.
  // If this value has alias from outer block,
  // then the update must be reflected back to outside.
  // Thus it needs to be registered as a subblock output.
  // This step can be skipped if other alias of this value has already been
  // registered as subblock output.
  if (!registered && from_outer_alias) {
    if (owning_block_nkind == prim::Loop) {
      owning_block->registerOutput(new_v);
      auto new_block_in = owning_block->addInput();
      new_block_in->setType(new_v->type());
      sorted_alias.insert(new_block_in);
      alias_to_value_[new_block_in] = root_v;
      owning_blocknode->addInput(root_v);
    } else if (owning_block_nkind == prim::If) {
      for (auto* if_sub_block : owning_blocknode->blocks()) {
        if (owning_block == if_sub_block) {
          if_sub_block->registerOutput(new_v);
        } else {
          if_sub_block->registerOutput(root_v);
        }
      }
    }
    auto* new_blocknode_out = owning_blocknode->addOutput();
    new_blocknode_out->setType(new_v->type());
    recordSetValue(root_v, new_blocknode_out);
  }

  GRAPH_UPDATE(
      "After recordSetValue for in: ",
      old_v->debugName(),
      ", out: ",
      new_v->debugName(),
      ". tracker status:");
  GRAPH_UPDATE(this->toString());
}

// Based on current value aliases record, pass over graph and correct alias
// reference for all the nodes.
void InplaceConverter::correctAliasReferences() {
  correctAliasReferences(graph_->block());
}

void InplaceConverter::correctAliasReferences(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    correctAliasReferences(n);

    auto nkind = n->kind();
    if (nkind == prim::If || nkind == prim::Loop) {
      for (auto* sub_block : n->blocks()) {
        correctAliasReferences(sub_block);
      }
    }
  }
  correctAliasReferences(block->return_node());
}

// For every input of Node n, find the correct alias representing that input.
void InplaceConverter::correctAliasReferences(Node* n) {
  for (size_t i = 0; i < n->inputs().size(); ++i) {
    auto* in = n->input(i);
    auto* alias = vt_.findAliasForValueAtNode(in, n);

    if (alias != in) {
      n->replaceInput(i, alias);
      GRAPH_UPDATE(
          "Replacing ",
          in->debugName(),
          " with ",
          alias->debugName(),
          " for ",
          *n);
    }
  }
}

// Find the correct alias representing Value v at Node n.
Value* InplaceConverter::ValueTracker::findAliasForValueAtNode(
    Value* v,
    const Node* n) const {
  GRAPH_UPDATE("Finding alias for value:", v->debugName(), " at node ", *n);
  if (alias_to_value_.find(v) == alias_to_value_.end()) {
    // This value was not affected by any inplace operator.
    return v;
  }

  auto* root_v = alias_to_value_.find(v)->second;
  TORCH_INTERNAL_ASSERT(
      value_to_sorted_aliases_.find(root_v) != value_to_sorted_aliases_.end());
  const auto& aliases = value_to_sorted_aliases_.find(root_v)->second;

  // alias is accessible only if
  // 1. alias owning block is ancestor of n.
  // 2. alias owning node is before n.
  // return the last alias that satisfies this condition.
  Value* found_alias = nullptr;
  for (auto* alias : aliases) {
    auto* alias_n = alias->node();
    if (alias_n->isBefore(n) &&
        isAncestor(alias_n->owningBlock(), n->owningBlock())) {
      found_alias = alias;
    }
  }

  TORCH_INTERNAL_ASSERT(
      nullptr != found_alias,
      "More details: \n",
      n->sourceRange().str(),
      "Input ",
      v->debugName(),
      " of node ",
      *n,
      " was modified by in-place operation, but we cannot find its updated value. ",
      "Please report a bug to PyTorch, and/or try to avoid using in-place operators on this value.");

  return found_alias;
}

// Pass over block, and gather the initial value for any attribute.
// Also cache the full name of the attribute for every GetAttr/SetAttr node.
void InplaceConverter::gatherAttrNameInitialValueMap(
    Block* block,
    std::unordered_map<std::string, Value*>& attr_name_value_map,
    std::unordered_map<Node*, std::string>& attr_node_fullname_map) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    for (auto* sub_block : n->blocks()) {
      gatherAttrNameInitialValueMap(
          sub_block, attr_name_value_map, attr_node_fullname_map);
    }

    if (n->kind() != prim::GetAttr && n->kind() != prim::SetAttr)
      continue;

    auto name = n->s(attr::name);
    auto attrModule = *module_;
    Value* paramConst = nullptr;

    auto moduleNames =
        findSubModuleAttr(n->inputs().at(0), name, attrModule, graph_);

    std::string fullName("");
    for (auto& name : moduleNames) {
      fullName += name + '.';
    }
    fullName += name;

    attr_node_fullname_map.insert({n, fullName});

    if (attr_name_value_map.find(fullName) == attr_name_value_map.end() &&
        attrModule.hasattr(name)) {
      auto attr = attrModule.attr(name);
      auto type = attrModule.type();
      auto slot = *type->findAttributeSlot(name);

      // Add model_parameters and model_buffers as model inputs. Order is
      // preserved based on the appearance in the graph.
      WithInsertPoint guard(graph_->nodes().front());
      if (type->is_parameter(slot) || type->is_buffer(slot) ||
          (attr.isObject() && !attr.toObjectRef().type()->is_module())) {
        paramConst = findArgumentAsInputParam(graph_, fullName, attr);
        attr_name_value_map.insert({fullName, paramConst});
      } else if (auto attrVal = tryInsertConstant(*graph_, attr)) {
        // TODO: Extend support for attribute of type List[Tensor] etc.
        for (size_t i = 0; i < type->getAttributes().size(); i++) {
          if (type->getAttributeName(i) == name) {
            paramConst = *attrVal;
            attr_name_value_map.insert({fullName, paramConst});
          }
        }
      } else {
        // If attribute is a custom class object, instead of primitive types,
        // Tensor, or List/Tuple/Dict of Tensors.
        GRAPH_DEBUG(
            attr.type()->cast<ClassType>() ? "" : "attribute: ",
            name,
            " is not materializable.");
      }
    }

    // Create dummy initial value, if initial value does not exist for this
    // attribute.
    if (attr_name_value_map.find(fullName) == attr_name_value_map.end()) {
      auto* noneNode = graph_->create(prim::Constant);
      noneNode->output()->setType(NoneType::get());
      noneNode->insertBefore(graph_->nodes().front());
      attr_name_value_map.insert({fullName, noneNode->output()});
    }
  }
}

// Replace prim::GetAttr and prim::SetAttr with ATen inplace operators.
// Example graph:
// clang-format off
//  Before graph(%x.1 : Float(12, strides=[1], requires_grad=0, device=cpu)):
//    %1 : __torch__.___torch_mangle_1.M = prim::CreateObject()
//    ...
//    %10 : Tensor = aten::arange(%6, %7, %7, %7, %7)
//     = prim::SetAttr[name="_bias"](%1, %10)
//     = prim::Loop(%5, %8)
//      block0(%i.1 : int):
//        %12 : bool = aten::eq(%i.1, %4)
//         = prim::If(%12)
//          block0():
//             = prim::Loop(%3, %8)
//              block0(%j : int):
//                %14 : Tensor = prim::GetAttr[name="_bias"](%1)
//                %15 : Tensor = aten::add_(%14, %2, %9)
//                 = prim::SetAttr[name="_bias"](%1, %15)
//                -> (%8)
//            -> ()
//          block1():
//            %16 : Tensor = aten::arange(%6, %7, %7, %7, %7)
//             = prim::SetAttr[name="_bias"](%1, %16)
//            -> ()
//        -> (%8)
//    %17 : Tensor = prim::GetAttr[name="_bias"](%1)
//    %18 : Tensor = aten::add(%17, %x.1, %9)
//    return (%18)
//
//  After graph(%x.1 : Float(12, strides=[1], requires_grad=0, device=cpu)):
//    %19 : Float(2, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value= 1  1 [ CPUFloatType{2} ]]()
//    %1 : __torch__.___torch_mangle_1.M = prim::CreateObject()
//    ...
//    %10 : Tensor = aten::arange(%6, %7, %7, %7, %7)
//    %28 : Tensor = aten::set_(%19, %10)
//     = prim::Loop(%5, %8)
//      block0(%i.1 : int):
//        %12 : bool = aten::eq(%i.1, %4)
//         = prim::If(%12)
//          block0():
//             = prim::Loop(%3, %8)
//              block0(%j : int):
//                %15 : Tensor = aten::add_(%19, %2, %9)
//                %25 : Tensor = aten::set_(%19, %15)
//                -> (%8)
//            -> ()
//          block1():
//            %16 : Tensor = aten::arange(%6, %7, %7, %7, %7)
//            %22 : Tensor = aten::set_(%19, %16)
//            -> ()
//        -> (%8)
//    %18 : Tensor = aten::add(%19, %x.1, %9)
//    return (%18)
// clang-format on
void InplaceConverter::replaceAttrWithInplaceOps(
    Block* block,
    const std::unordered_map<std::string, Value*>& attr_name_value_map,
    const std::unordered_map<Node*, std::string>& attr_node_fullname_map) {
  for (const auto& pair : attr_node_fullname_map) {
    auto* n = pair.first;
    auto fullName = pair.second;
    auto find_init_val = attr_name_value_map.find(fullName);
    TORCH_INTERNAL_ASSERT(find_init_val != attr_name_value_map.end());

    TORCH_INTERNAL_ASSERT(
        n->kind() == prim::GetAttr || n->kind() == prim::SetAttr);
    if (n->kind() == prim::SetAttr) {
      // Convert SetAttr to inplace op aten::set_.
      WithInsertPoint guard(n);
      auto* set_node = graph_->create(aten::set_, 1);
      set_node->addInput(find_init_val->second);
      set_node->addInput(n->input(1));
      set_node->copyMetadata(n);
      set_node->insertBefore(n);
    } else if (n->kind() == prim::GetAttr) {
      // Replace use of GetAttr with first seen alias (usually initial value) of
      // that particular value. Correct alias at point of this node will be
      // discovered and assigned in later pass.
      n->output()->replaceAllUsesWith(find_init_val->second);
    }

    n->destroy();
  }
}

void InplaceConverter::convertGetSetAttrToInplaceOps(Block* block) {
  std::unordered_map<std::string, Value*> attr_name_value_map = {};
  std::unordered_map<Node*, std::string> attr_node_fullname_map = {};
  // First pass over graph, to gather all attribute names, and their initial
  // values. Create dummy initial values for attributes if necessary. By the end
  // of this pass, these dummy initial values should have zero uses, and can be
  // safely removed. Otherwise it will imply an error in the model for using
  // uninitialized values.
  gatherAttrNameInitialValueMap(
      block, attr_name_value_map, attr_node_fullname_map);
  GRAPH_UPDATE("Graph after gatherAttrNameInitialValueMap", graph_->toString());

  // Second pass over graph,
  // replace GetAttr with first seen alias (usually initial value),
  // and replace SetAttr with inplace op, updating new value onto first seen
  // alias.
  replaceAttrWithInplaceOps(block, attr_name_value_map, attr_node_fullname_map);
}

// Convert inplace ops to outplace version, and record the associated new alias
// in ValueTracker.
void InplaceConverter::convertInplaceOpsAndTrackAlias(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    auto nkind = n->kind();
    if (nkind == prim::If || nkind == prim::Loop) {
      for (Block* sub_block : n->blocks()) {
        convertInplaceOpsAndTrackAlias(sub_block);
      }
    } else {
      Value *orig_data = nullptr, *new_out = nullptr;
      if (nkind == aten::copy_) {
        std::tie(orig_data, new_out) = PrepareCopyForONNX(n);
      } else if (nkind == aten::index_put || nkind == aten::index_put_) {
        std::tie(orig_data, new_out) = PrepareIndexPutForONNX(n);
        if (nkind == aten::index_put) {
          // special case, index_put is not inplace.
          continue;
        }
      } else if (nkind == aten::insert || nkind == aten::append) {
        std::tie(orig_data, new_out) = PrepareListAppendAndInsertForONNX(n);
      } else if (nkind == aten::set_) {
        std::tie(orig_data, new_out) = PrepareSetForONNX(n);
      } else if (mr_->inplaceOpVariant(n)) {
        std::tie(orig_data, new_out) = PrepareInplaceOpsInBlocksForONNX(n);
      } else if (nkind == aten::pop) {
        std::tie(orig_data, new_out) = PrepareListPopForONNX(n);
      } else if (nkind == aten::Delete) {
        std::tie(orig_data, new_out) = PrepareListDeleteForONNX(n);
      } else if (nkind == aten::_set_item) {
        std::tie(orig_data, new_out) = PrepareSetItemForONNX(n);
      } else {
        // Not inplace op.
        continue;
      }

      if (nullptr != orig_data && nullptr != new_out) {
        vt_.recordSetValue(orig_data, new_out);
      }
    }
  }
}

void InplaceConverter::convertInplaceOpsAndTrackAlias() {
  convertInplaceOpsAndTrackAlias(graph_->block());
  GRAPH_UPDATE(
      "Graph after convertInplaceOpsAndTrackAlias: ", graph_->toString());
  GRAPH_UPDATE(vt_.toString());
}

void InplaceConverter::convertMutationForONNX() {
  // First pass to convert all prim::GetAttr and prim::SetAttr to ATen inplace
  // operators.
  convertGetSetAttrToInplaceOps(graph_->block());
  GRAPH_UPDATE("Graph after convertGetSetAttrToInplaceOps", graph_->toString());
  vt_.init(graph_);
  // Second pass to convert all inplace operators to outplace version, and
  // record the associated new alias in ValueTracker.
  convertInplaceOpsAndTrackAlias();
  // Third pass to check and correct alias reference for all the nodes.
  correctAliasReferences();
}

} // namespace

void RemoveInplaceOpsForONNX(
    const std::shared_ptr<Graph>& graph,
    Module* model = nullptr) {
  ImplicitCastForBinaryInplaceOps(graph->block());
  PrepareForRemoveMutations(graph);
  MutationRemover mr(graph);
  mr.removeTensorMutation();
  mr.removeListMutation();
  InplaceConverter ic(graph, &mr, model);
  ic.convertMutationForONNX();
}

} // namespace torch::jit
