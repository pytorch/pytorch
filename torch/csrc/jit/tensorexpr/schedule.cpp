#include "torch/csrc/jit/tensorexpr/schedule.h"

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {
namespace schedule {

namespace {

// Evaluates a constant expression and returns its value.
template <typename T>
static T EvalConstExpr(const ExprHandle& expr) {
  ExprEval<SimpleIREvaluator> eval(expr);
  return eval.value<T>();
}

} // namespace

ScheduleNode::~ScheduleNode() {
  for (ScheduleObject* p : schedule_objects_) {
    delete p;
  }
}

class ScheduleNode::DependencyTracker : public IRVisitor {
 public:
  virtual ~DependencyTracker() = default;
  DependencyTracker(const std::vector<Tensor*>& output_tensors) {
    for (size_t i = 0; i < output_tensors.size(); i++) {
      const Tensor* node = output_tensors[i];
      to_process_.push(node);
      encountered_.insert(node);
      given_tensors_.insert(node);
    }

    // Extract all the consumer-producer relationship.
    while (!to_process_.empty()) {
      Tensor* tensor_node = const_cast<Tensor*>(to_process_.front());
      to_process_.pop();
      current_consumer_ = tensor_node;
      tensor_node->function()->body()->accept(this);
    }

    // Topologically sorted all the tensors in encountered_
    while (!encountered_.empty()) {
      sort_tensor_node(*encountered_.begin());
    }
  }

  std::vector<const Tensor*> GetTopologicallySorted() const {
    return topologically_sorted_;
  }

  bool is_internal(const Tensor* tensor_node) const {
    return (given_tensors_.count(tensor_node) == 0);
  }

 private:
  void visit(const FunctionCall* v) override {
    const Tensor* producer = v->tensor();
    add_producer_consumer_pair(current_consumer_, producer);
  }

  void add_producer_consumer_pair(
      const Tensor* consumer,
      const Tensor* producer) {
    producers_[consumer].insert(producer);
    consumers_[producer].insert(consumer);
    if (encountered_.count(producer) == 0) {
      encountered_.insert(producer);
      to_process_.push(producer);
    }
  }

  // topoligically sort the sub tensors under the current node
  void sort_tensor_node(const Tensor* tensor_node) {
    encountered_.erase(tensor_node);
    auto iter = producers_.find(tensor_node);
    if (iter != producers_.end()) {
      for (const Tensor* producer_node : iter->second) {
        if (encountered_.count(producer_node) != 0) {
          sort_tensor_node(producer_node);
        }
      }
    }
    topologically_sorted_.push_back(tensor_node);
  }

  std::unordered_map<const Tensor*, std::unordered_set<const Tensor*>>
      producers_;
  std::unordered_map<const Tensor*, std::unordered_set<const Tensor*>>
      consumers_;

  // the tensors given in the constructors. They are either the input or the
  // output of the entire schedule.
  std::unordered_set<const Tensor*> given_tensors_;

  const Tensor* current_consumer_ = nullptr;
  std::unordered_set<const Tensor*> encountered_;
  std::queue<const Tensor*> to_process_;
  std::vector<const Tensor*> topologically_sorted_;
};

ScheduleNode::ScheduleNode(const std::vector<Tensor*>& tensors)
    : output_tensors_(tensors) {
  dependency_tracker_.reset(new DependencyTracker(tensors));
  root_node_ = this->NewTensorExprNode();
  TensorExprNode* current_func = nullptr;
  std::vector<const Tensor*> sorted_tensors =
      dependency_tracker_->GetTopologicallySorted();
  for (const Tensor* tensor_node : sorted_tensors) {
    Function* func = tensor_node->function();
    if (current_func == nullptr) {
      current_func = root_node_->NewFirstChild();
    } else {
      current_func = current_func->NewNextSibling();
    }
    // TODO: handles the scalar case where ndims == 0
    TensorExprNode* expr_node = current_func;
    for (int i = 0; i < func->ndim(); i++) {
      expr_node = expr_node->NewFirstChild();
      LoopAxis* loop_axis = this->NewAxis(VarHandle(func->arg(i)), Range(0, ExprHandle(func->dim(i))));
      expr_node->set_loop_axis(loop_axis);
    }
    expr_node = expr_node->NewFirstChild();
    TensorExprOp* tensor_expr_op = this->NewTensorExprOp(func);
    expr_node->set_tensor_expr_op(tensor_expr_op);

    // attach the node to the user provided tensors.
    Tensor* tensor_mutable = const_cast<Tensor*>(tensor_node);
    tensor_mutable->expr_node_ = expr_node;

    if (dependency_tracker_->is_internal(tensor_node)) {
      internal_tensors_.push_back(const_cast<Tensor*>(tensor_node));
    }
  }
}

void ScheduleNode::ComputeInline(TensorExprNode* expr_node) {
  if (!expr_node->is_tensor_expr_op()) {
    throw std::runtime_error("expr_node must be tensor_expr_op");
  }

  TensorExprOp* texpr_op = expr_node->tensor_expr_op();
  inlined_functions_.push_back(texpr_op->func());
}

void ScheduleNode::GPUExecConfig(
    TensorExprNode* expr_node,
    const std::vector<VarHandle>& blockIdx,
    const std::vector<VarHandle>& threadIdx) {
  // Extract all the ancestors into a var* to loop-axis lookup table
  std::unordered_map<const Var*, LoopAxis*> var_to_loop;
  TensorExprNode* node = expr_node;
  while (node != nullptr) {
    if (node->is_loop_axis()) {
      LoopAxis* loop_axis = node->loop_axis();
      const VarHandle& loop_var = loop_axis->var();
      var_to_loop[loop_var.node()] = loop_axis;
    }
    node = node->parent();
  }

  // Set the blockIndex attr.
  for (int i = 0; i < blockIdx.size(); i++) {
    auto iter = var_to_loop.find(blockIdx[i].node());
    if (iter == var_to_loop.end()) {
      throw std::runtime_error(
          "Invalid blockIdx: " + std::to_string(i) + ", " +
          blockIdx[i].name_hint());
    }
    iter->second->set_gpu_block_index(i);
  }

  // Set the threadIdx attr.
  for (int i = 0; i < threadIdx.size(); i++) {
    auto iter = var_to_loop.find(threadIdx[i].node());
    if (iter == var_to_loop.end()) {
      throw std::runtime_error(
          "Invalid threadIdx: " + std::to_string(i) + ", " +
          threadIdx[i].name_hint());
    }
    iter->second->set_gpu_thread_index(i);
  }
}

void ScheduleNode::SplitWithTail(
    TensorExprNode* expr_node,
    const VarHandle& loop_var,
    int factor,
    bool factor_on_inner,
    VarHandle* outer_var,
    VarHandle* inner_var,
    VarHandle* tail_var,
    TensorExprNode** tail_op) {
  // find the loop_axis that contains loop_var in the ancestor
  TensorExprNode* loop_node = expr_node;
  while (loop_node != nullptr) {
    if (loop_node->is_loop_axis()) {
      LoopAxis* loop_axis = loop_node->loop_axis();
      if (loop_axis->var() == loop_var) {
        break;
      }
    }
    loop_node = loop_node->parent();
  }

  if (loop_node == nullptr) {
    // TODO: change to a recoverable error.
    LOG(FATAL) << "loop var cannot be found in the ancestors of node";
  }

  // create the new loop_axis
  SplitAxisWithTail* split_transform = this->NewSplitAxisWithTail(
      loop_node->loop_axis(), factor, factor_on_inner);
  CHECK(split_transform->output_group_count() >= 1);
  CHECK(split_transform->output_group_size(0) == 2);
  LoopAxis* outer_axis = split_transform->output(0, 0);
  LoopAxis* inner_axis = split_transform->output(0, 1);
  LoopAxis* tail_axis = nullptr;
  if (split_transform->output_group_count() >= 2) {
    tail_axis = split_transform->output(1, 0);
  }

  // replace loop_node with the new loop_axis
  TensorExprNode* outer_node = this->NewTensorExprNode();
  outer_node->set_loop_axis(outer_axis);
  *outer_var = outer_axis->var();
  TensorExprNode* inner_node = outer_node->NewFirstChild();
  inner_node->set_loop_axis(inner_axis);
  *inner_var = inner_axis->var();
  TensorExprNode* loop_sibling = loop_node->next_sibling();
  TensorExprNode* loop_child = loop_node->first_child();
  inner_node->SetFirstChild(loop_child);
  if (tail_axis != nullptr) {
    TensorExprNode* tail_node = outer_node->NewNextSibling();
    tail_node->set_loop_axis(tail_axis);
    TensorExprNode* loop_child_clone = nullptr;
    {
      ScopedCloneMap clone_map_scope(this);
      loop_child_clone = CloneObject(loop_child);
      CloneMap& clone_map = clone_map_scope.clone_map();
      CloneMap::iterator iter = clone_map.find(expr_node);
      if (iter == clone_map.end()) {
        LOG(FATAL) << "cannot find node in the clone-map";
      }
      TensorExprNode* expr_node_clone =
          dynamic_cast<TensorExprNode*>(iter->second);
      CHECK(!expr_node || expr_node_clone)
          << "expr_node is not null, but its clone is";
      *tail_op = expr_node_clone;
      DCHECK(expr_node_clone->is_tensor_expr_op());
      expr_node_clone->tensor_expr_op()->ApplyLoopTransform(split_transform, 1);
    }
    tail_node->SetFirstChild(loop_child_clone);
    tail_node->SetNextSibling(loop_sibling);
    *tail_var = tail_axis->var();
  } else {
    outer_node->SetNextSibling(loop_sibling);
  }
  CHECK(expr_node->is_tensor_expr_op());
  // This transform is left after the tail axis is cloned, so it doesn't affect
  // the tail axis.
  expr_node->tensor_expr_op()->ApplyLoopTransform(split_transform, 0);
  TensorExprNode::ReplaceSubtree(loop_node, outer_node);
}

// TODO: Merge with SplitWithTail
void ScheduleNode::SplitWithMask(
    TensorExprNode* expr_node,
    const VarHandle& loop_var,
    int factor,
    bool factor_on_inner,
    VarHandle* outer_var,
    VarHandle* inner_var) {
  // find the loop_axis that contains loop_var in the ancestor
  TensorExprNode* loop_node = expr_node;
  while (loop_node != nullptr) {
    if (loop_node->is_loop_axis()) {
      LoopAxis* loop_axis = loop_node->loop_axis();
      if (loop_axis->var() == loop_var) {
        break;
      }
    }
    loop_node = loop_node->parent();
  }

  if (loop_node == nullptr) {
    // TODO: change to a recoverable error.
    LOG(FATAL) << "loop var cannot be found in the ancestors of node";
  }

  // create the new loop_axis
  SplitAxisWithMask* split_transform = this->NewSplitAxisWithMask(
      loop_node->loop_axis(), factor, factor_on_inner);
  CHECK(split_transform->output_group_count() == 1);
  CHECK(split_transform->output_group_size(0) == 2);
  LoopAxis* outer_axis = split_transform->output(0, 0);
  LoopAxis* inner_axis = split_transform->output(0, 1);

  // replace loop_node with the new loop_axis
  TensorExprNode* outer_node = this->NewTensorExprNode();
  outer_node->set_loop_axis(outer_axis);
  *outer_var = outer_axis->var();
  TensorExprNode* inner_node = outer_node->NewFirstChild();
  inner_node->set_loop_axis(inner_axis);
  *inner_var = inner_axis->var();
  TensorExprNode* loop_sibling = loop_node->next_sibling();
  TensorExprNode* loop_child = loop_node->first_child();
  inner_node->SetFirstChild(loop_child);
  outer_node->SetNextSibling(loop_sibling);

  CHECK(expr_node->is_tensor_expr_op());
  expr_node->tensor_expr_op()->AddPredicate(split_transform->predicate().node());
  expr_node->tensor_expr_op()->ApplyLoopTransform(split_transform, 0);
  TensorExprNode::ReplaceSubtree(loop_node, outer_node);
}

void TensorExprNode::SetParent(TensorExprNode* parent) {
  TensorExprNode* n = this;
  while (n != nullptr) {
    n->parent_ = parent;
    n = n->next_sibling();
  }
}

void TensorExprNode::SetNextSibling(TensorExprNode* node) {
  TensorExprNode* old_sibling = this->next_sibling_;
  this->next_sibling_ = node;
  // reset all the parent links for the siblings
  if (node) {
    node->SetParent(this->parent());
  }
  // detach the parents in the previous next_sibling to prevent dangling
  // pointers.
  if (old_sibling) {
    old_sibling->SetParent(nullptr);
  }
}

void TensorExprNode::SetFirstChild(TensorExprNode* node) {
  TensorExprNode* old_child = this->first_child_;
  this->first_child_ = node;
  // reset all the parent links
  if (node) {
    node->SetParent(this);
  }
  if (old_child) {
    old_child->SetParent(nullptr);
  }
}

void ScheduleObject::AddClonePair(ScheduleObject* new_obj) {
  ScheduleNode* schedule = this->schedule();
  schedule->clone_map().insert(std::make_pair(this, new_obj));
}

ScheduleObject* ScheduleNode::CloneScheduleObject(ScheduleObject* object) {
  if (object == nullptr)
    return nullptr;

  bool map_initialized = false;
  if (!clone_map_) {
    map_initialized = true;
    clone_map_.reset(new CloneMap());
  }

  CloneMap::iterator iter = clone_map_->find(object);
  if (iter != clone_map_->end()) {
    return iter->second;
  }

  ScheduleObject* new_object = object->Clone();
  // TODO: Clone may have inseretd into the map. Only one insertion is needed.
  clone_map_->insert(std::make_pair(object, new_object));

  if (map_initialized) {
    clone_map_.reset();
  }

  return new_object;
}

class Flattener : public IRMutator {
 private:
  Expr* mutate(const FunctionCall* v) override {
    Buffer buffer(
        VarHandle(v->tensor()->function()->func_var()),
        v->tensor()->function()->body()->dtype(),
        ExprVectorToExprHandleVector(v->tensor()->function()->dims()));
    const std::vector<const Expr*>& params = v->params();
    std::vector<ExprHandle> params_expr(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_expr[i] = ExprHandle(params[i]);
    }
    return buffer(params_expr).node();
  }
};

class FunctionInliner : public IRMutator {
 public:
  FunctionInliner(const std::vector<Function*>& funcs) : funcs_(funcs) {
    for (Function* func : funcs) {
      func_var_set_.insert(func->func_var());
    }
  }

 private:
  // For the target function, insert the caller/callee pair into the replacement
  // mapping.
  const Expr* mutate(const FunctionCall* v) override {
    Function* func = v->tensor()->function();
    if (func_var_set_.count(func->func_var()) > 0) {
      // Insert the caller/callee pair into the mapping.
      for (int i = 0; i < func->ndim(); i++) {
        const Var* func_callee_arg = dynamic_cast<const Var*>(func->arg(i));
        const Expr* func_caller_param = v->param(i);
        auto iter = inline_mapping_.find(func_callee_arg);
        if (iter != inline_mapping_.end()) {
          throw std::runtime_error(
              "Duplicated variables: " + func_callee_arg->name_hint());
        }
        inline_mapping_[func_callee_arg] = func_caller_param;
      }

      // Call the actual replacement.
      const Expr* body = func->body();
      const Expr* result = body->accept_mutator(this);

      // Remove the caller/callee relationship.
      for (int i = 0; i < func->ndim(); i++) {
        const Var* func_callee_arg = dynamic_cast<const Var*>(func->arg(i));
        auto iter = inline_mapping_.find(func_callee_arg);
        if (iter == inline_mapping_.end()) {
          throw std::runtime_error(
              "Var already removed: " + func_callee_arg->name_hint());
        }
        inline_mapping_.erase(iter);
      }
      return result;
    } else {
      return IRMutator::mutate(v);
    }
  }

  // Replace the target variable with the caller expressions.
  const Expr* mutate(const Var* v) {
    auto iter = inline_mapping_.find(v);
    if (iter == inline_mapping_.end()) {
      return IRMutator::mutate(v);
    } else {
      const Expr* expr = iter->second;
      // Continue to transform the value from the lookup table.
      return expr->accept_mutator(this);
    }
  }

  // Remove the buffer write the inlined function.
  Stmt* mutate(const Store* v) override {
    if (func_var_set_.count(v->base_handle()) > 0) {
      return nullptr;
    } else {
      return IRMutator::mutate(v);
    }
  }

  std::unordered_map<const Var*, const Expr*> inline_mapping_;
  std::vector<Function*> funcs_;
  std::unordered_set<const Var*> func_var_set_;
};

static Stmt* InjectInlines(
    Stmt* stmt,
    const std::vector<Function*>& inlined_funcs) {
  FunctionInliner inliner(inlined_funcs);
  Stmt* stmt_old = stmt;
  Stmt* stmt_new = stmt_old->accept_mutator(&inliner);
  return stmt_new;
}

ScheduleObject* ScheduleNode::LookUpCloneScheduleObject(
    ScheduleObject* object) {
  if (object == nullptr) {
    return nullptr;
  }
  if (!clone_map_) {
    return nullptr;
  }

  CloneMap::iterator iter = clone_map_->find(object);
  if (iter == clone_map_->end()) {
    return nullptr;
  }

  return iter->second;
}

// TODO: change to a stack-based version without recursion
Stmt* ScheduleNode::Lower(TensorExprNode* node) {
  if (node == nullptr) {
    return nullptr;
  }
  if (node->next_sibling() != nullptr) {
    std::vector<Stmt*> siblings;
    TensorExprNode* n = node;
    while (n != nullptr) {
      Stmt* stmt = LowerNoSibling(n);
      siblings.push_back(stmt);
      n = n->next_sibling();
    }
    return Block::make(siblings);
  }
  return LowerNoSibling(node);
}

Stmt* ScheduleNode::Lower() {
  Stmt* core_stmt = Lower(root_node_);

  // Inject inlines
  core_stmt = InjectInlines(core_stmt, inlined_functions_);

  // Flatten function calls.
  Flattener flattener;
  core_stmt = core_stmt->accept_mutator(&flattener);

  // Add allocs and frees for intermediate buffers at the global level.
  // TODO: move allocs and frees to the imemediate areas to reuse buffers.
  if (internal_tensors_.size() == 0ULL) {
    return core_stmt;
  }

  std::unordered_set<Function*> inlined_func_set;
  for (size_t i = 0; i < inlined_functions_.size(); i++) {
    inlined_func_set.insert(inlined_functions_[i]);
  }
  std::unordered_set<const Tensor*> output_tensors_set;
  for (size_t i = 0; i < output_tensors_.size(); i++) {
    output_tensors_set.insert(output_tensors_[i]);
  }
  std::vector<Stmt*> allocs;
  std::vector<Stmt*> frees;
  for (size_t i = 0; i < internal_tensors_.size(); i++) {
    Tensor* tensor = internal_tensors_[i];
    if (inlined_func_set.count(tensor->function()) > 0) {
      // No need to allocation memory for intermediate tensors.
      continue;
    }
    if (output_tensors_set.count(tensor) > 0) {
      // No need to allocate memory if the tensors are given as input/output.
      continue;
    }
    Stmt* alloc = new Allocate(
        tensor->function()->func_var(),
        tensor->function()->body()->dtype(),
        tensor->function()->dims());
    allocs.push_back(alloc);
    Stmt* free = new Free(tensor->function()->func_var());
    frees.push_back(free);
  }
  std::reverse(frees.begin(), frees.end());
  Stmt* alloc_block = Block::make(allocs);
  Stmt* free_block = Block::make(frees);
  Stmt* combined_stmt = Block::make({alloc_block, core_stmt, free_block});
  return combined_stmt;
}

Stmt* ScheduleNode::LowerNoSibling(TensorExprNode* node) {
  if (node == nullptr) {
    return nullptr;
  }
  if (node->is_empty_value()) {
    return Lower(node->first_child());
  }
  if (node->is_tensor_expr_op()) {
    CHECK(node->first_child() == nullptr);
    TensorExprOp* expr_op = node->tensor_expr_op();
    Stmt* stmt = expr_op->ElementStmt();
    // TODO: the predicate should be hoisted to as high as possible in the
    // acestor chain.
    const std::vector<ExprHandle>& predicates = expr_op->predicates();
    for (int i = 0; i < predicates.size(); i++) {
      stmt = Cond::make(predicates[i], stmt, nullptr);
    }
    return stmt;
  } else if (node->is_loop_axis()) {
    CHECK(node->first_child() != nullptr);
    LoopAxis* loop_axis = node->loop_axis();
    Stmt* body = Lower(node->first_child());
    const VarHandle& var = loop_axis->var();
    const Range& range = loop_axis->range();
    Stmt* for_stmt = For::make(
        var, range.start(), range.stop(), body, loop_axis->loop_options());
    return for_stmt;
  } else if (node->is_empty_value()) {
    return Lower(node->first_child());
  } else {
    LOG(FATAL) << "Unsupported node type";
    return nullptr;
  }
}

void LoopAxis::CloneFrom(const LoopAxis* other) {
  this->loop_var_ = other->loop_var_;
  this->loop_range_ = other->loop_range_;
  this->axis_type_ = other->axis_type_;
  this->is_leaf_ = other->is_leaf_;
  this->output_group_index_ = other->output_group_index_;
  this->loop_options_ = other->loop_options_;

  this->loop_axis_transform_ = CloneObject(other->loop_axis_transform_);
}

void LoopAxisTransform::CloneFrom(const LoopAxisTransform* other) {
  inputs_.resize(other->inputs_.size());
  outputs_.resize(other->outputs_.size());

  for (size_t i = 0; i < inputs_.size(); i++) {
    inputs_[i] = CloneObject(other->inputs_[i]);
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    std::vector<LoopAxis*>& output = outputs_[i];
    const std::vector<LoopAxis*>& other_output = other->outputs_[i];
    output.resize(other_output.size());
    for (size_t j = 0; j < other_output.size(); j++) {
      output[j] = CloneObject(other_output[j]);
    }
  }
}

void SplitAxisTransform::CloneFrom(const SplitAxisTransform* other) {
  this->LoopAxisTransform::CloneFrom(other);
  this->factor_on_inner_ = other->factor_on_inner_;
  this->factor_ = other->factor_;
  this->start_ = other->start_;
  this->stop_ = other->stop_;
}

void SplitAxisWithTail::CloneFrom(const SplitAxisWithTail* other) {
  this->SplitAxisTransform::CloneFrom(other);
}

void SplitAxisWithMask::CloneFrom(const SplitAxisWithMask* other) {
  this->SplitAxisTransform::CloneFrom(other);
}

void TensorExprNode::CloneFrom(const TensorExprNode* other) {
  this->next_sibling_ = CloneObject(other->next_sibling_);
  this->first_child_ = CloneObject(other->first_child_);
  this->node_value_.CloneFrom(&other->node_value_);

  // the parent_ link is valid at this point, since it was updated within
  // Cloneable when the parent object. If the parent link points outside what
  // was cloned so far, it points to NULL.
  this->parent_ = LookUpCloneObject(other->parent_);
}

void TensorExprNode::NodeValue::CloneFrom(
    const TensorExprNode::NodeValue* other) {
  this->node_type = other->node_type;
  if (this->node_type == NodeType::kOperation) {
    this->tensor_expr_op = CloneObject(other->tensor_expr_op);
  } else if (node_type == NodeType::kAxis) {
    this->loop_axis = CloneObject(other->loop_axis);
  } else if (node_type == NodeType::kEmptyValue) {
    // no actdion taken
  } else {
    LOG(FATAL) << "Invalid node type: " << static_cast<int>(this->node_type);
  }
}

void TensorExprNode::ReplaceSubtree(
    TensorExprNode* old_node,
    TensorExprNode* new_node) {
  CHECK(old_node->parent() != nullptr) << "cannot replace a root node";

  TensorExprNode* parent = old_node->parent_;
  if (parent->first_child() == old_node) {
    parent->SetFirstChild(new_node);
  } else {
    TensorExprNode* n = parent->first_child();
    while (n != nullptr && n->next_sibling() != new_node) {
      n = n->next_sibling();
    }
    if (n == nullptr) {
      LOG(FATAL) << "Cannot find node as a child of its parent";
    }
    n->SetNextSibling(new_node);
  }
}

TensorExprNode* TensorExprNode::NewNextSibling() {
  DCHECK(next_sibling_ == nullptr);
  TensorExprNode* sibling = schedule()->NewTensorExprNode();
  sibling->parent_ = this->parent_;
  this->next_sibling_ = sibling;
  return sibling;
}

TensorExprNode* TensorExprNode::NewFirstChild() {
  DCHECK(first_child_ == nullptr);
  TensorExprNode* first_child = schedule()->NewTensorExprNode();
  first_child->parent_ = this;
  this->first_child_ = first_child;
  return first_child;
}

SplitAxisTransform::SplitAxisTransform(
    LoopAxis* loop_axis,
    int factor,
    bool factor_on_inner)
    : BaseClass(std::vector<LoopAxis*>({loop_axis})),
      factor_(factor),
      factor_on_inner_(factor_on_inner) {
  const Range& loop_range = loop_axis->range();
  const ExprHandle& start_expr = loop_range.start();
  const ExprHandle& stop_expr = loop_range.stop();

  start_ = start_expr;
  stop_ = stop_expr;
}

SplitAxisWithTail::SplitAxisWithTail(
    LoopAxis* loop_axis,
    int factor,
    bool factor_on_inner)
    : BaseClass(loop_axis, factor, factor_on_inner) {
  // TODO: support factor_on_inner == false;
  CHECK(factor_on_inner) << "only factor_on_inner = True is supported for now";

  auto const& size = this->stop() - this->start();
  int output_group_count = 2;
  if (this->stop().AsNode<IntImm>() && this->start().AsNode<IntImm>()) {
    int startVal = this->start().AsNode<IntImm>()->value();
    int stopVal = this->stop().AsNode<IntImm>()->value();
    int sizeVal = stopVal - startVal;
    int tail_size = sizeVal % factor;
    if (tail_size == 0) {
      output_group_count = 1;
    }
  }
  auto const& split_count = size / factor;
  auto const& tail_size = size % factor;

  this->set_output_group_count(output_group_count);
  // The main group
  const std::string& loop_var_name = loop_axis->var().name_hint();
  Dtype loop_var_dtype = loop_axis->var().dtype();
  LoopAxis* outer = this->NewAxis(
      VarHandle(loop_var_name + "_outer", loop_var_dtype), Range(0, split_count));
  LoopAxis* inner = this->NewAxis(
      VarHandle(loop_var_name + "_inner", loop_var_dtype), Range(0, factor));
  this->set_output_group(0, {outer, inner});

  // The tail group
  if (output_group_count == 2) {
    LoopAxis* tail = this->NewAxis(
        VarHandle(loop_var_name + "_tail", loop_var_dtype), Range(0, tail_size));
    this->set_output_group(1, {tail});
  }
}

// TODO: merge with SplitAxisWithTail
SplitAxisWithMask::SplitAxisWithMask(
    LoopAxis* loop_axis,
    int factor,
    bool factor_on_inner)
    : BaseClass(loop_axis, factor, factor_on_inner) {
  // TODO: support factor_on_inner == false;
  CHECK(factor_on_inner) << "only factor_on_inner = True is supported for now";

  // TODO: Support dynamic shapes
  auto const& sizeExpr = this->stop() - this->start();
  bool needsPredicate = true;
  if (this->stop().AsNode<IntImm>() && this->start().AsNode<IntImm>()) {
    int size = stop().AsNode<IntImm>()->value() - start().AsNode<IntImm>()->value();
    if ((size % factor) == 0) {
      needsPredicate = false;
    }
  }
  if (needsPredicate) {
    IntImm* start = this->start().AsNode<IntImm>();
    CHECK(start && start->value() == 0) << "Non-zero start is not implemented yet";
    predicate_ = CompareSelect::make(loop_axis->var(), this->stop(), kLT);
  }
  auto const& split_count = (sizeExpr + factor - 1) / factor;

  this->set_output_group_count(1);
  const std::string& loop_var_name = loop_axis->var().name_hint();
  Dtype loop_var_dtype = loop_axis->var().dtype();
  LoopAxis* outer = this->NewAxis(
      VarHandle(loop_var_name + "_outer", loop_var_dtype), Range(0, split_count));
  LoopAxis* inner = this->NewAxis(
      VarHandle(loop_var_name + "_inner", loop_var_dtype), Range(0, factor));
  this->set_output_group(0, {outer, inner});
}

ExprHandle SplitAxisWithTail::combined_loop_index(int output_group) {
  LoopAxis* original_axis = this->input(0);
  VarHandle original_var = original_axis->var();
  LoopAxis* outer = this->output(0, 0);
  LoopAxis* inner = this->output(0, 1);
  ExprHandle combined_index;
  if (output_group == 0) {
    // x -> x.outer * inner.size + x.inner
    combined_index = outer->var() * inner->range().stop() + inner->var();
  } else if (output_group == 1) {
    LoopAxis* tail = this->output(1, 0);
    // x -> x.tail + outer.size * inner.size
    combined_index =
        tail->var() + outer->range().stop() * inner->range().stop();
  } else {
    LOG(FATAL) << "invalid output_group: " << output_group;
  }
  return combined_index;
}

Stmt* SplitAxisWithTail::ConvertToNewArgs(Stmt* stmt, int output_group) {
  ExprHandle combined_index = combined_loop_index(output_group);
  Stmt* new_stmt = Substitute(stmt, {{input(0)->var(), combined_index}});
  return new_stmt;
}

ExprHandle SplitAxisWithTail::ConvertToNewArgs(ExprHandle* expr, int output_group) {
  ExprHandle combined_index = combined_loop_index(output_group);
  ExprHandle new_expr = Substitute(expr, {{input(0)->var(), combined_index}});
  return new_expr;
}

ExprHandle SplitAxisWithMask::combined_loop_index(int output_group) {
  DCHECK_EQ(output_group, 0) << "Ininvalid output group: " << output_group;
  LoopAxis* original_axis = this->input(0);
  VarHandle original_var = original_axis->var();
  LoopAxis* outer = this->output(0, 0);
  LoopAxis* inner = this->output(0, 1);
  ExprHandle combined_index = outer->var() * inner->range().stop() + inner->var();
  return combined_index;
}

Stmt* SplitAxisWithMask::ConvertToNewArgs(Stmt* stmt, int output_group) {
  ExprHandle combined_index = combined_loop_index(output_group);
  Stmt* new_stmt = Substitute(stmt, {{input(0)->var(), combined_index}});
  return new_stmt;
}

ExprHandle SplitAxisWithMask::ConvertToNewArgs(ExprHandle* expr, int output_group) {
  ExprHandle combined_index = combined_loop_index(output_group);
  ExprHandle new_expr = Substitute(expr, {{input(0)->var(), combined_index}});
  return new_expr;
}

LoopAxis* LoopAxisTransform::NewAxis(
    const VarHandle& loop_var,
    const Range& loop_range) {
  ScheduleNode* schedule = this->schedule();
  LoopAxis* axis = schedule->NewAxis(loop_var, loop_range);
  axis->set_loop_axis_transform(this);
  return axis;
}

} // namespace schedule
} // namespace tensorexpr
} // namespace jit
} // namespace torch
