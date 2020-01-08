#include "torch/csrc/jit/compiler/include/schedule.h"

#include <stdexcept>

#include "torch/csrc/jit/compiler/include/eval.h"

namespace torch {
namespace jit {
namespace compiler {
namespace schedule {

namespace {

// Evaluates a constant expression and returns its value.
template <typename T>
static T EvalConstExpr(const Expr& expr) {
  SimpleIREvaluator eval;
  expr.accept(&eval);
  return eval.value().as<T>();
}

} // namespace

ScheduleNode::~ScheduleNode() {
  for (ScheduleObject* p : schedule_objects_) {
    delete p;
  }
}

ScheduleNode::ScheduleNode(const std::vector<Tensor>& tensors)
    : tensors_(tensors) {
  root_node_ = this->NewTensorExprNode();
  TensorExprNode* current_func = nullptr;
  for (const Tensor& tensor : tensors) {
    const Function& func = tensor.function();
    if (current_func == nullptr) {
      current_func = root_node_->NewFirstChild();
    } else {
      current_func = current_func->NewNextSibling();
    }
    // TODO: handles the scalar case where ndims == 0
    TensorExprNode* node = current_func;
    for (int i = 0; i < func.ndim(); i++) {
      node = node->NewFirstChild();
      LoopAxis* loop_axis = this->NewAxis(func.arg(i), Range(0, func.dim(i)));
      node->set_loop_axis(loop_axis);
    }
    node = node->NewFirstChild();
    TensorExprOp* tensor_expr_op = this->NewTensorExprOp(func);
    node->set_tensor_expr_op(tensor_expr_op);

    // attach the node to the user provided tensors.
    Tensor* tensor_mutable = const_cast<Tensor*>(&tensor);
    tensor_mutable->node()->expr_node_ = node;
  }
}

void ScheduleNode::SplitWithTail(
    TensorExprNode* expr_node,
    const Var& loop_var,
    int factor,
    bool factor_on_inner,
    Var* outer_var,
    Var* inner_var,
    Var* tail_var,
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
    ;
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
    }
    tail_node->SetFirstChild(loop_child_clone);
    tail_node->SetNextSibling(loop_sibling);
    *tail_var = tail_axis->var();
  } else {
    outer_node->SetNextSibling(loop_sibling);
  }
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
Stmt ScheduleNode::Lower(TensorExprNode* node) {
  if (node == nullptr) {
    return Stmt();
  }
  if (node->next_sibling() != nullptr) {
    std::vector<Stmt> siblings;
    TensorExprNode* n = node;
    while (n != nullptr) {
      Stmt stmt = LowerNoSibling(n);
      siblings.push_back(stmt);
      n = n->next_sibling();
    }
    return Block::make(siblings);
  }
  return LowerNoSibling(node);
}

Stmt ScheduleNode::LowerNoSibling(TensorExprNode* node) {
  if (node == nullptr) {
    return Stmt();
  }
  if (node->is_empty_value()) {
    return Stmt();
  }
  if (node->is_tensor_expr_op()) {
    CHECK(node->first_child() == nullptr);
    TensorExprOp* expr_op = node->tensor_expr_op();
    Stmt stmt = expr_op->ElementStmt();
    return stmt;
  } else if (node->is_loop_axis()) {
    CHECK(node->first_child() != nullptr);
    LoopAxis* loop_axis = node->loop_axis();
    Stmt body = Lower(node->first_child());
    const Var& var = loop_axis->var();
    const Range& range = loop_axis->range();
    Stmt for_stmt = For::make(var, range.start(), range.stop(), body);
    return for_stmt;
  } else if (node->is_empty_value()) {
    return Lower(node->first_child());
  } else {
    LOG(FATAL) << "Unsupported node type";
  }
}

void LoopAxis::CloneFrom(const LoopAxis* other) {
  this->loop_var_ = other->loop_var_;
  this->loop_range_ = other->loop_range_;
  this->axis_type_ = other->axis_type_;
  this->is_leaf_ = other->is_leaf_;
  this->output_group_index_ = other->output_group_index_;

  this->loop_axis_transform_ = CloneObject(other->loop_axis_transform_);
}

void LoopAxisTransform::CloneFrom(const LoopAxisTransform* other) {
  inputs_.resize(other->inputs_.size());
  outputs_.resize(other->outputs_.size());

  for (int i = 0; i < inputs_.size(); i++) {
    inputs_[i] = CloneObject(other->inputs_[i]);
  }
  for (int i = 0; i < outputs_.size(); i++) {
    std::vector<LoopAxis*>& output = outputs_[i];
    const std::vector<LoopAxis*>& other_output = other->outputs_[i];
    output.resize(other_output.size());
    for (int j = 0; j < other_output.size(); j++) {
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
  this->node_type = this->node_type;
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
  const Expr& start_expr = loop_range.start();
  const Expr& stop_expr = loop_range.stop();

  // For now, only support static sizes for split axes.
  // TODO: Add support for dynamic ranges.
  start_ = EvalConstExpr<int>(start_expr);
  stop_ = EvalConstExpr<int>(stop_expr);
}

SplitAxisWithTail::SplitAxisWithTail(
    LoopAxis* loop_axis,
    int factor,
    bool factor_on_inner)
    : BaseClass(loop_axis, factor, factor_on_inner) {
  // TODO: support factor_on_inner == false;
  CHECK(factor_on_inner) << "only factor_on_inner = True is supported for now";

  int size = this->stop() - this->start();
  int split_count = size / factor;
  int trail_size = size % factor;
  int output_group_count = (trail_size > 0) ? 2 : 1;

  this->set_output_group_count(output_group_count);
  // The main group
  const std::string& loop_var_name = loop_axis->var().name_hint();
  Dtype loop_var_dtype = loop_axis->var().dtype();
  LoopAxis* outer = this->NewAxis(
      Var(loop_var_name + ".outer", loop_var_dtype), Range(0, split_count));
  LoopAxis* inner = this->NewAxis(
      Var(loop_var_name + ".inner", loop_var_dtype), Range(0, factor));
  this->set_output_group(0, {outer, inner});

  // The trail group
  if (trail_size) {
    LoopAxis* trail = this->NewAxis(
        Var(loop_var_name + ".trail", loop_var_dtype), Range(0, trail_size));
    this->set_output_group(1, {trail});
  }
}

Stmt SplitAxisWithTail::ConvertToNewArgs(const Stmt& stmt, int output_group) {
  LOG(FATAL) << "SplitAxisWithTail::ConvertToNewArgs unimplemented yet";
}

LoopAxis* LoopAxisTransform::NewAxis(
    const Var& loop_var,
    const Range& loop_range) {
  ScheduleNode* schedule = this->schedule();
  LoopAxis* axis = schedule->NewAxis(loop_var, loop_range);
  axis->set_loop_axis_transform(this);
}

} // namespace schedule
} // namespace compiler
} // namespace jit
} // namespace torch
