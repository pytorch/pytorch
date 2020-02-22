#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>
#include <torch/csrc/jit/fuser/common/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {
DataType aten_opt_type_map(const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? aten_to_data_type(scalar_type.value())
                                 : DataType::Null;
}

c10::optional<TensorContiguity> infer_contiguity_from_tensor_type(
    const std::shared_ptr<c10::TensorType>& tensor_type) {
  if (!tensor_type->isComplete()) {
    return c10::nullopt;
  } else {
    return TensorContiguity(
        *(tensor_type->sizes().concrete_sizes()),
        *(tensor_type->strides().concrete_sizes()));
  }
}

} // namespace

/*
 * Tensor member definitions
 */

Tensor::Tensor(const std::shared_ptr<c10::TensorType>& tensor_type)
    : Val(ValType::Tensor, aten_opt_type_map(tensor_type->scalarType())),
      contiguity_(infer_contiguity_from_tensor_type(tensor_type)),
      domain_(nullptr) {}

Tensor::Tensor(const std::shared_ptr<Value>& jit_value)
    : Tensor(jit_value->type()->cast<c10::TensorType>()) {}

bool Tensor::hasContiguityInfo() const {
  return contiguity_.has_value();
}

const c10::optional<TensorContiguity>& Tensor::getContiguityInfo() const {
  return contiguity_;
}

/*
 * Split, Merge, Reorder constructors
 */

Split::Split(
    TensorDomain* _out,
    TensorDomain* _in,
    int _axis,
    Int* _factor)
    : Expr(ExprType::Split),
      out_{_out},
      in_{_in},
      axis_{_axis},
      factor_{_factor} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Merge::Merge(TensorDomain* _out, TensorDomain* _in, int _axis)
    : Expr(ExprType::Merge), out_{_out}, in_{_in}, axis_{_axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Reorder::Reorder(
    TensorDomain* _out,
    TensorDomain* _in,
    std::vector<int> _pos2axis)
    : Expr(ExprType::Reorder), out_{_out}, in_{_in}, pos2axis_{_pos2axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}



TensorView* split(TensorView* tv, int axis, int factor) {
  
  TensorDomain* td = tv->domain();
  if(axis<0) axis+=tv->domain()->size();
  assert(axis >= 0 && axis < td->size());
  IterDomain* id = td->axis(axis);

  if (id->parallel_method() != ParallelType::Serial)
    throw std::runtime_error(
        "Splitting an axis of non-Serial iteration approach is not supported at this time. Parallelization strategy must be set after calling split.");

  if (tv->getComputeAtView() != nullptr)
    if (axis < tv->getComputeAtAxis())
        throw std::runtime_error("Cannot split axis within compute at range.");

  std::vector<IterDomain*> new_domain;

  Int* fact = new Int(factor);
  Int* one = new Int(1);
  
  for (decltype(td->size()) i = 0; i < td->size(); i++) {
    if (i != axis)
      new_domain.push_back(td->axis(i));
    else {
      // outer loop size
      Val* vo = add(div(sub(id->size(), one), fact), one);

      Int* so = static_cast<Int*>(vo);

      // outer loop IterDomain
      IterDomain* ido = new IterDomain(so, id->parallel_method(), id->isReduction());
      new_domain.push_back(ido);

      // inner loop IterDomain
      IterDomain* idi = new IterDomain(fact, id->parallel_method(), id->isReduction());
      new_domain.push_back(idi);
    }
  }
  TensorDomain* split_td = new TensorDomain(new_domain);
  Split* split_node = new Split(split_td, td, axis, fact); //For record keeping
  tv->setDomain(split_td);
  return tv;
}

TensorView* merge(TensorView* tv, int axis) {
  TensorDomain* td = tv->domain();
  assert(axis >= 0 && axis + 1 < td->size());

  if (tv->getComputeAtView() != nullptr)
    if (axis < tv->getComputeAtAxis())
        throw std::runtime_error("Cannot split axis within compute at range.");

  IterDomain* first = tv->domain()->axis(axis);
  IterDomain* second = tv->domain()->axis(axis+1);

  assert(first->isReduction() == second->isReduction());
  assert(first->parallel_method() == second->parallel_method());

  Val* merged_id_size = mul(first->size(), second->size());
  IterDomain* merged_id =
      new IterDomain(static_cast<Int*>(merged_id_size), first->parallel_method(), first->isReduction());

  std::vector<IterDomain*> new_domain;
  for (decltype(td->size()) i = 0; i < td->size(); i++) {
    if (i < axis || i > axis + 1)
      new_domain.push_back(td->axis(i));
    else if (i == axis) {
      new_domain.push_back(merged_id);
    }
  }
  TensorDomain* merged_td = new TensorDomain(new_domain);
  Merge* merge_node = new Merge(merged_td, td, axis); //For record keeping
  tv->setDomain(merged_td);
  return tv;
}

/*
 * Takes axis2pos map, axis2pos[old_pos] = new_pos, to modify the ordering of the iter
 * axes.
 */ 
TensorView* reorder(
    TensorView* tv,
    std::unordered_map<int, int> axis2pos) {
  TensorDomain* td = tv->domain();
  auto ndims = td->size();
  // Map to save from previous order, to new order.
  std::vector<int> pos2axis(ndims, -1);

  // Go through each old and new position, make sure they're within 0-ndims
  for (std::pair<int, int> elem : axis2pos) {
    int old_pos = elem.first;
    int new_pos = elem.second;

    if (old_pos < 0)
      old_pos += ndims;
    if (new_pos < 0)
      new_pos += ndims;

    assert(old_pos >= 0 && old_pos < ndims && new_pos >= 0 && new_pos < ndims);

    if (pos2axis[new_pos] != -1)
      throw std::runtime_error(
          std::string("Fatal error: found overlapped positions in reorder. ") +
          std::string(__FILE__) + " : " + std::to_string(__LINE__));

    pos2axis[new_pos] = old_pos;
  }

  std::set<int> old_positions(pos2axis.begin(), pos2axis.end());
  old_positions.erase(-1);

  if (old_positions.size() != axis2pos.size())
    throw std::runtime_error(
        std::string("Fatal error: found overlapped positions in reorder. ") +
        std::string(__FILE__) + " : " + std::to_string(__LINE__));

  std::set<int> all_positions;
  for (int i = 0; i < ndims; i++)
    all_positions.insert(i);

  // Check what positions haven't been specified.
  std::set<int> positions_left;
  std::set_difference(
      all_positions.begin(),
      all_positions.end(),
      old_positions.begin(),
      old_positions.end(),
      std::inserter(positions_left, positions_left.end()));

  // Fill in positions that weren't specified, in relative order,
  // in empty spots in the set of new positions.
  // pos2axis[new_position] = old_position
  auto it = positions_left.begin(); // old positions left
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    if (pos2axis[i] == -1)
      pos2axis[i] = *it++;
  }

  //pos2axis is now filled
  if(tv->getComputeAtView() != nullptr){
    for(int i = 0; i < tv->getComputeAtAxis(); i++){
      if(pos2axis[i] != i)
        throw std::runtime_error("Cannot reorder axis within compute at range.");
    }
  }

  std::vector<IterDomain*> reordered_domain;

  for (int i = 0; i < pos2axis.size(); i++) {
    reordered_domain.push_back(td->axis(pos2axis[i]));
  }
  TensorDomain* reordered_td = new TensorDomain(reordered_domain);
  Reorder* merge_node = new Reorder(reordered_td, td, pos2axis);
  tv->setDomain(reordered_td);
  return tv;
}

TensorView* TensorView::computeAt(TensorView* consumer, int axis){
  /*
   * TODO:
   * Recursive compute_at:
   * Recurse backward from consumer, to this, make sure there's a dependency chain there.
   * After recursing, recurse again, and call ComputeAt for all tensors between this and consumer.
   * 
   * Assert direct consumer/this relationship.
   * Compute at modifies the consumer, not the this.
   */

  std::stack<Val*> dep_chain = DependencyCheck::getDependencyChain(this, consumer);
  //forward apply to uses of this.
  //Recursively apply replay.
  TensorView* ref = consumer;
  //dep_chain = deps <- consumer (doesn't have this)
  //We want to apply:
  //  replay(consumer, dep)
  //  replay(dep, this)
  while(!dep_chain.empty()){
    Val* val = dep_chain.top(); dep_chain.pop();
    TORCH_CHECK(val->getValType() == ValType::TensorView);
    TensorView* tv = static_cast<TensorView*>(val);
    if(tv->same_as(consumer))
      continue;
    TransformReplay::replay(ref, tv, axis);
    ref = tv; //replay is in-place
  }

  if(FusionGuard::getCurFusion()->origin(this) == nullptr)
    return this;
    
  //Dep chain doesn't contain this, run on this manually.
  return TransformReplay::replay(ref, this, axis);
}

} // namespace fuser
} // namespace jit
} // namespace torch
