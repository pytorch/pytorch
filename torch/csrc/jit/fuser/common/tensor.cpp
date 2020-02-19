#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/tensor.h>

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
    const TensorDomain* _out,
    const TensorDomain* _in,
    int _axis,
    const Int* _factor)
    : Expr(ExprType::Split),
      out_{_out},
      in_{_in},
      axis_{_axis},
      factor_{_factor} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Merge::Merge(const TensorDomain* _out, const TensorDomain* _in, int _axis)
    : Expr(ExprType::Merge), out_{_out}, in_{_in}, axis_{_axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Reorder::Reorder(
    const TensorDomain* _out,
    const TensorDomain* _in,
    std::vector<int> _pos2axis)
    : Expr(ExprType::Reorder), out_{_out}, in_{_in}, pos2axis_{_pos2axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

const TensorView* split(const TensorView* tv, int axis, int factor) {
  
  const TensorDomain* td = tv->domain();
  if(axis<0) axis+=tv->domain()->size();
  assert(axis >= 0 && axis < td->size());
  const IterDomain* id = td->axis(axis);

  if (id->parallel_method() != ParallelType::Serial)
    throw std::runtime_error(
        "Splitting an axis of non-Serial iteration approach is not supported at this time. Parallelization strategy must be set after calling split.");

  if (tv->getComputeAtView() != nullptr)
    if (axis < tv->getComputeAtAxis())
        throw std::runtime_error("Cannot split axis within compute at range.");

  std::vector<const IterDomain*> new_domain;

  const Int* fact = new Int(factor);
  const Int* one = new Int(1);
  
  for (decltype(td->size()) i = 0; i < td->size(); i++) {
    if (i != axis)
      new_domain.push_back(td->axis(i));
    else {
      // outer loop size
      const Val* vo = add(div(sub(id->size(), one), fact), one);

      const Int* so = static_cast<const Int*>(vo);

      // outer loop IterDomain
      const IterDomain* ido = new IterDomain(so, id->parallel_method(), id->isReduction());
      new_domain.push_back(ido);

      // inner loop IterDomain
      const IterDomain* idi = new IterDomain(fact, id->parallel_method(), id->isReduction());
      new_domain.push_back(idi);
    }
  }
  const TensorDomain* split_td = new TensorDomain(new_domain);
  const TensorView* split_view = new TensorView(tv->tensor(), split_td);
  const Split* split_node = new Split(split_td, td, axis, fact); //For record keeping
  return split_view;
}

const TensorView* merge(const TensorView* tv, int axis) {
  const TensorDomain* td = tv->domain();
  assert(axis >= 0 && axis + 1 < td->size());

  if (tv->getComputeAtView() != nullptr)
    if (axis < tv->getComputeAtAxis())
        throw std::runtime_error("Cannot split axis within compute at range.");

  const IterDomain* first = tv->domain()->axis(axis);
  const IterDomain* second = tv->domain()->axis(axis+1);

  assert(first->isReduction() == second->isReduction());
  assert(first->parallel_method() == second->parallel_method());

  const Val* merged_id_size = mul(first->size(), second->size());
  const IterDomain* merged_id =
      new IterDomain(static_cast<const Int*>(merged_id_size), first->parallel_method(), first->isReduction());

  std::vector<const IterDomain*> new_domain;
  for (decltype(td->size()) i = 0; i < td->size(); i++) {
    if (i < axis || i > axis + 1)
      new_domain.push_back(td->axis(i));
    else if (i == axis) {
      new_domain.push_back(merged_id);
    }
  }
  const TensorDomain* merged_td = new TensorDomain(new_domain);
  const TensorView* merged_tv = new TensorView(tv->tensor(), merged_td);
  const Merge* merge_node = new Merge(merged_td, td, axis); //For record keeping
  return merged_tv;
}

/*
 * TODO: How do we coordinate the uses of tensor and the tensorview used here,
 * Do we only support these operations on tensorview? Do we replace all
 * instances of tensor with the tensorview created here?
 */
const TensorView* split(const Tensor* tensor, int axis, int factor) {
  throw std::runtime_error("For now tensors must be converted to tensor views before calling split.");
  //return split(new TensorView(tensor, tensor->domain()), axis, factor);
}

const TensorView* merge(const Tensor* tensor, int axis) {
  throw std::runtime_error("For now tensors must be converted to tensor views before calling merge.");
  //return merge(new TensorView(tensor, tensor->domain()), axis);
}

const TensorView* reorder(
    const Tensor* tensor,
    std::unordered_map<int, int> axis2pos) {
  throw std::runtime_error("For now tensors must be converted to tensor views before calling reorder.");
  //return reorder(new TensorView(tensor, tensor->domain()), axis2pos);
}

/*
 * Takes axis2pos map, axis2pos[old_pos] = new_pos, to modify the ordering of the iter
 * axes.
 * TODO: Figure out if this is a valid reorder. That will be dependant on compute_at
 * and any reduction axes in the tensor/tensorview. We can't reorder so that any reduction
 * dimension ends up inside the compute_at axis.
 */ 
const TensorView* reorder(
    const TensorView* tv,
    std::unordered_map<int, int> axis2pos) {
  const TensorDomain* td = tv->domain();
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
  if(tv->getComputeAtView != nullptr){
    for(int i = 0; i < tv->getComputeAtAxis(); i++){
      if(pos2axis[i] != i)
        throw std::runtime_error("Cannot reorder axis within compute at range.");
    }
  }

  std::vector<const IterDomain*> reordered_domain;

  for (int i = 0; i < pos2axis.size(); i++) {
    reordered_domain.push_back(td->axis(pos2axis[i]));
  }
  const TensorDomain* reordered_td = new TensorDomain(reordered_domain);
  const TensorView* reordered_view = new TensorView(tv->tensor(), reordered_td);
  const Reorder* merge_node = new Reorder(reordered_td, td, pos2axis);
  return reordered_view;
}


void ComputeAt_impl(const TensorView* consumer, const TensorView* producer, int axis){
  /*
   * TODO:
   * Recursive compute_at:
   * Recurse backward from consumer, to producer, make sure there's a dependency chain there.
   * After recursing, recurse again, and call ComputeAt for all tensors between producer and consumer.
   * 
   * Assert direct consumer/producer relationship.
   * Compute at modifies the consumer, not the producer.
   */

  const Fusion* fusion = FusionGuard::getCurFusion();
  const Expr* expr = fusion->origin(producer);
  bool direct_relationship = false;
  for(const Val* out : expr->outputs())
    if(out->same_as(consumer))
      direct_relationship = true;
  
  if(!direct_relationship)
    throw std::runtime_error("Compute at is currently only supported on direct producer/consumer relationships.");

  using size_type = decltype(consumer->domain()->size());

  bool matching_dims = true;
  if(consumer->domain()->size() != producer->domain()->size()){
    size_type producer_iter_dims = 0;
    for(size_type i = 0; i < producer->domain()->size(); i++)
      if(!producer->domain()->axis(i)->isReduction())
        producer_iter_dims++;
    if(producer_iter_dims != consumer->domain()->size())
      matching_dims = false;      
  }

  /*
   * Need replay function producer/consumer.
   */
}

} // namespace fuser
} // namespace jit
} // namespace torch
