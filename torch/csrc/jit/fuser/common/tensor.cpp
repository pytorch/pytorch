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
    const TensorView* _out,
    const TensorView* _in,
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

Merge::Merge(const TensorView* _out, const TensorView* _in, int _axis)
    : Expr(ExprType::Merge), out_{_out}, in_{_in}, axis_{_axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Reorder::Reorder(
    const TensorView* _out,
    const TensorView* _in,
    std::vector<int> _pos2axis)
    : Expr(ExprType::Reorder), out_{_out}, in_{_in}, pos2axis_{_pos2axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

const TensorView* split(const TensorView* tv, int axis, int factor) {
  
  const TensorDomain* td = tv->domain();
  assert(axis > 0 && axis < td->size());
  const IterDomain* id = td->axis(axis);

  if (id->parallel_method() != ParallelType::Serial)
    throw std::runtime_error(
        "Splitting an axis of non-Serial iteration approach is not supported at this time. Parallelization strategy must be set after calling split.");

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
  const TensorView* split_view = new TensorView(tv->tensor(), new TensorDomain(new_domain));
  const Split* split_node = new Split(split_view, tv, axis, fact); //For record keeping
  return split_view;
}

const TensorView* merge(const TensorView* tv, int axis) {
  const TensorDomain* td = tv->domain();
  assert(axis >= 0 && axis + 1 < td->size());

  const IterDomain* first = tv->domain()->axis(axis);
  const IterDomain* second = tv->domain()->axis(axis);

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
  const TensorView* new_tv = new TensorView(tv->tensor(), new TensorDomain(new_domain));
  const Merge* merge_node = new Merge(new_tv, tv, axis); //For record keeping
  return new_tv;
}

/*
 * TODO: How do we coordinate the uses of tensor and the tensorview used here,
 * Do we only support these operations on tensorview? Do we replace all
 * instances of tensor with the tensorview created here?
 */
const TensorView* split(const Tensor* tensor, int axis, int factor) {
  return split(new TensorView(tensor, tensor->domain()), axis, factor);
}

const TensorView* merge(const Tensor* tensor, int axis) {
  
  return merge(new TensorView(tensor, tensor->domain()), axis);
}

const TensorView* reorder(
    const Tensor* tensor,
    std::unordered_map<int, int> axis2pos) {
  return reorder(new TensorView(tensor, tensor->domain()), axis2pos);
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
  auto ndims = tv->domain()->size();
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
  for (int i = 0; i < pos2axis.size(); i++) {
    if (pos2axis[i] == -1)
      pos2axis[i] = *it++;
  }

  std::vector<const IterDomain*> domain;

  for (int i = 0; i < pos2axis.size(); i++) {
    domain.push_back(tv->domain()->axis(pos2axis[i]));
  }
  const TensorView* reordered_view = new TensorView(tv->tensor(), new TensorDomain(domain));
  const Reorder* merge_node = new Reorder(reordered_view, tv, pos2axis);
  return reordered_view;
}

} // namespace fuser
} // namespace jit
} // namespace torch
