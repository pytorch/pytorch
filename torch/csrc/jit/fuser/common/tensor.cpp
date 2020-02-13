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
      contiguity_(infer_contiguity_from_tensor_type(tensor_type)) {}

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

const TensorView* split(const Tensor* tensor, int axis, int factor) {
  using size_type = std::vector<const IterDomain*>::size_type;
  const TensorDomain* td = tensor->domain;

  assert(axis > 0 && axis < td->size());
  std::vector<const IterDomain*> new_domain;
  for (size_type i = 0; i < td->size(); i++) {
    if (i != axis)
      new_domain.push_back(td->axis(i));
    else {
      const Int* fact = new Int(factor);
      const Int* one = new Int(1);
      // outer loop size
      const Val* vo = add(div(sub(td->axis(i)->size(), one), fact), one);

      const Int* so = static_cast<const Int*>(vo);

      // outer loop IterDomain
      const IterDomain* ido = new IterDomain(so);
      // inner loop IterDomain
      const IterDomain* idi = new IterDomain(fact);
    }
  }
  // const TensorView* tv = ;
  return new TensorView(tensor, new TensorDomain(new_domain));
}

const TensorView* merge(const Tensor* tensor, int axis) {
  return new TensorView(tensor, tensor->domain);
}

} // namespace fuser
} // namespace jit
} // namespace torch
