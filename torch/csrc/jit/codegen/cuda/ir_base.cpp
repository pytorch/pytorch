#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base.h>

#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <torch/csrc/jit/ir.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

#include <torch/csrc/jit/codegen/cuda/ir_nodes.h>


/*
 * Statement member definitions & related functions
 */

// When we create a Val or EXPR we immediately register them with the active
// fusion.
Val::Val(ValType _vtype, DataType _dtype)
    : vtype_{_vtype}, dtype_{_dtype} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion != nullptr) {
    this->name_ = fusion->registerVal(this);
    this->fusion_ = fusion;
  } else {
    TORCH_CHECK(false, "No active fusion group found when creating a Val.");
  }
}

  c10::optional<ValType> Val::getValType() const noexcept {
    return vtype_;
  }

  c10::optional<DataType> Val::getDataType() const {
    TORCH_INTERNAL_ASSERT(dtype_ != DataType::Null, "Value does not have a data type.");
    return dtype_;
  }

  bool Val::isScalar() {
    return vtype_ == ValType::Scalar;
  }

Expr* Val::getOrigin() {
  return (fusion_->origin(this));
}

  bool IRInputOutput::hasInput(const Val* const input) const {
    for (auto val : inputs_)
      if (val == input)
        return true;
    return false;
  }

  bool IRInputOutput::hasOutput(const Val* const output) const {
    for (auto val : outputs_)
      if (val == output)
        return true;
    return false;
  }

void IRInputOutput::removeOutput(Val* val) {
    auto it = outputs_.begin();
    for (; it != outputs_.end(); ++it) {
      if ((*it) == val)
        break;
    }
    assert(it != outputs_.end());
    outputs_.erase(it);
  }
Expr::Expr(ExprType _type) : type_{_type} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion == nullptr)
    TORCH_CHECK(false, "No active fusion group found when creating an Expr.");
  this->fusion_ = fusion;
}

Statement::~Statement() {}

/*
 * Val member definitions
 */

Val::~Val() {}

/*
 * IRInputOutput member definitions
 */

IRInputOutput::~IRInputOutput() {}

/*
 * Expr member definitions
 */

Expr::~Expr() {}

} // namespace fuser
} // namespace jit
} // namespace torch
