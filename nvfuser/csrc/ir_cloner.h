#pragma once

#include <c10/macros/Export.h>
#include <dispatch.h>
#include <ir_builder.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class IrContainer;

//! Clones nodes from an exiting Fusion
//!
//! \warning IrCloner machinery is a specialized helper for implementing
//!   Fusion copy operations and the and limited scope of RecomputeTv below.
//!   It is not intended for any other uses.
//!
class TORCH_CUDA_CU_API IrCloner : private OptInConstDispatch {
  friend class Statement;
  friend class IrBuilder;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit IrCloner(IrContainer* container);

  Statement* clone(const Statement* statement);

  template <class T>
  T* clone(const T* node) {
    return node ? clone(node->template as<Statement>())->template as<T>()
                : nullptr;
  }

  template <class T>
  std::vector<T*> clone(const std::vector<T*>& container) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<T*> copy;
    copy.reserve(container.size());
    for (auto p : container) {
      copy.push_back(clone(p));
    }
    return copy;
  }

  IrContainer* container() const {
    return ir_container_;
  }

 protected:
  void registerClone(const Statement* src, Statement* clone);

  void handle(const Statement*) override;
  void handle(const Val*) override;
  void handle(const Expr*) override;

  void handle(const TensorDomain*) override;
  void handle(const TensorView*) override;
  void handle(const IterDomain*) override;

  void handle(const Bool*) override;
  void handle(const Double*) override;
  void handle(const Int*) override;
  void handle(const ComplexDouble*) override;
  void handle(const NamedScalar*) override;

  void handle(const FullOp*) override;
  void handle(const ARangeOp*) override;
  void handle(const EyeOp*) override;
  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;
  void handle(const TernaryOp*) override;
  void handle(const RNGOp*) override;
  void handle(const BroadcastOp*) override;
  void handle(const ReductionOp*) override;
  void handle(const GroupedReductionOp*) override;
  void handle(const WelfordOp*) override;
  void handle(const LoadStoreOp*) override;
  void handle(const MmaOp*) override;
  void handle(const TransposeOp*) override;
  void handle(const ExpandOp*) override;
  void handle(const ShiftOp*) override;
  void handle(const GatherOp*) override;
  void handle(const ViewAsScalar*) override;
  void handle(const ViewOp*) override;

  void handle(const Split*) override;
  void handle(const Merge*) override;
  void handle(const Swizzle2D*) override;

 protected:
  // We keep track of the original -> clone map so we don't
  // duplicate clones of the same object if referenced multiple times
  std::unordered_map<const Statement*, Statement*> clones_map_;

 private:
  // The destination Fusion container
  IrContainer* ir_container_ = nullptr;

  // The dispatch interface doesn't allow returning values from
  // individual `handle()` methods, so they are storing the
  // result here
  Statement* clone_ = nullptr;

  // Builder to make all the new nodes
  IrBuilder builder_;
};

// Replicates all expressions used to generate the provided TensorView. Does not
// replicate inputs. Does not replicate scalar values. In other words the value
// provided will be recomputed from the inputs of the fusion.
class RecomputeTv : private IrCloner {
 public:
  // Replicates expressions and values in provided expressions.
  static TensorView* recompute(TensorView* tv);

 private:
  RecomputeTv(Fusion* fusion, std::vector<Expr*> exprs);

  void handle(const TensorDomain*) final;

  Fusion* fusion_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
