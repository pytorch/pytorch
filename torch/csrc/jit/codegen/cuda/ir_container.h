#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class IrBuilderPasskey;
class ExprPasskey;
class OptOutMutator;

class Int;
class Bool;
class NamedScalar;

// Passkey for container to register names with statements
class IrContainerPasskey {
  friend class IrContainer;

 private:
  explicit IrContainerPasskey() {}
};

class TORCH_CUDA_CU_API IrContainer : public PolymorphicBase {
 public:
  IrContainer();

  IrContainer(const IrContainer& other);
  IrContainer(IrContainer&& other) noexcept;

  IrContainer& operator=(const IrContainer& other);
  IrContainer& operator=(IrContainer&& other) noexcept;

  virtual ~IrContainer();

  bool inContainer(const Statement* stmt) const;

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    TORCH_CHECK(
        inContainer(stmt), msg, " it was not found in the active container.");
  }

  //! Return in insertion order
  const std::deque<Val*> deterministic_vals() const noexcept {
    std::deque<Val*> vals_deque;
    std::transform(
        vals_up_.begin(),
        vals_up_.end(),
        std::back_inserter(vals_deque),
        [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
    return vals_deque;
  }

  //! Register the Statement with this container
  virtual void registerStmt(IrBuilderPasskey, Statement* stmt);

  //! Register the Val with this container
  virtual void registerVal(IrBuilderPasskey, Val* val);

  //! Register expr with this container.
  virtual void registerExpr(IrBuilderPasskey, Expr* expr);

  //! Allow expr's to register themselves with a container, this is only used
  //! for broadcastOp so it can register itself in its constructor so root maps
  //! can be built.
  virtual void registerExpr(ExprPasskey, Expr* expr);

  //! Return the set of Exprs registered with this fusion. Warning: This will
  //! return exprs outside inputs/outputs, so can be unsafe for use with
  //! segmented fusions.
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept {
    return exprs_;
  }

  //! Return the set of Vals registered with this fusion
  const std::unordered_set<Val*>& vals() const noexcept {
    return vals_;
  }

  // Shortcuts for frequently used vals
  Int* zeroVal();
  Int* oneVal();
  Bool* falseVal();
  Bool* trueVal();
  NamedScalar* magicZeroVal();

 protected:
  static IrCloner copy(const IrContainer* from, IrContainer* to);

  friend void swap(IrContainer& a, IrContainer& b) noexcept;

  // Let mutator remove Exprs.
  friend OptOutMutator;

  virtual void removeExpr(Expr* expr);

  //! Completely remove val from the fusion, break all dependencies associated
  //! with it
  virtual void removeVal(Val* val);

  //! Register the Val with this container
  virtual void registerVal(Val* val);

  //! Register expr with this container.
  virtual void registerExpr(Expr* expr);

  StmtNameType getValName(ValType vtype) {
    if (val_type_name_map_.find(vtype) == val_type_name_map_.end()) {
      val_type_name_map_[vtype] = 0;
    }
    return val_type_name_map_[vtype]++;
  }

  StmtNameType getExprName() {
    return expr_name_counter_++;
  }

  void clear() noexcept;

  // Deque of unique pointer is the memory owning data structure
  std::deque<std::unique_ptr<Val>> vals_up_;

  // A convenient set to return when we just need an unordered set to do
  // something like check if a Val is in this container
  std::unordered_set<Val*> vals_;

  // Deque of unique pointer is the memory owning data structure
  std::deque<std::unique_ptr<Expr>> exprs_up_;

  // A convenient set to return when we just need an unordered set to do
  // something like check if an Expr is in this container
  std::unordered_set<Expr*> exprs_;

  // Used to implement a generic "inContainer" that can be passed an invalid
  // pointer. Specifically a pointer to a Statement owned by another container
  // that has been freed. We can't check normally with the unordered_sets we
  // already have because it would require a const_cast from a constant
  // expr/val, or a dynamic cast from a Statement.
  std::unordered_set<void*> raw_ptrs_;

  // Values names counters
  std::unordered_map<ValType, StmtNameType, TypeHash> val_type_name_map_;

  // Expression names counter
  StmtNameType expr_name_counter_ = 0;

  // Manually store some persistent, frequently used nodes. It's very
  // challenging to do this anything but manually as detecting when a container
  // may or may not have one of these vals is tricky. Specifically because if
  // the container doesn't own it, it's hard to understand from the outside if
  // the node may have been removed then re-registered. It could also be tricky
  // to know when we're using a different container as in FusionCopy_test
  // demonstrates deleting then creating containers can result in the same
  // pointer for the container.
  std::unique_ptr<Bool> true_val_;
  std::unique_ptr<Bool> false_val_;
  std::unique_ptr<Int> one_val_;
  std::unique_ptr<Int> zero_val_;
  std::unique_ptr<NamedScalar> magic_zero_val_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
