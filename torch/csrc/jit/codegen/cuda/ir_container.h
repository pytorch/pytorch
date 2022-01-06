#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

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
// Passkey for container to register names with statements
class IrContainerPasskey {
  friend class IrContainer;

 private:
  explicit IrContainerPasskey() {}
};

class TORCH_CUDA_CU_API IrContainer : public PolymorphicBase {
 public:
  IrContainer() = default;

  IrContainer(const IrContainer& other);
  IrContainer(IrContainer&& other) noexcept;

  IrContainer& operator=(const IrContainer& other);
  IrContainer& operator=(IrContainer&& other) noexcept;

  virtual ~IrContainer();

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

 protected:
  static IrCloner copy(const IrContainer* from, IrContainer* to);

  friend void swap(IrContainer& a, IrContainer& b) noexcept;

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

  bool inContainer(const Statement* stmt) const;

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    TORCH_CHECK(
        inContainer(stmt), msg, " it was not found in the active fusion.");
  }

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

  // Values names counters
  std::unordered_map<ValType, StmtNameType, TypeHash> val_type_name_map_;

  // Expression names counter
  StmtNameType expr_name_counter_ = 0;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
