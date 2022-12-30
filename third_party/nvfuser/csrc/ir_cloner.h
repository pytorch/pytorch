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
class TORCH_CUDA_CU_API IrCloner {
  friend class Statement;
  friend class IrBuilder;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit IrCloner(IrContainer* container);
  virtual ~IrCloner() {}

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
  virtual Statement* handle(const Statement* s);

 protected:
  // We keep track of the original -> clone map so we don't
  // duplicate clones of the same object if referenced multiple times
  std::unordered_map<const Statement*, Statement*> clones_map_;

 private:
  // The destination Fusion container
  IrContainer* ir_container_ = nullptr;

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
  virtual Statement* handle(const Statement* s) override;
  Statement* handle(const TensorDomain*);

  Fusion* fusion_;
};

//! Clone an IR node, forwarding the arguments to the IrCloner constructor.
template <class T>
T* IrBuilder::clone(const T* src, IrCloner* ir_cloner) {
  TORCH_INTERNAL_ASSERT(
      ir_cloner != nullptr,
      "Cannot use create when a cloner object is set. Use clone.");

  TORCH_INTERNAL_ASSERT(
      ir_cloner->container() != nullptr,
      "Cloner doesn't have a valid container to store cloned object.");

  T* dest = new T(src, ir_cloner);
  const Statement* src_stmt = dynamic_cast<const Statement*>(src);
  Statement* dest_stmt = dynamic_cast<Statement*>(dest);

  auto dest_container = ir_cloner->container();
  auto src_container = src_stmt->container();

  dest_container->registerStmt(IrBuilderPasskey(dest_container), dest_stmt);

  if (src_container != dest_container) {
    dest_stmt->setName(IrBuilderPasskey(dest_container), src_stmt->name());
  }

  ir_cloner->registerClone(src_stmt, dest_stmt);

  return dest;
}

template <typename T>
NVFUSER_DEFINE_CLONE(Attribute<T>)

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
