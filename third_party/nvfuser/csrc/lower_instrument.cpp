#include <iter_visitor.h>
#include <kernel_ir_dispatch.h>
#include <lower2device.h>
#include <lower_magic_zero.h>

#include <lower_instrument.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class Instrumentor : private kir::IrVisitor {
 public:
  Instrumentor(const std::vector<Expr*>& exprs) {
    IrVisitor::handle(exprs);

    if (profile_.getNumberOfProfileEntries() == 0) {
      exprs_ = exprs;
      return;
    }

    // Allocate a new TensorView as a backing buffer
    allocateBuffer();
    profile_.setBuffer(buffer_);

    // Insert the allocation expression at the beginning of the
    // top-level expressions
    exprs_.push_back(buffer_alloc_);
    exprs_.insert(exprs_.end(), exprs.begin(), exprs.end());
  }

  const kir::KernelPerformanceProfile& profile() const {
    return profile_;
  }

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

 private:
  using IrVisitor::handle;

  //! Profile all grid reductions.
  //! TODO: support other variants of grid reductions (e.g.,
  //! GroupedGridReduction)
  void handle(kir::GridReduction* expr) final {
    profile_.registerExpr(expr);
  }

  void handle(kir::GroupedGridReduction* expr) final {
    profile_.registerExpr(expr);
  }

  void allocateBuffer() {
    const auto num_profile_entries = profile_.getNumberOfProfileEntries();

    // If nothing to profile, do not allocate anything
    if (num_profile_entries == 0) {
      return;
    }

    // Allocate two integers for each entry. One is used for accumulating
    // cycles, and another for couting the number of hits
    const std::vector<IterDomain*> new_buffer_ids = {
        IterDomainBuilder(
            GpuLower::current()->kernel()->zeroVal(),
            IrBuilder::create<Int>(num_profile_entries))
            .build(),
        IterDomainBuilder(
            GpuLower::current()->kernel()->zeroVal(), IrBuilder::create<Int>(2))
            .build()};

    const auto buffer_domain = IrBuilder::create<TensorDomain>(new_buffer_ids);

    buffer_ = IrBuilder::create<TensorView>(
        buffer_domain, DataType::Int, MemoryType::Global);

    buffer_alloc_ = IrBuilder::create<kir::Allocate>(
        buffer_, buffer_->getMemoryType(), nullptr, true);
  }

 private:
  std::vector<Expr*> exprs_;
  kir::KernelPerformanceProfile profile_;
  TensorView* buffer_ = nullptr;
  kir::Allocate* buffer_alloc_ = nullptr;
};

} // namespace

std::vector<Expr*> instrumentKernel(const std::vector<Expr*>& exprs) {
  if (!isOptionEnabled(EnableOption::KernelProfile)) {
    return exprs;
  }

  Instrumentor inst(exprs);

  GpuLower::current()->profile() = inst.profile();

  return inst.exprs();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
