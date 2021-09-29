#pragma once

#include <unordered_map>
#include <unordered_set>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// A class that analyzes the given program relevant for Cuda backends.
class CudaAnalysis : public IRVisitor {
 public:
  CudaAnalysis() {
    gpu_block_extents_ = {alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};
    gpu_thread_extents_ = {
        alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};
  }
  bool is_buf_store_target(BufPtr buf) const {
    return store_targets_.count(buf) > 0;
  }

  const std::unordered_set<VarPtr>& thread_local_bufs() const {
    return thread_local_bufs_;
  }

  const std::unordered_set<VarPtr>& cross_block_bufs() const {
    return cross_block_bufs_;
  }

  const std::vector<ExprPtr>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  const std::vector<ExprPtr>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

 private:
  void visit(StorePtr v) override {
    store_targets_.insert(v->buf());
  }

  void visit(AllocatePtr v) override;
  void visit(FreePtr v) override;
  void visit(ForPtr v) override;

  std::unordered_set<BufPtr> store_targets_;
  std::unordered_set<VarPtr> thread_local_bufs_;
  std::unordered_set<VarPtr> cross_block_bufs_;

  std::vector<ExprPtr> gpu_block_extents_;
  std::vector<ExprPtr> gpu_thread_extents_;
};

// An IRMutator that replaces binding loop options with Cuda metavars, and masks
// statements blocks which should execute with less reach than the launch
// parameter extent.
//
// We do this by segmenting each block into chunks which should have the same
// execution parameters, then if those params differ from the max mask each dim.
class GPUMetaVarRewriter : public IRMutator {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit GPUMetaVarRewriter(const CudaAnalysis* cuda_analysis)
      : cuda_analysis_(cuda_analysis) {
    gpu_block_vars_ = {
        alloc<Var>("blockIdx.x", kInt),
        alloc<Var>("blockIdx.y", kInt),
        alloc<Var>("blockIdx.z", kInt)};
    gpu_thread_vars_ = {
        alloc<Var>("threadIdx.x", kInt),
        alloc<Var>("threadIdx.y", kInt),
        alloc<Var>("threadIdx.z", kInt)};

    current_block_reach_ = {
        alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};
    current_thread_reach_ = {
        alloc<IntImm>(1), alloc<IntImm>(1), alloc<IntImm>(1)};
  }

  StmtPtr mutate(ForPtr v) override;
  StmtPtr mutate(BlockPtr v) override;

  const std::vector<VarPtr>& gpu_block_vars() const {
    return gpu_block_vars_;
  }

  const std::vector<VarPtr>& gpu_thread_vars() const {
    return gpu_thread_vars_;
  }

  const std::vector<ExprPtr>& gpu_block_extents() const {
    return cuda_analysis_->gpu_block_extents();
  }

  const std::vector<ExprPtr>& gpu_thread_extents() const {
    return cuda_analysis_->gpu_thread_extents();
  }

 private:
  // When processing a block, stores the contents of each sub-segment.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  class Segment {
   public:
    void reset(bool mask) {
      stmts_.clear();
      mask_ = mask;
    }

    bool empty() const {
      return stmts_.empty();
    }

    std::vector<StmtPtr>& stmts() {
      return stmts_;
    }
    bool mask() {
      return mask_;
    }

   private:
    std::vector<StmtPtr> stmts_;
    bool mask_{true};
  };

  // Returns true if the current execution scope is equivalent to the launch
  // parameters.
  bool isFullExtent();

  std::vector<VarPtr> gpu_block_vars_;
  std::vector<VarPtr> gpu_thread_vars_;

  std::vector<ExprPtr> current_block_reach_;
  std::vector<ExprPtr> current_thread_reach_;

  const CudaAnalysis* cuda_analysis_;
};

// A class that overrides the underlying IRPrinter to produce Cuda C.
class CudaPrinter : public IRPrinter {
 public:
  explicit CudaPrinter(
      std::ostream* os,
      const CudaAnalysis* cuda_analysis,
      bool has_random)
      : IRPrinter(*os), cuda_analysis_(cuda_analysis) {
    if (has_random) {
      rand_func_ = alloc<Var>("rand", kHandle);
    }
  }

  void visit(CastPtr v) override;
  void visit(IntrinsicsPtr v) override;
  void visit(ForPtr v) override;

  void visit(LoadPtr v) override;
  void visit(StorePtr v) override;
  void visit(AtomicAddPtr v) override;
  void visit(MaxPtr v) override;
  void visit(MinPtr v) override;
  void visit(IfThenElsePtr v) override;
  void visit(BlockPtr v) override;
  void visit(AllocatePtr v) override;
  void visit(FreePtr v) override;
  void visit(LetPtr v) override;

  void visit(ExternalCallPtr v) override;

  VarPtr rand_func() const {
    return rand_func_;
  }

  std::string dtypeToCppString(const Dtype& dtype) override;

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  VarPtr rand_func_;
  const CudaAnalysis* cuda_analysis_;

  void print_flat_alloc(AllocatePtr alloc);
};

// Construct Cuda C from the buffer and tensor input, and invoke the kernel
// when real arguments are provided.
class TORCH_CUDA_CU_API CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CudaCodeGen(StmtPtr stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCUDA, at::cuda::current_device())) {
    Initialize();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CudaCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::Device(at::kCUDA, at::cuda::current_device()),
      const std::string& kernel_func_name = "func")
      : CodeGen(stmt, buffer_args, device, kernel_func_name) {
    Initialize();
  }

  ~CudaCodeGen() override;

  void call_raw(const std::vector<void*>& args) override;
  void call(const std::vector<CallArg>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

  at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      c10::optional<c10::ScalarType> dtype_opt,
      c10::optional<c10::Layout> layout_opt,
      c10::optional<c10::Device> device_opt,
      c10::optional<bool> pin_memory_opt) override;

  const std::vector<ExprPtr>& gpu_block_extents() const {
    return cuda_analysis_->gpu_block_extents();
  }

  const std::vector<ExprPtr>& gpu_thread_extents() const {
    return cuda_analysis_->gpu_thread_extents();
  }

  std::string getCodeText(const std::string& attr = "") override {
    return oss_.str();
  }

 private:
  void Initialize();

  void CompileToNVRTC(const std::string& code, const std::string& func_name);

  UniqueNameManager* name_manager() {
    if (!printer_) {
      throw std::runtime_error("Null IRPrinter is not expected");
    }
    return printer_->name_manager();
  }

  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_;
  std::unique_ptr<CudaPrinter> printer_;
  std::unique_ptr<CudaAnalysis> cuda_analysis_;
  std::unique_ptr<GPUMetaVarRewriter> metavar_rewriter_;
  std::unordered_set<std::string> taken_func_names;
  CUfunction function_;
  bool has_random_ = false;

  std::string GetUniqueFuncName(const std::string& func_prefix);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
