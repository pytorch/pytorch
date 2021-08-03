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
    gpu_block_extents_ = {new IntImm(1), new IntImm(1), new IntImm(1)};
    gpu_thread_extents_ = {new IntImm(1), new IntImm(1), new IntImm(1)};
  }
  bool is_buf_store_target(const Buf* buf) const {
    return store_targets_.count(buf) > 0;
  }

  const std::unordered_set<const Var*>& thread_local_bufs() const {
    return thread_local_bufs_;
  }

  const std::unordered_set<const Var*>& cross_block_bufs() const {
    return cross_block_bufs_;
  }

  const std::vector<const Expr*>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  const std::vector<const Expr*>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

 private:
  void visit(const Store* v) override {
    store_targets_.insert(v->buf());
  }

  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const For* v) override;

  std::unordered_set<const Buf*> store_targets_;
  std::unordered_set<const Var*> thread_local_bufs_;
  std::unordered_set<const Var*> cross_block_bufs_;

  std::vector<const Expr*> gpu_block_extents_;
  std::vector<const Expr*> gpu_thread_extents_;
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
        new Var("blockIdx.x", kInt),
        new Var("blockIdx.y", kInt),
        new Var("blockIdx.z", kInt)};
    gpu_thread_vars_ = {
        new Var("threadIdx.x", kInt),
        new Var("threadIdx.y", kInt),
        new Var("threadIdx.z", kInt)};

    current_block_reach_ = {new IntImm(1), new IntImm(1), new IntImm(1)};
    current_thread_reach_ = {new IntImm(1), new IntImm(1), new IntImm(1)};
  }

  Stmt* mutate(const For* v) override;
  Stmt* mutate(const Block* v) override;

  const std::vector<const Var*>& gpu_block_vars() const {
    return gpu_block_vars_;
  }

  const std::vector<const Var*>& gpu_thread_vars() const {
    return gpu_thread_vars_;
  }

  const std::vector<const Expr*>& gpu_block_extents() const {
    return cuda_analysis_->gpu_block_extents();
  }

  const std::vector<const Expr*>& gpu_thread_extents() const {
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

    std::vector<Stmt*>& stmts() {
      return stmts_;
    }
    bool mask() {
      return mask_;
    }

   private:
    std::vector<Stmt*> stmts_;
    bool mask_{true};
  };

  // Returns true if the current execution scope is equivalent to the launch
  // parameters.
  bool isFullExtent();

  std::vector<const Var*> gpu_block_vars_;
  std::vector<const Var*> gpu_thread_vars_;

  std::vector<const Expr*> current_block_reach_;
  std::vector<const Expr*> current_thread_reach_;

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
      rand_func_ = new Var("rand", kHandle);
    }
  }

  void visit(const Cast* v) override;
  void visit(const Intrinsics* v) override;
  void visit(const For* v) override;

  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const AtomicAdd* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const Block* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const Let* v) override;

  void visit(const ExternalCall* v) override;

  const Var* rand_func() const {
    return rand_func_;
  }

  std::string dtypeToCppString(const Dtype& dtype) override;

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  const Var* rand_func_;
  const CudaAnalysis* cuda_analysis_;

  void print_flat_alloc(const Allocate* alloc);
};

// Construct Cuda C from the buffer and tensor input, and invoke the kernel
// when real arguments are provided.
class TORCH_CUDA_CU_API CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CudaCodeGen(Stmt* stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCUDA, at::cuda::current_device())) {
    Initialize();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CudaCodeGen(
      Stmt* stmt,
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

  const std::vector<const Expr*>& gpu_block_extents() const {
    return cuda_analysis_->gpu_block_extents();
  }

  const std::vector<const Expr*>& gpu_thread_extents() const {
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
