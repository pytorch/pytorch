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
  bool is_buf_store_target(const Buf* buf) const {
    return store_targets_.count(buf) > 0;
  }

  const std::unordered_set<const Var*>& thread_local_bufs() const {
    return thread_local_bufs_;
  }

  const std::unordered_set<const Var*>& cross_block_bufs() const {
    return cross_block_bufs_;
  }

 private:
  void visit(const Store* v) override {
    store_targets_.insert(v->buf());
  }

  void visit(const Allocate* v) override;
  void visit(const Free* v) override;

  std::unordered_set<const Buf*> store_targets_;
  std::unordered_set<const Var*> thread_local_bufs_;
  std::unordered_set<const Var*> cross_block_bufs_;
};

// An IRMutator that replaces binding loop options with Cuda metavars.
class GPUMetaVarRewriter : public IRMutator {
 public:
  explicit GPUMetaVarRewriter() {
    gpu_block_vars_ = {new Var("blockIdx.x", kInt),
                       new Var("blockIdx.y", kInt),
                       new Var("blockIdx.z", kInt)};
    gpu_thread_vars_ = {new Var("threadIdx.x", kInt),
                        new Var("threadIdx.y", kInt),
                        new Var("threadIdx.z", kInt)};
  }

  Stmt* mutate(const For* v) override;
  Stmt* mutate(const Block* v) override;

  const std::vector<const Expr*>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  const std::vector<const Expr*>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

  const std::vector<const Var*>& gpu_block_vars() const {
    return gpu_block_vars_;
  }

  const std::vector<const Var*>& gpu_thread_vars() const {
    return gpu_thread_vars_;
  }

 private:
  std::vector<const Expr*> gpu_block_extents_;
  std::vector<const Expr*> gpu_thread_extents_;

  std::vector<const Var*> gpu_block_vars_;
  std::vector<const Var*> gpu_thread_vars_;

  bool need_sync_ = false;
  const Expr* last_thread_dim_ = nullptr;
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

  const Var* rand_func() const {
    return rand_func_;
  }

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  const Var* rand_func_;
  const CudaAnalysis* cuda_analysis_;
};

// Construct Cuda C from the buffer and tensor input, and invoke the kernel
// when real arguments are provided.
class TORCH_CUDA_API CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  CudaCodeGen(Stmt* stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCUDA, at::cuda::current_device())) {
    Initialize();
  }

  CudaCodeGen(
      Stmt* stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::Device(at::kCUDA, at::cuda::current_device()))
      : CodeGen(stmt, buffer_args, device) {
    Initialize();
  }

  ~CudaCodeGen() override;

  void call(const std::vector<CallArg>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
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
  CUfunction function_;
  bool has_random_ = false;

  std::string GetUniqueFuncName(const std::string& func_prefix);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
