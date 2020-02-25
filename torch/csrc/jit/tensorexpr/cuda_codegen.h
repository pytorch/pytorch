#pragma once

#include <unordered_map>
#include <unordered_set>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/nvrtc_stub/ATenNVRTC.h"
#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAGuard.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/unique_name_manager.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// A class that overrides the underlying IRPrinter to produce Cuda C.
class CudaPrinter : public IRPrinter {
 public:
  explicit CudaPrinter(std::ostream* os, bool has_random) : IRPrinter(*os) {
    if (has_random) {
      rand_func_ = new Var("rand", kHandle);
    }
  }

  void visit(const Cast* v) {
    auto dtype = v->dtype();
    if (dtype == kFloat32) {
      os() << "float";
    } else {
      os() << dtype;
    }
    os() << "(";
    v->src_value()->accept(this);
    os() << ")";
  }

  void visit(const Intrinsics* v);
  void visit(const For* v);

  void visit(const Load* v);
  void visit(const Max* v);
  void visit(const Min* v);
  void visit(const IfThenElse* v);

  const std::vector<const Expr*>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  const std::vector<const Expr*>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

  const Var* rand_func() const {
    return rand_func_;
  }

  using IRPrinter::name_manager;

 private:
  std::vector<const Expr*> gpu_block_extents_;
  std::vector<const Expr*> gpu_thread_extents_;
  const Var* rand_func_;
};

// Construct Cuda C from the buffer and tensor input, and invoke the kernel
// when real arguments are provided.
class TORCH_API CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  CudaCodeGen(Stmt* stmt, Ts... ts)
      : CodeGen(stmt, std::forward<Ts>(ts)...) {
    Initialize();
  }

  CudaCodeGen(Stmt* stmt, const std::vector<BufferArg>& buffer_args)
      : CodeGen(stmt, buffer_args) {
    Initialize();
  }

  ~CudaCodeGen() override {}

  TORCH_API void call(const std::vector<CallArg>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

 private:
  TORCH_API void Initialize();

  void CompileToNVRTC(const std::string& code);

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
  CUfunction function_;
  bool has_random_ = false;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
