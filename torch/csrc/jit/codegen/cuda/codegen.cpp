
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <sstream>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace codegen {

namespace {

class CudaKernelGenerator : private OptInConstDispatch {
 public:
  static std::string generate(
      const Kernel* kernel,
      const std::string& kernel_name) {
    CudaKernelGenerator codegen(kernel);
    codegen.genDeclaration(kernel_name);
    codegen.genPrologue();
    codegen.startBlock();
    codegen.genBody();
    codegen.endBlock();
    TORCH_CHECK(codegen.block_nest_level_ == 0);
    return codegen.code_.str();
  }

 private:
  explicit CudaKernelGenerator(const Kernel* kernel) : kernel_(kernel) {}

  // Generates the kernel function declaration
  void genDeclaration(const std::string& kernel_name) {
    const auto& kernel_summary = kernel_->summary();

    code_ << "__global__ void " << kernel_name << "(";

    std::vector<Val*> params;

    // Inputs
    for (auto val : kernel_->inputs()) {
      params.push_back(val);
    }

    // Outputs
    for (auto val : kernel_->outputs()) {
      params.push_back(val);
    }

    // Global buffers
    for (auto allocate : kernel_summary.global_allocations) {
      params.push_back(allocate->buffer());
    }

    // Generate parameter declarations
    for (Val* val : params) {
      switch (val->getValType().value()) {
        case ValType::KirTensorView:
          // TODO(kir): review this
          code_ << "Tensor<" << val->getDataType().value() << ", "
                << TensorDomain::noReductions(val->as<kir::TensorView>()
                                                  ->fuserTv()
                                                  ->getMaybeRFactorDomain())
                       .size()
                << "> T" << val->name();
          break;
        case ValType::KirScalar:
          code_ << val->getDataType().value() << " " << val;
          break;
        default:
          TORCH_CHECK(!"Unexpected parameter type");
      }

      if (val != params.back()) {
        code_ << ", ";
      }
    }

    // Kernels generating random numbers take extra (seed, offset) arguments
    if (kernel_summary.is_stochastic) {
      code_ << ", unsigned long long seed, unsigned long long offset";
    }

    code_ << ") ";
  }

  // Generates setup code which is executed before the kernel body
  void genPrologue() {
    const auto& kernel_summary = kernel_->summary();

    // Random number generator (optional)
    if (kernel_summary.is_stochastic) {
      indent() << "const int idx = blockIdx.x*blockDim.x + threadIdx.x;\n";
      indent() << "Philox rnd(seed, idx, offset);\n";
    }

    // Do we have any dynamic shared memory buffers?
    const bool has_dynamic_smem =
        !kernel_summary.dynamic_smem_allocations.empty();

    // Do we have any reductions?
    const bool has_reductions = kernel_summary.has_block_reductions ||
        kernel_summary.has_grid_reductions;

    // Shared memory
    if (has_dynamic_smem || has_reductions) {
      indent() << "alignas("
               << dataTypeSize(kernel_summary.largest_smem_data_type)
               << ") extern __shared__ char array[];\n";

      if (has_dynamic_smem) {
        indent() << "unsigned offset = "
                 << "((blockDim.x * blockDim.y * blockDim.z) * sizeof("
                 << kernel_summary.largest_smem_data_type << "));\n";
      }

      if (has_reductions) {
        indent() << "void* shared_mem = array;\n";
      }
    }
  }

  void genBody() {
    for (auto expr : kernel_->exprs()) {
      OptInConstDispatch::handle(expr);
    }
  }

  void startBlock() {
    code_ << "{\n";
    ++block_nest_level_;
  }

  void endBlock() {
    --block_nest_level_;
    TORCH_CHECK(block_nest_level_ >= 0);
    code_ << "}\n";
  }

  std::ostream& indent() {
    for (int i = 0; i < block_nest_level_; ++i) {
      code_ << "  ";
    }
    return code_;
  }

  void handle(const kir::Bool* node) final {}

  void handle(const kir::Float* node) final {}

  void handle(const kir::Half* node) final {}

  void handle(const kir::Int* node) final {}

  void handle(const kir::NamedScalar* node) final {}

  void handle(const kir::IterDomain* node) final {}

  void handle(const kir::TensorDomain* node) final {}

  void handle(const kir::TensorView* node) final {}

  void handle(const kir::UnaryOp* node) final {}

  void handle(const kir::BinaryOp* node) final {}

  void handle(const kir::TernaryOp* node) final {}

  void handle(const kir::ReductionOp* node) final {}

  void handle(const kir::BroadcastOp* node) final {}

  void handle(const kir::GridReduction* node) final {}

  void handle(const kir::ForLoop* node) final {}

  void handle(const kir::IfThenElse* node) final {}

  void handle(const kir::Allocate* node) final {}

  void handle(const kir::Sync* node) final {}

 private:
  std::stringstream code_;
  const Kernel* kernel_;
  int block_nest_level_ = 0;
};

} // namespace

std::string generateCudaKernel(
    const Kernel* kernel,
    const std::string& kernel_name) {
  return CudaKernelGenerator::generate(kernel, kernel_name);
}

} // namespace codegen
} // namespace fuser
} // namespace jit
} // namespace torch
