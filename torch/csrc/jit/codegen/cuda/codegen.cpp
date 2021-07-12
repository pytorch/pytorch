#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <sstream>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace codegen {

namespace {

class CudaKernelGenerator : private OptInConstDispatch {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static constexpr char* kTab = "  ";

 public:
  static std::string generateKernelDefinition(
      const Kernel* kernel,
      const std::string& kernel_name) {
    CudaKernelGenerator codegen(kernel);
    codegen.genDeclaration(kernel_name);
    codegen.startBlock();
    codegen.genPrologue();
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
        case ValType::KirTensorView: {
          // TODO(kir): review this
          const auto tv = val->as<kir::TensorView>();
          code_ << "Tensor<" << val->getDataType().value() << ", "
                << TensorDomain::noReductions(
                       tv->fuserTv()->getMaybeRFactorDomain())
                       .size()
                << "> " << gen(tv);
          break;
        }
        case ValType::KirScalar:
          code_ << val->getDataType().value() << " " << gen(val);
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
#ifndef __HIP_PLATFORM_HCC__
               << dataTypeSize(kernel_summary.largest_smem_data_type)
#else
               << 8 // for HIP, we want 8-aligned even for smaller datatypes
#endif
               << ") extern __shared__ char array[];\n";

      if (has_dynamic_smem) {
        indent() << "unsigned offset = 0;\n";
      }

      if (has_reductions) {
        indent() << "void* shared_mem = array;\n";
        if (has_dynamic_smem) {
          indent() << "offset += "
                   << "((blockDim.x * blockDim.y * blockDim.z) * sizeof("
                   << kernel_summary.largest_smem_data_type << "));\n";
        }
      }
    }
  }

  void genBody() {
    for (auto expr : kernel_->topLevelExprs()) {
      OptInConstDispatch::handle(expr);
    }
  }

  void startBlock(bool continuation = false) {
    if (continuation) {
      code_ << "{\n";
    } else {
      indent() << "{\n";
    }
    ++block_nest_level_;
  }

  void endBlock(const char* sep = "\n") {
    --block_nest_level_;
    TORCH_CHECK(block_nest_level_ >= 0);
    indent() << "}" << sep;
  }

  std::ostream& indent() {
    for (int i = 0; i < block_nest_level_; ++i) {
      code_ << kTab;
    }
    return code_;
  }

  std::string gen(const Statement* stmt) {
    std::stringstream tmp_code;
    std::swap(tmp_code, code_);
    handle(stmt);
    std::swap(tmp_code, code_);
    return tmp_code.str();
  }

  std::string gen(const kir::TensorView* tv) {
    std::stringstream tv_name;
    tv_name << "T" << tv->name();
    return tv_name.str();
  }

  std::string genInline(const Statement* stmt) {
    const bool saved_inline = print_inline_;
    print_inline_ = true;
    const auto result = gen(stmt);
    print_inline_ = saved_inline;
    // NOLINTNEXTLINE(performance-no-automatic-move)
    return result;
  }

  void handle(const Statement* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const Expr* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const Val* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const kir::Bool* node) final {
    const auto def = node->getOrigin();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isSymbolic()) {
      code_ << "b" << node->name();
    } else {
      code_ << *node->value();
    }
  }

  void handle(const kir::Float* node) final {
    const auto def = node->getOrigin();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isSymbolic()) {
      code_ << "f" << node->name();
    } else {
      const int digits = std::numeric_limits<Float::ScalarType>::max_digits10;
      code_ << "float(" << std::setprecision(digits) << *node->value() << ")";
    }
  }

  void handle(const kir::Half* node) final {
    const auto def = node->getOrigin();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isSymbolic()) {
      code_ << "h" << node->name();
    } else {
      code_ << "__float2half(" << *node->value() << ")";
    }
  }

  void handle(const kir::Int* node) final {
    const auto def = node->getOrigin();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isSymbolic()) {
      code_ << "i" << node->name();
    } else {
      code_ << *node->value();
    }
  }

  void handle(const kir::NamedScalar* node) final {
    code_ << node->name();
  }

  void handle(const kir::TensorIndex* node) final {
    code_ << gen(node->view()) << "[";

    bool first = true;
    for (auto* ind : node->indices()) {
      if (!ind->isZeroInt()) {
        if (!first) {
          code_ << " + ";
        }
        code_ << genInline(ind);
        first = false;
      }
    }

    if (first) {
      code_ << "0";
    }

    code_ << "]";
  }

  void handle(const kir::IterDomain* node) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void handle(const kir::TensorDomain* node) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void handle(const kir::TensorView* node) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void handle(const kir::UnaryOp* node) final {
    if (!print_inline_) {
      indent() << gen(node->out());
      if (!node->out()->isScalar() && !node->in()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    if (auto op = inline_op_str(node->getUnaryOpType())) {
      code_ << *op << gen(node->in());
    } else {
      if (node->getUnaryOpType() == UnaryOpType::Cast) {
        const auto cast_str = cast_func_str(
            {node->in()->getDataType().value(),
             node->out()->getDataType().value()});
        code_ << cast_str.value();
      } else {
        code_ << node->getUnaryOpType();
      }

      code_ << "(";
      if (node->getUnaryOpType() == UnaryOpType::RandLike) {
        code_ << "rnd";
      } else {
        code_ << gen(node->in());
      }
      code_ << ")";
    }

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  std::string genBinaryOp(
      BinaryOpType op_type,
      const std::string& lhs,
      const std::string& rhs) {
    std::stringstream expr;
    if (auto op = inline_op_str(op_type)) {
      expr << lhs << " " << *op << " " << rhs;
    } else {
      expr << op_type << "(" << lhs << ", " << rhs << ")";
    }
    return expr.str();
  }

  void handle(const kir::BinaryOp* node) final {
    const auto op_type = node->getBinaryOpType();
    if (print_inline_) {
      // Inline expression: `lhs op rhs`
      code_ << genBinaryOp(op_type, gen(node->lhs()), gen(node->rhs()));
    } else {
      indent() << gen(node->out());
      if (node->out()->isScalar()) {
        // Single line: `out = lhs op rhs;`
        code_ << " = "
              << genBinaryOp(op_type, gen(node->lhs()), gen(node->rhs()));
      } else {
        // Split TensorView expressions across multiple lines:
        //
        // out
        //    =  lhs
        //    op rhs;
        //
        if (auto op = inline_op_str(op_type)) {
          code_ << "\n";
          indent() << kTab << "= " << gen(node->lhs()) << "\n";
          indent() << kTab << *op << " " << gen(node->rhs());
        } else {
          code_ << " = " << op_type << "(\n";
          indent() << kTab << gen(node->lhs()) << ",\n";
          indent() << kTab << gen(node->rhs()) << ")";
        }
      }
      code_ << ";\n";
    }
  }

  void handle(const kir::TernaryOp* node) final {
    if (!print_inline_) {
      indent() << gen(node->out());
      if (!node->out()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    code_ << node->getTernaryOpType() << "(" << gen(node->in1()) << ", "
          << gen(node->in2()) << ", " << gen(node->in3()) << ")";

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  std::string genReductionOp(BinaryOpType op_type, DataType data_type) {
    std::stringstream lambda;
    lambda << "[](" << data_type << " &a, " << data_type << " b) "
           << "{ a = " << genBinaryOp(op_type, "a", "b") << "; }";
    return lambda.str();
  }

  void handle(const kir::BroadcastOp* node) final {
    const ir_utils::ParallelTypeBitmap domains =
        ir_utils::getParallelBroadcastDomains(
            node->out(), kernel_->predicateMap());

    const bool thread_x = domains.get(ParallelType::TIDx);
    const bool thread_y = domains.get(ParallelType::TIDy);
    const bool thread_z = domains.get(ParallelType::TIDz);
    const bool block_x = domains.get(ParallelType::BIDx);
    const bool block_y = domains.get(ParallelType::BIDy);
    const bool block_z = domains.get(ParallelType::BIDz);

    const bool grid_broadcast_needed = block_x || block_y || block_z;
    const bool block_broadcast_needed = thread_x || thread_y || thread_z;

    TORCH_INTERNAL_ASSERT(
        !grid_broadcast_needed,
        "Parallel broadcast across blocks not supported");

    if (block_broadcast_needed) {
      const auto data_type = node->out()->getDataType().value();
      indent() << "broadcast::blockBroadcast<" << (thread_x ? "true" : "false")
               << ", " << (thread_y ? "true" : "false") << ", "
               << (thread_z ? "true" : "false") << ">(\n";
      indent() << kTab << gen(node->out()) << ",\n";
      indent() << kTab << gen(node->in()) << ",\n";
      indent() << kTab << "static_cast<" << data_type << "*>(shared_mem));\n";
    } else {
      indent() << gen(node->out()) << "\n";
      indent() << kTab << " = " << gen(node->in()) << ";\n";
    }
  }

  void handle(const kir::ReductionOp* node) final {
    TORCH_CHECK(node->out()->getValType() == ValType::TensorIndex);

    const auto out = node->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    if (!has_block_reduce && !has_grid_reduce) {
      const auto gen_out = gen(out);
      const auto op_type = node->getReductionOpType();
      indent() << gen_out << " = "
               << genBinaryOp(op_type, gen_out, gen(node->in())) << ";\n";
      return;
    }

    const auto par_domains = node->getParallelReductionDomains();
    const bool tidx = par_domains.find(ParallelType::TIDx) != par_domains.end();
    const bool tidy = par_domains.find(ParallelType::TIDy) != par_domains.end();
    const bool tidz = par_domains.find(ParallelType::TIDz) != par_domains.end();

    const auto data_type = node->out()->getDataType().value();
    const auto op_type = node->getReductionOpType();

    if (has_block_reduce) {
      if (has_grid_reduce) {
        indent() << data_type << " "
                 << "block_result"
                 << ";\n";
      }
      indent() << "blockReduce<" << (tidx ? "true" : "false") << ", "
               << (tidy ? "true" : "false") << ", " << (tidz ? "true" : "false")
               << ">(\n";
      if (has_grid_reduce) {
        indent() << kTab << "block_result"
                 << ",\n";
      } else {
        indent() << kTab << gen(node->out()) << ",\n";
      }
      indent() << kTab << gen(node->in()) << ",\n";
      indent() << kTab << genReductionOp(op_type, data_type) << ",\n";
      indent() << kTab << "threadIdx,\n";
      indent() << kTab << "blockDim,\n";
      indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
      if (node->pred() == nullptr) {
        indent() << kTab << "true,\n";
      } else {
        indent() << kTab << genInline(node->pred()) << ",\n";
      }
      indent() << kTab << genInline(node->init()) << ");\n";
    }
  }

  void handle(const kir::GridReduction* node) final {
    const auto rop = node->reduction_op();
    TORCH_INTERNAL_ASSERT(rop->out()->getValType() == ValType::TensorIndex);

    const auto out = rop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    const auto par_domains = rop->getParallelReductionDomains();
    const bool tidx = par_domains.find(ParallelType::TIDx) != par_domains.end();
    const bool tidy = par_domains.find(ParallelType::TIDy) != par_domains.end();
    const bool tidz = par_domains.find(ParallelType::TIDz) != par_domains.end();
    const bool bidx = par_domains.find(ParallelType::BIDx) != par_domains.end();
    const bool bidy = par_domains.find(ParallelType::BIDy) != par_domains.end();
    const bool bidz = par_domains.find(ParallelType::BIDz) != par_domains.end();

    const auto data_type = rop->out()->getDataType().value();
    const auto op_type = rop->getReductionOpType();

    TORCH_INTERNAL_ASSERT(
        node->reduction_buffer()->buffer()->getValType().value() ==
        ValType::KirTensorView);
    TORCH_INTERNAL_ASSERT(
        node->sync_buffer()->buffer()->getValType().value() ==
        ValType::KirTensorView);
    const auto work_buffer =
        node->reduction_buffer()->buffer()->as<kir::TensorView>();
    const auto sync_buffer =
        node->sync_buffer()->buffer()->as<kir::TensorView>();

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid reduction.
    indent() << kir::GridReduction::getPredicateFlagName(out->view()) << " = "
             << "reduction::gridReduce<" << (bidx ? "true" : "false") << ", "
             << (bidy ? "true" : "false") << ", " << (bidz ? "true" : "false")
             << ", " << (!tidx ? "true" : "false") << ", "
             << (!tidy ? "true" : "false") << ", " << (!tidz ? "true" : "false")
             << ">(\n";
    indent() << kTab << gen(rop->out()) << ",\n";
    if (domain->hasBlockReduction()) {
      indent() << kTab << "block_result"
               << ",\n";
    } else {
      indent() << kTab << gen(rop->in()) << ",\n";
    }
    indent() << kTab << genReductionOp(op_type, data_type) << ",\n";
    indent() << kTab << "&" << gen(work_buffer) << "[0],\n";
    indent() << kTab << gen(sync_buffer) << ",\n";
    indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
    if (node->pred() == nullptr) {
      indent() << kTab << "true,\n";
    } else {
      indent() << kTab << genInline(node->pred()) << ",\n";
    }
    indent() << kTab << genInline(node->reduction_op()->init()) << ");\n";
  }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
  // TODO(Kir): fix me
  void handle(const kir::Scope& scope) {
    for (auto expr : scope.exprs()) {
      handle(expr);
    }
  }
#pragma clang diagnostic pop

  void handle(const kir::ForLoop* node) final {
    // TODO(kir): handle this during lowering
    if (node->iter_domain()->isThread() || node->iter_domain()->isBroadcast()) {
      handle(node->body());
      return;
    }

    const auto gen_index = gen(node->index());
    const auto gen_start = genInline(node->iter_domain()->start());
    const auto gen_extent = genInline(node->iter_domain()->extent());
    indent() << "for(size_t " << gen_index << " = " << gen_start << "; "
             << gen_index << " < " << gen_extent << "; ++" << gen_index << ") ";

    startBlock(true);
    handle(node->body());
    endBlock();
  }

  void handle(const kir::IfThenElse* node) final {
    indent() << "if (" << genInline(node->cond()) << ") ";

    // "then" block
    startBlock(true);
    handle(node->thenBody());

    // "else" block (optional)
    if (node->hasElse()) {
      endBlock(" else ");
      startBlock(true);
      handle(node->elseBody());
    }

    endBlock();
  }

  // TODO(kir): fold initialization into Allocate
  void handle(const kir::Allocate* node) final {
    if (node->buffer()->getValType().value() != ValType::KirTensorView) {
      indent() << node->buffer_type() << " " << gen(node->buffer()) << ";\n";
      return;
    }

    const auto tv = node->buffer()->as<kir::TensorView>();
    TORCH_INTERNAL_ASSERT(tv->domain()->nDims() > 0);
    TORCH_INTERNAL_ASSERT(node->size() != nullptr);

    if (node->alias() != nullptr) {
      // Allocate alias another Allocate node
      const auto alias_tv = node->alias()->buffer()->as<kir::TensorView>();
      indent() << "// Alias Allocation - " << node->getMemoryType() << "\n";
      indent() << node->buffer_type() << "* " << gen(tv) << " = "
               << gen(alias_tv) << ";\n";
    } else {
      // Standard Memory Allocation
      switch (tv->memoryType()) {
        case MemoryType::Global:
          indent() << "// Allocate global tensor " << gen(tv) << "\n";
          break;
        case MemoryType::Shared:
          if (node->size()->isConstScalar()) {
            // Static shared memory
            indent() << "__shared__ " << node->buffer_type() << " " << gen(tv)
                     << "[" << genInline(node->size()) << "];\n";
          } else {
            // Align Offset Position
            indent() << "offset = alignBufferSize(offset,"
                     << dataTypeSize(node->buffer_type()) << ");\n";
            // Shared Memory Pointer
            indent() << node->buffer_type() << "* " << gen(tv)
                     << " = reinterpret_cast<" << node->buffer_type() << "*>"
                     << "(array + offset);\n";
            // Increment Offset Position
            indent() << "offset += (" << genInline(node->size()) << " * sizeof("
                     << node->buffer_type() << "));\n";
          }
          break;
        case MemoryType::Local:
          indent() << node->buffer_type() << " " << gen(tv) << "["
                   << genInline(node->size()) << "];\n";
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unexpected memory type");
      }
    }
  }

  void handle(const kir::Sync* node) final {
    indent() << "__syncthreads();\n";
  }

 private:
  std::stringstream code_;
  const Kernel* kernel_;
  int block_nest_level_ = 0;

  // TODO(kir): replace with explicit assignment statements
  bool print_inline_ = false;
};

} // namespace

std::string generateCudaKernel(
    const Kernel* kernel,
    const std::string& kernel_name) {
  FUSER_PERF_SCOPE("generateCudaKernel");
  return CudaKernelGenerator::generateKernelDefinition(kernel, kernel_name);
}

} // namespace codegen
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
