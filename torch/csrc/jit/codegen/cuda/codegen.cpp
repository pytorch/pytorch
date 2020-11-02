#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <array>
#include <sstream>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace codegen {

namespace {

class CudaKernelGenerator : private kir::IrVisitor {
  static constexpr const char* kTab = "  ";

 public:
  static std::string generateKernelDefinition(
      const kir::Kernel* kernel,
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
  explicit CudaKernelGenerator(const kir::Kernel* kernel) : kernel_(kernel) {}

  // Generates the kernel function declaration
  void genDeclaration(const std::string& kernel_name) {
    const auto& kernel_summary = kernel_->summary();

    code_ << "__global__ void " << kernel_name << "(";

    std::vector<kir::Val*> params;

    // Inputs & Outputs
    for (auto val : kernel_->inputs()) {
      params.push_back(val);
    }
    for (auto val : kernel_->outputs()) {
      params.push_back(val);
    }

    // Generate parameter declarations
    for (kir::Val* val : params) {
      if (const auto tv = dynamic_cast<kir::TensorView*>(val)) {
        code_ << "Tensor<" << val->dtype() << ", "
              << TensorDomain::noReductions(
                     tv->fuserTv()->getMaybeRFactorDomain())
                     .size()
              << "> " << varName(tv, "T");
      } else {
        TORCH_INTERNAL_ASSERT(val->isScalar());
        code_ << val->dtype() << " " << gen(val);
      }

      if (val != params.back()) {
        code_ << ", ";
      }
    }

    // Global buffers
    for (auto allocate : kernel_summary.global_allocations) {
      TORCH_INTERNAL_ASSERT(allocate->buffer()->isA<kir::TensorView>());
      const auto tv = allocate->buffer()->as<kir::TensorView>();
      code_ << ", Tensor<" << tv->dtype() << ", "
            << tv->domain()->rootDomain().size() << "> " << varName(tv, "T");
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
      expr->accept(this);
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

  std::string gen(const kir::Node* node) {
    std::stringstream tmp_code;
    std::swap(tmp_code, code_);
    node->accept(this);
    std::swap(tmp_code, code_);
    return tmp_code.str();
  }

  // TODO(kir): consider automatic var naming
  std::string varName(const kir::Val* val, const char* prefix) {
    std::stringstream value_name;
    if (val->name() != kInvalidStmName) {
      value_name << prefix << val->name();
    } else {
      value_name << "k" << prefix << val->id();
    }
    return value_name.str();
  }

  std::string genInline(const kir::Node* node) {
    const bool saved_inline = print_inline_;
    print_inline_ = true;
    const auto result = gen(node);
    print_inline_ = saved_inline;
    return result;
  }

  void visit(const kir::Bool* node) final {
    const auto def = node->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isConst()) {
      code_ << *node->value();
    } else {
      code_ << varName(node, "b");
    }
  }

  void visit(const kir::Float* node) final {
    const auto def = node->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isConst()) {
      const int digits = std::numeric_limits<Float::ScalarType>::max_digits10;
      code_ << "float(" << std::setprecision(digits) << *node->value() << ")";
    } else {
      code_ << varName(node, "f");
    }
  }

  void visit(const kir::Half* node) final {
    const auto def = node->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isConst()) {
      code_ << "__float2half(" << *node->value() << ")";
    } else {
      code_ << varName(node, "h");
    }
  }

  void visit(const kir::Int* node) final {
    const auto def = node->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isConst()) {
      code_ << *node->value();
    } else {
      code_ << varName(node, "i");
    }
  }

  void visit(const kir::NamedScalar* node) final {
    code_ << node->name();
  }

  void visit(const kir::TensorIndex* node) final {
    code_ << varName(node->view(), "T") << "[";

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

  void visit(const kir::IterDomain* node) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void visit(const kir::TensorDomain* node) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void visit(const kir::TensorView* tv) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void visit(const kir::UnaryOp* node) final {
    if (!print_inline_) {
      indent() << gen(node->out());
      if (!node->out()->isScalar() && !node->in()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    if (auto op = inline_op_str(node->operation())) {
      code_ << *op << gen(node->in());
    } else {
      if (node->operation() == UnaryOpType::Cast) {
        const auto cast_str =
            cast_func_str({node->in()->dtype(), node->out()->dtype()});
        code_ << cast_str.value();
      } else {
        code_ << node->operation();
      }

      code_ << "(";
      if (node->operation() == UnaryOpType::RandLike) {
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

  void visit(const kir::BinaryOp* node) final {
    const auto op_type = node->operation();
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

  void visit(const kir::TernaryOp* node) final {
    if (!print_inline_) {
      indent() << gen(node->out());
      if (!node->out()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    code_ << node->operation() << "(" << gen(node->in1()) << ", "
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

  void visit(const kir::BroadcastOp* node) final {
    TORCH_INTERNAL_ASSERT(node->out()->isA<kir::TensorIndex>());
    const auto tensor_index = node->out()->as<kir::TensorIndex>();

    const ParallelTypeBitmap domains = ir_utils::getParallelBroadcastDomains(
        tensor_index->view()->fuserTv(), kernel_->predicateMap());

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
      const auto data_type = node->out()->dtype();
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

  void visit(const kir::ReductionOp* node) final {
    TORCH_INTERNAL_ASSERT(node->out()->isA<kir::TensorIndex>());

    const auto out = node->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    if (!has_block_reduce && !has_grid_reduce) {
      const auto gen_out = gen(out);
      const auto op_type = node->operation();
      indent() << gen_out << " = "
               << genBinaryOp(op_type, gen_out, gen(node->in())) << ";\n";
      return;
    }

    const auto par_domains = node->getParallelReductionDomains();
    const bool tidx = par_domains.find(ParallelType::TIDx) != par_domains.end();
    const bool tidy = par_domains.find(ParallelType::TIDy) != par_domains.end();
    const bool tidz = par_domains.find(ParallelType::TIDz) != par_domains.end();

    const auto data_type = node->out()->dtype();
    const auto op_type = node->operation();

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
      if (node->predicate() == nullptr) {
        indent() << kTab << "true,\n";
      } else {
        indent() << kTab << genInline(node->predicate()) << ",\n";
      }
      indent() << kTab << genInline(node->init()) << ");\n";
    }
  }

  std::string generateGridReduceTemplateFlags(
      const kir::ReductionOp* rop,
      const ParallelTypeBitmap& thread_pred) {
    const auto par_domains = rop->getParallelReductionDomains();
    const std::array<ParallelType, 6> ptypes{ParallelType::BIDx,
                                             ParallelType::BIDy,
                                             ParallelType::BIDz,
                                             ParallelType::TIDx,
                                             ParallelType::TIDy,
                                             ParallelType::TIDz};
    std::stringstream flags;
    for (const ParallelType pt : ptypes) {
      const bool parallel_reduction = par_domains.find(pt) != par_domains.end();
      const bool pred = thread_pred.get(pt);
      TORCH_INTERNAL_ASSERT(
          !(parallel_reduction && pred), "Cannot reduce predicated axis: ", pt);
      bool flag = false;
      // Currently assumed that no dimensions parallelized with blocks
      // are predicated. This assumption may be lifted, but
      // gridReduction would need some changes.
      if (isParallelTypeBlockDim(pt)) {
        TORCH_INTERNAL_ASSERT(
            !pred, "Predication on block dimensions not allowed: ", pt);
        flag = parallel_reduction;
      } else {
        flag = !pred && !parallel_reduction;
      }
      if (pt != ptypes[0]) {
        flags << ", ";
      }
      flags << (flag ? "true" : "false");
    }
    return flags.str();
  }

  void visit(const kir::GridReduction* node) final {
    const auto rop = node->reduction_op();
    TORCH_INTERNAL_ASSERT(rop->out()->isA<kir::TensorIndex>());

    const auto out = rop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    const auto data_type = rop->out()->dtype();
    const auto op_type = rop->operation();

    TORCH_INTERNAL_ASSERT(
        node->reduction_buffer()->buffer()->isA<kir::TensorView>());
    TORCH_INTERNAL_ASSERT(
        node->sync_buffer()->buffer()->isA<kir::TensorView>());
    const auto work_buffer =
        node->reduction_buffer()->buffer()->as<kir::TensorView>();
    const auto sync_buffer =
        node->sync_buffer()->buffer()->as<kir::TensorView>();

    const std::string flags_str =
        generateGridReduceTemplateFlags(rop, node->threadPredicate());

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid reduction.
    indent() << kir::GridReduction::getPredicateFlagName(out->view()) << " = "
             << "reduction::gridReduce<" << flags_str << ">(\n";
    indent() << kTab << gen(rop->out()) << ",\n";
    if (domain->hasBlockReduction()) {
      indent() << kTab << "block_result"
               << ",\n";
    } else {
      indent() << kTab << gen(rop->in()) << ",\n";
    }
    indent() << kTab << genReductionOp(op_type, data_type) << ",\n";
    indent() << kTab << "&" << varName(work_buffer, "T") << "[0],\n";
    indent() << kTab << varName(sync_buffer, "T") << ",\n";
    indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
    if (node->predicate() == nullptr) {
      indent() << kTab << "true,\n";
    } else {
      indent() << kTab << genInline(node->predicate()) << ",\n";
    }
    indent() << kTab << genInline(node->reduction_op()->init()) << ");\n";
  }

  void handleScope(const kir::Scope& scope) {
    for (auto expr : scope.exprs()) {
      expr->accept(this);
    }
  }

  void visit(const kir::ForLoop* node) final {
    // TODO(kir): handle this during lowering
    if (node->iter_domain()->isThread() || node->iter_domain()->isBroadcast()) {
      handleScope(node->body());
      return;
    }

    const auto gen_index = gen(node->index());
    const auto gen_start = genInline(node->iter_domain()->start());
    const auto gen_extent = genInline(node->iter_domain()->extent());
    indent() << "for(size_t " << gen_index << " = " << gen_start << "; "
             << gen_index << " < " << gen_extent << "; ++" << gen_index << ") ";

    startBlock(true);
    handleScope(node->body());
    endBlock();
  }

  void visit(const kir::IfThenElse* node) final {
    indent() << "if (" << genInline(node->cond()) << ") ";

    // "then" block
    startBlock(true);
    handleScope(node->thenBody());

    // "else" block (optional)
    if (node->hasElse()) {
      endBlock(" else ");
      startBlock(true);
      handleScope(node->elseBody());
    }

    endBlock();
  }

  // TODO(kir): fold initialization into Allocate
  void visit(const kir::Allocate* node) final {
    const auto buffer_dtype = node->buffer()->dtype();

    if (!node->buffer()->isA<kir::TensorView>()) {
      indent() << buffer_dtype << " " << gen(node->buffer()) << ";\n";
      return;
    }

    const auto tv = node->buffer()->as<kir::TensorView>();
    TORCH_INTERNAL_ASSERT(tv->domain()->nDims() > 0);

    const auto size = node->size();
    TORCH_INTERNAL_ASSERT(size != nullptr);

    if (node->alias() != nullptr) {
      // Allocate alias another Allocate node
      const auto alias_tv = node->alias()->buffer()->as<kir::TensorView>();
      indent() << "// Alias Allocation - " << node->memoryType() << "\n";
      indent() << buffer_dtype << "* " << varName(tv, "T") << " = "
               << varName(alias_tv, "T") << ";\n";
    } else {
      // Standard Memory Allocation
      switch (tv->memoryType()) {
        case MemoryType::Global:
          indent() << "// Allocate global tensor " << varName(tv, "T") << "\n";
          break;
        case MemoryType::Shared:
          if (kir::ExpressionEvaluator::isConst(size)) {
            // Static shared memory
            indent() << "__shared__ " << buffer_dtype << " " << varName(tv, "T")
                     << "[" << genInline(size) << "];\n";
          } else {
            // Align Offset Position
            indent() << "offset = alignBufferSize(offset,"
                     << dataTypeSize(buffer_dtype) << ");\n";
            // Shared Memory Pointer
            indent() << buffer_dtype << "* " << varName(tv, "T")
                     << " = reinterpret_cast<" << buffer_dtype << "*>"
                     << "(array + offset);\n";
            // Increment Offset Position
            indent() << "offset += (" << genInline(size) << " * sizeof("
                     << buffer_dtype << "));\n";
          }
          break;
        case MemoryType::Local:
          indent() << buffer_dtype << " " << varName(tv, "T") << "["
                   << genInline(size) << "];\n";
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unexpected memory type");
      }
    }
  }

  void visit(const kir::Sync* node) final {
    indent() << "__syncthreads();\n";
  }

 private:
  std::stringstream code_;
  const kir::Kernel* kernel_;
  int block_nest_level_ = 0;

  // TODO(kir): replace with explicit assignment statements
  bool print_inline_ = false;
};

} // namespace

std::string generateCudaKernel(
    const kir::Kernel* kernel,
    const std::string& kernel_name) {
  FUSER_PERF_SCOPE("generateCudaKernel");
  return CudaKernelGenerator::generateKernelDefinition(kernel, kernel_name);
}

} // namespace codegen
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
