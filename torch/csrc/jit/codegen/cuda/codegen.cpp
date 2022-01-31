#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <array>
#include <cmath>
#include <sstream>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace codegen {

namespace {

class CudaKernelGenerator : private OptOutConstDispatch {
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

    std::vector<Val*> params;

    // Inputs & Outputs
    for (auto val : kernel_->inputs()) {
      params.push_back(val);
    }
    for (auto val : kernel_->outputs()) {
      params.push_back(val);
    }

    // Generate parameter declarations
    for (Val* val : params) {
      if (const auto tv = dynamic_cast<TensorView*>(val)) {
        if (tv->isCpuScalar()) {
          code_ << " CpuScalarTensor<" << val->dtype() << "> " << varName(tv);
        } else {
          code_
              << "Tensor<" << val->dtype() << ", "
              << TensorDomain::noReductions(tv->getMaybeRFactorDomain()).size()
              << "> " << varName(tv);
        }
      } else {
        TORCH_INTERNAL_ASSERT(val->isScalar()); // NOLINT (LLVM bug 48525)
        TORCH_INTERNAL_ASSERT(val->definition() == nullptr);
        code_ << val->dtype() << " " << gen(val);
      }

      if (val != params.back()) {
        code_ << ", ";
      }
    }

    // Global buffers
    for (auto allocate : kernel_summary.global_allocations) {
      TORCH_INTERNAL_ASSERT(allocate->buffer()->isA<TensorView>());
      const auto tv = allocate->buffer()->as<TensorView>();
      const auto& maybe_rfactor_domain = tv->domain()->hasRFactor()
          ? tv->domain()->getRFactorDomain()
          : tv->domain()->getRootDomain();
      const auto nDims = std::count_if(
          maybe_rfactor_domain.begin(),
          maybe_rfactor_domain.end(),
          [](const IterDomain* id) {
            return !id->isReduction() &&
                id->getIterType() != IterType::BroadcastWithoutStride;
          });
      code_ << ", Tensor<" << tv->dtype() << ", " << nDims << "> "
            << varName(tv);
    }

    // Kernels generating random numbers take extra (seed, offset) arguments
    if (kernel_summary.is_stochastic) {
      code_ << ", at::PhiloxCudaState philox_args";
    }

    code_ << ") ";
  }

  // Generates setup code which is executed before the kernel body
  void genPrologue() {
    const auto& kernel_summary = kernel_->summary();

    // Random number generator (optional)
    if (kernel_summary.is_stochastic) {
      indent()
          << "const auto idx = ((((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;";
      indent() << "auto offset = philox_args.captured_ ?\n";
      indent()
          << "  static_cast<uint64_t>(*(philox_args.offset_.ptr) + philox_args.offset_intragraph_) :\n";
      indent() << "  philox_args.offset_.val;\n";
      indent() << "Philox rnd(philox_args.seed_, idx, offset);\n";
    }

    // Do we have any dynamic shared memory buffers?
    const bool has_dynamic_smem =
        !kernel_summary.dynamic_smem_allocations.empty();

    // Do we have any reductions?
    const bool has_reductions = kernel_summary.has_block_reductions ||
        kernel_summary.has_grid_reductions;
    const bool has_parallel_welford =
        kernel_summary.has_block_welford || kernel_summary.has_grid_welford;

    // Shared memory
    if (has_dynamic_smem || has_reductions || has_parallel_welford) {
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

      if (has_reductions || has_parallel_welford) {
        indent() << "void* shared_mem = array;\n";
        if (has_dynamic_smem) {
          if (has_parallel_welford) {
            indent() << "offset += "
                     << "((blockDim.x * blockDim.y * blockDim.z) * 3 * sizeof("
                     << kernel_summary.largest_smem_data_type << "));\n";
          } else {
            indent() << "offset += "
                     << "((blockDim.x * blockDim.y * blockDim.z) * sizeof("
                     << kernel_summary.largest_smem_data_type << "));\n";
          }
        }

        if (has_parallel_welford) {
          // Unpack shared mem pointer
          auto space_type = kernel_summary.largest_smem_data_type;
          indent()
              << "nvfuser_index_t block_size = blockDim.x*blockDim.y*blockDim.z;\n";
          indent() << space_type << " *shared_mem_var = "
                   << "static_cast<" << space_type << "*>("
                   << "shared_mem);\n";
          indent() << space_type
                   << " *shared_mem_avg = shared_mem_var + block_size;\n";
          indent() << space_type
                   << " *shared_mem_n = shared_mem_avg + block_size;\n";
        }
      }
    }

    // Call the initialization function if using a custom block sync
    if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
      indent() << "block_sync::init();\n";
    }
  }

  void genBody() {
    for (auto expr : kernel_->topLevelExprs()) {
      OptOutConstDispatch::handle(expr);
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
    for (const auto i : c10::irange(block_nest_level_)) {
      (void)i; // Suppress unused variable warning
      code_ << kTab;
    }
    return code_;
  }

  std::string gen(const Statement* stmt) {
    std::stringstream tmp_code;
    std::swap(tmp_code, code_);
    auto replacement = replacement_map_.find(stmt);
    if (replacement != replacement_map_.end()) {
      stmt = replacement->second;
    }
    OptOutConstDispatch::handle(stmt);
    std::swap(tmp_code, code_);
    return tmp_code.str();
  }

  std::string varName(const Val* val) {
    std::stringstream name;
    if (val->isA<TensorView>()) {
      name << "T";
    } else {
      name << typePrefix(val->dtype());
    }
    name << val->name();
    return name.str();
  }

  std::string genInline(const Statement* stmt) {
    const bool saved_inline = print_inline_;
    print_inline_ = true;
    auto result = gen(stmt);
    print_inline_ = saved_inline;
    // NOLINTNEXTLINE(performance-no-automatic-move)
    return result;
  }

  void handle(const kir::Predicate* pred) final {
    TORCH_INTERNAL_ASSERT(pred->hasValue());
    code_ << gen(pred->value());
  }

  void handle(const Bool* pred) final {
    const auto def = pred->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (pred->isConst()) {
      code_ << (*pred->value() ? "true" : "false");
    } else {
      code_ << varName(pred);
    }
  }

  void handle(const Double* d) final {
    const auto def = d->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (d->isConst()) {
      const int digits = std::numeric_limits<Double::ScalarType>::max_digits10;
      code_ << std::setprecision(digits) << *d->value();
    } else {
      code_ << varName(d);
    }
  }

  void handle(const Int* i) final {
    const auto def = i->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (i->isConst()) {
      code_ << *i->value();
    } else {
      code_ << varName(i);
    }
  }

  void handle(const NamedScalar* ns) final {
    // dim3 components are unsigned int. Cast to signed integer to
    // support negative indexing
    if (ns->getParallelIndex().has_value() ||
        ns->getParallelDim().has_value()) {
      code_ << "((nvfuser_index_t)" << ns->name() << ")";
    } else {
      code_ << ns->name();
    }
  }

  void handle(const kir::TensorIndex* ti) final {
    code_ << varName(ti->view()) << "[";

    bool first = true;
    for (auto* ind : ti->indices()) {
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

  void handle(const IterDomain*) final {
    TORCH_INTERNAL_ASSERT(false, "Unreachable");
  }

  void handle(const TensorDomain*) final {
    TORCH_INTERNAL_ASSERT(false, "Unreachable");
  }

  void handle(const TensorView*) final {
    TORCH_INTERNAL_ASSERT(false, "Unreachable");
  }

  void handle(const UnaryOp* uop) final {
    bool is_vector_op = false;
    size_t vector_word_size = 1;

    if (vectorize_scope_ && uop->out()->isA<kir::TensorIndex>()) {
      auto ti = uop->out()->as<kir::TensorIndex>();

      bool vectorize_op = false;
      bool misaligned_op = false;

      for (auto id : ti->view()->domain()->domain()) {
        if (!isParallelTypeVectorize(id->getParallelType())) {
          continue;
        }

        ExpressionEvaluator expr_eval(id->fusion());
        auto vector_size_optional = expr_eval.evaluate(id->extent());

        TORCH_INTERNAL_ASSERT(
            vector_size_optional.has_value(),
            "Could not evaluate constant value bound to vectorized dim.");

        vector_word_size = vector_size_optional.value();

        vectorize_op = id->getParallelType() == ParallelType::Vectorize;
        misaligned_op =
            id->getParallelType() == ParallelType::MisalignedVectorize;
        break;
      }

      if (vectorize_op) {
        TORCH_INTERNAL_ASSERT(
            uop->getUnaryOpType() == UnaryOpType::Set,
            "Cannot vectorize operations that are not sets. ",
            "Use cache_before and cache_after to store/load with vectorized reads into buffers.");
        is_vector_op = true;
      }

      if (misaligned_op) {
        is_vector_op = (uop->getUnaryOpType() == UnaryOpType::Set);
      }

      if (is_vector_op && !uop->in()->isScalar()) {
        TORCH_INTERNAL_ASSERT(
            uop->out()->dtype() == uop->in()->dtype(),
            "Vectorized store/load requires input and output datatypes match.");
      }
    }

    if (is_vector_op) {
      if (uop->in()->isScalar()) {
        indent() << "reinterpret_cast<"
                 << "Array<" << uop->out()->dtype() << ", " << vector_word_size
                 << ">*>"
                 << "(&" << gen(uop->out()) << ")->set(" << gen(uop->in())
                 << ");\n";
      } else {
        indent() << "*reinterpret_cast<"
                 << "Array<" << uop->out()->dtype() << ", " << vector_word_size
                 << ">*>"
                 << "(&" << gen(uop->out()) << ")"
                 << " = *reinterpret_cast<"
                 << "Array<" << uop->in()->dtype() << ", " << vector_word_size
                 << ">*>"
                 << "(&" << gen(uop->in()) << ");\n";
      }
      return;
    }

    if (uop->out()->isA<NamedScalar>()) {
      const auto op_type = uop->getUnaryOpType();
      if (auto op = inline_op_str(op_type)) {
        indent() << gen(uop->out()) << " = " << *op << genInline(uop->in())
                 << ";\n";
      }
      return;
    }

    if (!print_inline_) {
      indent() << gen(uop->out());
      if (!uop->out()->isScalar() && !uop->in()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    const auto op_type = uop->getUnaryOpType();
    if (auto op = inline_op_str(op_type)) {
      if (alsoBooleanOperator(op_type) &&
          uop->out()->dtype() == DataType::Bool) {
        code_ << stringifyBooleanOp(op_type) << gen(uop->in());
      } else {
        code_ << *op << gen(uop->in());
      }
    } else {
      if (op_type == UnaryOpType::Cast) {
        const auto cast_str =
            cast_func_str({uop->in()->dtype(), uop->out()->dtype()});
        TORCH_INTERNAL_ASSERT(
            cast_str.has_value(),
            "Invalid cast. Input type: ",
            uop->in()->dtype(),
            ", output type: ",
            uop->out()->dtype());
        code_ << cast_str.value();
      } else {
        code_ << op_type;
        if (needFloatSuffix(op_type) &&
            uop->out()->dtype() == DataType::Float) {
          code_ << "f";
        }
      }

      code_ << "(";
      if (op_type == UnaryOpType::RandLike) {
        code_ << "rnd";
      } else {
        code_ << gen(uop->in());
      }
      code_ << ")";
    }

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  std::string genBinaryOp(
      BinaryOpType op_type,
      Val* out,
      const std::string& lhs,
      const std::string& rhs) {
    std::stringstream expr;
    if (auto op = inline_op_str(op_type)) {
      expr << lhs << " ";
      if (alsoBooleanOperator(op_type) && out->dtype() == DataType::Bool) {
        expr << stringifyBooleanOp(op_type);
      } else {
        expr << *op;
      }
      expr << " " << rhs;
    } else {
      if (integer_op_str(op_type) && isIntegralType(out->dtype())) {
        auto int_op = integer_op_str(op_type);
        expr << *int_op;
      } else {
        expr << op_type;
        if (needFloatSuffix(op_type) && out->dtype() == DataType::Float) {
          expr << "f";
        }
      }
      expr << "(" << lhs << ", " << rhs << ")";
    }
    return expr.str();
  }

  // If one argument is a tensorview and the other is a scalar, make sure we
  // cast the scalar to the tensorview type
  std::string scalarCast(Val* lhs, Val* rhs) {
    // If neither are scalars return
    if (!((lhs->isScalar() || rhs->isScalar()) &&
          (lhs->isA<kir::TensorIndex>() || rhs->isA<kir::TensorIndex>()))) {
      return "";
    }

    // Looking for mixed tensorview scalar options where types don't match
    // but are either both floating or both int types. We should cast
    // scalar to tensorview type in these instances.
    auto lhs_t = lhs->dtype();
    auto rhs_t = rhs->dtype();

    // If same type, don't cast anything
    if (lhs_t == rhs_t) {
      return "";
    }

    // Don't do anything when dealing with bools
    if (lhs_t == DataType::Bool || rhs_t == DataType::Bool) {
      return "";
    }

    // Mixing floating and int combination
    if ((isFloatingPointType(lhs_t) != isFloatingPointType(rhs_t)) ||
        (isIntegralType(lhs_t) != isIntegralType(rhs_t))) {
      return "";
    }

    std::stringstream cast;
    cast << "(" << (lhs->isA<kir::TensorIndex>() ? lhs_t : rhs_t) << ") ";
    return cast.str();
  }

  // If possible, replace pow with mul. Return true when successful.
  bool genPowerWithMul(const BinaryOp* bop) {
    if (bop->getBinaryOpType() != BinaryOpType::Pow) {
      return false;
    }

    auto rhs = bop->rhs();
    c10::optional<double> exponent;
    if (auto val_int = dynamic_cast<Int*>(rhs)) {
      if (val_int->isConst()) {
        exponent = val_int->value().value();
      }
    } else if (auto val_float = dynamic_cast<Double*>(rhs)) {
      if (val_float->isConst()) {
        auto fp_exp = val_float->value().value();
        double int_exp = 0;
        if (std::modf(fp_exp, &int_exp) == 0) {
          exponent = int_exp;
        }
      }
    }

    if (!exponent.has_value()) {
      return false;
    }

    // Only **2 and **3 are considered
    if (!(exponent.value() == 2 || exponent.value() == 3)) {
      return false;
    }

    auto lhs = gen(bop->lhs());

    if (print_inline_) {
      code_ << lhs << " * " << lhs;
      if (exponent.value() == 3) {
        code_ << " * " << lhs;
      }
    } else {
      indent() << gen(bop->out());
      if (bop->out()->isScalar()) {
        code_ << " = " << lhs << " * " << lhs;
        if (exponent.value() == 3) {
          code_ << " * " << lhs;
        }
      } else {
        code_ << "\n";
        indent() << kTab << "= " << lhs << "\n";
        indent() << kTab << "* " << lhs;
        if (exponent.value() == 3) {
          code_ << "\n";
          indent() << kTab << "* " << lhs;
        }
      }
    }

    code_ << ";\n";
    return true;
  }

  void handle(const BinaryOp* bop) final {
    // Try replacing pow with mul
    if (genPowerWithMul(bop)) {
      return;
    }

    const auto op_type = bop->getBinaryOpType();
    if (print_inline_) {
      // Inline expression: `lhs op rhs`
      code_ << genBinaryOp(
          op_type, bop->out(), gen(bop->lhs()), gen(bop->rhs()));
    } else {
      indent() << gen(bop->out());
      if (bop->out()->isScalar()) {
        // Single line: `out = lhs op rhs;`
        code_ << " = "
              << genBinaryOp(
                     op_type, bop->out(), gen(bop->lhs()), gen(bop->rhs()));
      } else {
        // Split TensorView expressions across multiple lines:
        //
        // out
        //    =  lhs
        //    op rhs;
        //

        auto cast = scalarCast(bop->lhs(), bop->rhs());
        if (auto op = inline_op_str(op_type)) {
          code_ << "\n";
          indent() << kTab << "= " << (bop->lhs()->isScalar() ? cast : "")
                   << gen(bop->lhs()) << "\n";
          indent() << kTab;
          if (alsoBooleanOperator(op_type) &&
              bop->out()->dtype() == DataType::Bool) {
            code_ << stringifyBooleanOp(op_type);
          } else {
            code_ << *op;
          }
          code_ << " " << (bop->rhs()->isScalar() ? cast : "")
                << gen(bop->rhs());
        } else {
          if (integer_op_str(op_type) && isIntegralType(bop->out()->dtype())) {
            auto int_op = integer_op_str(op_type);
            code_ << " = " << *int_op << "(\n";
          } else {
            std::stringstream op_str;
            op_str << op_type;
            if (needFloatSuffix(op_type) &&
                bop->out()->dtype() == DataType::Float) {
              op_str << "f";
            }
            code_ << " = " << op_str.str() << "(\n";
          }
          indent() << kTab << (bop->lhs()->isScalar() ? cast : "")
                   << gen(bop->lhs()) << ",\n";
          indent() << kTab << (bop->rhs()->isScalar() ? cast : "")
                   << gen(bop->rhs()) << ")";
        }
      }
      code_ << ";\n";
    }
  }

  void handle(const TernaryOp* top) final {
    if (!print_inline_) {
      indent() << gen(top->out());
      if (!top->out()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    code_ << top->getTernaryOpType() << "(" << gen(top->in1()) << ", ";

    // Make sure the two operands of where has the same
    // type. Note that compiling "where(0.0f, 0.0)" fails because of
    // the overloading ambiguity.
    if (top->getTernaryOpType() == TernaryOpType::Where) {
      auto cast = scalarCast(top->in2(), top->in3());
      code_ << (top->in2()->isScalar() ? cast : "") << gen(top->in2()) << ", "
            << (top->in3()->isScalar() ? cast : "") << gen(top->in3()) << ")";
    } else {
      code_ << gen(top->in2()) << ", " << gen(top->in3()) << ")";
    }

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  std::string genReductionOp(BinaryOpType op_type, Val* out) {
    std::stringstream lambda;
    DataType data_type = out->dtype();
    lambda << "[](" << data_type << " &a, " << data_type << " b) "
           << "{ a = " << genBinaryOp(op_type, out, "a", "b") << "; }";
    return lambda.str();
  }

  void handle(const BroadcastOp* stmt) final {
    TORCH_INTERNAL_ASSERT(stmt->out()->isA<kir::TensorIndex>());
    const auto tensor_index = stmt->out()->as<kir::TensorIndex>();

    const ParallelTypeBitmap parallel_types =
        kernel_->summary().broadcast_parallel_types.at(stmt);

    if (parallel_types.none()) {
      // Not parallelized
      indent() << gen(stmt->out()) << "\n";
      indent() << kTab << " = " << gen(stmt->in()) << ";\n";
      return;
    }

    TORCH_INTERNAL_ASSERT(
        !parallel_types.hasBID(),
        "Parallel broadcast across blocks should have been translated to a GridBroadcast IR node");

    std::stringstream flags_str;
    for (const ParallelType pt : kParallelTypeTIDs) {
      const bool parallel_bcast = parallel_types.get(pt);
      if (pt != kParallelTypeTIDs[0]) {
        flags_str << ", ";
      }
      flags_str << (parallel_bcast ? "true" : "false");
    }

    const auto data_type = stmt->out()->dtype();
    indent() << "broadcast::blockBroadcast<" << flags_str.str() << ">(\n";
    indent() << kTab << gen(stmt->out()) << ",\n";
    indent() << kTab << gen(stmt->in()) << ",\n";
    indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
    TORCH_INTERNAL_ASSERT(
        stmt->predicate() != nullptr && stmt->predicate()->hasValue());
    indent() << kTab << genInline(stmt->predicate()) << ");\n";
  }

  void genWarpReductionOp(
      const ReductionOp* rop,
      const IterDomain* reduction_id) {
    bool is_single_warp =
        kernel_->getWarpPaddedParallelInfo().is_tidx_single_warp;

    indent() << "warp::warpReduceTIDX";
    if (is_single_warp) {
      code_ << "<true>(\n";
    } else {
      code_ << "<false>(\n";
    }
    indent() << kTab << gen(rop->out()) << ",\n";
    indent() << kTab << gen(rop->in()) << ",\n";
    indent() << kTab << genReductionOp(rop->getReductionOpType(), rop->out())
             << ",\n";
    indent() << kTab << "threadIdx,\n";
    indent() << kTab << "blockDim,\n";
    indent() << kTab << "static_cast<" << rop->out()->dtype()
             << "*>(shared_mem),\n";
    TORCH_INTERNAL_ASSERT(
        rop->predicate() != nullptr && rop->predicate()->hasValue());
    indent() << kTab << genInline(rop->predicate()) << ",\n";
    indent() << kTab << rop->out()->dtype() << "(" << genInline(rop->init())
             << "));\n";
  }

  void handle(const ReductionOp* rop) final {
    TORCH_INTERNAL_ASSERT(rop->out()->isA<kir::TensorIndex>());

    const auto out = rop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    if (!has_block_reduce && !has_grid_reduce) {
      const auto gen_out = gen(out);
      const auto op_type = rop->getReductionOpType();
      indent() << gen_out << " = "
               << genBinaryOp(op_type, out, gen_out, gen(rop->in())) << ";\n";
      return;
    }

    if (auto reduction_id = ir_utils::getMaybeWarpReductionDim(rop)) {
      genWarpReductionOp(rop, reduction_id.value());
      return;
    }

    const auto par_domains = ir_utils::getParallelDomains(rop->out());
    // Get parallel reduction domains
    const bool tidx =
        par_domains.find(ParallelType::TIDx) != par_domains.end() &&
        par_domains.at(ParallelType::TIDx)->isReduction();
    const bool tidy =
        par_domains.find(ParallelType::TIDy) != par_domains.end() &&
        par_domains.at(ParallelType::TIDy)->isReduction();
    const bool tidz =
        par_domains.find(ParallelType::TIDz) != par_domains.end() &&
        par_domains.at(ParallelType::TIDz)->isReduction();

    const auto data_type = rop->out()->dtype();
    const auto op_type = rop->getReductionOpType();

    if (has_block_reduce) {
      if (has_grid_reduce) {
        indent() << data_type << " "
                 << "block_result_" << block_reduce_name_ << "="
                 << gen(rop->init()) << ";\n";
      }
      indent() << "blockReduce<" << (tidx ? "true" : "false") << ", "
               << (tidy ? "true" : "false") << ", " << (tidz ? "true" : "false")
               << ">(\n";
      if (has_grid_reduce) {
        indent() << kTab << "block_result_" << block_reduce_name_ << ",\n";
      } else {
        indent() << kTab << gen(rop->out()) << ",\n";
      }
      indent() << kTab << gen(rop->in()) << ",\n";
      indent() << kTab << genReductionOp(op_type, rop->out()) << ",\n";
      indent() << kTab << "threadIdx,\n";
      indent() << kTab << "blockDim,\n";
      indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
      TORCH_INTERNAL_ASSERT(
          rop->predicate() != nullptr && rop->predicate()->hasValue());
      auto read_pred = genInline(rop->predicate());
      indent() << kTab << read_pred << ",\n";
      // Pass the write predicate if available and different from the
      // default predicate. The blockReduce runtime function uses the
      // default predicate for both read and write when only the
      // default one is given.
      if (rop->writePredicate() != nullptr) {
        TORCH_INTERNAL_ASSERT(rop->writePredicate()->hasValue());
        auto write_pred = genInline(rop->writePredicate());
        indent() << kTab << write_pred << ",\n";
      }
      indent() << kTab << data_type << "(" << genInline(rop->init()) << "));\n";
    }
  }

  void handle(const WelfordOp* wop) final {
    TORCH_INTERNAL_ASSERT(wop->out()->isA<kir::TensorIndex>());

    const auto out = wop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();

    const auto out_var = wop->outVar();
    const auto out_avg = wop->outAvg();
    const auto out_N = wop->outN();

    const auto in_var = wop->inVar();
    const auto in_avg = wop->inAvg();
    const auto in_N = wop->inN();

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    // Serial WelfordOp generation
    if (!has_block_reduce && !has_grid_reduce) {
      indent() << "welfordCombine ("
               << "\n";
      indent() << " " << gen(out_avg) << ",\n";
      indent() << " " << gen(out_var) << ",\n";
      indent() << " " << gen(out_N) << ",\n";
      indent() << " " << gen(in_avg) << ",\n";
      if (in_var) {
        indent() << " " << gen(in_var) << ",\n";
      } else {
        indent() << " (" << in_avg->dtype() << ") 0"
                 << ",\n";
      }
      indent() << " (" << out_N->dtype() << ")" << gen(in_N) << ");\n";
      return;
    }

    const auto par_domains = ir_utils::getParallelDomains(wop->out());
    // Get parallel reduction domains
    const bool tidx =
        par_domains.find(ParallelType::TIDx) != par_domains.end() &&
        par_domains.at(ParallelType::TIDx)->isReduction();
    const bool tidy =
        par_domains.find(ParallelType::TIDy) != par_domains.end() &&
        par_domains.at(ParallelType::TIDy)->isReduction();
    const bool tidz =
        par_domains.find(ParallelType::TIDz) != par_domains.end() &&
        par_domains.at(ParallelType::TIDz)->isReduction();

    const auto data_type = wop->out()->dtype();

    if (has_block_reduce) {
      if (has_grid_reduce) {
        // allocate block result
        indent() << data_type << " "
                 << "block_result_avg_" << block_reduce_name_ << " = "
                 << gen(wop->initAvg()) << ";\n";
        indent() << data_type << " "
                 << "block_result_var_" << block_reduce_name_ << " = "
                 << gen(wop->initVar()) << ";\n";
        indent() << DataType::Int << " "
                 << "block_result_n_" << block_reduce_name_ << " = "
                 << gen(wop->initN()) << ";\n";
      }
      indent() << "blockWelford<" << (tidx ? "true" : "false") << ", "
               << (tidy ? "true" : "false") << ", " << (tidz ? "true" : "false")
               << ">(\n";
      if (has_grid_reduce) {
        indent() << kTab << "block_result_avg_" << block_reduce_name_ << ",\n"
                 << kTab << "block_result_var_" << block_reduce_name_ << ",\n"
                 << kTab << "block_result_n_" << block_reduce_name_ << ",\n";
      } else {
        indent() << kTab << gen(wop->outAvg()) << ",\n";
        indent() << kTab << gen(wop->outVar()) << ",\n";
        indent() << kTab << gen(wop->outN()) << ",\n";
      }
      indent() << " " << gen(in_avg) << ",\n";
      if (in_var) {
        indent() << " " << gen(in_var) << ",\n";
      } else {
        indent() << " (" << in_avg->dtype() << ") 0"
                 << ",\n";
      }
      indent() << out_N->dtype() << "(" << gen(in_N) << "),\n";
      indent() << kTab << "threadIdx,\n";
      indent() << kTab << "blockDim,\n";
      indent() << kTab << "reinterpret_cast<" << data_type
               << "*>(shared_mem_avg),\n";
      indent() << kTab << "reinterpret_cast<" << data_type
               << "*>(shared_mem_var),\n";
      indent() << kTab << "reinterpret_cast<" << DataType::Int
               << "*>(shared_mem_n),\n";
      TORCH_INTERNAL_ASSERT(wop->predicate() != nullptr);
      TORCH_INTERNAL_ASSERT(
          wop->predicate() != nullptr && wop->predicate()->hasValue());
      auto read_pred = genInline(wop->predicate());
      indent() << kTab << read_pred << ",\n";
      if (wop->writePredicate() != nullptr) {
        TORCH_INTERNAL_ASSERT(wop->writePredicate()->hasValue());
        auto write_pred = genInline(wop->writePredicate());
        indent() << kTab << write_pred << ",\n";
      }
      indent() << kTab << data_type << "(0));\n";
    }
  }

  // Support ReductionOp and WelfordOp
  template <typename REDUCTION_OP>
  std::string generateGridReduceTemplateFlags(
      const REDUCTION_OP* rop,
      const ParallelTypeBitmap& thread_pred) {
    const auto par_domains = ir_utils::getParallelDomains(rop->outputs()[0]);
    std::stringstream flags;
    for (const ParallelType pt : kParallelTypeThreads) {
      const bool parallel_reduction =
          par_domains.find(pt) != par_domains.end() &&
          par_domains.at(pt)->isReduction();
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
      if (pt != kParallelTypeThreads[0]) {
        flags << ", ";
      }
      flags << (flag ? "true" : "false");
    }
    return flags.str();
  }

  void handle(const kir::GridReduction* grop) final {
    const auto rop = grop->reduction_op();
    TORCH_INTERNAL_ASSERT(rop->out()->isA<kir::TensorIndex>());

    const auto out = rop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    const auto data_type = rop->out()->dtype();
    const auto op_type = rop->getReductionOpType();

    TORCH_INTERNAL_ASSERT(
        grop->reduction_buffer()->buffer()->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(grop->sync_buffer()->buffer()->isA<TensorView>());
    const auto work_buffer =
        grop->reduction_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = grop->sync_buffer()->buffer()->as<TensorView>();

    const std::string flags_str =
        generateGridReduceTemplateFlags(rop, grop->threadPredicate());

    const bool persistent_sync =
        kernel_->summary().has_cooperative_grid_reduction;

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid reduction.
    indent() << "reduction::gridReduce<" << flags_str << ", "
             << (persistent_sync ? "true" : "false") << ">(\n";
    indent() << kTab << gen(rop->out()) << ",\n";
    if (domain->hasBlockReduction()) {
      indent() << kTab << "block_result_" << block_reduce_name_ << ",\n";
      block_reduce_name_++;
    } else {
      indent() << kTab << gen(rop->in()) << ",\n";
    }
    indent() << kTab << genReductionOp(op_type, out) << ",\n";
    indent() << kTab << "&" << varName(work_buffer) << "[0],\n";
    indent() << kTab << varName(sync_buffer) << ",\n";
    indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
    TORCH_INTERNAL_ASSERT(
        grop->predicate() != nullptr && grop->predicate()->hasValue());
    auto read_pred = genInline(grop->predicate());
    indent() << kTab << read_pred << ",\n";
    if (grop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(grop->writePredicate()->hasValue());
      auto write_pred = genInline(grop->writePredicate());
      indent() << kTab << write_pred << ",\n";
    } else {
      indent() << kTab << read_pred << ",\n";
    }
    indent() << kTab << data_type << "("
             << genInline(grop->reduction_op()->init()) << "));\n";
  }

  void handle(const kir::GridBroadcast* grop) final {
    const auto bop = grop->broadcast_op();
    TORCH_INTERNAL_ASSERT(bop->out()->isA<kir::TensorIndex>());

    const ParallelTypeBitmap parallel_types =
        kernel_->summary().broadcast_parallel_types.at(bop);

    TORCH_INTERNAL_ASSERT(
        parallel_types.hasBID(),
        "GridBroadcast needs to be used with a broadcast op that is parallelized with the BID parallel types");

    const auto out = bop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();

    const auto data_type = bop->out()->dtype();

    TORCH_INTERNAL_ASSERT(
        grop->broadcast_buffer()->buffer()->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(grop->sync_buffer()->buffer()->isA<TensorView>());
    const auto work_buffer =
        grop->broadcast_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = grop->sync_buffer()->buffer()->as<TensorView>();

    std::stringstream flags_str;
    for (const ParallelType pt : kParallelTypeThreads) {
      const bool parallel_bcast = parallel_types.get(pt);
      if (pt != kParallelTypeThreads[0]) {
        flags_str << ", ";
      }
      flags_str << (parallel_bcast ? "true" : "false");
    }

    // Since block-level broadcast has not necessarily been performed before
    // this function call, so grid broadcast may be broadcasting across both
    // the grid and the block level.
    indent() << "grid_broadcast::broadcast<" << flags_str.str() << ">(\n";
    indent() << kTab << gen(bop->out()) << ",\n";
    indent() << kTab << gen(bop->in()) << ",\n";
    indent() << kTab << "&" << varName(work_buffer) << "[0],\n";
    indent() << kTab << varName(sync_buffer) << ",\n";
    TORCH_INTERNAL_ASSERT(
        grop->predicate() != nullptr && grop->predicate()->hasValue());
    indent() << kTab << genInline(grop->predicate()) << ");\n";
  }

  void handle(const kir::GridWelford* gwop) final {
    const auto wop = gwop->welford_op();
    TORCH_INTERNAL_ASSERT(wop->outAvg()->isA<kir::TensorIndex>());

    const auto out = wop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    const auto data_type = out->dtype();

    TORCH_INTERNAL_ASSERT(gwop->var_buffer()->buffer()->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(gwop->sync_buffer()->buffer()->isA<TensorView>());

    const auto avg_buffer = gwop->avg_buffer()->buffer()->as<TensorView>();
    const auto var_buffer = gwop->var_buffer()->buffer()->as<TensorView>();
    const auto n_buffer = gwop->N_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = gwop->sync_buffer()->buffer()->as<TensorView>();

    const bool persistent_sync =
        kernel_->summary().has_cooperative_grid_reduction;

    const std::string flags_str =
        generateGridReduceTemplateFlags(wop, gwop->threadPredicate());

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid reduction.
    indent() << "welford::gridWelford<" << flags_str << ", "
             << (persistent_sync ? "true" : "false") << ">(\n";
    indent() << kTab << gen(wop->outAvg()) << ",\n"
             << kTab << gen(wop->outVar()) << ",\n"
             << kTab << gen(wop->outN()) << ",\n";
    if (domain->hasBlockReduction()) {
      indent() << kTab << "block_result_avg_" << block_reduce_name_ << ",\n"
               << kTab << "block_result_var_" << block_reduce_name_ << ",\n"
               << kTab << "block_result_n_" << block_reduce_name_ << ",\n";
      block_reduce_name_++;
    } else {
      indent() << kTab << gen(wop->inAvg()) << ",\n";
      if (wop->inVar() == nullptr) {
        indent() << kTab << "(" << data_type << ") 0,\n";
      } else {
        indent() << kTab << gen(wop->inVar()) << ",\n";
      }
      indent() << kTab << "(" << wop->outN()->dtype() << ")" << gen(wop->inN())
               << ",\n";
    }
    indent() << kTab << "&" << varName(avg_buffer) << "[0],\n";
    indent() << kTab << "&" << varName(var_buffer) << "[0],\n";
    indent() << kTab << "&" << varName(n_buffer) << "[0],\n";
    indent() << kTab << varName(sync_buffer) << ",\n";
    indent() << kTab << "reinterpret_cast<" << data_type
             << "*>(shared_mem_avg),\n";
    indent() << kTab << "reinterpret_cast<" << data_type
             << "*>(shared_mem_var),\n";
    indent() << kTab << "reinterpret_cast<" << wop->outN()->dtype()
             << "*>(shared_mem_n),\n";
    TORCH_INTERNAL_ASSERT(
        gwop->predicate() != nullptr && gwop->predicate()->hasValue());
    auto read_pred = genInline(gwop->predicate());
    indent() << kTab << read_pred << ",\n";
    if (gwop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(gwop->writePredicate()->hasValue());
      auto write_pred = genInline(gwop->writePredicate());
      indent() << kTab << write_pred << ",\n";
    } else {
      indent() << kTab << read_pred << ",\n";
    }
    // TODO : init value support or remove.
    indent() << kTab << data_type << "(0));\n";
  }

  void handleScope(const kir::Scope& scope) {
    for (auto expr : scope.exprs()) {
      OptOutConstDispatch::handle(expr);
    }
  }

  void handle(const kir::ForLoop* loop) final {
    if (loop->iter_domain()->isBroadcast()) {
      handleScope(loop->body());
      return;
    } else if (loop->vectorize()) {
      vectorize_scope_ = loop->vectorize();
      handleScope(loop->body());
      vectorize_scope_ = false;
      return;
    } else if (loop->iter_domain()->isStride()) {
      // A stride domain only executes the loop body with the loop
      // index being zero.
      indent() << "constexpr "
               << "nvfuser_index_t"
               << " " << gen(loop->index()) << " = 0;\n";
      handleScope(loop->body());
      return;
    }

    // By default, a parallelized loop would look like:
    //
    //   for (int x = threadIdx.x; x < stop; x += blockDim.x) {
    //     do_some_comp(x);
    //   }
    //
    // When stop is guaranteed to be smaller or equal to the number of
    // threads, the for-loop is not necessary. In the above case, we
    // would just generate the loop body without the for clause but
    // references to the loop index replaced by the loop start value.
    //
    // When the loop end is the same as the IterDomain extent, the
    // assumption can be safely made. This is more conservative than
    // necessary since the loop stop value just needs to be <= the
    // IterDomain extent. However, at this point, this conservative
    // analysis seems sufficient.
    if (loop->stop() == loop->iter_domain()->extent() &&
        loop->iter_domain()->isThread()) {
      // Register a replacement of references to the loop index with
      // the loop start value.
      replacement_map_.insert({loop->index(), loop->start()});
      handleScope(loop->body());
      replacement_map_.erase(loop->index());
      return;
    }

    if (loop->start()->isZeroInt() && loop->stop()->isOneInt()) {
      indent() << "constexpr "
               << "nvfuser_index_t"
               << " " << gen(loop->index()) << " = 0;\n";
      handleScope(loop->body());
      return;
    } else if (
        // Special case handling for a pattern where start == end - 1.
        loop->start()->definition() != nullptr &&
        loop->start()->definition()->isA<BinaryOp>() &&
        loop->start()->definition()->as<BinaryOp>()->getBinaryOpType() ==
            BinaryOpType::Sub &&
        loop->start()->definition()->as<BinaryOp>()->lhs() == loop->stop() &&
        loop->start()->definition()->as<BinaryOp>()->rhs()->isOneInt()) {
      indent() << "const "
               << "nvfuser_index_t"
               << " " << gen(loop->index()) << " = " << genInline(loop->start())
               << ";\n";
      handleScope(loop->body());
      return;
    }

    const auto gen_index = gen(loop->index());
    const auto gen_start = genInline(loop->start());
    const auto gen_stop = genInline(loop->stop());
    const auto gen_step = genInline(loop->step());

    std::stringstream step_code;
    if (loop->step()->isOneInt()) {
      step_code << "++" << gen_index;
    } else {
      step_code << gen_index << " += " << gen_step;
    }
    if (loop->isUnrolled()) {
      indent() << "#pragma unroll\n";
    } else {
      indent() << "#pragma unroll 1\n";
    }

    indent() << "for(nvfuser_index_t " << gen_index;
    if (loop->iter_domain()->isParallelized()) {
      code_ << " = " << gen_start << "; ";
    } else {
      // Do not start at  the start of the ID when not parallelized. Instead,
      // start at 0. Predicates will protect buffers between 0 and ID->start(),
      // however if we started at ID->start and extent == ID->start, we could
      // have a "degenerate" loop (loop with no iterations). It may not be an
      // issue to have a 0-sized loop, but all potential consequences haven't
      // been covered. One example is WAR analysis which could incorrectly think
      // a barrier inside a 0-sized loop actually provides protection.
      code_ << " = 0; ";
    }
    code_ << gen_index << " < " << gen_stop << "; " << step_code.str() << ") ";
    startBlock(true);
    handleScope(loop->body());
    endBlock();
  }

  void handle(const kir::IfThenElse* ite) final {
    auto conditional = ite->predicate()->value();
    if (conditional->isConst()) {
      // If the conditional is a constant, then the IfThenElse is not required
      if (conditional->value().value()) {
        handleScope(ite->thenBody());
      } else {
        handleScope(ite->elseBody());
      }
      return;
    }

    indent() << "if (" << genInline(conditional) << ") ";

    // "then" block
    startBlock(true);
    handleScope(ite->thenBody());

    // "else" block (optional)
    if (ite->hasElse()) {
      endBlock(" else ");
      startBlock(true);
      handleScope(ite->elseBody());
    }

    endBlock();
  }

  void handle(const kir::Allocate* alloc) final {
    const auto buffer_dtype = alloc->buffer()->dtype();

    if (!alloc->buffer()->isA<TensorView>()) {
      indent() << buffer_dtype << " " << gen(alloc->buffer()) << ";\n";
      return;
    }

    const auto tv = alloc->buffer()->as<TensorView>();

    const auto size = alloc->size();
    TORCH_INTERNAL_ASSERT(size != nullptr);

    if (alloc->alias() != nullptr) {
      // Allocate alias another Allocate stmt
      const auto alias_tv = alloc->alias()->buffer()->as<TensorView>();
      indent() << "// Alias Allocation - " << alloc->memoryType() << "\n";
      indent() << buffer_dtype << "* " << varName(tv) << " = "
               << varName(alias_tv) << ";\n";
    } else {
      // Standard Memory Allocation
      switch (tv->getMemoryType()) {
        case MemoryType::Global:
          indent() << "// Allocate global tensor " << varName(tv) << "\n";
          break;
        case MemoryType::Shared:
          if (kir::ExpressionEvaluator::isConst(size)) {
            // Static shared memory
            indent() << "__shared__ " << buffer_dtype << " " << varName(tv)
                     << "[" << genInline(size) << "];\n";
          } else {
            // Align Offset Position
            indent() << "offset = alignBufferSize(offset,"
                     << dataTypeSize(buffer_dtype) << ");\n";
            // Shared Memory Pointer
            indent() << buffer_dtype << "* " << varName(tv)
                     << " = reinterpret_cast<" << buffer_dtype << "*>"
                     << "(array + offset);\n";
            // Increment Offset Position
            indent() << "offset += (" << genInline(size) << " * sizeof("
                     << buffer_dtype << "));\n";
          }
          break;
        case MemoryType::Local:
          indent() << buffer_dtype << " " << varName(tv) << "["
                   << genInline(size) << "];\n";
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unexpected memory type");
      }
    }
  }

  void handle(const kir::Sync*) final {
    // Use a custom synchronization method if enabled
    if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
      indent() << "block_sync::sync();\n";
    } else {
      indent() << "__barrier_sync(0);\n";
    }
  }

  void handle(const kir::InitMagicZero*) final {
    indent() << "NVFUSER_DEFINE_MAGIC_ZERO\n";
  }

  void handle(const kir::UpdateMagicZero*) final {
    indent() << "NVFUSER_UPDATE_MAGIC_ZERO\n";
  }

 private:
  std::stringstream code_;
  const kir::Kernel* kernel_;
  int block_nest_level_ = 0;
  int block_reduce_name_ = 0;
  bool print_inline_ = false;

  // Mark when we are inside of a vectorized for-loop
  bool vectorize_scope_ = false;

  //! Holds active replacement mappings during codegen
  std::unordered_map<const Statement*, const Statement*> replacement_map_;
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
