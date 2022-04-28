#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/mma_utils.h>
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

std::string ptrType(DataType dt) {
  std::stringstream ss;
  ss << dt << "*";
  return ss.str();
}

std::string refType(DataType dt) {
  std::stringstream ss;
  ss << dt << "&";
  return ss.str();
}

//! Utility class to build an argument list
class ArgumentBuilder {
 public:
  //! Build an argument list where each argument is separated with a comma
  ArgumentBuilder() = default;

  //! Build an argument list where each argument has its own line
  ArgumentBuilder(int indent_level, const char* tab) {
    std::stringstream ss;
    for (const auto i : c10::irange(indent_level)) {
      (void)i; // Suppress unused variable warning
      ss << tab;
    }
    sep_ = ",\n" + ss.str();
  }

  //! Add a new argument
  template <typename T>
  ArgumentBuilder& arg(const T& x) {
    addSeparator();
    return append(x);
  }

  //! Append to the last argument
  template <typename T>
  ArgumentBuilder& append(const T& arg) {
    ss_ << arg;
    return *this;
  }

  //! Get a string of the argument list
  std::string str() const {
    return ss_.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const ArgumentBuilder& ab) {
    return os << ab.str();
  }

 private:
  void addSeparator() {
    if (ss_.tellp() != 0) {
      ss_ << sep_;
    }
  }

 private:
  std::string sep_ = ", ";
  std::stringstream ss_;
};

//! Append to the last argument
template <>
ArgumentBuilder& ArgumentBuilder::append<bool>(const bool& arg) {
  ss_ << (arg ? "true" : "false");
  return *this;
}

//! Returns "template_name<template_arg>"
template <typename TemplateNameT, typename TemplateArgT>
std::string genTemplate(
    const TemplateNameT& template_name,
    const TemplateArgT& template_arg) {
  std::stringstream ss;
  ss << template_name << "<" << template_arg << ">";
  return ss.str();
}

//! Returns "func_name(func_arg)"
template <typename FuncNameT, typename FuncArgT>
std::string genCall(const FuncNameT& func_name, const FuncArgT& func_arg) {
  std::stringstream ss;
  ss << func_name << "(" << func_arg << ")";
  return ss.str();
}

//! Returns "func_name<template_arg>(func_arg)"
template <typename FuncNameT, typename TemplateArgT, typename FuncArgT>
std::string genCall(
    const FuncNameT& func_name,
    const TemplateArgT& template_arg,
    const FuncArgT& func_arg) {
  std::stringstream ss;
  ss << func_name << "<" << template_arg << ">(" << func_arg << ")";
  return ss.str();
}

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

    std::unordered_set<Val*> unique_args;

    std::vector<Val*> params;

    // Inputs & Outputs
    for (auto val : kernel_->inputs()) {
      params.push_back(val);
    }
    for (auto val : kernel_->outputs()) {
      TORCH_INTERNAL_ASSERT(
          !val->isScalar(), "No scalar output is allowed: ", val->toString());
      params.push_back(val);
    }

    // Generate parameter declarations
    unsigned int duplicate_counter = 0;
    for (auto i : c10::irange(params.size())) {
      std::stringstream var_name_ss;
      if (params[i]->isA<TensorView>()) {
        var_name_ss << varName(params[i]->as<TensorView>());
      } else {
        var_name_ss << gen(params[i]);
      }

      // If value is duplicate in arguments change the name to avoid name
      // conflicts in args.
      if (!unique_args.emplace(params[i]).second) {
        var_name_ss << "_duplicate_" << duplicate_counter++;
      }

      if (const auto tv = dynamic_cast<TensorView*>(params[i])) {
        if (tv->isCpuScalar()) {
          code_ << " CpuScalarTensor<" << params[i]->dtype() << "> "
                << var_name_ss.str();
        } else {
          code_
              << "Tensor<" << params[i]->dtype() << ", "
              << TensorDomain::noReductions(tv->getMaybeRFactorDomain()).size()
              << "> " << var_name_ss.str();
        }
      } else {
        TORCH_INTERNAL_ASSERT(params[i]->isScalar()); // NOLINT (LLVM bug 48525)
        TORCH_INTERNAL_ASSERT(params[i]->definition() == nullptr);
        code_ << params[i]->dtype() << " " << var_name_ss.str();
      }

      if (i + 1 != params.size()) {
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
    const bool has_alloc = alloc_map_.find(pred) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << gen(def) << ")";
    } else if (pred->isConst()) {
      code_ << (*pred->value() ? "true" : "false");
    } else {
      code_ << varName(pred);
    }
  }

  void handle(const Double* d) final {
    const auto def = d->definition();
    const bool has_alloc = alloc_map_.find(d) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << gen(def) << ")";
    } else if (d->isConst()) {
      auto val = *d->value();
      // note: default inf/nan doesn't work and should be replaced with macros
      // `NAN`, `POS_INFINITY` and `NEG_INFINITY` instead.
      if (std::isinf(val)) {
        if (val > 0) {
          code_ << "POS_INFINITY";
        } else {
          code_ << "NEG_INFINITY";
        }
      } else if (std::isnan(val)) {
        code_ << "NAN";
      } else {
        const int digits =
            std::numeric_limits<Double::ScalarType>::max_digits10;
        code_ << std::setprecision(digits) << val;
      }
    } else {
      code_ << varName(d);
    }
  }

  void handle(const Int* i) final {
    const auto def = i->definition();
    const bool has_alloc = alloc_map_.find(i) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << genInline(def) << ")";
    } else if (i->isConst()) {
      code_ << *i->value();
    } else {
      code_ << varName(i);
    }
  }

  void handle(const ComplexDouble* c) final {
    const auto def = c->definition();
    const bool has_alloc = alloc_map_.find(c) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << gen(def) << ")";
    } else if (c->isConst()) {
      const int digits = std::numeric_limits<double>::max_digits10;
      code_ << "std::complex<double>" << std::setprecision(digits)
            << *c->value();
    } else {
      code_ << varName(c);
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
    bool first = true;
    std::stringstream index;
    for (auto* ind : ti->indices()) {
      if (!ind->isZeroInt()) {
        if (!first) {
          index << " + ";
        }
        index << genInline(ind);
        first = false;
      }
    }

    if (first) {
      index << "0";
    }
    bool is_volatile = ti->view()->getMemoryType() == MemoryType::Global &&
        kernel_->summary().sync_map.needsRawSync(ti->view()).hasBID();
    if (is_volatile) {
      code_ << "*(volatile " << ti->getDataType().value() << "*)&";
    }
    code_ << varName(ti->view()) << "[" << index.str() << "]";
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

    if (uop->out()->isA<kir::TensorIndex>()) {
      auto out_tv = uop->out()->as<kir::TensorIndex>()->view();
      if (std::any_of(
              out_tv->domain()->domain().begin(),
              out_tv->domain()->domain().end(),
              [&](IterDomain* id) { return id->isMma(); })) {
        auto mma = dynamic_cast<MmaOp*>(
            uop->out()->as<kir::TensorIndex>()->view()->definition());
        TORCH_INTERNAL_ASSERT(
            mma != nullptr, "CodeGen: mma op not in mma loop");
        genMmaInitialization(mma, uop);
        return;
      }
    }

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

      if (is_vector_op) {
        auto out_tv = uop->out()->as<kir::TensorIndex>()->view();
        if (uop->in()->isScalar()) {
          // Note:
          //  Double buffered local tensors need indexed initialization,
          //   so will need to use `arraySet` option.
          if (out_tv->getMemoryType() == MemoryType::Local &&
              !out_tv->isDoubleBuffered()) {
            // Vectorized initialization
            indent() << varName(out_tv) << ".set(" << gen(uop->in()) << ");\n";
          } else {
            // Note: currently arraySet option is not vectorized, so it will
            //  rely on auto vectorization pass of cuda compiler.
            indent() << "arraySet<" << out_tv->getDataType().value() << ", "
                     << vector_word_size << ">(&" << gen(uop->out()) << ", "
                     << "(" << out_tv->getDataType().value() << ")"
                     << gen(uop->in()) << ");\n";
          }
        } else {
          // Vectorized load
          TORCH_INTERNAL_ASSERT(
              uop->in()->isA<kir::TensorIndex>(),
              "Invalid input to unary op with tensor output, found: ",
              uop->in()->toString());

          auto in_tv = uop->in()->as<kir::TensorIndex>()->view();
          bool localToGlobal = out_tv->getMemoryType() == MemoryType::Global &&
              in_tv->getMemoryType() == MemoryType::Local;

          bool globalToLocal = out_tv->getMemoryType() == MemoryType::Local &&
              in_tv->getMemoryType() == MemoryType::Global;

          bool globalToGlobal = out_tv->getMemoryType() == MemoryType::Global &&
              in_tv->getMemoryType() == MemoryType::Global;

          bool is_volatile_to = out_tv->getMemoryType() == MemoryType::Global &&
              kernel_->summary().sync_map.needsRawSync(out_tv).hasBID();

          bool is_volatile_from =
              in_tv->getMemoryType() == MemoryType::Global &&
              kernel_->summary().sync_map.needsRawSync(in_tv).hasBID();

          if (localToGlobal) {
            indent() << "loadLocalToGlobal<" << uop->out()->dtype() << ", "
                     << vector_word_size << ", "
                     << (is_volatile_to ? "true" : "false") << ">(";
            code_ << " &" << gen(uop->out()) << ", &" << gen(uop->in())
                  << ");\n";
          } else if (globalToLocal) {
            indent() << "loadGlobalToLocal<" << uop->out()->dtype() << ", "
                     << vector_word_size << ", "
                     << (is_volatile_from ? "true" : "false") << ">(&"
                     << gen(uop->out()) << ", ";
            code_ << " &" << gen(uop->in()) << ");\n";
          } else if (globalToGlobal) {
            indent() << "loadGlobalToGlobal<" << uop->out()->dtype() << ", "
                     << vector_word_size << ", "
                     << (is_volatile_to ? "true" : "false") << ", "
                     << (is_volatile_from ? "true" : "false") << ">(";
            code_ << " &" << gen(uop->out()) << ", ";
            code_ << " &" << gen(uop->in()) << ");\n";
          } else {
            indent() << "loadGeneric<" << uop->out()->dtype() << ", "
                     << vector_word_size << ">(";
            code_ << " &" << gen(uop->out()) << ", ";
            code_ << " &" << gen(uop->in()) << ");\n";
          }
        }
        return;
      }
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
      } else if (bool_op_str(op_type) && isBooleanType(out->dtype())) {
        auto bool_op = bool_op_str(op_type);
        expr << *bool_op;
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
          } else if (
              bool_op_str(op_type) && isBooleanType(bop->out()->dtype())) {
            auto bool_op = bool_op_str(op_type);
            code_ << " = " << *bool_op << "(\n";
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

  std::string genArchString(MmaOptions options) {
    std::stringstream ss;
    if (isVolta(options.macro)) {
      ss << "Volta";
    } else if (isTuring(options.macro)) {
      ss << "Turing";
    } else if (isAmpere(options.macro)) {
      ss << "Ampere";
    } else {
      TORCH_INTERNAL_ASSERT(false, "mma macro unknown arch");
    }
    return ss.str();
  }

  std::string genMmaOp(const MmaOp* mma, bool init = false) {
    std::stringstream ss;
    auto options = mma->options();
    ss << genArchString(options) << "::";
    if (init) {
      ss << "init";
    }
    ss << toString(options.macro) << toString(options.operand_layout);
    // TODO: additional parameter could be removed by swizzling iterdomain
    auto acc_stride = mma->accStride();
    TORCH_INTERNAL_ASSERT(acc_stride > 0);
    ss << "<" << acc_stride << ">";
    return ss.str();
  }

  void genMmaOperands(const MmaOp* mma) {
    std::stringstream ss;
    auto options = mma->options();
    auto in_a = mma->inA()->as<kir::TensorIndex>()->view();
    auto dtype = in_a->getDataType().value();
    indent() << kTab << "reinterpret_cast<Array<" << dtype << ","
             << getInputARegisterSize(options.macro) << ","
             << getInputARegisterSize(options.macro) << ">*>(&"
             << gen(mma->inA()) << "),\n";
    indent() << kTab << "reinterpret_cast<Array<" << dtype << ","
             << getInputBRegisterSize(options.macro) << ","
             << getInputBRegisterSize(options.macro) << ">*>(&"
             << gen(mma->inB()) << ")";
  }

  void genMmaInitialization(const MmaOp* mma, const UnaryOp* uop) {
    auto options = mma->options();

    indent() << genMmaOp(mma, true) << "(reinterpret_cast<Array<"
             << mma->out()->getDataType().value() << ","
             << getOutputRegisterSize(options.macro) << ","
             << getOutputRegisterSize(options.macro) << ">*>"
             << "(&" << gen(uop->out()) << "));\n";
  }

  void handle(const MmaOp* mma) final {
    auto options = mma->options();
    auto out = mma->out()->as<kir::TensorIndex>();
    indent() << genMmaOp(mma) << "(\n";
    indent() << kTab << "reinterpret_cast<Array<"
             << out->view()->getDataType().value() << ","
             << getOutputRegisterSize(options.macro) << ","
             << getOutputRegisterSize(options.macro) << ">*>(&"
             << gen(mma->out()) << "),\n";
    genMmaOperands(mma);
    code_ << ");\n";
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
        indent() << out_N->dtype() << " "
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
      indent() << kTab << "reinterpret_cast<" << out_N->dtype()
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
    TORCH_INTERNAL_ASSERT(
        !rop->isFused(), "This is not for the fused reduction kernel\n");

    const auto par_domains = ir_utils::getParallelDomains(rop->outputs()[0]);
    ArgumentBuilder flags;
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
      flags.arg(flag);
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

    if (rop->isFused()) {
      generateFusedGridReduction(grop);
      return;
    }

    const std::string flags_str =
        generateGridReduceTemplateFlags(rop, grop->threadPredicate());

    const bool persistent_sync =
        kernel_->summary().has_cooperative_grid_reduction;

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid
    // reduction.
    ArgumentBuilder template_args;
    template_args.arg(flags_str).arg(persistent_sync);

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    func_args.arg(gen(rop->out()));
    if (domain->hasBlockReduction()) {
      func_args.arg("block_result_").append(block_reduce_name_);
      block_reduce_name_++;
    } else {
      func_args.arg(gen(rop->in()));
    }
    func_args.arg(genReductionOp(op_type, out));
    func_args.arg("&").append(varName(work_buffer)).append("[0]");
    func_args.arg(varName(sync_buffer));
    func_args.arg(genCall("static_cast", ptrType(data_type), "shared_mem"));
    // read and write predicates
    TORCH_INTERNAL_ASSERT(
        grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    func_args.arg(read_pred);
    if (grop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(grop->writePredicate()->hasValue());
      func_args.arg(genInline(grop->writePredicate()));
    } else {
      func_args.arg(read_pred);
    }
    // Init val
    func_args.arg(genCall(data_type, genInline(grop->reduction_op()->init())));

    indent() << "reduction::gridReduce<" << template_args << ">(\n";
    indent() << kTab << func_args << ");\n";
  }

  std::string genFusedReductionName(const kir::TensorIndex* reduction_out) {
    return varName(reduction_out->view()) + "_reduction";
  }

  void generateFusedGridReduction(const kir::GridReduction* grop) {
    const auto rop = grop->reduction_op();
    TORCH_INTERNAL_ASSERT(rop->isFused());

    const auto out = rop->out()->as<kir::TensorIndex>();

    const auto data_type = rop->out()->dtype();
    const auto op_type = rop->getReductionOpType();

    const auto work_buffer =
        grop->reduction_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = grop->sync_buffer()->buffer()->as<TensorView>();

    const auto reduction_name = genFusedReductionName(out);

    // template <typename Func, typename... Types>
    // __device__ __inline__ void reduce(
    //   RefTuple<Types...> out,
    //   const LocalTuple<Types...>& inp,
    //   VolatilePtrTuple<Types...> global_work_buffer,
    //   int64_t* global_sync_buffer, // Allocated as product of all
    //                                // non-participating Grid dimension
    //   PtrTuple<Types...> shared_buf,
    //   bool read_pred, // Prevent reading from out of bounds memory
    //   bool write_pred, // Prevent from writing out of bounds
    //   const LocalTuple<Types...>& init_val,
    //   Func reduction_op);

    indent() << reduction_name << ".reduce(\n";

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    // out
    func_args.arg(genCall("RefTuple", data_type, gen(rop->out())));
    // inp
    func_args.arg(genCall("ConstRefTuple", data_type, gen(rop->in())));
    // global_work_buffer
    func_args.arg(genCall(
        "VolatilePtrTuple", data_type, "&" + varName(work_buffer) + "[0]"));
    // global_sync_buffer
    func_args.arg("&").append(varName(sync_buffer)).append("[0]");
    // shared_buf
    func_args.arg(genCall(
        "PtrTuple",
        data_type,
        genCall("static_cast", ptrType(data_type), "shared_mem")));
    // read and write predicates
    TORCH_INTERNAL_ASSERT(
        grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    auto write_pred = read_pred;
    if (grop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(grop->writePredicate()->hasValue());
      write_pred = genInline(grop->writePredicate());
    }
    func_args.arg(read_pred).arg(write_pred);
    // init_val
    func_args.arg(genCall(
        "LocalTuple", data_type, genInline(grop->reduction_op()->init())));
    // reduction_op
    func_args.arg(genReductionOp(op_type, out));

    indent() << kTab << func_args << ");\n";
  }

  void handle(const kir::GridBroadcast* grop) final {
    const auto bop = grop->broadcast_op();
    TORCH_INTERNAL_ASSERT(bop->out()->isA<kir::TensorIndex>());

    const ParallelTypeBitmap parallel_types =
        kernel_->summary().broadcast_parallel_types.at(bop);

    TORCH_INTERNAL_ASSERT(
        parallel_types.hasBID(),
        "GridBroadcast needs to be used with a broadcast op that is parallelized with the BID parallel types");

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

    if (wop->isFused()) {
      generateFusedGridWelford(gwop);
      return;
    }

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

  void generateFusedGridWelford(const kir::GridWelford* gwop) {
    const auto wop = gwop->welford_op();
    TORCH_INTERNAL_ASSERT(wop->isFused());

    const auto out = wop->out()->as<kir::TensorIndex>();

    const auto data_type = wop->outAvg()->dtype();
    const auto index_type = wop->outN()->dtype();
    TORCH_INTERNAL_ASSERT(wop->outAvg()->dtype() == wop->outVar()->dtype());

    ArgumentBuilder data_type_args;
    data_type_args.arg(data_type).arg(data_type).arg(index_type);

    const auto sync_buffer = gwop->sync_buffer()->buffer()->as<TensorView>();

    const auto reduction_name = genFusedReductionName(out);

    // template <typename Func, typename... Types>
    // __device__ __inline__ void reduce(
    //   RefTuple<Types...> out,
    //   const LocalTuple<Types...>& inp,
    //   VolatilePtrTuple<Types...> global_work_buffer,
    //   int64_t* global_sync_buffer, // Allocated as product of all
    //                                // non-participating Grid dimension
    //   PtrTuple<Types...> shared_buf,
    //   bool read_pred, // Prevent reading from out of bounds memory
    //   bool write_pred, // Prevent from writing out of bounds
    //   const LocalTuple<Types...>& init_val,
    //   Func reduction_op);

    ArgumentBuilder out_args;
    out_args.arg(gen(wop->outAvg()));
    out_args.arg(gen(wop->outVar()));
    out_args.arg(gen(wop->outN()));

    ArgumentBuilder in_args;
    in_args.arg(gen(wop->inAvg()));
    if (wop->inVar() != nullptr) {
      in_args.arg(gen(wop->inVar()));
    } else {
      in_args.arg("(").append(data_type).append(")0");
    }
    in_args.arg(gen(wop->inN()));

    ArgumentBuilder init_args;
    init_args.arg(gen(wop->initAvg()));
    init_args.arg(gen(wop->initVar()));
    init_args.arg(gen(wop->initN()));

    ArgumentBuilder work_buffer_args;
    work_buffer_args.arg("&")
        .append(varName(gwop->avg_buffer()->buffer()->as<TensorView>()))
        .append("[0]");
    work_buffer_args.arg("&")
        .append(varName(gwop->var_buffer()->buffer()->as<TensorView>()))
        .append("[0]");
    work_buffer_args.arg("&")
        .append(varName(gwop->N_buffer()->buffer()->as<TensorView>()))
        .append("[0]");

    ArgumentBuilder smem_buffer_args;
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(data_type), "shared_mem_avg"));
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(data_type), "shared_mem_var"));
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(index_type), "shared_mem_n"));

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    // out
    func_args.arg(genCall("RefTuple", data_type_args, out_args));
    // inp
    func_args.arg(genCall("ConstRefTuple", data_type_args, in_args));
    // global_work_buffer
    func_args.arg(
        genCall("VolatilePtrTuple", data_type_args, work_buffer_args));
    // global_sync_buffer
    func_args.arg("&").append(varName(sync_buffer)).append("[0]");
    // shared_buf
    func_args.arg(genCall("PtrTuple", data_type_args, smem_buffer_args));
    // read and write predicates
    TORCH_INTERNAL_ASSERT(
        gwop->predicate() != nullptr && gwop->predicate()->hasValue());
    const auto read_pred = genInline(gwop->predicate());
    auto write_pred = read_pred;
    if (gwop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(gwop->writePredicate()->hasValue());
      write_pred = genInline(gwop->writePredicate());
    }
    func_args.arg(read_pred).arg(write_pred);
    // init_val
    func_args.arg(genCall("LocalTuple", data_type_args, init_args));
    // reduction_op
    func_args.arg(genTemplate(
        "welfordCombine", ArgumentBuilder().arg(data_type).arg(index_type)));

    indent() << reduction_name << ".reduce(\n";
    indent() << kTab << func_args << ");\n";
  }

  void handle(const kir::AllocateFusedReduction* alloc_fused_reduction) final {
    // See the runtime file of the fused reduction
    enum class ReductionParallelTypeState { Reduce, Iter, Pred, Inactive };

    using ReductionParallelTypeStateArray =
        ParallelTypeMap<ReductionParallelTypeState>;

    ReductionParallelTypeStateArray states(
        ReductionParallelTypeState::Inactive);

    for (const ParallelType pt : kParallelTypeThreads) {
      // It may be better to predicate grid reductions on dimensions they don't
      // actively use, however since that should generally be discouraged (they
      // should be part of the iter portion of the operation, or they should be
      // predciated out) we're just going to assume they're part of the iter
      // dimension. This would cause more communication than strictly necessary
      // but should not be a common use case.
      auto pt_dim = kernel_->summary().parallel_dimension_map_.get(pt);
      if (pt_dim == nullptr || pt_dim->isOneInt()) {
        continue;
      }
      // Initialize pt_dim if used to an iter dimension. It may change to a
      // reduction or predicated dimension later.
      states[pt] = ReductionParallelTypeState::Iter;
    }

    for (auto id : alloc_fused_reduction->out()->view()->domain()->domain()) {
      auto pt = id->getParallelType();
      if (isParallelTypeThread(pt)) {
        auto state = id->isReduction() ? ReductionParallelTypeState::Reduce
                                       : ReductionParallelTypeState::Iter;
        states[pt] = state;
      }
    }

    for (const auto predicated_pt : alloc_fused_reduction->threadPredicate()) {
      auto& state = states[predicated_pt];
      TORCH_INTERNAL_ASSERT(
          state != ReductionParallelTypeState::Reduce,
          "Invalid thread predication: ",
          predicated_pt);
      state = ReductionParallelTypeState::Pred;
    }

    ArgumentBuilder flags;
    for (auto pt : kParallelTypeThreads) {
      flags.arg(static_cast<int>(states[pt]));
    }

    // Persistent
    flags.arg(true);

    // Broadcast is fused
    flags.arg(true);

    const auto reduction_name =
        genFusedReductionName(alloc_fused_reduction->out());

    indent() << genTemplate("fused_reduction::ParallelReduce", flags) << " "
             << reduction_name << ";\n";
  }

  void handleScope(const kir::Scope& scope) {
    for (auto expr : scope.exprs()) {
      OptOutConstDispatch::handle(expr);
    }
  }

  void handleTrivialLoop(const kir::ForLoop* loop) {
    if (loop->vectorize()) {
      vectorize_scope_ = loop->vectorize();
    }
    handleScope(loop->body());
    if (loop->vectorize()) {
      vectorize_scope_ = false;
    }
  }

  void handle(const kir::ForLoop* loop) final {
    if (loop->isTrivial()) {
      handleTrivialLoop(loop);
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

    TORCH_INTERNAL_ASSERT(alloc->buffer() != nullptr);
    alloc_map_.emplace(alloc->buffer(), alloc);

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
      indent() << "auto& " << varName(tv) << " = " << varName(alias_tv)
               << ";\n";

    } else {
      // Standard Memory Allocation
      switch (tv->getMemoryType()) {
        case MemoryType::Global:
          indent() << "// Allocate global tensor " << varName(tv) << "\n";
          break;
        case MemoryType::Shared:
          if (kir::ExpressionEvaluator::isConst(size)) {
            // Static shared memory
            //  Always align to 16B for tensorview buffers
            //   with any vectorized access.
            //  TODO:
            //   This path will be less commonly exercised once we
            //    start dynamically allocate all the tensors and
            //    might be removed in a follow up.
            auto va = kernel_->summary().vectorized_accesses;
            if (va.count(tv)) {
              indent() << "__align__(16) ";
            } else {
              indent();
            }
            code_ << "__shared__ " << buffer_dtype << " " << varName(tv) << "["
                  << genInline(size) << "];\n";
          } else {
            // Align Offset Position
            indent() << "offset = alignBufferSize(offset, "
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
        case MemoryType::Local: {
          auto va = kernel_->summary().vectorized_accesses;
          if (va.find(tv) != va.end()) {
            indent() << "Array<" << buffer_dtype << ", " << genInline(size)
                     << ", " << va.at(tv) << "> " << varName(tv) << ";\n";
          } else {
            indent() << buffer_dtype << " " << varName(tv) << "["
                     << genInline(size) << "];\n";
          }
        } break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unexpected memory type");
      }
    }
  }

  void handle(const kir::BlockSync*) final {
    // Use a custom synchronization method if enabled
    if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
      indent() << "block_sync::sync();\n";
    } else {
      indent() << "__barrier_sync(0);\n";
    }
  }

  void handle(const kir::GridSync* sync) final {
    // Use a custom synchronization method if enabled
    bool bidx = sync->syncDims().get(ParallelType::BIDx);
    bool bidy = sync->syncDims().get(ParallelType::BIDy);
    bool bidz = sync->syncDims().get(ParallelType::BIDz);
    auto bool2str = [](bool b) { return (b ? "true" : "false"); };
    std::stringstream sync_str;
    sync_str << bool2str(bidx) << ", " << bool2str(bidy) << ", "
             << bool2str(bidz);

    std::stringstream sync_segment_size;
    sync_segment_size << "index_utils::maskedSize<" << sync_str.str()
                      << ">(gridDim)";

    std::stringstream sync_idx;
    sync_idx << "index_utils::maskedOffset<" << bool2str(!bidx) << ", "
             << bool2str(!bidy) << ", " << bool2str(!bidz)
             << ">(gridDim, blockDim)";

    indent() << "grid_sync::sync<" << sync_str.str() << ", true>(\n";
    indent() << "  " << varName(sync->syncBuffer()) << "[" << sync_idx.str()
             << "],\n";
    indent() << "  " << sync_segment_size.str() << ");\n";
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

  //! Keep track of Allocate node for Val. Used to determine if Val
  //! should be inlined.
  std::unordered_map<const Val*, const kir::Allocate*> alloc_map_;
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
