
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/mma_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace mma_util {

namespace {

// Utility for mma dimension matching
enum class MmaDimension { M = 0, N, K };

// Utility for mma dimension matching before broadcast,
//   assumes the innermost 2 dimensions are the mma
//   operand dimensions, i.e. mnk.
IterDomain* getMmaOperandRootDimension2d(
    TensorView* tv,
    MmaOptions options,
    MmaDimension mma_dimension) {
  TORCH_INTERNAL_ASSERT(tv->getMaybeRFactorDomain().size() >= 2);
  // NT : K,M x K,N -> K,M,N
  // TT : M,K X K,N -> M,K,N
  // TN : M,K X N,K -> M,N,K
  int axis_id = mma_dimension == MmaDimension::K ? 1 : 0;
  bool is_transposed = isOperandTransposed(options);

  // Decode the transpostion
  if ((options.operand == MmaOptions::Operand::A && !is_transposed) ||
      (options.operand == MmaOptions::Operand::B && is_transposed)) {
    axis_id = 1 - axis_id;
  }

  int root_size = tv->getMaybeRFactorDomain().size();
  // Convert to index from right.
  return tv->getMaybeRFactorDomain().at(root_size + axis_id - 2);
}

// Utility for mma dimension matching, assumes the innermost
//  3 dimensions are the mma operand dimensions, i.e. mnk, but
// not necessarily in this order.
// For matmul use cases the root domains are always 3 dimensional,
//  but this wouldn't be the case for other kernels such as batched gemm.
// This utility only applies to the case where the innermost 3 dims
//  are the one that mma's are used. We probably don't want to use
//  mma intrinsics if that's not the case.
IterDomain* getMmaOperandRootDimension3d(
    TensorView* tv,
    MmaOptions::MmaInputLayout layout,
    MmaDimension mma_dimension) {
  TORCH_INTERNAL_ASSERT(tv->getMaybeRFactorDomain().size() >= 3);
  // NT : K,M x K,N -> K,M,N
  // TT : M,K X K,N -> M,K,N
  // TN : M,K X N,K -> M,N,K
  int axis_id = -1;
  switch (mma_dimension) {
    case MmaDimension::K:
      axis_id = (int)layout;
      break;
    case MmaDimension::M:
      axis_id = layout == MmaOptions::MmaInputLayout::NT ? 1 : 0;
      break;
    case MmaDimension::N:
      axis_id = layout == MmaOptions::MmaInputLayout::TN ? 1 : 2;
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unreachable");
      break;
  }

  int root_size = tv->getMaybeRFactorDomain().size();
  // Convert to index from right.
  return tv->getMaybeRFactorDomain().at(root_size + axis_id - 3);
}

// Locate the root id corresponding to the given mma dimension
//  Assumes the mma dimension always the innermost 2 or 3, might
//  need to extend for more complex fusions.
IterDomain* getMmaOperandRootDimension(
    TensorView* tv,
    MmaOptions options,
    MmaDimension mma_dimension) {
  if (isVolta(options.macro)) {
    return getMmaOperandRootDimension3d(
        tv, options.operand_layout, mma_dimension);
  } else if (isTuring(options.macro) || isAmpere(options.macro)) {
    // Volta mma swizzle requires the broadcast dimension to
    //  participate, which is not true in Turing+. So the two
    //  cases w/ or w/o the broadcast are supported here for
    //  mma pre-swizzle validation.
    bool has_broadcast_or_reduction = std::any_of(
        tv->getMaybeRFactorDomain().begin(),
        tv->getMaybeRFactorDomain().end(),
        [](IterDomain* id) { return id->isBroadcast() || id->isReduction(); });
    if (has_broadcast_or_reduction) {
      TORCH_INTERNAL_ASSERT(tv->nDims() >= 3);
      return getMmaOperandRootDimension3d(
          tv, options.operand_layout, mma_dimension);
    } else {
      TORCH_INTERNAL_ASSERT(tv->nDims() >= 2);
      return getMmaOperandRootDimension2d(tv, options, mma_dimension);
    }
  }
  TORCH_INTERNAL_ASSERT(false, "unreachable");
  return nullptr;
}

// Preliminary checks to try to validate that leaf is
//  a innermost dim of root of exactly the given size.
bool canValidateIsInnerDim(
    IterDomain* root,
    IterDomain* leaf,
    int inner_dim_size) {
  // Accept boundary case for Volta.
  if (leaf == root && leaf->isBroadcast()) {
    return true;
  }
  auto expr = leaf->definition();
  ExpressionEvaluator const_eval(leaf->fusion());
  auto maybe_leaf_size = const_eval.evaluate(leaf->extent());
  if (!maybe_leaf_size.has_value()) {
    return false;
  }
  if (maybe_leaf_size.value() != inner_dim_size) {
    return false;
  }

  while (expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      // Inner split only
      if (leaf != split->inner()) {
        return false;
      }
      // Const split only
      auto maybe_factor = const_eval.evaluate(split->factor());
      if (!maybe_factor.has_value()) {
        return false;
      }
      int factor = maybe_factor.value();
      if (factor < inner_dim_size) {
        // This might be too restrictive. Would need more
        //   bookkeeping to relax.
        return false;
      }
      leaf = split->in();
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // Might consider just rejecting merge.
      auto outer = merge->outer();
      if (outer->isBroadcast()) {
        return false;
      }

      // Only support merging with constant sized dims
      maybe_leaf_size = const_eval.evaluate(leaf->extent());
      if (!maybe_leaf_size.has_value()) {
        return false;
      }
      if (maybe_leaf_size.value() != inner_dim_size) {
        return false;
      }
      leaf = merge->inner();
    } else {
      // No support for swizzled inner dim for now.
      //  Might need to add transpose swizzle here.
      return false;
    }
    expr = leaf->definition();
  }
  return leaf == root;
}

} // namespace

void checkDimSize(
    TensorView* tv,
    std::vector<int> axis,
    std::vector<int> expect) {
  TORCH_INTERNAL_ASSERT(
      axis.size() == expect.size(),
      "CheckDimSize: Mismatched axis and expect size");
  ExpressionEvaluator const_eval(tv->fusion());
  for (auto axis_index : c10::irange(axis.size())) {
    TORCH_INTERNAL_ASSERT(
        ((axis[axis_index] + static_cast<int>(tv->nDims())) >= 0) &&
            (axis[axis_index] < (int)tv->nDims()),
        "CheckDimSize: axis position out of bound ",
        axis[axis_index],
        " ",
        tv->nDims());
    auto id = tv->axis(axis[axis_index]);
    auto maybe_extent = const_eval.evaluate(id->extent());
    TORCH_CHECK(
        maybe_extent.has_value(),
        "Mma warp mapping: instruction tile has to be constant");
    TORCH_CHECK(
        maybe_extent.value() == expect[axis_index],
        "Mma warp mapping: unexpected tile size at",
        axis_index,
        ":",
        maybe_extent.value(),
        "vs",
        expect[axis_index]);
  }
}

void WarpMmaSwizzler::scheduleMmaWarpOutput(
    TensorView* tv,
    MmaOptions options) {
  auto macro = options.macro;
  switch (macro) {
    case MmaOptions::MacroType::Volta_16_16_4:
      scheduleVoltaM16N16K4Fp32Output(tv, options);
      if (tv->definition()->isA<MmaOp>()) {
        setWarpMapped(tv, 5);
      }
      break;
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
      scheduleTuringM16N8K16MmaWarpOutput(tv, options);
      if (tv->definition()->isA<MmaOp>()) {
        setWarpMapped(tv, 4);
      }
      break;
    default:
      TORCH_CHECK(
          false, "scheduleMmaWarp: unsupported mma option ", toString(macro));
      break;
  }
}

void WarpMmaSwizzler::scheduleOperandRead(TensorView* tv, MmaOptions options) {
  // Schedules operand for inner most 3 contiguous dimensions
  // Assumes M, N, K

  switch (options.macro) {
    case MmaOptions::MacroType::Volta_16_16_4:
      scheduleVoltaOperandRead(tv, options);
      break;
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
      scheduleTuringOperandRead(tv, options);
      break;
    default:
      TORCH_CHECK(false, "WarpMmaSwizzler: please specify macro");
      break;
  }
}

void WarpMmaSwizzler::setWarpMapped(TensorView* tv, int number_of_dims) {
  for (int id : c10::irange(number_of_dims)) {
    tv->axis(-id - 1)->toMmaSwizzled();
  }
}

namespace {

// Utility to check operand innermost scheduled dimensions
void validateInnerMNK(TensorView* tv, MmaOptions options, int m, int n, int k) {
  TORCH_INTERNAL_ASSERT(tv->nDims() >= 3);
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(
          getMmaOperandRootDimension(tv, options, MmaDimension::M),
          tv->axis(-3),
          m),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(
          getMmaOperandRootDimension(tv, options, MmaDimension::N),
          tv->axis(-2),
          n),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(
          getMmaOperandRootDimension(tv, options, MmaDimension::K),
          tv->axis(-1),
          k),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
}

void validateResultInnerMN(TensorView* tv, int m, int n) {
  TORCH_INTERNAL_ASSERT(tv->nDims() >= 2);
  int root_dim = tv->getMaybeRFactorDomain().size();
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(
          tv->getMaybeRFactorDomain()[root_dim - 2], tv->axis(-2), m),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(
          tv->getMaybeRFactorDomain()[root_dim - 1], tv->axis(-1), n),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
}

//! Performs checks on tv given to schedule ld matrix.
//!  Currently only allowed ones are either:
//!    1. direct output of an ldmatrix op  or
//!    2. direct output of a broadcast op following a ldmatrix op
//!  Returns true if the tv is an immediate output of ldmatrix op
//!
//! TODO: this check is a WAR with pattern matching for now.
//!  The two patterns mentioned above are the only supported use
//!    cases of ldmatrix currently. This restriction can be greatly
//!    relaxed after the iterdomain swizzle infrastructure, which
//!    will provide the capability to directly model the exact
//!    data format of ldmatrix output.
bool checkLdMatrixTv(TensorView* tv) {
  // First check if tv is an ldmatrix output:
  auto tv_def = tv->definition();
  TORCH_CHECK(tv_def != nullptr, "ldmatrix : invalid tv");
  bool is_immediate_output = true;
  if (!ir_utils::isLdMatrixOp(tv_def)) {
    // Only allow one broadcast in between tv and the ldmatrix op
    TORCH_CHECK(
        tv_def->isA<BroadcastOp>(),
        "ldmatrix: only allow serial broadcast between ldmatrix and mma");
    tv_def = tv_def->input(0)->definition();
    TORCH_CHECK(tv_def != nullptr, "ldmatrix : invalid tv");
    is_immediate_output = false;
  }
  TORCH_CHECK(ir_utils::isLdMatrixOp(tv_def), "ldmatrix : invalid op type");
  TORCH_CHECK(
      tv->nDims() > 2,
      "ldmatrix: scheduled tv needs to be more than 2 dimensional");
  TORCH_CHECK(
      !tv->axis(-1)->isBroadcast(), "ldmatrix: unsupported scheduled axes");
  TORCH_CHECK(
      !tv->axis(-1)->isReduction(), "ldmatrix: unsupported scheduled axes");
  TORCH_CHECK(
      !tv->axis(-2)->isBroadcast(), "ldmatrix: unsupported scheduled axes");
  TORCH_CHECK(
      !tv->axis(-2)->isReduction(), "ldmatrix: unsupported scheduled axes");
  return is_immediate_output;
}

void scheduleVoltaA(TensorView* tv, MmaOptions options) {
  // Assumed:
  // [..., 16, 16 ,4]
  // [..., M,  BN, K]
  // Some validation:
  validateInnerMNK(tv, options, 16, 16, 4);
  bool transposed = isOperandTransposed(options);

  tv->split(-3, 4);

  // Split out 16 from the bcast
  tv->split(-2, 16);
  tv->split(-2, 8);

  // -6   -5    -4  -3   -2  -1
  //[Mo4, Mi4, Noo, No2, Ni8, K]

  if (transposed) {
    tv->reorder({{-5, -3}, {-3, -5}});
    // -6   -5    -4  -3   -2  -1
    //[Mo4, No2, Noo, Mi4, Ni8, K]

  } else {
    tv->reorder({{-5, -1}, {-3, -5}, {-1, -3}});
    // -6   -5    -4  -3  -2  -1
    //[Mo4, No2, Noo,  K, Ni8, Mi4]
  }

  tv->merge(-6);
  tv->merge(-5);
  tv->merge(-4);

  //[Warp, Ni8, K/Mi4]
  tv->axis(-3)->parallelize(ParallelType::TIDx);
}

void scheduleVoltaB(TensorView* tv, MmaOptions options) {
  // Assumed:
  // [..., 16,16,4]
  // [..., BM, N, K]
  // Some validation:
  validateInnerMNK(tv, options, 16, 16, 4);

  bool transposed = isOperandTransposed(options);
  tv->split(-3, 16);
  tv->split(-3, 8);

  tv->split(-2, 8);
  tv->split(-2, 4);

  // -7   -6   -5   -4   -3    -2   -1
  //[Moo, Mo2, Mi8, No2, Nio2, Nii4, K]
  tv->reorder({{-6, -4}, {-5, -6}, {-4, -3}, {-3, -5}});

  // -7   -6   -5   -4    -3    -2   -1
  //[Moo, Mi8, Nio2, Mo2, No2,  Nii4, K ]
  if (transposed) {
    tv->reorder({{-2, -1}, {-1, -2}});
    //  -7   -6   -5   -4    -3  -2   -1
    //[Moo, Mi8, Nio2, Mo2, No2, K, Nii4]
  }

  tv->merge(-5);
  tv->merge(-4);
  tv->merge(-3);

  //[Moo, Mi8, Warp, K/Nii4]
  tv->axis(-2)->parallelize(ParallelType::TIDx);
}

void scheduleLdMatrix(TensorView* tv, MmaOptions options) {
  // Check if tv should use ldmatrix layout and
  //   if tv is immediate output of ldmatrix
  bool is_immediate_output = checkLdMatrixTv(tv);

  // Decode transposition requirement for turing mma
  bool transposed = options.operand == MmaOptions::Operand::A
      ? !isOperandTransposed(options)
      : isOperandTransposed(options);
  // Check mma option is supported
  TORCH_CHECK(
      options.macro == MmaOptions::MacroType::Ampere_16_8_16 ||
          options.macro == MmaOptions::MacroType::Turing_16_8_16,
      "scheduleLdMatrix: unknown macro for ldmatrix");

  if (options.operand == MmaOptions::Operand::A) {
    TORCH_INTERNAL_ASSERT(tv->nDims() >= 2);
    // validation:
    TORCH_INTERNAL_ASSERT(
        canValidateIsInnerDim(
            getMmaOperandRootDimension(tv, options, MmaDimension::M),
            tv->axis(-2),
            16),
        "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
    TORCH_INTERNAL_ASSERT(
        canValidateIsInnerDim(
            getMmaOperandRootDimension(tv, options, MmaDimension::K),
            tv->axis(-1),
            16),
        "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");

    //[16m, 16k]
    tv->split(-2, 8);
    tv->split(-1, 8);

    // -4  -3  -2  -1
    //[2o, 8o, 2i, 8i]
    tv->reorder({{-4, -3}, {-3, -2}, {-2, -4}});

    //  -4  -3 -2  -1
    // [2i, 2o, 8o, 8i]

    if (transposed) {
      tv->reorder({{-1, -2}, {-2, -1}});
    }

    tv->merge(-4);
    tv->merge(-3);
    // [warp, 8i/o]

    tv->axis(-2)->parallelize(ParallelType::TIDx);
  } else if (options.operand == MmaOptions::Operand::B) {
    // validation:
    TORCH_INTERNAL_ASSERT(
        canValidateIsInnerDim(
            getMmaOperandRootDimension(tv, options, MmaDimension::N),
            tv->axis(-2),
            8),
        "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
    TORCH_INTERNAL_ASSERT(
        canValidateIsInnerDim(
            getMmaOperandRootDimension(tv, options, MmaDimension::K),
            tv->axis(-1),
            16),
        "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");

    if (transposed) {
      // [8, 16]
      tv->split(-2, 4);

      // [2i, 4i, 16]
      tv->reorder({{-1, -2}, {-2, -1}});
      // [2i, 16, 4i]

      tv->merge(-3);
      // [warp, 4i]
    } else {
      //[8, 16]
      tv->split(-1, 4);
      tv->split(-2, 2);

      // 0  1   2   3
      //[8, oo2,oi2,i4]
      tv->reorder({{-4, -2}, {-2, -4}});

      // 0     1   2  3
      //[oi2, oo2, 8,i4]

      tv->merge(-4);
      tv->merge(-3);
      //  0    1
      //[warp, i4]
    }

    tv->axis(-2)->parallelize(ParallelType::TIDx);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unreachable");
  }

  if (is_immediate_output) {
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }
}

} // namespace

void WarpMmaSwizzler::scheduleVoltaOperandRead(
    TensorView* tv,
    MmaOptions options) {
  switch (options.operand) {
    case MmaOptions::Operand::A:
      scheduleVoltaA(tv, options);
      setWarpMapped(tv, 3);
      break;
    case MmaOptions::Operand::B:
      scheduleVoltaB(tv, options);
      setWarpMapped(tv, 4);
      break;
    default:
      TORCH_CHECK(false, "WarpMmaSwizzler: please specify operand");
  }
}

// Fp32 and Fp16 outputs have different layouts on volta,
//   but we only support fp32 accumulate at this stage.
void WarpMmaSwizzler::scheduleVoltaM16N16K4Fp32Output(
    TensorView* tv,
    const MmaOptions& options) {
  // Assume last 2 dims [M16, N16] or [M16, N16, R]
  bool is_reduction = tv->axis(-1)->isReduction();

  // Make sure instruction tile size is correct.
  if (is_reduction) {
    validateInnerMNK(tv, options, 16, 16, 4);
  } else {
    validateResultInnerMN(tv, 16, 16);
  }

  int m_pos = is_reduction ? -3 : -2;

  // Assumed:
  //       m
  // [..., 16,16, (4)]
  // [..., M, N,  (R)]
  tv->split(m_pos, 4);
  tv->split(m_pos, 2);
  tv->split(m_pos + 1, 8);
  tv->split(m_pos + 1, 4);
  tv->split(m_pos + 1, 2);

  //        m-5  m-4   m-3   m-2   m-1    m     m+1   m+2
  // [..., Mo4, Mio2, Mii2,  No2, Nio2, Niio2, Niii2, (R)]
  tv->reorder(
      {{m_pos - 4, m_pos - 1},
       {m_pos - 3, m_pos - 2},
       {m_pos - 2, m_pos - 4},
       {m_pos - 1, m_pos},
       {m_pos, m_pos - 3}});

  //        m-5  m-4   m-3   m-2   m-1    m     m+1   m+2
  //  [..., Mo4, No2, Niio2, Mii2, Mio2, Nio2, Niii2, (R)]

  tv->merge(m_pos - 5);
  tv->merge(m_pos - 4);
  tv->merge(m_pos - 3);

  //  m-2   m-1   m     m+1   m+2
  //[Warp, Mio2, Nio2, Niii2, (R)]
  tv->axis(m_pos - 2)->parallelize(ParallelType::TIDx);

  if (is_reduction && tv->definition()->isA<MmaOp>()) {
    // Set instruction loops for mma reduce output
    for (int pos : c10::irange(5)) {
      if (!tv->axis(-pos - 1)->isThread()) {
        tv->axis(-pos - 1)->parallelize(ParallelType::Mma);
      }
      tv->axis(-pos - 1)->toMmaSwizzled();
    }
  }
}

void WarpMmaSwizzler::scheduleTuringOperandRead(
    TensorView* tv,
    MmaOptions options) {
  scheduleLdMatrix(tv, options);
  setWarpMapped(tv, 2);
}

void WarpMmaSwizzler::scheduleTuringM16N8K16MmaWarpOutput(
    TensorView* tv,
    const MmaOptions& options) {
  // Assume last 2 dims [M16, N8] or [M16, N8, R]
  // Locate instruction m
  bool is_reduction = tv->axis(-1)->isReduction();

  // Make sure instruction tile size is correct.
  if (is_reduction) {
    validateInnerMNK(tv, options, 16, 8, 16);
  } else {
    validateResultInnerMN(tv, 16, 8);
  }

  int m_pos = is_reduction ? -3 : -2;

  //  m
  // [16, 8  (,R)]
  tv->split(m_pos, 8);
  tv->split(m_pos + 1, 2);

  //          m
  // [2o, 8o, 4i, 2i (,R)]
  tv->merge(m_pos - 1);

  //       m
  // [2o, Warp, 2i (,R)]
  TORCH_CHECK(tv->definition() != nullptr);

  if (is_reduction && tv->definition()->isA<MmaOp>()) {
    // Set instruction loops for mma reduce
    for (int pos : c10::irange(4)) {
      tv->axis(-pos - 1)->parallelize(ParallelType::Mma);
    }
  }

  tv->axis(m_pos)->parallelize(ParallelType::TIDx);
}

namespace {

bool isMmaInitLoop(const kir::Scope& loop_body) {
  for (auto expr : loop_body.exprs()) {
    if (auto inner_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      if (!isMmaInitLoop(inner_loop->body())) {
        return false;
      }
    } else if (auto uop = dynamic_cast<UnaryOp*>(expr)) {
      if (!ir_utils::isTvOp(expr) ||
          uop->getUnaryOpType() != UnaryOpType::Set) {
        return false;
      }
      if (auto ti = dynamic_cast<kir::TensorIndex*>(expr->output(0))) {
        if (!ti->view()->definition() ||
            !ti->view()->definition()->isA<MmaOp>()) {
          return false;
        }
      }
      if (auto tv = dynamic_cast<TensorView*>(expr->output(0))) {
        if (!tv->definition() || !tv->definition()->isA<MmaOp>()) {
          return false;
        }
      }
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      if (!isMmaInitLoop(ite->thenBody())) {
        return false;
      }
      if (!isMmaInitLoop(ite->elseBody())) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

} // namespace

bool isMmaInitLoop(const kir::ForLoop* loop) {
  return isMmaInitLoop(loop->body());
}

} // namespace mma_util

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
