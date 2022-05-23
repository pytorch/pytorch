#pragma once
#include <c10/macros/Export.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Utility data structure for recording gemm tiles
struct GemmTile {
  int m, n, k;
  GemmTile(int m_, int n_, int k_) : m(m_), n(n_), k(k_) {}

  bool operator==(const GemmTile& other) {
    return m == other.m && n == other.n && k == other.k;
  }

  GemmTile operator/(const GemmTile& other) {
    return GemmTile(m / other.m, n / other.n, k / other.k);
  }
};

//! Utility data structure for recording gemm tiles
struct TORCH_CUDA_CU_API MatMulTileOptions {
  GemmTile cta_tile = GemmTile(128, 128, 32);
  GemmTile warp_tile = GemmTile(64, 64, 32);
  GemmTile instruction_tile = GemmTile(16, 8, 16);

  MatMulTileOptions() = default;
  MatMulTileOptions(
      GemmTile cta_tile_,
      GemmTile warp_tile_,
      GemmTile instruction_tile_)
      : cta_tile(cta_tile_),
        warp_tile(warp_tile_),
        instruction_tile(instruction_tile_) {}

  bool operator==(const MatMulTileOptions& other) {
    return cta_tile == other.cta_tile && warp_tile == other.warp_tile &&
        instruction_tile == other.instruction_tile;
  }
};

//! Information for configuring and lowering mma ops
struct MmaOptions {
  //! Type of mma instrinsic macro to use
  //!  This will translate to which mma intrinsic from runtime string
  //!    to be generated to implement the mma op. The current plan
  //!    is to have exactly one macro for each
  //!  (arch, datatype, operand layout) triple, though there
  //!  exists multiple possibilities for some cases, e.g. for Turing and fp16
  //!  one can use 16_8_8 or 16_8_16.
  //! Will consider adding more choices that the scheduler can pick from
  //!  when our perf target becomes more fine grained, which is more likely in
  //!  latency bound kernels.
  enum class MacroType {
    NoMMA = 0,
    Volta_16_16_4,
    Turing_16_8_16, // place holder for turing/ampere mma
    Ampere_16_8_8 // place holder for tf32
  };

  //! [Operand Layout Convention]
  //! Operand layout, T=transposed/row_major, N=normal/col_major
  //!   We don't support calling NN mma directly since it implies
  //!    a fused transpose. User needs to swap the operands and use
  //!    TT mma to make the transpose explicit.
  //! Ordered by position of K
  //! NT : K,M x K,N -> K,M,N
  //! TT : M,K X K,N -> M,K,N
  //! TN : M,K X N,K -> M,N,K
  enum class MmaInputLayout { NT = 0, TT, TN };

  //! Utility to annotate which input of mma this option struct describes
  enum class Operand { NotOperand = 0, A, B };

  //! Utility to annotate which mma macro this config uses.
  MacroType macro = MacroType::NoMMA;

  //! Utility to annotate transposition of operands
  MmaInputLayout operand_layout = MmaInputLayout::TT;

  //! Utility to annotate which input of mma this option struct describes
  Operand operand = Operand::A;

  //! Accumulator register stride, will be removed when the swizzle op
  //!  is introduced and the output can be labeled with a transpose swizzle.
  int accumulator_stride = 0;

  bool operator==(const MmaOptions& other) const {
    return macro == other.macro && operand_layout == other.operand_layout &&
        operand == other.operand &&
        accumulator_stride == other.accumulator_stride;
  }
};

//! User interface generating mma options for mma op
class TORCH_CUDA_CU_API MmaBuilder {
 public:
  MmaBuilder(MmaOptions::MacroType macro, MatMulTileOptions gemm_tile);
  MmaBuilder& layout(MmaOptions::MmaInputLayout layout);
  MmaBuilder& operand(MmaOptions::Operand a_or_b);
  MmaOptions build() const;

 private:
  MmaOptions option_;
};

//! GPU arch check for macro type
bool isVolta(MmaOptions::MacroType macro);
bool isTuring(MmaOptions::MacroType macro);
bool isAmpere(MmaOptions::MacroType macro);

//! Returns true if the given option describes a transposed operand
bool isOperandTransposed(MmaOptions options);

// Unpacked constants from macro type:
//   exact numbers are defined by each individual instruction.
int getOutputRegisterSize(MmaOptions::MacroType macro);
int getInputARegisterSize(MmaOptions::MacroType macro);
int getInputBRegisterSize(MmaOptions::MacroType macro);

// MMA stringify utils
std::string toString(MmaOptions::MacroType macro);
std::string toString(MmaOptions::MmaInputLayout input_layout);
std::string toString(MmaOptions::MacroType mt);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
