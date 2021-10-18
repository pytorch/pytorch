#pragma once
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API LaunchParams {
 public:
  static constexpr int64_t UNINITIALIZED_VAL = -1;

  LaunchParams(
      int64_t gdimx = UNINITIALIZED_VAL,
      int64_t gdimy = UNINITIALIZED_VAL,
      int64_t gdimz = UNINITIALIZED_VAL,
      int64_t bdimx = UNINITIALIZED_VAL,
      int64_t bdimy = UNINITIALIZED_VAL,
      int64_t bdimz = UNINITIALIZED_VAL)
      : gdimx_(gdimx),
        gdimy_(gdimy),
        gdimz_(gdimz),
        bdimx_(bdimx),
        bdimy_(bdimy),
        bdimz_(bdimz) {
    assertValid();
  }

  void assertValid();

  void setSmem(int64_t smem) {
    smem_ = smem;
  }

  int64_t smem() const {
    return smem_;
  }

  int64_t nBlocks() const {
    return std::abs(gdimx_ * gdimy_ * gdimz_);
  }

  int64_t nThreads() const {
    return std::abs(bdimx_ * bdimy_ * bdimz_);
  }

  int64_t bdimx() const {
    return static_cast<int64_t>(bdimx_ == UNINITIALIZED_VAL ? 1 : bdimx_);
  }

  int64_t gdimx() const {
    return static_cast<int64_t>(gdimx_ == UNINITIALIZED_VAL ? 1 : gdimx_);
  }

  int64_t bdimy() const {
    return static_cast<int64_t>(bdimy_ == UNINITIALIZED_VAL ? 1 : bdimy_);
  }

  int64_t gdimy() const {
    return static_cast<int64_t>(gdimy_ == UNINITIALIZED_VAL ? 1 : gdimy_);
  }

  int64_t bdimz() const {
    return static_cast<int64_t>(bdimz_ == UNINITIALIZED_VAL ? 1 : bdimz_);
  }

  int64_t gdimz() const {
    return static_cast<int64_t>(gdimz_ == UNINITIALIZED_VAL ? 1 : gdimz_);
  }

  void checkAndSet(
      const int64_t incoming_val,
      int64_t& class_val,
      std::string val) {
    TORCH_INTERNAL_ASSERT(
        class_val == UNINITIALIZED_VAL || incoming_val == class_val,
        "Tried to set ",
        val,
        " from ",
        class_val,
        " to ",
        incoming_val,
        ", but it was already set and new value does not match.",
        " Thread dims all have to be bound to the same value.");
    TORCH_CHECK(
        incoming_val > 0,
        "Received a thread binding on ",
        val,
        " that is ",
        incoming_val,
        ". Cannot create negative threads.");
    if (class_val == UNINITIALIZED_VAL) {
      class_val = incoming_val;
    }
    assertValid();
  }

  // Binds dim assocaited with p_type to val
  void bind(int64_t val, ParallelType p_type);

  // Adjusted value based on get functions above for each value
  int64_t getDim(ParallelType p_type) const;

  // Returns raw value which may be UNINITIALIZED_VAL
  const int64_t& getRawVal(ParallelType p_type) const;

  // Returns false if value associated with p_type == UNINITIALIZED_VAL
  bool hasDim(ParallelType p_type) const;

  bool operator==(const LaunchParams& other) const;

  void print() const;

  std::string toString() const;

 private:
  // Spell them out because I want signed ints to know if they were initialized
  // or not.
  // TODO: convert to c10::optional
  int64_t gdimx_ = UNINITIALIZED_VAL;
  int64_t gdimy_ = UNINITIALIZED_VAL;
  int64_t gdimz_ = UNINITIALIZED_VAL;
  int64_t bdimx_ = UNINITIALIZED_VAL;
  int64_t bdimy_ = UNINITIALIZED_VAL;
  int64_t bdimz_ = UNINITIALIZED_VAL;

  int64_t smem_ = 0;

  // TODO: Fill in output sizes
  std::vector<std::vector<int64_t>> output_sizes;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
