#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>

// Double buffering a tensor doubles its allocation size and uses two
// buffers to facilitate computation and memory access
// overlapping. The basic form of code looks like as follows:
//
// Before:
// for i
//   x[S]; // allocation
//   for j:
//     x[j] = y[i, j]
//   for j:
//     ... = x[j]
//
// After:
// X[S * 2]; // allocation
// for i in 0 to 1: // Prologue
//   for j:
//     x[j] = y[i, j]
//
// for i in 0 to N-1: // Main
//   for j:
//     x[j + (1 - i % 2) * S] = y[i + 1, j]
//   for j:
//     ... = x[j + (i % 2) * S]
//
// for i in N-1 to N: // Epilogue
//   for j:
//     ... = x[j + (i % 2) * S]
//
// Here, S is the original size of tensor x.
//
// The i loop is the double buffer loop of tensor x, where double
// buffering is applied to the tensor. The first step of lowering is
// to find the double buffering axis for each double buffered
// tensor. It must not be parallelized as it isn't possible to double
// buffer parallelized loops. Also, an unrolled axis expands the
// allocation and is intended to make the loop completely unrolled,
// which also conflicts with double buffering. So, basically, the double
// buffering axis is the inner-most axis within the axes left
// of the CA position. However, when it is parallelized or unrolled, a
// further left axis is picked.
//
// Once the double buffer axis is determined, the main task is to
// replicate the corresponding double buffer loop as illustrated
// above. The Prologue loop is to just fetch the first element to
// populate the buffer. The main loop is mostly the same as the
// original loop, except for the indexing change to switch the two
// buffers. When used as a consumer, an offset of (1 - i % 2) * S is
// added, whereas (i % 2) * S is added when used as a producer. Here,
// i is the index of the double buffer loop. The Epilogue loop is just
// for the last iteration of the loop. Since the main loop reads one
// element ahead of the producer of the double buffered tensor, it
// would require an additional guard to prevent buffer overruns with
// the producer if the main loop were also used for the last
// iteration. However, the value loaded by the invalid load would not
// be used, so instead of adding the additional predicate, the Epilogue
// loop is replicated from the original loop, except for the load
// expression since it's not used. Note that this overrun does not
// happen when the producer is on gmem, so in that case, this
// additional replication is not done.
//
// When creating those three types of loops, additional care must be
// taken when multiple tensors are double buffered. When multiple
// tensors use the same loop as their double buffer loop, one pass of
// replication takes care of them at once, meaning the same Prologue,
// Main, Epilogue loops are used for the multiple tensors.
//
// Other tasks to do for a double buffer tensor include:
// - Move allocation to outside of the double buffer loop
// - Double the allocation size
// - Omit the RAW sync in the Main and Epilogue loops

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

unsigned int getDoubleBufferAxisPosition(const TensorView* tv);

IterDomain* getDoubleBufferAxis(const TensorView* tv);

void validateDoubleBufferedTensor(const TensorView* tv);

class TORCH_CUDA_CU_API DoubleBufferPass {
 public:
  //! Apply double buffering transformations
  static std::vector<Expr*> run(const std::vector<Expr*>& exprs);
};

class TORCH_CUDA_CU_API DoubleBufferInfo {
  // Lowering information of double buffered tensors.
  struct TvInfo {
    IterDomain* double_buffer_axis = nullptr;
    Val* original_alloc_size = nullptr;
  };

 public:
  void build(Fusion* fusion);

  void setDoubleBufferAxis(const TensorView* tv, IterDomain* id);

  IterDomain* getDoubleBufferAxis(const TensorView* tv);

  //! Get a loop that matches with a given double-buffer axis. If
  //! ignore_prologue is true, a matched loop is ignored if it's a
  //! prologue loop.
  static kir::ForLoop* getDoubleBufferLoop(
      IterDomain* axis,
      const std::vector<kir::ForLoop*>& loops,
      bool ignore_prologue = false);

  //! Get a loop that matches with the double-buffer axis of a given
  //! double-buffered tensor. If ignore_prologue is true, a matched
  //! loop is ignored if it's a prologue loop.
  kir::ForLoop* getDoubleBufferLoop(
      const TensorView* tv,
      const std::vector<kir::ForLoop*>& loops,
      bool ignore_prologue = false);

  void setOriginalAllocSize(const TensorView* tv, Val* size);

  Val* getOriginalAllocSize(const TensorView* tv);

  //! Returns true if the iterdomain will be realized
  //!  as a double buffer loop.
  bool isDoubleBufferedIterDomain(IterDomain* id);

 private:
  TvInfo& getTvInfo(const TensorView* tv);

 private:
  //! Keeps track of information for lowering double buffered tensors
  std::unordered_map<const TensorView*, TvInfo> map_;

  //! Keeps track of which concrete loop map is realizing double buffer
  //!  iterdomains.
  std::unordered_set<const IterDomain*> concrete_double_buffered_loop_id_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
