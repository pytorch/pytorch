#pragma once

#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <vector>

/*
 * Index compute takes in a list of indices typically generated from the
 * surrounding for loop nest. The number of indicies are intended to match the
 * number of dimensions of the incomming TensorView which may have less or more
 * dimensions than its root due to split/merge/reorder operations.
 * Split/merge/reorder operations are then replayed backwards produce resulting
 * indices (based on input indices) that match the root dimension.
 *
 * For example with GLOBAL tensor:
 * TV[I, J]
 * TV[Io, Ii{4}, J] = TV.split(I, factor=4)
 * ALLOC: NONE
 * INDEX: indexCompute {i, j, k} -> {i * 4 + j, k}
 * FLATTENED_INDEX: {i * 4 + j, k} -> {i * 4 + j * J + k}
 * PREDICATE: {i * 4 + j, k} -> i * 4 + j < I
 *
 *
 * For example with SHARED tensor:
 *
 * global_TV[I, J]
 * global_TV[Io, Ii{4}, J] = global_TV.split(I, factor=4)
 * smem_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 *
 * ALLOC: alloc(smem_TV, 4 x J)
 * INDEX: indexCompute(smem_TV, {threadIdx.x, k}) -> {threadIdx.x, k}
 * FLATTENED_INDEX: {threadIdx.x * 4 + j, k} -> {threadIdx.x * 4 + j * J + k}
 * PREDICATE: {threadIdx.x * 4 + j, k} -> threadIdx.x * 4 + j < I // Same as if
 * global
 *
 *
 * For example with LOCAL tensor:
 * global_TV[I, J, K]
 * global_TV[Io, Ii{4}, J] = global_TV.split(I, factor=4)
 * reg_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 * global_TV{i, j, k, l} -> { i * 4 + j, k, l }
 * global_TV{ i * 4 + j, k, l } -> { i * 4 + j * J * K  +  k * K  +  l}
 *
 * ALLOC: alloc(reg_TV, J x K)
 * INDEX: {k, l} -> {k, l}
 * FLATTENED_INDEX: {k, l} -> {k * J + l}
 * PREDICATE: i * 4 + j < I && k < J && l < K ->  // Same as if global
 *
 * These indices can then be flattened later based on strides.
 */

namespace torch {
namespace jit {
namespace fuser {

struct IndexCompute : public TransformIter {
 private:
  TensorDomain* replayBackward(Split*, TensorDomain*) override;
  TensorDomain* replayBackward(Merge*, TensorDomain*) override;
  TensorDomain* replayBackward(Reorder*, TensorDomain*) override;

  TensorDomain* runBackward(std::vector<Expr*> history);

  // Otherwise warning on runBackward as it hides an overloaded virtual function
  using TransformIter::runBackward;

  IndexCompute(TensorDomain* td, std::vector<Val*> _indices);
  std::vector<Val*> indices;

 public:
  static std::vector<Val*> get(
      TensorDomain* td,
      const std::vector<Val*>& _indices);
};

// Simple interface for IndexCompute
struct Index : public TransformIter {
 private:
  // Producer indexing if it's in shared or local memory
  static TensorIndex* getProducerIndex_impl(
      TensorView* producer,
      TensorView* consumer,
      const std::vector<ForLoop*>& loops);

  // Consumer indexing if it's in shared or local memory
  static TensorIndex* getConsumerIndex_impl(
      TensorView* consumer,
      const std::vector<ForLoop*>& loops);

 public:
  // Producer if it's in global memory
  static TensorIndex* getGlobalProducerIndex(
      TensorView* producer,
      TensorView* consumer,
      const std::vector<ForLoop*>& loops);

  // Consumer indexing if it's in global memory
  static TensorIndex* getGlobalConsumerIndex(
      TensorView* consumer,
      const std::vector<ForLoop*>& loops);

  // Indexing functions
  // Consumer = Producer
  // i.e. T0 = T1... -> T0 is the consumer, T1 is the producer
  // Producer indexing dispatch
  static TensorIndex* getProducerIndex(
      TensorView* producer,
      TensorView* consumer,
      const std::vector<ForLoop*>& loops);

  // Consumer index dispatch
  static TensorIndex* getConsumerIndex(
      TensorView* consumer,
      const std::vector<ForLoop*>& loops);

  // Will run inds through back prop index computation for tv
  static TensorIndex* manualBackprop(TensorView tv, std::vector<Val*> inds);
};

} // namespace fuser
} // namespace jit
} // namespace torch
