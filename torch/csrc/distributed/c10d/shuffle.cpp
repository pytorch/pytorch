#include <ATen/ATen.h>
#include <torch/library.h>

#ifdef USE_CUDA
void fsdpAllGatherCopyOut(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize);
#endif

namespace {

void fsdp_all_gather_copy_out(
    std::vector<at::Tensor> params,
    at::Tensor all_gather_res,
    int64_t world_size) {
#ifdef USE_CUDA
  return fsdpAllGatherCopyOut(params, all_gather_res, world_size);
#else
  C10_THROW_ERROR(NotImplementedError, "Not implemented for CPU");
#endif
}

} // namespace

TORCH_LIBRARY_FRAGMENT(c10d, m) {
  /**
   * An optimized kernel for FSDP all_gather copy out.
   *
   * `allGatherRes` is a 3D jagged tensor of shape (world_size, num_features,
   * shard_length*) where feature_lengths is a jagged dim. The kernel logically
   * performs the following operations:
   *
   * - Transpose allGatherRes into (num_features, world_size, shard_length*)
   * - Reshape the transposed tensor into (num_features, feature_length*)
   * - Split reshaped the reshaped tensor by feature_lengths
   * - Copy each split into each tensor in `params`
   *
   * This op is dtype agnostic and performs byte-wise copy.
   *
   * - `params[i]`'s dtype must be the same
   */
  m.def(
      "fsdp_all_gather_copy_out("
      "Tensor[] params, Tensor all_gather_res, int world_size) -> ()",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::fsdp_all_gather_copy_out),
      {at::Tag::pt2_compliant_tag});
}
