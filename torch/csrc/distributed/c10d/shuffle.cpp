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
  m.def(
      "fsdp_all_gather_copy_out("
      "Tensor[] params, Tensor all_gather_res, int world_size) -> ()",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::fsdp_all_gather_copy_out),
      {at::Tag::pt2_compliant_tag});
}
