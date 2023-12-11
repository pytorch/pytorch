#include <torch/library.h>

void fsdpCopyOut(
    std::vector<at::Tensor> outputs,
    at::Tensor input,
    int64_t worldSize);

namespace {

void fsdp_copy_out(
    std::vector<at::Tensor> outputs,
    at::Tensor input,
    int64_t world_size) {
  return fsdpCopyOut(outputs, input, world_size);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(c10d, m) {
  m.def(
      "fsdp_copy_out(Tensor[] outputs, Tensor input, int world_size) -> ()",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::fsdp_copy_out),
      {at::Tag::pt2_compliant_tag});
}
