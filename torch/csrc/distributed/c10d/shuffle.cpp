#include <ATen/ATen.h>
#include <torch/library.h>

#ifdef USE_CUDA
void fsdpAllGatherCopyOut(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize);

void fsdpAllGatherCopyOut_no_align(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize);

void fsdpAllGatherCopyOut_no_align_2(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize,
    int64_t warpsPerShard);
#endif

namespace {

static inline int64_t divUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

at::Tensor fsdp_all_gather_copy_in(
    std::vector<at::Tensor> params,
    at::ScalarType dtype) {
  constexpr int64_t BYTES_PER_THREAD = 16;
  const auto elemSize = elementSize(dtype);
  const auto numelPerThread = BYTES_PER_THREAD / elemSize;
  const auto device = params[0].device();

  int64_t alignedTotalNumel = 0;
  std::vector<int64_t> alignedOffsets{0};
  for (size_t i = 0; i < params.size(); ++i) {
    const auto& param = params[i];
    TORCH_CHECK(param.device() == device);
    const auto alignedNumel =
        divUp(param.numel(), numelPerThread) * numelPerThread;
    alignedTotalNumel += alignedNumel;
    alignedOffsets.push_back(alignedOffsets[i] + alignedNumel);
  }

  auto allGatherInput = at::zeros(
      {alignedTotalNumel}, at::TensorOptions().dtype(dtype).device(device));
  for (size_t i = 0; i < params.size(); ++i) {
    at::NoGradGuard no_grad;
    const auto& param = params[i];
    auto view = at::narrow(allGatherInput, 0, alignedOffsets[i], param.numel());
    // TODO: don't materialize the intermediate
    view.copy_(param.to(dtype));
  }
  return allGatherInput;
}

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

void fsdp_all_gather_copy_out_no_align(
    std::vector<at::Tensor> params,
    at::Tensor all_gather_res,
    int64_t world_size) {
#ifdef USE_CUDA
  return fsdpAllGatherCopyOut_no_align(params, all_gather_res, world_size);
#else
  C10_THROW_ERROR(NotImplementedError, "Not implemented for CPU");
#endif
}

void fsdp_all_gather_copy_out_no_align_2(
    std::vector<at::Tensor> params,
    at::Tensor all_gather_res,
    int64_t world_size,
    int64_t warps_per_shard) {
#ifdef USE_CUDA
  return fsdpAllGatherCopyOut_no_align_2(
      params, all_gather_res, world_size, warps_per_shard);
#else
  C10_THROW_ERROR(NotImplementedError, "Not implemented for CPU");
#endif
}

} // namespace

TORCH_LIBRARY_FRAGMENT(c10d, m) {
  m.def(
      "fsdp_all_gather_copy_in(Tensor[] params, ScalarType dtype) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::fsdp_all_gather_copy_in),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "fsdp_all_gather_copy_out(Tensor[] params, Tensor all_gather_res, int world_size) -> ()",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::fsdp_all_gather_copy_out),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "fsdp_all_gather_copy_out_no_align(Tensor[] params, Tensor all_gather_res, int world_size) -> ()",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::fsdp_all_gather_copy_out_no_align),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "fsdp_all_gather_copy_out_no_align_2("
      "Tensor[] params, Tensor all_gather_res, int world_size, int warps_per_shard=4) -> ()",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::fsdp_all_gather_copy_out_no_align_2),
      {at::Tag::pt2_compliant_tag});
}
