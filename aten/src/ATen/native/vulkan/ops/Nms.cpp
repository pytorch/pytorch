#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

namespace at {
namespace native {

Tensor nms_cpu(
    const Tensor& dets_arg,
    const Tensor& scores_arg,
    double iou_threshold) {
  const auto& dets = dets_arg.is_vulkan() ? dets_arg.cpu() : dets_arg;
  const auto& scores = scores_arg.is_vulkan() ? scores_arg.cpu() : scores_arg;
  
  std::cout << "XXX nms dets:" << dets << std::endl;
  std::cout << "XXX nms scores:" << scores << std::endl;
  std::cout << "XXX nms iou_threshold:" << iou_threshold << std::endl;

  // cpu implementation from torchvision
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong));

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();
  using scalar_t = float;
  auto x1 = x1_t.data_ptr<scalar_t>();
  auto y1 = y1_t.data_ptr<scalar_t>();
  auto x2 = x2_t.data_ptr<scalar_t>();
  auto y2 = y2_t.data_ptr<scalar_t>();
  auto areas = areas_t.data_ptr<scalar_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold)
        suppressed[j] = 1;
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor nms_vulkan(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {

  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  auto n_dets = dets.size(0);
  std::cout << "XXX nms_vulkan " << __LINE__ << " n_dets:" << n_dets << std::endl;
  auto scores_cpu = scores.cpu();
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  auto indices_cpu = std::get<1>(scores_cpu.sort(0, /* descending */ true));
  auto indices_float_cpu = indices_cpu.to(indices_cpu.options().dtype(at::kFloat));
  std::cout << "XXX nms_vulkan indices_float_cpu:" << indices_float_cpu << std::endl;
  auto indices_float_vulkan = indices_float_cpu.vulkan();
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;

  api::Context* const context = api::context();
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  vTensor v_suppress{context, {n_dets}, dets.options()};
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;

  const vTensor& v_dets = convert(dets);
 // std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  const vTensor& v_indices = convert(indices_float_vulkan);
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  const struct Block final {
    float iou_threshold;
  } block {
    iou_threshold,
  };
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;

  context->dispatch(
      command_buffer,
      {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      VK_KERNEL(nms),
      {
        n_dets,
        n_dets,
        1
      },
      context->gpu().adapter->local_work_group_size(),
      // Write-only access bypasses synchronization but inserts appropriate
      // barriers if necessary.
      v_suppress.buffer(command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
      // Read-only access is implied on const tensors and triggers an async
      // synchronization if necessary.
      v_dets.buffer(command_buffer, vTensor::Stage::Compute),
      //// Read-only access is implied on const tensors and triggers an async
      //// synchronization if necessary.
      v_indices.buffer(command_buffer, vTensor::Stage::Compute),
      // Object lifetime is managed by the resource pool.
      // It is OK not to keep track of the handle.
      context->resource().pool.uniform(block).object);
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  command_pool.submit(context->gpu().queue, command_buffer);
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  auto suppress_vulkan = convert(v_suppress);
  auto suppress_cpu = suppress_vulkan.cpu();
  std::cout << "XXX suppress_cpu:" << suppress_cpu << std::endl;
  float* suppress_data = suppress_cpu.data_ptr<float>();
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  int num_to_keep = 0;
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  at::Tensor keep = at::empty(
      {n_dets}, 
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  int64_t* keep_data = keep.data_ptr<int64_t>();
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  for (int i = 0; i < n_dets; ++i) {
    if (suppress_data[i] != 1.f) { 
      keep_data[num_to_keep++] = i;
    }
  }
  std::cout << "XXX nms_vulkan " << __LINE__ << std::endl;
  return indices_cpu.index({keep.narrow(0, 0, num_to_keep)});
}

//#ifdef USE_VULKAN_API
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("nms", TORCH_FN(nms_vulkan));
}

//#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
