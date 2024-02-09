#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

template <typename T>
void stage_pack_weights(
    api::Context* const context,
    vTensor& v_weight,
    const Tensor& weight,
    const int64_t src_kb_sz,
    const int64_t src_kh_sz,
    const int64_t src_kw_sz,
    const int64_t dst_kh_sz,
    const int64_t dst_kw_sz) {
  const int64_t src_matrix_sz = src_kw_sz * src_kh_sz;
  const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;
  const int64_t dst_matrix_sz = dst_plane_sz * 4;
  const T* const src_weight_ptr = weight.data_ptr<T>();
  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    T* dst_weight_ptr = mapping.template data<T>();

    memset(dst_weight_ptr, 0, v_weight.nbytes());

    for (const auto src_b : c10::irange(src_kb_sz)) {
      for (const auto src_h : c10::irange(src_kh_sz)) {
        for (const auto src_w : c10::irange(src_kw_sz)) {
          int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
          int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
          memcpy(
              dst_weight_ptr + src_b * dst_matrix_sz +
                  dst_plane * dst_plane_sz + dst_index,
              src_weight_ptr + src_b * src_matrix_sz + src_h * src_kw_sz +
                  src_w,
              sizeof(T));
        }
      }
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), v_weight);
}

class LinearPackedContext final : virtual public VulkanPackedContext,
                                  public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;

 public:
  LinearPackedContext(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      const bool use_batch = false);

  /*
   * Assigns a name to each index in the unpacked list.
   */
  struct Unpacked final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;

    static constexpr uint32_t NumArgs = 2u;
  };

  /*
   * Assigns a name to each index in the packed list.
   */
  struct Packed final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;
    static constexpr uint32_t WeightSizes = 2u;
    static constexpr uint32_t BiasDefined = 3u;

    static constexpr uint32_t NumArgs = 4u;
  };

  static LinearPackedContext pack(c10::impl::GenericList);

  const c10::impl::GenericList unpack() const override {
    TORCH_CHECK(unpacked_.size() > 0u, "unpacked_ does not have any elements!");

    return unpacked_;
  }
};

c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias);

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& context);

Tensor run_qlinear_context(
    const Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    const c10::intrusive_ptr<LinearPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
