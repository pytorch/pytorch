#include <cstdint>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/GroupMM.h>
#include <ATen/ceil_div.h>

#ifdef USE_MSLK
#include <mslk/gemm/gemm_torch.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace at::cuda::scaled {

static bool _scaled_mm_allowed_device(bool sm90_only=false, bool sm100_only=false) {
#ifdef USE_ROCM
    static const std::vector<std::string> archs = {
        "gfx942",
#if ROCM_VERSION >= 60300
        "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 60500
        "gfx950"
#endif
    };
    return at::detail::getCUDAHooks().isGPUArch(archs);
#else
    auto dprops = at::cuda::getCurrentDeviceProperties();

    if (sm90_only || sm100_only) {
      return (sm90_only && dprops->major == 9) || (sm100_only && dprops->major == 10);
    } else {
      return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
    }
#endif
}

#ifdef USE_ROCM
static bool _scaled_mm_is_fnuz() {
    return at::detail::getCUDAHooks().isGPUArch({"gfx942"});
}
#endif
/**
 * Track concrete implementations available
 */
enum class ScaledGemmImplementation {
  NONE = 0,
  TENSORWISE_TENSORWISE = 1,
  ROWWISE_ROWWISE = 2,
  BLOCK_128x128_1x128 = 3,
  BLOCK_1x128_128x128 = 4,
  BLOCK_1x128_1x128 = 5,
  MXFP8_MXFP8 = 6,
  NVFP4_NVFP4 = 7,
  NVFP4_NVFP4_SINGLE_SCALE = 8,
  MXFP4_MXFP4 = 9,
};

/**
 * Convert passed int (enum) from python back into a
 * strictly-typed enum
 */
template <class EnumType, class ArrayType>
std::vector<EnumType> convert_int_to_enum(ArrayType& v) {
  std::vector<EnumType> converted;
  converted.reserve(v.size());

  for (auto vi : v) {
    converted.push_back(static_cast<EnumType>(vi));
  }
  return converted;
}

bool check_tensorwise_recipe(c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&,
                             c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&);


bool check_rowwise_recipe(c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&,
                             c10::ScalarType,
                             std::vector<ScalingType>&,
                             ArrayRef<Tensor>&);

bool check_nvfp4_recipe(c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

bool check_nvfp4_recipe_single_scale
                       (c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

bool check_deepseek_recipe(ScalingType,
                           ScalingType,
                           c10::ScalarType,
                           std::vector<ScalingType>&,
                           ArrayRef<Tensor>&,
                           c10::ScalarType,
                           std::vector<ScalingType>&,
                           ArrayRef<Tensor>&);

bool check_mxfp8_recipe(c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

bool check_mxfp4_recipe(c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&,
                        c10::ScalarType,
                        std::vector<ScalingType>&,
                        ArrayRef<Tensor>&);

} // namespace at::native::cuda::blas::scaled
