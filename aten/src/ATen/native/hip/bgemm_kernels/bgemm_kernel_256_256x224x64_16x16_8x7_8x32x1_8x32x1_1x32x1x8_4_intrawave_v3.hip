#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/bgemm_kernels/bgemm_kernel_template.h>

namespace at::native {

void bgemm_kernel_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_intrawave_v3(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {

  using BGemmInstance = DeviceBGemmInstance<
      256,
      256,
      224,
      64,
      16,
      16,
      8,
      7,
      S<8, 32, 1>,
      S<8, 32, 1>,
      S<1, 32, 1, 8>,
      S<4>,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3,
      ck::tensor_operation::device::GemmSpecialization::Default>;

  // Run kernel instance.
  bgemm_kernel_impl<BGemmInstance>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}

};
