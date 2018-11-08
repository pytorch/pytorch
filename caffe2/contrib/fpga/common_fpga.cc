#include "common_fpga.h"
#include "context.h"

std::string FPGAEngine::name = "FPGA";

namespace caffe2 {

// Copy from CPU(float32) to OpenCL(bfloat16)
// Note that to Caffe2 the tensors have to still be of type float
// The FPGA only supports matrices of sizes 2^(N+9) for N[0...], so if
// we get smaller inputs, we allocate a buffer of the next smallest valid size
template <>
void OpenCLContext::Copy<float, bfloat16, CPUContext, OpenCLContext>(
    const Tensor& src,
    Tensor& dst) {
  // Read from CPU to OpenCL
  // Read FLOAT32 -> output to BFLOAT16
  auto& ctx = *GetSingleton(engine_);
  dst.Resize(src.sizes());

  size_t n = src.numel();

  dst.template mutable_data<float>();
  src.template data<float>();

  // Create temporary buffer
  bfloat16* tmp = new bfloat16[n];
  assert(tmp);
  const float* srcf = static_cast<const float*>(src.template data<float>());

  for (auto i = 0; i < n; i++) {
    tmp[i] = *(unsigned int*)(&srcf[i]) >> 16;
  }

  // use last queue to write
  ctx.queues.back().enqueueWriteBuffer(
      *((const cl::Buffer*)dst.template mutable_data<float>()),
      true,
      0,
      n * sizeof(bfloat16),
      tmp);

  delete[] tmp;
}

// Copy from OpenCL(bfloat16) to CPU(float32)
// Note that to Caffe2 the tensors have to still be of type float
template <>
void OpenCLContext::Copy<bfloat16, float, OpenCLContext, CPUContext>(
    const Tensor& src,
    Tensor& dst) {
  auto& ctx = *GetSingleton(engine_);
  // Read from OpenCL to CPU
  // Read BFLOAT16 -> output to FLOAT32 CPU
  dst.Resize(src.sizes());
  size_t n = src.numel();

  dst.template mutable_data<float>();
  src.template data<float>();

  bfloat16* tmp = new bfloat16[n];
  assert(tmp);
  bool allocated = (tmp != nullptr);
  CAFFE_ENFORCE_EQ(allocated, true);

  // use first queue to read
  ctx.queues.begin()->enqueueReadBuffer(
      *((const cl::Buffer*)src.template data<float>()),
      true,
      0,
      n * sizeof(bfloat16),
      tmp);

  float* dst_f = dst.template mutable_data<float>();
  union bfp_converter x;
  x.bfp[0] = 0;
  for (int i = 0; i < n; i++) {
    x.bfp[1] = tmp[i];
    dst_f[i] = x.fp32;
  }
  delete[] tmp;
}

} // namespace caffe2
