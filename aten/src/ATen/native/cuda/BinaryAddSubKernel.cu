#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>

// TODO: update to use lazynvrtc
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/jit/resource_guard.h>
#include <sstream>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t>
struct AddFunctor {
  AddFunctor(scalar_t a): alpha(a) {}
  __device__ __forceinline__ scalar_t operator() (const scalar_t a, const scalar_t b) const {
    return a + alpha * b;
  }
  private:
    scalar_t alpha;
};

void add_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
    AddFunctor<scalar_t> f(alpha_scalar.to<scalar_t>());
    gpu_kernel_with_scalars(iter, f);
  });
}

static void sub_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  add_kernel_cuda(iter, -alpha_scalar);
}

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);

// TODO: update this
static void getMajorMinor(

  const cudaDeviceProp* const prop,
  int& major,
  int& minor) {
int nvrtc_major, nvrtc_minor;
AT_CUDA_NVRTC_CHECK(at::globalContext().getNVRTC().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

// Short-circuits if NVRTC version too low
AT_ASSERT(nvrtc_major >= 6);

// Major and minor is determined by device properties and
// possibly "downcompiled" to a lower (compatible) compute architecture
// based on the NVRTC version
major = prop->major;
minor = prop->minor;
if (nvrtc_major <= 7 && prop->major > 5) { // 7 supports 2-5.x
  major = 5;
  minor = 0;
} else if (nvrtc_major <= 8 && prop->major > 6) { // 8 supports 2-6.x
  major = 6;
  minor = 0;
} else if (nvrtc_major <= 9 && prop->major >= 7) { // 9 supports 3-7.2
  major = 7;
  if (prop->major == 7 && prop->minor <= 2)
    minor = prop->minor;
  else
    minor = 0;
} else if (nvrtc_major <= 10 && prop->major >= 7) { // 10 supports 3-7.5
  major = 7;
  if (prop->major == 7 && prop->minor <= 5)
    minor = prop->minor;
  else
    minor = 0;
}
}

Tensor foo_cuda(const Tensor& self, const Tensor& other) {
  // return at::empty({0}, self.options);

  // NOTE: may need/want to initialize CUDA context here (refactor into nvrtc call)

  //void* out, void* a, void* b
  // TODO: provide code (a std::string)
  const std::string name{"foo_kernel"};
  const std::string code{R"foo(
  extern "C" __global__
  void foo_kernel(void* out, void* a, void* b) {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //   printf("%f\n", a);
    //   printf("%i\n", b);
    //   printf("%f\n", ((float*)ptr)[0]);
    // }
    float* out_float = static_cast<float*>(out);
    float* a_float = static_cast<float*>(a);
    float* b_float = static_cast<float*>(b);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      *out_float = *a_float + *b_float;
    }
  })foo"};

  // Acquires device and NVRTC properties (for compile arch and occupancy
  // calculations)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop, major, minor);

  // Creates the NVRTC program
  nvrtcProgram program;
  const auto& nvrtc = at::globalContext().getNVRTC();
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

  // constructs nvrtc arguments
  const std::string compute = "--gpu-architecture=compute_" +
    std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> build_args = {
    "--std=c++14", compute.c_str(), "-default-device"};

  const auto result =
        nvrtc.nvrtcCompileProgram(program, build_args.size(), build_args.data());

  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLogSize(program, &logsize));
    std::vector<char> log(logsize);
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLog(program, log.data()));
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }

  CUmodule module;
  CUfunction function;
  // Note: this should probably not be checked to avoid throwing in a destructor
  ::torch::jit::ResourceGuard holdProgram(
    [&] { AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcDestroyProgram(&program)); });
  AT_CUDA_NVRTC_CHECK(result);
  std::vector<char> ptx;
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTX(program, ptx.data()));

  AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&module, ptx.data()));
  AT_CUDA_DRIVER_CHECK(
    nvrtc.cuModuleGetFunction(&function, module, name.c_str()));

  int maxBlocks;
  AT_CUDA_DRIVER_CHECK(nvrtc.cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, function, 128, 0));
  maxBlocks *= prop->multiProcessorCount;

  // const auto nBlocks = std::min(maxBlocks_, ceilDiv(numel, kBlockSize));
  const int nBlocks = 1;

  // Packs arguments into void**
  auto out_tensor = at::empty({1}, self.options());
  std::cout << out_tensor << std::endl;

  // std::cout << "typeid(data_ptr): " << typeid(self.data_ptr()).name() << std::endl;
  // std::cout << "self.data_ptr: " << self.data_ptr() << std::endl;

  void* out_ptr = out_tensor.data_ptr();
  void* self_ptr = self.data_ptr();
  void* other_ptr = other.data_ptr();

  std::vector<void*> args{(void*)&out_ptr, (void*)&self_ptr, (void*)&other_ptr};
  std::cout << "args_size: " << args.size() << std::endl;

  constexpr int32_t kBlockSize = 128;

  // Launches kernel on current stream (device was set by executor)
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_DRIVER_CHECK(nvrtc.cuLaunchKernel(
    function,
    nBlocks,
    1,
    1,
    kBlockSize,
    1,
    1,
    0,
    stream,
    args.data(),
    nullptr));

  return out_tensor;
}

}} // namespace at::native
