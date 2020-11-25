#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>

// TODO: update to use lazynvrtc
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/jit/resource_guard.h>
#include <sstream>
#include <torch/csrc/jit/frontend/code_template.h>

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
// stringify here?

void add_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  // stringify here?
  // create template here
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
    // NOTE: we don't need compile-time switching this does at all, so maybe use alternative?
    // Question: is instantiating worthwhile vs. just recompiling?
      // Cons of recompilation: string manipulation done every time
      // Cons of recompilation: need your own code template
      // Cons of instantiation: complicated
    // instantiate dispatched scalar types using the template here
    // this happens at runtime before the call
    // cache whether instantiated or not
    // call
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
    if (prop->major == 7 && prop->minor <= 5) {
      minor = prop->minor;
    } else {
      minor = 0;
    }
  }
}

namespace {

// Host-side version of the Tensor Accessor struct
struct TensorAccessor {
  // NOTE: strides must be in bytes, not elements!
  TensorAccessor(
      const ScalarType scalar_type,
      const int64_t _element_size,
      const IntArrayRef shape,
      const IntArrayRef strides,
      void* _data)
      : element_size_(_element_size),
        ndims_(shape.size()),
        data_{static_cast<char*>(_data)} {

    // TODO: improve this CUDA-compatible scalar type passing
    if (scalar_type == kFloat) {
      scalar_type_ = 0;
    } else if (scalar_type == kDouble) {
      scalar_type_ = 1;
    }

    // TODO: is there a better way to acquire and pass these to the device?
    std::copy(shape.cbegin(), shape.cend(), std::begin(sizes_));
    std::copy(strides.cbegin(), strides.cend(), std::begin(strides_));
  }

  short scalar_type_;
  short element_size_;
  short ndims_;
  int sizes_[25];
  // NOTE: strides is in bytes, not elements!
  int strides_[25];
  char* data_;
};

#define stringify(...) std::string(#__VA_ARGS__); __VA_ARGS__
const auto jittable_foo_functor = stringify(
  template<typename scalar_t>
  struct FooFunctor {
    FooFunctor(scalar_t a): alpha{a} {}
    __device__ __forceinline__ scalar_t operator() (const scalar_t a, const scalar_t b) const {
      return a + alpha * b;
    }

    scalar_t alpha;
  };
);
#undef stringify

// TODO: create a stringify-like macro so this looks like C++
//   but produces the appropriate type with newlines
//   NOTE: this probably requires creating macros for the template inserts, too
static auto cuda_template = torch::jit::CodeTemplate(R"(
  typedef long long int int64_t;

  template <typename T>
  struct DivMod {
    T div;
    T mod;

    __device__ DivMod(T _div, T _mod) {
      div = _div;
      mod = _mod;
    }
  };

  //<unsigned int>
  struct IntDivider {
    IntDivider() = default;

  __device__ inline unsigned int div(unsigned int n) const {
    unsigned int t = __umulhi(n, m1);
    return (t + n) >> shift;
  }

  __device__ inline unsigned int mod(unsigned int n) const {
    return n - div(n) * divisor;
  }

  __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
    unsigned int q = div(n);
    return DivMod<unsigned int>(q, n - q * divisor);
  }

  unsigned int divisor;  // d above.
  unsigned int m1;  // Magic number: m' above.
  unsigned int shift;  // Shift amounts.
};

  struct OffsetCalculator {
    OffsetCalculator() = default;
    __device__ void index_to_offset(${index_type} offsets[${nInputs}], ${index_type} linear_idx) const {
      #pragma unroll
      for (int arg = 0; arg < ${nInputs}; ++arg) {
        offsets[arg] = 0;
      }

      #pragma unroll
      for (int dim = 0; dim < 25; ++dim) {
        if (dim == dims) {
          break;
        }

        auto divmod = sizes_[dim].divmod(linear_idx);
        linear_idx = divmod.div;

        #pragma unroll
        for (int arg = 0; arg < ${nInputs}; ++arg) {
          offsets[arg] += divmod.mod * strides_[dim][arg];
        }
      }
    }

    int dims;
    IntDivider sizes_[25];
    // NOTE: this approach will not support nInputs == 0
    ${index_type} strides_[25][${nInputs}];
  };

  ${functor}

  // NOTE: assumes the op is binary (i.e. has three arguments out, a, and b)
  // TODO: setup grid-stride loop
  extern "C" __global__
  void ${name}_kernel(
      ${name}<${scalar_type}> functor,
      const int numel,
      char* data,
      OffsetCalculator input_calculator,
      OffsetCalculator output_calculator) {


    // NOTE: only the first thread operates on the first element for now
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      // ${scalar_type} a_value;
      // int a_offset = a.index_to_offset(0);

      // ${scalar_type} b_value;
      // int b_offset = b.index_to_offset(0);

      // int out_offset = out.index_to_offset(0);

      // // TODO: refactor the loading, see c10::fetch_and_cast
      // if (a.scalar_type_ == 0) {
      //   a_value = static_cast<${scalar_type}>(*(reinterpret_cast<float*>(a.data_ + a_offset)));
      // } else if (a.scalar_type_ == 1) {
      //   a_value = static_cast<${scalar_type}>(*(reinterpret_cast<double*>(a.data_ + a_offset)));
      // }

      // if (b.scalar_type_ == 0) {
      //   b_value = static_cast<${scalar_type}>(*(reinterpret_cast<float*>(b.data_ + b_offset)));
      // } else if (b.scalar_type_ == 1) {
      //   b_value = static_cast<${scalar_type}>(*(reinterpret_cast<double*>(b.data_ + b_offset)));
      // }

      // ${scalar_type} out_value = functor(a_value, b_value);

      // // TODO: refactor the storing, see c10::cast_and_store
      // if (out.scalar_type_ == 0) {
      //   *(reinterpret_cast<float*>(out.data_ + out_offset)) = static_cast<float>(out_value);
      // } else if (out.scalar_type_ == 1) {
      //   *(reinterpret_cast<double*>(out.data_ + out_offset)) = static_cast<double>(out_value);
      // }

      // printf("%f\n", out_value);
    }
  }

// instantiations here
)");

static int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

} // anonymous namespace

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define THREAD_WORK_SIZE 4
#define BLOCK_WORK_SIZE (THREAD_WORK_SIZE * num_threads)

Tensor foo_cuda(const Tensor& self, const Tensor& other, Scalar alpha_scalar) {
  TORCH_CHECK(self.scalar_type() == kFloat);
  TORCH_CHECK(other.scalar_type() == kDouble);

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);

  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  std::cout << "dtype 0: " << iter.dtype(0) << std::endl;
  std::cout << "dtype 1: " << iter.dtype(0) << std::endl;
  std::cout << "dtype 2: " << iter.dtype(0) << std::endl;
  std::cout << "iter.tensor(0).scalar_type(): " << iter.tensor(0).scalar_type() << std::endl;
  std::cout << "iter.tensor(1).scalar_type(): " << iter.tensor(1).scalar_type() << std::endl;
  std::cout << "iter.tensor(2).scalar_type(): " << iter.tensor(2).scalar_type() << std::endl;
  std::cout << "common_dtype: " << iter.common_dtype() << std::endl;

  std::cout << "jittable functor string" << std::endl;
  std::cout << jittable_foo_functor << std::endl;


  const auto output_dtype = iter.tensor(0).scalar_type();
  const int32_t output_dtype_as_int = static_cast<int32_t>(output_dtype);
  std::cout << "output_dtype_as_int: " << output_dtype_as_int << std::endl;

  // Constructs kernel args
  std::vector<void*> args;

  // Creates functor arg
  // TODO: refactor with dispatch macro?
  // TODO: support float or double dynamically
  FooFunctor<double> my_functor{alpha_scalar.to<double>()};
  args.push_back((void*)&my_functor);

  // Adds numel arg
  // NOTE: the intermediate capture is neccessary
  int64_t numel = iter.numel();
  args.push_back((void*)&numel);

  // Adds data ptrs
  at::detail::Array<char*, 3> data;
  for (auto i = decltype(iter.ntensors()){0}; i < iter.ntensors(); i++) {
    data[i] = (char*)iter.data_ptr(i);
  }
  args.push_back((void*)&data);

  // Addds offset calculators
  // TODO: maybe combine into one offset calculator?
  auto input_offset_calculator = make_input_offset_calculator<2>(iter);
  auto output_offset_calculator = make_output_offset_calculator(iter);
  args.push_back((void*)&input_offset_calculator);
  args.push_back((void*)&output_offset_calculator);

  // Constructs kernel code
  torch::jit::TemplateEnv env;
  env.s("name", "FooFunctor");
  env.s("functor", jittable_foo_functor);
  env.s("index_type", "unsigned int");
  env.s("nInputs", "2");

  // Identifies scalar type
  // TODO: there has to be an existing way of doing this (i.e. converting scalar type to string)
  const auto& common_dtype = iter.common_dtype();
  std::string common_dtype_string;
  if (common_dtype == kFloat) {
    common_dtype_string = "float";
  } else if (common_dtype == kDouble) {
    common_dtype_string = "double";
  }
  env.s("scalar_type", common_dtype_string);

  std::string code = cuda_template.format(env);
  std::cout << "code: " << code << std::endl;

  // Compiles kernel
  // Acquires device and NVRTC properties (for compile arch and occupancy calculations)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop, major, minor);

  // Creates the NVRTC program
  nvrtcProgram program;
  const auto& nvrtc = at::globalContext().getNVRTC();
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

  // constructs nvrtc build arguments
  const std::string compute = "--gpu-architecture=compute_" +
    std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> build_args = {
    "--std=c++14", compute.c_str(), "-default-device"};

  // compiles and validates result
  const auto compilation_result =
        nvrtc.nvrtcCompileProgram(program, build_args.size(), build_args.data());
  if (compilation_result != NVRTC_SUCCESS) {
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
  ::torch::jit::ResourceGuard holdProgram([&] { nvrtc.nvrtcDestroyProgram(&program); });
  std::vector<char> ptx;
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTX(program, ptx.data()));

  AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&module, ptx.data()));
  const std::string kernel_name = "FooFunctor_kernel";
  AT_CUDA_DRIVER_CHECK(
    nvrtc.cuModuleGetFunction(&function, module, kernel_name.c_str()));

  // Computes blocks and block size
  // TODO: review this block computation vs cuda loops
  int maxBlocks;
  AT_CUDA_DRIVER_CHECK(nvrtc.cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, function, 128, 0));
  maxBlocks *= prop->multiProcessorCount;

  constexpr int32_t kBlockSize = 128;
  const auto nBlocks = std::min(maxBlocks, ceilDiv(numel, kBlockSize));

  // Launches kernel on current stream
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

  return iter.output();

  // NOTE: may need/want to initialize CUDA context here (refactor into nvrtc call)

  // void* out, void* a, void* b
  // TODO: provide code (a std::string)
  // const std::string name{"foo_kernel"};
  // const std::string code{R"foo(
  // extern "C" __global__
  // void foo_kernel(void* out, void* a, void* b) {
  //   // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   //   printf("%f\n", a);
  //   //   printf("%i\n", b);
  //   //   printf("%f\n", ((float*)ptr)[0]);
  //   // }
  //   float* out_float = static_cast<float*>(out);
  //   float* a_float = static_cast<float*>(a);
  //   float* b_float = static_cast<float*>(b);

  //   if (blockIdx.x == 0 && threadIdx.x == 0) {
  //     *out_float = *a_float + *b_float;
  //   }
  // })foo"};

  // OLD AND EXPERIMENTAL CODE BELOW HERE

  // // Constructs accessor arguments
  // std::vector<std::string> accessor_strings;
  // int accessor_name_counter = 65;
  // for (const auto& accessor : accesors) {
  //   torch::jit::TemplateEnv local;
  //   local.s("tensor_name", static_cast<char>(accessor_name_counter++));
  //   accessor_strings.emplace_back(torch::jit::format(", TensorAccessor ${tensor_name}", env);
  // }

  // env.v("tensor_accessors", accessor_strings);
}

}} // namespace at::native
