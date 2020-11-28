#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>

// TODO: update to use lazynvrtc
#include <ATen/cuda/detail/LazyNVRTC.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/jit/resource_guard.h>
#include <sstream>
#include <torch/csrc/jit/frontend/code_template.h>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <mutex>

namespace at { namespace native {
namespace {

// TODO jiterator cache design does not handle multiple gpus currently
using JiteratorKey = ScalarType;
using JiteratorCache = std::unordered_map<JiteratorKey, CUfunction>;

// global jiterator mutex
// TODO: currently caches are per function but the mutex is global,
//   so maybe mutexes should be per function, too, or the caches should
//   be consolidated
std::mutex jiterator_mutex;

JiteratorKey construct_jiterator_key(const ScalarType scalar_type) {
  return scalar_type;
}

// NOTE: get does not acquire the lock
c10::optional<CUfunction> get_jitted_function(const JiteratorCache& cache, JiteratorKey key) {
  auto it = cache.find(key);
  if (it == cache.end()) {
    return c10::nullopt;
  }
  return it->second;
}

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

void store_jitted_function(
    JiteratorCache& cache,
    const JiteratorKey key,
    CUfunction function) {
  cache.emplace(key, function);
}

constexpr int num_threads = 64;
constexpr int thread_work_size = 4; //TODO make template substitution once we decide where those vars live
constexpr int block_work_size = thread_work_size * num_threads;

CUfunction jit_binary_pwise_function(
    JiteratorCache& cache,
    JiteratorKey key,
    const std::string& code,
    const std::string& kernel_name) {

  // TODO: this lock is could be acquired around the cache updates
  std::lock_guard<std::mutex> guard{jiterator_mutex};

  // Compiles the kernel ---

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
  std::vector<char> ptx;
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTX(program, ptx.data()));
  AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&module, ptx.data()));
  AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleGetFunction(&function, module, kernel_name.c_str()));


  // Updates (or not) the cache and returns the function ---
  c10::optional<CUfunction> maybe_function = get_jitted_function(cache, key);
  if (maybe_function) {
    // Destroys the just compiled but unneccessary program
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcDestroyProgram(&program));
    return *maybe_function;
  }

  store_jitted_function(cache, key, function);
  return function;
}

// TODO: may need/want to initialize CUDA context here (refactor into nvrtc call)
void launch_jitted_binary_pwise_function(
    CUfunction function,
    std::vector<void*>& args,
    const int nBlocks,
    const int kBlockSize) {

  const auto& nvrtc = at::globalContext().getNVRTC();

  // TODO: seems like this and block calculation should be cached per device
  // Acquires device and NVRTC properties (for compile arch and occupancy calculations)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop, major, minor);


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
}

// Creates a string from the scalar type and lowercases it to produce the c-type
// TODO: this works for float and double, but a more general solution is needed
std::string scalartype_to_type_string(const ScalarType scalar_type) {
  std::string s{c10::toString(scalar_type)};
  std::transform(
    s.cbegin(),
    s.cend(),
    s.begin(),
    [](unsigned char c){ return std::tolower(c); });
  return s;
}

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



JiteratorCache foo_cache;

Tensor foo_cuda(const Tensor& self, const Tensor& other, Scalar alpha_scalar) {
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

  // std::cout << "jittable functor string" << std::endl;
  // std::cout << jittable_foo_functor << std::endl;

  // Constructs kernel args
  std::vector<void*> args;

  // Creates functor arg
  // TODO: refactor with dispatch macro?
  // TODO: support float or double dynamically
  FooFunctor<float> my_functor{alpha_scalar.to<float>()};
  args.push_back((void*)&my_functor);

  // Adds numel arg
  // NOTE: the intermediate capture is neccessary
  const int64_t numel = iter.numel();
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
  const int nInputs = iter.ninputs();
  torch::jit::TemplateEnv env;
  env.s("name", "FooFunctor");
  env.s("functor", jittable_foo_functor);
  env.s("index_type", "unsigned int");
  env.s("nInputs", std::to_string(nInputs));
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
  std::stringstream declare_load_arrays;
  for (int i=0; i < nInputs; i++){
//TODO these arrays are potentially of the different types, use function traits to determine the types
    declare_load_arrays << common_dtype_string << " arg" << std::to_string(i) << "[" << std::to_string(thread_work_size) << "];\n";
  }
  env.s("declare_load_arrays", declare_load_arrays.str());
  std::stringstream  load_inputs;
  for (int i=0; i < nInputs; i++){
    load_inputs << "arg" << std::to_string(i) << "[j] = *(reinterpret_cast<" << common_dtype_string << "*>(data[" <<
    std::to_string(i + iter.noutputs()) << "]) + input_offsets[" << std::to_string(i) << "]);\n";
  }
  env.s("load_inputs", load_inputs.str());
  std::stringstream functor_args;
  for (int i=0; i < nInputs - 1; i++){
    functor_args << "arg" << std::to_string(i) << "[j], ";
  }
  functor_args << "arg" << std::to_string(nInputs-1) << "[j]";
  env.s("args", functor_args.str());

  cuda_template = at::cuda::detail::load_code_template("/private/home/ngimel/pytorch/aten/src/ATen/native/cuda/code_template.cuh");

  std::string code = cuda_template.format(env);
//  std::cout << "code: \n" << code << std::endl;

  JiteratorKey key = construct_jiterator_key(iter.common_dtype());
  c10::optional<CUfunction> maybe_function = get_jitted_function(foo_cache, key);
  CUfunction function;
  if (maybe_function) {
    std::cout << "found function" << std::endl;
    function = *maybe_function;
  } else {
    std::cout << "jitting function" << std::endl;
    // TODO: make kernel name generic
    const std::string kernel_name{"FooFunctor_kernel"};
    function = jit_binary_pwise_function(foo_cache, key, code, kernel_name);
  }

  int64_t grid = (numel + block_work_size - 1) / block_work_size;

  launch_jitted_binary_pwise_function(function, args, grid, num_threads);

  return iter.output();
}

}} // namespace at::native
