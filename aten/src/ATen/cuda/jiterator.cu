#include <ATen/cuda/jiterator.h>

#include <ATen/native/TensorIterator.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/code_template.h>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/cuda/llvm_jit_strings.h>

#include <ATen/native/cuda/JitLoops.cuh>

#include <c10/util/variant.h>

#include <iostream>

namespace at {
namespace native {

c10::SmallVector<std::string> get_extra_args_typenames(const std::vector<at::Scalar>& extra_args) {
  c10::SmallVector<std::string> args_typenames(extra_args.size());
  for (auto i = 0; i < extra_args.size(); ++i) {
    args_typenames[i] = at::cuda::jit::typeName(extra_args[i].type());
  }
  return args_typenames;
}

struct OffsetCalculatorContainer {
  OffsetCalculatorContainer(const TensorIteratorBase& iter) {
    int N = iter.ninputs();
    switch(N) {
      case 0: v.v0 = make_input_offset_calculator<0>(iter); break;
      case 1: v.v1 = make_input_offset_calculator<1>(iter); break;
      case 2: v.v2 = make_input_offset_calculator<2>(iter); break;
      case 3: v.v3 = make_input_offset_calculator<3>(iter); break;
      default:
        AT_ERROR("make_input_offset_calculator not implemented for ninputs = ", N);
    }
  }
  void* data_ptr() {
    return static_cast<void*>(&v);
  }

private:
  union v_t {
    OffsetCalculator<0> v0;
    OffsetCalculator<1> v1;
    OffsetCalculator<2> v2;
    OffsetCalculator<3> v3;
    v_t() {} // default constructor
  } v;
};

struct ArrayVariant {
  using ArrayTypes = c10::variant<
    at::detail::Array<char*, 2>,
    at::detail::Array<char*, 3>,
    at::detail::Array<char*, 4>>;

  ArrayVariant(const TensorIteratorBase& iter) {
    int N = iter.ntensors();
    switch(N) {
      // jitted kernels must have at least 1 input and 1 output
      case 2: array = at::detail::Array<char*, 2>{}; break;
      case 3: array = at::detail::Array<char*, 3>{}; break;
      case 4: array = at::detail::Array<char*, 4>{}; break;
      default:
        AT_ERROR("ArrayVariant not implemented for ninputs = ", N);
    }

    c10::visit([&](auto& a) {
      for (auto i = 0; i < N; ++i) {
        a[i] = (char*)iter.data_ptr(i);
      }
    }, array);
  }

  void* data_ptr() {
    return c10::visit([](auto & a){ return static_cast<void*>(&a); }, array);
  }

private:
  ArrayTypes array;
};



static void* make_trivial_offset_calculator(int arity) {
  switch(arity) {
    case 0: return static_cast<void*>(new TrivialOffsetCalculator<0>());
    case 1: return static_cast<void*>(new TrivialOffsetCalculator<1>());
    case 2: return static_cast<void*>(new TrivialOffsetCalculator<2>());
    case 3: return static_cast<void*>(new TrivialOffsetCalculator<3>());
    default:
      AT_ERROR("make_trivial_offset_calculator not implemented for ninputs = ", arity);
  }
}

static void* make_load_with_cast(const TensorIteratorBase& iter) {
  int arity = iter.ninputs();
  switch(arity) {
    case 0:
    {
      at::detail::Array<ScalarType, 0> dtypes;
      for (auto i = 0; i < arity; ++i) {
        dtypes[i] = iter.dtype(i + 1);
      }
      return static_cast<void*>(new memory::LoadWithCast<0>(dtypes));
    }
    case 1:
    {
      at::detail::Array<ScalarType, 1> dtypes;
      for (auto i = 0; i < arity; ++i) {
        dtypes[i] = iter.dtype(i + 1);
      }
      return static_cast<void*>(new memory::LoadWithCast<1>(dtypes));
    }
    case 2:
    {
      at::detail::Array<ScalarType, 2> dtypes;
      for (auto i = 0; i < arity; ++i) {
        dtypes[i] = iter.dtype(i + 1);
      }
      return static_cast<void*>(new memory::LoadWithCast<2>(dtypes));
    }
    default:
      AT_ERROR("make_input_offset_calculator not implemented for ninputs = ", arity);
  }
}

static inline void launch_jitted_vectorized_kernel(
  const std::string& name, TensorIteratorBase& iter,
  DeviceIndex dev_idx, int64_t N, const std::string& f, void* data_ptr,
  const std::vector<at::Scalar>& extra_args) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  // N is still int64_t for the computation, but it's always safe to cast result to int
  const uint32_t grid = (N + block_work_size() - 1) / block_work_size();

  // TODO: fix hard code here
  //const int vec_size = memory::jitted_can_vectorize_up_to<result_type, f_inputs_type, arity>(data);
  const int vec_size = 4;

  // Different kernels are compiled depending on what we're vectorizing up to (1, 2 or 4 elements)
  //   fn_ptr is set to the appropriate function based on the vec size and GPU used
  // TODO: Memory use can probably be optimized by re-using kernels across GPUs with
  //   the same compute capability
  static std::mutex _jiterator_mutex;
  static std::vector<at::cuda::jit::NvrtcFunction> fns4(c10::cuda::device_count());
  static std::vector<at::cuda::jit::NvrtcFunction> fns2(c10::cuda::device_count());
  static std::vector<at::cuda::jit::NvrtcFunction> fns1(c10::cuda::device_count());


  at::cuda::jit::NvrtcFunction* fn_ptr;
  if (vec_size == 4) {
    fn_ptr = &fns4[dev_idx];
  } else if (vec_size == 2) {
    fn_ptr = &fns2[dev_idx];
  } else if (vec_size ==1) {
    fn_ptr = &fns1[dev_idx];
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected vec_size for jitter vectorized kernel");
  }

  bool vectorized = vec_size > 1;

  // if (!fn_ptr->function) {
  {
    const std::lock_guard<std::mutex> lock{_jiterator_mutex};
    // if (!fn_ptr->function) { // cache miss!
    {
      // Generates program
      int nTensors =  iter.ntensors();

      const at::ScalarType common_dtype = iter.common_dtype();
      std::string f_inputs_type_str = at::cuda::jit::typeName(common_dtype);
      std::string compute_type_str = at::cuda::jit::typeName(toOpMathType(common_dtype));
      std::string result_type_str = at::cuda::jit::typeName(common_dtype);

      std::cout<<"launch_jitted_vectorized_kernel_raw_ptr"<<std::endl;

      c10::SmallVector<std::string> extra_args_types = get_extra_args_typenames(extra_args);
      auto code = at::cuda::jit::generate_code(nTensors, f, name,
                                               f_inputs_type_str, compute_type_str, result_type_str,
                                               /*contiguous=*/true, /*dynamic_casting=*/false,
                                               at::cuda::jit::BinaryFuncVariant::NoScalar,
                                               extra_args_types,
                                               vectorized, vec_size);
      std::string kernel_name = vectorized ? name + "_vectorized" + std::to_string(vec_size) : name;

      // Acquires the program
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, kernel_name);
    }
  }

  // size of `extra_args` is unknown at compile-time
  auto extra_args_size = extra_args.size();

  float scalar_val = 0;

  if (vectorized) {
    // pack args for kernel launch
    constexpr int kernel_args = 3;
    void* args[kernel_args + extra_args_size];
    args[0] = static_cast<void*>(&N);
    args[1] = data_ptr;
    args[2] = static_cast<void*>(&scalar_val);

    for (const auto i : c10::irange(extra_args_size)) {
      // since 3 slots are already filled in `args`
      args[i + 3] = const_cast<void*>(extra_args[i].data_ptr());
    }
    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, {grid, 1u, 1u}, {num_threads(), 1u, 1u});
  } else {
    // TODO: fix raw ptr here
    void* ic_ptr = make_trivial_offset_calculator(iter.ninputs());
    auto oc = TrivialOffsetCalculator<1>();
    auto l = memory::LoadWithoutCast();
    auto s = memory::StoreWithoutCast();

    // pack args for kernel launch
    constexpr int kernel_args = 7;
    void* args[kernel_args + extra_args_size];
    args[0] = static_cast<void*>(&N);
    args[1] = data_ptr;
    args[2] = ic_ptr;
    args[3] = static_cast<void*>(&oc);
    args[4] = static_cast<void*>(&l);
    args[5] = static_cast<void*>(&s);
    args[6] = static_cast<void*>(&scalar_val);

    for (const auto i : c10::irange(extra_args_size)) {
      // since 7 slots are already filled in `args`
      args[i + 7] = const_cast<void*>(extra_args[i].data_ptr());
    }

    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, {grid, 1u, 1u}, {num_threads(), 1u, 1u});
  }
}

static inline void launch_jitted_unrolled_kernel(
  const std::string& name, TensorIteratorBase& iter,
  DeviceIndex dev_idx, int64_t N, const std::string& f, void* data_ptr,
  void* ic_ptr, void* oc_ptr, void* l_ptr, void* s_ptr, bool contiguous, bool dynamic_casting,
  const std::vector<at::Scalar>& extra_args) {

  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  //casting result to int is always safe, intermediate is int64 and won't overflow
  const uint32_t grid = (N + block_work_size() - 1) / block_work_size();

  static std::mutex _jiterator_mutex;
  static std::vector<at::cuda::jit::NvrtcFunction> fns(c10::cuda::device_count());

  at::cuda::jit::NvrtcFunction* fn_ptr = &fns[dev_idx];
  // if (!fn_ptr->function) {
  {
    const std::lock_guard<std::mutex> lock{_jiterator_mutex};
    // if (!fn_ptr->function) {
    {
      int nTensors = iter.ntensors();

      const at::ScalarType common_dtype = iter.common_dtype();
      std::string f_inputs_type_str = at::cuda::jit::typeName(common_dtype);
      std::string compute_type_str = at::cuda::jit::typeName(toOpMathType(common_dtype));
      std::string result_type_str = at::cuda::jit::typeName(common_dtype);

      std::cout<<"launch_jitted_unrolled_kernel_raw_ptr\n";

      c10::SmallVector<std::string> extra_args_types = get_extra_args_typenames(extra_args);
      auto code = at::cuda::jit::generate_code(nTensors, f, name,
                                               f_inputs_type_str, compute_type_str, result_type_str,
                                               contiguous, dynamic_casting,
                                               at::cuda::jit::BinaryFuncVariant::NoScalar,
                                               extra_args_types);
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, name);
    }
  }

  float scalar_val = 0;

  // pack args for kernel launch
  constexpr int kernel_args = 7;
  // size of `extra_args` is unknown at compile-time
  auto extra_args_size = extra_args.size();
  void* args[kernel_args + extra_args_size];
  args[0] = static_cast<void*>(&N);
  args[1] = data_ptr;
  args[2] = ic_ptr;
  args[3] = oc_ptr;
  args[4] = l_ptr;
  args[5] = s_ptr;
  args[6] = static_cast<void*>(&scalar_val);

  for (const auto i : c10::irange(extra_args_size)) {
    // since 7 slots are already filled in `args`
    args[i + 7] = const_cast<void*>(extra_args[i].data_ptr());
  }
  at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, {grid, 1u, 1u},
  {num_threads(), 1u, 1u});
}

void jitted_gpu_kernel_dynamic_impl(
    const std::string& kernel_name,
    TensorIteratorBase& iter,
    const std::string& f,
    const bool dynamic_casting,
    const std::vector<at::Scalar>& extra_args) {

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  // TODO: assuming supported ninputs <=8, with only one output
  TORCH_INTERNAL_ASSERT(iter.ninputs() <= 8);

  ArrayVariant data(iter);
  void* data_ptr = data.data_ptr();

  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();

  std::cout<<"contiguous: "<< contiguous << " dynamic_casting: "<< dynamic_casting << std::endl;

  // Decides which of 4 kernel types to launch
  // Variations are:
  //   - Case 1: no dynamic casting and contiguous
  //   - Case 2: no dynamic casting and noncontiguous
  //   - Case 3: dynamic casting and contiguous
  //   - Case 4: dynamic casting and noncontiguous
  // These cases align with the non-jitted CUDALoops.cuh cases in gpu_kernel_impl

  if (!dynamic_casting) {
    if (contiguous) {
      // Case 1: no dynamic casting and contiguous
      launch_jitted_vectorized_kernel(kernel_name, iter,
         iter.device().index(), numel, f, data_ptr, extra_args);
      return;
    }

    // Case 2: no dynamic casting and noncontiguous

    // TODO: fix raw ptr

    OffsetCalculatorContainer input_offset_calculator(iter);
    void* ic_ptr = input_offset_calculator.data_ptr();

    auto output_offset_calculator = make_output_offset_calculator(iter);
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();

    void* oc_ptr = static_cast<void*>(&output_offset_calculator);
    void* l_ptr = static_cast<void*>(&loader);
    void* s_ptr = static_cast<void*>(&storer);

    launch_jitted_unrolled_kernel(
      kernel_name, iter, iter.device().index(), numel, f, data_ptr,
      ic_ptr, oc_ptr, l_ptr, s_ptr, contiguous, dynamic_casting, extra_args);

    return;
  }


  // Cases 3 and 4 are handled below
  // Both require construction of a storer (this asserts 1 output) and one or more loaders

  // Creates store cast to output (the zeroth tensor in TensorIterator)
  auto storer = memory::StoreWithCast(iter.dtype(0));

  // Creates load casts from inputs (note offset indexing into the iterators 1...n tensors)
  // at::detail::Array<ScalarType, arity> dtypes;
  // for (auto i = decltype(arity){0}; i < arity; ++i) {
  //   dtypes[i] = iter.dtype(i + 1);
  // }
  // auto loader = memory::LoadWithCast<arity>(dtypes);
  void* l_ptr = make_load_with_cast(iter);
  void* s_ptr = static_cast<void*>(&storer);

  if (contiguous) {
    // Case 3: dynamic casting and contiguous
    void* ic_ptr = make_trivial_offset_calculator(iter.ninputs());
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    void* oc_ptr = static_cast<void*>(&output_offset_calculator);

    launch_jitted_unrolled_kernel(
      kernel_name, iter, iter.device().index(), numel, f, data_ptr,
      ic_ptr, oc_ptr, l_ptr, s_ptr, contiguous, dynamic_casting, extra_args);
    return;
  }

  // Case 4: dynamic casting and noncontiguous
  OffsetCalculatorContainer input_offset_calculator(iter);
  void* ic_ptr = input_offset_calculator.data_ptr();

  auto output_offset_calculator = make_output_offset_calculator(iter);
  void* oc_ptr = static_cast<void*>(&output_offset_calculator);

  launch_jitted_unrolled_kernel(
      kernel_name, iter, iter.device().index(), numel, f, data_ptr,
      ic_ptr, oc_ptr, l_ptr, s_ptr, contiguous, dynamic_casting, extra_args);
}

void jitted_gpu_kernel_dynamic(
    const std::string& kernel_name,
    TensorIteratorBase& iter,
    const std::string& f,
    const std::vector<at::Scalar>& extra_args) {

  // TODO: much of preamble is common to both jitted_gpu_kernel and gpu_kernel
  //   Maybe it could be refactored?
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      jitted_gpu_kernel_dynamic(kernel_name, sub_iter, f, extra_args);
    }
    return;
  }

  // Computes if dynamic casting is needed
  // Dynamic casting is needed if an input's or output's dtype differs from the common dtype
  // TODO: double check! this is different from jitted_gpu_kernel's logic
  bool needs_dynamic_casting = false;
  const at::ScalarType common_dtype = iter.common_dtype();
  for (auto i = 0; i < iter.ntensors(); ++i) {
    if (iter.dtype(i) != common_dtype) {
      needs_dynamic_casting = true;
      break;
    }
  }

  jitted_gpu_kernel_dynamic_impl(kernel_name, iter, f, needs_dynamic_casting, extra_args);
}


} // namespace native



namespace cuda {

at::Tensor CompileKernel(
  const std::string& op_string,
  const std::string& optional_name,
  const std::vector<at::Tensor>& tensors,
  const std::vector<at::Scalar>& extra_args) {

  Tensor output;
  // TODO: double check if any other flags needs to be set
  TensorIteratorConfig config;
  config
    .set_check_mem_overlap(true)
    .allow_cpu_scalars(false)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    // TODO:  add_output or add_owned_output
    .add_owned_output(output);
  for (const auto& t: tensors){
    config.add_input(t);
  }
  TensorIterator iter = config.build();

  at::native::jitted_gpu_kernel_dynamic(optional_name, iter, op_string, extra_args);

  return iter.output();
}


}} // namespace at::cuda

