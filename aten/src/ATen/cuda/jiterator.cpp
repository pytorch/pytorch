#include <ATen/cuda/jiterator.h>

#include <ATen/native/TensorIterator.h>

// #include <ATen/native/cuda/JitLoops.cuh>

#include <iostream>

namespace at {
namespace cuda {


void CompileKernel(
  const std::string& op_string,
  const std::string& optional_name,
  const std::string& optional_fusion_class,
  const std::vector<at::Tensor>& tensors) {

  std::cout<< "at::cuda::jit::CompileKernel" << std::endl;
  std::cout<< op_string << std::endl;
  std::cout<< optional_name << std::endl;
  std::cout<< optional_fusion_class << std::endl;
  for (const auto& t: tensors){
    std::cout<< t.toString() <<std::endl;
  }

  // TODO: creating a locally scoped tensor, is this correct?
  at::Tensor result;

  TensorIteratorConfig config{};
  config.add_output(result);
  for (const auto& t: tensors){
    config.add_input(t);
  }

  // TODO: not sure how to populate the followings
  // config
  // .set_check_mem_overlap(true)
  // .allow_cpu_scalars(true)
  // .promote_inputs_to_common_dtype(true)
  // .cast_common_dtype_to_outputs(true)
  // .enforce_safe_casting_to_output(true));

  TensorIterator iter = config.build();

  // const char name[] = "python_jitted";
  // at::native::jitted_gpu_kernel<
  //       /*name=*/ name,
  //       /*return_dtype=*/ float,
  //       /*common_dtype=*/ float,
  //       /*arity=*/ 2>(iter, op_string);




  return;
}


}} // namespace at::cuda

