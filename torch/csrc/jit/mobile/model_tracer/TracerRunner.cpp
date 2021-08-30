#include <ATen/Functions.h>
#include <ATen/core/dispatch/ObservedOperators.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h>
#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>
#include <torch/csrc/jit/mobile/model_tracer/TracerRunner.h>
#include <torch/script.h>

namespace torch {
namespace jit {
namespace mobile {

const std::vector<std::string> always_included_traced_ops = {
    // The following are called from setup sections.
    "aten::resize_",
    "aten::slice.Tensor",
};

// Fetched from caffe2/aten/src/ATen/native/metal/MetalAten.mm
// Diffusion Link: https://fburl.com/diffusion/atwwmax2
const std::vector<std::string> gpu_metal_operators = {
    "aten::conv2d",
    "aten::add.Tensor",
    "aten::add_.Tensor",
    "aten::addmm",
    "aten::empty.memory_format",
    "aten::empty_strided",
    "aten::log_softmax.int",
    "aten::max_pool2d",
    "aten::mul.Tensor",
    "aten::relu",
    "aten::relu_",
    "aten::sigmoid",
    "aten::sub.Tensor",
    "aten::upsample_nearest2d.vec",
    "aten::view",
    "aten::adaptive_avg_pool2d",
    "aten::hardtanh_",
    "aten::reshape",
    "aten::flatten.using_ints",
};

/**
 * These are a collection of some common ATen methods that are usually
 * called outside of the Model's forward() run, and they need to be
 * traced to ensure that the used operators are included in the build.
 * If/When this list becomes too long, we can consider making it a
 * per-model list.
 */
void call_setup_methods() {
  at::zeros({2, 2});
  at::ones({2, 2});
  at::Tensor t1 = at::empty({7, 7});
  at::Tensor t2 = t1.fill_(3);
  at::narrow(t2, 1, 0, 1);
  at::eq(t1, t2);
  const volatile bool nz = at::zeros({1}).is_nonzero();
  (void)nz;

  // Create a byte tensor and copy it
  auto zb = at::zeros({10}, at::kByte);
  auto zf = at::zeros({10}, at::kFloat);
  zb.copy_(zf);
  t2.div(1);

  // Typically, failures show up in CopyKernel.cpp, so enumerating
  // common dtypes that may show up.
  const auto all_dtypes_for_copy = {
      at::kByte,
      at::kFloat,
      at::kInt,
      at::kChar,
      at::kDouble,
      at::kShort,
      at::kLong};
  for (const auto dtype : all_dtypes_for_copy) {
    auto tensor1 = at::empty({10}, dtype);
    tensor1.copy_(at::zeros({10}, at::kFloat));
  }

  torch::zeros({0, 0}, torch::ScalarType::Float);
  std::vector<float> storage(20, 1.0);
  std::vector<int64_t> sizes({2, 10});
  torch::from_blob(storage.data(), at::IntArrayRef(sizes), at::kFloat);
}

/**
 * Call methods on the Tensor object that we expect to be called
 * in production on this Tensor.
 */
void consume_tensor(at::Tensor& t) {
  const at::Tensor c = t;
  c.copy_(t.cpu());
}

TracerResult trace_run(const std::string& input_module_path) {
  at::globalContext().setQEngine(at::QEngine::QNNPACK);
  c10::ObservedOperators::getUnobservedOperatorList().clear();

  torch::jit::mobile::OperatorCallTracer op_tracer;
  torch::jit::mobile::KernelDTypeTracer kdtype_tracer;

  call_setup_methods();

  std::set<std::string> root_ops, traced_operators;
  torch::jit::mobile::KernelDTypeTracer::kernel_tags_type called_kernel_tags;

  std::vector<std::string> enabled_backends;

  using torch::jit::MobileModuleLoadOptions;

  try {
    // Load the module on CPU with the flag to skip the operator exists check.
    // This is needed so that we can load any TorchBind objects (custom classes)
    // that this model refers to so that any operators being called from those
    // TorchBind objects can be traced by the model tracer.
    //
    torch::jit::mobile::MobileModelRunner module_runner(input_module_path, 0);
    root_ops = module_runner.get_root_operators();
    std::cout << "Got " << root_ops.size() << " Root Operators." << std::endl;

    if (torch::jit::mobile::MobileModelRunner::set_has_metal_gpu_operators(
            root_ops)) {
      std::cout << "Inferred Metal GPU Model." << std::endl;
      root_ops.insert(gpu_metal_operators.begin(), gpu_metal_operators.end());
      called_kernel_tags["__unused__"] = {"Float"};
      enabled_backends.emplace_back("Metal GPU");

      // When we encounter a GPU model, we should call .cpu().copy_() on the
      // tensors in the bundled inputs, since this is what will happen when
      // such a model is executed on an iOS device (to copy the Tensor to Metal
      // memory via a call to .metal()).
      module_runner.for_each_tensor_in_bundled_inputs(consume_tensor);
    } else {
      std::cout << "Inferred CPU Model." << std::endl;
      enabled_backends.emplace_back("CPU");
      torch::jit::mobile::MobileModelRunner mobile_module_runner(
          input_module_path);

      // When we encounter a CPU model, we should call .cpu().copy_() on the
      // tensors in the bundled inputs, since this is what will happen when
      // such a model is executed on an Android device since the PyTorch JNI
      // bindings call .cpu() in JIValue::newJIValueFromAtIValue().
      module_runner.for_each_tensor_in_bundled_inputs(consume_tensor);

      // If a user has bundled inputs since that api was updated to accept
      // bundled inputs for multiple methods They should go down this route.
      // Even if they only bundle inputs for forward they will have the new
      // style bundled inputs. Since at this time in tracer.cpp we do not know
      // what functions have bundled inputs we must call
      // get_bundled_inputs_functions_and_info if it exists to get the set.
      if (mobile_module_runner.has_new_style_bundled_inputs()) {
        auto bundled_inputs_mapping =
            mobile_module_runner.get_many_functions_bundled_inputs();
        for (auto& entry : bundled_inputs_mapping) {
          std::string function_name = entry.first;
          std::vector<std::vector<at::IValue>> bundled_inputs = entry.second;
          std::cout << "Got " << bundled_inputs.size()
                    << " bundled input(s) for " << function_name << "\n\n";
          std::vector<at::IValue> results =
              mobile_module_runner.run_with_inputs(
                  function_name, bundled_inputs);

          for (auto& result : results) {
            // Consume the result Tensor(s) when tracing on CPU since the
            // Android/Java JNI bindings will do the same.
            torch::jit::mobile::for_each_tensor_in_ivalue(
                result, consume_tensor);
          }
        }
        // If get_bundled_inputs_functions_and_info does not exists we default
        // to assuming they bundled before that change was made. If no bundled
        // inputs are found here either an error will be thrown
      } else {
        std::vector<std::vector<at::IValue>> bundled_inputs =
            mobile_module_runner.get_all_bundled_inputs();
        std::cout << "Got " << bundled_inputs.size() << " bundled input(s)\n\n";
        std::vector<at::IValue> results =
            mobile_module_runner.run_with_inputs(bundled_inputs);

        for (auto& result : results) {
          // Consume the result Tensor(s) when tracing on CPU since the
          // Android/Java JNI bindings will do the same.
          torch::jit::mobile::for_each_tensor_in_ivalue(result, consume_tensor);
        }
      }
    }
  } catch (std::exception& ex) {
    std::cerr
        << "ModelTracer has not been able to load the module for the following reasons:\n"
        << ex.what()
        << "\nPlease consider posting to the PyTorch Edge Users workplace group (https://fb.workplace.com/groups/pytorch.edge.users) with the error message."
        << std::endl;

    throw ex;
  }

  traced_operators = op_tracer.getCalledOperators();
  called_kernel_tags.insert(
      kdtype_tracer.getCalledKernelTags().begin(),
      kdtype_tracer.getCalledKernelTags().end());
  traced_operators.insert(
      always_included_traced_ops.begin(), always_included_traced_ops.end());
  TracerResult tracer_result = {
      root_ops, traced_operators, called_kernel_tags, enabled_backends};

  if (tracer_result.traced_operators.size() <=
          always_included_traced_ops.size() ||
      tracer_result.called_kernel_tags.size() == 0) {
    throw std::runtime_error(
        "Error traced_operators size: " +
        std::to_string(tracer_result.traced_operators.size()) +
        " , Kernel_metadata size: " +
        std::to_string(tracer_result.called_kernel_tags.size()) +
        ", Expected kernel to be > 0 and the traced operator list " +
        "to be bigger then the default size " +
        std::to_string(always_included_traced_ops.size()) +
        ". Please ensure tracer was run with " +
        "'buck run -c pt.disable_per_op_profiling=0 -c pt.enable_record_kernel_dtype=1'");
  }

  return tracer_result;
}

} // namespace mobile
} // namespace jit
} // namespace torch
