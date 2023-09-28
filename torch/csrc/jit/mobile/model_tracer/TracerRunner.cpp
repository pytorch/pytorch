#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/ObservedOperators.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h>
#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>
#include <torch/csrc/jit/mobile/model_tracer/TracerRunner.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>

namespace torch {
namespace jit {
namespace mobile {

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
  at::Tensor t3 = t1.new_empty_strided(
      {2, 3},
      {3,
       1}); // TODO investigate how this is different from normal empty_strided
  at::narrow(t2, 1, 0, 1);
  at::eq(t1, t2);
  const volatile bool nz = at::native::is_nonzero(at::zeros({1}));
  (void)nz;

  // Create a byte tensor and copy it
  auto zb = at::zeros({10}, at::kByte);
  auto zf = at::zeros({10}, at::kFloat);
  zb.copy_(zf);
  t2.div(1);

  // Typically, failures show up in CopyKernel.cpp, so enumerating
  // common dtypes that may show up.
  const auto all_dtypes_for_copy = {
      at::kBool,
      at::kByte,
      at::kFloat,
      at::kInt,
      at::kChar,
      at::kDouble,
      at::kShort,
      at::kLong};
  for (const auto dtype : all_dtypes_for_copy) {
    auto tensor1 = at::empty({10}, dtype);
    tensor1.copy_(at::zeros({10}, at::kBool));
    tensor1.copy_(at::zeros({10}, at::kFloat));
    tensor1.copy_(at::zeros({10}, at::kInt));
  }

  torch::zeros({0, 0}, torch::ScalarType::Float);
  std::vector<float> storage(20, 1.0);
  std::vector<int64_t> sizes({2, 10});
  torch::from_blob(storage.data(), at::IntArrayRef(sizes), at::kFloat);
}

/**
 * Similar to setup methods there are a suite a functions that often appear
 * under certain conditions but may avoid getting called in the trace due to the
 * narrow nature of bundled inputs
 */
void call_dependent_methods(std::set<std::string>& root_ops) {
  bool is_training = false;
  bool has_batchnorm = false;
  bool has_dropout = false;
  for (const std::string& op : root_ops) {
    if (op.find("backward") != std::string::npos ||
        op.find("requires_grad_") != std::string::npos) {
      is_training = true;
    }
    if (op.find("batch_norm") != std::string::npos) {
      has_batchnorm = true;
    }
    if (op.find("dropout") != std::string::npos) {
      has_dropout = true;
    }
  }
  if (is_training && has_batchnorm) {
    at::batch_norm(
        at::ones({2, 2}),
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        true,
        0.1,
        0.1,
        false);
  }
  if (is_training && has_dropout) {
    at::dropout(at::ones({20, 20, 20}), 0.2, true);
  }
}

/**
 * Call methods on the Tensor object that we expect to be called
 * in production on this Tensor.
 */
void consume_tensor(const at::Tensor& t) {
  const at::Tensor& c = t;
  c.copy_(t.cpu());
}

std::unordered_map<std::string, c10::FunctionSchema>
_get_runtime_ops_and_schema() {
  std::unordered_map<std::string, c10::FunctionSchema> result;

  // Grab the jit operators
  auto nonDispatcherOperators = torch::jit::getAllOperators();
  for (const auto& full_op : nonDispatcherOperators) {
    auto op = full_op->schema();
    auto op_name = op.name();
    if (!op.overload_name().empty()) {
      op_name += ("." + op.overload_name());
    }
    result.emplace(op_name, op);
  }

  // Grab the dispatcher operators
  auto dispatcherOperators = c10::Dispatcher::singleton().getAllOpNames();
  for (auto& op : dispatcherOperators) {
    // grab schema
    const auto op_handle = c10::Dispatcher::singleton().findOp(op);
    if (op_handle->hasSchema()) {
      auto op_name = op.name;
      if (!op.overload_name.empty()) {
        op_name += ("." + op.overload_name);
      }
      result.emplace(op_name, op_handle->schema());
    }
  }

  return result;
}

/**
 * For the vast majority of usecases the instrumentation in getCustomClass will
 * catch any custom classes referenced by a model. There are however, niche
 * situations that avoid the getCustomClass instrumentation due to some nuances
 * of mobile model deserialization. To get around that we can search through all
 * the used ops, and inspect their schemas to search for any referenced classes.
 * Example schema: prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None,
 *   Scalar? output_min=None, Scalar? output_max=None) ->
 *   __torch__.torch.classes.xnnpack.LinearOpContext"
 */
void recordCustomClassesFromOpSchemas(
    std::set<std::string>& root_ops,
    std::set<std::string>& traced_ops,
    std::set<std::string>& loaded_classes) {
  std::set<std::string> ops;
  ops.insert(root_ops.begin(), root_ops.end());
  ops.insert(traced_ops.begin(), traced_ops.end());
  auto ops_and_schemas = _get_runtime_ops_and_schema();

  auto record_if_class = [&](std::string type_name) {
    // All custom class types start with __torch__ not sure if this is by
    // chance or guaranteed
    if (type_name.find("__torch__") != std::string::npos) {
      // The name of a customClassType here is its fully qualified name, but
      // in registration only the class name is used so only record that
      auto class_name = type_name.substr(type_name.find_last_of('.') + 1);
      // Function schemas can include other type indicators such as [] so we
      // need to trim to just alphanumeric + '_' characters as well
      class_name = class_name.substr(
          0,
          class_name.find_first_not_of(
              "aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ_1234567890"));
      loaded_classes.insert(class_name);
    }
  };

  for (auto& op_name : ops) {
    // This check is only necessary because of GPU models.
    // Certain models can only run on a specific backend say metal.
    // Those ops will be present in the models root ops, but likely
    // not the tracer on linux
    if (ops_and_schemas.find(op_name) != ops_and_schemas.end()) {
      auto& schema = ops_and_schemas.at(op_name);
      for (auto& arg : schema.arguments()) {
        record_if_class(arg.type()->annotation_str());
      }
      for (auto& ret : schema.returns()) {
        record_if_class(ret.type()->annotation_str());
      }
    }
  }
}

void run_model(
    const std::string& input_module_path,
    std::set<std::string>& root_ops,
    std::set<std::string>& enabled_backends,
    KernelDTypeTracer::kernel_tags_type& called_kernel_tags) {
  // Load the module on CPU with the flag to skip the operator exists check.
  // This is needed so that we can load any TorchBind objects (custom classes)
  // that this model refers to so that any operators being called from those
  // TorchBind objects can be traced by the model tracer.
  torch::jit::mobile::MobileModelRunner module_runner(input_module_path, 0);
  root_ops = module_runner.get_root_operators();
  std::cout << "Got " << root_ops.size() << " Root Operators." << std::endl;

  if (torch::jit::mobile::MobileModelRunner::set_has_metal_gpu_operators(
          root_ops)) {
    std::cout << "Inferred Metal GPU Model." << std::endl;
    root_ops.insert(gpu_metal_operators.begin(), gpu_metal_operators.end());
    called_kernel_tags["__unused__"] = {"Float"};
    enabled_backends.insert("Metal GPU");

    // When we encounter a GPU model, we should call .cpu().copy_() on the
    // tensors in the bundled inputs, since this is what will happen when
    // such a model is executed on an iOS device (to copy the Tensor to Metal
    // memory via a call to .metal()).
    module_runner.for_each_tensor_in_bundled_inputs(consume_tensor);
  } else {
    std::cout << "Inferred CPU Model." << std::endl;
    enabled_backends.insert("CPU");
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
        std::cout << "Got " << bundled_inputs.size() << " bundled input(s) for "
                  << function_name << "\n\n";
        std::vector<at::IValue> results =
            mobile_module_runner.run_with_inputs(function_name, bundled_inputs);

        for (auto& result : results) {
          // Consume the result Tensor(s) when tracing on CPU since the
          // Android/Java JNI bindings will do the same.
          torch::jit::mobile::for_each_tensor_in_ivalue(result, consume_tensor);
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
}

TracerResult trace_run(const std::string& input_module_path) {
  return trace_run(std::vector<std::string>(1, input_module_path));
}

TracerResult trace_run(const std::vector<std::string>& input_module_paths) {
  at::globalContext().setQEngine(at::QEngine::QNNPACK);
  c10::ObservedOperators::getUnobservedOperatorList().clear();

  torch::jit::mobile::OperatorCallTracer op_tracer;
  torch::jit::mobile::KernelDTypeTracer kdtype_tracer;
  torch::jit::mobile::CustomClassTracer custom_class_tracer;
  torch::jit::mobile::BuildFeatureTracer build_feature_tracer;

  call_setup_methods();

  std::set<std::string> root_ops, traced_operators, enabled_backends,
      loaded_classes, build_features;
  torch::jit::mobile::KernelDTypeTracer::kernel_tags_type called_kernel_tags;

  using torch::jit::MobileModuleLoadOptions;

  for (auto& input_module_path : input_module_paths) {
    // run with QNNPACK
    at::globalContext().setQEngine(at::QEngine::QNNPACK);

    run_model(
        input_module_path, root_ops, enabled_backends, called_kernel_tags);
    // Not every model can be successfully run with fbgemm,
    // but for those that can this can help broaden the tracers scope around
    // hyper optimized QNNPack paths
    try {
      at::globalContext().setQEngine(at::QEngine::FBGEMM);
      run_model(
          input_module_path, root_ops, enabled_backends, called_kernel_tags);
    } catch (std::exception& ex) {
      std::cerr
          << "ModelTracer encountered an error while attempting to run the model in FBGEMM mode"
          << ex.what() << "\n Skipping FBGEMM execution" << std::endl;
    }
  }

  call_dependent_methods(root_ops);

  op_tracer.getCalledOperators().withLock(
      [&](std::set<std::string>& called_operators) {
        traced_operators = called_operators;
      });

  recordCustomClassesFromOpSchemas(root_ops, traced_operators, loaded_classes);

  kdtype_tracer.getCalledKernelTags().withLock(
      [&](KernelDTypeTracer::kernel_tags_type& kernel_tags) {
        called_kernel_tags.insert(kernel_tags.begin(), kernel_tags.end());
      });

  traced_operators.insert(
      always_included_traced_ops.begin(), always_included_traced_ops.end());

  custom_class_tracer.getLoadedClasses().withLock(
      [&](CustomClassTracer::custom_classes_type& custom_classes) {
        loaded_classes.insert(custom_classes.begin(), custom_classes.end());
      });

  build_feature_tracer.getBuildFeatures().withLock(
      [&](BuildFeatureTracer::build_feature_type& bf) {
        build_features.insert(bf.begin(), bf.end());
      });

  TracerResult tracer_result = {
      root_ops,
      traced_operators,
      called_kernel_tags,
      loaded_classes,
      build_features,
      enabled_backends};

  return tracer_result;
}

} // namespace mobile
} // namespace jit
} // namespace torch
