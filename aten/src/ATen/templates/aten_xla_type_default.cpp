// ${generated_comment}
#include <torch_xla/csrc/aten_xla_type_default.h>

#include <ATen/Context.h>
#include <torch/library.h>
#include <ATen/CPUGeneratorImpl.h>

#include <tensorflow/compiler/xla/xla_client/debug_macros.h>
#include <tensorflow/compiler/xla/xla_client/metrics.h>
#include <tensorflow/compiler/xla/xla_client/tf_logging.h>
#include <torch_xla/csrc/aten_xla_bridge.h>
#include <torch_xla/csrc/XLANativeFunctions.h>
#include <torch_xla/csrc/function_call_tracker.h>

namespace ${cpp_namespace} {

// convenience helpers for extracting out an optional c10::Device

c10::optional<c10::Device> get_device_arg(at::Tensor tensor) {
    return tensor.device();
}

c10::optional<c10::Device> get_device_arg(c10::optional<at::Tensor> tensor) {
    return tensor ? c10::optional<c10::Device>((*tensor).device()) : c10::nullopt;
}

c10::optional<c10::Device> get_device_arg(std::vector<at::Tensor> tensors) {
    return tensors.size() > 0 ? c10::optional<c10::Device>(tensors[0].device()) : c10::nullopt;
}

c10::optional<c10::Device> get_device_arg(at::TensorList tensors) {
    return tensors.size() > 0 ? c10::optional<c10::Device>(tensors[0].device()) : c10::nullopt;
}

c10::optional<c10::Device> get_device_arg(c10::optional<c10::Device> device) {
    return device;
}

c10::optional<c10::Device> get_device_arg(c10::Device device) {
    return c10::optional<c10::Device>(device);
}

// convenience helpers for converting tensors to an optional device

at::Tensor to_device_opt(const at::Tensor tensor, c10::optional<c10::Device> device) {
    return device ? tensor.to(*device) : tensor;
}

std::vector<at::Tensor> to_device_opt(const std::vector<at::Tensor>& tensors, c10::optional<c10::Device> device) {
    std::vector<at::Tensor> output_tensors;
    for (const auto& t : tensors) {
        output_tensors.push_back(to_device_opt(t, device));
    }
    return output_tensors;
}

// convenience helper for converting tensors to cpu

std::vector<at::Tensor> to_cpu(const at::TensorList& tensors) {
    // We can't just call at::to_cpu() on the entire list of Tensors
    // Because it will break on undefined tensors. Separate out undefined tensors first.
    std::vector<at::Tensor> cpu_tensors(tensors.size());
    std::vector<at::Tensor> valid_tensors;
    std::vector<bool> to_translate(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const at::Tensor& tensor = tensors[i];
        if (tensor.defined()) {
            to_translate[i] = true;
            valid_tensors.push_back(tensor);
        } else {
            cpu_tensors[i] = tensor;
        }
    }
    auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
    for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
        if (to_translate[i]) {
            cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
        }
    }
  return cpu_tensors;
}

std::vector<c10::optional<at::Tensor>> to_cpu(const std::vector<c10::optional<at::Tensor>>& tensors) {
    std::vector<c10::optional<at::Tensor>> opt_tensors(tensors.size());
    std::vector<at::Tensor> materialized_tensors;
    std::vector<bool> to_translate(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        if (tensor.has_value()) {
            to_translate[i] = true;
            materialized_tensors.push_back(*tensor);
        }
    }
    auto aten_materialized_tensors = to_cpu(materialized_tensors);
    for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
        if (to_translate[i]) {
          opt_tensors[i] =
          std::move(aten_materialized_tensors[defined_pos++]);
        }
    }
    return opt_tensors;
}

${dispatch_aten_fallback_definitions}



TORCH_LIBRARY_IMPL(aten, XLA, m) {
${dispatch_registrations}

}

}  // namespace torch_xla
