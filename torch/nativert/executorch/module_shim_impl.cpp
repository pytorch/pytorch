/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

#include <torch/csrc/executorch/shim/module_shim.h>

#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/stable/stableivalue_conversions.h>

#include <torch/nativert/ModelRunner.h>

using namespace::torch::nativert;

namespace {
  c10::IValue to_ivalue(const TypedStableIValue& v) {
    switch (v.tag) {
      case StableIValueTag::None:
        return c10::IValue();
      case StableIValueTag::Int:
        return c10::IValue(to<int64_t>(v.val));
      case StableIValueTag::Bool:
        return c10::IValue(to<bool>(v.val));
      case StableIValueTag::Double:
        return c10::IValue(to<double>(v.val));
      case StableIValueTag::Tensor:
        auto ret_raiiath = torch::aot_inductor::RAIIAtenTensorHandle(
          to<AtenTensorHandle>(v.val));
        return (c10::IValue(*torch::aot_inductor::tensor_handle_to_tensor_pointer(
            ret_raiiath.get())));
    }
  }

  TypedStableIValue from_ivalue(const c10::IValue& v) {
    if (v.isNone()) {
      return TypedStableIValue{from(v.toNone()), StableIValueTag::None};
    } else if (v.isInt()) {
      return TypedStableIValue{from(v.toInt()), StableIValueTag::Int};
    } else if (v.isBool()) {
      return TypedStableIValue{from(v.toBool()), StableIValueTag::Int};
    } else if (v.isDouble()) {
      return TypedStableIValue{from(v.toDouble()), StableIValueTag::Int};
    } else if (v.isTensor()) {
      AtenTensorHandle ath = torch::aot_inductor::new_tensor_handle(
          std::move(const_cast<at::Tensor&>(v.toTensor())));
      return TypedStableIValue{from(ath), StableIValueTag::Tensor};
    } else {
      TORCH_CHECK(false, "Unsupported type");
    }
  }
} // namespace


AOTITorchError experimental_torch_load_module_from_file(const char* package_path, uint64_t package_path_len, const char* model_name, uint64_t model_name_len, ModuleHandle* ret_value) {
  std::string package_path_str(package_path, package_path_len);
  std::string model_name_str(model_name, model_name_len);
  *ret_value = reinterpret_cast<ModuleHandle>(new ModelRunner(package_path_str, model_name_str));
  return 0;
}

AOTITorchError experimental_torch_delete_module_object(ModuleHandle handle) {
  delete reinterpret_cast<ModelRunner*>(handle);
  return 0;
}

AOTITorchError experimental_torch_module_num_outputs(ModuleHandle handle, uint64_t* ret_value) {
  *ret_value = reinterpret_cast<ModelRunner*>(handle)->numOutputs();
  return 0;
}

AOTITorchError experimental_torch_module_forward_flattened(ModuleHandle handle, const TypedStableIValue* args, uint64_t num_args, TypedStableIValue* ret_values, uint64_t num_outputs) {
  std::vector<c10::IValue> vec;
  vec.reserve(num_args);
  for (uint64_t i = 0; i < num_args; ++i) {
    vec.push_back(to_ivalue(args[i]));
  }

  std::vector<c10::IValue> out = reinterpret_cast<ModelRunner*>(handle)->runWithFlatInputsAndOutputs(std::move(vec));
  TORCH_CHECK(out.size() == num_outputs);

  for (uint64_t i = 0; i < num_outputs; ++i) {
    ret_values[i] = from_ivalue(out[i]);
  }
  return 0;
}
