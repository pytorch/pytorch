/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

#include <torch/csrc/executorch/shim/module_shim.h>

#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/stableivalue_conversions.h>

#include <torch/nativert/ModelRunner.h>

using namespace ::torch::nativert;

namespace {
AOTITorchError to_ivalue(const TypedStableIValue& v, c10::IValue* ret_val) {
  switch (v.tag) {
    case StableIValueTag::None:
      *ret_val = c10::IValue();
      return 0;
    case StableIValueTag::Int:
      *ret_val = c10::IValue(to<int64_t>(v.val));
      return 0;
    case StableIValueTag::Bool:
      *ret_val = c10::IValue(to<bool>(v.val));
      return 0;
    case StableIValueTag::Double:
      *ret_val = c10::IValue(to<double>(v.val));
      return 0;
    case StableIValueTag::Tensor:
      auto ret_raiiath = torch::aot_inductor::RAIIAtenTensorHandle(
          to<AtenTensorHandle>(v.val));
      *ret_val =
          (c10::IValue(*torch::aot_inductor::tensor_handle_to_tensor_pointer(
              ret_raiiath.get())));
      return 0;
  }
  // Should be unreachable.
  return 1;
}

AOTITorchError from_ivalue(const c10::IValue& v, TypedStableIValue* ret_val) {
  if (v.isNone()) {
    *ret_val = TypedStableIValue{from(std::nullopt), StableIValueTag::None};
    return 0;
  } else if (v.isInt()) {
    *ret_val = TypedStableIValue{from(v.toInt()), StableIValueTag::Int};
    return 0;
  } else if (v.isBool()) {
    *ret_val = TypedStableIValue{from(v.toBool()), StableIValueTag::Int};
    return 0;
  } else if (v.isDouble()) {
    *ret_val = TypedStableIValue{from(v.toDouble()), StableIValueTag::Int};
    return 0;
  } else if (v.isTensor()) {
    AtenTensorHandle ath = torch::aot_inductor::new_tensor_handle(
        std::move(const_cast<at::Tensor&>(v.toTensor())));
    *ret_val = TypedStableIValue{from(ath), StableIValueTag::Tensor};
    return 0;
  } else {
    return 1;
  }
}
} // namespace

AOTITorchError experimental_torch_load_module_from_file(
    const char* package_path,
    uint64_t package_path_len,
    const char* model_name,
    uint64_t model_name_len,
    ModuleHandle* ret_value) {
  std::string package_path_str(package_path, package_path_len);
  std::string model_name_str(model_name, model_name_len);
  *ret_value = reinterpret_cast<ModuleHandle>(
      new ModelRunner(package_path_str, model_name_str));
  return 0;
}

AOTITorchError experimental_torch_delete_module_object(ModuleHandle handle) {
  delete reinterpret_cast<ModelRunner*>(handle);
  return 0;
}

AOTITorchError experimental_torch_module_num_outputs(
    ModuleHandle handle,
    uint64_t* ret_value) {
  *ret_value = reinterpret_cast<ModelRunner*>(handle)->numOutputs();
  return 0;
}

AOTITorchError experimental_torch_module_forward_flattened(
    ModuleHandle handle,
    TypedStableIValue* args,
    uint64_t num_args,
    TypedStableIValue* ret_values,
    uint64_t num_outputs) {
  std::vector<c10::IValue> vec;
  vec.reserve(num_args);
  for (uint64_t i = 0; i < num_args; ++i) {
    auto err = to_ivalue(args[i], &vec.emplace_back());
    if (err != 0) {
      return err;
    }
  }

  std::vector<c10::IValue> out =
      reinterpret_cast<ModelRunner*>(handle)->runWithFlatInputsAndOutputs(
          std::move(vec));
  if (out.size() != num_outputs) {
    return 1;
  }

  for (uint64_t i = 0; i < num_outputs; ++i) {
    auto err = from_ivalue(out[i], &ret_values[i]);
    if (err != 0) {
      return err;
    }
  }
  return 0;
}
