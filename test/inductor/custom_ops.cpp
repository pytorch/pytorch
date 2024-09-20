#include <torch/csrc/api/include/torch/types.h>  // @manual=fbcode//caffe2:libtorch

#include <cstdint>
#include <iostream>
#include <string>

namespace at {

Tensor custom_add_impl(Tensor t1, Tensor t2) {
  return t1 + t2;
}

Tensor fn_with_all_inputs_impl(
    const Tensor& tensor,
    const c10::List<Tensor>& tensors,
    const c10::List<std::optional<Tensor>>& optional_tensors,
    const bool b8,
    const c10::List<bool>& b8s,
    const int64_t i64,
    const c10::List<int64_t>& i64s,
    const int64_t& symint,
    const IntArrayRef symints,
    const double f64,
    const c10::List<double>& f64s,
    const at::Scalar& scalar,
    at::ArrayRef<at::Scalar> scalars,
    const std::string& string,
    const std::vector<std::string>& strings,
    // const c10::ScalarType& dtype,
    // const MemoryFormat& memory_format,
    // const Layout& layout,
    const Device& device,
    // optional
    const std::optional<Tensor>& o_tensor,
    const std::optional<c10::List<Tensor>>& o_tensors,
    const std::optional<bool>& o_b8,
    const std::optional<c10::List<bool>>& o_b8s,
    const std::optional<int64_t>& o_i64,
    const std::optional<c10::List<int64_t>>& o_i64s,
    const std::optional<int64_t>& o_symint,
    const std::optional<IntArrayRef>& o_symints,
    const std::optional<double>& o_f64,
    const std::optional<c10::List<double>>& o_f64s,
    const std::optional<at::Scalar>& o_scalar,
    const std::optional<at::ArrayRef<at::Scalar>>& o_scalars,
    const std::optional<std::string>& o_string,
    const std::optional<std::vector<std::string>>& o_strings,
    // const std::optional<c10::ScalarType>& o_dtype,
    // const std::optional<MemoryFormat>& o_memory_format,
    // const std::optional<Layout>& o_layout,
    const std::optional<Device>& o_device) {
  std::cout << "tensor shape: " << tensor.sizes() << std::endl;

  std::cout << "tensors shape: ";
  for (auto t : tensors) {
    std::cout << t.get().toTensor().sizes() << ", ";
  }
  std::cout << std::endl;

  std::cout << "optional tensors shape: ";
  for (auto t : optional_tensors) {
    if (t.get().toOptional<Tensor>().has_value()) {
      std::cout << t.get().toTensor().sizes() << ", ";
    } else {
      std::cout << "None, ";
    }
  }
  std::cout << std::endl;

  std::cout << "b8 " << c10::IValue(b8) << std::endl;
  std::cout << "b8s " << c10::IValue(b8s) << std::endl;
  std::cout << "i64 " << c10::IValue(i64) << std::endl;
  std::cout << "i64s " << c10::IValue(i64s) << std::endl;
  std::cout << "symint " << c10::IValue(symint) << std::endl;
  std::cout << "symints " << c10::IValue(symints) << std::endl;
  std::cout << "f64 " << c10::IValue(f64) << std::endl;
  std::cout << "f64s " << c10::IValue(f64s) << std::endl;
  std::cout << "scalar " << c10::IValue(scalar) << std::endl;
  std::cout << "scalars " << c10::IValue(scalars) << std::endl;
  std::cout << "string " << c10::IValue(string) << std::endl;
  std::cout << "strings " << c10::IValue(strings) << std::endl;
  // std::cout << "dtype " << c10::IValue(dtype) << std::endl;
  // std::cout << "memory_format " << c10::IValue(memory_format) << std::endl;
  // std::cout << "layout " << c10::IValue(layout) << std::endl;
  std::cout << "device " << c10::IValue(device) << std::endl;

  std::cout << "o_tensor "
            << (o_tensor.has_value() ? c10::IValue(o_tensor.value().sizes())
                                     : "None")
            << std::endl;

  std::cout << "o_tensors shape: ";
  if (o_tensors.has_value()) {
    for (auto t : o_tensors.value()) {
      std::cout << t.get().toTensor().sizes() << ", ";
    }
  } else {
    std::cout << "None";
  }
  std::cout << std::endl;

  std::cout << "o_b8 "
            << (o_b8.has_value() ? c10::IValue(o_b8.value()) : "None")
            << std::endl;
  std::cout << "o_b8s "
            << (o_b8s.has_value() ? c10::IValue(o_b8s.value()) : "None")
            << std::endl;
  std::cout << "o_i64 "
            << (o_i64.has_value() ? c10::IValue(o_i64.value()) : "None")
            << std::endl;
  std::cout << "o_i64s "
            << (o_i64s.has_value() ? c10::IValue(o_i64s.value()) : "None")
            << std::endl;
  std::cout << "o_symint "
            << (o_symint.has_value() ? c10::IValue(o_symint.value()) : "None")
            << std::endl;
  std::cout << "o_symints "
            << (o_symints.has_value() ? c10::IValue(o_symints.value()) : "None")
            << std::endl;
  std::cout << "o_f64 "
            << (o_f64.has_value() ? c10::IValue(o_f64.value()) : "None")
            << std::endl;
  std::cout << "o_f64s "
            << (o_f64s.has_value() ? c10::IValue(o_f64s.value()) : "None")
            << std::endl;
  std::cout << "o_scalar "
            << (o_scalar.has_value() ? c10::IValue(o_scalar.value()) : "None")
            << std::endl;
  std::cout << "o_scalars "
            << (o_scalars.has_value() ? c10::IValue(o_scalars.value()) : "None")
            << std::endl;
  std::cout << "o_string "
            << (o_string.has_value() ? c10::IValue(o_string.value()) : "None")
            << std::endl;
  std::cout << "o_strings "
            << (o_strings.has_value() ? c10::IValue(o_strings.value()) : "None")
            << std::endl;
  // std::cout << "o_dtype "
  //           << (o_dtype.has_value() ? c10::IValue(o_dtype.value()) : "None")
  //           << std::endl;
  // std::cout << "o_memory_format "
  //           << (o_memory_format.has_value()
  //                   ? c10::IValue(o_memory_format.value())
  //                   : "None")
  //           << std::endl;
  // std::cout << "o_layout "
  //           << (o_layout.has_value() ? c10::IValue(o_layout.value()) : "None")
  //           << std::endl;
  std::cout << "o_device "
            << (o_device.has_value() ? c10::IValue(o_device.value()) : "None")
            << std::endl;

  int64_t int_hash = 0;
  int_hash ^= i64;
  for (auto i : i64s) {
    int_hash ^= i;
  }
  if (o_i64.has_value()) {
    int_hash ^= o_i64.value();
  }
  if (o_i64s.has_value()) {
    for (auto i : o_i64s.value()) {
      int_hash ^= i;
    }
  }

  int_hash ^= symint;
  for (auto i : symints) {
    int_hash ^= i;
  }
  if (o_symint.has_value()) {
    int_hash ^= o_symint.value();
  }
  if (o_symints.has_value()) {
    for (auto i : o_symints.value()) {
      int_hash ^= i;
    }
  }

  return tensor + int_hash;
}

Tensor fn_with_default_input_impl(const Tensor& tensor, const int64_t i64) {
  return tensor + i64;
}

std::tuple<Tensor, Tensor> fn_with_tuple_output_impl(
    const Tensor& tensor,
    const int64_t i64) {
  return {tensor + i64, tensor - i64};
}

std::vector<Tensor> fn_with_list_output_impl(
    TensorList tensors,
    const int64_t i64) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.emplace_back(t + i64);
  }
  return outputs;
}

std::tuple<Tensor, std::vector<Tensor>> fn_with_mix_outputs_impl(
    const Tensor& tensor,
    TensorList tensors) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.emplace_back(t + 2);
  }
  return {tensor + 1, outputs};
}

std::tuple<Tensor, Tensor> fn_with_input_mutation_impl(
    Tensor& t0,
    const Tensor& t1,
    Tensor& t2) {
  t0.add_(1);
  t2.sub_(1);
  return {t1 + 1, t1 + 2};
}

// NOLINTBEGIN(clang-diagnostic-unused-parameter)
Tensor fn_with_all_inputs_meta(
    const Tensor& tensor,
    const c10::List<Tensor>& tensors,
    const c10::List<std::optional<Tensor>>& optional_tensors,
    const bool b8,
    const c10::List<bool>& b8s,
    const int64_t i64,
    const c10::List<int64_t>& i64s,
    const c10::SymInt& symint,
    c10::SymIntArrayRef symints,
    const double f64,
    const c10::List<double>& f64s,
    const at::Scalar& scalar,
    at::ArrayRef<at::Scalar> scalars,
    const std::string& string,
    const std::vector<std::string>& strings,
    // const c10::ScalarType& dtype,
    // const MemoryFormat& memory_format,
    // const Layout& layout,
    const Device& device,
    // optional
    const std::optional<Tensor>& o_tensor,
    const std::optional<c10::List<Tensor>>& o_tensors,
    const std::optional<bool>& o_b8,
    const std::optional<c10::List<bool>>& o_b8s,
    const std::optional<int64_t>& o_i64,
    const std::optional<c10::List<int64_t>>& o_i64s,
    const std::optional<c10::SymInt>& o_symint,
    at::OptionalSymIntArrayRef o_symints,
    const std::optional<double>& o_f64,
    const std::optional<c10::List<double>>& o_f64s,
    const std::optional<at::Scalar>& o_scalar,
    const std::optional<at::ArrayRef<at::Scalar>>& o_scalars,
    const std::optional<std::string>& o_string,
    const std::optional<std::vector<std::string>>& o_strings,
    // const std::optional<c10::ScalarType>& o_dtype,
    // const std::optional<MemoryFormat>& o_memory_format,
    // const std::optional<Layout>& o_layout,
    const std::optional<Device>& o_device) {
  return tensor;
}

Tensor fn_with_default_input_meta(const Tensor& tensor, const int64_t i64) {
  return tensor.clone();
}

std::tuple<Tensor, Tensor> fn_with_tuple_output_meta(
    const Tensor& tensor,
    const int64_t i64) {
  return {tensor.clone(), tensor.clone()};
}

std::vector<Tensor> fn_with_list_output_meta(
    TensorList tensors,
    const int64_t i64) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.push_back(t.clone());
  }
  return outputs;
}

std::tuple<Tensor, std::vector<Tensor>> fn_with_mix_outputs_meta(
    const Tensor& tensor,
    TensorList tensors) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.push_back(t.clone());
  }
  return {tensor.clone(), outputs};
}

std::tuple<Tensor, Tensor> fn_with_input_mutation_meta(
    Tensor& t0,
    const Tensor& t1,
    Tensor& t2) {
  return {t1.clone(), t1.clone()};
}

} // namespace at

TORCH_LIBRARY(aoti_custom_ops, m) {
  m.def("custom_add(Tensor t1, Tensor t2) -> Tensor");
  m.def(
      "fn_with_all_inputs(Tensor tensor, "
      "Tensor[] tensors, "
      "Tensor?[] optional_tensors, "
      "bool b8, bool[] b8s, "
      "int i64, int[] i64s, "
      "SymInt symint, SymInt[] symints, "
      "float f64, float[] f64s, "
      "Scalar scalar, Scalar[] scalars, "
      "str string, str[] strings, "
      // "ScalarType dtype, "
      // "MemoryFormat memory_format, "
      // "Layout layout, "
      "Device device, "
      "*, "
      "Tensor? o_tensor, Tensor[]? o_tensors, "
      "bool? o_b8, bool[]? o_b8s, "
      "int? o_i64, int[]? o_i64s, "
      "SymInt? o_symint, SymInt[]? o_symints, "
      "float? o_f64, float[]? o_f64s, "
      "Scalar? o_scalar, Scalar[]? o_scalars, "
      "str? o_string, str[]? o_strings, "
      // "ScalarType? o_dtype, "
      // "MemoryFormat? o_memory_format, "
      // "Layout? o_layout, "
      "Device? o_device) -> Tensor");

  m.def("fn_with_default_input(Tensor t, int i=3) -> Tensor");

  m.def("fn_with_tuple_output(Tensor t, int i) -> (Tensor, Tensor)");

  m.def("fn_with_list_output(Tensor[] tensors, int i) -> Tensor[]");

  m.def(
      "fn_with_mix_outputs(Tensor t, Tensor[] tensors) -> (Tensor, Tensor[])");

  m.def(
      "fn_with_input_mutation(Tensor(a!) t0, Tensor t1, Tensor(b!) t2) -> (Tensor, Tensor)");

}

TORCH_LIBRARY_IMPL(aoti_custom_ops, CompositeExplicitAutograd, m) {
  m.impl("custom_add", at::custom_add_impl);
  m.impl("fn_with_all_inputs", at::fn_with_all_inputs_impl);
  m.impl("fn_with_default_input", at::fn_with_default_input_impl);
  m.impl("fn_with_tuple_output", at::fn_with_tuple_output_impl);
  m.impl("fn_with_list_output", at::fn_with_list_output_impl);
  m.impl("fn_with_mix_outputs", at::fn_with_mix_outputs_impl);
  m.impl("fn_with_input_mutation", at::fn_with_input_mutation_impl);
}

TORCH_LIBRARY_IMPL(aoti_custom_ops, Meta, m) {
  m.impl("fn_with_all_inputs", at::fn_with_all_inputs_meta);
  m.impl("fn_with_default_input", at::fn_with_default_input_meta);
  m.impl("fn_with_tuple_output", at::fn_with_tuple_output_meta);
  m.impl("fn_with_list_output", at::fn_with_list_output_meta);
  m.impl("fn_with_mix_outputs", at::fn_with_mix_outputs_meta);
  m.impl("fn_with_input_mutation", at::fn_with_input_mutation_meta);
}
