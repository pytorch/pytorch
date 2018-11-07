#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/core/ArrayRef.h>

#include <iostream>


namespace at { namespace native {

static inline std::ostream& operator<<(std::ostream& out, dim3 dim) {
  if (dim.y == 1 && dim.z == 1) {
    out << dim.x;
  } else {
    out << "[" << dim.x << "," << dim.y << "," << dim.z << "]";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const ReduceConfig& config) {
  out << "ReduceConfig(";
  out << "element_size_bytes=" << config.element_size_bytes << ", ";
  out << "num_inputs=" << config.num_inputs << ", ";
  out << "num_outputs=" << config.num_outputs << ", ";
  out << "step_input=" << config.step_input << ", ";
  out << "step_output=" << config.step_output << ", ";
  out << "ctas_per_output=" << config.ctas_per_output << ", ";
  out << "input_mult=[";
  for (int i = 0; i < 3; i++) {
    if (i != 0) {
      out << ",";
    }
    out << config.input_mult[i];
  }
  out << "], ";
  out << "output_mult=[";
  for (int i = 0; i < 2; i++) {
    if (i != 0) {
      out << ",";
    }
    out << config.output_mult[i];
  }
  out << "], ";
  out << "values_per_thread=" << config.values_per_thread() << ", ";
  out << "block=" << config.block() << ", ";
  out << "grid=" << config.grid() << ", ";
  out << "global_memory_size=" << config.global_memory_size();
  out << ")";
  return out;
}

}}  // namespace at::native
