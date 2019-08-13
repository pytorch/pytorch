#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <cstdint>

namespace at {
namespace native {

int64_t bench__zero_args_at() {
  return 0;
}

int64_t bench__zero_args_c10() {
  return 0;
}

int64_t bench__one_arg_at(const Tensor& in) {
  return 0;
}

int64_t bench__one_arg_c10(const Tensor& in) {
  return 0;
}

Tensor bench__one_arg_return_at(const Tensor& in) {
  return in;
}

Tensor bench__one_arg_return_c10(const Tensor& in) {
  return in;
}

int64_t bench__two_args_at(const Tensor& arg1, const Tensor& arg2) {
  return 0;
}

int64_t bench__two_args_c10(const Tensor& arg1, const Tensor& arg2) {
  return 0;
}

Tensor bench__two_args_return_at(const Tensor& arg1, const Tensor& arg2) {
  return arg1;
}

Tensor bench__two_args_return_c10(const Tensor& arg1, const Tensor& arg2) {
  return arg1;
}

int64_t bench__three_args_at(const Tensor& arg1, const Tensor& arg2, const Tensor& arg3) {
  return 0;
}

int64_t bench__three_args_c10(const Tensor& arg1, const Tensor& arg2, const Tensor& arg3) {
  return 0;
}

Tensor bench__three_args_return_at(const Tensor& arg1, const Tensor& arg2, const Tensor& arg3) {
  return arg1;
}

Tensor bench__three_args_return_c10(const Tensor& arg1, const Tensor& arg2, const Tensor& arg3) {
  return arg1;
}

Tensor bench__add_at(const Tensor& arg1, const Tensor& arg2) {
  return arg1.add(arg2);
}

Tensor bench__add_c10(const Tensor& arg1, const Tensor& arg2) {
  return arg1.add(arg2);
}

}
}
