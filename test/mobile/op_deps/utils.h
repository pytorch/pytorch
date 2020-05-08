#pragma once

#include <ATen/Tensor.h>

at::Tensor global_helper_call_AA_op_1(const at::Tensor& self);
at::Tensor global_helper_call_AA_op_2(const at::Tensor& self);
at::Tensor global_helper_call_AA_op_3(const at::Tensor& self);

namespace torch {
namespace jit {

class C10_EXPORT API_Class {
public:
  at::Tensor API_Method(const at::Tensor& self);
};

}  // namespace jit
}  // namespace torch
