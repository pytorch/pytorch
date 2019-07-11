
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <iostream>
    
void tensor_list_abs(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::abs_out(out[i], input1[i]);
  }
}
    
void tensor_list_acos(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::acos_out(out[i], input1[i]);
  }
}
    
void tensor_list_asin(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::asin_out(out[i], input1[i]);
  }
}
    
void tensor_list_atan(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::atan_out(out[i], input1[i]);
  }
}
    
void tensor_list_ceil(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::ceil_out(out[i], input1[i]);
  }
}
    
void tensor_list_clamp(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::clamp_out(out[i], input1[i]);
  }
}
    
void tensor_list_cos(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::cos_out(out[i], input1[i]);
  }
}
    
void tensor_list_cosh(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::cosh_out(out[i], input1[i]);
  }
}
    
void tensor_list_digamma(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::digamma_out(out[i], input1[i]);
  }
}
    
void tensor_list_erf(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::erf_out(out[i], input1[i]);
  }
}
    
void tensor_list_erfc(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::erfc_out(out[i], input1[i]);
  }
}
    
void tensor_list_erfinv(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::erfinv_out(out[i], input1[i]);
  }
}
    
void tensor_list_exp(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::exp_out(out[i], input1[i]);
  }
}
    
void tensor_list_expm1(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::expm1_out(out[i], input1[i]);
  }
}
    
void tensor_list_floor(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::floor_out(out[i], input1[i]);
  }
}
    
void tensor_list_frac(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::frac_out(out[i], input1[i]);
  }
}
    
void tensor_list_lgamma(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::lgamma_out(out[i], input1[i]);
  }
}
    
void tensor_list_log(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::log_out(out[i], input1[i]);
  }
}
    
void tensor_list_log10(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::log10_out(out[i], input1[i]);
  }
}
    
void tensor_list_log1p(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::log1p_out(out[i], input1[i]);
  }
}
    
void tensor_list_log2(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::log2_out(out[i], input1[i]);
  }
}
    
void tensor_list_neg(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::neg_out(out[i], input1[i]);
  }
}
    
void tensor_list_nonzero(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::nonzero_out(out[i], input1[i]);
  }
}
    
void tensor_list_reciprocal(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::reciprocal_out(out[i], input1[i]);
  }
}
    
void tensor_list_round(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::round_out(out[i], input1[i]);
  }
}
    
void tensor_list_rsqrt(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::rsqrt_out(out[i], input1[i]);
  }
}
    
void tensor_list_sigmoid(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::sigmoid_out(out[i], input1[i]);
  }
}
    
void tensor_list_sign(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::sign_out(out[i], input1[i]);
  }
}
    
void tensor_list_sin(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::sin_out(out[i], input1[i]);
  }
}
    
void tensor_list_sinh(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::sinh_out(out[i], input1[i]);
  }
}
    
void tensor_list_sqrt(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::sqrt_out(out[i], input1[i]);
  }
}
    
void tensor_list_tan(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::tan_out(out[i], input1[i]);
  }
}
    
void tensor_list_tanh(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::tanh_out(out[i], input1[i]);
  }
}
    
void tensor_list_tril(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::tril_out(out[i], input1[i]);
  }
}
    
void tensor_list_triu(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::triu_out(out[i], input1[i]);
  }
}
    
void tensor_list_trunc(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::trunc_out(out[i], input1[i]);
  }
}
    

void tensor_list_add(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::add_out(out[i], input1[i], input2[i]);
  }
}
    
void tensor_list_mul(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::mul_out(out[i], input1[i], input2[i]);
  }
}
    
void tensor_list_sub(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::sub_out(out[i], input1[i], input2[i]);
  }
}
    
void tensor_list_div(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::div_out(out[i], input1[i], input2[i]);
  }
}
    

void tensor_list_eq(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::eq_out(out[i], input1[i], input2[i]);
  }
}
    
void tensor_list_ge(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::ge_out(out[i], input1[i], input2[i]);
  }
}
    
void tensor_list_gt(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::gt_out(out[i], input1[i], input2[i]);
  }
}
    
void tensor_list_le(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::le_out(out[i], input1[i], input2[i]);
  }
}
    
void tensor_list_ne(std::vector<at::Tensor>& input1,
                       std::vector<at::Tensor>& input2,
                       std::vector<at::Tensor>& out) {
  for (int64_t i = 0; i < input1.size(); i++) {
    at::ne_out(out[i], input1[i], input2[i]);
  }
}
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("abs", &tensor_list_abs, "abs");
  m.def("acos", &tensor_list_acos, "acos");
  m.def("asin", &tensor_list_asin, "asin");
  m.def("atan", &tensor_list_atan, "atan");
  m.def("ceil", &tensor_list_ceil, "ceil");
  m.def("clamp", &tensor_list_clamp, "clamp");
  m.def("cos", &tensor_list_cos, "cos");
  m.def("cosh", &tensor_list_cosh, "cosh");
  m.def("digamma", &tensor_list_digamma, "digamma");
  m.def("erf", &tensor_list_erf, "erf");
  m.def("erfc", &tensor_list_erfc, "erfc");
  m.def("erfinv", &tensor_list_erfinv, "erfinv");
  m.def("exp", &tensor_list_exp, "exp");
  m.def("expm1", &tensor_list_expm1, "expm1");
  m.def("floor", &tensor_list_floor, "floor");
  m.def("frac", &tensor_list_frac, "frac");
  m.def("lgamma", &tensor_list_lgamma, "lgamma");
  m.def("log", &tensor_list_log, "log");
  m.def("log10", &tensor_list_log10, "log10");
  m.def("log1p", &tensor_list_log1p, "log1p");
  m.def("log2", &tensor_list_log2, "log2");
  m.def("neg", &tensor_list_neg, "neg");
  m.def("nonzero", &tensor_list_nonzero, "nonzero");
  m.def("reciprocal", &tensor_list_reciprocal, "reciprocal");
  m.def("round", &tensor_list_round, "round");
  m.def("rsqrt", &tensor_list_rsqrt, "rsqrt");
  m.def("sigmoid", &tensor_list_sigmoid, "sigmoid");
  m.def("sign", &tensor_list_sign, "sign");
  m.def("sin", &tensor_list_sin, "sin");
  m.def("sinh", &tensor_list_sinh, "sinh");
  m.def("sqrt", &tensor_list_sqrt, "sqrt");
  m.def("tan", &tensor_list_tan, "tan");
  m.def("tanh", &tensor_list_tanh, "tanh");
  m.def("tril", &tensor_list_tril, "tril");
  m.def("triu", &tensor_list_triu, "triu");
  m.def("trunc", &tensor_list_trunc, "trunc");
  m.def("add", &tensor_list_add, "add");
  m.def("mul", &tensor_list_mul, "mul");
  m.def("sub", &tensor_list_sub, "sub");
  m.def("div", &tensor_list_div, "div");
  m.def("eq", &tensor_list_eq, "eq");
  m.def("ge", &tensor_list_ge, "ge");
  m.def("gt", &tensor_list_gt, "gt");
  m.def("le", &tensor_list_le, "le");
  m.def("ne", &tensor_list_ne, "ne");
}