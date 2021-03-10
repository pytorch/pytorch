// #include <torch/script.h>

// torch::Tensor ort_execute(torch::Tensor image, torch::Tensor warp) {
//   torch::Tensor output = image;
//   return output.clone();
// }

// #include <../torch/*.h>
// #include "../torch/script.h"
#include <cstdio>

void ort_execute() {
  printf("Hello World");
  return ;
}

// static auto registry =
//   torch::RegisterOperators("nezha_ops::ort_execute", &ort_execute);