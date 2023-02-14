//#include <gtest/gtest.h>
#include <iostream>

#include "build/aot_inductor_output.h"

//def func(x):
//    return (torch.sigmoid(torch.sin(x)), torch.sigmoid(torch.cos(x)))I
std::vector<at::Tensor> func(std::vector<at::Tensor> args) {
  return {torch::sigmoid(torch::sin(args[0])), torch::sigmoid(torch::cos(args[0]))};
}

int main() {
    auto args = {at::randn({8, 4, 16, 16})};
    auto results_ref = func(args);
    auto results_opt = __aot_inductor_entry(args);

    assert(torch::allclose(results_ref[0], results_opt[0]));
    assert(torch::allclose(results_ref[1], results_opt[1]));
    printf("PASS\n");
    return 0;
}
