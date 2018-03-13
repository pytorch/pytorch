#pragma once

// It initially lies in torch/csrc/cuda, but to unconditionlly compile it
// we have to put it here.

namespace torch {
namespace utils {

void cuda_lazy_init();

}
}
