#include <torch/csrc/jit/fuser/cuda/thnvrtc.h>
#include <iostream>

THNVRTC* torch_load_nvrtc() {
  auto self = new THNVRTC();
#define CREATE_ASSIGN(name) self->name = name;
  TORCH_FORALL_NVRTC(CREATE_ASSIGN)
  return self;
}
