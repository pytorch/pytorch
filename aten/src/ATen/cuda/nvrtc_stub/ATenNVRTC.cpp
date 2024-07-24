#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <iostream>

namespace at { namespace cuda {

NVRTC* load_nvrtc() {
  auto self = new NVRTC();
#define CREATE_ASSIGN(name) self->name = name;
  AT_FORALL_NVRTC(CREATE_ASSIGN)
  return self;
}

}} // at::cuda
