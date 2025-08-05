#pragma once

#include <ATen/native/TensorIterator.h>


namespace at{
namespace native{

void gelu_bf16_lut_kernel(at::TensorIteratorBase& iter);

}
}