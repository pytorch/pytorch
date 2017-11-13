#pragma once

#include "ATen/ATen.h"
#include <vector>

/**
 * ATen native functions are ways to write ATen methods which make only
 * make use of other ATen operations (e.g., it is not necessary to
 * bind into TH/THC code or drop into CUDA kernels.)  These functions
 * are written as both functions as well as cwrap fragments, which are
 * then folded into the ATen code generation process; define a function
 * here, and it will show up as a method on at::Tensor.
 *
 * At the moment, only type_method_definition_level: base is supported.
 */

namespace at {
namespace native {

// [NativeFunction]
Tensor type_as(const Tensor &self, const Tensor &other);

// [NativeFunction]
Tensor expand_as(const Tensor &self, const Tensor &other);

// [NativeFunction]
std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim=0);

// [NativeFunction]
std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim=0);

// [NativeFunction]
int64_t size(const Tensor& self, int64_t dim);


// [NativeFunction]
int64_t stride(const Tensor& self, int64_t dim);


// [NativeFunction]
bool is_same_size(const Tensor& self, const Tensor& other);

// [NativeFunction]
Tensor permute(const Tensor& self, IntList dims);


// [NativeFunction]
Tensor expand(const Tensor& self, IntList size);

// [NativeFunction]
Tensor squeeze(const Tensor& self);

// [NativeFunction]
Tensor squeeze(const Tensor& self, int64_t dim);

// [NativeFunction]
Tensor & squeeze_(Tensor& self);


// [NativeFunction]
Tensor & squeeze_(Tensor& self, int64_t dim);

// [NativeFunction]
Tensor unsqueeze(const Tensor& self, int64_t dim);

// [NativeFunction]
Tensor & unsqueeze_(Tensor& self, int64_t dim);

/*
[NativeFunction]
variants: function
*/
Tensor stack(TensorList tensors, int64_t dim=0);

/*
[NativeFunction]
variants: function
type_method_definition_dispatch: {
  - CPU: at::native::SpatialRoIPooling_forward
  - CUDA: at::native::SpatialRoIPooling_forward_cuda
}
*/
std::tuple<Tensor, Tensor> SpatialRoIPooling_forward(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale);

std::tuple<Tensor, Tensor> SpatialRoIPooling_forward_cuda(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale);

/*
[NativeFunction]
variants: function
type_method_definition_dispatch: {
  - CPU: at::native::SpatialRoIPooling_backward
  - CUDA: at::native::SpatialRoIPooling_backward_cuda
}
*/
Tensor SpatialRoIPooling_backward(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale,
  const Tensor& gradOutput,
  const Tensor& argmaxes);

Tensor SpatialRoIPooling_backward_cuda(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale,
  const Tensor& gradOutput,
  const Tensor& argmaxes);

}
}
