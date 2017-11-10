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

/*
[NativeFunction]
name: type_as
arg: Tensor self
arg: Tensor other
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::type_as
[/NativeFunction]
*/
static inline Tensor type_as(const Tensor &self, const Tensor &other) {
  return self.toType(other.type());;
}

/*
[NativeFunction]
name: expand_as
arg: Tensor self
arg: Tensor other
return: Tensor
variants: method, function
type_method_definition_level: base
type_method_definition_dispatch: at::native::expand_as
[/NativeFunction]
*/
static inline Tensor expand_as(const Tensor &self, const Tensor &other) {
  return self.expand(other.sizes());
}

/*
[NativeFunction]
type_method_definition_dispatch: at::native::split
[/NativeFunction]
*/
std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim=0);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::chunk
*/
std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim=0);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::size
*/
int64_t size(const Tensor& self, int64_t dim);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::stride
*/
int64_t stride(const Tensor& self, int64_t dim);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::is_same_size
*/
bool is_same_size(const Tensor& self, const Tensor& other);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::permute
*/
Tensor permute(const Tensor& self, IntList dims);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::expand
*/
Tensor expand(const Tensor& self, IntList size);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::squeeze
*/
Tensor squeeze(const Tensor& self);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::squeeze
*/
Tensor squeeze(const Tensor& self, int64_t dim);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::squeeze_
*/
Tensor & squeeze_(Tensor& self);
/*
[NativeFunction]
type_method_definition_dispatch: at::native::squeeze_
*/
Tensor & squeeze_(Tensor& self, int64_t dim);

/*
[NativeFunction]
type_method_definition_dispatch: at::native::unsqueeze
*/
Tensor unsqueeze(const Tensor& self, int64_t dim);

/*
[NativeFunction]
variants: method, function
type_method_definition_dispatch: at::native::unsqueeze_
*/
Tensor & unsqueeze_(Tensor& self, int64_t dim);

/*
[NativeFunction]
variants: function
type_method_definition_dispatch: at::native::stack
*/
Tensor stack(TensorList tensors, int64_t dim=0);

/*
[NativeFunction]
variants: function
type_method_definition_level: backend
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
type_method_definition_level: backend
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
