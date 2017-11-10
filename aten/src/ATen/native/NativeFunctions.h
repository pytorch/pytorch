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
arg: Tensor self
arg: int64_t split_size
arg: int64_t dim=0
variants: method, function
type_method_definition_dispatch: at::native::split
[/NativeFunction]
*/
std::vector<Tensor> split(const Tensor &self, int64_t split_size, int64_t dim=0);

/*
[NativeFunction]
arg: Tensor self
arg: int64_t chunks
arg: int64_t dim=0
variants: method, function
type_method_definition_dispatch: at::native::chunk
[/NativeFunction]
*/
std::vector<Tensor> chunk(const Tensor &self, int64_t chunks, int64_t dim=0);

/*
[NativeFunction]
arg: Tensor self
arg: int64_t dim
variants: method, function
type_method_definition_dispatch: at::native::size
[/NativeFunction]
*/
int64_t size(const Tensor &self, int64_t dim);

/*
[NativeFunction]
arg: Tensor self
arg: int64_t dim
variants: method, function
type_method_definition_dispatch: at::native::stride
[/NativeFunction]
*/
int64_t stride(const Tensor &self, int64_t dim);

/*
[NativeFunction]
arg: Tensor self
arg: Tensor other
variants: method, function
type_method_definition_dispatch: at::native::is_same_size
[/NativeFunction]
*/
bool is_same_size(const Tensor &self, const Tensor &other);

/*
[NativeFunction]
arg: Tensor self
arg: IntList dims
variants: method, function
type_method_definition_dispatch: at::native::permute
[/NativeFunction]
*/
Tensor permute(const Tensor & self, IntList dims);

/*
[NativeFunction]
arg: Tensor self
arg: IntList size
variants: method, function
type_method_definition_dispatch: at::native::expand
[/NativeFunction]
*/
Tensor expand(const Tensor &self, IntList size);

/*
[NativeFunction]
arg: Tensor self
variants: method, function
type_method_definition_dispatch: at::native::squeeze
[/NativeFunction]
*/
Tensor squeeze(const Tensor & self);

/*
[NativeFunction]
arg: Tensor self
arg: int64_t dim
variants: method, function
type_method_definition_dispatch: at::native::squeeze
[/NativeFunction]
*/
Tensor squeeze(const Tensor & self, int64_t dim);

/*
[NativeFunction]
name: squeeze_
arg: Tensor self
variants: method, function
type_method_definition_dispatch: at::native::squeeze_
[/NativeFunction]
*/
Tensor & squeeze_(Tensor & self);
/*
[NativeFunction]
arg: Tensor self
arg: int64_t dim
variants: method, function
type_method_definition_dispatch: at::native::squeeze_
[/NativeFunction]
*/
Tensor & squeeze_(Tensor & self, int64_t dim);

/*
[NativeFunction]
arg: Tensor self
arg: int64_t dim
variants: method, function
type_method_definition_dispatch: at::native::unsqueeze
[/NativeFunction]
*/
Tensor unsqueeze(const Tensor & self, int64_t dim);

/*
[NativeFunction]
arg: Tensor self
arg: int64_t dim
variants: method, function
type_method_definition_dispatch: at::native::unsqueeze_
[/NativeFunction]
*/
Tensor & unsqueeze_(Tensor & self, int64_t dim);

/*
[NativeFunction]
arg: TensorList tensors
arg: int64_t dim=0
variants: function
type_method_definition_dispatch: at::native::stack
[/NativeFunction]
*/
Tensor stack(TensorList tensors, int64_t dim=0);

/*
[NativeFunction]
arg: Tensor input
arg: Tensor rois
arg: int64_t pooledHeight
arg: int64_t pooledWidth
arg: double spatialScale
variants: function
type_method_definition_level: backend
type_method_definition_dispatch: {
  - CPU: at::native::SpatialRoIPooling_forward
  - CUDA: at::native::SpatialRoIPooling_forward_cuda
}
[/NativeFunction]
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
arg: Tensor input
arg: Tensor rois
arg: int64_t pooledHeight
arg: int64_t pooledWidth
arg: double spatialScale
arg: Tensor gradOutput
arg: Tensor argmaxes
variants: function
type_method_definition_level: backend
type_method_definition_dispatch: {
  - CPU: at::native::SpatialRoIPooling_backward
  - CUDA: at::native::SpatialRoIPooling_backward_cuda
}
[/NativeFunction]
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
