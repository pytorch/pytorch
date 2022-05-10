#pragma once

#include <ATen/core/DimVector.h>
#include <c10/core/TensorImpl.h>
#include <caffe2/core/context_base.h>
#include <caffe2/core/context_base.h>

namespace caffe2 {
  using at::ToVectorint64_t;
  using at::size_from_dim_;
  using at::size_to_dim_;
  using at::size_between_dim_;
  using at::canonical_axis_index_;
  using at::TensorImpl;
}
