#pragma once

#include <caffe2/core/macros.h>  // For caffe2 macros.

// All caffe2 ideep related headers
#include <ideep.hpp>
#include <caffe2/ideep/utils/ideep_context.h>
#include <caffe2/ideep/utils/ideep_operator.h>

namespace caffe2 {

#define USE_IDEEP_DEF_ALIASES()                                                \
  using itensor = ideep::tensor;                                               \
  using iformat = ideep::format;                                               \
  using ialgo = ideep::algorithm;                                              \
  using iprop = ideep::prop_kind;                                              \
  using ipadding = ideep::padding_kind;                                        \
  using iattr = ideep::descriptor_group::attr_t;                               \
  using ibn_flag = ideep::batch_normalization_flag;

} // namespace caffe2
