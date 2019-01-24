#pragma once

#include <caffe2/core/macros.h>  // For caffe2 macros.

// All caffe2 ideep related headers
#include <ideep.hpp>
#include <caffe2/ideep/utils/ideep_context.h>
#include <caffe2/ideep/utils/ideep_operator.h>

namespace caffe2 {

enum ConvAlgorithm {
  CONV_ALGORITHM_AUTO = 0,
  CONV_ALGORITHM_WINOGRAD = 1,
  CONV_ALGORITHM_MAX = CONV_ALGORITHM_WINOGRAD + 1
};

enum FusionType {
  FUSION_UNKNOWN = 0,
  FUSION_CONV_RELU = 1,
  FUSION_CONV_SUM = 2,
  FUSION_CONV_SUM_RELU = 3,
  FUSION_MAX = FUSION_CONV_SUM_RELU + 1
};

#define USE_IDEEP_DEF_ALIASES()                                                \
  using ikey = ideep::key_t;                                                   \
  using itensor = ideep::tensor;                                               \
  using iformat = ideep::format;                                               \
  using iscale = ideep::scale_t;                                               \
  using ialgo = ideep::algorithm;                                              \
  using iprop = ideep::prop_kind;                                              \
  using ilowp_kind = ideep::lowp_kind;                                         \
  using ipadding = ideep::padding_kind;                                        \
  using idtype = ideep::tensor::data_type;                                     \
  using itdesc = ideep::tensor::descriptor;                                    \
  using iattr = ideep::descriptor_group::attr_t;                               \
  using ibn_flag = ideep::batch_normalization_flag;                            \

} // namespace caffe2
