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
  CONV_ALGORITHM_MAX
};

enum FusionType {
  FUSION_UNKNOWN = 0,
  FUSION_CONV_RELU = 1,
  FUSION_CONV_SUM = 2,
  FUSION_CONV_SUM_RELU = 3,
  FUSION_MAX
};

#define USE_IDEEP_DEF_ALIASES()                                                \
  /* the tensor type created/handled by iDEEP  */                              \
  using itensor = ideep::tensor;                                               \
  /* the date layout of iDEEP tensor */                                        \
  using iformat = ideep::format_tag;                                           \
  /* the scales for iDEEP tensor with different data type */                   \
  using iscale = ideep::scale_t;                                               \
  /* the detial algorithm for iDEEP operators, e.g. winograd */                \
  using ialgo = ideep::algorithm;                                              \
  /* the kind of propagation for iDEEP operators, e.g. forward, training */    \
  using iprop = ideep::prop_kind;                                              \
  /* the kind of low precision operators, e.g. signed/unsigned activation */   \
  using ilowp_kind = ideep::lowp_kind;                                         \
  /* the data type of iDEEP tensor, e.g. f32, u8, s8 */                        \
  using idtype = ideep::tensor::data_type;                                     \
  /* the descriptor of iDEEP tensor */                                         \
  using itdesc = ideep::tensor::descriptor;                                    \
  /* the attribute for operator to describe the details of inputs&fusion */    \
  using iattr = ideep::attr_t;                                                 \
  /* the detail flags for batch normalization */                               \
  using ibn_flag = ideep::batch_normalization_flag;

} // namespace caffe2
