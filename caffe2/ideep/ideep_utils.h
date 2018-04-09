/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
