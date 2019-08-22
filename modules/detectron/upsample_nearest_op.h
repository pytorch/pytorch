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

#ifndef UPSAMPLE_NEAREST_OP_H_
#define UPSAMPLE_NEAREST_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#ifdef CAFFE2_USE_MKLDNN
#include <caffe2/ideep/ideep_utils.h>
#endif

namespace caffe2 {

template <typename T, class Context>
class UpsampleNearestOp final : public Operator<Context> {
 public:
  UpsampleNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(this->template GetSingleArgument<int>("scale", 2)) {
    DCHECK_GE(scale_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);

    auto out_shape = X.sizes().vec();
    out_shape[X.dim() - 1] *= scale_;
    out_shape[X.dim() - 2] *= scale_;
    auto* Y = Output(0, out_shape, at::dtype<T>());

    int d1;
    int d2;
    int d3;
    if (X.dim() == 3) {
      d1 = Y->dim32(0);
      d2 = Y->dim32(1);
      d3 = Y->dim32(2);
    } else {
      d1 = Y->dim32(0) * Y->dim32(1);
      d2 = Y->dim32(2);
      d3 = Y->dim32(3);
    }

    const T *input_data = X.template data<T>();
    T *output_data = Y->template mutable_data<T>();
    int scaled_d2 = d2 / scale_;
    int scaled_d3 = d3 / scale_;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < d1; ++i) {
      for (int j = 0; j < d2; ++j) {
        for (int u = 0; u < d3; ++u) {
          int ii = (i * d2 + j) * d3 + u;
          int scaled_u = u / scale_;
          int scaled_j = j / scale_;
          int ipidx = ((i * scaled_d2) + scaled_j) * scaled_d3 + scaled_u;
          output_data[ii] = input_data[ipidx];
        }
      }
    }

    return true;
  }

 protected:
  int scale_;
};

template <typename T, class Context>
class UpsampleNearestGradientOp final : public Operator<Context> {
 public:
  UpsampleNearestGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(this->template GetSingleArgument<int>("scale", 2)) {
    DCHECK_GE(scale_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int scale_;
};

#ifdef CAFFE2_USE_MKLDNN
USE_IDEEP_DEF_ALIASES();

template <typename T>
void getData(T Ydata, const T Xdata, 
  int d0, int d1, int d2, int d3, int upsample_scale) {
  int scaled_d1 = d1 / upsample_scale;
  int scaled_d2 = d2 / upsample_scale;
  
  for (int j = 0; j < d0; ++j) {    
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int u = 0; u < d1; ++u) {
      for (int v = 0; v < d2; ++v) {
        for (int k = 0; k < d3; ++k) {
          int scaled_u = u / upsample_scale;
          int scaled_v = v / upsample_scale;
          int ii = ((j * d1 + u) * d2 + v) * d3 + k;
          int ipidx = ((j * scaled_d1 + scaled_u) * scaled_d2 + scaled_v) * d3 + k;
          Ydata[ii] = Xdata[ipidx];
        }
      }
    }
  }
}

class IDEEPUpsampleNearestOp final : public IDEEPOperator {
public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPUpsampleNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        upsample_scale_(this->template GetSingleArgument<int>("scale", 2)) {
  }
  virtual ~IDEEPUpsampleNearestOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto X_ = X;
    bool nhwc_format = X_.is_nhwc_format();
    if (!X_.is_nchw_channel_blocking() && !X_.is_nhwc_format()) {
      X_.init({X.get_dims(), X.get_data_type(), iformat::nchw});
      X_.feed_from(X);
    }

    auto Y_dims = X_.get_dims();
    int d0, d1, d2, d3;
    if (X_.ndims() == 3) {
      d0 = 1;
      d1 = Y_dims[0];
      d2 = Y_dims[1] * upsample_scale_;
      d3 = Y_dims[2] * upsample_scale_;
      Y_dims = {d1, d2, d3};
    } else {
      d0 = Y_dims[0];
      d1 = Y_dims[1];
      d2 = Y_dims[2] * upsample_scale_;
      d3 = Y_dims[3] * upsample_scale_;
      Y_dims = {d0, d1, d2, d3};
    }

    int c_blocking = X_.is_nhwc_format() ? Y_dims[1] : X_.get_block_dims()[1];
    int nc_num = d0 * ceil(float(d1) / c_blocking);
   
    auto* Y = Output(OUTPUT);
    Y->init({Y_dims, X_.get_data_type(), X_.get_internal_format()});
 
    if (X_.get_data_type() == idtype::s8 || 
      X_.get_data_type() == idtype::u8) {
      Y->set_scale(X.get_scale());
      if (X_.get_data_type() == idtype::s8) {
        auto Xdata = static_cast<int8_t*>(X_.get_data_handle());
        auto Ydata = static_cast<int8_t*>(Y->get_data_handle());
        getData<int8_t*>(Ydata, Xdata, nc_num, d2, d3, c_blocking, upsample_scale_);
      } else {
        auto Xdata = static_cast<uint8_t*>(X_.get_data_handle());
        auto Ydata = static_cast<uint8_t*>(Y->get_data_handle());
        getData<uint8_t*>(Ydata, Xdata, nc_num, d2, d3, c_blocking, upsample_scale_);
      }
    } else {
      auto Xdata = static_cast<float*>(X_.get_data_handle());
      auto Ydata = static_cast<float*>(Y->get_data_handle());
      getData<float*>(Ydata, Xdata, nc_num, d2, d3, c_blocking, upsample_scale_);
    } 
    return true;
  }

 protected:
  int upsample_scale_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};
#endif // CAFFE2_USE_MKLDNN
} // namespace caffe2

#endif // UPSAMPLE_NEAREST_OP_H_
