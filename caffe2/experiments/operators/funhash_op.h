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

#ifndef CAFFE2_OPERATORS_FUNHASH_OP_H_
#define CAFFE2_OPERATORS_FUNHASH_OP_H_

#include <xxhash.h>
#include <array>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#define SIGN_MAGIC 0x9e3779b97f4a7c15
#define INDEX_MAGIC 0xf39cc0605cedc834

#define USE_SIGN

namespace caffe2 {

template <typename T, class Context>
class FunHashOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FunHashOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_outputs_(
            OperatorBase::GetSingleArgument<int64_t>("num_outputs", -1)),
        num_segments_(
            OperatorBase::GetSingleArgument<int64_t>("num_segments", -1)),
        seed_(OperatorBase::GetSingleArgument<uint64_t>("seed", 0)) {
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("num_outputs"),
        "Argument `num_outputs` is missing.");
    // If alpha is provided, use adaptive hashing parameterized by alpha.
    adaptive_ = (InputSize() == 5);
  }

  bool RunOnDevice() override {
    const auto& val = Input(0);
    const auto& key = Input(1);
    const auto& seg = Input(2);
    const auto& weight = Input(3);

    int64_t num_alpha = 1;
    if (adaptive_) {
      const auto& alpha = Input(4);
      num_alpha = alpha.size(0);
    }

    const auto* seg_data = seg.template data<int>();

    int64_t num_weight = weight.size(0);
    int64_t num_nz_ent = seg.size(0);

    int64_t n_segments = num_segments_;
    if (num_segments_ == -1) {
      for (int64_t i = 0; i < num_nz_ent; ++i) {
        if (seg_data[i] > n_segments) {
          n_segments = seg_data[i];
        }
      }
      ++n_segments;
    }

    auto* output = Output(0, {n_segments, num_outputs_}, at::dtype<T>());

    T* output_data = output->template mutable_data<T>();

    memset(output_data, 0, sizeof(T) * n_segments * num_outputs_);

    const auto* weight_data = weight.template data<T>();
    const auto* alpha_data = adaptive_ ? Input(4).template data<T>() : 0;
    const auto* val_data = val.template data<T>();
    const auto* key_data = key.template data<int64_t>();

    for (int64_t j = 0; j < num_nz_ent; ++j) {
      int64_t cur_seg = seg_data[j];
      int64_t cur_key = key_data[j];
      T cur_val = val_data[j];
      int64_t output_stride = cur_seg * num_outputs_;
      for (int64_t i = 0; i < num_outputs_; ++i) {
        T sum = 0;
        for (int64_t k = 0; k < num_alpha; ++k) {
          uint64_t hash;
          // The hash function takes as input four integers:
          // 1. feature index
          // 2. output index
          // 3. alpha index
          // 4. magic number: SIGN_MAGIC for sign (-1/+1)
          //                  INDEX_MAGIC for weight index
          hash_data[0] = cur_key;
          hash_data[1] = i;
          hash_data[2] = k;

          hash_data[3] = INDEX_MAGIC;
          hash = XXH64(hash_data.data(), hash_data.size(), seed_);
          int64_t index = hash % num_weight;

          T cur_weight = weight_data[index];
#ifdef USE_SIGN
          hash_data[3] = SIGN_MAGIC;
          hash = XXH64(hash_data.data(), hash_data.size(), seed_);
          if (hash % 2) {
            cur_weight = -cur_weight;
          }
#endif // USE_SIGN

          if (adaptive_) {
            sum += cur_weight * alpha_data[k];
          } else {
            sum += cur_weight;
          }
        }
        output_data[output_stride + i] += sum * cur_val;
      }
    }

    return true;
  }

 protected:
  int64_t num_outputs_;
  int64_t num_segments_;
  uint64_t seed_;
  std::array<uint64_t, 4> hash_data;
  bool adaptive_;
};

template <typename T, class Context>
class FunHashGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FunHashGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_outputs_(
            OperatorBase::GetSingleArgument<int64_t>("num_outputs", -1)),
        seed_(OperatorBase::GetSingleArgument<uint64_t>("seed", 0)) {
    adaptive_ = (InputSize() == 6);
  }

  bool RunOnDevice() override {
    const auto& grad_out = Input(0);
    const auto& val = Input(1);
    const auto& key = Input(2);
    const auto& seg = Input(3);
    const auto& weight = Input(4);

    int64_t num_alpha = 1;
    T* grad_alpha_data = 0;

    if (adaptive_) {
      const auto& alpha = Input(5);
      num_alpha = alpha.size(0);

      auto* grad_alpha = Output(1, alpha.sizes(), at::dtype<T>());
      grad_alpha_data = grad_alpha->template mutable_data<T>();
      memset(grad_alpha_data, 0, sizeof(T) * num_alpha);
    }

    const auto* seg_data = seg.template data<int>();

    int64_t num_weight = weight.size(0);
    int64_t num_nz_ent = seg.size(0);

    auto* grad_weight = Output(0, weight.sizes(), at::dtype<T>());
    T* grad_weight_data = grad_weight->template mutable_data<T>();

    const auto* grad_out_data = grad_out.template data<T>();
    const auto* weight_data = weight.template data<T>();
    const auto* alpha_data = adaptive_ ? Input(5).template data<T>() : 0;
    const auto* val_data = val.template data<T>();
    const auto* key_data = key.template data<int64_t>();

    memset(grad_weight_data, 0, sizeof(T) * num_weight);

    for (int64_t j = 0; j < num_nz_ent; ++j) {
      int64_t cur_seg = seg_data[j];
      int64_t cur_key = key_data[j];
      T cur_val = val_data[j];
      int64_t grad_out_stride = cur_seg * num_outputs_;
      for (int64_t i = 0; i < num_outputs_; ++i) {
        T grad_out_scale = grad_out_data[grad_out_stride + i] * cur_val;
        for (int64_t k = 0; k < num_alpha; ++k) {
          uint64_t hash;
          hash_data[0] = cur_key;
          hash_data[1] = i;
          hash_data[2] = k;

          hash_data[3] = INDEX_MAGIC;
          hash = XXH64(hash_data.data(), hash_data.size(), seed_);
          int64_t index = hash % num_weight;

          T cur_grad_out_scale = grad_out_scale;
#ifdef USE_SIGN
          hash_data[3] = SIGN_MAGIC;
          hash = XXH64(hash_data.data(), hash_data.size(), seed_);
          if (hash % 2) {
            cur_grad_out_scale = -cur_grad_out_scale;
          }
#endif // USE_SIGN

          if (adaptive_) {
            grad_alpha_data[k] += cur_grad_out_scale * weight_data[index];
            grad_weight_data[index] += alpha_data[k] * cur_grad_out_scale;
          } else {
            grad_weight_data[index] += cur_grad_out_scale;
          }
        }
      }
    }
    return true;
  }

 protected:
  int64_t num_outputs_;
  uint64_t seed_;
  std::array<uint64_t, 4> hash_data;
  bool adaptive_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FUNHASH_OP_H_
