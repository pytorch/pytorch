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

#ifndef CAFFE2_OPERATORS_SPARSE_FUNHASH_OP_H_
#define CAFFE2_OPERATORS_SPARSE_FUNHASH_OP_H_

#include <xxhash.h>
#include <array>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#define HASH_MAGIC 0x9e3779b97f4a7c15

#define USE_SIGN

namespace caffe2 {

template <typename T, class Context>
class SparseFunHashOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseFunHashOp(const OperatorDef& operator_def, Workspace* ws)
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
      for (const auto i : c10::irange(num_nz_ent)) {
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

    for (const auto j : c10::irange(num_nz_ent)) {
      int64_t cur_seg = seg_data[j];
      int64_t cur_key = key_data[j];
      T cur_val = val_data[j];
      int64_t output_stride = cur_seg * num_outputs_;
      for (const auto i : c10::irange(num_outputs_)) {
        T sum = 0;
        for (const auto k : c10::irange(num_alpha)) {
          // The hash function takes as input three integers:
          // 1. feature index
          // 2. output index
          // 3. alpha index
          // 4. magic number to improve hashing
          hash_data[0] = cur_key;
          hash_data[1] = i;
          hash_data[2] = k;
          hash_data[3] = HASH_MAGIC;

          uint64_t hash = XXH64(hash_data.data(), hash_data.size(), seed_);

#ifdef USE_SIGN
          // Use the least significant bit for sign, the rest for weights.
          int64_t index = (hash >> 1) % num_weight;
          T cur_weight = weight_data[index];
          if (hash & 1) {
            cur_weight = -cur_weight;
          }
#else
          int64_t index = hash % num_weight;
          T cur_weight = weight_data[index];
#endif

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
class SparseFunHashGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseFunHashGradientOp(const OperatorDef& operator_def, Workspace* ws)
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

      auto* grad_alpha = Output(2, alpha.sizes(), at::dtype<T>());
      grad_alpha_data = grad_alpha->template mutable_data<T>();
      memset(grad_alpha_data, 0, sizeof(T) * num_alpha);
    }

    const auto* seg_data = seg.template data<int>();

    int64_t num_weight = weight.size(0);
    int64_t num_nz_ent = seg.size(0);

    int64_t grad_weight_size = num_nz_ent * num_outputs_ * num_alpha;

    auto* grad_weight_val = Output(0, {grad_weight_size}, at::dtype<T>());
    T* grad_weight_val_data = grad_weight_val->template mutable_data<T>();

    auto* grad_weight_ind = Output(1, {grad_weight_size}, at::dtype<int64_t>());
    auto* grad_weight_ind_data =
        grad_weight_ind->template mutable_data<int64_t>();

    const auto* grad_out_data = grad_out.template data<T>();
    const auto* weight_data = weight.template data<T>();
    const auto* alpha_data = adaptive_ ? Input(5).template data<T>() : 0;
    const auto* val_data = val.template data<T>();
    const auto* key_data = key.template data<int64_t>();

    int64_t w_ind = 0;
    for (const auto j : c10::irange(num_nz_ent)) {
      int64_t cur_seg = seg_data[j];
      int64_t cur_key = key_data[j];
      T cur_val = val_data[j];
      int64_t grad_out_stride = cur_seg * num_outputs_;
      for (const auto i : c10::irange(num_outputs_)) {
        T grad_out_scale = grad_out_data[grad_out_stride + i] * cur_val;
        for (const auto k : c10::irange(num_alpha)) {
          hash_data[0] = cur_key;
          hash_data[1] = i;
          hash_data[2] = k;
          hash_data[3] = HASH_MAGIC;

          uint64_t hash = XXH64(hash_data.data(), hash_data.size(), seed_);

          T cur_grad_out_scale = grad_out_scale;
#ifdef USE_SIGN
          int64_t index = (hash >> 1) % num_weight;
          if (hash & 1) {
            cur_grad_out_scale = -cur_grad_out_scale;
          }
#else
          int64_t index = hash % num_weight;
#endif

          if (adaptive_) {
            grad_alpha_data[k] += cur_grad_out_scale * weight_data[index];
            grad_weight_val_data[w_ind] = alpha_data[k] * cur_grad_out_scale;
          } else {
            grad_weight_val_data[w_ind] = cur_grad_out_scale;
          }
          grad_weight_ind_data[w_ind] = index;
          ++w_ind;
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

#endif // CAFFE2_OPERATORS_SPARSE_FUNHASH_OP_H_
