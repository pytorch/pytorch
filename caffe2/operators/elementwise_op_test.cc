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

#include "caffe2/operators/elementwise_op_test.h"

#include "caffe2/core/flags.h"

CAFFE2_DECLARE_string(caffe_test_root);

template <>
void CopyVector<caffe2::CPUContext, bool>(const int N, const bool* x, bool* y) {
  memcpy(y, x, N * sizeof(bool));
}

template <>
void CopyVector<caffe2::CPUContext, int32_t>(
    const int N,
    const int32_t* x,
    int32_t* y) {
  memcpy(y, x, N * sizeof(int32_t));
}

TEST(ElementwiseCPUTest, And) {
  elementwiseAnd<caffe2::CPUContext>();
}

TEST(ElementwiseTest, Or) {
  elementwiseOr<caffe2::CPUContext>();
}

TEST(ElementwiseTest, Xor) {
  elementwiseXor<caffe2::CPUContext>();
}

TEST(ElementwiseTest, Not) {
  elementwiseNot<caffe2::CPUContext>();
}

TEST(ElementwiseTest, EQ) {
  elementwiseEQ<caffe2::CPUContext>();
}
