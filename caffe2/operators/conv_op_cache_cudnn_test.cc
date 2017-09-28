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

#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

TEST(AlgorithmsCacheTest, CachesCorrectly) {
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<TIndex>(1), std::vector<TIndex>(1), []() { return 5; });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<TIndex>(1), std::vector<TIndex>(1), []() { return 10; });

  EXPECT_EQ(res2, 5);
}

TEST(AlgorithmsCacheTest, DoesNotCacheEmptyKeys) {
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<TIndex>(), std::vector<TIndex>(), []() { return 5; });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<TIndex>(), std::vector<TIndex>(), []() { return 10; });

  EXPECT_EQ(res2, 10);
}

TEST(AlgorithmsCacheTest, KeysDifferIfOneVectorIsEmpty) {
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<TIndex>(1, 10), std::vector<TIndex>(), []() { return 5; });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<TIndex>(), std::vector<TIndex>(1, 10), []() { return 10; });

  EXPECT_EQ(res2, 10);
}

} // namespace caffe2
