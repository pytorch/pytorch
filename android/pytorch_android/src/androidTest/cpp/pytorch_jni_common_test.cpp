// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <ATen/core/type_factory.h>
#include "caffe2/android/pytorch_android/src/main/cpp/pytorch_jni_common.h"

using namespace ::testing;

TEST(pytorch_jni_common_test, newJIValueFromAtIValue) {
  auto dict = c10::impl::GenericDict(
      c10::dynT<c10::IntType>(), c10::dynT<c10::StringType>());
  auto dictCallback = [](auto&&) {
    return facebook::jni::local_ref<pytorch_jni::JIValue>{};
  };
  EXPECT_NO_THROW(pytorch_jni::JIValue::newJIValueFromAtIValue(
      dict, dictCallback, dictCallback));
}
