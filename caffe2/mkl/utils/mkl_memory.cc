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

#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

CAFFE2_DEFINE_bool(
    caffe2_mkl_implicit_layout_change, false,
    "Controls the behavior when we call View() on an MKLMemory: if it is set "
    "true, then the View() function will actually change the underlying "
    "storage. If it is set false, an implicit copy is triggered but the "
    "original storage is not affected."
    );

namespace caffe2 {

CAFFE_KNOWN_TYPE(mkl::MKLMemory<float>);
CAFFE_KNOWN_TYPE(mkl::MKLMemory<double>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
