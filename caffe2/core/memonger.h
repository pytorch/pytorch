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

#ifndef CAFFE2_CORE_MEMONGER_H_
#define CAFFE2_CORE_MEMONGER_H_

#include <unordered_set>

#include "caffe2/core/common.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace memonger {

NetDef optimize_inference_net(
    const NetDef& net,
    const std::set<string>& static_blobs);

NetDef compute_blob_recycling_for_dag(
    const NetDef& net,
    const std::vector<string>& heads,
    const std::vector<int>& op_indices,
    const std::unordered_set<string>& shareable_blob_names,
    const string& namescope,
    const std::unordered_set<string>& dont_share_blob_names,
    const std::unordered_map<string, vector<int>>& blob_shapes);

} // memonger
} // caffe2

#endif
