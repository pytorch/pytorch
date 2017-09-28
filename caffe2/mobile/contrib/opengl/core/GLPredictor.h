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

#include "GLImage.h"
#include "caffe2/core/net.h"
#include "caffe2/core/predictor.h"

namespace caffe2 {
class GLPredictor : public Predictor {
 public:
  GLPredictor(const NetDef& init_net,
              const NetDef& run_net,
              bool use_texture_input = false,
              Workspace* parent = nullptr);

  template <class T>
  bool run(std::vector<GLImageVector<T>*>& inputs, std::vector<const GLImageVector<T>*>* outputs);

  ~GLPredictor();
};
} // namespace caffe2
