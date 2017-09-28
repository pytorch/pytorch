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

#include "arm_neon_support.h"

void interleaveSlice(void* output,
                     const float* input,
                     size_t width,
                     size_t height,
                     size_t row_stride,
                     uint16_t input_channels);
void deInterleaveSlice(float* output,
                       const void* input,
                       size_t width,
                       size_t height,
                       size_t input_stride,
                       uint32_t output_channels);
