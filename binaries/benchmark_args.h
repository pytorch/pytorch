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

#include "c10/util/Flags.h"

C10_DEFINE_string(
    backend,
    "builtin",
    "The backend to use when running the model. The allowed "
    "backend choices are: builtin, default, nnpack, eigen, mkl, cuda");

C10_DEFINE_string(init_net, "", "The given net to initialize any parameters.");
C10_DEFINE_string(
    input,
    "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");
C10_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
C10_DEFINE_string(
    input_file,
    "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");
C10_DEFINE_string(
    input_type,
    "float",
    "Input type when specifying the input dimension."
    "The supported types are float, uint8_t.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");
C10_DEFINE_bool(
    measure_memory,
    false,
    "Whether to measure increase in allocated memory while "
    "loading and running the net.");
C10_DEFINE_string(net, "", "The given net to benchmark.");
C10_DEFINE_string(
    output,
    "",
    "Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.");
C10_DEFINE_string(
    output_folder,
    "",
    "The folder that the output should be written to. This "
    "folder must already exist in the file system.");
C10_DEFINE_bool(
    run_individual,
    false,
    "Whether to benchmark individual operators.");
C10_DEFINE_int(
    sleep_before_run,
    0,
    "The seconds to sleep before starting the benchmarking.");
C10_DEFINE_int(
    sleep_between_iteration,
    0,
    "The seconds to sleep between the individual iterations.");
C10_DEFINE_int(
    sleep_between_net_and_operator,
    0,
    "The seconds to sleep between net and operator runs.");
C10_DEFINE_bool(
    text_output,
    false,
    "Whether to write out output in text format for regression purpose.");
C10_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
C10_DEFINE_bool(
    wipe_cache,
    false,
    "Whether to evict the cache before running network.");
