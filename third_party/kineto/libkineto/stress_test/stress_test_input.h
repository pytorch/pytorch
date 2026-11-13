/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <folly/dynamic.h>
#include <folly/json.h>
#include <fstream>
#include <iostream>
#include <string>

#include "kineto/libkineto/stress_test/random_ops_stress_test.cuh"
#include "kineto/libkineto/stress_test/tensor_cache.cuh"

#pragma once

namespace kineto_stress_test {

void do_checks_and_fixes(
    stress_test_args* test_args,
    tensor_cache_args* cache_args) {
  if (test_args->sz_nccl_buff_KB <= cache_args->sz_max_tensor_KB) {
    std::cout
        << "sz_nccl_buff_KB must be greater than sz_max_tensor_KB. Setting it now."
        << std::endl;
    test_args->sz_nccl_buff_KB = cache_args->sz_max_tensor_KB + 1;
  }
}

void read_inputs_from_json(
    std::string sJsonFile,
    stress_test_args* test_args,
    tensor_cache_args* cache_args) {
  std::ifstream fJson(sJsonFile.c_str());
  std::string sJson;

  if (!fJson.fail()) {
    fJson.seekg(0, std::ios::end);
    sJson.reserve(fJson.tellg());
    fJson.seekg(0, std::ios::beg);
    sJson.assign(
        (std::istreambuf_iterator<char>(fJson)),
        std::istreambuf_iterator<char>());
    fJson.close();

    folly::dynamic sJsonParsed = folly::parseJson(sJson);

    folly::dynamic jsonTestArgs = sJsonParsed["test_args"];
    test_args->num_operations =
        (uint32_t)jsonTestArgs["num_operations"].asInt();
    test_args->num_cuda_streams =
        (uint32_t)jsonTestArgs["num_cuda_streams"].asInt();
    test_args->prob_cuda_malloc =
        (double)jsonTestArgs["prob_cuda_malloc"].asDouble();
    test_args->min_iters_kernel =
        (uint32_t)jsonTestArgs["min_iters_kernel"].asInt();
    test_args->max_iters_kernel =
        (uint32_t)jsonTestArgs["max_iters_kernel"].asInt();
    test_args->thread_blocks_per_kernel =
        (uint32_t)jsonTestArgs["thread_blocks_per_kernel"].asInt();
    test_args->kernel_block_size =
        (uint32_t)jsonTestArgs["kernel_block_size"].asInt();
    test_args->memset_prob = (double)jsonTestArgs["memset_prob"].asDouble();
    test_args->min_idle_us = (uint32_t)jsonTestArgs["min_idle_us"].asInt();
    test_args->max_idle_us = (uint32_t)jsonTestArgs["max_idle_us"].asInt();
    test_args->do_warmup = (bool)jsonTestArgs["do_warmup"].asBool();
    test_args->simulate_host_time =
        (bool)jsonTestArgs["simulate_host_time"].asBool();
    test_args->num_workers = (uint32_t)jsonTestArgs["num_workers"].asInt();
    test_args->use_uvm_buffers = (bool)jsonTestArgs["use_uvm_buffers"].asBool();
    test_args->uvm_kernel_prob =
        (double)jsonTestArgs["uvm_kernel_prob"].asDouble();
    test_args->parallel_uvm_alloc =
        (bool)jsonTestArgs["parallel_uvm_alloc"].asBool();
    test_args->uvm_len = (uint64_t)jsonTestArgs["uvm_len"].asDouble();
    test_args->is_multi_rank = (bool)jsonTestArgs["is_multi_rank"].asBool();
    test_args->sz_nccl_buff_KB =
        (uint32_t)jsonTestArgs["sz_nccl_buff_KB"].asInt();
    test_args->num_iters_nccl_sync =
        (uint32_t)jsonTestArgs["num_iters_nccl_sync"].asInt();
    test_args->pre_alloc_streams =
        (bool)jsonTestArgs["pre_alloc_streams"].asBool();
    test_args->use_memcpy_stream =
        (bool)jsonTestArgs["use_memcpy_stream"].asBool();
    test_args->use_uvm_stream = (bool)jsonTestArgs["use_uvm_stream"].asBool();
    test_args->monitor_mem_usage =
        (bool)jsonTestArgs["monitor_mem_usage"].asBool();
    test_args->trace_delay_us =
        (uint32_t)jsonTestArgs["trace_delay_us"].asInt();
    test_args->trace_length_us =
        (uint32_t)jsonTestArgs["trace_length_us"].asInt();
    test_args->cupti_buffer_mb =
        (uint32_t)jsonTestArgs["cupti_buffer_mb"].asInt();

    folly::dynamic cacheArgs = sJsonParsed["cache_args"];
    cache_args->sz_cache_KB = (uint32_t)cacheArgs["sz_cache_KB"].asInt();
    cache_args->sz_GPU_memory_KB =
        (uint32_t)cacheArgs["sz_GPU_memory_KB"].asInt();
    cache_args->sz_min_tensor_KB =
        (uint32_t)cacheArgs["sz_min_tensor_KB"].asInt();
    cache_args->sz_max_tensor_KB =
        (uint32_t)cacheArgs["sz_max_tensor_KB"].asInt();
    cache_args->prob_h2d = (double)cacheArgs["prob_h2d"].asDouble();
    cache_args->prob_d2h = (double)cacheArgs["prob_d2h"].asDouble();
    cache_args->num_increments = (uint32_t)cacheArgs["num_increments"].asInt();
    cache_args->num_pairs_per_increment =
        (uint32_t)cacheArgs["num_pairs_per_increment"].asInt();
  } else {
    std::cout << "Reading input " << sJsonFile << " failed !" << std::endl;
  }

  do_checks_and_fixes(test_args, cache_args);
}

} // namespace kineto_stress_test
