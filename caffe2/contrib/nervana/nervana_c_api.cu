/*
 * Copyright 2015 Baidu USA, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>
#include <string>
#include <map>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <mutex>
#include <tuple>
#include "nervana_c_api.h"

std::map<CUdevice, int> nervana_sm_counts_;
std::map<std::string, CUfunction> nervana_kernels_;
std::vector<CUmodule> nervana_modules_;

//for when we need to modify the above data structures
std::mutex nervana_load_kernels_mutex_;
std::mutex nervana_sm_count_mutex_;

extern "C" bool nervana_loadKernels(const char* const base_path_cstr) {
    std::lock_guard<std::mutex> lock(nervana_load_kernels_mutex_);

    //better would be a vector<string>, but there is a bug in nvcc that prevents this
    // (bug report filed) (fixed in 7.5)
    std::string names[36] = {
        "hgemm_nn_vec_128x128",
        "hgemm_nn_128x128",
        "hgemm_nt_vec_128x128",
        "hgemm_nt_128x128",
        "hgemm_tn_vec_128x128",
        "hgemm_tn_128x128",
        "hgemm_nn_vec_128x64",
        "hgemm_nn_128x64",
        "hgemm_tn_vec_128x64",
        "hgemm_tn_128x64",
        "hgemm_nn_vec_128x32",
        "hgemm_nn_128x32",
        "hgemm_tn_vec_128x32",
        "hgemm_tn_128x32",
        "hgemm_nn_32x128",
        "hgemm_nn_vec_32x128",
        "hgemm_nt_32x128",
        "hgemm_nt_vec_32x128",
        "sgemm_nn_vec_128x128",
        "sgemm_nn_128x128",
        "sgemm_nt_vec_128x128",
        "sgemm_nt_128x128",
        "sgemm_tn_vec_128x128",
        "sgemm_tn_128x128",
        "sgemm_nn_vec_128x64",
        "sgemm_nn_128x64",
        "sgemm_tn_vec_128x64",
        "sgemm_tn_128x64",
        "sgemm_nn_vec_128x32",
        "sgemm_nn_128x32",
        "sgemm_tn_vec_128x32",
        "sgemm_tn_128x32",
        "sgemm_nn_32x128",
        "sgemm_nn_vec_32x128",
        "sgemm_nt_32x128",
        "sgemm_nt_vec_32x128"
    };

    std::string base_path(base_path_cstr);

    for (auto kernel : names) {
        if (nervana_kernels_.count(kernel) > 0)
            continue;

        CUmodule module;

        std::string path = base_path + kernel + std::string(".cubin");
        CUresult res = cuModuleLoad(&module, path.c_str());

        if (res != CUDA_SUCCESS) {
            // std::cerr << "Failed to load: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_modules_.push_back(module);

        CUfunction function;
        res = cuModuleGetFunction(&function, module, kernel.c_str());
        if (res != CUDA_SUCCESS) {
            // std::cerr << "Failed to extract: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_kernels_.insert(std::make_pair(kernel, function));
    }

    return true;
}

extern "C" bool nervana_unloadKernels() {
    std::lock_guard<std::mutex> lock(nervana_load_kernels_mutex_);
    while(nervana_modules_.size() > 0) {
        auto module = nervana_modules_.back();
        CUresult res = cuModuleUnload(module);

        nervana_modules_.pop_back();

        if (res != CUDA_SUCCESS)
            return false;
    }

    nervana_kernels_.clear();

    return true;
}

extern "C" size_t nervana_randStateSizeBytes() {
    return 2048 * 32 * sizeof(int);
}

std::tuple<int, int, int> get_grid_dimensions(int grid, int m, int n, int sm_count, const std::string& trans)
{
    int sizeA, sizeB, threads;
    if (grid >= 0) {
        if (grid == 0) {
            sizeA = 32;
            sizeB = 128;
            threads = 128;
        } else if (grid == 1) {
            sizeA = 128;
            sizeB = 32;
            threads = 128;
        } else if (grid == 2) {
            sizeA = 128;
            sizeB = 64;
            threads = 128;
        } else if (grid == 3) {
            sizeA = 128;
            sizeB = 128;
            threads = 256;
        }
    } else {
        int sh = min(m, n);

        int size;
        if (sh < 384 - 16) {
            int sh128 = sh % 128;
            if (sh128 > 0 && sh128 < 112) {
                if (sh128 > 48 && sh128 <= 64) {
                    int sh64 = sh / 64;
                    int wide = max(m, n);
                    sh64 *= (wide / 128 + (wide % 128 != 0)) / sm_count;
                    if (sh64 > 1) {
                        size = 64;
                    }
                    else {
                        size = 32;
                    }
                }
                else {
                    size = 32;
                }
            }
            else {
                size = 128;
            }
        } else {
            size = 128;
        }

        if (m >= n) {
            if (trans == "nt") {
                size = 128;
            }
            sizeA = 128;
            sizeB = size;
        } else {
            if (trans == "tn") {
                size = 128;
            } else if (size == 64) {
                //temporary until kernels exist
                size = 32;
            }
            sizeA = size;
            sizeB = 128;
        }
        threads = (sizeA == 128 && sizeB == 128) ? 256 : 128;
    }

    return std::make_tuple(sizeA, sizeB, threads);
}

extern "C" bool nervana_sgemm(float *A, float *B, float *C,
                              bool a_t, bool b_t,
                              int m, int n, int k,
                              int lda, int ldb, int ldc,
                              float alpha, float beta,
                              unsigned int *rand_state,
                              bool stochastic_round, bool apply_relu,
                              CUstream stream, int grid
                             )
{
    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    std::string name = "sgemm_";

    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';

    name += trans;

    int sizeA, sizeB, threads;

    std::tie(sizeA, sizeB, threads) = get_grid_dimensions(grid, m, n, sm_count, trans);

    int k_vec = (sizeA == 32 || sizeB == 32) ? 4 : 16;

    if ( (trans == "tn" && m % 4 == 0  && n % 4 == 0) ||
         (trans == "nn" && k % k_vec == 0 && n % 4 == 0) ||
         (trans == "nt" && k % k_vec == 0)) {
         name += "_vec";
    }

    int gridA = m / sizeA + (m % sizeA != 0);
    int gridB = n / sizeB + (n % sizeB != 0);
    std::stringstream ss;
    ss << "_" << sizeA << "x" << sizeB;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    CUresult res;

    if (a_t)
        lda *= (8 * sizeof(float));

    if (!b_t)
        ldb *= (8 * sizeof(float));

    int zero = 0;
    void *args[17] = {&rand_state, &A, &B, &C, &lda, &ldb, &ldc, &m, &n, &k, &alpha, &beta, &flags,
                      &zero, &zero, &zero, &zero};

    res = cuLaunchKernel(nervana_kernels_[name],
                         1, gridA, gridB,
                         threads, 1, 1,
                         0,
                         stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << name << " " << res << std::endl;
        return false;
    }

    return true;
}

extern "C" bool nervana_hgemm(short *A, short *B, short *C,
                              bool a_t, bool b_t,
                              int m, int n, int k,
                              int lda, int ldb, int ldc,
                              float alpha, float beta,
                              unsigned int *rand_state,
                              bool stochastic_round, bool apply_relu,
                              CUstream stream, int grid
                             )
{
    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    std::string name = "hgemm_";

    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';

    name += trans;

    int sizeA, sizeB, threads;

    std::tie(sizeA, sizeB, threads) = get_grid_dimensions(grid, m, n, sm_count, trans);

    int k_vec = (sizeA == 32 || sizeB == 32) ? 4 : 16;

    if ( (trans == "tn" && m % 4 == 0 && n % 4 == 0) ||
         (trans == "nn" && k % k_vec == 0 && n % 4 == 0) ||
         (trans == "nt" && k % k_vec == 0)) {
         name += "_vec";
    }

    int gridA = m / sizeA + (m % sizeA != 0);
    int gridB = n / sizeB + (n % sizeB != 0);
    std::stringstream ss;
    ss << "_" << sizeA << "x" << sizeB;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    CUresult res;

    if (a_t)
        lda *= (8 * sizeof(short));

    if (!b_t)
        ldb *= (8 * sizeof(short));

    int zero = 0;
    void *args[17] = {&rand_state, &A, &B, &C, &lda, &ldb, &ldc, &m, &n, &k, &alpha, &beta, &flags,
                      &zero, &zero, &zero, &zero};

    res = cuLaunchKernel(nervana_kernels_[name],
                         1, gridA, gridB,
                         threads, 1, 1,
                         0,
                         stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << name << " " << res << std::endl;
        return false;
    }

    return true;
}

extern "C" bool nervana_sgemm_colmajor(float *A, float *B, float *C,
                                       bool a_t, bool b_t,
                                       int m, int n, int k,
                                       int lda, int ldb, int ldc,
                                       float alpha, float beta,
                                       unsigned int *rand_state,
                                       bool stochastic_round, bool apply_relu,
                                       CUstream stream, int grid
                                      )
{
    return nervana_sgemm(B, A, C,
                         b_t, a_t,
                         n, m, k,
                         ldb, lda, ldc,
                         alpha, beta,
                         rand_state, stochastic_round, apply_relu,
                         stream, grid);
}

extern "C" bool nervana_hgemm_colmajor(short *A, short *B, short *C,
                                       bool a_t, bool b_t,
                                       int m, int n, int k,
                                       int lda, int ldb, int ldc,
                                       float alpha, float beta,
                                       unsigned int *rand_state,
                                       bool stochastic_round, bool apply_relu,
                                       CUstream stream, int grid
                                      )
{
    return nervana_hgemm(B, A, C,
                         b_t, a_t,
                         n, m, k,
                         ldb, lda, ldc,
                         alpha, beta,
                         rand_state, stochastic_round, apply_relu,
                         stream, grid);
}
