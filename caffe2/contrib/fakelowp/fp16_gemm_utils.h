// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once
#include "caffe2/core/context.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
void custom_fp16_gemm(
    const int m,
    const int k,
    const int n,
    const float* A_fp16,
    const float* B_fp16,
    const float beta,
    float* C,
    const bool use_acc_fp16,
    const bool use_temp_accumulator);

void custom_fp16_gemm_with_trans(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int m,
    const int k,
    const int n,
    const float* A_fp16,
    const float* B_fp16,
    const float beta,
    float* C,
    const bool use_acc_fp16,
    const bool use_temp_accumulator);

void transpose(const float* A, float* A_trans, int M, int N);
void custom_fp16_gemv(
    const bool use_acc_fp16,
    const bool use_custom_acc32,
    const bool use_temp_accumulator,
    const CBLAS_TRANSPOSE trans_A,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CPUContext* context);

void custom_fp16_gemm_batched(
    const bool use_acc_fp16,
    const bool use_custom_acc32,
    const bool use_temp_accumulator,
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float** A,
    const float** B,
    const float beta,
    float** C,
    CPUContext* context);
void custom_fp16_gemm_strided_batched(
    const bool use_acc_fp16,
    const bool use_custom_acc32,
    const bool use_temp_accumulator,
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha /* unused */,
    const float* A,
    const int A_stride,
    const float* B,
    const int B_stride,
    const float beta,
    float* C,
    const int C_stride,
    CPUContext* context);
} // namespace caffe2
