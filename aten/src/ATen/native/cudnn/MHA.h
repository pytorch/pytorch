#pragma once
#include <ATen/ATen.h>
/*
enum class MHA_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2,
    SBH_INTERLEAVED = 3
};
*/
namespace at { namespace native {

void run_cudnn_LLM_fprop(int64_t b, 
                    int64_t h, 
                    int64_t s_q,
                    int64_t s_kv,
                    int64_t d,
                    float scaling_factor,
                    bool isTraining,
                    double dropout_probability,
                    Tensor& q, 
                    Tensor& k,   
                    Tensor& v,
                    Tensor& softmaxstats,
                    Tensor& o,
                    Tensor& dropoutseed,
                    Tensor& dropoutoffset);

}}
