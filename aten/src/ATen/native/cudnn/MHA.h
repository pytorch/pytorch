#pragma once
#include <ATen/ATen.h>

namespace at { namespace native {

void run_cudnn_LLM_fprop(int64_t b,
    		 int64_t h,
    		 int64_t s_q,
    		 int64_t s_kv,
    		 int64_t d,
    		 float scaling_factor,
    		 bool isTraining,
    		 double dropout_probability,
    		 const Tensor& q,
    		 const Tensor& k,
    		 const Tensor& v,
    		 Tensor& softmaxstats,
    		 Tensor& o,
    		 Tensor& dropoutseed,
    		 Tensor& dropoutoffset);
}}
