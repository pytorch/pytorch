#pragma once
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/packed_stride.hpp>
#include <sycl/sycl.hpp>

#include "collective/xe_flash_attn_prefill_mma_bshd.h"
#include "collective/xe_flash_attn_sdpa_fwd_bshd_epilogue.h"
#include "collective/xe_flash_attn_sdpa_fwd_bshd_softmax_epilogue.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "kernel/tile_scheduler_sdpa_fwd_bshd.h"
#include "kernel/xe_sdpa_fwd_bshd.h"