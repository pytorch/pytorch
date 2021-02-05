/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <cpuinfo.h>
#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/params.h>
#include <qnnpack/q8avgpool.h>
#include <qnnpack/q8conv.h>
#include <qnnpack/q8dwconv.h>
#include <qnnpack/q8gavgpool.h>
#include <qnnpack/q8gemm.h>
#include <qnnpack/q8gemm_sparse.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/u8clamp.h>
#include <qnnpack/u8lut32norm.h>
#include <qnnpack/u8maxpool.h>
#include <qnnpack/u8rmax.h>
#include <qnnpack/x8lut.h>
#include <qnnpack/x8zip.h>


#ifdef _MSC_VER
static INIT_ONCE init_guard;
BOOL CALLBACK pytorch_qnnp_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContex);
#else
static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

struct pytorch_qnnp_parameters pytorch_qnnp_params = {.initialized = false};

static void init(void) {
#if CPUINFO_ARCH_ARM
  if (!cpuinfo_has_arm_neon()) {
    pytorch_qnnp_log_error(
        "QNNPACK initialization failed: NEON is not supported");
    return;
  }
  pytorch_qnnp_params.q8conv = (struct pytorch_q8conv_parameters){
      .gemm = pytorch_q8gemm_ukernel_4x8__aarch32_neon,
      .conv = pytorch_q8conv_ukernel_4x8__aarch32_neon,
      .gemm_dq = pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 1,
  };
  pytorch_qnnp_params.q8gemm_sparse = (struct pytorch_q8gemm_sparse_parameters){
      .gemm_dq = pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon,
      .packedA_gemm_dq = pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon,
      .packA = pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      .mr = 8,
      .nr = 4,
      .kr = 4,
      .log2_mr = 3,
  };
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
  pytorch_qnnp_params.q8conv_xzp = (struct pytorch_q8conv_xzp_parameters){
      .gemm = pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 2,
      .kc = 8,
      .kthreshold = SIZE_MAX,
  };
  /* setup xzp threshold based on measurements */
  switch (cpuinfo_get_core(0)->uarch) {
    case cpuinfo_uarch_cortex_a72:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 64;
      break;
    case cpuinfo_uarch_cortex_a73:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 256;
      break;
    case cpuinfo_uarch_cortex_a75:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 32;
      break;
    case cpuinfo_uarch_cortex_a76:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 16;
      break;
    default:
      break;
  }
#else
  pytorch_qnnp_params.q8conv_xzp = (struct pytorch_q8conv_xzp_parameters){
      .kthreshold = SIZE_MAX,
  };
#endif
  pytorch_qnnp_params.q8dw9 = (struct pytorch_q8dwconv_up_parameters){
      .updw = pytorch_q8dwconv_ukernel_up8x9__aarch32_neon,
      .updw_per_channel = pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon,
      .cr = 8,
  };
  pytorch_qnnp_params.q8dw25 = (struct pytorch_q8dwconv_mp_parameters){
      .mpdw = pytorch_q8dwconv_ukernel_mp8x25__neon,
      .mpdw_per_channel = pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon,
      .cr = 8,
  };
  pytorch_qnnp_params.q8sum_rows = (struct pytorch_q8sum_rows_parameters){
      .sum_rows = pytorch_q8sumrows_ukernel_4x__neon,
      .m = 4,
  };
  pytorch_qnnp_params.q8vadd = pytorch_q8vadd_ukernel__neon;
  pytorch_qnnp_params.q8gavgpool = (struct pytorch_q8gavgpool_parameters){
      .ltnr = pytorch_q8gavgpool_ukernel_up8xm__neon,
      .genr_lemr = pytorch_q8gavgpool_ukernel_up8x7__neon,
      .genr_gtmr = pytorch_q8gavgpool_ukernel_mp8x7p7q__neon,
      .mr = 7,
      .nr = 8,
  };
  pytorch_qnnp_params.q8avgpool = (struct pytorch_q8avgpool_parameters){
      .ltkr = pytorch_q8avgpool_ukernel_up8xm__neon,
      .gekr_lemr = pytorch_q8avgpool_ukernel_up8x9__neon,
      .gekr_gtmr = pytorch_q8avgpool_ukernel_mp8x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  pytorch_qnnp_params.u8maxpool = (struct pytorch_u8maxpool_parameters){
      .ltkr = pytorch_u8maxpool_ukernel_sub16__neon,
      .gekr = pytorch_u8maxpool_ukernel_16x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  pytorch_qnnp_params.x8zip = (struct pytorch_x8zip_parameters){
      .x2 = pytorch_qnnp_x8zip_x2__neon,
      .x3 = pytorch_qnnp_x8zip_x3__neon,
      .x4 = pytorch_qnnp_x8zip_x4__neon,
      .xm = pytorch_qnnp_x8zip_xm__neon,
  };
  pytorch_qnnp_params.u8clamp = pytorch_u8clamp_ukernel__neon;
  pytorch_qnnp_params.u8rmax = pytorch_u8rmax_ukernel__neon;
  pytorch_qnnp_params.u8lut32norm = pytorch_u8lut32norm_ukernel__scalar;
  pytorch_qnnp_params.x8lut = pytorch_x8lut_ukernel__scalar;
#elif CPUINFO_ARCH_ARM64
  pytorch_qnnp_params.q8conv = (struct pytorch_q8conv_parameters){
      .gemm = pytorch_q8gemm_ukernel_8x8__aarch64_neon,
      .conv = pytorch_q8conv_ukernel_8x8__aarch64_neon,
      .gemm_dq = pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon,
      .mr = 8,
      .nr = 8,
      .kr = 1,
  };
  pytorch_qnnp_params.q8conv_xzp = (struct pytorch_q8conv_xzp_parameters){
      .kthreshold = SIZE_MAX,
  };
  pytorch_qnnp_params.q8dw9 = (struct pytorch_q8dwconv_up_parameters){
      .updw = pytorch_q8dwconv_ukernel_up8x9__neon,
      .updw_per_channel = pytorch_q8dwconv_ukernel_up8x9_per_channel__neon,
      .cr = 8,
  };
  pytorch_qnnp_params.q8dw25 = (struct pytorch_q8dwconv_mp_parameters){
      .mpdw = pytorch_q8dwconv_ukernel_mp8x25__neon,
      .mpdw_per_channel = pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon,
      .cr = 8,
  };
  pytorch_qnnp_params.q8vadd = pytorch_q8vadd_ukernel__neon;
  pytorch_qnnp_params.q8gavgpool = (struct pytorch_q8gavgpool_parameters){
      .ltnr = pytorch_q8gavgpool_ukernel_up8xm__neon,
      .genr_lemr = pytorch_q8gavgpool_ukernel_up8x7__neon,
      .genr_gtmr = pytorch_q8gavgpool_ukernel_mp8x7p7q__neon,
      .mr = 7,
      .nr = 8,
  };
  pytorch_qnnp_params.q8avgpool = (struct pytorch_q8avgpool_parameters){
      .ltkr = pytorch_q8avgpool_ukernel_up8xm__neon,
      .gekr_lemr = pytorch_q8avgpool_ukernel_up8x9__neon,
      .gekr_gtmr = pytorch_q8avgpool_ukernel_mp8x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  pytorch_qnnp_params.u8maxpool = (struct pytorch_u8maxpool_parameters){
      .ltkr = pytorch_u8maxpool_ukernel_sub16__neon,
      .gekr = pytorch_u8maxpool_ukernel_16x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  pytorch_qnnp_params.x8zip = (struct pytorch_x8zip_parameters){
      .x2 = pytorch_qnnp_x8zip_x2__neon,
      .x3 = pytorch_qnnp_x8zip_x3__neon,
      .x4 = pytorch_qnnp_x8zip_x4__neon,
      .xm = pytorch_qnnp_x8zip_xm__neon,
  };
  pytorch_qnnp_params.u8clamp = pytorch_u8clamp_ukernel__neon;
  pytorch_qnnp_params.u8rmax = pytorch_u8rmax_ukernel__neon;
  pytorch_qnnp_params.u8lut32norm = pytorch_u8lut32norm_ukernel__scalar;
  pytorch_qnnp_params.x8lut = pytorch_x8lut_ukernel__scalar;
#elif CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  if (!cpuinfo_has_x86_sse2()) {
    pytorch_qnnp_log_error(
        "QNNPACK initialization failed: SSE2 is not supported");
    return;
  }
  pytorch_qnnp_params.q8conv = (struct pytorch_q8conv_parameters){
      .gemm = pytorch_q8gemm_ukernel_4x4c2__sse2,
      .conv = pytorch_q8conv_ukernel_4x4c2__sse2,
      .gemm_dq = pytorch_q8gemm_dq_ukernel_4x4c2__sse2,
      .mr = 4,
      .nr = 4,
      .kr = 2,
  };
  pytorch_qnnp_params.q8gemm_sparse = (struct pytorch_q8gemm_sparse_parameters){
      .gemm_dq = pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2,
      .packedA_gemm_dq = pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2,
      .packA = pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      .mr = 8,
      .nr = 4,
      .kr = 4,
      .log2_mr = 3,
  };
  pytorch_qnnp_params.q8conv_xzp = (struct pytorch_q8conv_xzp_parameters){
      .kthreshold = SIZE_MAX,
  };
  pytorch_qnnp_params.q8dw9 = (struct pytorch_q8dwconv_up_parameters){
      .updw = pytorch_q8dwconv_ukernel_up8x9__sse2,
      .updw_per_channel = pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2,
      .cr = 8,
  };
  pytorch_qnnp_params.q8dw25 = (struct pytorch_q8dwconv_mp_parameters){
      .mpdw = pytorch_q8dwconv_ukernel_mp8x25__sse2,
      .mpdw_per_channel = pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2,
      .cr = 8,
  };
  pytorch_qnnp_params.q8vadd = pytorch_q8vadd_ukernel__sse2;
  pytorch_qnnp_params.q8gavgpool = (struct pytorch_q8gavgpool_parameters){
      .ltnr = pytorch_q8gavgpool_ukernel_up8xm__sse2,
      .genr_lemr = pytorch_q8gavgpool_ukernel_up8x7__sse2,
      .genr_gtmr = pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2,
      .mr = 7,
      .nr = 8,
  };
  pytorch_qnnp_params.q8avgpool = (struct pytorch_q8avgpool_parameters){
      .ltkr = pytorch_q8avgpool_ukernel_up8xm__sse2,
      .gekr_lemr = pytorch_q8avgpool_ukernel_up8x9__sse2,
      .gekr_gtmr = pytorch_q8avgpool_ukernel_mp8x9p8q__sse2,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  pytorch_qnnp_params.u8maxpool = (struct pytorch_u8maxpool_parameters){
      .ltkr = pytorch_u8maxpool_ukernel_sub16__sse2,
      .gekr = pytorch_u8maxpool_ukernel_16x9p8q__sse2,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  pytorch_qnnp_params.x8zip = (struct pytorch_x8zip_parameters){
      .x2 = pytorch_qnnp_x8zip_x2__sse2,
      .x3 = pytorch_qnnp_x8zip_x3__sse2,
      .x4 = pytorch_qnnp_x8zip_x4__sse2,
      .xm = pytorch_qnnp_x8zip_xm__sse2,
  };
  pytorch_qnnp_params.u8clamp = pytorch_u8clamp_ukernel__sse2;
  pytorch_qnnp_params.u8rmax = pytorch_u8rmax_ukernel__sse2;
  pytorch_qnnp_params.u8lut32norm = pytorch_u8lut32norm_ukernel__scalar;
  pytorch_qnnp_params.x8lut = pytorch_x8lut_ukernel__scalar;
#else
#error "Unsupported architecture"
#endif
  pytorch_qnnp_params.initialized = true;
}

enum pytorch_qnnp_status pytorch_qnnp_initialize(void) {
  if (!cpuinfo_initialize()) {
    return pytorch_qnnp_status_out_of_memory;
  }
#ifdef _MSC_VER
  InitOnceExecuteOnce(&init_guard, pytorch_qnnp_init_win, NULL, NULL);
#else
  pthread_once(&init_guard, &init);
#endif
  if (pytorch_qnnp_params.initialized) {
    return pytorch_qnnp_status_success;
  } else {
    return pytorch_qnnp_status_unsupported_hardware;
  }
}

enum pytorch_qnnp_status pytorch_qnnp_deinitialize(void) {
  cpuinfo_deinitialize();
  return pytorch_qnnp_status_success;
}

#ifdef _MSC_VER
BOOL CALLBACK pytorch_qnnp_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContex) {
  init();
  return TRUE;
}
#endif
