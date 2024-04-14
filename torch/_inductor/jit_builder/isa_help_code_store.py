def get_x86_isa_detect_code():
    _x86_isa_detect_code = """
#include <stdint.h>

#if defined(__GNUC__)
#include <cpuid.h>
#define __forceinline __attribute__((always_inline)) inline
#define EXTERN_DLL_EXPORT extern "C"
#elif defined(_MSC_VER)
#include <intrin.h>
#define EXTERN_DLL_EXPORT extern "C" __declspec(dllexport)
#endif

/* get value from bits [n:m] */
#define BIT_M_TO_N(x, m, n) ((uint32_t)(x << (31 - (n))) >> ((31 - (n)) + (m)))
#define BIT_M_TO_N_64(x, m, n) \
  ((uint64_t)(x << (63 - (n))) >> ((63 - (n)) + (m)))

__forceinline bool check_reg_bit(uint32_t reg, int bit_idx) {
  return (reg & ((uint32_t)1 << bit_idx));
}

__forceinline void read_cpuid(
    uint32_t func_id,
    uint32_t* p_eax,
    uint32_t* p_ebx,
    uint32_t* p_ecx,
    uint32_t* p_edx) {
  int reg_data[4] = {0};
#if defined(__GNUC__)
  __cpuid(func_id, reg_data[0], reg_data[1], reg_data[2], reg_data[3]);
#elif defined(_MSC_VER)
  __cpuid(reg_data, func_id);
#endif
  *p_eax = reg_data[0];
  *p_ebx = reg_data[1];
  *p_ecx = reg_data[2];
  *p_edx = reg_data[3];
}

__forceinline void read_cpuidex(
    uint32_t func_id,
    uint32_t sub_func_id,
    uint32_t* p_eax,
    uint32_t* p_ebx,
    uint32_t* p_ecx,
    uint32_t* p_edx) {
  int reg_data[4] = {0};
#if defined(__GNUC__)
  __cpuid_count(
      func_id, sub_func_id, reg_data[0], reg_data[1], reg_data[2], reg_data[3]);
#elif defined(_MSC_VER)
  __cpuidex(reg_data, func_id, sub_func_id);
#endif
  *p_eax = reg_data[0];
  *p_ebx = reg_data[1];
  *p_ecx = reg_data[2];
  *p_edx = reg_data[3];
}


void check_feature_via_cpuid(bool &bit_fma,
  bool &bit_avx2,
  bool &bit_avx512_f,
  bool &bit_avx512_vl,
  bool &bit_avx512_bw,
  bool &bit_avx512_dq)
{
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  /*
  Initial version reference from:
  ----------------------------------------------------
  Intel® Architecture
  Instruction Set Extensions
  and Future Features
  Programming Reference
  May 2021
  319433-044
  */
  read_cpuid(0, &eax, &ebx, &ecx, &edx);
  uint32_t max_basic_id = eax;

  read_cpuid(0x80000000, &eax, &ebx, &ecx, &edx);
  uint32_t max_extend_id = eax;

  if (max_basic_id >= 0x00000001) {
    read_cpuidex(0x00000001, 0, &eax, &ebx, &ecx, &edx);

    bit_fma = check_reg_bit(ecx, 12);
  }

  if (max_basic_id >= 0x00000007) {
    uint32_t max_sub_leaf = 0;
    read_cpuidex(0x00000007, 0, &eax, &ebx, &ecx, &edx);
    max_sub_leaf = eax;

    bit_avx2 = check_reg_bit(ebx, 5);

    bit_avx512_f = check_reg_bit(ebx, 16);
    bit_avx512_vl = check_reg_bit(ebx, 31);
    bit_avx512_bw = check_reg_bit(ebx, 30);
    bit_avx512_dq = check_reg_bit(ebx, 17);
  }
}

EXTERN_DLL_EXPORT bool check_avx2_feature()
{
  bool bit_fma = false;
  bool bit_avx2 = false;

  bool bit_avx512_f = false;
  bool bit_avx512_vl = false;
  bool bit_avx512_bw = false;
  bool bit_avx512_dq = false;

  check_feature_via_cpuid(bit_fma, bit_avx2, bit_avx512_f, bit_avx512_vl, bit_avx512_bw, bit_avx512_dq);

  return (bit_fma && bit_avx2);
}

EXTERN_DLL_EXPORT bool check_avx512_feature()
{
  bool bit_fma = false;
  bool bit_avx2 = false;

  bool bit_avx512_f = false;
  bool bit_avx512_vl = false;
  bool bit_avx512_bw = false;
  bool bit_avx512_dq = false;

  check_feature_via_cpuid(bit_fma, bit_avx2, bit_avx512_f, bit_avx512_vl, bit_avx512_bw, bit_avx512_dq);

  return (bit_fma && bit_avx512_f &&
  bit_avx512_vl && bit_avx512_bw && bit_avx512_dq);
}
"""
    return _x86_isa_detect_code
