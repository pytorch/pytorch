#include <ATen/cpu/Utils.h>
#if !defined(__s390x__ ) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace at::cpu {

static constexpr const char* get_cpu_architecture() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__powerpc64__) || defined(__PPC64__)
  return "ppc64";
#elif defined(__s390x__)
  return "s390x";
#elif defined(__riscv) && (__riscv_xlen == 64)
  return "riscv64";
#else
  return "unknown";
#endif
}

std::unordered_map<std::string, c10::IValue> get_cpu_capabilities() {
  std::unordered_map<std::string, c10::IValue> capabilities;

  capabilities["architecture"] = std::string(get_cpu_architecture());

#if !defined(__s390x__) && !defined(__powerpc__)
  if (!cpuinfo_initialize()) {
    return capabilities;
  }

  auto get_cache_size = [](int level) -> int64_t {
    const auto processors = cpuinfo_get_processors();
    if (!processors) {
      return 0;
    }
    const struct cpuinfo_cache* cache = nullptr;
    switch (level) {
      case 1:
        cache = processors[0].cache.l1d;
        break;
      case 2:
        cache = processors[0].cache.l2;
        break;
      default:
        TORCH_CHECK(false, "Unsupported cache level");
    }
    return cache ? static_cast<int64_t>(cache->size) : 0;
  };

  const auto packages = cpuinfo_get_packages();
  if (packages && cpuinfo_get_packages_count() > 0) {
    capabilities["cpu_name"] = std::string(packages[0].name);
  }

  capabilities["num_sockets"] =
      static_cast<int64_t>(cpuinfo_get_packages_count());
  capabilities["num_physical_cores"] =
      static_cast<int64_t>(cpuinfo_get_cores_count());
  capabilities["num_logical_cores"] =
      static_cast<int64_t>(cpuinfo_get_processors_count());

  capabilities["l1d_cache_size"] = get_cache_size(1);
  capabilities["l2_cache_size"] = get_cache_size(2);

#if defined(__x86_64__) || defined(_M_X64)
  // SSE family
  capabilities["sse"] = cpuinfo_has_x86_sse();
  capabilities["sse2"] = cpuinfo_has_x86_sse2();
  capabilities["sse3"] = cpuinfo_has_x86_sse3();
  capabilities["ssse3"] = cpuinfo_has_x86_ssse3();
  capabilities["sse4_1"] = cpuinfo_has_x86_sse4_1();
  capabilities["sse4_2"] = cpuinfo_has_x86_sse4_2();
  capabilities["sse4a"] = cpuinfo_has_x86_sse4a();

  // AVX family
  capabilities["avx"] = cpuinfo_has_x86_avx();
  capabilities["avx2"] = cpuinfo_has_x86_avx2();
  capabilities["avx_vnni"] = cpuinfo_has_x86_avxvnni();

  // AVX-512 family
  capabilities["avx512_f"] = cpuinfo_has_x86_avx512f();
  capabilities["avx512_cd"] = cpuinfo_has_x86_avx512cd();
  capabilities["avx512_dq"] = cpuinfo_has_x86_avx512dq();
  capabilities["avx512_bw"] = cpuinfo_has_x86_avx512bw();
  capabilities["avx512_vl"] = cpuinfo_has_x86_avx512vl();
  capabilities["avx512_ifma"] = cpuinfo_has_x86_avx512ifma();
  capabilities["avx512_vbmi"] = cpuinfo_has_x86_avx512vbmi();
  capabilities["avx512_vbmi2"] = cpuinfo_has_x86_avx512vbmi2();
  capabilities["avx512_bitalg"] = cpuinfo_has_x86_avx512bitalg();
  capabilities["avx512_vpopcntdq"] = cpuinfo_has_x86_avx512vpopcntdq();
  capabilities["avx512_vnni"] = cpuinfo_has_x86_avx512vnni();
  capabilities["avx512_bf16"] = cpuinfo_has_x86_avx512bf16();
  capabilities["avx512_fp16"] = cpuinfo_has_x86_avx512fp16();
  capabilities["avx512_vp2intersect"] = cpuinfo_has_x86_avx512vp2intersect();
  capabilities["avx512_4vnniw"] = cpuinfo_has_x86_avx512_4vnniw();
  capabilities["avx512_4fmaps"] = cpuinfo_has_x86_avx512_4fmaps();

  // AVX10 family
  capabilities["avx10_1"] = cpuinfo_has_x86_avx10_1();
  capabilities["avx10_2"] = cpuinfo_has_x86_avx10_2();

  // AVX-VNNI-INT variants
  capabilities["avx_vnni_int8"] = cpuinfo_has_x86_avx_vnni_int8();
  capabilities["avx_vnni_int16"] = cpuinfo_has_x86_avx_vnni_int16();
  capabilities["avx_ne_convert"] = cpuinfo_has_x86_avx_ne_convert();

  // AMX (Advanced Matrix Extensions)
  capabilities["amx_bf16"] = cpuinfo_has_x86_amx_bf16();
  capabilities["amx_tile"] = cpuinfo_has_x86_amx_tile();
  capabilities["amx_int8"] = cpuinfo_has_x86_amx_int8();
  capabilities["amx_fp16"] = cpuinfo_has_x86_amx_fp16();

  // FMA
  capabilities["fma3"] = cpuinfo_has_x86_fma3();
  capabilities["fma4"] = cpuinfo_has_x86_fma4();

  // Other useful capabilities
  capabilities["f16c"] = cpuinfo_has_x86_f16c();
  capabilities["bmi"] = cpuinfo_has_x86_bmi();
  capabilities["bmi2"] = cpuinfo_has_x86_bmi2();
  capabilities["popcnt"] = cpuinfo_has_x86_popcnt();
  capabilities["lzcnt"] = cpuinfo_has_x86_lzcnt();
  capabilities["aes"] = cpuinfo_has_x86_aes();
  capabilities["sha"] = cpuinfo_has_x86_sha();
  capabilities["clflush"] = cpuinfo_isa.clflush;
  capabilities["clflushopt"] = cpuinfo_isa.clflushopt;
  capabilities["clwb"] = cpuinfo_has_x86_clwb();
#endif

  // ARM64 specific capabilities
#if defined(__aarch64__) || defined(_M_ARM64)
  capabilities["neon"] = cpuinfo_has_arm_neon();
  capabilities["fp16_arith"] = cpuinfo_has_arm_fp16_arith();
  capabilities["bf16"] = cpuinfo_has_arm_bf16();
  capabilities["i8mm"] = cpuinfo_has_arm_i8mm();
  capabilities["dot"] = cpuinfo_has_arm_neon_dot();
  capabilities["sve"] = cpuinfo_has_arm_sve();
  capabilities["sve2"] = cpuinfo_has_arm_sve2();
  capabilities["sve_bf16"] = cpuinfo_has_arm_sve_bf16();
  capabilities["sme"] = cpuinfo_has_arm_sme();
  capabilities["sme2"] = cpuinfo_has_arm_sme2();
  capabilities["atomics"] = cpuinfo_has_arm_atomics();
  capabilities["fhm"] = cpuinfo_has_arm_fhm();
  capabilities["rdm"] = cpuinfo_has_arm_neon_rdm();
  capabilities["crc32"] = cpuinfo_has_arm_crc32();
  capabilities["aes"] = cpuinfo_has_arm_aes();
  capabilities["sha1"] = cpuinfo_has_arm_sha1();
  capabilities["sha2"] = cpuinfo_has_arm_sha2();
  capabilities["pmull"] = cpuinfo_has_arm_pmull();
  if (cpuinfo_has_arm_sve()) {
    capabilities["sve_max_length"] =
        static_cast<int64_t>(cpuinfo_get_max_arm_sve_length());
  }
  if (cpuinfo_has_arm_sme()) {
    capabilities["sme_max_length"] =
        static_cast<int64_t>(cpuinfo_get_max_arm_sme_length());
  }
#endif

#endif // !defined(__s390x__) && !defined(__powerpc__)

  return capabilities;
}

bool is_avx512_vnni_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
#else
  return false;
#endif
}

static bool is_amx_tile_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_amx_tile();
#else
  return false;
#endif
}

bool init_amx() {
  if (!is_amx_tile_supported()) {
    return false;
  }

#if defined(__linux__) && !defined(__ANDROID__) && defined(__x86_64__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

  unsigned long bitmask = 0;
  // Request permission to use AMX instructions
  long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc) {
      return false;
  }
  // Check if the system supports AMX instructions
  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (rc) {
      return false;
  }
  if (bitmask & XFEATURE_MASK_XTILE) {
      return true;
  }
  return false;
#else
  return true;
#endif
}

} // namespace at::cpu
