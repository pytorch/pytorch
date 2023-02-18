// Utility macro for this file
#define DEVICE_INLINE __device__ inline

// Utility for converting generic pointer to SMEM pointer in PTX.
//  We should review vectorized load/stores with shared memory.
//  SMEM memory movement PTX is only Global -> SMEM, SMEM -> Local, Local ->
//  SMEM, and this is needed for these PTX instructions to provide the SMEM
//  pointer.
DEVICE_INLINE unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));

  return smem_ptr_uint;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))

namespace Turing {

namespace util {

// LdMatrix has .x1, .x2 and .x4 options, currently we actively use .x2 and
//  .x4. In .x2 option. the the address register of upper half warp (lane 16-31)
//  are un-used but on Turing [sm75,sm80) architecture these un-used addresses
//  need to be valid, in the sense that:
//     1. The data it points to has to be within allocated shared mem buffer.
//     2. The address needs to be aligned to 16 byte.
//  See also:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
//  This function addresses 2. above by masking out the sub-16B component
//    of the address in upper warp and 1. is guaranteed by ldmatrix swizzle
//    util.
//  This will **not** affect any functionality. This is just modification
//    of unused pointers to satisfy the alignment requirement on Turing
//    hardware.
//  The alignment requirement is lifted on sm80+,
//    so this function is a no-op on Ampere or above.
DEVICE_INLINE void adjustPartialLdMatrixAddrInTuring(unsigned& addr_in_byte) {
#if (__CUDA_ARCH__ < 800)
  const unsigned thread_id = threadIdx.x;
  // Upper half warp has 8 bytes offset from aligned in .x2 option
  //  of ldmatrix. Currently no support for .x1 so assume always
  //  adjust by half warp.
  constexpr unsigned half_warp = 16;
  // Need to adjust to 16 byte alignment, mask out un-aligned component.
  constexpr unsigned mask_out = 16 - 1;
  // Adjust only in upper half warp.
  // use bit math to reduce strength
  if (thread_id & half_warp) {
    // mask out the bits where adjust_mask has 1.
    addr_in_byte &= (~mask_out);
  }
#endif //(__CUDA_ARCH__ < 800)
}

} // namespace util

// Load Matrix (per warp instruction) is to take data from SMEM to Local Memory.
//   Automatically handles vectorized loads/stores in the MMA operation.
//   Loads 8x8 matrix into a warp. Thread 0-7 provide the ptr that is the start
//   of each row. All other threads can simply point to something valid
//   (including 0).
// The x2 modifier on the instruction will actually load 2x8 rows to make a
// 16x8,
//   then thread 0-15 will specify the start of each row.
// Finally is an x4 modifier producing a 32x8 using addrs from 0-31 in each
// warp.
DEVICE_INLINE void ldMatrix(Array<__half, 4, 4>& out, unsigned addr) {
  uint2& val = reinterpret_cast<uint2&>(out);
  util::adjustPartialLdMatrixAddrInTuring(addr);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
               : "=r"(val.x), "=r"(val.y)
               : "r"(addr));
}

// Same as previous, 8x8 matrix is vectorized loaded, then scattered (to perform
// transpose) so threads will hold 2 values down a column (instead of the
// previous instruction that's across a row).
DEVICE_INLINE void ldMatrixT(Array<__half, 4, 4>& out, unsigned addr) {
  uint2& val = reinterpret_cast<uint2&>(out);
  util::adjustPartialLdMatrixAddrInTuring(addr);
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];"
               : "=r"(val.x), "=r"(val.y)
               : "r"(addr));
}

DEVICE_INLINE void ldMatrix(Array<__half, 8, 8>& out, unsigned addr) {
  uint4& val = reinterpret_cast<uint4&>(out);
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "r"(addr));
}

DEVICE_INLINE void ldMatrixT(Array<__half, 8, 8>& out, unsigned addr) {
  uint4& val = reinterpret_cast<uint4&>(out);
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
      : "r"(addr));
}

} // namespace Turing

#endif // Arch 75

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

namespace Ampere {

// MMA instruction wrappers (sm_80+):

// Global to SMEM load that is asynchronous,
// not guaranteed to be completed until cpAsyncBarrier() is called.
// if predicate is set to false, then gmem_ptr won't be read and smem_addr will be zero-initialized
// gmem_ptr must be `sizeof(dtype) * len` aligned
template <typename dtype, int len>
DEVICE_INLINE void cpAsyncCa(
    unsigned smem_addr,
    void const* gmem_ptr,
    bool predicate) {
  constexpr int byte_size = sizeof(dtype) * len;

  static_assert(
      byte_size == 4 || byte_size == 8 || byte_size == 16,
      "cp_async : unsupported byte size");

  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.eq.b32 p, %3, 0;\n"
      "  cp.async.ca.shared.global [%0], [%1], %2, p;\n"
      "}\n" ::"r"(smem_addr),
      "l"(gmem_ptr),
      "n"(byte_size),
      "r"((int)predicate));
}

// Global to SMEM load that is asynchronous,
//  The cache global variant, i.e. skip L1 caching.
// more details see:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
// not guaranteed to be completed until cpAsyncBarrier() is called.
// if predicate is set to false, then gmem_ptr won't be read and smem_addr will be zero-initialized
// gmem_ptr must be 16B aligned
template <typename dtype, int len>
DEVICE_INLINE void cpAsyncCg(
    unsigned smem_addr,
    void const* gmem_ptr,
    bool predicate) {
  constexpr int byte_size = sizeof(dtype) * len;

  static_assert(byte_size == 16, "cp_async : unsupported byte size");

  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.eq.b32 p, %2, 0;\n"
      "  cp.async.cg.shared.global [%0], [%1], 16, p;\n"
      "}\n" ::"r"(smem_addr),
      "l"(gmem_ptr),
      "r"((int)predicate));
}

// TODO: Might have a different category of sync if we want to build out this:
DEVICE_INLINE void cpAsyncBarrier() {
  asm volatile("cp.async.wait_all;");
}

DEVICE_INLINE void cpAsyncCommit() {
  asm volatile("cp.async.commit_group;");
}

template <int keep_stages>
DEVICE_INLINE void cpAsyncPartialBarrier() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(keep_stages));
}

} // namespace Ampere

#endif // Arch 80

#undef DEVICE_INLINE
