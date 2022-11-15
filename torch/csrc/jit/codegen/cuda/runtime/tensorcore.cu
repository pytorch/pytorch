// Utility macro for this file
#define DEVICE_INLINE __device__ inline

// MMA instruction wrappers:
//  The wrappers are subroutines that implement matrix of size
//    A(M,K) X B(K,N) = C(M,N)
//  The naming of the wrappers follow similar naming conventions
//    as the mma instructions.
//  All the mma macros follow the namespace and naming like
//    Arch::M (M-dim) N (N-dim) K(K-dim) (Layout), eg.
//    Volta::M16N16K4TT,
//  with the dimensions describing the size of the sub-matrices being
//   multiplied by this wrapper.
//  see [Operand Layout Convention] in mma_type.h for details on the layout
//   notation.
namespace Volta {

namespace util {
// MMA instruction wrappers (sm_70+):
// The instruction wrappers below are quarter-warp macros, which currently
//  nvfuser doesn't explicitly model.
// So they are currently only meant to be
//  used as building blocks in warp level mma macros

//  8x8x4 mma instruction, per quarter warp (8 threads), fp32 accumulate
//  per thread register:
//   A[4] x B[4] -> C[8]
DEVICE_INLINE void mmaM8n8k4tt(
    Array<float, 8, 8>* C,
    Array<__half, 4, 4>* A,
    Array<__half, 4, 4>* B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(B);
  unsigned* _C = reinterpret_cast<unsigned*>(C);

  asm("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(_C[0]),
        "=r"(_C[1]),
        "=r"(_C[2]),
        "=r"(_C[3]),
        "=r"(_C[4]),
        "=r"(_C[5]),
        "=r"(_C[6]),
        "=r"(_C[7])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_C[0]),
        "r"(_C[1]),
        "r"(_C[2]),
        "r"(_C[3]),
        "r"(_C[4]),
        "r"(_C[5]),
        "r"(_C[6]),
        "r"(_C[7]));
}

DEVICE_INLINE void mmaM8n8k4tn(
    Array<float, 8, 8>* C,
    Array<__half, 4, 4>* A,
    Array<__half, 4, 4>* B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(B);
  unsigned* _C = reinterpret_cast<unsigned*>(C);

  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(_C[0]),
        "=r"(_C[1]),
        "=r"(_C[2]),
        "=r"(_C[3]),
        "=r"(_C[4]),
        "=r"(_C[5]),
        "=r"(_C[6]),
        "=r"(_C[7])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_C[0]),
        "r"(_C[1]),
        "r"(_C[2]),
        "r"(_C[3]),
        "r"(_C[4]),
        "r"(_C[5]),
        "r"(_C[6]),
        "r"(_C[7]));
}

DEVICE_INLINE void mmaM8n8k4nt(
    Array<float, 8, 8>* C,
    Array<__half, 4, 4>* A,
    Array<__half, 4, 4>* B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(B);
  unsigned* _C = reinterpret_cast<unsigned*>(C);

  asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(_C[0]),
        "=r"(_C[1]),
        "=r"(_C[2]),
        "=r"(_C[3]),
        "=r"(_C[4]),
        "=r"(_C[5]),
        "=r"(_C[6]),
        "=r"(_C[7])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_C[0]),
        "r"(_C[1]),
        "r"(_C[2]),
        "r"(_C[3]),
        "r"(_C[4]),
        "r"(_C[5]),
        "r"(_C[6]),
        "r"(_C[7]));
}

// TODO: in a follow up,
//    lift this part onto iterdomain ops, once the
//    swizzle ops are ready.
template <int acc_stride>
DEVICE_INLINE Array<float, 8, 8> accToMma(float* _C) {
  float C_data[8] = {
      _C[0],
      _C[1],
      _C[acc_stride],
      _C[acc_stride + 1],
      _C[2],
      _C[3],
      _C[acc_stride + 2],
      _C[acc_stride + 3],
  };

  return *reinterpret_cast<Array<float, 8, 8>*>(&C_data[0]);
}

template <int acc_stride>
DEVICE_INLINE void mmaToAcc(float* _C, Array<float, 8, 8>& C) {
  float* C_data = reinterpret_cast<float*>(&C);
  _C[0] = C_data[0];
  _C[1] = C_data[1];
  _C[acc_stride] = C_data[2];
  _C[acc_stride + 1] = C_data[3];
  _C[2] = C_data[4];
  _C[3] = C_data[5];
  _C[acc_stride + 2] = C_data[6];
  _C[acc_stride + 3] = C_data[7];
}

// Should be able to lift this with transpose op as well.
template <int acc_stride>
DEVICE_INLINE void initM16N16K4(Array<float, 8, 8>& accumulator) {
  float* _C = reinterpret_cast<float*>(&accumulator);
  float zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  mmaToAcc<acc_stride>(_C, *reinterpret_cast<Array<float, 8, 8>*>(&zeros[0]));
}

} // namespace util

template <int acc_stride>
DEVICE_INLINE void M16N16K4TT(
    Array<float, 8, 8>* C,
    Array<__half, 4, 4>* A,
    Array<__half, 4, 4>* B) {
  float* _C = reinterpret_cast<float*>(C);
  Array<float, 8, 8> C_data = util::accToMma<acc_stride>(_C);
  util::mmaM8n8k4tt(&C_data, A, B);
  util::mmaToAcc<acc_stride>(_C, C_data);
}

template <int acc_stride>
DEVICE_INLINE void M16N16K4TN(
    Array<float, 8, 8>* C,
    Array<__half, 4, 4>* A,
    Array<__half, 4, 4>* B) {
  float* _C = reinterpret_cast<float*>(C);
  Array<float, 8, 8> C_data = util::accToMma<acc_stride>(_C);
  util::mmaM8n8k4tn(&C_data, A, B);
  util::mmaToAcc<acc_stride>(_C, C_data);
}

template <int acc_stride>
DEVICE_INLINE void M16N16K4NT(
    Array<float, 8, 8>* C,
    Array<__half, 4, 4>* A,
    Array<__half, 4, 4>* B) {
  float* _C = reinterpret_cast<float*>(C);
  Array<float, 8, 8> C_data = util::accToMma<acc_stride>(_C);
  util::mmaM8n8k4nt(&C_data, A, B);
  util::mmaToAcc<acc_stride>(_C, C_data);
}

// Same initialization for now, will be different in interleaved
//   macros
template <int acc_stride>
DEVICE_INLINE void initM16N16K4TT(Array<float, 8, 8>* accumulator) {
  util::initM16N16K4<acc_stride>(*accumulator);
}

template <int acc_stride>
DEVICE_INLINE void initM16N16K4TN(Array<float, 8, 8>* accumulator) {
  util::initM16N16K4<acc_stride>(*accumulator);
}

template <int acc_stride>
DEVICE_INLINE void initM16N16K4NT(Array<float, 8, 8>* accumulator) {
  util::initM16N16K4<acc_stride>(*accumulator);
}

} // namespace Volta

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))

namespace Turing {

namespace util {
// MMA instruction wrappers (sm_75+):
DEVICE_INLINE void m16n8k16TN(
    Array<float, 4, 4>* C,
    Array<__half, 8, 8>* A,
    Array<__half, 4, 4>* B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(B);
  unsigned* _C = reinterpret_cast<unsigned*>(C);
  const unsigned* _D = reinterpret_cast<const unsigned*>(C);

  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=r"(_C[0]), "=r"(_C[1]), "=r"(_C[2]), "=r"(_C[3])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_D[0]),
        "r"(_D[1]),
        "r"(_D[2]),
        "r"(_D[3]));
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=r"(_C[0]), "=r"(_C[1]), "=r"(_C[2]), "=r"(_C[3])
      : "r"(_A[2]),
        "r"(_A[3]),
        "r"(_B[1]),
        "r"(_D[0]),
        "r"(_D[1]),
        "r"(_D[2]),
        "r"(_D[3]));
}

} // namespace util

template <int acc_stride>
DEVICE_INLINE void initM16N8K16TN(Array<float, 4, 4>* accumulator) {
  float* _C = reinterpret_cast<float*>(accumulator);
  _C[0] = 0;
  _C[1] = 0;
  _C[acc_stride] = 0;
  _C[acc_stride + 1] = 0;
}

template <int acc_stride = 2>
DEVICE_INLINE void M16N8K16TN(
    Array<float, 4, 4>* C,
    Array<__half, 8, 8>* A,
    Array<__half, 4, 4>* B) {
  // TODO: in a follow up,
  //    lift this fused swizzle onto iterdomain
  float* _C = reinterpret_cast<float*>(C);
  float C_data[4] = {_C[0], _C[1], _C[acc_stride], _C[acc_stride + 1]};

  util::m16n8k16TN(reinterpret_cast<Array<float, 4, 4>*>(&C_data[0]), A, B);

  _C[0] = C_data[0];
  _C[1] = C_data[1];
  _C[acc_stride] = C_data[2];
  _C[acc_stride + 1] = C_data[3];
}

template <int acc_stride>
DEVICE_INLINE void initM16N16K16TN(Array<float, 8, 8>* accumulator) {
  float* _C = reinterpret_cast<float*>(accumulator);
  initM16N8K16TN<acc_stride>(reinterpret_cast<Array<float, 4, 4>*>(&_C[0]));
  initM16N8K16TN<acc_stride>(reinterpret_cast<Array<float, 4, 4>*>(&_C[2]));
}

template <int acc_stride = 2>
DEVICE_INLINE void M16N16K16TN(
    Array<float, 8, 8>* C,
    Array<__half, 8, 8>* A,
    Array<__half, 8, 8>* B) {
  float* _C = reinterpret_cast<float*>(C);
  __half* _B = reinterpret_cast<__half*>(B);
  M16N8K16TN<acc_stride>(
      reinterpret_cast<Array<float, 4, 4>*>(&_C[0]),
      A,
      reinterpret_cast<Array<__half, 4, 4>*>(&_B[0]));
  M16N8K16TN<acc_stride>(
      reinterpret_cast<Array<float, 4, 4>*>(&_C[2]),
      A,
      reinterpret_cast<Array<__half, 4, 4>*>(&_B[4]));
}

} // namespace Turing

#endif // Arch 75

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

namespace Ampere {

namespace util {
// MMA instruction wrappers (sm_75+):
DEVICE_INLINE void m16n8k16TN(
    Array<float, 4, 4>* C,
    Array<__half, 8, 8>* A,
    Array<__half, 4, 4>* B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(B);
  unsigned* _C = reinterpret_cast<unsigned*>(C);
  const unsigned* _D = reinterpret_cast<const unsigned*>(C);

  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(_C[0]), "=r"(_C[1]), "=r"(_C[2]), "=r"(_C[3])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_A[2]),
        "r"(_A[3]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_D[0]),
        "r"(_D[1]),
        "r"(_D[2]),
        "r"(_D[3]));
}

} // namespace util

template <int acc_stride>
DEVICE_INLINE void initM16N8K16TN(Array<float, 4, 4>* accumulator) {
  float* _C = reinterpret_cast<float*>(accumulator);
  _C[0] = 0;
  _C[1] = 0;
  _C[acc_stride] = 0;
  _C[acc_stride + 1] = 0;
}

template <int acc_stride = 2>
DEVICE_INLINE void M16N8K16TN(
    Array<float, 4, 4>* C,
    Array<__half, 8, 8>* A,
    Array<__half, 4, 4>* B) {
  // TODO: in a follow up,
  //    lift this fused swizzle onto iterdomain
  float* _C = reinterpret_cast<float*>(C);
  float C_data[4] = {_C[0], _C[1], _C[acc_stride], _C[acc_stride + 1]};

  util::m16n8k16TN(reinterpret_cast<Array<float, 4, 4>*>(&C_data[0]), A, B);

  _C[0] = C_data[0];
  _C[1] = C_data[1];
  _C[acc_stride] = C_data[2];
  _C[acc_stride + 1] = C_data[3];
}

template <int acc_stride>
DEVICE_INLINE void initM16N16K16TN(Array<float, 8, 8>* accumulator) {
  float* _C = reinterpret_cast<float*>(accumulator);
  initM16N8K16TN<acc_stride>(reinterpret_cast<Array<float, 4, 4>*>(&_C[0]));
  initM16N8K16TN<acc_stride>(reinterpret_cast<Array<float, 4, 4>*>(&_C[2]));
}

template <int acc_stride = 2>
DEVICE_INLINE void M16N16K16TN(
    Array<float, 8, 8>* C,
    Array<__half, 8, 8>* A,
    Array<__half, 8, 8>* B) {
  float* _C = reinterpret_cast<float*>(C);
  __half* _B = reinterpret_cast<__half*>(B);
  M16N8K16TN<acc_stride>(
      reinterpret_cast<Array<float, 4, 4>*>(&_C[0]),
      A,
      reinterpret_cast<Array<__half, 4, 4>*>(&_B[0]));
  M16N8K16TN<acc_stride>(
      reinterpret_cast<Array<float, 4, 4>*>(&_C[2]),
      A,
      reinterpret_cast<Array<__half, 4, 4>*>(&_B[4]));
}

} // namespace Ampere

#endif // Arch 80

#undef DEVICE_INLINE
