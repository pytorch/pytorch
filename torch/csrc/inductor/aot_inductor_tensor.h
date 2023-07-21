#ifndef __AOT_INDUCTOR_TENSOR__
#define __AOT_INDUCTOR_TENSOR__

#include <stdint.h>
#include <memory>

#ifdef __GNUC__
#define AOT_INDUCTOR_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
#define AOT_INDUCTOR_EXPORT __declspec(dllexport)
#else // !_WIN32
#define AOT_INDUCTOR_EXPORT
#endif // _WIN32
#endif // __GNUC__

#define AOT_INDUCTOR_CHECK(cond, msg) \
  { assert((cond) && (msg)); }

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  kAotInductorCPU = 0,
  kAotInductorCUDA = 1,
} AotInductorDeviceType;

typedef enum {
  kAotInductorByte = 0,
  kAotInductorChar = 1,
  kAotInductorShort = 2,
  kAotInductorInt = 3,
  kAotInductorLong = 4,
  kAotInductorHalf = 5,
  kAotInductorFloat = 6,
  kAotInductorDouble = 7,
  kAotInductorComplexHalf = 8,
  kAotInductorComplexFloat = 9,
  kAotInductorComplexDouble = 10,
  kAotInductorBool = 11,
  kAotInductorQInt8 = 12,
  kAotInductorQUInt8 = 13,
  kAotInductorQInt32 = 14,
  kAotInductorBFloat16 = 15,
} AotInductorScalarType;

typedef struct {
  AotInductorDeviceType device_type;
  int8_t device_id;
} AotInductorDevice;

// using DeleterFnPtr  void (*)(void*);

// Bare minumum tensor struct to interchange data between AOTInducotor
// generated shared library and libtorch to avoid ABI compatibility issues
typedef struct {
  void* data_ptr;
  void* deleter;
  AotInductorDevice device;
  AotInductorScalarType dtype;
  int64_t ndim;
  const int64_t* sizes;
  const int64_t* strides;
  int64_t storage_offset;
} AotInductorTensor;

AOT_INDUCTOR_EXPORT void aot_inductor_initialize();

AOT_INDUCTOR_EXPORT void aot_inductor_destroy();

AOT_INDUCTOR_EXPORT void aot_inductor_tensor_free(AotInductorTensor aot_tensor);

AOT_INDUCTOR_EXPORT AotInductorTensor
aten_tensor_to_aot_tensor(void* aten_tensor, char keep_live);

AOT_INDUCTOR_EXPORT void aot_tensor_to_aten_tensor(
    AotInductorTensor aot_tensor,
    void* aten_tensor);

AOT_INDUCTOR_EXPORT AotInductorTensor aot_inductor_empty_strided(
    int64_t ndim,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    AotInductorDevice device,
    AotInductorScalarType type);

AOT_INDUCTOR_EXPORT AotInductorTensor aot_inductor_as_strided(
    AotInductorTensor self,
    int64_t ndim,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    int64_t offset);

AOT_INDUCTOR_EXPORT AotInductorTensor aot_inductor_addmm_out(
    AotInductorTensor out,
    AotInductorTensor self,
    AotInductorTensor mat1,
    AotInductorTensor mat2,
    float beta,
    float alpha);

#ifdef __cplusplus
}
#endif

#endif
