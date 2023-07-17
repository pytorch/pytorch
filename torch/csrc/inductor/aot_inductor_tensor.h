#ifndef __AOT_INDUCTOR_TENSOR__
#define __AOT_INDUCTOR_TENSOR__

#include <stdint.h>

#define EXPORT __attribute__((__visibility__("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  kAotInductorCPU = 1,
  kAotInductorCUDA = 2,
} AotInductorDeviceType;

typedef struct {
  AotInductorDeviceType device_type;
  int8_t device_id;
} AotInductorDevice;

typedef enum {
  kAotInductorByte = 1,
  kAotInductorChar = 2,
  kAotInductorShort = 3,
  kAotInductorInt = 4,
  kAotInductorLong = 5,
  kAotInductorHalf = 6,
  kAotInductorFloat = 7,
  kAotInductorDouble = 8,
  kAotInductorComplexHalf = 9,
  kAotInductorComplexFloat = 10,
  kAotInductorComplexDouble = 11,
  kAotInductorBool = 12,
  kAotInductorBFloat16 = 13,
} AotInductorScalarType;

// Bare minumum tensor struct to interchange data between AOTInducotor
// generated shared library and libtorch to avoid ABI compatibility issues
typedef struct {
  void* data_ptr;
  AotInductorDevice device;
  AotInductorScalarType type;
  int64_t ndim;
  const int64_t* sizes;
  const int64_t* strides;
} AotInductorTensor;

EXPORT AotInductorTensor convert_to_aot_inductor_tensor(void* aten_tensor);

EXPORT void convert_to_aten_tensor(
    AotInductorTensor inductor_tensor,
    void* aten_tensor);

EXPORT AotInductorTensor aot_inductor_empty_strided(
    int64_t dim,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    AotInductorDevice device,
    AotInductorScalarType type);

EXPORT AotInductorTensor aot_inductor_as_strided(
    AotInductorTensor self,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    int64_t offset);

EXPORT AotInductorTensor aot_inductor_addmm_out(
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
