#ifndef CAFFE2_SNPE_FFI_H_
#define CAFFE2_SNPE_FFI_H_

#include <stdint.h>
#include <string>

namespace caffe2 {

std::string& gSNPELocation();

const char* const snpe_ffi_so = "libsnpe_ffi.so";

}

extern "C" {

bool snpe_has_gpu();

void* snpe_create(const uint8_t* container, size_t size, const char* input_name);

void snpe_destroy(void* ctx);

void snpe_get_input_dims(void* ctx, size_t const** dims, size_t* size);

void snpe_run(void* ctx,
              const float* inputData,
              size_t inputSize,
              size_t const** outputDims,
              size_t* outputSize);

void snpe_copy_output_to(void* ctx, float* outputData);

}

#endif  // CAFFE2_SNPE_FFI_H_
