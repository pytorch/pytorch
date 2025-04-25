from __future__ import annotations

from typing import Optional

import torch

from ...utils import triton_version_uses_attrs_dict
from ..common import DeviceOpOverrides, register_device_op_overrides


class CUDADeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    def set_device(self, device_idx: int) -> str:
        return f"torch.cuda.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.cuda.synchronize()"

    def device_guard(self, device_idx: int) -> str:
        return f"torch.cuda._DeviceGuard({device_idx})"

    def cpp_device_guard(self) -> str:
        return "at::cuda::CUDAGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "AOTICudaGuard"

    def cpp_stream_guard(self) -> str:
        return "at::cuda::CUDAStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "AOTICudaStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        return "at::cuda::getStreamFromExternal"

    def kernel_header(self) -> str:
        source_codes = """
        #include <c10/cuda/CUDAGuard.h>
        #include <c10/cuda/CUDAStream.h>
        #include <ATen/cuda/EmptyTensor.h>
        """
        return source_codes

    def kernel_driver(self) -> str:
        source_codes = """
            #define CUDA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                CUresult code = EXPR;                          \\
                const char *msg;                               \\
                CUresult code_get_error = cuGetErrorString(code, &msg); \\
                if (code_get_error != CUDA_SUCCESS) {          \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string("invalid error code!"));   \\
                }                                              \\
                if (code != CUDA_SUCCESS) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

            static inline CUfunction loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                CUmodule mod;
                CUfunction func;
                CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
                CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                        func,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline void launchKernel(
                    CUfunction func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    cudaStream_t stream) {
                CUDA_DRIVER_CHECK(cuLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
        """
        if torch.version.hip is not None:
            # Adjusting the warp size to GPU supported wavefront size on AMD GPU
            prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            source_codes = source_codes.replace(
                "32*numWarps", str(prop.warp_size) + "*numWarps"
            )
        return source_codes

    def tma_descriptor_helpers(self) -> str:
        if torch.version.hip is not None:
            raise RuntimeError("Host-side TMA descriptors not supported on HIP.")

        # helper functions for initializing 1D and 2D TMA descriptors in C++. borrowed from the Triton code here:
        # https://github.com/triton-lang/triton/blob/6af4f88591c85de079d8a36a4d7dba67918e2b39/third_party/nvidia/backend/driver.c#L283
        return """
            #if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000
            [[maybe_unused]] static void init1DTMADescriptor(
                    CUtensorMap* m,
                    void* globalAddress,
                    uint64_t dim,
                    uint32_t blockDim,
                    uint32_t elementSize) {
                uint64_t dims[1] = {dim};
                uint64_t globalStrides[1] = {dim * elementSize};
                uint32_t tensorDims[1] = {blockDim};
                uint32_t elementStrides[1] = {1};

                CUtensorMapDataType type;
                switch (elementSize) {
                case 1:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
                    break;
                case 2:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
                    break;
                case 4:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
                    break;
                default:
                    throw std::runtime_error("elementSize must be 1, 2, or 4");
                }

                if (elementSize * blockDim < 32) {
                    throw std::runtime_error("block size too small");
                }

                int rank = 1;

                CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
                    m, type, rank, globalAddress, dims,
                    globalStrides, tensorDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            }

            [[maybe_unused]] static void init2DTMADescriptor(
                    CUtensorMap* m,
                    void* globalAddress,
                    uint64_t dim1,
                    uint64_t dim0,
                    uint32_t blockDim1,
                    uint32_t blockDim0,
                    uint32_t elementSize) {
                uint64_t dims[2] = {dim0, dim1};
                uint32_t tensorDims[2] = {blockDim0, blockDim1};
                uint64_t globalStrides[2] = {dims[0] * elementSize,
                                             dims[0] * dims[1] * elementSize};
                uint32_t elementStrides[2] = {1, 1};

                CUtensorMapDataType type;
                switch (elementSize) {
                case 1:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
                    break;
                case 2:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
                    break;
                case 4:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
                    break;
                default:
                    throw std::runtime_error("elementSize must be 1, 2, or 4");
                }

                int rank = 2;

                CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
                uint32_t contigDimSizeInByte = elementSize * tensorDims[0];
                if (contigDimSizeInByte >= 128) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
                } else if (contigDimSizeInByte >= 64) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
                } else if (contigDimSizeInByte >= 32) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
                } else {
                    throw std::runtime_error("block size too small");
                }

                if (contigDimSizeInByte > 128) {
                    tensorDims[0] = 128 / elementSize;
                }

                CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
                    m, type, rank, globalAddress, dims,
                    globalStrides, tensorDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            }
            #endif
        """

    def cpp_stream_type(self) -> str:
        return "cudaStream_t"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_get_current_cuda_stream"

    def cpp_kernel_type(self) -> str:
        return "CUfunction"

    def cpp_device_ptr(self) -> str:
        return "CUdeviceptr"

    def cpp_global_scratch(self, idx: int) -> Optional[tuple[str, str]]:
        if triton_version_uses_attrs_dict():
            return f"CUdeviceptr global_scratch_{idx} = 0;", f"global_scratch_{idx}"
        return None


register_device_op_overrides("cuda", CUDADeviceOpOverrides())
