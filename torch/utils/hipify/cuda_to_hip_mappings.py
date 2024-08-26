import collections

from .constants import (API_BLAS, API_C10, API_CAFFE2, API_DRIVER, API_FFT,
                        API_PYTORCH, API_RAND, API_ROCTX, API_RTC, API_RUNTIME,
                        API_SPECIAL, API_ROCMSMI, CONV_CACHE, CONV_CONTEXT, CONV_D3D9,
                        CONV_D3D10, CONV_D3D11, CONV_DEF, CONV_DEVICE,
                        CONV_DEVICE_FUNC, CONV_EGL, CONV_ERROR, CONV_EVENT,
                        CONV_EXEC, CONV_GL, CONV_GRAPHICS, CONV_INCLUDE,
                        CONV_INCLUDE_CUDA_MAIN_H, CONV_INIT, CONV_JIT,
                        CONV_MATH_FUNC, CONV_MEM, CONV_MODULE,
                        CONV_NUMERIC_LITERAL, CONV_OCCUPANCY, CONV_OTHER,
                        CONV_PEER, CONV_SPECIAL_FUNC, CONV_STREAM,
                        CONV_SURFACE, CONV_TEX, CONV_THREAD, CONV_TYPE,
                        CONV_VDPAU, CONV_VERSION, HIP_UNSUPPORTED)

""" Mapping of CUDA functions, include files, constants, and types to ROCm/HIP equivalents
This closely follows the implementation in hipify-clang
https://github.com/ROCm-Developer-Tools/HIP/blob/master/hipify-clang/src/CUDA2HipMap.cpp
and its structure.
There are different maps for fundamental names, include files, identifies, sparse, and
PyTorch specific translations.
Each of the entries in these maps translates a CUDA string to a tuple containing the
ROCm/HIP string, a type and API annotation and - optionally - an annotation if it is not
supported in ROCm/HIP yet.
"""

# List of math functions that should be replaced inside device code only.
MATH_TRANSPILATIONS = collections.OrderedDict(
    [
        ("std::max", ("::max")),
        ("std::min", ("::min")),
        ("std::ceil", ("::ceil")),
        ("std::floor", ("::floor")),
        ("std::exp", ("::exp")),
        ("std::log", ("::log")),
        ("std::pow", ("::pow")),
        ("std::fabs", ("::fabs")),
        ("std::fmod", ("::fmod")),
        ("std::remainder", ("::remainder")),
        ("std::frexp", ("::frexp")),
    ]
)

CUDA_TYPE_NAME_MAP = collections.OrderedDict(
    [
        ("CUresult", ("hipError_t", CONV_TYPE, API_DRIVER)),
        ("cudaError_t", ("hipError_t", CONV_TYPE, API_RUNTIME)),
        ("cudaError", ("hipError_t", CONV_TYPE, API_RUNTIME)),
        (
            "CUDA_ARRAY3D_DESCRIPTOR",
            ("HIP_ARRAY3D_DESCRIPTOR", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUDA_ARRAY_DESCRIPTOR", ("HIP_ARRAY_DESCRIPTOR", CONV_TYPE, API_DRIVER)),
        ("CUDA_MEMCPY2D", ("hip_Memcpy2D", CONV_TYPE, API_DRIVER)),
        ("CUDA_MEMCPY3D", ("HIP_MEMCPY3D", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUDA_MEMCPY3D_PEER",
            ("HIP_MEMCPY3D_PEER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_POINTER_ATTRIBUTE_P2P_TOKENS",
            (
                "HIP_POINTER_ATTRIBUTE_P2P_TOKENS",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_RESOURCE_DESC",
            ("HIP_RESOURCE_DESC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_RESOURCE_VIEW_DESC",
            ("HIP_RESOURCE_VIEW_DESC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUipcEventHandle",
            ("hipIpcEventHandle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUipcMemHandle", ("hipIpcMemHandle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUaddress_mode", ("hipAddress_mode", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUarray_cubemap_face",
            ("hipArray_cubemap_face", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUarray_format", ("hipArray_format", CONV_TYPE, API_DRIVER)),
        ("CUcomputemode", ("hipComputemode", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUmem_advise", ("hipMemAdvise", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUmem_range_attribute",
            ("hipMemRangeAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUctx_flags", ("hipCctx_flags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUdevice", ("hipDevice_t", CONV_TYPE, API_DRIVER)),
        ("CUdevice_attribute_enum", ("hipDeviceAttribute_t", CONV_TYPE, API_DRIVER)),
        ("CUdevice_attribute", ("hipDeviceAttribute_t", CONV_TYPE, API_DRIVER)),
        ("CUpointer_attribute", ("hipPointer_attribute", CONV_TYPE, API_DRIVER)),
        ("CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL", ("HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL", CONV_TYPE, API_DRIVER)),
        ("CU_POINTER_ATTRIBUTE_BUFFER_ID", ("HIP_POINTER_ATTRIBUTE_BUFFER_ID", CONV_TYPE, API_DRIVER)),
        ("CUdeviceptr", ("hipDeviceptr_t", CONV_TYPE, API_DRIVER)),
        ("CUarray_st", ("hipArray", CONV_TYPE, API_DRIVER)),
        ("CUarray", ("hipArray *", CONV_TYPE, API_DRIVER)),
        ("CUdevprop_st", ("hipDeviceProp_t", CONV_TYPE, API_DRIVER)),
        ("CUdevprop", ("hipDeviceProp_t", CONV_TYPE, API_DRIVER)),
        ("CUfunction", ("hipFunction_t", CONV_TYPE, API_DRIVER)),
        (
            "CUgraphicsResource",
            ("hipGraphicsResource_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUmipmappedArray",
            ("hipMipmappedArray_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUfunction_attribute",
            ("hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUfunction_attribute_enum",
            ("hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsMapResourceFlags",
            ("hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsMapResourceFlags_enum",
            ("hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsRegisterFlags",
            ("hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsRegisterFlags_enum",
            ("hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUoccupancy_flags",
            ("hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUoccupancy_flags_enum",
            ("hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUfunc_cache_enum", ("hipFuncCache", CONV_TYPE, API_DRIVER)),
        ("CUfunc_cache", ("hipFuncCache", CONV_TYPE, API_DRIVER)),
        ("CUipcMem_flags", ("hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUipcMem_flags_enum",
            ("hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUjit_cacheMode", ("hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUjit_cacheMode_enum",
            ("hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUjit_fallback", ("hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUjit_fallback_enum",
            ("hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUjit_option", ("hipJitOption", CONV_JIT, API_DRIVER)),
        ("CUjit_option_enum", ("hipJitOption", CONV_JIT, API_DRIVER)),
        ("CUjit_target", ("hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUjit_target_enum", ("hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUjitInputType", ("hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUjitInputType_enum",
            ("hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUlimit", ("hipLimit_t", CONV_TYPE, API_DRIVER)),
        ("CUlimit_enum", ("hipLimit_t", CONV_TYPE, API_DRIVER)),
        (
            "CUmemAttach_flags",
            ("hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUmemAttach_flags_enum",
            ("hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUmemorytype", ("hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUmemorytype_enum", ("hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUresourcetype", ("hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUresourcetype_enum",
            ("hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUresourceViewFormat", ("hipResourceViewFormat", CONV_TEX, API_DRIVER)),
        ("CUresourceViewFormat_enum", ("hipResourceViewFormat", CONV_TEX, API_DRIVER)),
        ("CUsharedconfig", ("hipSharedMemConfig", CONV_TYPE, API_DRIVER)),
        ("CUsharedconfig_enum", ("hipSharedMemConfig", CONV_TYPE, API_DRIVER)),
        ("CUcontext", ("hipCtx_t", CONV_TYPE, API_DRIVER)),
        ("CUmodule", ("hipModule_t", CONV_TYPE, API_DRIVER)),
        ("CUstream", ("hipStream_t", CONV_TYPE, API_DRIVER)),
        ("CUstream_st", ("ihipStream_t", CONV_TYPE, API_DRIVER)),
        ("CUstreamCallback", ("hipStreamCallback_t", CONV_TYPE, API_DRIVER)),
        ("CUsurfObject", ("hipSurfaceObject", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUsurfref",
            ("hipSurfaceReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUtexObject", ("hipTextureObject_t", CONV_TYPE, API_DRIVER)),
        ("CUtexref", ("textureReference", CONV_TYPE, API_DRIVER)),
        ("CUstream_flags", ("hipStreamFlags", CONV_TYPE, API_DRIVER)),
        (
            "CUstreamWaitValue_flags",
            ("hipStreamWaitValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUstreamWriteValue_flags",
            ("hipStreamWriteValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUstreamBatchMemOpType",
            ("hipStreamBatchMemOpType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUdevice_P2PAttribute",
            ("hipDeviceP2PAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUevent", ("hipEvent_t", CONV_TYPE, API_DRIVER)),
        ("CUevent_st", ("ihipEvent_t", CONV_TYPE, API_DRIVER)),
        ("CUevent_flags", ("hipEventFlags", CONV_EVENT, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUfilter_mode", ("hipTextureFilterMode", CONV_TEX, API_DRIVER)),
        ("CUGLDeviceList", ("hipGLDeviceList", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUGLmap_flags", ("hipGLMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUd3d9DeviceList",
            ("hipD3D9DeviceList", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d9map_flags",
            ("hipD3D9MapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d9register_flags",
            ("hipD3D9RegisterFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10DeviceList",
            ("hipd3d10DeviceList", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10map_flags",
            ("hipD3D10MapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10register_flags",
            ("hipD3D10RegisterFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d11DeviceList",
            ("hipd3d11DeviceList", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUeglStreamConnection_st",
            ("hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUeglStreamConnection",
            ("hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "libraryPropertyType_t",
            ("hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "libraryPropertyType",
            ("hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaStreamCallback_t", ("hipStreamCallback_t", CONV_TYPE, API_RUNTIME)),
        ("cudaArray", ("hipArray", CONV_MEM, API_RUNTIME)),
        ("cudaArray_t", ("hipArray_t", CONV_MEM, API_RUNTIME)),
        ("cudaArray_const_t", ("hipArray_const_t", CONV_MEM, API_RUNTIME)),
        ("cudaMipmappedArray_t", ("hipMipmappedArray_t", CONV_MEM, API_RUNTIME)),
        (
            "cudaMipmappedArray_const_t",
            ("hipMipmappedArray_const_t", CONV_MEM, API_RUNTIME),
        ),
        ("cudaArrayDefault", ("hipArrayDefault", CONV_MEM, API_RUNTIME)),
        ("cudaArrayLayered", ("hipArrayLayered", CONV_MEM, API_RUNTIME)),
        (
            "cudaArraySurfaceLoadStore",
            ("hipArraySurfaceLoadStore", CONV_MEM, API_RUNTIME),
        ),
        ("cudaArrayCubemap", ("hipArrayCubemap", CONV_MEM, API_RUNTIME)),
        ("cudaArrayTextureGather", ("hipArrayTextureGather", CONV_MEM, API_RUNTIME)),
        ("cudaMemoryAdvise", ("hipMemoryAdvise", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaMemRangeAttribute",
            ("hipMemRangeAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMemcpyKind", ("hipMemcpyKind", CONV_MEM, API_RUNTIME)),
        ("cudaMemoryType", ("hipMemoryType", CONV_MEM, API_RUNTIME)),
        ("cudaExtent", ("hipExtent", CONV_MEM, API_RUNTIME)),
        ("cudaPitchedPtr", ("hipPitchedPtr", CONV_MEM, API_RUNTIME)),
        ("cudaPos", ("hipPos", CONV_MEM, API_RUNTIME)),
        ("cudaEvent_t", ("hipEvent_t", CONV_TYPE, API_RUNTIME)),
        ("cudaStream_t", ("hipStream_t", CONV_TYPE, API_RUNTIME)),
        ("cudaPointerAttributes", ("hipPointerAttribute_t", CONV_TYPE, API_RUNTIME)),
        ("cudaDeviceAttr", ("hipDeviceAttribute_t", CONV_TYPE, API_RUNTIME)),
        ("cudaDeviceProp", ("hipDeviceProp_t", CONV_TYPE, API_RUNTIME)),
        (
            "cudaDeviceP2PAttr",
            ("hipDeviceP2PAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeMode",
            ("hipComputeMode", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaFuncCache", ("hipFuncCache_t", CONV_CACHE, API_RUNTIME)),
        (
            "cudaFuncAttributes",
            ("hipFuncAttributes", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaSharedMemConfig", ("hipSharedMemConfig", CONV_TYPE, API_RUNTIME)),
        ("cudaLimit", ("hipLimit_t", CONV_TYPE, API_RUNTIME)),
        ("cudaOutputMode", ("hipOutputMode", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED)),
        ("cudaTextureReadMode", ("hipTextureReadMode", CONV_TEX, API_RUNTIME)),
        ("cudaTextureFilterMode", ("hipTextureFilterMode", CONV_TEX, API_RUNTIME)),
        ("cudaChannelFormatKind", ("hipChannelFormatKind", CONV_TEX, API_RUNTIME)),
        ("cudaChannelFormatDesc", ("hipChannelFormatDesc", CONV_TEX, API_RUNTIME)),
        ("cudaResourceDesc", ("hipResourceDesc", CONV_TEX, API_RUNTIME)),
        ("cudaResourceViewDesc", ("hipResourceViewDesc", CONV_TEX, API_RUNTIME)),
        ("cudaTextureDesc", ("hipTextureDesc", CONV_TEX, API_RUNTIME)),
        (
            "surfaceReference",
            ("hipSurfaceReference", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaTextureObject_t", ("hipTextureObject_t", CONV_TEX, API_RUNTIME)),
        ("cudaResourceType", ("hipResourceType", CONV_TEX, API_RUNTIME)),
        ("cudaResourceViewFormat", ("hipResourceViewFormat", CONV_TEX, API_RUNTIME)),
        ("cudaTextureAddressMode", ("hipTextureAddressMode", CONV_TEX, API_RUNTIME)),
        (
            "cudaSurfaceBoundaryMode",
            ("hipSurfaceBoundaryMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaSurfaceFormatMode",
            ("hipSurfaceFormatMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaTextureType1D", ("hipTextureType1D", CONV_TEX, API_RUNTIME)),
        ("cudaTextureType2D", ("hipTextureType2D", CONV_TEX, API_RUNTIME)),
        ("cudaTextureType3D", ("hipTextureType3D", CONV_TEX, API_RUNTIME)),
        ("cudaTextureTypeCubemap", ("hipTextureTypeCubemap", CONV_TEX, API_RUNTIME)),
        (
            "cudaTextureType1DLayered",
            ("hipTextureType1DLayered", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaTextureType2DLayered",
            ("hipTextureType2DLayered", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaTextureTypeCubemapLayered",
            ("hipTextureTypeCubemapLayered", CONV_TEX, API_RUNTIME),
        ),
        ("cudaIpcEventHandle_t", ("hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME)),
        ("cudaIpcEventHandle_st", ("hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME)),
        ("cudaIpcMemHandle_t", ("hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME)),
        ("cudaIpcMemHandle_st", ("hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME)),
        (
            "cudaGraphicsCubeFace",
            ("hipGraphicsCubeFace", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsMapFlags",
            ("hipGraphicsMapFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsRegisterFlags",
            ("hipGraphicsRegisterFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLDeviceList",
            ("hipGLDeviceList", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaGLMapFlags", ("hipGLMapFlags", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaD3D9DeviceList",
            ("hipD3D9DeviceList", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9MapFlags",
            ("hipD3D9MapFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9RegisterFlags",
            ("hipD3D9RegisterFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10DeviceList",
            ("hipd3d10DeviceList", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10MapFlags",
            ("hipD3D10MapFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10RegisterFlags",
            ("hipD3D10RegisterFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11DeviceList",
            ("hipd3d11DeviceList", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaEglStreamConnection",
            ("hipEglStreamConnection", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cublasHandle_t", ("hipblasHandle_t", CONV_TYPE, API_BLAS)),
        ("cublasOperation_t", ("hipblasOperation_t", CONV_TYPE, API_BLAS)),
        ("cublasStatus_t", ("hipblasStatus_t", CONV_TYPE, API_BLAS)),
        ("cublasFillMode_t", ("hipblasFillMode_t", CONV_TYPE, API_BLAS)),
        ("cublasDiagType_t", ("hipblasDiagType_t", CONV_TYPE, API_BLAS)),
        ("cublasSideMode_t", ("hipblasSideMode_t", CONV_TYPE, API_BLAS)),
        ("cublasPointerMode_t", ("hipblasPointerMode_t", CONV_TYPE, API_BLAS)),
        ("cublasGemmAlgo_t", ("hipblasGemmAlgo_t", CONV_TYPE, API_BLAS)),
        (
            "cublasAtomicsMode_t",
            ("hipblasAtomicsMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDataType_t",
            ("hipblasDatatype_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("curandStatus", ("hiprandStatus_t", CONV_TYPE, API_RAND)),
        ("curandStatus_t", ("hiprandStatus_t", CONV_TYPE, API_RAND)),
        ("curandRngType", ("hiprandRngType_t", CONV_TYPE, API_RAND)),
        ("curandRngType_t", ("hiprandRngType_t", CONV_TYPE, API_RAND)),
        ("curandGenerator_st", ("hiprandGenerator_st", CONV_TYPE, API_RAND)),
        ("curandGenerator_t", ("hiprandGenerator_t", CONV_TYPE, API_RAND)),
        (
            "curandDirectionVectorSet",
            ("hiprandDirectionVectorSet_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDirectionVectorSet_t",
            ("hiprandDirectionVectorSet_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandOrdering", ("hiprandOrdering_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED)),
        (
            "curandOrdering_t",
            ("hiprandOrdering_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistribution_st",
            ("hiprandDistribution_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2V_st",
            ("hiprandDistribution_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistribution_t",
            ("hiprandDistribution_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2V_t",
            ("hiprandDistribution_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionShift_st",
            ("hiprandDistributionShift_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionShift_t",
            ("hiprandDistributionShift_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionM2Shift_st",
            ("hiprandDistributionM2Shift_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionM2Shift_t",
            ("hiprandDistributionM2Shift_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2_st",
            ("hiprandHistogramM2_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2_t",
            ("hiprandHistogramM2_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2K_st",
            ("hiprandHistogramM2K_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2K_t",
            ("hiprandHistogramM2K_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDiscreteDistribution_st",
            ("hiprandDiscreteDistribution_st", CONV_TYPE, API_RAND),
        ),
        (
            "curandDiscreteDistribution_t",
            ("hiprandDiscreteDistribution_t", CONV_TYPE, API_RAND),
        ),
        ("curandMethod", ("hiprandMethod_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED)),
        ("curandMethod_t", ("hiprandMethod_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED)),
        (
            "curandDirectionVectors32_t",
            ("hiprandDirectionVectors32_t", CONV_TYPE, API_RAND),
        ),
        (
            "curandDirectionVectors64_t",
            ("hiprandDirectionVectors64_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandStateMtgp32_t", ("hiprandStateMtgp32_t", CONV_TYPE, API_RAND)),
        ("curandStateMtgp32", ("hiprandStateMtgp32_t", CONV_TYPE, API_RAND)),
        (
            "curandStateScrambledSobol64_t",
            ("hiprandStateScrambledSobol64_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandStateSobol64_t",
            ("hiprandStateSobol64_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandStateScrambledSobol32_t",
            ("hiprandStateScrambledSobol32_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandStateSobol32_t", ("hiprandStateSobol32_t", CONV_TYPE, API_RAND)),
        ("curandStateMRG32k3a_t", ("hiprandStateMRG32k3a_t", CONV_TYPE, API_RAND)),
        (
            "curandStatePhilox4_32_10_t",
            ("hiprandStatePhilox4_32_10_t", CONV_TYPE, API_RAND),
        ),
        ("curandStateXORWOW_t", ("hiprandStateXORWOW_t", CONV_TYPE, API_RAND)),
        ("curandState_t", ("hiprandState_t", CONV_TYPE, API_RAND)),
        ("curandState", ("hiprandState_t", CONV_TYPE, API_RAND)),
        ("CUuuid", ("hipUUID", CONV_TYPE, API_RUNTIME)),
        ("cudaGraph_t", ("hipGraph_t", CONV_TYPE, API_RAND)),
        ("cudaGraphExec_t", ("hipGraphExec_t", CONV_TYPE, API_RAND)),
        ("__nv_bfloat16", ("__hip_bfloat16", CONV_TYPE, API_RUNTIME)),
        ("__nv_bfloat162", ("__hip_bfloat162", CONV_TYPE, API_RUNTIME)),
    ]
)

CUDA_INCLUDE_MAP = collections.OrderedDict(
    [
        # since pytorch uses "\b{pattern}\b" as the actual re pattern,
        # patterns listed here have to begin and end with alnum chars
        (
            "include <cuda.h",
            ("include <hip/hip_runtime.h", CONV_INCLUDE_CUDA_MAIN_H, API_DRIVER),
        ),
        (
            'include "cuda.h',
            ('include "hip/hip_runtime.h', CONV_INCLUDE_CUDA_MAIN_H, API_DRIVER),
        ),
        (
            "cuda_runtime.h",
            ("hip/hip_runtime.h", CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME),
        ),
        ("cuda_runtime_api.h", ("hip/hip_runtime_api.h", CONV_INCLUDE, API_RUNTIME)),
        ("cuda_profiler_api.h", ("hip/hip_runtime_api.h", CONV_INCLUDE, API_RUNTIME)),
        (
            "channel_descriptor.h",
            ("hip/channel_descriptor.h", CONV_INCLUDE, API_RUNTIME),
        ),
        ("device_functions.h", ("hip/device_functions.h", CONV_INCLUDE, API_RUNTIME)),
        ("driver_types.h", ("hip/driver_types.h", CONV_INCLUDE, API_RUNTIME)),
        ("library_types.h", ("hip/library_types.h", CONV_INCLUDE, API_RUNTIME)),
        ("cuComplex.h", ("hip/hip_complex.h", CONV_INCLUDE, API_RUNTIME)),
        ("cuda_fp16.h", ("hip/hip_fp16.h", CONV_INCLUDE, API_RUNTIME)),
        ("cuda_bf16.h", ("hip/hip_bf16.h", CONV_INCLUDE, API_RUNTIME)),
        (
            "cuda_texture_types.h",
            ("hip/hip_texture_types.h", CONV_INCLUDE, API_RUNTIME),
        ),
        ("cooperative_groups.h", ("hip/hip_cooperative_groups.h", CONV_INCLUDE, API_RUNTIME)),
        ("vector_types.h", ("hip/hip_vector_types.h", CONV_INCLUDE, API_RUNTIME)),
        ("cublas.h", ("hipblas/hipblas.h", CONV_INCLUDE_CUDA_MAIN_H, API_BLAS)),
        ("cublas_v2.h", ("hipblas/hipblas.h", CONV_INCLUDE_CUDA_MAIN_H, API_BLAS)),
        ("cublasLt.h", ("hipblaslt/hipblaslt.h", CONV_INCLUDE_CUDA_MAIN_H, API_BLAS)),
        ("curand.h", ("hiprand/hiprand.h", CONV_INCLUDE_CUDA_MAIN_H, API_RAND)),
        ("curand_kernel.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_discrete.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_discrete2.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_globals.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_lognormal.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_mrg32k3a.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_mtgp32.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_mtgp32_host.h", ("hiprand/hiprand_mtgp32_host.h", CONV_INCLUDE, API_RAND)),
        ("curand_mtgp32_kernel.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        (
            "curand_mtgp32dc_p_11213.h",
            ("rocrand/rocrand_mtgp32_11213.h", CONV_INCLUDE, API_RAND),
        ),
        ("curand_normal.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_normal_static.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_philox4x32_x.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_poisson.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_precalc.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_uniform.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("cusparse.h", ("hipsparse/hipsparse.h", CONV_INCLUDE, API_RAND)),
        ("cufft.h", ("hipfft/hipfft.h", CONV_INCLUDE, API_BLAS)),
        ("cufftXt.h", ("hipfft/hipfftXt.h", CONV_INCLUDE, API_BLAS)),
        # PyTorch also has a source file named "nccl.h", so we need to "<"">" to differentiate
        ("<nccl.h>", ("<rccl/rccl.h>", CONV_INCLUDE, API_RUNTIME)),
        ("nvrtc.h", ("hip/hiprtc.h", CONV_INCLUDE, API_RTC)),
        ("thrust/system/cuda", ("thrust/system/hip", CONV_INCLUDE, API_BLAS)),
        ("cub/util_allocator.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_reduce.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_raking_layout.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/cub.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/config.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/util_ptx.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/util_type.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_run_length_encode.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_load.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_store.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_scan.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_radix_sort.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_reduce.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_scan.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_select.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("nvtx3/nvtx3.hpp", ("roctracer/roctx.h", CONV_INCLUDE, API_ROCTX)),
        ("nvToolsExt.h", ("roctracer/roctx.h", CONV_INCLUDE, API_ROCTX)),
        ("nvml.h", ("rocm_smi/rocm_smi.h", CONV_INCLUDE, API_ROCMSMI)),
    ]
)

CUDA_IDENTIFIER_MAP = collections.OrderedDict(
    [
        ("__CUDACC__", ("__HIPCC__", CONV_DEF, API_RUNTIME)),
        (
            "CUDA_ERROR_INVALID_CONTEXT",
            ("hipErrorInvalidContext", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            ("hipErrorContextAlreadyCurrent", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_ARRAY_IS_MAPPED",
            ("hipErrorArrayIsMapped", CONV_TYPE, API_DRIVER),
        ),
        ("CUDA_ERROR_ALREADY_MAPPED", ("hipErrorAlreadyMapped", CONV_TYPE, API_DRIVER)),
        (
            "CUDA_ERROR_ALREADY_ACQUIRED",
            ("hipErrorAlreadyAcquired", CONV_TYPE, API_DRIVER),
        ),
        ("CUDA_ERROR_NOT_MAPPED", ("hipErrorNotMapped", CONV_TYPE, API_DRIVER)),
        (
            "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
            ("hipErrorNotMappedAsArray", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
            ("hipErrorNotMappedAsPointer", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
            ("hipErrorContextAlreadyInUse", CONV_TYPE, API_DRIVER),
        ),
        ("CUDA_ERROR_INVALID_SOURCE", ("hipErrorInvalidSource", CONV_TYPE, API_DRIVER)),
        ("CUDA_ERROR_FILE_NOT_FOUND", ("hipErrorFileNotFound", CONV_TYPE, API_DRIVER)),
        ("CUDA_ERROR_NOT_FOUND", ("hipErrorNotFound", CONV_TYPE, API_DRIVER)),
        (
            "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
            (
                "hipErrorLaunchIncompatibleTexturing",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            ("hipErrorPrimaryContextActive", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_CONTEXT_IS_DESTROYED",
            ("hipErrorContextIsDestroyed", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NOT_PERMITTED",
            ("hipErrorNotPermitted", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NOT_SUPPORTED",
            ("hipErrorNotSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMissingConfiguration",
            ("hipErrorMissingConfiguration", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorPriorLaunchFailure",
            ("hipErrorPriorLaunchFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidDeviceFunction",
            ("hipErrorInvalidDeviceFunction", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidConfiguration",
            ("hipErrorInvalidConfiguration", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidPitchValue",
            ("hipErrorInvalidPitchValue", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidSymbol",
            ("hipErrorInvalidSymbol", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidHostPointer",
            ("hipErrorInvalidHostPointer", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidDevicePointer",
            ("hipErrorInvalidDevicePointer", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaErrorInvalidTexture",
            ("hipErrorInvalidTexture", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidTextureBinding",
            ("hipErrorInvalidTextureBinding", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidChannelDescriptor",
            (
                "hipErrorInvalidChannelDescriptor",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorInvalidMemcpyDirection",
            ("hipErrorInvalidMemcpyDirection", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorAddressOfConstant",
            ("hipErrorAddressOfConstant", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTextureFetchFailed",
            ("hipErrorTextureFetchFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTextureNotBound",
            ("hipErrorTextureNotBound", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSynchronizationError",
            ("hipErrorSynchronizationError", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidFilterSetting",
            ("hipErrorInvalidFilterSetting", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidNormSetting",
            ("hipErrorInvalidNormSetting", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMixedDeviceExecution",
            ("hipErrorMixedDeviceExecution", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNotYetImplemented",
            ("hipErrorNotYetImplemented", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMemoryValueTooLarge",
            ("hipErrorMemoryValueTooLarge", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInsufficientDriver",
            ("hipErrorInsufficientDriver", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSetOnActiveProcess",
            ("hipErrorSetOnActiveProcess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidSurface",
            ("hipErrorInvalidSurface", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateVariableName",
            ("hipErrorDuplicateVariableName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateTextureName",
            ("hipErrorDuplicateTextureName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateSurfaceName",
            ("hipErrorDuplicateSurfaceName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDevicesUnavailable",
            ("hipErrorDevicesUnavailable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorIncompatibleDriverContext",
            (
                "hipErrorIncompatibleDriverContext",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorDeviceAlreadyInUse",
            ("hipErrorDeviceAlreadyInUse", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchMaxDepthExceeded",
            ("hipErrorLaunchMaxDepthExceeded", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFileScopedTex",
            ("hipErrorLaunchFileScopedTex", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFileScopedSurf",
            ("hipErrorLaunchFileScopedSurf", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSyncDepthExceeded",
            ("hipErrorSyncDepthExceeded", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchPendingCountExceeded",
            (
                "hipErrorLaunchPendingCountExceeded",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorNotPermitted",
            ("hipErrorNotPermitted", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNotSupported",
            ("hipErrorNotSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorStartupFailure",
            ("hipErrorStartupFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorApiFailureBase",
            ("hipErrorApiFailureBase", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_SUCCESS", ("hipSuccess", CONV_TYPE, API_DRIVER)),
        ("cudaSuccess", ("hipSuccess", CONV_TYPE, API_RUNTIME)),
        ("CUDA_ERROR_INVALID_VALUE", ("hipErrorInvalidValue", CONV_TYPE, API_DRIVER)),
        ("cudaErrorInvalidValue", ("hipErrorInvalidValue", CONV_TYPE, API_RUNTIME)),
        (
            "CUDA_ERROR_OUT_OF_MEMORY",
            ("hipErrorMemoryAllocation", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorMemoryAllocation",
            ("hipErrorMemoryAllocation", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_NOT_INITIALIZED",
            ("hipErrorNotInitialized", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInitializationError",
            ("hipErrorInitializationError", CONV_TYPE, API_RUNTIME),
        ),
        ("CUDA_ERROR_DEINITIALIZED", ("hipErrorDeinitialized", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorCudartUnloading",
            ("hipErrorDeinitialized", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_DISABLED",
            ("hipErrorProfilerDisabled", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerDisabled",
            ("hipErrorProfilerDisabled", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
            ("hipErrorProfilerNotInitialized", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerNotInitialized",
            ("hipErrorProfilerNotInitialized", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_ALREADY_STARTED",
            ("hipErrorProfilerAlreadyStarted", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerAlreadyStarted",
            ("hipErrorProfilerAlreadyStarted", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
            ("hipErrorProfilerAlreadyStopped", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerAlreadyStopped",
            ("hipErrorProfilerAlreadyStopped", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_NO_DEVICE", ("hipErrorNoDevice", CONV_TYPE, API_DRIVER)),
        ("cudaErrorNoDevice", ("hipErrorNoDevice", CONV_TYPE, API_RUNTIME)),
        ("CUDA_ERROR_INVALID_DEVICE", ("hipErrorInvalidDevice", CONV_TYPE, API_DRIVER)),
        ("cudaErrorInvalidDevice", ("hipErrorInvalidDevice", CONV_TYPE, API_RUNTIME)),
        ("CUDA_ERROR_INVALID_IMAGE", ("hipErrorInvalidImage", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorInvalidKernelImage",
            ("hipErrorInvalidImage", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_MAP_FAILED", ("hipErrorMapFailed", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorMapBufferObjectFailed",
            ("hipErrorMapFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_UNMAP_FAILED", ("hipErrorUnmapFailed", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorUnmapBufferObjectFailed",
            ("hipErrorUnmapFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NO_BINARY_FOR_GPU",
            ("hipErrorNoBinaryForGpu", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorNoKernelImageForDevice",
            ("hipErrorNoBinaryForGpu", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_ECC_UNCORRECTABLE",
            ("hipErrorECCNotCorrectable", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorECCUncorrectable",
            ("hipErrorECCNotCorrectable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_UNSUPPORTED_LIMIT",
            ("hipErrorUnsupportedLimit", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorUnsupportedLimit",
            ("hipErrorUnsupportedLimit", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
            ("hipErrorPeerAccessUnsupported", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessUnsupported",
            ("hipErrorPeerAccessUnsupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_PTX",
            ("hipErrorInvalidKernelFile", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInvalidPtx",
            ("hipErrorInvalidKernelFile", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
            ("hipErrorInvalidGraphicsContext", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInvalidGraphicsContext",
            ("hipErrorInvalidGraphicsContext", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NVLINK_UNCORRECTABLE",
            ("hipErrorNvlinkUncorrectable", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNvlinkUncorrectable",
            ("hipErrorNvlinkUncorrectable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
            ("hipErrorSharedObjectSymbolNotFound", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorSharedObjectSymbolNotFound",
            (
                "hipErrorSharedObjectSymbolNotFound",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
            ("hipErrorSharedObjectInitFailed", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorSharedObjectInitFailed",
            ("hipErrorSharedObjectInitFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_OPERATING_SYSTEM",
            ("hipErrorOperatingSystem", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorOperatingSystem",
            ("hipErrorOperatingSystem", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_HANDLE",
            ("hipErrorInvalidResourceHandle", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInvalidResourceHandle",
            ("hipErrorInvalidResourceHandle", CONV_TYPE, API_RUNTIME),
        ),
        ("CUDA_ERROR_NOT_READY", ("hipErrorNotReady", CONV_TYPE, API_DRIVER)),
        ("cudaErrorNotReady", ("hipErrorNotReady", CONV_TYPE, API_RUNTIME)),
        (
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            ("hipErrorIllegalAddress", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorIllegalAddress",
            ("hipErrorIllegalAddress", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            ("hipErrorLaunchOutOfResources", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorLaunchOutOfResources",
            ("hipErrorLaunchOutOfResources", CONV_TYPE, API_RUNTIME),
        ),
        ("CUDA_ERROR_LAUNCH_TIMEOUT", ("hipErrorLaunchTimeOut", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorLaunchTimeout",
            ("hipErrorLaunchTimeOut", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
            ("hipErrorPeerAccessAlreadyEnabled", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessAlreadyEnabled",
            ("hipErrorPeerAccessAlreadyEnabled", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
            ("hipErrorPeerAccessNotEnabled", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessNotEnabled",
            ("hipErrorPeerAccessNotEnabled", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_ASSERT",
            ("hipErrorAssert", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorAssert",
            ("hipErrorAssert", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_TOO_MANY_PEERS",
            ("hipErrorTooManyPeers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTooManyPeers",
            ("hipErrorTooManyPeers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
            ("hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorHostMemoryAlreadyRegistered",
            ("hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
            ("hipErrorHostMemoryNotRegistered", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorHostMemoryNotRegistered",
            ("hipErrorHostMemoryNotRegistered", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_HARDWARE_STACK_ERROR",
            ("hipErrorHardwareStackError", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorHardwareStackError",
            ("hipErrorHardwareStackError", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            ("hipErrorIllegalInstruction", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorIllegalInstruction",
            ("hipErrorIllegalInstruction", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_MISALIGNED_ADDRESS",
            ("hipErrorMisalignedAddress", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMisalignedAddress",
            ("hipErrorMisalignedAddress", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_ADDRESS_SPACE",
            ("hipErrorInvalidAddressSpace", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidAddressSpace",
            ("hipErrorInvalidAddressSpace", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_PC",
            ("hipErrorInvalidPc", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidPc",
            ("hipErrorInvalidPc", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_LAUNCH_FAILED",
            ("hipErrorLaunchFailure", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFailure",
            ("hipErrorLaunchFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_UNKNOWN",
            ("hipErrorUnknown", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cudaErrorUnknown", ("hipErrorUnknown", CONV_TYPE, API_RUNTIME)),
        (
            "CU_TR_ADDRESS_MODE_WRAP",
            ("HIP_TR_ADDRESS_MODE_WRAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_CLAMP",
            ("HIP_TR_ADDRESS_MODE_CLAMP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_MIRROR",
            ("HIP_TR_ADDRESS_MODE_MIRROR", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_BORDER",
            ("HIP_TR_ADDRESS_MODE_BORDER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_X",
            ("HIP_CUBEMAP_FACE_POSITIVE_X", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_X",
            ("HIP_CUBEMAP_FACE_NEGATIVE_X", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_Y",
            ("HIP_CUBEMAP_FACE_POSITIVE_Y", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_Y",
            ("HIP_CUBEMAP_FACE_NEGATIVE_Y", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_Z",
            ("HIP_CUBEMAP_FACE_POSITIVE_Z", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_Z",
            ("HIP_CUBEMAP_FACE_NEGATIVE_Z", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT8",
            ("HIP_AD_FORMAT_UNSIGNED_INT8", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT16",
            ("HIP_AD_FORMAT_UNSIGNED_INT16", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT32",
            ("HIP_AD_FORMAT_UNSIGNED_INT32", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT8",
            ("HIP_AD_FORMAT_SIGNED_INT8", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT16",
            ("HIP_AD_FORMAT_SIGNED_INT16", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT32",
            ("HIP_AD_FORMAT_SIGNED_INT32", CONV_TYPE, API_DRIVER),
        ),
        ("CU_AD_FORMAT_HALF", ("HIP_AD_FORMAT_HALF", CONV_TYPE, API_DRIVER)),
        ("CU_AD_FORMAT_FLOAT", ("HIP_AD_FORMAT_FLOAT", CONV_TYPE, API_DRIVER)),
        (
            "CU_COMPUTEMODE_DEFAULT",
            ("hipComputeModeDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_EXCLUSIVE",
            ("hipComputeModeExclusive", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_PROHIBITED",
            ("hipComputeModeProhibited", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_EXCLUSIVE_PROCESS",
            ("hipComputeModeExclusiveProcess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_SET_READ_MOSTLY",
            ("hipMemAdviseSetReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_UNSET_READ_MOSTLY",
            ("hipMemAdviseUnsetReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
            (
                "hipMemAdviseSetPreferredLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
            (
                "hipMemAdviseUnsetPreferredLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_ADVISE_SET_ACCESSED_BY",
            ("hipMemAdviseSetAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_UNSET_ACCESSED_BY",
            ("hipMemAdviseUnsetAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
            ("hipMemRangeAttributeReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",
            (
                "hipMemRangeAttributePreferredLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",
            ("hipMemRangeAttributeAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION",
            (
                "hipMemRangeAttributeLastPrefetchLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_CTX_SCHED_AUTO",
            ("HIP_CTX_SCHED_AUTO", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_SPIN",
            ("HIP_CTX_SCHED_SPIN", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_YIELD",
            ("HIP_CTX_SCHED_YIELD", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_BLOCKING_SYNC",
            ("HIP_CTX_SCHED_BLOCKING_SYNC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_BLOCKING_SYNC",
            ("HIP_CTX_BLOCKING_SYNC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_MASK",
            ("HIP_CTX_SCHED_MASK", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_MAP_HOST",
            ("HIP_CTX_MAP_HOST", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_LMEM_RESIZE_TO_MAX",
            ("HIP_CTX_LMEM_RESIZE_TO_MAX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_FLAGS_MASK",
            ("HIP_CTX_FLAGS_MASK", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LAUNCH_PARAM_BUFFER_POINTER",
            ("HIP_LAUNCH_PARAM_BUFFER_POINTER", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_LAUNCH_PARAM_BUFFER_SIZE",
            ("HIP_LAUNCH_PARAM_BUFFER_SIZE", CONV_TYPE, API_DRIVER),
        ),
        ("CU_LAUNCH_PARAM_END", ("HIP_LAUNCH_PARAM_END", CONV_TYPE, API_DRIVER)),
        (
            "CU_IPC_HANDLE_SIZE",
            ("HIP_IPC_HANDLE_SIZE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_DEVICEMAP",
            ("HIP_MEMHOSTALLOC_DEVICEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_PORTABLE",
            ("HIP_MEMHOSTALLOC_PORTABLE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_WRITECOMBINED",
            ("HIP_MEMHOSTALLOC_WRITECOMBINED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_DEVICEMAP",
            ("HIP_MEMHOSTREGISTER_DEVICEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_IOMEMORY",
            ("HIP_MEMHOSTREGISTER_IOMEMORY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_PORTABLE",
            ("HIP_MEMHOSTREGISTER_PORTABLE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_PARAM_TR_DEFAULT",
            ("HIP_PARAM_TR_DEFAULT", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_LEGACY",
            ("HIP_STREAM_LEGACY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_PER_THREAD",
            ("HIP_STREAM_PER_THREAD", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSA_OVERRIDE_FORMAT",
            ("HIP_TRSA_OVERRIDE_FORMAT", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSF_NORMALIZED_COORDINATES",
            ("HIP_TRSF_NORMALIZED_COORDINATES", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSF_READ_AS_INTEGER",
            ("HIP_TRSF_READ_AS_INTEGER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CU_TRSF_SRGB", ("HIP_TRSF_SRGB", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUDA_ARRAY3D_2DARRAY",
            ("HIP_ARRAY3D_LAYERED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_CUBEMAP",
            ("HIP_ARRAY3D_CUBEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_DEPTH_TEXTURE",
            ("HIP_ARRAY3D_DEPTH_TEXTURE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_LAYERED",
            ("HIP_ARRAY3D_LAYERED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_SURFACE_LDST",
            ("HIP_ARRAY3D_SURFACE_LDST", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_TEXTURE_GATHER",
            ("HIP_ARRAY3D_TEXTURE_GATHER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxThreadsPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
            ("hipDeviceAttributeMaxBlockDimX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
            ("hipDeviceAttributeMaxBlockDimY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
            ("hipDeviceAttributeMaxBlockDimZ", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
            ("hipDeviceAttributeMaxGridDimX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",
            ("hipDeviceAttributeMaxGridDimY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",
            ("hipDeviceAttributeMaxGridDimZ", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
            (
                "hipDeviceAttributeMaxSharedMemoryPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK",
            (
                "hipDeviceAttributeMaxSharedMemoryPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",
            (
                "hipDeviceAttributeTotalConstantMemory",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_WARP_SIZE",
            ("hipDeviceAttributeWarpSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_PITCH",
            ("hipDeviceAttributeMaxPitch", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxRegistersPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxRegistersPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CLOCK_RATE",
            ("hipDeviceAttributeClockRate", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",
            (
                "hipDeviceAttributeTextureAlignment",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP",
            (
                "hipDeviceAttributeAsyncEngineCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
            (
                "hipDeviceAttributeMultiprocessorCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",
            (
                "hipDeviceAttributeKernelExecTimeout",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_INTEGRATED",
            ("hipDeviceAttributeIntegrated", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",
            (
                "hipDeviceAttributeCanMapHostMemory",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE",
            ("hipDeviceAttributeComputeMode", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture3DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture3DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH",
            (
                "hipDeviceAttributeMaxTexture3DDepth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",
            (
                "hipDeviceAttributeSurfaceAlignment",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",
            ("hipDeviceAttributeConcurrentKernels", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_ECC_ENABLED",
            ("hipDeviceAttributeEccEnabled", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",
            ("hipDeviceAttributePciBusId", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
            ("hipDeviceAttributePciDeviceId", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TCC_DRIVER",
            ("hipDeviceAttributeTccDriver", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE",
            (
                "hipDeviceAttributeMemoryClockRate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",
            ("hipDeviceAttributeMemoryBusWidth", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",
            ("hipDeviceAttributeL2CacheSize", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
            ("hipDeviceAttributeMaxThreadsPerMultiProcessor", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",
            (
                "hipDeviceAttributeAsyncEngineCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",
            (
                "hipDeviceAttributeUnifiedAddressing",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTexture1DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER",
            (
                "hipDeviceAttributeCanTex2DGather",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DGatherWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DGatherHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DWidthAlternate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DHeightAlternate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DDepthAlternate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
            ("hipDeviceAttributePciDomainId", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",
            (
                "hipDeviceAttributeTexturePitchAlignment",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH",
            (
                "hipDeviceAttributeMaxTextureCubemapWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface1DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface2DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface2DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface3DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface3DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH",
            (
                "hipDeviceAttributeMaxSurface3DDepth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurface1DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurface1DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurface2DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface2DLayeredHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurface2DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH",
            (
                "hipDeviceAttributeMaxSurfaceCubemapWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DLinearWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLinearWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLinearHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH",
            (
                "hipDeviceAttributeMaxTexture2DLinearPitch",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
            ("hipDeviceAttributeComputeCapabilityMajor", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
            ("hipDeviceAttributeComputeCapabilityMinor", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DMipmappedWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",
            (
                "hipDeviceAttributeStreamPrioritiesSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",
            (
                "hipDeviceAttributeGlobalL1CacheSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED",
            (
                "hipDeviceAttributeLocalL1CacheSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
            (
                "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
                CONV_TYPE,
                API_DRIVER,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",
            (
                "hipDeviceAttributeMaxRegistersPerMultiprocessor",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY",
            ("hipDeviceAttributeManagedMemory", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",
            ("hipDeviceAttributeIsMultiGpuBoard", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID",
            (
                "hipDeviceAttributeMultiGpuBoardGroupId",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
            (
                "hipDeviceAttributeHostNativeAtomicSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
            (
                "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
            (
                "hipDeviceAttributePageableMemoryAccess",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
            (
                "hipDeviceAttributeConcurrentManagedAccess",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED",
            (
                "hipDeviceAttributeComputePreemptionSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
            (
                "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX",
            ("hipDeviceAttributeMax", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_CONTEXT",
            ("hipPointerAttributeContext", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_MEMORY_TYPE",
            ("hipPointerAttributeMemoryType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_DEVICE_POINTER",
            (
                "hipPointerAttributeDevicePointer",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_POINTER_ATTRIBUTE_HOST_POINTER",
            ("hipPointerAttributeHostPointer", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_P2P_TOKENS",
            ("hipPointerAttributeP2pTokens", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS",
            ("hipPointerAttributeSyncMemops", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_BUFFER_ID",
            ("hipPointerAttributeBufferId", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_IS_MANAGED",
            ("hipPointerAttributeIsManaged", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
            (
                "hipFuncAttributeMaxThreadsPerBlocks",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
            ("hipFuncAttributeSharedSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
            ("hipFuncAttributeMaxDynamicSharedMemorySize", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
            ("hipFuncAttributeConstSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
            ("hipFuncAttributeLocalSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_NUM_REGS",
            ("hipFuncAttributeNumRegs", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_PTX_VERSION",
            ("hipFuncAttributePtxVersion", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_BINARY_VERSION",
            ("hipFuncAttributeBinaryVersion", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA",
            ("hipFuncAttributeCacheModeCA", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX",
            ("hipFuncAttributeMax", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE",
            ("hipGraphicsMapFlagsNone", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY",
            ("hipGraphicsMapFlagsReadOnly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
            ("hipGraphicsMapFlagsWriteDiscard", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_NONE",
            ("hipGraphicsRegisterFlagsNone", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY",
            (
                "hipGraphicsRegisterFlagsReadOnly",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD",
            (
                "hipGraphicsRegisterFlagsWriteDiscard",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST",
            (
                "hipGraphicsRegisterFlagsSurfaceLoadStore",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER",
            (
                "hipGraphicsRegisterFlagsTextureGather",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_OCCUPANCY_DEFAULT",
            ("hipOccupancyDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE",
            (
                "hipOccupancyDisableCachingOverride",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_FUNC_CACHE_PREFER_NONE",
            ("hipFuncCachePreferNone", CONV_CACHE, API_DRIVER),
        ),
        (
            "CU_FUNC_CACHE_PREFER_SHARED",
            ("hipFuncCachePreferShared", CONV_CACHE, API_DRIVER),
        ),
        ("CU_FUNC_CACHE_PREFER_L1", ("hipFuncCachePreferL1", CONV_CACHE, API_DRIVER)),
        (
            "CU_FUNC_CACHE_PREFER_EQUAL",
            ("hipFuncCachePreferEqual", CONV_CACHE, API_DRIVER),
        ),
        (
            "CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS",
            ("hipIpcMemLazyEnablePeerAccess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUDA_IPC_HANDLE_SIZE", ("HIP_IPC_HANDLE_SIZE", CONV_TYPE, API_DRIVER)),
        (
            "CU_JIT_CACHE_OPTION_NONE",
            ("hipJitCacheModeOptionNone", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_CACHE_OPTION_CG",
            ("hipJitCacheModeOptionCG", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_CACHE_OPTION_CA",
            ("hipJitCacheModeOptionCA", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_PREFER_PTX",
            ("hipJitFallbackPreferPtx", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_PREFER_BINARY",
            ("hipJitFallbackPreferBinary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CU_JIT_MAX_REGISTERS", ("hipJitOptionMaxRegisters", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_THREADS_PER_BLOCK",
            ("hipJitOptionThreadsPerBlock", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_WALL_TIME", ("hipJitOptionWallTime", CONV_JIT, API_DRIVER)),
        ("CU_JIT_INFO_LOG_BUFFER", ("hipJitOptionInfoLogBuffer", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES",
            ("hipJitOptionInfoLogBufferSizeBytes", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_ERROR_LOG_BUFFER",
            ("hipJitOptionErrorLogBuffer", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES",
            ("hipJitOptionErrorLogBufferSizeBytes", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_OPTIMIZATION_LEVEL",
            ("hipJitOptionOptimizationLevel", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_TARGET_FROM_CUCONTEXT",
            ("hipJitOptionTargetFromContext", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_TARGET", ("hipJitOptionTarget", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_FALLBACK_STRATEGY",
            ("hipJitOptionFallbackStrategy", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_GENERATE_DEBUG_INFO",
            ("hipJitOptionGenerateDebugInfo", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_LOG_VERBOSE", ("hipJitOptionLogVerbose", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_GENERATE_LINE_INFO",
            ("hipJitOptionGenerateLineInfo", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_CACHE_MODE", ("hipJitOptionCacheMode", CONV_JIT, API_DRIVER)),
        ("CU_JIT_NEW_SM3X_OPT", ("hipJitOptionSm3xOpt", CONV_JIT, API_DRIVER)),
        ("CU_JIT_FAST_COMPILE", ("hipJitOptionFastCompile", CONV_JIT, API_DRIVER)),
        ("CU_JIT_NUM_OPTIONS", ("hipJitOptionNumOptions", CONV_JIT, API_DRIVER)),
        (
            "CU_TARGET_COMPUTE_10",
            ("hipJitTargetCompute10", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_11",
            ("hipJitTargetCompute11", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_12",
            ("hipJitTargetCompute12", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_13",
            ("hipJitTargetCompute13", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_20",
            ("hipJitTargetCompute20", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_21",
            ("hipJitTargetCompute21", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_30",
            ("hipJitTargetCompute30", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_32",
            ("hipJitTargetCompute32", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_35",
            ("hipJitTargetCompute35", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_37",
            ("hipJitTargetCompute37", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_50",
            ("hipJitTargetCompute50", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_52",
            ("hipJitTargetCompute52", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_53",
            ("hipJitTargetCompute53", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_60",
            ("hipJitTargetCompute60", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_61",
            ("hipJitTargetCompute61", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_62",
            ("hipJitTargetCompute62", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_CUBIN",
            ("hipJitInputTypeBin", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_PTX",
            ("hipJitInputTypePtx", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_FATBINARY",
            ("hipJitInputTypeFatBinary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_OBJECT",
            ("hipJitInputTypeObject", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_LIBRARY",
            ("hipJitInputTypeLibrary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_NUM_INPUT_TYPES",
            ("hipJitInputTypeNumInputTypes", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_STACK_SIZE",
            ("hipLimitStackSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_PRINTF_FIFO_SIZE",
            ("hipLimitPrintfFifoSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_MALLOC_HEAP_SIZE",
            ("hipLimitMallocHeapSize", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH",
            ("hipLimitDevRuntimeSyncDepth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT",
            (
                "hipLimitDevRuntimePendingLaunchCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_LIMIT_STACK_SIZE",
            ("hipLimitStackSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_GLOBAL",
            ("hipMemAttachGlobal", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_HOST",
            ("hipMemAttachHost", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_SINGLE",
            ("hipMemAttachSingle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_HOST",
            ("hipMemTypeHost", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_DEVICE",
            ("hipMemTypeDevice", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_ARRAY",
            ("hipMemTypeArray", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_UNIFIED",
            ("hipMemTypeUnified", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_ARRAY",
            ("hipResourceTypeArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY",
            ("hipResourceTypeMipmappedArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_LINEAR",
            ("hipResourceTypeLinear", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_PITCH2D",
            ("hipResourceTypePitch2D", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CU_RES_VIEW_FORMAT_NONE", ("hipResViewFormatNone", CONV_TEX, API_DRIVER)),
        (
            "CU_RES_VIEW_FORMAT_UINT_1X8",
            ("hipResViewFormatUnsignedChar1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_2X8",
            ("hipResViewFormatUnsignedChar2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_4X8",
            ("hipResViewFormatUnsignedChar4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_1X8",
            ("hipResViewFormatSignedChar1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_2X8",
            ("hipResViewFormatSignedChar2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_4X8",
            ("hipResViewFormatSignedChar4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_1X16",
            ("hipResViewFormatUnsignedShort1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_2X16",
            ("hipResViewFormatUnsignedShort2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_4X16",
            ("hipResViewFormatUnsignedShort4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_1X16",
            ("hipResViewFormatSignedShort1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_2X16",
            ("hipResViewFormatSignedShort2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_4X16",
            ("hipResViewFormatSignedShort4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_1X32",
            ("hipResViewFormatUnsignedInt1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_2X32",
            ("hipResViewFormatUnsignedInt2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_4X32",
            ("hipResViewFormatUnsignedInt4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_1X32",
            ("hipResViewFormatSignedInt1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_2X32",
            ("hipResViewFormatSignedInt2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_4X32",
            ("hipResViewFormatSignedInt4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_1X16",
            ("hipResViewFormatHalf1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_2X16",
            ("hipResViewFormatHalf2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_4X16",
            ("hipResViewFormatHalf4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_1X32",
            ("hipResViewFormatFloat1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_2X32",
            ("hipResViewFormatFloat2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_4X32",
            ("hipResViewFormatFloat4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC1",
            ("hipResViewFormatUnsignedBlockCompressed1", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC2",
            ("hipResViewFormatUnsignedBlockCompressed2", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC3",
            ("hipResViewFormatUnsignedBlockCompressed3", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC4",
            ("hipResViewFormatUnsignedBlockCompressed4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SIGNED_BC4",
            ("hipResViewFormatSignedBlockCompressed4", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC5",
            ("hipResViewFormatUnsignedBlockCompressed5", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SIGNED_BC5",
            ("hipResViewFormatSignedBlockCompressed5", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC6H",
            ("hipResViewFormatUnsignedBlockCompressed6H", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SIGNED_BC6H",
            ("hipResViewFormatSignedBlockCompressed6H", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC7",
            ("hipResViewFormatUnsignedBlockCompressed7", CONV_TEX, API_DRIVER),
        ),
        (
            "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE",
            ("hipSharedMemBankSizeDefault", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE",
            ("hipSharedMemBankSizeFourByte", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE",
            ("hipSharedMemBankSizeEightByte", CONV_TYPE, API_DRIVER),
        ),
        ("CU_STREAM_DEFAULT", ("hipStreamDefault", CONV_TYPE, API_DRIVER)),
        ("CU_STREAM_NON_BLOCKING", ("hipStreamNonBlocking", CONV_TYPE, API_DRIVER)),
        (
            "CU_STREAM_WAIT_VALUE_GEQ",
            ("hipStreamWaitValueGeq", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WAIT_VALUE_EQ",
            ("hipStreamWaitValueEq", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WAIT_VALUE_AND",
            ("hipStreamWaitValueAnd", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WAIT_VALUE_FLUSH",
            ("hipStreamWaitValueFlush", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WRITE_VALUE_DEFAULT",
            ("hipStreamWriteValueDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER",
            (
                "hipStreamWriteValueNoMemoryBarrier",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_STREAM_MEM_OP_WAIT_VALUE_32",
            ("hipStreamBatchMemOpWaitValue32", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_MEM_OP_WRITE_VALUE_32",
            ("hipStreamBatchMemOpWriteValue32", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES",
            (
                "hipStreamBatchMemOpFlushRemoteWrites",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGetErrorName",
            ("hipGetErrorName", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGetErrorString",
            ("hipDrvGetErrorString", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuInit", ("hipInit", CONV_INIT, API_DRIVER)),
        ("cuDriverGetVersion", ("hipDriverGetVersion", CONV_VERSION, API_DRIVER)),
        ("cuCtxCreate", ("hipCtxCreate", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxCreate_v2", ("hipCtxCreate", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxDestroy", ("hipCtxDestroy", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxDestroy_v2", ("hipCtxDestroy", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxGetApiVersion", ("hipCtxGetApiVersion", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxGetCacheConfig", ("hipCtxGetCacheConfig", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxGetCurrent", ("hipCtxGetCurrent", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxGetDevice", ("hipCtxGetDevice", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxGetFlags", ("hipCtxGetFlags", CONV_CONTEXT, API_DRIVER)),
        ("cuDeviceGetUuid", ("hipDeviceGetUuid", CONV_CONTEXT, API_DRIVER)),
        (
            "cuCtxGetLimit",
            ("hipCtxGetLimit", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuCtxGetSharedMemConfig",
            ("hipCtxGetSharedMemConfig", CONV_CONTEXT, API_DRIVER),
        ),
        (
            "cuCtxGetStreamPriorityRange",
            ("hipCtxGetStreamPriorityRange", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuCtxPopCurrent_v2", ("hipCtxPopCurrent", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxPushCurrent_v2", ("hipCtxPushCurrent", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxSetCacheConfig", ("hipCtxSetCacheConfig", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxSetCurrent", ("hipCtxSetCurrent", CONV_CONTEXT, API_DRIVER)),
        (
            "cuCtxSetLimit",
            ("hipCtxSetLimit", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuCtxSetSharedMemConfig",
            ("hipCtxSetSharedMemConfig", CONV_CONTEXT, API_DRIVER),
        ),
        ("cuCtxSynchronize", ("hipCtxSynchronize", CONV_CONTEXT, API_DRIVER)),
        ("cuCtxAttach", ("hipCtxAttach", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuCtxDetach", ("hipCtxDetach", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuCtxEnablePeerAccess", ("hipCtxEnablePeerAccess", CONV_PEER, API_DRIVER)),
        ("cuCtxDisablePeerAccess", ("hipCtxDisablePeerAccess", CONV_PEER, API_DRIVER)),
        ("cuDeviceCanAccessPeer", ("hipDeviceCanAccessPeer", CONV_PEER, API_DRIVER)),
        (
            "cuDeviceGetP2PAttribute",
            ("hipDeviceGetP2PAttribute", CONV_PEER, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuDevicePrimaryCtxGetState",
            ("hipDevicePrimaryCtxGetState", CONV_CONTEXT, API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxRelease",
            ("hipDevicePrimaryCtxRelease", CONV_CONTEXT, API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxReset",
            ("hipDevicePrimaryCtxReset", CONV_CONTEXT, API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxRetain",
            ("hipDevicePrimaryCtxRetain", CONV_CONTEXT, API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxSetFlags",
            ("hipDevicePrimaryCtxSetFlags", CONV_CONTEXT, API_DRIVER),
        ),
        ("cuDeviceGet", ("hipDeviceGet", CONV_DEVICE, API_DRIVER)),
        ("cuDeviceGetName", ("hipDeviceGetName", CONV_DEVICE, API_DRIVER)),
        ("cuDeviceGetCount", ("hipGetDeviceCount", CONV_DEVICE, API_DRIVER)),
        ("cuDeviceGetAttribute", ("hipDeviceGetAttribute", CONV_DEVICE, API_DRIVER)),
        ("cuDeviceGetPCIBusId", ("hipDeviceGetPCIBusId", CONV_DEVICE, API_DRIVER)),
        ("cuDeviceGetByPCIBusId", ("hipDeviceGetByPCIBusId", CONV_DEVICE, API_DRIVER)),
        ("cuDeviceTotalMem_v2", ("hipDeviceTotalMem", CONV_DEVICE, API_DRIVER)),
        (
            "cuDeviceComputeCapability",
            ("hipDeviceComputeCapability", CONV_DEVICE, API_DRIVER),
        ),
        ("cuDeviceGetProperties", ("hipGetDeviceProperties", CONV_DEVICE, API_DRIVER)),
        ("cuLinkAddData", ("hipLinkAddData", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuLinkAddFile", ("hipLinkAddFile", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuLinkComplete",
            ("hipLinkComplete", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuLinkCreate", ("hipLinkCreate", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuLinkDestroy", ("hipLinkDestroy", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuModuleGetFunction", ("hipModuleGetFunction", CONV_MODULE, API_DRIVER)),
        ("cuModuleGetGlobal_v2", ("hipModuleGetGlobal", CONV_MODULE, API_DRIVER)),
        (
            "cuModuleGetSurfRef",
            ("hipModuleGetSurfRef", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuModuleGetTexRef", ("hipModuleGetTexRef", CONV_MODULE, API_DRIVER)),
        ("cuModuleLoad", ("hipModuleLoad", CONV_MODULE, API_DRIVER)),
        ("cuModuleLoadData", ("hipModuleLoadData", CONV_MODULE, API_DRIVER)),
        ("cuModuleLoadDataEx", ("hipModuleLoadDataEx", CONV_MODULE, API_DRIVER)),
        (
            "cuModuleLoadFatBinary",
            ("hipModuleLoadFatBinary", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuModuleUnload", ("hipModuleUnload", CONV_MODULE, API_DRIVER)),
        (
            "CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK",
            (
                "hipDeviceP2PAttributePerformanceRank",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED",
            (
                "hipDeviceP2PAttributeAccessSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED",
            (
                "hipDeviceP2PAttributeNativeAtomicSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        ("CU_EVENT_DEFAULT", ("hipEventDefault", CONV_EVENT, API_DRIVER)),
        ("CU_EVENT_BLOCKING_SYNC", ("hipEventBlockingSync", CONV_EVENT, API_DRIVER)),
        ("CU_EVENT_DISABLE_TIMING", ("hipEventDisableTiming", CONV_EVENT, API_DRIVER)),
        ("CU_EVENT_INTERPROCESS", ("hipEventInterprocess", CONV_EVENT, API_DRIVER)),
        ("cuEventCreate", ("hipEventCreate", CONV_EVENT, API_DRIVER)),
        ("cuEventDestroy", ("hipEventDestroy", CONV_EVENT, API_DRIVER)),
        ("cuEventDestroy_v2", ("hipEventDestroy", CONV_EVENT, API_DRIVER)),
        ("cuEventElapsedTime", ("hipEventElapsedTime", CONV_EVENT, API_DRIVER)),
        ("cuEventQuery", ("hipEventQuery", CONV_EVENT, API_DRIVER)),
        ("cuEventRecord", ("hipEventRecord", CONV_EVENT, API_DRIVER)),
        ("cuEventSynchronize", ("hipEventSynchronize", CONV_EVENT, API_DRIVER)),
        ("cuFuncSetAttribute", ("hipFuncSetAttribute", CONV_EVENT, API_DRIVER)),
        (
            "cuFuncGetAttribute",
            ("hipFuncGetAttribute", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuFuncSetCacheConfig", ("hipFuncSetCacheConfig", CONV_MODULE, API_DRIVER)),
        (
            "cuFuncSetSharedMemConfig",
            ("hipFuncSetSharedMemConfig", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuLaunchKernel", ("hipModuleLaunchKernel", CONV_MODULE, API_DRIVER)),
        (
            "cuFuncSetBlockShape",
            ("hipFuncSetBlockShape", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuFuncSetSharedSize",
            ("hipFuncSetSharedSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuLaunch", ("hipLaunch", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuLaunchGrid", ("hipLaunchGrid", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuLaunchGridAsync",
            ("hipLaunchGridAsync", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuParamSetf", ("hipParamSetf", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuParamSeti", ("hipParamSeti", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuParamSetSize",
            ("hipParamSetSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuParamSetSize",
            ("hipParamSetSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuParamSetv", ("hipParamSetv", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            (
                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                CONV_OCCUPANCY,
                API_DRIVER,
            ),
        ),
        (
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            (
                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                CONV_OCCUPANCY,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuOccupancyMaxPotentialBlockSize",
            ("hipModuleOccupancyMaxPotentialBlockSize", CONV_OCCUPANCY, API_DRIVER),
        ),
        (
            "cuOccupancyMaxPotentialBlockSizeWithFlags",
            (
                "hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
                CONV_OCCUPANCY,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cuStreamAddCallback", ("hipStreamAddCallback", CONV_STREAM, API_DRIVER)),
        (
            "cuStreamAttachMemAsync",
            ("hipStreamAttachMemAsync", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuStreamCreate",
            ("hipStreamCreate__", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuStreamCreateWithPriority",
            ("hipStreamCreateWithPriority", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuStreamDestroy", ("hipStreamDestroy", CONV_STREAM, API_DRIVER)),
        ("cuStreamDestroy_v2", ("hipStreamDestroy", CONV_STREAM, API_DRIVER)),
        ("cuStreamGetFlags", ("hipStreamGetFlags", CONV_STREAM, API_DRIVER)),
        (
            "cuStreamGetPriority",
            ("hipStreamGetPriority", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuStreamQuery", ("hipStreamQuery", CONV_STREAM, API_DRIVER)),
        ("cuStreamSynchronize", ("hipStreamSynchronize", CONV_STREAM, API_DRIVER)),
        ("cuStreamWaitEvent", ("hipStreamWaitEvent", CONV_STREAM, API_DRIVER)),
        (
            "cuStreamWaitValue32",
            ("hipStreamWaitValue32", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuStreamWriteValue32",
            ("hipStreamWriteValue32", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuStreamBatchMemOp",
            ("hipStreamBatchMemOp", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuArray3DCreate", ("hipArray3DCreate", CONV_MEM, API_DRIVER)),
        (
            "cuArray3DGetDescriptor",
            ("hipArray3DGetDescriptor", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuArrayCreate", ("hipArrayCreate", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuArrayDestroy", ("hipArrayDestroy", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuArrayGetDescriptor",
            ("hipArrayGetDescriptor", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuIpcCloseMemHandle",
            ("hipIpcCloseMemHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuIpcGetEventHandle",
            ("hipIpcGetEventHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuIpcGetMemHandle",
            ("hipIpcGetMemHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuIpcOpenEventHandle",
            ("hipIpcOpenEventHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuIpcOpenMemHandle",
            ("hipIpcOpenMemHandle", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemAlloc_v2", ("hipMalloc", CONV_MEM, API_DRIVER)),
        ("cuMemAllocHost", ("hipMemAllocHost", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemAllocManaged",
            ("hipMemAllocManaged", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemAllocPitch",
            ("hipMemAllocPitch__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemcpy", ("hipMemcpy__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuMemcpy2D", ("hipMemcpy2D__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemcpy2DAsync",
            ("hipMemcpy2DAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemcpy2DUnaligned",
            ("hipMemcpy2DUnaligned", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemcpy3D", ("hipMemcpy3D__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemcpy3DAsync",
            ("hipMemcpy3DAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemcpy3DPeer",
            ("hipMemcpy3DPeer__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemcpy3DPeerAsync",
            ("hipMemcpy3DPeerAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemcpyAsync", ("hipMemcpyAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuMemcpyAtoA", ("hipMemcpyAtoA", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuMemcpyAtoD", ("hipMemcpyAtoD", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuMemcpyAtoH", ("hipMemcpyAtoH", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemcpyAtoHAsync",
            ("hipMemcpyAtoHAsync", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemcpyDtoA", ("hipMemcpyDtoA", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuMemcpyDtoD_v2", ("hipMemcpyDtoD", CONV_MEM, API_DRIVER)),
        ("cuMemcpyDtoDAsync_v2", ("hipMemcpyDtoDAsync", CONV_MEM, API_DRIVER)),
        ("cuMemcpyDtoH_v2", ("hipMemcpyDtoH", CONV_MEM, API_DRIVER)),
        ("cuMemcpyDtoHAsync_v2", ("hipMemcpyDtoHAsync", CONV_MEM, API_DRIVER)),
        ("cuMemcpyHtoA", ("hipMemcpyHtoA", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemcpyHtoAAsync",
            ("hipMemcpyHtoAAsync", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemcpyHtoD_v2", ("hipMemcpyHtoD", CONV_MEM, API_DRIVER)),
        ("cuMemcpyHtoDAsync_v2", ("hipMemcpyHtoDAsync", CONV_MEM, API_DRIVER)),
        (
            "cuMemcpyPeerAsync",
            ("hipMemcpyPeerAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemcpyPeer", ("hipMemcpyPeer__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuMemFree", ("hipFree", CONV_MEM, API_DRIVER)),
        ("cuMemFree_v2", ("hipFree", CONV_MEM, API_DRIVER)),
        ("cuMemFreeHost", ("hipHostFree", CONV_MEM, API_DRIVER)),
        (
            "cuMemGetAddressRange",
            ("hipMemGetAddressRange", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemGetInfo_v2", ("hipMemGetInfo", CONV_MEM, API_DRIVER)),
        ("cuMemHostAlloc", ("hipHostMalloc", CONV_MEM, API_DRIVER)),
        (
            "cuMemHostGetDevicePointer",
            ("hipMemHostGetDevicePointer", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemHostGetFlags",
            ("hipMemHostGetFlags", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemHostRegister_v2", ("hipHostRegister", CONV_MEM, API_DRIVER)),
        ("cuMemHostUnregister", ("hipHostUnregister", CONV_MEM, API_DRIVER)),
        ("cuMemsetD16_v2", ("hipMemsetD16", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemsetD16Async",
            ("hipMemsetD16Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemsetD2D16_v2", ("hipMemsetD2D16", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemsetD2D16Async",
            ("hipMemsetD2D16Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemsetD2D32_v2", ("hipMemsetD2D32", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemsetD2D32Async",
            ("hipMemsetD2D32Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemsetD2D8_v2", ("hipMemsetD2D8", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemsetD2D8Async",
            ("hipMemsetD2D8Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemsetD32_v2", ("hipMemset", CONV_MEM, API_DRIVER)),
        ("cuMemsetD32Async", ("hipMemsetAsync", CONV_MEM, API_DRIVER)),
        ("cuMemsetD8_v2", ("hipMemsetD8", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemsetD8Async",
            ("hipMemsetD8Async", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMipmappedArrayCreate",
            ("hipMipmappedArrayCreate", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMipmappedArrayDestroy",
            ("hipMipmappedArrayDestroy", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMipmappedArrayGetLevel",
            ("hipMipmappedArrayGetLevel", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemPrefetchAsync",
            ("hipMemPrefetchAsync__", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuMemAdvise", ("hipMemAdvise", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuMemRangeGetAttribute",
            ("hipMemRangeGetAttribute", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemRangeGetAttributes",
            ("hipMemRangeGetAttributes", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuPointerGetAttribute",
            ("hipPointerGetAttribute", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuMemGetAddressRange_v2",
            ("hipMemGetAddressRange", CONV_MEM, API_DRIVER),
        ),
        (
            "cuPointerGetAttributes",
            ("hipPointerGetAttributes", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuPointerSetAttribute",
            ("hipPointerSetAttribute", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CU_TR_FILTER_MODE_POINT", ("hipFilterModePoint", CONV_TEX, API_DRIVER)),
        (
            "CU_TR_FILTER_MODE_LINEAR",
            ("hipFilterModeLinear", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetAddress",
            ("hipTexRefGetAddress", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetAddressMode",
            ("hipTexRefGetAddressMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetArray",
            ("hipTexRefGetArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetBorderColor",
            ("hipTexRefGetBorderColor", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetFilterMode",
            ("hipTexRefGetFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetFlags",
            ("hipTexRefGetFlags", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetFormat",
            ("hipTexRefGetFormat", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMaxAnisotropy",
            ("hipTexRefGetMaxAnisotropy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmapFilterMode",
            ("hipTexRefGetMipmapFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmapLevelBias",
            ("hipTexRefGetMipmapLevelBias", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmapLevelClamp",
            ("hipTexRefGetMipmapLevelClamp", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmappedArray",
            ("hipTexRefGetMipmappedArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetAddress",
            ("hipTexRefSetAddress", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetAddress2D",
            ("hipTexRefSetAddress2D", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuTexRefSetAddressMode", ("hipTexRefSetAddressMode", CONV_TEX, API_DRIVER)),
        ("cuTexRefSetArray", ("hipTexRefSetArray", CONV_TEX, API_DRIVER)),
        (
            "cuTexRefSetBorderColor",
            ("hipTexRefSetBorderColor", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuTexRefSetFilterMode", ("hipTexRefSetFilterMode", CONV_TEX, API_DRIVER)),
        ("cuTexRefSetFlags", ("hipTexRefSetFlags", CONV_TEX, API_DRIVER)),
        ("cuTexRefSetFormat", ("hipTexRefSetFormat", CONV_TEX, API_DRIVER)),
        (
            "cuTexRefSetMaxAnisotropy",
            ("hipTexRefSetMaxAnisotropy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmapFilterMode",
            ("hipTexRefSetMipmapFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmapLevelBias",
            ("hipTexRefSetMipmapLevelBias", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmapLevelClamp",
            ("hipTexRefSetMipmapLevelClamp", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmappedArray",
            ("hipTexRefSetMipmappedArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuTexRefCreate", ("hipTexRefCreate", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuTexRefDestroy",
            ("hipTexRefDestroy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuSurfRefGetArray",
            ("hipSurfRefGetArray", CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuSurfRefSetArray",
            ("hipSurfRefSetArray", CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectCreate",
            ("hipTexObjectCreate", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectDestroy",
            ("hipTexObjectDestroy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectGetResourceDesc",
            ("hipTexObjectGetResourceDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectGetResourceViewDesc",
            ("hipTexObjectGetResourceViewDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectGetTextureDesc",
            ("hipTexObjectGetTextureDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuSurfObjectCreate",
            ("hipSurfObjectCreate", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuSurfObjectDestroy",
            ("hipSurfObjectDestroy", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuSurfObjectGetResourceDesc",
            ("hipSurfObjectGetResourceDesc", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsMapResources",
            ("hipGraphicsMapResources", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsResourceGetMappedMipmappedArray",
            (
                "hipGraphicsResourceGetMappedMipmappedArray",
                CONV_GRAPHICS,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsResourceGetMappedPointer",
            (
                "hipGraphicsResourceGetMappedPointer",
                CONV_GRAPHICS,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsResourceSetMapFlags",
            (
                "hipGraphicsResourceSetMapFlags",
                CONV_GRAPHICS,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsSubResourceGetMappedArray",
            (
                "hipGraphicsSubResourceGetMappedArray",
                CONV_GRAPHICS,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsUnmapResources",
            ("hipGraphicsUnmapResources", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsUnregisterResource",
            (
                "hipGraphicsUnregisterResource",
                CONV_GRAPHICS,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuProfilerInitialize",
            ("hipProfilerInitialize", CONV_OTHER, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuProfilerStart", ("hipProfilerStart", CONV_OTHER, API_DRIVER)),
        ("cuProfilerStop", ("hipProfilerStop", CONV_OTHER, API_DRIVER)),
        (
            "CU_GL_DEVICE_LIST_ALL",
            ("HIP_GL_DEVICE_LIST_ALL", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GL_DEVICE_LIST_CURRENT_FRAME",
            ("HIP_GL_DEVICE_LIST_CURRENT_FRAME", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GL_DEVICE_LIST_NEXT_FRAME",
            ("HIP_GL_DEVICE_LIST_NEXT_FRAME", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuGLGetDevices", ("hipGLGetDevices", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuGraphicsGLRegisterBuffer",
            ("hipGraphicsGLRegisterBuffer", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsGLRegisterImage",
            ("hipGraphicsGLRegisterImage", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cuWGLGetDevice", ("hipWGLGetDevice", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CU_GL_MAP_RESOURCE_FLAGS_NONE",
            ("HIP_GL_MAP_RESOURCE_FLAGS_NONE", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY",
                CONV_GL,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
                CONV_GL,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cuGLCtxCreate", ("hipGLCtxCreate", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        ("cuGLInit", ("hipGLInit", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "cuGLMapBufferObject",
            ("hipGLMapBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGLMapBufferObjectAsync",
            ("hipGLMapBufferObjectAsync", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGLRegisterBufferObject",
            ("hipGLRegisterBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGLSetBufferObjectMapFlags",
            ("hipGLSetBufferObjectMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGLUnmapBufferObject",
            ("hipGLUnmapBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGLUnmapBufferObjectAsync",
            ("hipGLUnmapBufferObjectAsync", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGLUnregisterBufferObject",
            ("hipGLUnregisterBufferObject", CONV_GL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_DEVICE_LIST_ALL",
            ("HIP_D3D9_DEVICE_LIST_ALL", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_DEVICE_LIST_CURRENT_FRAME",
            (
                "HIP_D3D9_DEVICE_LIST_CURRENT_FRAME",
                CONV_D3D9,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D9_DEVICE_LIST_NEXT_FRAME",
            ("HIP_D3D9_DEVICE_LIST_NEXT_FRAME", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9CtxCreate",
            ("hipD3D9CtxCreate", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9CtxCreateOnDevice",
            ("hipD3D9CtxCreateOnDevice", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9GetDevice",
            ("hipD3D9GetDevice", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9GetDevices",
            ("hipD3D9GetDevices", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9GetDirect3DDevice",
            ("hipD3D9GetDirect3DDevice", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsD3D9RegisterResource",
            ("hipGraphicsD3D9RegisterResource", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_MAPRESOURCE_FLAGS_NONE",
            ("HIP_D3D9_MAPRESOURCE_FLAGS_NONE", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_MAPRESOURCE_FLAGS_READONLY",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",
                CONV_D3D9,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",
                CONV_D3D9,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D9_REGISTER_FLAGS_NONE",
            ("HIP_D3D9_REGISTER_FLAGS_NONE", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_REGISTER_FLAGS_ARRAY",
            ("HIP_D3D9_REGISTER_FLAGS_ARRAY", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9MapResources",
            ("hipD3D9MapResources", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9RegisterResource",
            ("hipD3D9RegisterResource", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedArray",
            ("hipD3D9ResourceGetMappedArray", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedPitch",
            ("hipD3D9ResourceGetMappedPitch", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedPointer",
            ("hipD3D9ResourceGetMappedPointer", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedSize",
            ("hipD3D9ResourceGetMappedSize", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetSurfaceDimensions",
            (
                "hipD3D9ResourceGetSurfaceDimensions",
                CONV_D3D9,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D9ResourceSetMapFlags",
            ("hipD3D9ResourceSetMapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9UnmapResources",
            ("hipD3D9UnmapResources", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9UnregisterResource",
            ("hipD3D9UnregisterResource", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D10_DEVICE_LIST_ALL",
            ("HIP_D3D10_DEVICE_LIST_ALL", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D10_DEVICE_LIST_CURRENT_FRAME",
            (
                "HIP_D3D10_DEVICE_LIST_CURRENT_FRAME",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_DEVICE_LIST_NEXT_FRAME",
            (
                "HIP_D3D10_DEVICE_LIST_NEXT_FRAME",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D10GetDevice",
            ("hipD3D10GetDevice", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10GetDevices",
            ("hipD3D10GetDevices", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsD3D10RegisterResource",
            (
                "hipGraphicsD3D10RegisterResource",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_MAPRESOURCE_FLAGS_NONE",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_NONE",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_MAPRESOURCE_FLAGS_READONLY",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_REGISTER_FLAGS_NONE",
            ("HIP_D3D10_REGISTER_FLAGS_NONE", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D10_REGISTER_FLAGS_ARRAY",
            ("HIP_D3D10_REGISTER_FLAGS_ARRAY", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10CtxCreate",
            ("hipD3D10CtxCreate", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10CtxCreateOnDevice",
            ("hipD3D10CtxCreateOnDevice", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10GetDirect3DDevice",
            ("hipD3D10GetDirect3DDevice", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10MapResources",
            ("hipD3D10MapResources", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10RegisterResource",
            ("hipD3D10RegisterResource", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetMappedArray",
            ("hipD3D10ResourceGetMappedArray", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetMappedPitch",
            ("hipD3D10ResourceGetMappedPitch", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetMappedPointer",
            (
                "hipD3D10ResourceGetMappedPointer",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D10ResourceGetMappedSize",
            ("hipD3D10ResourceGetMappedSize", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetSurfaceDimensions",
            (
                "hipD3D10ResourceGetSurfaceDimensions",
                CONV_D3D10,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD310ResourceSetMapFlags",
            ("hipD3D10ResourceSetMapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10UnmapResources",
            ("hipD3D10UnmapResources", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10UnregisterResource",
            ("hipD3D10UnregisterResource", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D11_DEVICE_LIST_ALL",
            ("HIP_D3D11_DEVICE_LIST_ALL", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D11_DEVICE_LIST_CURRENT_FRAME",
            (
                "HIP_D3D11_DEVICE_LIST_CURRENT_FRAME",
                CONV_D3D11,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D11_DEVICE_LIST_NEXT_FRAME",
            (
                "HIP_D3D11_DEVICE_LIST_NEXT_FRAME",
                CONV_D3D11,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D11GetDevice",
            ("hipD3D11GetDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D11GetDevices",
            ("hipD3D11GetDevices", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsD3D11RegisterResource",
            (
                "hipGraphicsD3D11RegisterResource",
                CONV_D3D11,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D11CtxCreate",
            ("hipD3D11CtxCreate", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D11CtxCreateOnDevice",
            ("hipD3D11CtxCreateOnDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuD3D11GetDirect3DDevice",
            ("hipD3D11GetDirect3DDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsVDPAURegisterOutputSurface",
            (
                "hipGraphicsVDPAURegisterOutputSurface",
                CONV_VDPAU,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsVDPAURegisterVideoSurface",
            (
                "hipGraphicsVDPAURegisterVideoSurface",
                CONV_VDPAU,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuVDPAUGetDevice",
            ("hipVDPAUGetDevice", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuVDPAUCtxCreate",
            ("hipVDPAUCtxCreate", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerAcquireFrame",
            ("hipEGLStreamConsumerAcquireFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerConnect",
            ("hipEGLStreamConsumerConnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerConnectWithFlags",
            (
                "hipEGLStreamConsumerConnectWithFlags",
                CONV_EGL,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuEGLStreamConsumerDisconnect",
            ("hipEGLStreamConsumerDisconnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerReleaseFrame",
            ("hipEGLStreamConsumerReleaseFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerConnect",
            ("hipEGLStreamProducerConnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerDisconnect",
            ("hipEGLStreamProducerDisconnect", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerPresentFrame",
            ("hipEGLStreamProducerPresentFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerReturnFrame",
            ("hipEGLStreamProducerReturnFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsEGLRegisterImage",
            ("hipGraphicsEGLRegisterImage", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsResourceGetMappedEglFrame",
            (
                "hipGraphicsResourceGetMappedEglFrame",
                CONV_EGL,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cudaDataType_t", ("hipDataType", CONV_TYPE, API_RUNTIME)),
        ("cudaDataType", ("hipDataType", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_32F", ("HIP_R_32F", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_64F", ("HIP_R_64F", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_16F", ("HIP_R_16F", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_8I", ("HIP_R_8I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_32F", ("HIP_C_32F", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_64F", ("HIP_C_64F", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_16F", ("HIP_C_16F", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_8I", ("HIP_C_8I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_8U", ("HIP_R_8U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_8U", ("HIP_C_8U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_32I", ("HIP_R_32I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_32I", ("HIP_C_32I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_32U", ("HIP_R_32U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_32U", ("HIP_C_32U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_16BF", ("HIP_R_16BF", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_16BF", ("HIP_C_16BF", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_4I", ("HIP_R_4I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_4I", ("HIP_C_4I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_4U", ("HIP_R_4U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_4U", ("HIP_C_4U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_16I", ("HIP_R_16I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_16I", ("HIP_C_16I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_16U", ("HIP_R_16U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_16U", ("HIP_C_16U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_64I", ("HIP_R_64I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_64I", ("HIP_C_64I", CONV_TYPE, API_RUNTIME)),
        ("CUDA_R_64U", ("HIP_R_64U", CONV_TYPE, API_RUNTIME)),
        ("CUDA_C_64U", ("HIP_C_64U", CONV_TYPE, API_RUNTIME)),
        (
            "MAJOR_VERSION",
            ("hipLibraryMajorVersion", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "MINOR_VERSION",
            ("hipLibraryMinorVersion", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "PATCH_LEVEL",
            ("hipLibraryPatchVersion", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAttachGlobal",
            ("hipMemAttachGlobal", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAttachHost",
            ("hipMemAttachHost", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAttachSingle",
            ("hipMemAttachSingle", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaOccupancyDefault",
            ("hipOccupancyDefault", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaOccupancyDisableCachingOverride",
            (
                "hipOccupancyDisableCachingOverride",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cudaGetLastError", ("hipGetLastError", CONV_ERROR, API_RUNTIME)),
        ("cudaPeekAtLastError", ("hipPeekAtLastError", CONV_ERROR, API_RUNTIME)),
        ("cudaGetErrorName", ("hipGetErrorName", CONV_ERROR, API_RUNTIME)),
        ("cudaGetErrorString", ("hipGetErrorString", CONV_ERROR, API_RUNTIME)),
        ("cudaMemcpy3DParms", ("hipMemcpy3DParms", CONV_MEM, API_RUNTIME)),
        (
            "cudaMemcpy3DPeerParms",
            ("hipMemcpy3DPeerParms", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMemcpy", ("hipMemcpy", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpyToArray", ("hipMemcpyToArray", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpyToSymbol", ("hipMemcpyToSymbol", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpyToSymbolAsync", ("hipMemcpyToSymbolAsync", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpyAsync", ("hipMemcpyAsync", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpy2D", ("hipMemcpy2D", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpy2DAsync", ("hipMemcpy2DAsync", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpy2DToArray", ("hipMemcpy2DToArray", CONV_MEM, API_RUNTIME)),
        (
            "cudaMemcpy2DArrayToArray",
            ("hipMemcpy2DArrayToArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy2DFromArray",
            ("hipMemcpy2DFromArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy2DFromArrayAsync",
            ("hipMemcpy2DFromArrayAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy2DToArrayAsync",
            ("hipMemcpy2DToArrayAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMemcpy3D", ("hipMemcpy3D", CONV_MEM, API_RUNTIME)),
        (
            "cudaMemcpy3DAsync",
            ("hipMemcpy3DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy3DPeer",
            ("hipMemcpy3DPeer", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy3DPeerAsync",
            ("hipMemcpy3DPeerAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpyArrayToArray",
            ("hipMemcpyArrayToArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpyFromArrayAsync",
            ("hipMemcpyFromArrayAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMemcpyFromSymbol", ("hipMemcpyFromSymbol", CONV_MEM, API_RUNTIME)),
        (
            "cudaMemcpyFromSymbolAsync",
            ("hipMemcpyFromSymbolAsync", CONV_MEM, API_RUNTIME),
        ),
        ("cudaMemAdvise", ("hipMemAdvise", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaMemRangeGetAttribute",
            ("hipMemRangeGetAttribute", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeGetAttributes",
            ("hipMemRangeGetAttributes", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseSetReadMostly",
            ("hipMemAdviseSetReadMostly", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseUnsetReadMostly",
            ("hipMemAdviseUnsetReadMostly", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseSetPreferredLocation",
            (
                "hipMemAdviseSetPreferredLocation",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaMemAdviseUnsetPreferredLocation",
            (
                "hipMemAdviseUnsetPreferredLocation",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaMemAdviseSetAccessedBy",
            ("hipMemAdviseSetAccessedBy", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseUnsetAccessedBy",
            ("hipMemAdviseUnsetAccessedBy", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeAttributeReadMostly",
            ("hipMemRangeAttributeReadMostly", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeAttributePreferredLocation",
            (
                "hipMemRangeAttributePreferredLocation",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaMemRangeAttributeAccessedBy",
            ("hipMemRangeAttributeAccessedBy", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeAttributeLastPrefetchLocation",
            (
                "hipMemRangeAttributeLastPrefetchLocation",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cudaMemcpyHostToHost", ("hipMemcpyHostToHost", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpyHostToDevice", ("hipMemcpyHostToDevice", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpyDeviceToHost", ("hipMemcpyDeviceToHost", CONV_MEM, API_RUNTIME)),
        (
            "cudaMemcpyDeviceToDevice",
            ("hipMemcpyDeviceToDevice", CONV_MEM, API_RUNTIME),
        ),
        ("cudaMemcpyDefault", ("hipMemcpyDefault", CONV_MEM, API_RUNTIME)),
        ("cudaMemset", ("hipMemset", CONV_MEM, API_RUNTIME)),
        ("cudaMemsetAsync", ("hipMemsetAsync", CONV_MEM, API_RUNTIME)),
        ("cudaMemset2D", ("hipMemset2D", CONV_MEM, API_RUNTIME)),
        (
            "cudaMemset2DAsync",
            ("hipMemset2DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMemset3D", ("hipMemset3D", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaMemset3DAsync",
            ("hipMemset3DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMemGetInfo", ("hipMemGetInfo", CONV_MEM, API_RUNTIME)),
        (
            "cudaArrayGetInfo",
            ("hipArrayGetInfo", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaFreeMipmappedArray",
            ("hipFreeMipmappedArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGetMipmappedArrayLevel",
            ("hipGetMipmappedArrayLevel", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSymbolAddress",
            ("hipGetSymbolAddress", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSymbolSize",
            ("hipGetSymbolSize", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMemPrefetchAsync",
            ("hipMemPrefetchAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMallocHost", ("hipHostMalloc", CONV_MEM, API_RUNTIME)),
        ("cudaMallocArray", ("hipMallocArray", CONV_MEM, API_RUNTIME)),
        ("cudaMalloc", ("hipMalloc", CONV_MEM, API_RUNTIME)),
        ("cudaMalloc3D", ("hipMalloc3D", CONV_MEM, API_RUNTIME)),
        ("cudaMalloc3DArray", ("hipMalloc3DArray", CONV_MEM, API_RUNTIME)),
        (
            "cudaMallocManaged",
            ("hipMallocManaged", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaMallocMipmappedArray",
            ("hipMallocMipmappedArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMallocPitch", ("hipMallocPitch", CONV_MEM, API_RUNTIME)),
        ("cudaFreeHost", ("hipHostFree", CONV_MEM, API_RUNTIME)),
        ("cudaFreeArray", ("hipFreeArray", CONV_MEM, API_RUNTIME)),
        ("cudaFree", ("hipFree", CONV_MEM, API_RUNTIME)),
        ("cudaHostRegister", ("hipHostRegister", CONV_MEM, API_RUNTIME)),
        ("cudaHostUnregister", ("hipHostUnregister", CONV_MEM, API_RUNTIME)),
        ("cudaHostAlloc", ("hipHostMalloc", CONV_MEM, API_RUNTIME)),
        ("cudaMemoryTypeHost", ("hipMemoryTypeHost", CONV_MEM, API_RUNTIME)),
        ("cudaMemoryTypeDevice", ("hipMemoryTypeDevice", CONV_MEM, API_RUNTIME)),
        ("make_cudaExtent", ("make_hipExtent", CONV_MEM, API_RUNTIME)),
        ("make_cudaPitchedPtr", ("make_hipPitchedPtr", CONV_MEM, API_RUNTIME)),
        ("make_cudaPos", ("make_hipPos", CONV_MEM, API_RUNTIME)),
        ("cudaHostAllocDefault", ("hipHostMallocDefault", CONV_MEM, API_RUNTIME)),
        ("cudaHostAllocPortable", ("hipHostMallocPortable", CONV_MEM, API_RUNTIME)),
        ("cudaHostAllocMapped", ("hipHostMallocMapped", CONV_MEM, API_RUNTIME)),
        (
            "cudaHostAllocWriteCombined",
            ("hipHostMallocWriteCombined", CONV_MEM, API_RUNTIME),
        ),
        ("cudaHostGetFlags", ("hipHostGetFlags", CONV_MEM, API_RUNTIME)),
        ("cudaHostRegisterDefault", ("hipHostRegisterDefault", CONV_MEM, API_RUNTIME)),
        (
            "cudaHostRegisterPortable",
            ("hipHostRegisterPortable", CONV_MEM, API_RUNTIME),
        ),
        ("cudaHostRegisterMapped", ("hipHostRegisterMapped", CONV_MEM, API_RUNTIME)),
        (
            "cudaHostRegisterIoMemory",
            ("hipHostRegisterIoMemory", CONV_MEM, API_RUNTIME),
        ),
        # ("warpSize", ("hipWarpSize", CONV_SPECIAL_FUNC, API_RUNTIME), (HIP actually uses warpSize...)),
        ("cudaEventCreate", ("hipEventCreate", CONV_EVENT, API_RUNTIME)),
        (
            "cudaEventCreateWithFlags",
            ("hipEventCreateWithFlags", CONV_EVENT, API_RUNTIME),
        ),
        ("cudaEventDestroy", ("hipEventDestroy", CONV_EVENT, API_RUNTIME)),
        ("cudaEventRecord", ("hipEventRecord", CONV_EVENT, API_RUNTIME)),
        ("cudaEventElapsedTime", ("hipEventElapsedTime", CONV_EVENT, API_RUNTIME)),
        ("cudaEventSynchronize", ("hipEventSynchronize", CONV_EVENT, API_RUNTIME)),
        ("cudaEventQuery", ("hipEventQuery", CONV_EVENT, API_RUNTIME)),
        ("cudaEventDefault", ("hipEventDefault", CONV_EVENT, API_RUNTIME)),
        ("cudaEventBlockingSync", ("hipEventBlockingSync", CONV_EVENT, API_RUNTIME)),
        ("cudaEventDisableTiming", ("hipEventDisableTiming", CONV_EVENT, API_RUNTIME)),
        ("cudaEventInterprocess", ("hipEventInterprocess", CONV_EVENT, API_RUNTIME)),
        ("cudaStreamCreate", ("hipStreamCreate", CONV_STREAM, API_RUNTIME)),
        (
            "cudaStreamCreateWithFlags",
            ("hipStreamCreateWithFlags", CONV_STREAM, API_RUNTIME),
        ),
        (
            "cudaStreamCreateWithPriority",
            ("hipStreamCreateWithPriority", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaStreamDestroy", ("hipStreamDestroy", CONV_STREAM, API_RUNTIME)),
        ("cudaStreamWaitEvent", ("hipStreamWaitEvent", CONV_STREAM, API_RUNTIME)),
        ("cudaStreamSynchronize", ("hipStreamSynchronize", CONV_STREAM, API_RUNTIME)),
        ("cudaStreamGetFlags", ("hipStreamGetFlags", CONV_STREAM, API_RUNTIME)),
        ("cudaStreamQuery", ("hipStreamQuery", CONV_STREAM, API_RUNTIME)),
        ("cudaStreamAddCallback", ("hipStreamAddCallback", CONV_STREAM, API_RUNTIME)),
        (
            "cudaStreamAttachMemAsync",
            ("hipStreamAttachMemAsync", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaStreamGetPriority",
            ("hipStreamGetPriority", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaCpuDeviceId", ("hipCpuDeviceId", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamDefault", ("hipStreamDefault", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamNonBlocking", ("hipStreamNonBlocking", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamGetCaptureInfo", ("hipStreamGetCaptureInfo", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamGetCaptureInfo_v2", ("hipStreamGetCaptureInfo_v2", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamCaptureStatus", ("hipStreamCaptureStatus", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamCaptureStatusActive", ("hipStreamCaptureStatusActive", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamCaptureMode", ("hipStreamCaptureMode", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamCaptureModeGlobal", ("hipStreamCaptureModeGlobal", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamCaptureModeRelaxed", ("hipStreamCaptureModeRelaxed", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamCaptureModeThreadLocal", ("hipStreamCaptureModeThreadLocal", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamBeginCapture", ("hipStreamBeginCapture", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamEndCapture", ("hipStreamEndCapture", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphInstantiate", ("hipGraphInstantiate", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphDestroy", ("hipGraphDestroy", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphExecDestroy", ("hipGraphExecDestroy", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphLaunch", ("hipGraphLaunch", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphGetNodes", ("hipGraphGetNodes", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphDebugDotPrint", ("hipGraphDebugDotPrint", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphDebugDotFlagsVerbose", ("hipGraphDebugDotFlagsVerbose", CONV_NUMERIC_LITERAL, API_RUNTIME)),
        ("cudaGraphRetainUserObject", ("hipGraphRetainUserObject", CONV_TYPE, API_RUNTIME)),
        ("cudaGraphUserObjectMove", ("hipGraphUserObjectMove", CONV_TYPE, API_RUNTIME)),
        ("cudaUserObject_t", ("hipUserObject_t", CONV_TYPE, API_RUNTIME)),
        ("cudaUserObjectCreate", ("hipUserObjectCreate", CONV_TYPE, API_RUNTIME)),
        ("cudaUserObjectNoDestructorSync", ("hipUserObjectNoDestructorSync", CONV_TYPE, API_RUNTIME)),
        ("cudaThreadExchangeStreamCaptureMode", ("hipThreadExchangeStreamCaptureMode", CONV_TYPE, API_RUNTIME)),
        ("cudaStreamIsCapturing", ("hipStreamIsCapturing", CONV_TYPE, API_RUNTIME)),
        ("cudaDeviceSynchronize", ("hipDeviceSynchronize", CONV_DEVICE, API_RUNTIME)),
        ("cudaDeviceReset", ("hipDeviceReset", CONV_DEVICE, API_RUNTIME)),
        ("cudaSetDevice", ("hipSetDevice", CONV_DEVICE, API_RUNTIME)),
        ("cudaGetDevice", ("hipGetDevice", CONV_DEVICE, API_RUNTIME)),
        ("cudaGetDeviceCount", ("hipGetDeviceCount", CONV_DEVICE, API_RUNTIME)),
        ("cudaChooseDevice", ("hipChooseDevice", CONV_DEVICE, API_RUNTIME)),
        ("cudaThreadExit", ("hipDeviceReset", CONV_THREAD, API_RUNTIME)),
        (
            "cudaThreadGetCacheConfig",
            ("hipDeviceGetCacheConfig", CONV_THREAD, API_RUNTIME),
        ),
        (
            "cudaThreadGetLimit",
            ("hipThreadGetLimit", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaThreadSetCacheConfig",
            ("hipDeviceSetCacheConfig", CONV_THREAD, API_RUNTIME),
        ),
        (
            "cudaThreadSetLimit",
            ("hipThreadSetLimit", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaThreadSynchronize", ("hipDeviceSynchronize", CONV_THREAD, API_RUNTIME)),
        ("cudaDeviceGetAttribute", ("hipDeviceGetAttribute", CONV_DEVICE, API_RUNTIME)),
        (
            "cudaDevAttrMaxThreadsPerBlock",
            ("hipDeviceAttributeMaxThreadsPerBlock", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxBlockDimX",
            ("hipDeviceAttributeMaxBlockDimX", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxBlockDimY",
            ("hipDeviceAttributeMaxBlockDimY", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxBlockDimZ",
            ("hipDeviceAttributeMaxBlockDimZ", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxGridDimX",
            ("hipDeviceAttributeMaxGridDimX", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxGridDimY",
            ("hipDeviceAttributeMaxGridDimY", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxGridDimZ",
            ("hipDeviceAttributeMaxGridDimZ", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxSharedMemoryPerBlock",
            ("hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxSharedMemoryPerBlockOptin",
            ("hipDeviceAttributeMaxSharedMemoryPerBlock", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrTotalConstantMemory",
            ("hipDeviceAttributeTotalConstantMemory", CONV_TYPE, API_RUNTIME),
        ),
        ("cudaDevAttrWarpSize", ("hipDeviceAttributeWarpSize", CONV_TYPE, API_RUNTIME)),
        (
            "cudaDevAttrMaxPitch",
            ("hipDeviceAttributeMaxPitch", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrMaxRegistersPerBlock",
            ("hipDeviceAttributeMaxRegistersPerBlock", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrClockRate",
            ("hipDeviceAttributeClockRate", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrTextureAlignment",
            (
                "hipDeviceAttributeTextureAlignment",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrGpuOverlap",
            ("hipDeviceAttributeGpuOverlap", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrMultiProcessorCount",
            ("hipDeviceAttributeMultiprocessorCount", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrKernelExecTimeout",
            (
                "hipDeviceAttributeKernelExecTimeout",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrIntegrated",
            ("hipDeviceAttributeIntegrated", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrCanMapHostMemory",
            (
                "hipDeviceAttributeCanMapHostMemory",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrComputeMode",
            ("hipDeviceAttributeComputeMode", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxTexture1DWidth",
            (
                "hipDeviceAttributeMaxTexture1DWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DWidth",
            (
                "hipDeviceAttributeMaxTexture2DWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DHeight",
            (
                "hipDeviceAttributeMaxTexture2DHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DWidth",
            (
                "hipDeviceAttributeMaxTexture3DWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DHeight",
            (
                "hipDeviceAttributeMaxTexture3DHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DDepth",
            (
                "hipDeviceAttributeMaxTexture3DDepth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLayeredWidth",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLayeredHeight",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLayeredLayers",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrSurfaceAlignment",
            (
                "hipDeviceAttributeSurfaceAlignment",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrConcurrentKernels",
            ("hipDeviceAttributeConcurrentKernels", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrEccEnabled",
            ("hipDeviceAttributeEccEnabled", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaDevAttrPciBusId", ("hipDeviceAttributePciBusId", CONV_TYPE, API_RUNTIME)),
        (
            "cudaDevAttrPciDeviceId",
            ("hipDeviceAttributePciDeviceId", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrTccDriver",
            ("hipDeviceAttributeTccDriver", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrMemoryClockRate",
            ("hipDeviceAttributeMemoryClockRate", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrGlobalMemoryBusWidth",
            ("hipDeviceAttributeMemoryBusWidth", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrL2CacheSize",
            ("hipDeviceAttributeL2CacheSize", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxThreadsPerMultiProcessor",
            ("hipDeviceAttributeMaxThreadsPerMultiProcessor", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrAsyncEngineCount",
            (
                "hipDeviceAttributeAsyncEngineCount",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrUnifiedAddressing",
            (
                "hipDeviceAttributeUnifiedAddressing",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture1DLayeredWidth",
            (
                "hipDeviceAttributeMaxTexture1DLayeredWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture1DLayeredLayers",
            (
                "hipDeviceAttributeMaxTexture1DLayeredLayers",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DGatherWidth",
            (
                "hipDeviceAttributeMaxTexture2DGatherWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DGatherHeight",
            (
                "hipDeviceAttributeMaxTexture2DGatherHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DWidthAlt",
            (
                "hipDeviceAttributeMaxTexture3DWidthAlternate",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DHeightAlt",
            (
                "hipDeviceAttributeMaxTexture3DHeightAlternate",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DDepthAlt",
            (
                "hipDeviceAttributeMaxTexture3DDepthAlternate",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrPciDomainId",
            ("hipDeviceAttributePciDomainId", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrTexturePitchAlignment",
            (
                "hipDeviceAttributeTexturePitchAlignment",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTextureCubemapWidth",
            (
                "hipDeviceAttributeMaxTextureCubemapWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTextureCubemapLayeredWidth",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTextureCubemapLayeredLayers",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredLayers",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface1DWidth",
            (
                "hipDeviceAttributeMaxSurface1DWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DWidth",
            (
                "hipDeviceAttributeMaxSurface2DWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DHeight",
            (
                "hipDeviceAttributeMaxSurface2DHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface3DWidth",
            (
                "hipDeviceAttributeMaxSurface3DWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface3DHeight",
            (
                "hipDeviceAttributeMaxSurface3DHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface3DDepth",
            (
                "hipDeviceAttributeMaxSurface3DDepth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface1DLayeredWidth",
            (
                "hipDeviceAttributeMaxSurface1DLayeredWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface1DLayeredLayers",
            (
                "hipDeviceAttributeMaxSurface1DLayeredLayers",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DLayeredWidth",
            (
                "hipDeviceAttributeMaxSurface2DLayeredWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DLayeredHeight",
            (
                "hipDeviceAttributeMaxSurface2DLayeredHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DLayeredLayers",
            (
                "hipDeviceAttributeMaxSurface2DLayeredLayers",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurfaceCubemapWidth",
            (
                "hipDeviceAttributeMaxSurfaceCubemapWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurfaceCubemapLayeredWidth",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurfaceCubemapLayeredLayers",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture1DLinearWidth",
            (
                "hipDeviceAttributeMaxTexture1DLinearWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLinearWidth",
            (
                "hipDeviceAttributeMaxTexture2DLinearWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLinearHeight",
            (
                "hipDeviceAttributeMaxTexture2DLinearHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLinearPitch",
            (
                "hipDeviceAttributeMaxTexture2DLinearPitch",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DMipmappedWidth",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DMipmappedHeight",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedHeight",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrComputeCapabilityMajor",
            ("hipDeviceAttributeComputeCapabilityMajor", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrComputeCapabilityMinor",
            ("hipDeviceAttributeComputeCapabilityMinor", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxTexture1DMipmappedWidth",
            (
                "hipDeviceAttributeMaxTexture1DMipmappedWidth",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrStreamPrioritiesSupported",
            (
                "hipDeviceAttributeStreamPrioritiesSupported",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrGlobalL1CacheSupported",
            (
                "hipDeviceAttributeGlobalL1CacheSupported",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrLocalL1CacheSupported",
            (
                "hipDeviceAttributeLocalL1CacheSupported",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSharedMemoryPerMultiprocessor",
            (
                "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
                CONV_TYPE,
                API_RUNTIME,
            ),
        ),
        (
            "cudaDevAttrMaxRegistersPerMultiprocessor",
            (
                "hipDeviceAttributeMaxRegistersPerMultiprocessor",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrManagedMemory",
            (
                "hipDeviceAttributeManagedMemory",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrIsMultiGpuBoard",
            ("hipDeviceAttributeIsMultiGpuBoard", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDevAttrMultiGpuBoardGroupID",
            (
                "hipDeviceAttributeMultiGpuBoardGroupID",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrHostNativeAtomicSupported",
            (
                "hipDeviceAttributeHostNativeAtomicSupported",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrSingleToDoublePrecisionPerfRatio",
            (
                "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrPageableMemoryAccess",
            (
                "hipDeviceAttributePageableMemoryAccess",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrConcurrentManagedAccess",
            (
                "hipDeviceAttributeConcurrentManagedAccess",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrComputePreemptionSupported",
            (
                "hipDeviceAttributeComputePreemptionSupported",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrCanUseHostPointerForRegisteredMem",
            (
                "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaPointerGetAttributes",
            ("hipPointerGetAttributes", CONV_MEM, API_RUNTIME),
        ),
        (
            "cudaHostGetDevicePointer",
            ("hipHostGetDevicePointer", CONV_MEM, API_RUNTIME),
        ),
        (
            "cudaGetDeviceProperties",
            ("hipGetDeviceProperties", CONV_DEVICE, API_RUNTIME),
        ),
        ("cudaDeviceGetPCIBusId", ("hipDeviceGetPCIBusId", CONV_DEVICE, API_RUNTIME)),
        (
            "cudaDeviceGetByPCIBusId",
            ("hipDeviceGetByPCIBusId", CONV_DEVICE, API_RUNTIME),
        ),
        (
            "cudaDeviceGetStreamPriorityRange",
            (
                "hipDeviceGetStreamPriorityRange",
                CONV_DEVICE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaSetValidDevices",
            ("hipSetValidDevices", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaDevP2PAttrPerformanceRank",
            (
                "hipDeviceP2PAttributePerformanceRank",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevP2PAttrAccessSupported",
            (
                "hipDeviceP2PAttributeAccessSupported",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevP2PAttrNativeAtomicSupported",
            (
                "hipDeviceP2PAttributeNativeAtomicSupported",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDeviceGetP2PAttribute",
            ("hipDeviceGetP2PAttribute", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeDefault",
            ("hipComputeModeDefault", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeExclusive",
            ("hipComputeModeExclusive", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeProhibited",
            ("hipComputeModeProhibited", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeExclusiveProcess",
            ("hipComputeModeExclusiveProcess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGetDeviceFlags",
            ("hipGetDeviceFlags", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaSetDeviceFlags", ("hipSetDeviceFlags", CONV_DEVICE, API_RUNTIME)),
        ("cudaDeviceScheduleAuto", ("hipDeviceScheduleAuto", CONV_TYPE, API_RUNTIME)),
        ("cudaDeviceScheduleSpin", ("hipDeviceScheduleSpin", CONV_TYPE, API_RUNTIME)),
        ("cudaDeviceScheduleYield", ("hipDeviceScheduleYield", CONV_TYPE, API_RUNTIME)),
        (
            "cudaDeviceBlockingSync",
            ("hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDeviceScheduleBlockingSync",
            ("hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDeviceScheduleMask",
            ("hipDeviceScheduleMask", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaDeviceMapHost", ("hipDeviceMapHost", CONV_TYPE, API_RUNTIME)),
        (
            "cudaDeviceLmemResizeToMax",
            ("hipDeviceLmemResizeToMax", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaDeviceMask", ("hipDeviceMask", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaDeviceSetCacheConfig",
            ("hipDeviceSetCacheConfig", CONV_CACHE, API_RUNTIME),
        ),
        (
            "cudaDeviceGetCacheConfig",
            ("hipDeviceGetCacheConfig", CONV_CACHE, API_RUNTIME),
        ),
        (
            "cudaFuncAttributes",
            ("hipFuncAttributes", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaFuncAttributeMaxDynamicSharedMemorySize",
            ("hipFuncAttributeMaxDynamicSharedMemorySize", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaFuncAttributePreferredSharedMemoryCarveout",
            ("hipFuncAttributePreferredSharedMemoryCarveout", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaFuncSetAttribute",
            ("hipFuncSetAttribute", CONV_EXEC, API_RUNTIME),
        ),
        ("cudaFuncSetCacheConfig", ("hipFuncSetCacheConfig", CONV_CACHE, API_RUNTIME)),
        (
            "cudaFuncCachePreferNone",
            ("hipFuncCachePreferNone", CONV_CACHE, API_RUNTIME),
        ),
        (
            "cudaFuncCachePreferShared",
            ("hipFuncCachePreferShared", CONV_CACHE, API_RUNTIME),
        ),
        ("cudaFuncCachePreferL1", ("hipFuncCachePreferL1", CONV_CACHE, API_RUNTIME)),
        (
            "cudaFuncCachePreferEqual",
            ("hipFuncCachePreferEqual", CONV_CACHE, API_RUNTIME),
        ),
        (
            "cudaFuncGetAttributes",
            ("hipFuncGetAttributes", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaFuncSetSharedMemConfig",
            ("hipFuncSetSharedMemConfig", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGetParameterBuffer",
            ("hipGetParameterBuffer", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaSetDoubleForDevice",
            ("hipSetDoubleForDevice", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaSetDoubleForHost",
            ("hipSetDoubleForHost", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaConfigureCall",
            ("hipConfigureCall", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaLaunch", ("hipLaunch", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaLaunchCooperativeKernel",
            ("hipLaunchCooperativeKernel", CONV_EXEC, API_RUNTIME),
        ),
        (
            "cudaSetupArgument",
            ("hipSetupArgument", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaDriverGetVersion", ("hipDriverGetVersion", CONV_VERSION, API_RUNTIME)),
        (
            "cudaRuntimeGetVersion",
            ("hipRuntimeGetVersion", CONV_VERSION, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSize",
            ("hipOccupancyMaxPotentialBlockSize", CONV_OCCUPANCY, API_RUNTIME),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSizeWithFlags",
            (
                "hipOccupancyMaxPotentialBlockSizeWithFlags",
                CONV_OCCUPANCY,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            (
                "hipOccupancyMaxActiveBlocksPerMultiprocessor",
                CONV_OCCUPANCY,
                API_RUNTIME,
            ),
        ),
        (
            "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            (
                "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                CONV_OCCUPANCY,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSizeVariableSMem",
            (
                "hipOccupancyMaxPotentialBlockSizeVariableSMem",
                CONV_OCCUPANCY,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
            (
                "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
                CONV_OCCUPANCY,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cudaDeviceCanAccessPeer", ("hipDeviceCanAccessPeer", CONV_PEER, API_RUNTIME)),
        (
            "cudaDeviceDisablePeerAccess",
            ("hipDeviceDisablePeerAccess", CONV_PEER, API_RUNTIME),
        ),
        (
            "cudaDeviceEnablePeerAccess",
            ("hipDeviceEnablePeerAccess", CONV_PEER, API_RUNTIME),
        ),
        ("cudaMemcpyPeerAsync", ("hipMemcpyPeerAsync", CONV_MEM, API_RUNTIME)),
        ("cudaMemcpyPeer", ("hipMemcpyPeer", CONV_MEM, API_RUNTIME)),
        (
            "cudaIpcMemLazyEnablePeerAccess",
            ("hipIpcMemLazyEnablePeerAccess", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaDeviceSetSharedMemConfig",
            ("hipDeviceSetSharedMemConfig", CONV_DEVICE, API_RUNTIME),
        ),
        (
            "cudaDeviceGetSharedMemConfig",
            ("hipDeviceGetSharedMemConfig", CONV_DEVICE, API_RUNTIME),
        ),
        (
            "cudaSharedMemBankSizeDefault",
            ("hipSharedMemBankSizeDefault", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaSharedMemBankSizeFourByte",
            ("hipSharedMemBankSizeFourByte", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaSharedMemBankSizeEightByte",
            ("hipSharedMemBankSizeEightByte", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaLimitStackSize",
            ("hipLimitStackSize", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaLimitPrintfFifoSize",
            ("hipLimitPrintfFifoSize", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaLimitMallocHeapSize", ("hipLimitMallocHeapSize", CONV_TYPE, API_RUNTIME)),
        (
            "cudaLimitDevRuntimeSyncDepth",
            ("hipLimitDevRuntimeSyncDepth", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaLimitDevRuntimePendingLaunchCount",
            (
                "hipLimitDevRuntimePendingLaunchCount",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cudaDeviceGetLimit", ("hipDeviceGetLimit", CONV_DEVICE, API_RUNTIME)),
        (
            "cudaProfilerInitialize",
            ("hipProfilerInitialize", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaProfilerStart", ("hipProfilerStart", CONV_OTHER, API_RUNTIME)),
        ("cudaProfilerStop", ("hipProfilerStop", CONV_OTHER, API_RUNTIME)),
        (
            "cudaKeyValuePair",
            ("hipKeyValuePair", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaCSV", ("hipCSV", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED)),
        ("cudaReadModeElementType", ("hipReadModeElementType", CONV_TEX, API_RUNTIME)),
        (
            "cudaReadModeNormalizedFloat",
            ("hipReadModeNormalizedFloat", CONV_TEX, API_RUNTIME),
        ),
        ("cudaFilterModePoint", ("hipFilterModePoint", CONV_TEX, API_RUNTIME)),
        ("cudaFilterModeLinear", ("hipFilterModeLinear", CONV_TEX, API_RUNTIME)),
        ("cudaBindTexture", ("hipBindTexture", CONV_TEX, API_RUNTIME)),
        ("cudaUnbindTexture", ("hipUnbindTexture", CONV_TEX, API_RUNTIME)),
        ("cudaBindTexture2D", ("hipBindTexture2D", CONV_TEX, API_RUNTIME)),
        ("cudaBindTextureToArray", ("hipBindTextureToArray", CONV_TEX, API_RUNTIME)),
        (
            "cudaBindTextureToMipmappedArray",
            ("hipBindTextureToMipmappedArray", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaGetTextureAlignmentOffset",
            ("hipGetTextureAlignmentOffset", CONV_TEX, API_RUNTIME),
        ),
        ("cudaGetTextureReference", ("hipGetTextureReference", CONV_TEX, API_RUNTIME)),
        (
            "cudaChannelFormatKindSigned",
            ("hipChannelFormatKindSigned", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaChannelFormatKindUnsigned",
            ("hipChannelFormatKindUnsigned", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaChannelFormatKindFloat",
            ("hipChannelFormatKindFloat", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaChannelFormatKindNone",
            ("hipChannelFormatKindNone", CONV_TEX, API_RUNTIME),
        ),
        ("cudaCreateChannelDesc", ("hipCreateChannelDesc", CONV_TEX, API_RUNTIME)),
        ("cudaGetChannelDesc", ("hipGetChannelDesc", CONV_TEX, API_RUNTIME)),
        ("cudaResourceTypeArray", ("hipResourceTypeArray", CONV_TEX, API_RUNTIME)),
        (
            "cudaResourceTypeMipmappedArray",
            ("hipResourceTypeMipmappedArray", CONV_TEX, API_RUNTIME),
        ),
        ("cudaResourceTypeLinear", ("hipResourceTypeLinear", CONV_TEX, API_RUNTIME)),
        ("cudaResourceTypePitch2D", ("hipResourceTypePitch2D", CONV_TEX, API_RUNTIME)),
        ("cudaResViewFormatNone", ("hipResViewFormatNone", CONV_TEX, API_RUNTIME)),
        (
            "cudaResViewFormatUnsignedChar1",
            ("hipResViewFormatUnsignedChar1", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedChar2",
            ("hipResViewFormatUnsignedChar2", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedChar4",
            ("hipResViewFormatUnsignedChar4", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedChar1",
            ("hipResViewFormatSignedChar1", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedChar2",
            ("hipResViewFormatSignedChar2", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedChar4",
            ("hipResViewFormatSignedChar4", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedShort1",
            ("hipResViewFormatUnsignedShort1", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedShort2",
            ("hipResViewFormatUnsignedShort2", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedShort4",
            ("hipResViewFormatUnsignedShort4", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedShort1",
            ("hipResViewFormatSignedShort1", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedShort2",
            ("hipResViewFormatSignedShort2", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedShort4",
            ("hipResViewFormatSignedShort4", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedInt1",
            ("hipResViewFormatUnsignedInt1", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedInt2",
            ("hipResViewFormatUnsignedInt2", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedInt4",
            ("hipResViewFormatUnsignedInt4", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedInt1",
            ("hipResViewFormatSignedInt1", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedInt2",
            ("hipResViewFormatSignedInt2", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedInt4",
            ("hipResViewFormatSignedInt4", CONV_TEX, API_RUNTIME),
        ),
        ("cudaResViewFormatHalf1", ("hipResViewFormatHalf1", CONV_TEX, API_RUNTIME)),
        ("cudaResViewFormatHalf2", ("hipResViewFormatHalf2", CONV_TEX, API_RUNTIME)),
        ("cudaResViewFormatHalf4", ("hipResViewFormatHalf4", CONV_TEX, API_RUNTIME)),
        ("cudaResViewFormatFloat1", ("hipResViewFormatFloat1", CONV_TEX, API_RUNTIME)),
        ("cudaResViewFormatFloat2", ("hipResViewFormatFloat2", CONV_TEX, API_RUNTIME)),
        ("cudaResViewFormatFloat4", ("hipResViewFormatFloat4", CONV_TEX, API_RUNTIME)),
        (
            "cudaResViewFormatUnsignedBlockCompressed1",
            ("hipResViewFormatUnsignedBlockCompressed1", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed2",
            ("hipResViewFormatUnsignedBlockCompressed2", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed3",
            ("hipResViewFormatUnsignedBlockCompressed3", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed4",
            ("hipResViewFormatUnsignedBlockCompressed4", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedBlockCompressed4",
            ("hipResViewFormatSignedBlockCompressed4", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed5",
            ("hipResViewFormatUnsignedBlockCompressed5", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedBlockCompressed5",
            ("hipResViewFormatSignedBlockCompressed5", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed6H",
            ("hipResViewFormatUnsignedBlockCompressed6H", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedBlockCompressed6H",
            ("hipResViewFormatSignedBlockCompressed6H", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed7",
            ("hipResViewFormatUnsignedBlockCompressed7", CONV_TEX, API_RUNTIME),
        ),
        ("cudaAddressModeWrap", ("hipAddressModeWrap", CONV_TEX, API_RUNTIME)),
        ("cudaAddressModeClamp", ("hipAddressModeClamp", CONV_TEX, API_RUNTIME)),
        ("cudaAddressModeMirror", ("hipAddressModeMirror", CONV_TEX, API_RUNTIME)),
        ("cudaAddressModeBorder", ("hipAddressModeBorder", CONV_TEX, API_RUNTIME)),
        ("cudaCreateTextureObject", ("hipCreateTextureObject", CONV_TEX, API_RUNTIME)),
        (
            "cudaDestroyTextureObject",
            ("hipDestroyTextureObject", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaGetTextureObjectResourceDesc",
            ("hipGetTextureObjectResourceDesc", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaGetTextureObjectResourceViewDesc",
            ("hipGetTextureObjectResourceViewDesc", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaGetTextureObjectTextureDesc",
            ("hipGetTextureObjectTextureDesc", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaBindSurfaceToArray",
            ("hipBindSurfaceToArray", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSurfaceReference",
            ("hipGetSurfaceReference", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaBoundaryModeZero",
            ("hipBoundaryModeZero", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaBoundaryModeClamp",
            ("hipBoundaryModeClamp", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaBoundaryModeTrap",
            ("hipBoundaryModeTrap", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaFormatModeForced",
            ("hipFormatModeForced", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaFormatModeAuto",
            ("hipFormatModeAuto", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaCreateSurfaceObject",
            ("hipCreateSurfaceObject", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaDestroySurfaceObject",
            ("hipDestroySurfaceObject", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSurfaceObjectResourceDesc",
            (
                "hipGetSurfaceObjectResourceDesc",
                CONV_SURFACE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cudaIpcCloseMemHandle", ("hipIpcCloseMemHandle", CONV_DEVICE, API_RUNTIME)),
        ("cudaIpcGetEventHandle", ("hipIpcGetEventHandle", CONV_DEVICE, API_RUNTIME)),
        ("cudaIpcGetMemHandle", ("hipIpcGetMemHandle", CONV_DEVICE, API_RUNTIME)),
        ("cudaIpcOpenEventHandle", ("hipIpcOpenEventHandle", CONV_DEVICE, API_RUNTIME)),
        ("cudaIpcOpenMemHandle", ("hipIpcOpenMemHandle", CONV_DEVICE, API_RUNTIME)),
        (
            "cudaGLGetDevices",
            ("hipGLGetDevices", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterBuffer",
            ("hipGraphicsGLRegisterBuffer", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterImage",
            ("hipGraphicsGLRegisterImage", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaWGLGetDevice",
            ("hipWGLGetDevice", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsMapResources",
            ("hipGraphicsMapResources", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsResourceGetMappedMipmappedArray",
            (
                "hipGraphicsResourceGetMappedMipmappedArray",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsResourceGetMappedPointer",
            (
                "hipGraphicsResourceGetMappedPointer",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsResourceSetMapFlags",
            (
                "hipGraphicsResourceSetMapFlags",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsSubResourceGetMappedArray",
            (
                "hipGraphicsSubResourceGetMappedArray",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsUnmapResources",
            ("hipGraphicsUnmapResources", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsUnregisterResource",
            (
                "hipGraphicsUnregisterResource",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFacePositiveX",
            (
                "hipGraphicsCubeFacePositiveX",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFaceNegativeX",
            (
                "hipGraphicsCubeFaceNegativeX",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFacePositiveY",
            (
                "hipGraphicsCubeFacePositiveY",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFaceNegativeY",
            (
                "hipGraphicsCubeFaceNegativeY",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFacePositiveZ",
            (
                "hipGraphicsCubeFacePositiveZ",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFaceNegativeZ",
            (
                "hipGraphicsCubeFaceNegativeZ",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsMapFlagsNone",
            ("hipGraphicsMapFlagsNone", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsMapFlagsReadOnly",
            (
                "hipGraphicsMapFlagsReadOnly",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsMapFlagsWriteDiscard",
            (
                "hipGraphicsMapFlagsWriteDiscard",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsNone",
            (
                "hipGraphicsRegisterFlagsNone",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsReadOnly",
            (
                "hipGraphicsRegisterFlagsReadOnly",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsWriteDiscard",
            (
                "hipGraphicsRegisterFlagsWriteDiscard",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsSurfaceLoadStore",
            (
                "hipGraphicsRegisterFlagsSurfaceLoadStore",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsTextureGather",
            (
                "hipGraphicsRegisterFlagsTextureGather",
                CONV_GRAPHICS,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGLDeviceListAll",
            ("HIP_GL_DEVICE_LIST_ALL", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLDeviceListCurrentFrame",
            ("HIP_GL_DEVICE_LIST_CURRENT_FRAME", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLDeviceListNextFrame",
            ("HIP_GL_DEVICE_LIST_NEXT_FRAME", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLGetDevices",
            ("hipGLGetDevices", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterBuffer",
            ("hipGraphicsGLRegisterBuffer", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterImage",
            ("hipGraphicsGLRegisterImage", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaWGLGetDevice",
            ("hipWGLGetDevice", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLMapFlagsNone",
            ("HIP_GL_MAP_RESOURCE_FLAGS_NONE", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLMapFlagsReadOnly",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY",
                CONV_GL,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGLMapFlagsWriteDiscard",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
                CONV_GL,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGLMapBufferObject",
            ("hipGLMapBufferObject__", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLMapBufferObjectAsync",
            ("hipGLMapBufferObjectAsync__", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLRegisterBufferObject",
            ("hipGLRegisterBufferObject", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLSetBufferObjectMapFlags",
            ("hipGLSetBufferObjectMapFlags", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLSetGLDevice",
            ("hipGLSetGLDevice", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLUnmapBufferObject",
            ("hipGLUnmapBufferObject", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLUnmapBufferObjectAsync",
            ("hipGLUnmapBufferObjectAsync", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLUnregisterBufferObject",
            ("hipGLUnregisterBufferObject", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9DeviceListAll",
            ("HIP_D3D9_DEVICE_LIST_ALL", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9DeviceListCurrentFrame",
            (
                "HIP_D3D9_DEVICE_LIST_CURRENT_FRAME",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9DeviceListNextFrame",
            (
                "HIP_D3D9_DEVICE_LIST_NEXT_FRAME",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9GetDevice",
            ("hipD3D9GetDevice", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9GetDevices",
            ("hipD3D9GetDevices", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9GetDirect3DDevice",
            ("hipD3D9GetDirect3DDevice", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9SetDirect3DDevice",
            ("hipD3D9SetDirect3DDevice", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D9RegisterResource",
            (
                "hipGraphicsD3D9RegisterResource",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9MapFlags",
            ("hipD3D9MapFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9MapFlagsNone",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_NONE",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9MapFlagsReadOnly",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9MapFlagsWriteDiscard",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9RegisterFlagsNone",
            ("HIP_D3D9_REGISTER_FLAGS_NONE", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9RegisterFlagsArray",
            ("HIP_D3D9_REGISTER_FLAGS_ARRAY", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9MapResources",
            ("hipD3D9MapResources", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9RegisterResource",
            ("hipD3D9RegisterResource", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetMappedArray",
            ("hipD3D9ResourceGetMappedArray", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetMappedPitch",
            ("hipD3D9ResourceGetMappedPitch", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetMappedPointer",
            (
                "hipD3D9ResourceGetMappedPointer",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9ResourceGetMappedSize",
            ("hipD3D9ResourceGetMappedSize", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetSurfaceDimensions",
            (
                "hipD3D9ResourceGetSurfaceDimensions",
                CONV_D3D9,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9ResourceSetMapFlags",
            ("hipD3D9ResourceSetMapFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9UnmapResources",
            ("hipD3D9UnmapResources", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9UnregisterResource",
            ("hipD3D9UnregisterResource", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10DeviceListAll",
            ("HIP_D3D10_DEVICE_LIST_ALL", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10DeviceListCurrentFrame",
            (
                "HIP_D3D10_DEVICE_LIST_CURRENT_FRAME",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10DeviceListNextFrame",
            (
                "HIP_D3D10_DEVICE_LIST_NEXT_FRAME",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10GetDevice",
            ("hipD3D10GetDevice", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10GetDevices",
            ("hipD3D10GetDevices", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D10RegisterResource",
            (
                "hipGraphicsD3D10RegisterResource",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10MapFlagsNone",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_NONE",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10MapFlagsReadOnly",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10MapFlagsWriteDiscard",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10RegisterFlagsNone",
            ("HIP_D3D10_REGISTER_FLAGS_NONE", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10RegisterFlagsArray",
            (
                "HIP_D3D10_REGISTER_FLAGS_ARRAY",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10GetDirect3DDevice",
            ("hipD3D10GetDirect3DDevice", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10MapResources",
            ("hipD3D10MapResources", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10RegisterResource",
            ("hipD3D10RegisterResource", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10ResourceGetMappedArray",
            (
                "hipD3D10ResourceGetMappedArray",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceGetMappedPitch",
            (
                "hipD3D10ResourceGetMappedPitch",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceGetMappedPointer",
            (
                "hipD3D10ResourceGetMappedPointer",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceGetMappedSize",
            ("hipD3D10ResourceGetMappedSize", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10ResourceGetSurfaceDimensions",
            (
                "hipD3D10ResourceGetSurfaceDimensions",
                CONV_D3D10,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceSetMapFlags",
            ("hipD3D10ResourceSetMapFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10SetDirect3DDevice",
            ("hipD3D10SetDirect3DDevice", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10UnmapResources",
            ("hipD3D10UnmapResources", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10UnregisterResource",
            ("hipD3D10UnregisterResource", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11DeviceListAll",
            ("HIP_D3D11_DEVICE_LIST_ALL", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11DeviceListCurrentFrame",
            (
                "HIP_D3D11_DEVICE_LIST_CURRENT_FRAME",
                CONV_D3D11,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D11DeviceListNextFrame",
            (
                "HIP_D3D11_DEVICE_LIST_NEXT_FRAME",
                CONV_D3D11,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D11GetDevice",
            ("hipD3D11GetDevice", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11GetDevices",
            ("hipD3D11GetDevices", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D11RegisterResource",
            (
                "hipGraphicsD3D11RegisterResource",
                CONV_D3D11,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D11GetDevice",
            ("hipD3D11GetDevice", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11GetDevices",
            ("hipD3D11GetDevices", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D11RegisterResource",
            (
                "hipGraphicsD3D11RegisterResource",
                CONV_D3D11,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsVDPAURegisterOutputSurface",
            (
                "hipGraphicsVDPAURegisterOutputSurface",
                CONV_VDPAU,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsVDPAURegisterVideoSurface",
            (
                "hipGraphicsVDPAURegisterVideoSurface",
                CONV_VDPAU,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaVDPAUGetDevice",
            ("hipVDPAUGetDevice", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaVDPAUSetVDPAUDevice",
            ("hipVDPAUSetDevice", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamConsumerAcquireFrame",
            (
                "hipEGLStreamConsumerAcquireFrame",
                CONV_EGL,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamConsumerConnect",
            ("hipEGLStreamConsumerConnect", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamConsumerConnectWithFlags",
            (
                "hipEGLStreamConsumerConnectWithFlags",
                CONV_EGL,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamConsumerReleaseFrame",
            (
                "hipEGLStreamConsumerReleaseFrame",
                CONV_EGL,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamProducerConnect",
            ("hipEGLStreamProducerConnect", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamProducerDisconnect",
            ("hipEGLStreamProducerDisconnect", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamProducerPresentFrame",
            (
                "hipEGLStreamProducerPresentFrame",
                CONV_EGL,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamProducerReturnFrame",
            ("hipEGLStreamProducerReturnFrame", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsEGLRegisterImage",
            ("hipGraphicsEGLRegisterImage", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsResourceGetMappedEglFrame",
            (
                "hipGraphicsResourceGetMappedEglFrame",
                CONV_EGL,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cublasInit", ("hipblasInit", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasShutdown",
            ("hipblasShutdown", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasGetVersion",
            ("hipblasGetVersion", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasGetError",
            ("hipblasGetError", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasAlloc", ("hipblasAlloc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasFree", ("hipblasFree", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasSetKernelStream",
            ("hipblasSetKernelStream", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasGetAtomicsMode",
            ("hipblasGetAtomicsMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSetAtomicsMode",
            ("hipblasSetAtomicsMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasGetMathMode",
            ("hipblasGetMathMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSetMathMode",
            ("hipblasSetMathMode", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("CUBLAS_OP_N", ("HIPBLAS_OP_N", CONV_NUMERIC_LITERAL, API_BLAS)),
        (
            "CUBLAS_OP_T",
            ("HIPBLAS_OP_T", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_OP_C",
            ("HIPBLAS_OP_C", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_SUCCESS",
            ("HIPBLAS_STATUS_SUCCESS", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_NOT_INITIALIZED",
            ("HIPBLAS_STATUS_NOT_INITIALIZED", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_ALLOC_FAILED",
            ("HIPBLAS_STATUS_ALLOC_FAILED", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_INVALID_VALUE",
            ("HIPBLAS_STATUS_INVALID_VALUE", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_MAPPING_ERROR",
            ("HIPBLAS_STATUS_MAPPING_ERROR", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_EXECUTION_FAILED",
            ("HIPBLAS_STATUS_EXECUTION_FAILED", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_INTERNAL_ERROR",
            ("HIPBLAS_STATUS_INTERNAL_ERROR", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_NOT_SUPPORTED",
            ("HIPBLAS_STATUS_NOT_SUPPORTED", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_STATUS_ARCH_MISMATCH",
            ("HIPBLAS_STATUS_ARCH_MISMATCH", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_FILL_MODE_LOWER",
            ("HIPBLAS_FILL_MODE_LOWER", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_FILL_MODE_UPPER",
            ("HIPBLAS_FILL_MODE_UPPER", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_DIAG_NON_UNIT",
            ("HIPBLAS_DIAG_NON_UNIT", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        ("CUBLAS_DIAG_UNIT", ("HIPBLAS_DIAG_UNIT", CONV_NUMERIC_LITERAL, API_BLAS)),
        ("CUBLAS_SIDE_LEFT", ("HIPBLAS_SIDE_LEFT", CONV_NUMERIC_LITERAL, API_BLAS)),
        ("CUBLAS_SIDE_RIGHT", ("HIPBLAS_SIDE_RIGHT", CONV_NUMERIC_LITERAL, API_BLAS)),
        (
            "CUBLAS_POINTER_MODE_HOST",
            ("HIPBLAS_POINTER_MODE_HOST", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_POINTER_MODE_DEVICE",
            ("HIPBLAS_POINTER_MODE_DEVICE", CONV_NUMERIC_LITERAL, API_BLAS),
        ),
        (
            "CUBLAS_ATOMICS_NOT_ALLOWED",
            (
                "HIPBLAS_ATOMICS_NOT_ALLOWED",
                CONV_NUMERIC_LITERAL,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_ATOMICS_ALLOWED",
            (
                "HIPBLAS_ATOMICS_ALLOWED",
                CONV_NUMERIC_LITERAL,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_DATA_FLOAT",
            (
                "HIPBLAS_DATA_FLOAT",
                CONV_NUMERIC_LITERAL,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_DATA_DOUBLE",
            (
                "HIPBLAS_DATA_DOUBLE",
                CONV_NUMERIC_LITERAL,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_DATA_HALF",
            ("HIPBLAS_DATA_HALF", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "CUBLAS_DATA_INT8",
            ("HIPBLAS_DATA_INT8", CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("CUBLAS_GEMM_DEFAULT", ("HIPBLAS_GEMM_DEFAULT", CONV_NUMERIC_LITERAL, API_BLAS)),
        ("CUBLAS_GEMM_DEFAULT_TENSOR_OP", ("HIPBLAS_GEMM_DEFAULT", CONV_NUMERIC_LITERAL, API_BLAS)),
        ("cublasCreate", ("hipblasCreate", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDestroy", ("hipblasDestroy", CONV_MATH_FUNC, API_BLAS)),
        ("cublasSetVector", ("hipblasSetVector", CONV_MATH_FUNC, API_BLAS)),
        ("cublasGetVector", ("hipblasGetVector", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasSetVectorAsync",
            ("hipblasSetVectorAsync", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasGetVectorAsync",
            ("hipblasGetVectorAsync", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSetMatrix", ("hipblasSetMatrix", CONV_MATH_FUNC, API_BLAS)),
        ("cublasGetMatrix", ("hipblasGetMatrix", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasGetMatrixAsync",
            ("hipblasGetMatrixAsync", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSetMatrixAsync",
            ("hipblasSetMatrixAsync", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasXerbla", ("hipblasXerbla", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSnrm2", ("hipblasSnrm2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDnrm2", ("hipblasDnrm2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasScnrm2", ("hipblasScnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDznrm2", ("hipblasDznrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasNrm2Ex",
            ("hipblasNrm2Ex", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSdot", ("hipblasSdot", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasSdotBatched",
            ("hipblasSdotBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasDdot", ("hipblasDdot", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasDdotBatched",
            ("hipblasDdotBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasCdotu", ("hipblasCdotu", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCdotc", ("hipblasCdotc", CONV_MATH_FUNC, API_BLAS)),
        ("cublasZdotu", ("hipblasZdotu", CONV_MATH_FUNC, API_BLAS)),
        ("cublasZdotc", ("hipblasZdotc", CONV_MATH_FUNC, API_BLAS)),
        ("cublasSscal", ("hipblasSscal", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasSscalBatched",
            ("hipblasSscalBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasDscal", ("hipblasDscal", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasDscalBatched",
            ("hipblasDscalBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasCscal", ("hipblasCscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsscal", ("hipblasCsscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZscal", ("hipblasZscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZdscal", ("hipblasZdscal", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSaxpy", ("hipblasSaxpy", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasSaxpyBatched",
            ("hipblasSaxpyBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasDaxpy", ("hipblasDaxpy", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCaxpy", ("hipblasCaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZaxpy", ("hipblasZaxpy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasScopy", ("hipblasScopy", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasScopyBatched",
            ("hipblasScopyBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasDcopy", ("hipblasDcopy", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasDcopyBatched",
            ("hipblasDcopyBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasCcopy", ("hipblasCcopy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZcopy", ("hipblasZcopy", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSswap", ("hipblasSswap", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDswap", ("hipblasDswap", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCswap", ("hipblasCswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZswap", ("hipblasZswap", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasIsamax", ("hipblasIsamax", CONV_MATH_FUNC, API_BLAS)),
        ("cublasIdamax", ("hipblasIdamax", CONV_MATH_FUNC, API_BLAS)),
        ("cublasIcamax", ("hipblasIcamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasIzamax", ("hipblasIzamax", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasIsamin", ("hipblasIsamin", CONV_MATH_FUNC, API_BLAS)),
        ("cublasIdamin", ("hipblasIdamin", CONV_MATH_FUNC, API_BLAS)),
        ("cublasIcamin", ("hipblasIcamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasIzamin", ("hipblasIzamin", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSasum", ("hipblasSasum", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasSasumBatched",
            ("hipblasSasumBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasDasum", ("hipblasDasum", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasDasumBatched",
            ("hipblasDasumBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasScasum", ("hipblasScasum", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDzasum", ("hipblasDzasum", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSrot", ("hipblasSrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDrot", ("hipblasDrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCrot", ("hipblasCrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsrot", ("hipblasCsrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZrot", ("hipblasZrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZdrot", ("hipblasZdrot", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSrotg", ("hipblasSrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDrotg", ("hipblasDrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCrotg", ("hipblasCrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZrotg", ("hipblasZrotg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSrotm", ("hipblasSrotm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDrotm", ("hipblasDrotm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSrotmg", ("hipblasSrotmg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDrotmg", ("hipblasDrotmg", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSgemv", ("hipblasSgemv", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasSgemvBatched",
            ("hipblasSgemvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasDgemv", ("hipblasDgemv", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCgemv", ("hipblasCgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZgemv", ("hipblasZgemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSgbmv", ("hipblasSgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDgbmv", ("hipblasDgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCgbmv", ("hipblasCgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZgbmv", ("hipblasZgbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStrmv", ("hipblasStrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtrmv", ("hipblasDtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtrmv", ("hipblasCtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtrmv", ("hipblasZtrmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStbmv", ("hipblasStbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtbmv", ("hipblasDtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtbmv", ("hipblasCtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtbmv", ("hipblasZtbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStpmv", ("hipblasStpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtpmv", ("hipblasDtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtpmv", ("hipblasCtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtpmv", ("hipblasZtpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStrsv", ("hipblasStrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtrsv", ("hipblasDtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtrsv", ("hipblasCtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtrsv", ("hipblasZtrsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStpsv", ("hipblasStpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtpsv", ("hipblasDtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtpsv", ("hipblasCtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtpsv", ("hipblasZtpsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStbsv", ("hipblasStbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtbsv", ("hipblasDtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtbsv", ("hipblasCtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtbsv", ("hipblasZtbsv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSsymv", ("hipblasSsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsymv", ("hipblasDsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsymv", ("hipblasCsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZsymv", ("hipblasZsymv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasChemv", ("hipblasChemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZhemv", ("hipblasZhemv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSsbmv", ("hipblasSsbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsbmv", ("hipblasDsbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasChbmv", ("hipblasChbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZhbmv", ("hipblasZhbmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSspmv", ("hipblasSspmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDspmv", ("hipblasDspmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasChpmv", ("hipblasChpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZhpmv", ("hipblasZhpmv", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSger", ("hipblasSger", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDger", ("hipblasDger", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCgeru", ("hipblasCgeru", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCgerc", ("hipblasCgerc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZgeru", ("hipblasZgeru", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZgerc", ("hipblasZgerc", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSsyr", ("hipblasSsyr", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDsyr", ("hipblasDsyr", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCher", ("hipblasCher", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZher", ("hipblasZher", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSspr", ("hipblasSspr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDspr", ("hipblasDspr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasChpr", ("hipblasChpr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZhpr", ("hipblasZhpr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSsyr2", ("hipblasSsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsyr2", ("hipblasDsyr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCher2", ("hipblasCher2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZher2", ("hipblasZher2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSspr2", ("hipblasSspr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDspr2", ("hipblasDspr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasChpr2", ("hipblasChpr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZhpr2", ("hipblasZhpr2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasSgemmBatched",
            ("hipblasSgemmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDgemmBatched",
            ("hipblasDgemmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasHgemmBatched",
            ("hipblasHgemmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSgemmStridedBatched",
            ("hipblasSgemmStridedBatched", CONV_MATH_FUNC, API_BLAS),
        ),
        (
            "cublasDgemmStridedBatched",
            ("hipblasDgemmStridedBatched", CONV_MATH_FUNC, API_BLAS),
        ),
        (
            "cublasHgemmStridedBatched",
            ("hipblasHgemmStridedBatched", CONV_MATH_FUNC, API_BLAS),
        ),
        (
            "cublasCgemmBatched",
            ("hipblasCgemmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemm3mBatched",
            ("hipblasCgemm3mBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemmBatched",
            ("hipblasZgemmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemmStridedBatched",
            (
                "hipblasCgemmStridedBatched",
                CONV_MATH_FUNC,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cublasCgemm3mStridedBatched",
            (
                "hipblasCgemm3mStridedBatched",
                CONV_MATH_FUNC,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cublasZgemmStridedBatched",
            (
                "hipblasZgemmStridedBatched",
                CONV_MATH_FUNC,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cublasHgemmStridedBatched",
            (
                "hipblasHgemmStridedBatched",
                CONV_MATH_FUNC,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cublasSgemm", ("hipblasSgemm", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDgemm", ("hipblasDgemm", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCgemm", ("hipblasCgemm", CONV_MATH_FUNC, API_BLAS)),
        ("cublasZgemm", ("hipblasZgemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasHgemm", ("hipblasHgemm", CONV_MATH_FUNC, API_BLAS)),
        ("cublasSsyrk", ("hipblasSsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsyrk", ("hipblasDsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsyrk", ("hipblasCsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZsyrk", ("hipblasZsyrk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCherk", ("hipblasCherk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZherk", ("hipblasZherk", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSsyr2k", ("hipblasSsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsyr2k", ("hipblasDsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsyr2k", ("hipblasCsyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZsyr2k", ("hipblasZyr2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSsyrkx", ("hipblasSsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsyrkx", ("hipblasDsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsyrkx", ("hipblasCsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZsyrkx", ("hipblasZsyrkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCher2k", ("hipblasCher2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZher2k", ("hipblasZher2k", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCherkx", ("hipblasCherkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZherkx", ("hipblasZherkx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSsymm", ("hipblasSsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsymm", ("hipblasDsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsymm", ("hipblasCsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZsymm", ("hipblasZsymm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasChemm", ("hipblasChemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZhemm", ("hipblasZhemm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStrsm", ("hipblasStrsm", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDtrsm", ("hipblasDtrsm", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCtrsm", ("hipblasCtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtrsm", ("hipblasZtrsm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasStrsmBatched",
            ("hipblasStrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsmBatched",
            ("hipblasDtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsmBatched",
            ("hipblasCtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsmBatched",
            ("hipblasZtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasStrmm", ("hipblasStrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtrmm", ("hipblasDtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtrmm", ("hipblasCtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtrmm", ("hipblasZtrmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSgeam", ("hipblasSgeam", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDgeam", ("hipblasDgeam", CONV_MATH_FUNC, API_BLAS)),
        ("cublasCgeam", ("hipblasCgeam", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZgeam", ("hipblasZgeam", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasSgetrfBatched",
            ("hipblasSgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDgetrfBatched",
            ("hipblasDgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgetrfBatched",
            ("hipblasCgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgetrfBatched",
            ("hipblasZgetrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSgetriBatched",
            ("hipblasSgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDgetriBatched",
            ("hipblasDgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgetriBatched",
            ("hipblasCgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgetriBatched",
            ("hipblasZgetriBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSgetrsBatched",
            ("hipblasSgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDgetrsBatched",
            ("hipblasDgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgetrsBatched",
            ("hipblasCgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgetrsBatched",
            ("hipblasZgetrsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStrsmBatched",
            ("hipblasStrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsmBatched",
            ("hipblasDtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsmBatched",
            ("hipblasCtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsmBatched",
            ("hipblasZtrsmBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSmatinvBatched",
            ("hipblasSmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDmatinvBatched",
            ("hipblasDmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCmatinvBatched",
            ("hipblasCmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZmatinvBatched",
            ("hipblasZmatinvBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSgeqrfBatched",
            ("hipblasSgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDgeqrfBatched",
            ("hipblasDgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgeqrfBatched",
            ("hipblasCgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgeqrfBatched",
            ("hipblasZgeqrfBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSgelsBatched",
            ("hipblasSgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDgelsBatched",
            ("hipblasDgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgelsBatched",
            ("hipblasCgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgelsBatched",
            ("hipblasZgelsBatched", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSdgmm", ("hipblasSdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDdgmm", ("hipblasDdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCdgmm", ("hipblasCdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZdgmm", ("hipblasZdgmm", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStpttr", ("hipblasStpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtpttr", ("hipblasDtpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtpttr", ("hipblasCtpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtpttr", ("hipblasZtpttr", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasStrttp", ("hipblasStrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDtrttp", ("hipblasDtrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCtrttp", ("hipblasCtrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZtrttp", ("hipblasZtrttp", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCreate_v2", ("hipblasCreate_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDestroy_v2", ("hipblasDestroy_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasGetVersion_v2",
            ("hipblasGetVersion_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSetStream", ("hipblasSetStream", CONV_MATH_FUNC, API_BLAS)),
        ("cublasGetStream", ("hipblasGetStream", CONV_MATH_FUNC, API_BLAS)),
        ("cublasSetStream_v2", ("hipblasSetStream_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasGetStream_v2", ("hipblasGetStream_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasGetPointerMode",
            ("hipblasGetPointerMode", CONV_MATH_FUNC, API_BLAS),
        ),
        (
            "cublasSetPointerMode",
            ("hipblasSetPointerMode", CONV_MATH_FUNC, API_BLAS),
        ),
        (
            "cublasGetPointerMode_v2",
            ("hipblasGetPointerMode_v2", CONV_MATH_FUNC, API_BLAS),
        ),
        (
            "cublasSetPointerMode_v2",
            ("hipblasSetPointerMode_v2", CONV_MATH_FUNC, API_BLAS),
        ),
        ("cublasSgemv_v2", ("hipblasSgemv_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDgemv_v2", ("hipblasDgemv_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCgemv_v2",
            ("hipblasCgemv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemv_v2",
            ("hipblasZgemv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSgbmv_v2",
            ("hipblasSgbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDgbmv_v2",
            ("hipblasDgbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgbmv_v2",
            ("hipblasCgbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgbmv_v2",
            ("hipblasZgbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStrmv_v2",
            ("hipblasStrmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrmv_v2",
            ("hipblasDtrmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrmv_v2",
            ("hipblasCtrmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrmv_v2",
            ("hipblasZtrmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStbmv_v2",
            ("hipblasStbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtbmv_v2",
            ("hipblasDtbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtbmv_v2",
            ("hipblasCtbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtbmv_v2",
            ("hipblasZtbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStpmv_v2",
            ("hipblasStpmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtpmv_v2",
            ("hipblasDtpmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtpmv_v2",
            ("hipblasCtpmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtpmv_v2",
            ("hipblasZtpmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStrsv_v2",
            ("hipblasStrsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsv_v2",
            ("hipblasDtrsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsv_v2",
            ("hipblasCtrsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsv_v2",
            ("hipblasZtrsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStpsv_v2",
            ("hipblasStpsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtpsv_v2",
            ("hipblasDtpsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtpsv_v2",
            ("hipblasCtpsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtpsv_v2",
            ("hipblasZtpsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStbsv_v2",
            ("hipblasStbsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtbsv_v2",
            ("hipblasDtbsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtbsv_v2",
            ("hipblasCtbsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtbsv_v2",
            ("hipblasZtbsv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSsymv_v2",
            ("hipblasSsymv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDsymv_v2",
            ("hipblasDsymv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsymv_v2",
            ("hipblasCsymv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZsymv_v2",
            ("hipblasZsymv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasChemv_v2",
            ("hipblasChemv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZhemv_v2",
            ("hipblasZhemv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSsbmv_v2",
            ("hipblasSsbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDsbmv_v2",
            ("hipblasDsbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasChbmv_v2",
            ("hipblasChbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZhbmv_v2",
            ("hipblasZhbmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSspmv_v2",
            ("hipblasSspmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDspmv_v2",
            ("hipblasDspmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasChpmv_v2",
            ("hipblasChpmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZhpmv_v2",
            ("hipblasZhpmv_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSger_v2", ("hipblasSger_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDger_v2", ("hipblasDger_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCgeru_v2",
            ("hipblasCgeru_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgerc_v2",
            ("hipblasCergc_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgeru_v2",
            ("hipblasZgeru_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgerc_v2",
            ("hipblasZgerc_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSsyr_v2", ("hipblasSsyr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDsyr_v2", ("hipblasDsyr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCsyr_v2", ("hipblasCsyr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZsyr_v2", ("hipblasZsyr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCher_v2", ("hipblasCher_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZher_v2", ("hipblasZher_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSspr_v2", ("hipblasSspr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDspr_v2", ("hipblasDspr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasChpr_v2", ("hipblasChpr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasZhpr_v2", ("hipblasZhpr_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasSsyr2_v2",
            ("hipblasSsyr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDsyr2_v2",
            ("hipblasDsyr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyr2_v2",
            ("hipblasCsyr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZsyr2_v2",
            ("hipblasZsyr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCher2_v2",
            ("hipblasCher2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZher2_v2",
            ("hipblasZher2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSspr2_v2",
            ("hipblasSspr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDspr2_v2",
            ("hipblasDspr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasChpr2_v2",
            ("hipblasChpr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZhpr2_v2",
            ("hipblasZhpr2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSgemm_v2", ("hipblasSgemm_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDgemm_v2", ("hipblasDgemm_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCgemm_v2",
            ("hipblasCgemm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemm3m",
            ("hipblasCgemm3m", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemm3mEx",
            ("hipblasCgemm3mEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemm_v2",
            ("hipblasZgemm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemm3m",
            ("hipblasZgemm3m", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSgemmEx",
            ("hipblasSgemmEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasGemmEx", ("hipblasGemmEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasGemmBatchedEx",
            ("hipblasGemmBatchedEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasGemmStridedBatchedEx",
            ("hipblasGemmStridedBatchedEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemmEx",
            ("hipblasCgemmEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasUint8gemmBias",
            ("hipblasUint8gemmBias", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSsyrk_v2",
            ("hipblasSsyrk_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDsyrk_v2",
            ("hipblasDsyrk_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyrk_v2",
            ("hipblasCsyrk_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZsyrk_v2",
            ("hipblasZsyrk_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyrkEx",
            ("hipblasCsyrkEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyrk3mEx",
            ("hipblasCsyrk3mEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCherk_v2",
            ("hipblasCherk_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCherkEx",
            ("hipblasCherkEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCherk3mEx",
            ("hipblasCherk3mEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZherk_v2",
            ("hipblasZherk_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSsyr2k_v2",
            ("hipblasSsyr2k_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDsyr2k_v2",
            ("hipblasDsyr2k_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyr2k_v2",
            ("hipblasCsyr2k_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZsyr2k_v2",
            ("hipblasZsyr2k_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCher2k_v2",
            ("hipblasCher2k_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZher2k_v2",
            ("hipblasZher2k_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSsymm_v2",
            ("hipblasSsymm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDsymm_v2",
            ("hipblasDsymm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsymm_v2",
            ("hipblasCsymm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZsymm_v2",
            ("hipblasZsymm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasChemm_v2",
            ("hipblasChemm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZhemm_v2",
            ("hipblasZhemm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStrsm_v2",
            ("hipblasStrsm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsm_v2",
            ("hipblasDtrsm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsm_v2",
            ("hipblasCtrsm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsm_v2",
            ("hipblasZtrsm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasStrmm_v2",
            ("hipblasStrmm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrmm_v2",
            ("hipblasDtrmm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrmm_v2",
            ("hipblasCtrmm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrmm_v2",
            ("hipblasZtrmm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSnrm2_v2", ("hipblasSnrm2_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDnrm2_v2", ("hipblasDnrm2_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasScnrm2_v2",
            ("hipblasScnrm2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDznrm2_v2",
            ("hipblasDznrm2_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasDotEx", ("hipblasDotEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDotcEx", ("hipblasDotcEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSdot_v2", ("hipblasSdot_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDdot_v2", ("hipblasDdot_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCdotu_v2",
            ("hipblasCdotu_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCdotc_v2",
            ("hipblasCdotc_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZdotu_v2",
            ("hipblasZdotu_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZdotc_v2",
            ("hipblasZdotc_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasScalEx", ("hipblasScalEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSscal_v2", ("hipblasSscal_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDscal_v2", ("hipblasDscal_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCscal_v2",
            ("hipblasCscal_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCsscal_v2",
            ("hipblasCsscal_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZscal_v2",
            ("hipblasZcsal_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZdscal_v2",
            ("hipblasZdscal_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasAxpyEx", ("hipblasAxpyEx", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasSaxpy_v2", ("hipblasSaxpy_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDaxpy_v2", ("hipblasDaxpy_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCaxpy_v2",
            ("hipblasCaxpy_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZaxpy_v2",
            ("hipblasZaxpy_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasScopy_v2", ("hipblasScopy_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDcopy_v2", ("hipblasDcopy_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCcopy_v2",
            ("hipblasCcopy_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZcopy_v2",
            ("hipblasZcopy_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSswap_v2", ("hipblasSswap_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDswap_v2", ("hipblasDswap_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasCswap_v2",
            ("hipblasCswap_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZswap_v2",
            ("hipblasZswap_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasIsamax_v2", ("hipblasIsamax_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasIdamax_v2", ("hipblasIdamax_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasIcamax_v2",
            ("hipblasIcamax_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasIzamax_v2",
            ("hipblasIzamax_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasIsamin_v2", ("hipblasIsamin_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasIdamin_v2", ("hipblasIdamin_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasIcamin_v2",
            ("hipblasIcamin_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasIzamin_v2",
            ("hipblasIzamin_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSasum_v2", ("hipblasSasum_v2", CONV_MATH_FUNC, API_BLAS)),
        ("cublasDasum_v2", ("hipblasDasum_v2", CONV_MATH_FUNC, API_BLAS)),
        (
            "cublasScasum_v2",
            ("hipblasScasum_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDzasum_v2",
            ("hipblasDzasum_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasSrot_v2", ("hipblasSrot_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasDrot_v2", ("hipblasDrot_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        ("cublasCrot_v2", ("hipblasCrot_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasCsrot_v2",
            ("hipblasCsrot_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("cublasZrot_v2", ("hipblasZrot_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED)),
        (
            "cublasZdrot_v2",
            ("hipblasZdrot_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSrotg_v2",
            ("hipblasSrotg_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDrotg_v2",
            ("hipblasDrotg_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasCrotg_v2",
            ("hipblasCrotg_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasZrotg_v2",
            ("hipblasZrotg_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSrotm_v2",
            ("hipblasSrotm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDrotm_v2",
            ("hipblasDrotm_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasSrotmg_v2",
            ("hipblasSrotmg_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDrotmg_v2",
            ("hipblasDrotmg_v2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasComputeType_t",
            ("hipblasComputeType_t", CONV_MATH_FUNC, API_BLAS)
        ),
        (
            "CUBLAS_COMPUTE_32I",
            ("HIPBLAS_COMPUTE_32I", CONV_MATH_FUNC, API_BLAS)
        ),
        (
            "CUBLAS_COMPUTE_32F",
            ("HIPBLAS_COMPUTE_32F", CONV_MATH_FUNC, API_BLAS)
        ),
        (
            "CUBLAS_COMPUTE_64F",
            ("HIPBLAS_COMPUTE_64F", CONV_MATH_FUNC, API_BLAS)
        ),
        ("cublasLtEpilogue_t", ("hipblasLtEpilogue_t", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_EPILOGUE_DEFAULT", ("HIPBLASLT_EPILOGUE_DEFAULT", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_EPILOGUE_RELU", ("HIPBLASLT_EPILOGUE_RELU", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_EPILOGUE_BIAS", ("HIPBLASLT_EPILOGUE_BIAS", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_EPILOGUE_RELU_BIAS", ("HIPBLASLT_EPILOGUE_RELU_BIAS", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_EPILOGUE_GELU", ("HIPBLASLT_EPILOGUE_GELU", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_EPILOGUE_GELU_BIAS", ("HIPBLASLT_EPILOGUE_GELU_BIAS", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtHandle_t", ("hipblasLtHandle_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulDesc_t", ("hipblasLtMatmulDesc_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulDescOpaque_t", ("hipblasLtMatmulDescOpaque_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulDescAttributes_t", ("hipblasLtMatmulDescAttributes_t", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_TRANSA", ("HIPBLASLT_MATMUL_DESC_TRANSA", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_TRANSB", ("HIPBLASLT_MATMUL_DESC_TRANSB", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_EPILOGUE", ("HIPBLASLT_MATMUL_DESC_EPILOGUE", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_BIAS_POINTER", ("HIPBLASLT_MATMUL_DESC_BIAS_POINTER", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_A_SCALE_POINTER", ("HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_B_SCALE_POINTER", ("HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_D_SCALE_POINTER", ("HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_AMAX_D_POINTER", ("HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE", ("HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatrixLayout_t", ("hipblasLtMatrixLayout_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatrixLayoutOpaque_t", ("hipblasLtMatrixLayoutOpaque_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatrixLayoutAttribute_t", ("hipblasLtMatrixLayoutAttribute_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatrixLayoutCreate", ("hipblasLtMatrixLayoutCreate", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatrixLayoutDestroy", ("hipblasLtMatrixLayoutDestroy", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatrixLayoutSetAttribute", ("hipblasLtMatrixLayoutSetAttribute", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT", ("HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET", ("HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulPreference_t", ("hipblasLtMatmulPreference_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulPreferenceOpaque_t", ("hipblasLtMatmulPreferenceOpaque_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulPreferenceAttributes_t", ("hipblasLtMatmulPreferenceAttributes_t", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_PREF_SEARCH_MODE", ("HIPBLASLT_MATMUL_PREF_SEARCH_MODE", CONV_MATH_FUNC, API_BLAS)),
        ("CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES", ("HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulAlgo_t", ("hipblasLtMatmulAlgo_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulHeuristicResult_t", ("hipblasLtMatmulHeuristicResult_t", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtCreate", ("hipblasLtCreate", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtDestroy", ("hipblasLtDestroy", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulDescCreate", ("hipblasLtMatmulDescCreate", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulDescDestroy", ("hipblasLtMatmulDescDestroy", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulDescSetAttribute", ("hipblasLtMatmulDescSetAttribute", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulPreferenceCreate", ("hipblasLtMatmulPreferenceCreate", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulPreferenceDestroy", ("hipblasLtMatmulPreferenceDestroy", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulPreferenceSetAttribute", ("hipblasLtMatmulPreferenceSetAttribute", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmulAlgoGetHeuristic", ("hipblasLtMatmulAlgoGetHeuristic", CONV_MATH_FUNC, API_BLAS)),
        ("cublasLtMatmul", ("hipblasLtMatmul", CONV_MATH_FUNC, API_BLAS)),
        (
            "CURAND_STATUS_SUCCESS",
            ("HIPRAND_STATUS_SUCCESS", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_VERSION_MISMATCH",
            ("HIPRAND_STATUS_VERSION_MISMATCH", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_NOT_INITIALIZED",
            ("HIPRAND_STATUS_NOT_INITIALIZED", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_ALLOCATION_FAILED",
            ("HIPRAND_STATUS_ALLOCATION_FAILED", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_TYPE_ERROR",
            ("HIPRAND_STATUS_TYPE_ERROR", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_OUT_OF_RANGE",
            ("HIPRAND_STATUS_OUT_OF_RANGE", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
            ("HIPRAND_STATUS_LENGTH_NOT_MULTIPLE", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
            (
                "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED",
                CONV_NUMERIC_LITERAL,
                API_RAND,
            ),
        ),
        (
            "CURAND_STATUS_LAUNCH_FAILURE",
            ("HIPRAND_STATUS_LAUNCH_FAILURE", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_PREEXISTING_FAILURE",
            ("HIPRAND_STATUS_PREEXISTING_FAILURE", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_INITIALIZATION_FAILED",
            ("HIPRAND_STATUS_INITIALIZATION_FAILED", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_ARCH_MISMATCH",
            ("HIPRAND_STATUS_ARCH_MISMATCH", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_STATUS_INTERNAL_ERROR",
            ("HIPRAND_STATUS_INTERNAL_ERROR", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        ("CURAND_RNG_TEST", ("HIPRAND_RNG_TEST", CONV_NUMERIC_LITERAL, API_RAND)),
        (
            "mtgp32dc_params_fast_11213",
            ("mtgp32dc_params_fast_11213", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_DEFAULT",
            ("HIPRAND_RNG_PSEUDO_DEFAULT", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_XORWOW",
            ("HIPRAND_RNG_PSEUDO_XORWOW", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_MRG32K3A",
            ("HIPRAND_RNG_PSEUDO_MRG32K3A", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_MTGP32",
            ("HIPRAND_RNG_PSEUDO_MTGP32", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_MT19937",
            ("HIPRAND_RNG_PSEUDO_MT19937", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_PHILOX4_32_10",
            ("HIPRAND_RNG_PSEUDO_PHILOX4_32_10", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_DEFAULT",
            ("HIPRAND_RNG_QUASI_DEFAULT", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SOBOL32",
            ("HIPRAND_RNG_QUASI_SOBOL32", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",
            ("HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SOBOL64",
            ("HIPRAND_RNG_QUASI_SOBOL64", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",
            ("HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64", CONV_NUMERIC_LITERAL, API_RAND),
        ),
        (
            "curand_ORDERING_PSEUDO_BEST",
            (
                "HIPRAND_ORDERING_PSEUDO_BEST",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_ORDERING_PSEUDO_DEFAULT",
            (
                "HIPRAND_ORDERING_PSEUDO_DEFAULT",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_ORDERING_PSEUDO_SEEDED",
            (
                "HIPRAND_ORDERING_PSEUDO_SEEDED",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_ORDERING_QUASI_DEFAULT",
            (
                "HIPRAND_ORDERING_QUASI_DEFAULT",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_DIRECTION_VECTORS_32_JOEKUO6",
            (
                "HIPRAND_DIRECTION_VECTORS_32_JOEKUO6",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
            (
                "HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_DIRECTION_VECTORS_64_JOEKUO6",
            (
                "HIPRAND_DIRECTION_VECTORS_64_JOEKUO6",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
            (
                "HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
                CONV_NUMERIC_LITERAL,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_CHOOSE_BEST",
            ("HIPRAND_CHOOSE_BEST", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_ITR",
            ("HIPRAND_ITR", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_KNUTH",
            ("HIPRAND_KNUTH", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_HITR",
            ("HIPRAND_HITR", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curand_M1", ("HIPRAND_M1", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED)),
        ("curand_M2", ("HIPRAND_M2", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED)),
        (
            "curand_BINARY_SEARCH",
            ("HIPRAND_BINARY_SEARCH", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_DISCRETE_GAUSS",
            ("HIPRAND_DISCRETE_GAUSS", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_REJECTION",
            ("HIPRAND_REJECTION", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_DEVICE_API",
            ("HIPRAND_DEVICE_API", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_FAST_REJECTION",
            ("HIPRAND_FAST_REJECTION", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_3RD",
            ("HIPRAND_3RD", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_DEFINITION",
            ("HIPRAND_DEFINITION", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_POISSON",
            ("HIPRAND_POISSON", CONV_NUMERIC_LITERAL, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandCreateGenerator", ("hiprandCreateGenerator", CONV_MATH_FUNC, API_RAND)),
        (
            "curandCreateGeneratorHost",
            ("hiprandCreateGeneratorHost", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandCreatePoissonDistribution",
            ("hiprandCreatePoissonDistribution", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandDestroyDistribution",
            ("hiprandDestroyDistribution", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandDestroyGenerator",
            ("hiprandDestroyGenerator", CONV_MATH_FUNC, API_RAND),
        ),
        ("curandGenerate", ("hiprandGenerate", CONV_MATH_FUNC, API_RAND)),
        (
            "curandGenerateLogNormal",
            ("hiprandGenerateLogNormal", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandGenerateLogNormalDouble",
            ("hiprandGenerateLogNormalDouble", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandGenerateLongLong",
            ("hiprandGenerateLongLong", CONV_MATH_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandGenerateNormal", ("hiprandGenerateNormal", CONV_MATH_FUNC, API_RAND)),
        (
            "curandGenerateNormalDouble",
            ("hiprandGenerateNormalDouble", CONV_MATH_FUNC, API_RAND),
        ),
        ("curandGeneratePoisson", ("hiprandGeneratePoisson", CONV_MATH_FUNC, API_RAND)),
        ("curandGenerateSeeds", ("hiprandGenerateSeeds", CONV_MATH_FUNC, API_RAND)),
        ("curandGenerateUniform", ("hiprandGenerateUniform", CONV_MATH_FUNC, API_RAND)),
        (
            "curandGenerateUniformDouble",
            ("hiprandGenerateUniformDouble", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandGetDirectionVectors32",
            ("hiprandGetDirectionVectors32", CONV_MATH_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandGetDirectionVectors64",
            ("hiprandGetDirectionVectors64", CONV_MATH_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandGetProperty",
            ("hiprandGetProperty", CONV_MATH_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandGetScrambleConstants32",
            (
                "hiprandGetScrambleConstants32",
                CONV_MATH_FUNC,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curandGetScrambleConstants64",
            (
                "hiprandGetScrambleConstants64",
                CONV_MATH_FUNC,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        ("curandGetVersion", ("hiprandGetVersion", CONV_MATH_FUNC, API_RAND)),
        (
            "curandSetGeneratorOffset",
            ("hiprandSetGeneratorOffset", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandSetGeneratorOrdering",
            ("hiprandSetGeneratorOrdering", CONV_MATH_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandSetPseudoRandomGeneratorSeed",
            ("hiprandSetPseudoRandomGeneratorSeed", CONV_MATH_FUNC, API_RAND),
        ),
        (
            "curandSetQuasiRandomGeneratorDimensions",
            ("hiprandSetQuasiRandomGeneratorDimensions", CONV_MATH_FUNC, API_RAND),
        ),
        ("curandSetStream", ("hiprandSetStream", CONV_MATH_FUNC, API_RAND)),
        ("curand", ("hiprand", CONV_DEVICE_FUNC, API_RAND)),
        ("curand4", ("hiprand4", CONV_DEVICE_FUNC, API_RAND)),
        ("curand_init", ("hiprand_init", CONV_DEVICE_FUNC, API_RAND)),
        ("curand_log_normal", ("hiprand_log_normal", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_log_normal_double",
            ("hiprand_log_normal_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        ("curand_log_normal2", ("hiprand_log_normal2", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_log_normal2_double",
            ("hiprand_log_normal2_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        ("curand_log_normal4", ("hiprand_log_normal4", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_log_normal4_double",
            ("hiprand_log_normal4_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        (
            "curand_mtgp32_single",
            ("hiprand_mtgp32_single", CONV_DEVICE_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curand_mtgp32_single_specific",
            (
                "hiprand_mtgp32_single_specific",
                CONV_DEVICE_FUNC,
                API_RAND,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_mtgp32_specific",
            ("hiprand_mtgp32_specific", CONV_DEVICE_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curand_normal", ("hiprand_normal", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curandMakeMTGP32Constants",
            ("hiprandMakeMTGP32Constants", CONV_DEVICE_FUNC, API_RAND),
        ),
        (
            "curandMakeMTGP32KernelState",
            ("hiprandMakeMTGP32KernelState", CONV_DEVICE_FUNC, API_RAND),
        ),
        ("curand_normal_double", ("hiprand_normal_double", CONV_DEVICE_FUNC, API_RAND)),
        ("curand_normal2", ("hiprand_normal2", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_normal2_double",
            ("hiprand_normal2_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        ("curand_normal4", ("hiprand_normal4", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_normal4_double",
            ("hiprand_normal4_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        ("curand_uniform", ("hiprand_uniform", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_uniform_double",
            ("hiprand_uniform_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        (
            "curand_uniform2_double",
            ("hiprand_uniform2_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        ("curand_uniform4", ("hiprand_uniform4", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_uniform4_double",
            ("hiprand_uniform4_double", CONV_DEVICE_FUNC, API_RAND),
        ),
        ("curand_discrete", ("hiprand_discrete", CONV_DEVICE_FUNC, API_RAND)),
        ("curand_discrete4", ("hiprand_discrete4", CONV_DEVICE_FUNC, API_RAND)),
        ("curand_poisson", ("hiprand_poisson", CONV_DEVICE_FUNC, API_RAND)),
        ("curand_poisson4", ("hiprand_poisson4", CONV_DEVICE_FUNC, API_RAND)),
        (
            "curand_Philox4x32_10",
            ("hiprand_Philox4x32_10", CONV_DEVICE_FUNC, API_RAND, HIP_UNSUPPORTED),
        ),
        ("mtgp32_kernel_params", ("mtgp32_kernel_params_t", CONV_MATH_FUNC, API_RAND)),
        ("CUFFT_FORWARD", ("HIPFFT_FORWARD", CONV_NUMERIC_LITERAL, API_BLAS)),
        ("CUFFT_INVERSE", ("HIPFFT_BACKWARD", CONV_NUMERIC_LITERAL, API_BLAS)),
        (
            "CUFFT_COMPATIBILITY_DEFAULT",
            (
                "HIPFFT_COMPATIBILITY_DEFAULT",
                CONV_NUMERIC_LITERAL,
                API_BLAS,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cuComplex", ("hipComplex", CONV_TYPE, API_BLAS)),
        ("cuDoubleComplex", ("hipDoubleComplex", CONV_TYPE, API_BLAS)),
        ("cufftResult_t", ("hipfftResult_t", CONV_TYPE, API_FFT)),
        ("cufftResult", ("hipfftResult", CONV_TYPE, API_FFT)),
        ("CUFFT_SUCCESS", ("HIPFFT_SUCCESS", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_INVALID_PLAN", ("HIPFFT_INVALID_PLAN", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_ALLOC_FAILED", ("HIPFFT_ALLOC_FAILED", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_INVALID_TYPE", ("HIPFFT_INVALID_TYPE", CONV_NUMERIC_LITERAL, API_FFT)),
        (
            "CUFFT_INVALID_VALUE",
            ("HIPFFT_INVALID_VALUE", CONV_NUMERIC_LITERAL, API_FFT),
        ),
        (
            "CUFFT_INTERNAL_ERROR",
            ("HIPFFT_INTERNAL_ERROR", CONV_NUMERIC_LITERAL, API_FFT),
        ),
        ("CUFFT_EXEC_FAILED", ("HIPFFT_EXEC_FAILED", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_SETUP_FAILED", ("HIPFFT_SETUP_FAILED", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_INVALID_SIZE", ("HIPFFT_INVALID_SIZE", CONV_NUMERIC_LITERAL, API_FFT)),
        (
            "CUFFT_UNALIGNED_DATA",
            ("HIPFFT_UNALIGNED_DATA", CONV_NUMERIC_LITERAL, API_FFT),
        ),
        (
            "CUFFT_INCOMPLETE_PARAMETER_LIST",
            ("HIPFFT_INCOMPLETE_PARAMETER_LIST", CONV_NUMERIC_LITERAL, API_FFT),
        ),
        (
            "CUFFT_INVALID_DEVICE",
            ("HIPFFT_INVALID_DEVICE", CONV_NUMERIC_LITERAL, API_FFT),
        ),
        ("CUFFT_PARSE_ERROR", ("HIPFFT_PARSE_ERROR", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_NO_WORKSPACE", ("HIPFFT_NO_WORKSPACE", CONV_NUMERIC_LITERAL, API_FFT)),
        (
            "CUFFT_NOT_IMPLEMENTED",
            ("HIPFFT_NOT_IMPLEMENTED", CONV_NUMERIC_LITERAL, API_FFT),
        ),
        (
            "CUFFT_LICENSE_ERROR",
            ("HIPFFT_LICENSE_ERROR", CONV_NUMERIC_LITERAL, API_FFT, HIP_UNSUPPORTED),
        ),
        (
            "CUFFT_NOT_SUPPORTED",
            ("HIPFFT_NOT_SUPPORTED", CONV_NUMERIC_LITERAL, API_FFT),
        ),
        ("cufftType_t", ("hipfftType_t", CONV_TYPE, API_FFT)),
        ("cufftType", ("hipfftType", CONV_TYPE, API_FFT)),
        ("CUFFT_R2C", ("HIPFFT_R2C", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_C2R", ("HIPFFT_C2R", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_C2C", ("HIPFFT_C2C", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_D2Z", ("HIPFFT_D2Z", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_Z2D", ("HIPFFT_Z2D", CONV_NUMERIC_LITERAL, API_FFT)),
        ("CUFFT_Z2Z", ("HIPFFT_Z2Z", CONV_NUMERIC_LITERAL, API_FFT)),
        (
            "cufftCompatibility_t",
            ("hipfftCompatibility_t", CONV_TYPE, API_FFT, HIP_UNSUPPORTED),
        ),
        (
            "cufftCompatibility",
            ("hipfftCompatibility", CONV_TYPE, API_FFT, HIP_UNSUPPORTED),
        ),
        (
            "CUFFT_COMPATIBILITY_FFTW_PADDING",
            (
                "HIPFFT_COMPATIBILITY_FFTW_PADDING",
                CONV_NUMERIC_LITERAL,
                API_FFT,
                HIP_UNSUPPORTED,
            ),
        ),
        ("cufftReal", ("hipfftReal", CONV_TYPE, API_FFT)),
        ("cufftDoubleReal", ("hipfftDoubleReal", CONV_TYPE, API_FFT)),
        ("cufftComplex", ("hipfftComplex", CONV_TYPE, API_FFT)),
        ("cufftDoubleComplex", ("hipfftDoubleComplex", CONV_TYPE, API_FFT)),
        ("cufftHandle", ("hipfftHandle", CONV_TYPE, API_FFT)),
        ("cufftPlan1d", ("hipfftPlan1d", CONV_MATH_FUNC, API_FFT)),
        ("cufftPlan2d", ("hipfftPlan2d", CONV_MATH_FUNC, API_FFT)),
        ("cufftPlan3d", ("hipfftPlan3d", CONV_MATH_FUNC, API_FFT)),
        ("cufftPlanMany", ("hipfftPlanMany", CONV_MATH_FUNC, API_FFT)),
        ("cufftMakePlan1d", ("hipfftMakePlan1d", CONV_MATH_FUNC, API_FFT)),
        ("cufftMakePlan2d", ("hipfftMakePlan2d", CONV_MATH_FUNC, API_FFT)),
        ("cufftMakePlan3d", ("hipfftMakePlan3d", CONV_MATH_FUNC, API_FFT)),
        ("cufftMakePlanMany", ("hipfftMakePlanMany", CONV_MATH_FUNC, API_FFT)),
        ("cufftMakePlanMany64", ("hipfftMakePlanMany64", CONV_MATH_FUNC, API_FFT)),
        ("cufftGetSizeMany64", ("hipfftGetSizeMany64", CONV_MATH_FUNC, API_FFT)),
        ("cufftEstimate1d", ("hipfftEstimate1d", CONV_MATH_FUNC, API_FFT)),
        ("cufftEstimate2d", ("hipfftEstimate2d", CONV_MATH_FUNC, API_FFT)),
        ("cufftEstimate3d", ("hipfftEstimate3d", CONV_MATH_FUNC, API_FFT)),
        ("cufftEstimateMany", ("hipfftEstimateMany", CONV_MATH_FUNC, API_FFT)),
        ("cufftCreate", ("hipfftCreate", CONV_MATH_FUNC, API_FFT)),
        ("cufftGetSize1d", ("hipfftGetSize1d", CONV_MATH_FUNC, API_FFT)),
        ("cufftGetSize2d", ("hipfftGetSize2d", CONV_MATH_FUNC, API_FFT)),
        ("cufftGetSize3d", ("hipfftGetSize3d", CONV_MATH_FUNC, API_FFT)),
        ("cufftGetSizeMany", ("hipfftGetSizeMany", CONV_MATH_FUNC, API_FFT)),
        ("cufftGetSize", ("hipfftGetSize", CONV_MATH_FUNC, API_FFT)),
        ("cufftSetWorkArea", ("hipfftSetWorkArea", CONV_MATH_FUNC, API_FFT)),
        (
            "cufftSetAutoAllocation",
            ("hipfftSetAutoAllocation", CONV_MATH_FUNC, API_FFT),
        ),
        ("cufftXtExec", ("hipfftXtExec", CONV_MATH_FUNC, API_FFT)),
        ("cufftXtMakePlanMany", ("hipfftXtMakePlanMany", CONV_MATH_FUNC, API_FFT)),
        ("cufftExecC2C", ("hipfftExecC2C", CONV_MATH_FUNC, API_FFT)),
        ("cufftExecR2C", ("hipfftExecR2C", CONV_MATH_FUNC, API_FFT)),
        ("cufftExecC2R", ("hipfftExecC2R", CONV_MATH_FUNC, API_FFT)),
        ("cufftExecZ2Z", ("hipfftExecZ2Z", CONV_MATH_FUNC, API_FFT)),
        ("cufftExecD2Z", ("hipfftExecD2Z", CONV_MATH_FUNC, API_FFT)),
        ("cufftExecZ2D", ("hipfftExecZ2D", CONV_MATH_FUNC, API_FFT)),
        ("cufftSetStream", ("hipfftSetStream", CONV_MATH_FUNC, API_FFT)),
        ("cufftDestroy", ("hipfftDestroy", CONV_MATH_FUNC, API_FFT)),
        ("cufftGetVersion", ("hipfftGetVersion", CONV_MATH_FUNC, API_FFT)),
        (
            "cufftGetProperty",
            ("hipfftGetProperty", CONV_MATH_FUNC, API_FFT, HIP_UNSUPPORTED),
        ),
        ("nvrtcResult", ("hiprtcResult", CONV_TYPE, API_RTC)),
        ("NVRTC_SUCCESS", ("HIPRTC_SUCCESS", CONV_TYPE, API_RTC)),
        (
            "NVRTC_ERROR_OUT_OF_MEMORY",
            ("HIPRTC_ERROR_OUT_OF_MEMORY", CONV_TYPE, API_RTC),
        ),
        (
            "NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
            ("HIPRTC_ERROR_PROGRAM_CREATION_FAILURE", CONV_TYPE, API_RTC),
        ),
        (
            "NVRTC_ERROR_INVALID_INPUT",
            ("HIPRTC_ERROR_INVALID_INPUT", CONV_TYPE, API_RTC),
        ),
        (
            "NVRTC_ERROR_INVALID_PROGRAM",
            ("HIPRTC_ERROR_INVALID_PROGRAM", CONV_TYPE, API_RTC),
        ),
        ("NVRTC_ERROR_COMPILATION", ("HIPRTC_ERROR_COMPILATION", CONV_TYPE, API_RTC)),
        (
            "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
            ("HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE", CONV_TYPE, API_RTC),
        ),
        (
            "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
            ("HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION", CONV_TYPE, API_RTC),
        ),
        (
            "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
            ("HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID", CONV_TYPE, API_RTC),
        ),
        (
            "NVRTC_ERROR_INTERNAL_ERROR",
            ("HIPRTC_ERROR_INTERNAL_ERROR", CONV_TYPE, API_RTC),
        ),
        ("nvrtcGetErrorString", ("hiprtcGetErrorString", CONV_JIT, API_RTC)),
        ("nvrtcVersion", ("hiprtcVersion", CONV_JIT, API_RTC)),
        ("nvrtcProgram", ("hiprtcProgram", CONV_TYPE, API_RTC)),
        ("nvrtcAddNameExpression", ("hiprtcAddNameExpression", CONV_JIT, API_RTC)),
        ("nvrtcCompileProgram", ("hiprtcCompileProgram", CONV_JIT, API_RTC)),
        ("nvrtcCreateProgram", ("hiprtcCreateProgram", CONV_JIT, API_RTC)),
        ("nvrtcDestroyProgram", ("hiprtcDestroyProgram", CONV_JIT, API_RTC)),
        ("nvrtcGetLoweredName", ("hiprtcGetLoweredName", CONV_JIT, API_RTC)),
        ("nvrtcGetProgramLog", ("hiprtcGetProgramLog", CONV_JIT, API_RTC)),
        ("nvrtcGetProgramLogSize", ("hiprtcGetProgramLogSize", CONV_JIT, API_RTC)),
        ("nvrtcGetPTX", ("hiprtcGetCode", CONV_JIT, API_RTC)),
        ("nvrtcGetPTXSize", ("hiprtcGetCodeSize", CONV_JIT, API_RTC)),
        ("thrust::cuda", ("thrust::hip", CONV_MATH_FUNC, API_BLAS)),
        (
            "cudaCpuDeviceId",
            ("hipCpuDeviceId", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        # The caffe2 directory does a string match; pytorch does a word-boundary match.
        # Patterns such as 'cub::' will not match for pytorch.
        # We list all current uses of cub symbols for this reason.
        ("cub::", ("hipcub::", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::ArgMax", ("hipcub::ArgMax", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::ArgMin", ("hipcub::ArgMin", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BLOCK_SCAN_WARP_SCANS", ("hipcub::BLOCK_SCAN_WARP_SCANS", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BLOCK_REDUCE_WARP_REDUCTIONS", ("hipcub::BLOCK_REDUCE_WARP_REDUCTIONS", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BLOCK_STORE_WARP_TRANSPOSE", ("hipcub::BLOCK_STORE_WARP_TRANSPOSE", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BLOCK_LOAD_DIRECT", ("hipcub::BLOCK_LOAD_DIRECT", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BLOCK_STORE_DIRECT", ("hipcub::BLOCK_STORE_DIRECT", CONV_SPECIAL_FUNC, API_RUNTIME)),
        (
            "cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY",
            ("hipcub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY", CONV_SPECIAL_FUNC, API_RUNTIME)
        ),
        ("cub::BlockReduce", ("hipcub::BlockReduce", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BlockScan", ("hipcub::BlockScan", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BlockLoad", ("hipcub::BlockLoad", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BlockStore", ("hipcub::BlockStore", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::BlockRakingLayout", ("hipcub::BlockRakingLayout", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::Uninitialized", ("hipcub::Uninitialized", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::RowMajorTid", ("hipcub::RowMajorTid", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::CachingDeviceAllocator", ("hipcub::CachingDeviceAllocator", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::CountingInputIterator", ("hipcub::CountingInputIterator", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::DeviceRadixSort", ("hipcub::DeviceRadixSort", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::DeviceReduce", ("hipcub::DeviceReduce", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::DeviceRunLengthEncode", ("hipcub::DeviceRunLengthEncode", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::DeviceScan", ("hipcub::DeviceScan", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::DeviceSegmentedRadixSort", ("hipcub::DeviceSegmentedRadixSort", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::DeviceSegmentedReduce", ("hipcub::DeviceSegmentedReduce", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::DeviceSelect", ("hipcub::DeviceSelect", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::KeyValuePair", ("hipcub::KeyValuePair", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::Max", ("hipcub::Max", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::Min", ("hipcub::Min", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::Sum", ("hipcub::Sum", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::Log2", ("hipcub::Log2", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::LaneId", ("hipcub::LaneId", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::WarpMask", ("hipcub::WarpMask", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::ShuffleIndex", ("hipcub::ShuffleIndex", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::ShuffleDown", ("hipcub::ShuffleDown", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::ArgIndexInputIterator", ("hipcub::ArgIndexInputIterator", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::TransformInputIterator", ("hipcub::TransformInputIterator", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::WarpReduce", ("hipcub::WarpReduce", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("cub::CTA_SYNC", ("hipcub::CTA_SYNC", CONV_SPECIAL_FUNC, API_RUNTIME)),
        ("nvtxMark", ("roctxMark", CONV_OTHER, API_ROCTX)),
        ("nvtxMarkA", ("roctxMarkA", CONV_OTHER, API_ROCTX)),
        ("nvtxRangePushA", ("roctxRangePushA", CONV_OTHER, API_ROCTX)),
        ("nvtxRangePop", ("roctxRangePop", CONV_OTHER, API_ROCTX)),
        ("nvtxRangeStartA", ("roctxRangeStartA", CONV_OTHER, API_ROCTX)),
        ("nvtxRangeEnd", ("roctxRangeStop", CONV_OTHER, API_ROCTX)),
        ("nvmlReturn_t", ("rsmi_status_t", CONV_OTHER, API_ROCMSMI)),
        ("NVML_SUCCESS", ("RSMI_STATUS_SUCCESS", CONV_OTHER, API_ROCMSMI)),
        ("NVML_P2P_CAPS_INDEX_READ", ("RSMI_STATUS_SUCCESS", CONV_OTHER, API_ROCMSMI)),
        ("NVML_P2P_STATUS_OK", ("RSMI_STATUS_SUCCESS", CONV_OTHER, API_ROCMSMI)),
        ("NVML_ERROR_INSUFFICIENT_SIZE", ("RSMI_STATUS_INSUFFICIENT_SIZE", CONV_OTHER, API_ROCMSMI)),
        ("nvmlDevice_t", ("uint32_t", CONV_OTHER, API_ROCMSMI)),
        ("nvmlGpuP2PStatus_t", ("bool", CONV_OTHER, API_ROCMSMI)),
        ("nvmlProcessInfo_t", ("rsmi_process_info_t", CONV_OTHER, API_ROCMSMI)),
        ("nvmlGpuP2PCapsIndex_t", ("uint32_t", CONV_OTHER, API_ROCMSMI)),
    ]
)

CUDA_SPECIAL_MAP = collections.OrderedDict(
    [
        # SPARSE
        ("cusparseStatus_t", ("hipsparseStatus_t", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseHandle_t", ("hipsparseHandle_t", CONV_MATH_FUNC, API_SPECIAL)),
        ("cuComplex", ("hipComplex", CONV_TYPE, API_SPECIAL)),
        ("cuDoubleComplex", ("hipDoubleComplex", CONV_TYPE, API_SPECIAL)),
        (
            "CUSPARSE_POINTER_MODE_HOST",
            ("HIPSPARSE_POINTER_MODE_HOST", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        ("cusparseOperation_t", ("hipsparseOperation_t", CONV_TYPE, API_SPECIAL)),
        (
            "cusparseCreateMatDescr",
            ("hipsparseCreateMatDescr", CONV_MATH_FUNC, API_SPECIAL),
        ),
        ("cusparseCreate", ("hipsparseCreate", CONV_MATH_FUNC, API_SPECIAL)),
        (
            "cusparseDestroyMatDescr",
            ("hipsparseDestroyMatDescr", CONV_MATH_FUNC, API_SPECIAL),
        ),
        ("cusparseDestroy", ("hipsparseDestroy", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseXcoo2csr", ("hipsparseXcoo2csr", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseMatDescr_t", ("hipsparseMatDescr_t", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDiagType_t", ("hipsparseDiagType_t", CONV_TYPE, API_SPECIAL)),
        ("CUSPARSE_DIAG_TYPE_UNIT", ("HIPSPARSE_DIAG_TYPE_UNIT", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_DIAG_TYPE_NON_UNIT", ("HIPSPARSE_DIAG_TYPE_NON_UNIT", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("cusparseSetMatDiagType", ("hipsparseSetMatDiagType", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseFillMode_t", ("hipsparseFillMode_t", CONV_TYPE, API_SPECIAL)),
        ("CUSPARSE_FILL_MODE_UPPER", ("HIPSPARSE_FILL_MODE_UPPER", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_FILL_MODE_LOWER", ("HIPSPARSE_FILL_MODE_LOWER", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("cusparseSetMatFillMode", ("hipsparseSetMatFillMode", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDirection_t", ("hipsparseDirection_t", CONV_TYPE, API_SPECIAL)),
        ("CUSPARSE_DIRECTION_ROW", ("HIPSPARSE_DIRECTION_ROW", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_DIRECTION_COLUMN", ("HIPSPARSE_DIRECTION_COLUMN", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("cusparseSolvePolicy_t", ("hipsparseSolvePolicy_t", CONV_TYPE, API_SPECIAL)),
        ("CUSPARSE_SOLVE_POLICY_NO_LEVEL", ("HIPSPARSE_SOLVE_POLICY_NO_LEVEL", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_SOLVE_POLICY_USE_LEVEL", ("HIPSPARSE_SOLVE_POLICY_USE_LEVEL", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("cusparseCreateBsrsv2Info", ("hipsparseCreateBsrsv2Info", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCreateBsrsm2Info", ("hipsparseCreateBsrsm2Info", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDestroyBsrsv2Info", ("hipsparseDestroyBsrsv2Info", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDestroyBsrsm2Info", ("hipsparseDestroyBsrsm2Info", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrmm", ("hipsparseSbsrmm", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrmm", ("hipsparseDbsrmm", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrmm", ("hipsparseCbsrmm", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrmm", ("hipsparseZbsrmm", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrmv", ("hipsparseSbsrmv", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrmv", ("hipsparseDbsrmv", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrmv", ("hipsparseCbsrmv", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrmv", ("hipsparseZbsrmv", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrsv2_bufferSize", ("hipsparseSbsrsv2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrsv2_bufferSize", ("hipsparseDbsrsv2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrsv2_bufferSize", ("hipsparseCbsrsv2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrsv2_bufferSize", ("hipsparseZbsrsv2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrsv2_analysis", ("hipsparseSbsrsv2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrsv2_analysis", ("hipsparseDbsrsv2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrsv2_analysis", ("hipsparseCbsrsv2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrsv2_analysis", ("hipsparseZbsrsv2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrsv2_solve", ("hipsparseSbsrsv2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrsv2_solve", ("hipsparseDbsrsv2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrsv2_solve", ("hipsparseCbsrsv2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrsv2_solve", ("hipsparseZbsrsv2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrsm2_bufferSize", ("hipsparseSbsrsm2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrsm2_bufferSize", ("hipsparseDbsrsm2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrsm2_bufferSize", ("hipsparseCbsrsm2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrsm2_bufferSize", ("hipsparseZbsrsm2_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrsm2_analysis", ("hipsparseSbsrsm2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrsm2_analysis", ("hipsparseDbsrsm2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrsm2_analysis", ("hipsparseCbsrsm2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrsm2_analysis", ("hipsparseZbsrsm2_analysis", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSbsrsm2_solve", ("hipsparseSbsrsm2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDbsrsm2_solve", ("hipsparseDbsrsm2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCbsrsm2_solve", ("hipsparseCbsrsm2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZbsrsm2_solve", ("hipsparseZbsrsm2_solve", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseScsrmm2", ("hipsparseScsrmm2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDcsrmm2", ("hipsparseDcsrmm2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCcsrmm2", ("hipsparseCcsrmm2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZcsrmm2", ("hipsparseZcsrmm2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseScsrmm", ("hipsparseScsrmm", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDcsrmm", ("hipsparseDcsrmm", CONV_MATH_FUNC, API_SPECIAL)),
        (
            "cusparseXcsrsort_bufferSizeExt",
            ("hipsparseXcsrsort_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL),
        ),
        ("cusparseCreateCsrgemm2Info", ("hipsparseCreateCsrgemm2Info", CONV_MATH_FUNC, API_SPECIAL)),
        (
            "cusparseDestroyCsrgemm2Info",
            ("hipsparseDestroyCsrgemm2Info", CONV_MATH_FUNC, API_SPECIAL),
        ),
        ("cusparseXcsrgemm2Nnz", ("hipsparseXcsrgemm2Nnz", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDcsrgemm2_bufferSizeExt", ("hipsparseDcsrgemm2_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseScsrgemm2_bufferSizeExt", ("hipsparseScsrgemm2_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDcsrgemm2", ("hipsparseDcsrgemm2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseScsrgemm2", ("hipsparseScsrgemm2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSetPointerMode", ("hipsparseSetPointerMode", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseXcsrgeam2Nnz", ("hipsparseXcsrgeam2Nnz", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseScsrgeam2_bufferSizeExt", ("hipsparseScsrgeam2_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDcsrgeam2_bufferSizeExt", ("hipsparseDcsrgeam2_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCcsrgeam2_bufferSizeExt", ("hipsparseCcsrgeam2_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZcsrgeam2_bufferSizeExt", ("hipsparseZcsrgeam2_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseScsrgeam2", ("hipsparseScsrgeam2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDcsrgeam2", ("hipsparseDcsrgeam2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCcsrgeam2", ("hipsparseCcsrgeam2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseZcsrgeam2", ("hipsparseZcsrgeam2", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseXcsrsort", ("hipsparseXcsrsort", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseXbsrsm2_zeroPivot", ("hipsparseXbsrsm2_zeroPivot", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseXbsrsv2_zeroPivot", ("hipsparseXbsrsv2_zeroPivot", CONV_MATH_FUNC, API_SPECIAL)),
        (
            "cusparseXcoosort_bufferSizeExt",
            ("hipsparseXcoosort_bufferSizeExt", CONV_MATH_FUNC, API_SPECIAL),
        ),
        (
            "cusparseXcoosortByRow",
            ("hipsparseXcoosortByRow", CONV_MATH_FUNC, API_SPECIAL),
        ),
        ("cusparseSetStream", ("hipsparseSetStream", CONV_MATH_FUNC, API_SPECIAL)),
        (
            "cusparseCreateIdentityPermutation",
            ("hipsparseCreateIdentityPermutation", CONV_MATH_FUNC, API_SPECIAL),
        ),
        (
            "cusparseSetMatIndexBase",
            ("hipsparseSetMatIndexBase", CONV_MATH_FUNC, API_SPECIAL),
        ),
        ("cusparseSetMatType", ("hipsparseSetMatType", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpMV", ("hipsparseSpMV", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpMV_bufferSize", ("hipsparseSpMV_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpMM", ("hipsparseSpMM", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpMM_bufferSize", ("hipsparseSpMM_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCreateDnMat", ("hipsparseCreateDnMat", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDnMatSetStridedBatch", ("hipsparseDnMatSetStridedBatch", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCsrSetStridedBatch", ("hipsparseCsrSetStridedBatch", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCreateDnVec", ("hipsparseCreateDnVec", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCreateCsr", ("hipsparseCreateCsr", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDestroyDnMat", ("hipsparseDestroyDnMat", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDestroyDnVec", ("hipsparseDestroyDnVec", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDestroySpMat", ("hipsparseDestroySpMat", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpGEMM_destroyDescr", ("hipsparseSpGEMM_destroyDescr", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCreateCoo", ("hipsparseCreateCoo", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCreateCsr", ("hipsparseCreateCsr", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpGEMM_createDescr", ("hipsparseSpGEMM_createDescr", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseDnMatSetStridedBatch", ("hipsparseDnMatSetStridedBatch", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpGEMM_copy", ("hipsparseSpGEMM_copy", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSDDMM_bufferSize", ("hipsparseSDDMM_bufferSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSDDMM_preprocess", ("hipsparseSDDMM_preprocess", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSDDMM", ("hipsparseSDDMM", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpGEMM_compute", ("hipsparseSpGEMM_compute", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpGEMM_workEstimation", ("hipsparseSpGEMM_workEstimation", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpMatGetSize", ("hipsparseSpMatGetSize", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseCsrSetPointers", ("hipsparseCsrSetPointers", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusparseSpMVAlg_t", ("hipsparseSpMVAlg_t", CONV_TYPE, API_SPECIAL)),
        ("cusparseSpMMAlg_t", ("hipsparseSpMMAlg_t", CONV_TYPE, API_SPECIAL)),
        ("cusparseIndexType_t", ("hipsparseIndexType_t", CONV_TYPE, API_SPECIAL)),
        # Unsupported ("cusparseMatDescr", ("hipsparseMatDescr", CONV_TYPE, API_SPECIAL)),
        # Unsupported ("cusparseDnMatDescr", ("hipsparseDnMatDescr", CONV_TYPE, API_SPECIAL)),
        # Unsupported ("cusparseDnVecDescr", ("hipsparseDnVecDescr", CONV_TYPE, API_SPECIAL)),
        # Unsupported ("cusparseSpMatDescr", ("hipsparseSpMatDescr", CONV_TYPE, API_SPECIAL)),
        # Unsupported ("cusparseSpGEMMDescr", ("hipsparseSpGEMMDescr", CONV_TYPE, API_SPECIAL)),
        ("cusparseDnMatDescr_t", ("hipsparseDnMatDescr_t", CONV_TYPE, API_SPECIAL)),
        ("cusparseDnVecDescr_t", ("hipsparseDnVecDescr_t", CONV_TYPE, API_SPECIAL)),
        ("cusparseSpMatDescr_t", ("hipsparseSpMatDescr_t", CONV_TYPE, API_SPECIAL)),
        ("cusparseSpGEMMDescr_t", ("hipsparseSpGEMMDescr_t", CONV_TYPE, API_SPECIAL)),
        ("CUSPARSE_INDEX_32I", ("HIPSPARSE_INDEX_32I", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_INDEX_64I", ("HIPSPARSE_INDEX_64I", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_ORDER_COL", ("HIPSPARSE_ORDER_COLUMN", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_MV_ALG_DEFAULT", ("HIPSPARSE_MV_ALG_DEFAULT", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_MM_ALG_DEFAULT", ("HIPSPARSE_MM_ALG_DEFAULT", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_SPMM_COO_ALG1", ("HIPSPARSE_SPMM_COO_ALG1", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_SPMM_COO_ALG2", ("HIPSPARSE_SPMM_COO_ALG2", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_COOMV_ALG", ("HIPSPARSE_COOMV_ALG", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_SPMM_CSR_ALG1", ("HIPSPARSE_CSRMM_ALG1", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_SPGEMM_DEFAULT", ("HIPSPARSE_SPGEMM_DEFAULT", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSPARSE_SDDMM_ALG_DEFAULT", ("HIPSPARSE_SDDMM_ALG_DEFAULT", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        (
            "CUSPARSE_STATUS_SUCCESS",
            ("HIPSPARSE_STATUS_SUCCESS", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_NOT_INITIALIZED",
            ("HIPSPARSE_STATUS_NOT_INITIALIZED", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_ALLOC_FAILED",
            ("HIPSPARSE_STATUS_ALLOC_FAILED", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_INVALID_VALUE",
            ("HIPSPARSE_STATUS_INVALID_VALUE", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_MAPPING_ERROR",
            ("HIPSPARSE_STATUS_MAPPING_ERROR", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_EXECUTION_FAILED",
            ("HIPSPARSE_STATUS_EXECUTION_FAILED", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_INTERNAL_ERROR",
            ("HIPSPARSE_STATUS_INTERNAL_ERROR", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
            (
                "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
                CONV_NUMERIC_LITERAL,
                API_SPECIAL,
            ),
        ),
        (
            "CUSPARSE_STATUS_ARCH_MISMATCH",
            ("HIPSPARSE_STATUS_ARCH_MISMATCH", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_STATUS_ZERO_PIVOT",
            ("HIPSPARSE_STATUS_ZERO_PIVOT", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_OPERATION_TRANSPOSE",
            ("HIPSPARSE_OPERATION_TRANSPOSE", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_OPERATION_NON_TRANSPOSE",
            ("HIPSPARSE_OPERATION_NON_TRANSPOSE", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE",
            (
                "HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE",
                CONV_NUMERIC_LITERAL,
                API_SPECIAL,
            ),
        ),
        (
            "CUSPARSE_INDEX_BASE_ZERO",
            ("HIPSPARSE_INDEX_BASE_ZERO", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_INDEX_BASE_ONE",
            ("HIPSPARSE_INDEX_BASE_ONE", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUSPARSE_MATRIX_TYPE_GENERAL",
            ("HIPSPARSE_MATRIX_TYPE_GENERAL", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        # SOLVER
        ("cublasOperation_t", ("hipsolverOperation_t", CONV_TYPE, API_SPECIAL)),
        ("CUBLAS_OP_N", ("HIPSOLVER_OP_N", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        (
            "CUBLAS_OP_T",
            ("HIPSOLVER_OP_T", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUBLAS_OP_C",
            ("HIPSOLVER_OP_C", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        ("cublasFillMode_t", ("hipsolverFillMode_t", CONV_TYPE, API_SPECIAL)),
        (
            "CUBLAS_FILL_MODE_LOWER",
            ("HIPSOLVER_FILL_MODE_LOWER", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        (
            "CUBLAS_FILL_MODE_UPPER",
            ("HIPSOLVER_FILL_MODE_UPPER", CONV_NUMERIC_LITERAL, API_SPECIAL),
        ),
        ("cublasSideMode_t", ("hipsolverSideMode_t", CONV_TYPE, API_SPECIAL)),
        ("CUBLAS_SIDE_LEFT", ("HIPSOLVER_SIDE_LEFT", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUBLAS_SIDE_RIGHT", ("HIPSOLVER_SIDE_RIGHT", CONV_NUMERIC_LITERAL, API_SPECIAL)),

        ("cusolverEigMode_t", ("hipsolverEigMode_t", CONV_TYPE, API_SPECIAL)),
        ("CUSOLVER_EIG_MODE_VECTOR", ("HIPSOLVER_EIG_MODE_VECTOR", CONV_NUMERIC_LITERAL, API_SPECIAL)),
        ("CUSOLVER_EIG_MODE_NOVECTOR", ("HIPSOLVER_EIG_MODE_NOVECTOR", CONV_NUMERIC_LITERAL, API_SPECIAL)),

        ("syevjInfo_t", ("hipsolverSyevjInfo_t", CONV_TYPE, API_SPECIAL)),
        ("cusolverDnCreateSyevjInfo", ("hipsolverDnCreateSyevjInfo", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusolverDnXsyevjSetSortEig", ("hipsolverDnXsyevjSetSortEig", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusolverDnDestroySyevjInfo", ("hipsolverDnDestroySyevjInfo", CONV_MATH_FUNC, API_SPECIAL)),

        ("gesvdjInfo_t", ("hipsolverGesvdjInfo_t", CONV_TYPE, API_SPECIAL)),
        ("cusolverDnCreateGesvdjInfo", ("hipsolverDnCreateGesvdjInfo", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusolverDnXgesvdjSetSortEig", ("hipsolverDnXgesvdjSetSortEig", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusolverDnDestroyGesvdjInfo", ("hipsolverDnDestroyGesvdjInfo", CONV_MATH_FUNC, API_SPECIAL)),

        ("cusolverDnHandle_t", ("hipsolverDnHandle_t", CONV_TYPE, API_SPECIAL)),
        ("cusolverDnCreate", ("hipsolverDnCreate", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusolverDnSetStream", ("hipsolverDnSetStream", CONV_MATH_FUNC, API_SPECIAL)),
        ("cusolverDnDestroy", ("hipsolverDnDestroy", CONV_MATH_FUNC, API_SPECIAL)),

        # from aten/src/ATen/native/hip/linalg/HIPSolver.cpp
        ('cusolverDnParams_t', ('hipsolverDnParams_t', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgeqrf', ('hipsolverDnCgeqrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgeqrf_bufferSize', ('hipsolverDnCgeqrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgesvd', ('hipsolverDnCgesvd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgesvd_bufferSize', ('hipsolverDnCgesvd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgesvdj', ('hipsolverDnCgesvdj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgesvdjBatched', ('hipsolverDnCgesvdjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgesvdjBatched_bufferSize', ('hipsolverDnCgesvdjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgesvdj_bufferSize', ('hipsolverDnCgesvdj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgetrf', ('hipsolverDnCgetrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgetrf_bufferSize', ('hipsolverDnCgetrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgetrs', ('hipsolverDnCgetrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCheevd', ('hipsolverDnCheevd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCheevd_bufferSize', ('hipsolverDnCheevd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCheevj', ('hipsolverDnCheevj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCheevjBatched', ('hipsolverDnCheevjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCheevjBatched_bufferSize', ('hipsolverDnCheevjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCheevj_bufferSize', ('hipsolverDnCheevj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCpotrf', ('hipsolverDnCpotrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCpotrfBatched', ('hipsolverDnCpotrfBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCpotrf_bufferSize', ('hipsolverDnCpotrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCpotrs', ('hipsolverDnCpotrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCpotrsBatched', ('hipsolverDnCpotrsBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCungqr', ('hipsolverDnCungqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCungqr_bufferSize', ('hipsolverDnCungqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCunmqr', ('hipsolverDnCunmqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCunmqr_bufferSize', ('hipsolverDnCunmqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgeqrf', ('hipsolverDnDgeqrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgeqrf_bufferSize', ('hipsolverDnDgeqrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgesvd', ('hipsolverDnDgesvd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgesvd_bufferSize', ('hipsolverDnDgesvd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgesvdj', ('hipsolverDnDgesvdj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgesvdjBatched', ('hipsolverDnDgesvdjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgesvdjBatched_bufferSize', ('hipsolverDnDgesvdjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgesvdj_bufferSize', ('hipsolverDnDgesvdj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgetrf', ('hipsolverDnDgetrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgetrf_bufferSize', ('hipsolverDnDgetrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgetrs', ('hipsolverDnDgetrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDorgqr', ('hipsolverDnDorgqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDorgqr_bufferSize', ('hipsolverDnDorgqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDormqr', ('hipsolverDnDormqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDormqr_bufferSize', ('hipsolverDnDormqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDpotrf', ('hipsolverDnDpotrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDpotrfBatched', ('hipsolverDnDpotrfBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDpotrf_bufferSize', ('hipsolverDnDpotrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDpotrs', ('hipsolverDnDpotrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDpotrsBatched', ('hipsolverDnDpotrsBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDsyevd', ('hipsolverDnDsyevd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDsyevd_bufferSize', ('hipsolverDnDsyevd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDsyevj', ('hipsolverDnDsyevj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDsyevjBatched', ('hipsolverDnDsyevjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDsyevjBatched_bufferSize', ('hipsolverDnDsyevjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDsyevj_bufferSize', ('hipsolverDnDsyevj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgeqrf', ('hipsolverDnSgeqrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgeqrf_bufferSize', ('hipsolverDnSgeqrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgesvd', ('hipsolverDnSgesvd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgesvd_bufferSize', ('hipsolverDnSgesvd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgesvdj', ('hipsolverDnSgesvdj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgesvdjBatched', ('hipsolverDnSgesvdjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgesvdjBatched_bufferSize', ('hipsolverDnSgesvdjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgesvdj_bufferSize', ('hipsolverDnSgesvdj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgetrf', ('hipsolverDnSgetrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgetrf_bufferSize', ('hipsolverDnSgetrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSgetrs', ('hipsolverDnSgetrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSorgqr', ('hipsolverDnSorgqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSorgqr_bufferSize', ('hipsolverDnSorgqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSormqr', ('hipsolverDnSormqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSormqr_bufferSize', ('hipsolverDnSormqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSpotrf', ('hipsolverDnSpotrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSpotrfBatched', ('hipsolverDnSpotrfBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSpotrf_bufferSize', ('hipsolverDnSpotrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSpotrs', ('hipsolverDnSpotrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSpotrsBatched', ('hipsolverDnSpotrsBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsyevd', ('hipsolverDnSsyevd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsyevd_bufferSize', ('hipsolverDnSsyevd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsyevj', ('hipsolverDnSsyevj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsyevjBatched', ('hipsolverDnSsyevjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsyevjBatched_bufferSize', ('hipsolverDnSsyevjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsyevj_bufferSize', ('hipsolverDnSsyevj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXgeqrf', ('hipsolverDnXgeqrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXgeqrf_bufferSize', ('hipsolverDnXgeqrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXpotrf', ('hipsolverDnXpotrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXpotrf_bufferSize', ('hipsolverDnXpotrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXpotrs', ('hipsolverDnXpotrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXsyevd', ('hipsolverDnXsyevd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXsyevd_bufferSize', ('hipsolverDnXsyevd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgeqrf', ('hipsolverDnZgeqrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgeqrf_bufferSize', ('hipsolverDnZgeqrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgesvd', ('hipsolverDnZgesvd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgesvd_bufferSize', ('hipsolverDnZgesvd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgesvdj', ('hipsolverDnZgesvdj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgesvdjBatched', ('hipsolverDnZgesvdjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgesvdjBatched_bufferSize', ('hipsolverDnZgesvdjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgesvdj_bufferSize', ('hipsolverDnZgesvdj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgetrf', ('hipsolverDnZgetrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgetrf_bufferSize', ('hipsolverDnZgetrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgetrs', ('hipsolverDnZgetrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZheevd', ('hipsolverDnZheevd', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZheevd_bufferSize', ('hipsolverDnZheevd_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZheevj', ('hipsolverDnZheevj', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZheevjBatched', ('hipsolverDnZheevjBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZheevjBatched_bufferSize', ('hipsolverDnZheevjBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZheevj_bufferSize', ('hipsolverDnZheevj_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZpotrf', ('hipsolverDnZpotrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZpotrfBatched', ('hipsolverDnZpotrfBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZpotrf_bufferSize', ('hipsolverDnZpotrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZpotrs', ('hipsolverDnZpotrs', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZpotrsBatched', ('hipsolverDnZpotrsBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZungqr', ('hipsolverDnZungqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZungqr_bufferSize', ('hipsolverDnZungqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZunmqr', ('hipsolverDnZunmqr', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZunmqr_bufferSize', ('hipsolverDnZunmqr_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),

        # sytrf
        ('cusolverDnDsytrf_bufferSize', ('hipsolverDnDsytrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsytrf_bufferSize', ('hipsolverDnSsytrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZsytrf_bufferSize', ('hipsolverDnZsytrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCsytrf_bufferSize', ('hipsolverDnCsytrf_bufferSize', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDsytrf', ('hipsolverDnDsytrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnSsytrf', ('hipsolverDnSsytrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZsytrf', ('hipsolverDnZsytrf', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCsytrf', ('hipsolverDnCsytrf', CONV_MATH_FUNC, API_SPECIAL)),

        # gesdva strided
        (
            'cusolverDnSgesvdaStridedBatched_bufferSize',
            ('hipsolverDnSgesvdaStridedBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)
        ),
        (
            'cusolverDnDgesvdaStridedBatched_bufferSize',
            ('hipsolverDnDgesvdaStridedBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)
        ),
        (
            'cusolverDnCgesvdaStridedBatched_bufferSize',
            ('hipsolverDnCgesvdaStridedBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)
        ),
        (
            'cusolverDnZgesvdaStridedBatched_bufferSize',
            ('hipsolverDnZgesvdaStridedBatched_bufferSize', CONV_MATH_FUNC, API_SPECIAL)
        ),
        ('cusolverDnSgesvdaStridedBatched', ('hipsolverDnSgesvdaStridedBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnDgesvdaStridedBatched', ('hipsolverDnDgesvdaStridedBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnCgesvdaStridedBatched', ('hipsolverDnCgesvdaStridedBatched', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnZgesvdaStridedBatched', ('hipsolverDnZgesvdaStridedBatched', CONV_MATH_FUNC, API_SPECIAL)),

        # gesvdj SetXXX
        ('cusolverDnXgesvdjSetTolerance', ('hipsolverDnXgesvdjSetTolerance', CONV_MATH_FUNC, API_SPECIAL)),
        ('cusolverDnXgesvdjSetMaxSweeps', ('hipsolverDnXgesvdjSetMaxSweeps', CONV_MATH_FUNC, API_SPECIAL)),
    ]
)

PYTORCH_SPECIFIC_MAPPINGS = collections.OrderedDict(
    [
        ("USE_CUDA", ("USE_ROCM", API_PYTORCH)),
        ("CUDA_VERSION", ("TORCH_HIP_VERSION", API_PYTORCH)),
        ("cudaHostAllocator", ("hipHostAllocator", API_PYTORCH)),
        ("cudaDeviceAllocator", ("hipDeviceAllocator", API_PYTORCH)),
        ("define MAX_NUM_BLOCKS 200", ("define MAX_NUM_BLOCKS 64", API_PYTORCH)),
        ("cuda::CUDAGuard", ("hip::HIPGuardMasqueradingAsCUDA", API_PYTORCH)),
        ("CUDAGuard", ("HIPGuardMasqueradingAsCUDA", API_PYTORCH)),
        (
            "cuda::OptionalCUDAGuard",
            ("hip::OptionalHIPGuardMasqueradingAsCUDA", API_PYTORCH),
        ),
        ("OptionalCUDAGuard", ("OptionalHIPGuardMasqueradingAsCUDA", API_PYTORCH)),
        (
            "cuda::CUDAStreamGuard",
            ("hip::HIPStreamGuardMasqueradingAsCUDA", API_PYTORCH),
        ),
        ("CUDAStreamGuard", ("HIPStreamGuardMasqueradingAsCUDA", API_PYTORCH)),
        (
            "cuda::OptionalCUDAStreamGuard",
            ("hip::OptionalHIPStreamGuardMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "OptionalCUDAStreamGuard",
            ("OptionalHIPStreamGuardMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "cuda::CUDAMultiStreamGuard",
            ("hip::HIPMultiStreamGuardMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "CUDAMultiStreamGuard",
            ("HIPMultiStreamGuardMasqueradingAsCUDA", API_PYTORCH),
        ),
        # Only get needs to be transformed this way; all the other ones can go
        # straight to the normal versions hip::HIPCachingAllocator
        (
            "cuda::CUDACachingAllocator::get",
            ("hip::HIPCachingAllocatorMasqueradingAsCUDA::get", API_PYTORCH),
        ),
        (
            "CUDACachingAllocator::get",
            ("HIPCachingAllocatorMasqueradingAsCUDA::get", API_PYTORCH),
        ),
        (
            "cuda::CUDACachingAllocator::recordStream",
            (
                "hip::HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA",
                API_PYTORCH,
            ),
        ),
        (
            "CUDACachingAllocator::recordStream",
            (
                "HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA",
                API_PYTORCH,
            ),
        ),
        (
            "cuda::CUDAAllocator::recordStream",
            (
                "hip::HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA",
                API_PYTORCH,
            ),
        ),
        (
            "CUDAAllocator::recordStream",
            (
                "HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA",
                API_PYTORCH,
            ),
        ),
        ("cuda::CUDAStream", ("hip::HIPStreamMasqueradingAsCUDA", API_PYTORCH)),
        ("CUDAStream", ("HIPStreamMasqueradingAsCUDA", API_PYTORCH)),
        (
            "cuda::getStreamFromPool",
            ("hip::getStreamFromPoolMasqueradingAsCUDA", API_PYTORCH),
        ),
        ("getStreamFromPool", ("getStreamFromPoolMasqueradingAsCUDA", API_PYTORCH)),
        (
            "cuda::getDefaultCUDAStream",
            ("hip::getDefaultHIPStreamMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "cuda::getStreamFromExternal",
            ("hip::getStreamFromExternalMasqueradingAsCUDA", API_PYTORCH),
        ),
        ("getStreamFromExternal", ("getStreamFromExternalMasqueradingAsCUDA", API_PYTORCH)),
        (
            "cuda::getDefaultCUDAStream",
            ("hip::getDefaultHIPStreamMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "getDefaultCUDAStream",
            ("getDefaultHIPStreamMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "cuda::getCurrentCUDAStream",
            ("hip::getCurrentHIPStreamMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "getCurrentCUDAStream",
            ("getCurrentHIPStreamMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "cuda::setCurrentCUDAStream",
            ("hip::setCurrentHIPStreamMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "setCurrentCUDAStream",
            ("setCurrentHIPStreamMasqueradingAsCUDA", API_PYTORCH),
        ),
        (
            "ATen/cudnn/Handle.h",
            ("ATen/miopen/Handle.h", API_PYTORCH),
        ),
        # TODO: Undo this special-case; see the header for motivation behind this
        # hack.  It's VERY important this is only applied to PyTorch HIPify.
        (
            "c10/cuda/CUDAGuard.h",
            ("ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h", API_PYTORCH),
        ),
        (
            "c10/cuda/CUDACachingAllocator.h",
            ("ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h", API_PYTORCH),
        ),
        (
            "c10/cuda/CUDAStream.h",
            ("ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h", API_PYTORCH),
        ),
        ("gloo/cuda.h", ("gloo/hip.h", API_PYTORCH)),
        (
            "gloo/cuda_allreduce_halving_doubling.h",
            ("gloo/hip_allreduce_halving_doubling.h", API_PYTORCH),
        ),
        (
            "gloo/cuda_allreduce_halving_doubling_pipelined.h",
            ("gloo/hip_allreduce_halving_doubling_pipelined.h", API_PYTORCH),
        ),
        ("gloo/cuda_allreduce_ring.h", ("gloo/hip_allreduce_ring.h", API_PYTORCH)),
        (
            "gloo/cuda_broadcast_one_to_all.h",
            ("gloo/hip_broadcast_one_to_all.h", API_PYTORCH),
        ),
        (
            "gloo::CudaAllreduceHalvingDoublingPipelined",
            ("gloo::HipAllreduceHalvingDoublingPipelined", API_PYTORCH),
        ),
        ("gloo::CudaBroadcastOneToAll", ("gloo::HipBroadcastOneToAll", API_PYTORCH)),
        ("gloo::CudaHostWorkspace", ("gloo::HipHostWorkspace", API_PYTORCH)),
        ("gloo::CudaDeviceWorkspace", ("gloo::HipDeviceWorkspace", API_PYTORCH)),
        ("CUDNN_RNN_RELU", ("miopenRNNRELU", API_PYTORCH)),
        ("CUDNN_RNN_TANH", ("miopenRNNTANH", API_PYTORCH)),
        ("CUDNN_LSTM", ("miopenLSTM", API_PYTORCH)),
        ("CUDNN_GRU", ("miopenGRU", API_PYTORCH)),
        ("cudnnRNNMode_t", ("miopenRNNMode_t", API_PYTORCH)),
        ("magma_queue_create_from_cuda", ("magma_queue_create_from_hip", API_PYTORCH)),
    ]
)

CAFFE2_SPECIFIC_MAPPINGS = collections.OrderedDict(
    [
        ("cuda_stream", ("hip_stream", API_CAFFE2)),
        # if the header is a native hip folder (under hip directory),
        # there is no need to add a hip path to it; the trie in hipify script
        # takes this mapping order to forbid further replacement
        ("/hip/", ("/hip/", API_CAFFE2)),
        ("/context_gpu", ("/hip/context_gpu", API_CAFFE2)),
        ("/common_gpu", ("/hip/common_gpu", API_CAFFE2)),
        ("/cuda_nccl_gpu", ("/hip/hip_nccl_gpu", API_CAFFE2)),
        ("/mixed_utils", ("/hip/mixed_utils", API_CAFFE2)),
        ("/operator_fallback_gpu", ("/hip/operator_fallback_gpu", API_CAFFE2)),
        (
            "/spatial_batch_norm_op_impl",
            ("/hip/spatial_batch_norm_op_impl", API_CAFFE2),
        ),
        (
            "/recurrent_network_executor_gpu",
            ("/hip/recurrent_network_executor_gpu", API_CAFFE2),
        ),
        (
            "/generate_proposals_op_util_nms_gpu",
            ("/hip/generate_proposals_op_util_nms_gpu", API_CAFFE2),
        ),
        ("/max_pool_with_index_gpu", ("/hip/max_pool_with_index_gpu", API_CAFFE2)),
        ("/THCCachingAllocator_gpu", ("/hip/THCCachingAllocator_gpu", API_CAFFE2)),
        ("/top_k_heap_selection", ("/hip/top_k_heap_selection", API_CAFFE2)),
        ("/top_k_radix_selection", ("/hip/top_k_radix_selection", API_CAFFE2)),
        ("/GpuAtomics", ("/hip/GpuAtomics", API_CAFFE2)),
        ("/GpuDefs", ("/hip/GpuDefs", API_CAFFE2)),
        ("/GpuScanUtils", ("/hip/GpuScanUtils", API_CAFFE2)),
        ("/GpuBitonicSort", ("/hip/GpuBitonicSort", API_CAFFE2)),
        ("/math/reduce.cuh", ("/math/hip/reduce.cuh", API_CAFFE2)),
        ("/sgd/adagrad_fused_op_gpu.cuh", ("/sgd/hip/adagrad_fused_op_gpu.cuh", API_CAFFE2)),
        ("/operators/segment_reduction_op_gpu.cuh", ("/operators/hip/segment_reduction_op_gpu.cuh", API_CAFFE2)),
        ("/gather_op.cuh", ("/hip/gather_op.cuh", API_CAFFE2)),
        ("caffe2/core/common_cudnn.h", ("caffe2/core/hip/common_miopen.h", API_CAFFE2)),
        ("REGISTER_CUDA_OPERATOR", ("REGISTER_HIP_OPERATOR", API_CAFFE2)),
        ("CUDA_1D_KERNEL_LOOP", ("HIP_1D_KERNEL_LOOP", API_CAFFE2)),
        ("CUDAContext", ("HIPContext", API_CAFFE2)),
        ("CAFFE_CUDA_NUM_THREADS", ("CAFFE_HIP_NUM_THREADS", API_CAFFE2)),
        ("HasCudaGPU", ("HasHipGPU", API_CAFFE2)),
        ("__expf", ("expf", API_CAFFE2)),
        ("CUBLAS_ENFORCE", ("HIPBLAS_ENFORCE", API_CAFFE2)),
        ("CUBLAS_CHECK", ("HIPBLAS_CHECK", API_CAFFE2)),
        ("cublas_handle", ("hipblas_handle", API_CAFFE2)),
        ("CURAND_ENFORCE", ("HIPRAND_ENFORCE", API_CAFFE2)),
        ("CURAND_CHECK", ("HIPRAND_CHECK", API_CAFFE2)),
        ("curandGenerateUniform", ("hiprandGenerateUniform", API_CAFFE2)),
        ("curand_generator", ("hiprand_generator", API_CAFFE2)),
        ("CaffeCudaGetDevice", ("CaffeHipGetDevice", API_CAFFE2)),
        # do not rename CUDA_KERNEL_ASSERT, lazyInitCUDA in caffe2 sources
        # the ordered dict guarantees this pattern will match first, before "CUDA"
        ("CUDA_KERNEL_ASSERT", ("CUDA_KERNEL_ASSERT", API_CAFFE2)),
        ("lazyInitCUDA", ("lazyInitCUDA", API_CAFFE2)),
        ("CUDA_VERSION", ("TORCH_HIP_VERSION", API_CAFFE2)),
        ("CUDA", ("HIP", API_CAFFE2)),
        ("Cuda", ("Hip", API_CAFFE2)),
        ("cuda_", ("hip_", API_CAFFE2)),
        ("_cuda", ("_hip", API_CAFFE2)),
        ("CUDNN", ("MIOPEN", API_CAFFE2)),
        ("CuDNN", ("MIOPEN", API_CAFFE2)),
        ("cudnn", ("miopen", API_CAFFE2)),
        ("namespace cuda", ("namespace hip", API_CAFFE2)),
        ("cuda::CUDAGuard", ("hip::HIPGuard", API_CAFFE2)),
        ("cuda::OptionalCUDAGuard", ("hip::OptionalHIPGuard", API_CAFFE2)),
        ("cuda::CUDAStreamGuard", ("hip::HIPStreamGuard", API_CAFFE2)),
        ("cuda::OptionalCUDAStreamGuard", ("hip::OptionalHIPStreamGuard", API_CAFFE2)),
        ("c10/cuda/CUDAGuard.h", ("c10/hip/HIPGuard.h", API_CAFFE2)),
        ("gloo/cuda", ("gloo/hip", API_CAFFE2)),
    ]
)

# We must tread very carefully here.  Blanket conversions like are done
# in CAFFE2_SPECIFIC_MAPPINGS are not presently supported on PyTorch,
# because a regex for CUDA will also match a filename like CUDAGuard.h,
# but the HIPIFY script doesn't presently move the file and so the substitution
# will be invalid.  Instead, we specifically list out every identifier
# and file from c10/cuda which may be used externally, and do substitutions this
# way.
#
# NB: if you want a transformation to ONLY apply to the c10/ directory,
# put it as API_CAFFE2
C10_MAPPINGS = collections.OrderedDict(
    [
        ("CUDA_VERSION", ("TORCH_HIP_VERSION", API_PYTORCH)),
        ("CUDA_LAUNCH_BLOCKING=1", ("AMD_SERIALIZE_KERNEL=3", API_C10)),
        ("CUDA_LAUNCH_BLOCKING", ("AMD_SERIALIZE_KERNEL", API_C10)),
        ("cuda::compat::", ("hip::compat::", API_C10)),
        ("c10/cuda/CUDAAlgorithm.h", ("c10/hip/HIPAlgorithm.h", API_C10)),
        ("c10/cuda/CUDADeviceAssertion.h", ("c10/hip/HIPDeviceAssertion.h", API_C10)),
        ("c10/cuda/CUDADeviceAssertionHost.h", ("c10/hip/HIPDeviceAssertionHost.h", API_C10)),
        ("c10/cuda/CUDAException.h", ("c10/hip/HIPException.h", API_C10)),
        ("c10/cuda/CUDAMacros.h", ("c10/hip/HIPMacros.h", API_C10)),
        ("c10/cuda/CUDAMathCompat.h", ("c10/hip/HIPMathCompat.h", API_C10)),
        ("c10/cuda/CUDAFunctions.h", ("c10/hip/HIPFunctions.h", API_C10)),
        ("c10/cuda/CUDAMiscFunctions.h", ("c10/hip/HIPMiscFunctions.h", API_C10)),
        ("c10/cuda/CUDAStream.h", ("c10/hip/HIPStream.h", API_C10)),
        ("c10/cuda/CUDAGraphsC10Utils.h", ("c10/hip/HIPGraphsC10Utils.h", API_C10)),
        ("c10/cuda/CUDAAllocatorConfig.h", ("c10/hip/HIPAllocatorConfig.h", API_C10)),
        ("c10/cuda/CUDACachingAllocator.h", ("c10/hip/HIPCachingAllocator.h", API_C10)),
        ("c10/cuda/impl/CUDATest.h", ("c10/hip/impl/HIPTest.h", API_C10)),
        ("c10/cuda/impl/CUDAGuardImpl.h", ("c10/hip/impl/HIPGuardImpl.h", API_C10)),
        (
            "c10/cuda/impl/cuda_cmake_macros.h",
            ("c10/hip/impl/hip_cmake_macros.h", API_C10),
        ),
        ("C10_CUDA_CHECK", ("C10_HIP_CHECK", API_C10)),
        ("C10_CUDA_CHECK_WARN", ("C10_HIP_CHECK_WARN", API_C10)),
        ("C10_CUDA_ERROR_HANDLED", ("C10_HIP_ERROR_HANDLED", API_C10)),
        ("C10_CUDA_IGNORE_ERROR", ("C10_HIP_IGNORE_ERROR", API_C10)),
        ("C10_CUDA_CLEAR_ERROR", ("C10_HIP_CLEAR_ERROR", API_C10)),
        ("c10::cuda", ("c10::hip", API_C10)),
        ("cuda::CUDAStream", ("hip::HIPStream", API_C10)),
        ("CUDAStream", ("HIPStream", API_C10)),
        # This substitution is not permissible, because there's another copy of this
        # function in torch/cuda.h
        # ("cuda::device_count", ("hip::device_count", API_C10)),
        ("cuda::current_device", ("hip::current_device", API_C10)),
        ("cuda::set_device", ("hip::set_device", API_C10)),
        ("cuda::device_synchronize", ("hip::device_synchronize", API_C10)),
        ("cuda::getStreamFromPool", ("hip::getStreamFromPool", API_C10)),
        ("getStreamFromPool", ("getStreamFromPool", API_C10)),
        ("cuda::getDefaultCUDAStream", ("hip::getDefaultHIPStream", API_C10)),
        ("getDefaultCUDAStream", ("getDefaultHIPStream", API_C10)),
        ("cuda::getCurrentCUDAStream", ("hip::getCurrentHIPStream", API_C10)),
        ("getCurrentCUDAStream", ("getCurrentHIPStream", API_C10)),
        ("cuda::get_cuda_check_prefix", ("hip::get_cuda_check_prefix", API_C10)),
        ("cuda::setCurrentCUDAStream", ("hip::setCurrentHIPStream", API_C10)),
        ("setCurrentCUDAStream", ("setCurrentHIPStream", API_C10)),
        ("cuda::CUDACachingAllocator", ("hip::HIPCachingAllocator", API_C10)),
        ("CUDACachingAllocator", ("HIPCachingAllocator", API_C10)),
        ("cuda::CUDAAllocatorConfig", ("hip::HIPAllocatorConfig", API_C10)),
        ("CUDAAllocatorConfig", ("HIPAllocatorConfig", API_C10)),
        ("pinned_use_cuda_host_register", ("pinned_use_hip_host_register", API_C10)),
        ("c10::cuda::CUDAAllocator", ("c10::hip::HIPAllocator", API_C10)),
        ("cuda::CUDAAllocator", ("hip::HIPAllocator", API_C10)),
        ("CUDAStreamCaptureModeGuard", ("HIPStreamCaptureModeGuard", API_C10)),
        ("cuda::CUDAStreamCaptureModeGuard", ("cuda::HIPStreamCaptureModeGuard", API_C10)),
        ("CUDAAllocator", ("HIPAllocator", API_C10)),
        ("C10_CUDA_KERNEL_LAUNCH_CHECK", ("C10_HIP_KERNEL_LAUNCH_CHECK", API_C10))
    ]
)

# NB: C10 mappings are more specific than Caffe2 mappings, so run them
# first
CUDA_TO_HIP_MAPPINGS = [
    CUDA_IDENTIFIER_MAP,
    CUDA_TYPE_NAME_MAP,
    CUDA_INCLUDE_MAP,
    CUDA_SPECIAL_MAP,
    C10_MAPPINGS,
    PYTORCH_SPECIFIC_MAPPINGS,
    CAFFE2_SPECIFIC_MAPPINGS,
]
