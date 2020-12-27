import collections

from . import constants as c

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
    ]
)

CUDA_TYPE_NAME_MAP = collections.OrderedDict(
    [
        ("CUresult", ("hipError_t", c.CONV_TYPE, c.API_DRIVER)),
        ("cudaError_t", ("hipError_t", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "CUDA_ARRAY3D_DESCRIPTOR",
            ("HIP_ARRAY3D_DESCRIPTOR", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUDA_ARRAY_DESCRIPTOR", ("HIP_ARRAY_DESCRIPTOR", c.CONV_TYPE, c.API_DRIVER)),
        ("CUDA_MEMCPY2D", ("hip_Memcpy2D", c.CONV_TYPE, c.API_DRIVER)),
        ("CUDA_MEMCPY3D", ("HIP_MEMCPY3D", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUDA_MEMCPY3D_PEER",
            ("HIP_MEMCPY3D_PEER", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_POINTER_ATTRIBUTE_P2P_TOKENS",
            (
                "HIP_POINTER_ATTRIBUTE_P2P_TOKENS",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_RESOURCE_DESC",
            ("HIP_RESOURCE_DESC", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_RESOURCE_VIEW_DESC",
            ("HIP_RESOURCE_VIEW_DESC", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUipcEventHandle",
            ("hipIpcEventHandle", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUipcMemHandle", ("hipIpcMemHandle", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUaddress_mode", ("hipAddress_mode", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUarray_cubemap_face",
            ("hipArray_cubemap_face", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUarray_format", ("hipArray_format", c.CONV_TYPE, c.API_DRIVER)),
        ("CUcomputemode", ("hipComputemode", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUmem_advise", ("hipMemAdvise", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUmem_range_attribute",
            ("hipMemRangeAttribute", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUctx_flags", ("hipCctx_flags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUdevice", ("hipDevice_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUdevice_attribute_enum", ("hipDeviceAttribute_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUdevice_attribute", ("hipDeviceAttribute_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUdeviceptr", ("hipDeviceptr_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUarray_st", ("hipArray", c.CONV_TYPE, c.API_DRIVER)),
        ("CUarray", ("hipArray *", c.CONV_TYPE, c.API_DRIVER)),
        ("CUdevprop_st", ("hipDeviceProp_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUdevprop", ("hipDeviceProp_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUfunction", ("hipFunction_t", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CUgraphicsResource",
            ("hipGraphicsResource_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUmipmappedArray",
            ("hipMipmappedArray_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUfunction_attribute",
            ("hipFuncAttribute_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUfunction_attribute_enum",
            ("hipFuncAttribute_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsMapResourceFlags",
            ("hipGraphicsMapFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsMapResourceFlags_enum",
            ("hipGraphicsMapFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsRegisterFlags",
            ("hipGraphicsRegisterFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsRegisterFlags_enum",
            ("hipGraphicsRegisterFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUoccupancy_flags",
            ("hipOccupancyFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUoccupancy_flags_enum",
            ("hipOccupancyFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUfunc_cache_enum", ("hipFuncCache", c.CONV_TYPE, c.API_DRIVER)),
        ("CUfunc_cache", ("hipFuncCache", c.CONV_TYPE, c.API_DRIVER)),
        ("CUipcMem_flags", ("hipIpcMemFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUipcMem_flags_enum",
            ("hipIpcMemFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUjit_cacheMode", ("hipJitCacheMode", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUjit_cacheMode_enum",
            ("hipJitCacheMode", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUjit_fallback", ("hipJitFallback", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUjit_fallback_enum",
            ("hipJitFallback", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUjit_option", ("hipJitOption", c.CONV_JIT, c.API_DRIVER)),
        ("CUjit_option_enum", ("hipJitOption", c.CONV_JIT, c.API_DRIVER)),
        ("CUjit_target", ("hipJitTarget", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUjit_target_enum", ("hipJitTarget", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUjitInputType", ("hipJitInputType", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUjitInputType_enum",
            ("hipJitInputType", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUlimit", ("hipLimit_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUlimit_enum", ("hipLimit_t", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CUmemAttach_flags",
            ("hipMemAttachFlags_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUmemAttach_flags_enum",
            ("hipMemAttachFlags_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUmemorytype", ("hipMemType_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUmemorytype_enum", ("hipMemType_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUresourcetype", ("hipResourceType", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUresourcetype_enum",
            ("hipResourceType", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUresourceViewFormat", ("hipResourceViewFormat", c.CONV_TEX, c.API_DRIVER)),
        ("CUresourceViewFormat_enum", ("hipResourceViewFormat", c.CONV_TEX, c.API_DRIVER)),
        ("CUsharedconfig", ("hipSharedMemConfig", c.CONV_TYPE, c.API_DRIVER)),
        ("CUsharedconfig_enum", ("hipSharedMemConfig", c.CONV_TYPE, c.API_DRIVER)),
        ("CUcontext", ("hipCtx_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUmodule", ("hipModule_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUstream", ("hipStream_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUstream_st", ("ihipStream_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUstreamCallback", ("hipStreamCallback_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUsurfObject", ("hipSurfaceObject", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUsurfref",
            ("hipSurfaceReference_t", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUtexObject", ("hipTextureObject_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUtexref", ("textureReference", c.CONV_TYPE, c.API_DRIVER)),
        ("CUstream_flags", ("hipStreamFlags", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CUstreamWaitValue_flags",
            ("hipStreamWaitValueFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUstreamWriteValue_flags",
            ("hipStreamWriteValueFlags", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUstreamBatchMemOpType",
            ("hipStreamBatchMemOpType", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUdevice_P2PAttribute",
            ("hipDeviceP2PAttribute", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUevent", ("hipEvent_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUevent_st", ("ihipEvent_t", c.CONV_TYPE, c.API_DRIVER)),
        ("CUevent_flags", ("hipEventFlags", c.CONV_EVENT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUfilter_mode", ("hipTextureFilterMode", c.CONV_TEX, c.API_DRIVER)),
        ("CUGLDeviceList", ("hipGLDeviceList", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("CUGLmap_flags", ("hipGLMapFlags", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUd3d9DeviceList",
            ("hipD3D9DeviceList", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUd3d9map_flags",
            ("hipD3D9MapFlags", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUd3d9register_flags",
            ("hipD3D9RegisterFlags", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10DeviceList",
            ("hipd3d10DeviceList", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10map_flags",
            ("hipD3D10MapFlags", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10register_flags",
            ("hipD3D10RegisterFlags", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUd3d11DeviceList",
            ("hipd3d11DeviceList", c.CONV_D3D11, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUeglStreamConnection_st",
            ("hipEglStreamConnection", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUeglStreamConnection",
            ("hipEglStreamConnection", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "libraryPropertyType_t",
            ("hipLibraryPropertyType_t", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "libraryPropertyType",
            ("hipLibraryPropertyType_t", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaStreamCallback_t", ("hipStreamCallback_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaArray", ("hipArray", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaArray_t", ("hipArray_t", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaArray_const_t", ("hipArray_const_t", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMipmappedArray_t", ("hipMipmappedArray_t", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMipmappedArray_const_t",
            ("hipMipmappedArray_const_t", c.CONV_MEM, c.API_RUNTIME),
        ),
        ("cudaArrayDefault", ("hipArrayDefault", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaArrayLayered", ("hipArrayLayered", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaArraySurfaceLoadStore",
            ("hipArraySurfaceLoadStore", c.CONV_MEM, c.API_RUNTIME),
        ),
        ("cudaArrayCubemap", ("hipArrayCubemap", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaArrayTextureGather", ("hipArrayTextureGather", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemoryAdvise", ("hipMemAdvise", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        (
            "cudaMemRangeAttribute",
            ("hipMemRangeAttribute", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMemcpyKind", ("hipMemcpyKind", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemoryType", ("hipMemoryType", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaExtent", ("hipExtent", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaPitchedPtr", ("hipPitchedPtr", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaPos", ("hipPos", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaEvent_t", ("hipEvent_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaStream_t", ("hipStream_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaPointerAttributes", ("hipPointerAttribute_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaDeviceAttr", ("hipDeviceAttribute_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaDeviceProp", ("hipDeviceProp_t", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "cudaDeviceP2PAttr",
            ("hipDeviceP2PAttribute", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeMode",
            ("hipComputeMode", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaFuncCache", ("hipFuncCache_t", c.CONV_CACHE, c.API_RUNTIME)),
        (
            "cudaFuncAttributes",
            ("hipFuncAttributes", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaSharedMemConfig", ("hipSharedMemConfig", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaLimit", ("hipLimit_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaOutputMode", ("hipOutputMode", c.CONV_OTHER, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("cudaTextureReadMode", ("hipTextureReadMode", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaTextureFilterMode", ("hipTextureFilterMode", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaChannelFormatKind", ("hipChannelFormatKind", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaChannelFormatDesc", ("hipChannelFormatDesc", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResourceDesc", ("hipResourceDesc", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResourceViewDesc", ("hipResourceViewDesc", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaTextureDesc", ("hipTextureDesc", c.CONV_TEX, c.API_RUNTIME)),
        (
            "surfaceReference",
            ("hipSurfaceReference", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaTextureObject_t", ("hipTextureObject_t", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResourceType", ("hipResourceType", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResourceViewFormat", ("hipResourceViewFormat", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaTextureAddressMode", ("hipTextureAddressMode", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaSurfaceBoundaryMode",
            ("hipSurfaceBoundaryMode", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaSurfaceFormatMode",
            ("hipSurfaceFormatMode", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaTextureType1D", ("hipTextureType1D", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaTextureType2D", ("hipTextureType2D", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaTextureType3D", ("hipTextureType3D", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaTextureTypeCubemap", ("hipTextureTypeCubemap", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaTextureType1DLayered",
            ("hipTextureType1DLayered", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaTextureType2DLayered",
            ("hipTextureType2DLayered", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaTextureTypeCubemapLayered",
            ("hipTextureTypeCubemapLayered", c.CONV_TEX, c.API_RUNTIME),
        ),
        ("cudaIpcEventHandle_t", ("hipIpcEventHandle_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaIpcEventHandle_st", ("hipIpcEventHandle_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaIpcMemHandle_t", ("hipIpcMemHandle_t", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaIpcMemHandle_st", ("hipIpcMemHandle_t", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "cudaGraphicsCubeFace",
            ("hipGraphicsCubeFace", c.CONV_GRAPHICS, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsMapFlags",
            ("hipGraphicsMapFlags", c.CONV_GRAPHICS, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsRegisterFlags",
            ("hipGraphicsRegisterFlags", c.CONV_GRAPHICS, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLDeviceList",
            ("hipGLDeviceList", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaGLMapFlags", ("hipGLMapFlags", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        (
            "cudaD3D9DeviceList",
            ("hipD3D9DeviceList", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9MapFlags",
            ("hipD3D9MapFlags", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9RegisterFlags",
            ("hipD3D9RegisterFlags", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10DeviceList",
            ("hipd3d10DeviceList", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10MapFlags",
            ("hipD3D10MapFlags", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10RegisterFlags",
            ("hipD3D10RegisterFlags", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11DeviceList",
            ("hipd3d11DeviceList", c.CONV_D3D11, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaEglStreamConnection",
            ("hipEglStreamConnection", c.CONV_EGL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cublasHandle_t", ("rocblas_handle", c.CONV_TYPE, c.API_BLAS)),
        ("cublasOperation_t", ("rocblas_operation", c.CONV_TYPE, c.API_BLAS)),
        ("cublasStatus_t", ("rocblas_status", c.CONV_TYPE, c.API_BLAS)),
        ("cublasFillMode_t", ("rocblas_fill", c.CONV_TYPE, c.API_BLAS)),
        ("cublasDiagType_t", ("rocblas_diagonal", c.CONV_TYPE, c.API_BLAS)),
        ("cublasSideMode_t", ("rocblas_side", c.CONV_TYPE, c.API_BLAS)),
        ("cublasPointerMode_t", ("rocblas_pointer_mode", c.CONV_TYPE, c.API_BLAS)),
        (
            "cublasAtomicsMode_t",
            ("rocblas_atomics_mode", c.CONV_TYPE, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDataType_t",
            ("rocblas_data_type", c.CONV_TYPE, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("curandStatus", ("hiprandStatus_t", c.CONV_TYPE, c.API_RAND)),
        ("curandStatus_t", ("hiprandStatus_t", c.CONV_TYPE, c.API_RAND)),
        ("curandRngType", ("hiprandRngType_t", c.CONV_TYPE, c.API_RAND)),
        ("curandRngType_t", ("hiprandRngType_t", c.CONV_TYPE, c.API_RAND)),
        ("curandGenerator_st", ("hiprandGenerator_st", c.CONV_TYPE, c.API_RAND)),
        ("curandGenerator_t", ("hiprandGenerator_t", c.CONV_TYPE, c.API_RAND)),
        (
            "curandDirectionVectorSet",
            ("hiprandDirectionVectorSet_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDirectionVectorSet_t",
            ("hiprandDirectionVectorSet_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("curandOrdering", ("hiprandOrdering_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED)),
        (
            "curandOrdering_t",
            ("hiprandOrdering_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDistribution_st",
            ("hiprandDistribution_st", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2V_st",
            ("hiprandDistribution_st", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDistribution_t",
            ("hiprandDistribution_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2V_t",
            ("hiprandDistribution_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionShift_st",
            ("hiprandDistributionShift_st", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionShift_t",
            ("hiprandDistributionShift_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionM2Shift_st",
            ("hiprandDistributionM2Shift_st", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionM2Shift_t",
            ("hiprandDistributionM2Shift_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2_st",
            ("hiprandHistogramM2_st", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2_t",
            ("hiprandHistogramM2_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2K_st",
            ("hiprandHistogramM2K_st", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2K_t",
            ("hiprandHistogramM2K_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandDiscreteDistribution_st",
            ("hiprandDiscreteDistribution_st", c.CONV_TYPE, c.API_RAND),
        ),
        (
            "curandDiscreteDistribution_t",
            ("hiprandDiscreteDistribution_t", c.CONV_TYPE, c.API_RAND),
        ),
        ("curandMethod", ("hiprandMethod_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED)),
        ("curandMethod_t", ("hiprandMethod_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED)),
        (
            "curandDirectionVectors32_t",
            ("hiprandDirectionVectors32_t", c.CONV_TYPE, c.API_RAND),
        ),
        (
            "curandDirectionVectors64_t",
            ("hiprandDirectionVectors64_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("curandStateMtgp32_t", ("hiprandStateMtgp32_t", c.CONV_TYPE, c.API_RAND)),
        ("curandStateMtgp32", ("hiprandStateMtgp32_t", c.CONV_TYPE, c.API_RAND)),
        (
            "curandStateScrambledSobol64_t",
            ("hiprandStateScrambledSobol64_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandStateSobol64_t",
            ("hiprandStateSobol64_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandStateScrambledSobol32_t",
            ("hiprandStateScrambledSobol32_t", c.CONV_TYPE, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("curandStateSobol32_t", ("hiprandStateSobol32_t", c.CONV_TYPE, c.API_RAND)),
        ("curandStateMRG32k3a_t", ("hiprandStateMRG32k3a_t", c.CONV_TYPE, c.API_RAND)),
        (
            "curandStatePhilox4_32_10_t",
            ("hiprandStatePhilox4_32_10_t", c.CONV_TYPE, c.API_RAND),
        ),
        ("curandStateXORWOW_t", ("hiprandStateXORWOW_t", c.CONV_TYPE, c.API_RAND)),
        ("curandState_t", ("hiprandState_t", c.CONV_TYPE, c.API_RAND)),
        ("curandState", ("hiprandState_t", c.CONV_TYPE, c.API_RAND)),
    ]
)

CUDA_INCLUDE_MAP = collections.OrderedDict(
    [
        # since pytorch uses "\b{pattern}\b" as the actual re pattern,
        # patterns listed here have to begin and end with alnum chars
        (
            "include <cuda.h",
            ("include <hip/hip_runtime.h", c.CONV_INCLUDE_CUDA_MAIN_H, c.API_DRIVER),
        ),
        (
            'include "cuda.h',
            ('include "hip/hip_runtime.h', c.CONV_INCLUDE_CUDA_MAIN_H, c.API_DRIVER),
        ),
        (
            "cuda_runtime.h",
            ("hip/hip_runtime.h", c.CONV_INCLUDE_CUDA_MAIN_H, c.API_RUNTIME),
        ),
        ("cuda_runtime_api.h", ("hip/hip_runtime_api.h", c.CONV_INCLUDE, c.API_RUNTIME)),
        (
            "channel_descriptor.h",
            ("hip/channel_descriptor.h", c.CONV_INCLUDE, c.API_RUNTIME),
        ),
        ("device_functions.h", ("hip/device_functions.h", c.CONV_INCLUDE, c.API_RUNTIME)),
        ("driver_types.h", ("hip/driver_types.h", c.CONV_INCLUDE, c.API_RUNTIME)),
        ("cuComplex.h", ("hip/hip_complex.h", c.CONV_INCLUDE, c.API_RUNTIME)),
        ("cuda_fp16.h", ("hip/hip_fp16.h", c.CONV_INCLUDE, c.API_RUNTIME)),
        (
            "cuda_texture_types.h",
            ("hip/hip_texture_types.h", c.CONV_INCLUDE, c.API_RUNTIME),
        ),
        ("vector_types.h", ("hip/hip_vector_types.h", c.CONV_INCLUDE, c.API_RUNTIME)),
        ("cublas.h", ("rocblas.h", c.CONV_INCLUDE_CUDA_MAIN_H, c.API_BLAS)),
        ("cublas_v2.h", ("rocblas.h", c.CONV_INCLUDE_CUDA_MAIN_H, c.API_BLAS)),
        ("curand.h", ("hiprand/hiprand.h", c.CONV_INCLUDE_CUDA_MAIN_H, c.API_RAND)),
        ("curand_kernel.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_discrete.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_discrete2.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_globals.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_lognormal.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_mrg32k3a.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_mtgp32.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_mtgp32_host.h", ("hiprand/hiprand_mtgp32_host.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_mtgp32_kernel.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        (
            "curand_mtgp32dc_p_11213.h",
            ("rocrand/rocrand_mtgp32_11213.h", c.CONV_INCLUDE, c.API_RAND),
        ),
        ("curand_normal.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_normal_static.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_philox4x32_x.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_poisson.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_precalc.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("curand_uniform.h", ("hiprand/hiprand_kernel.h", c.CONV_INCLUDE, c.API_RAND)),
        ("cusparse.h", ("hipsparse.h", c.CONV_INCLUDE, c.API_RAND)),
        ("cufft.h", ("hipfft.h", c.CONV_INCLUDE, c.API_BLAS)),
        ("cufftXt.h", ("hipfft.h", c.CONV_INCLUDE, c.API_BLAS)),
        # PyTorch also has a source file named "nccl.h", so we need to "<"">" to differentiate
        ("<nccl.h>", ("<rccl.h>", c.CONV_INCLUDE, c.API_RUNTIME)),
        ("nvrtc.h", ("hip/hiprtc.h", c.CONV_INCLUDE, c.API_RTC)),
        ("thrust/system/cuda", ("thrust/system/hip", c.CONV_INCLUDE, c.API_BLAS)),
        ("cub/util_allocator.cuh", ("hipcub/hipcub.hpp", c.CONV_INCLUDE, c.API_BLAS)),
        ("cub/block/block_reduce.cuh", ("hipcub/hipcub.hpp", c.CONV_INCLUDE, c.API_BLAS)),
        ("cub/cub.cuh", ("hipcub/hipcub.hpp", c.CONV_INCLUDE, c.API_BLAS)),
        ("cub/block/block_load.cuh", ("hipcub/hipcub.hpp", c.CONV_INCLUDE, c.API_BLAS)),
        ("cub/device/device_radix_sort.cuh", ("hipcub/hipcub.hpp", c.CONV_INCLUDE, c.API_BLAS)),
        ("cub/device/device_reduce.cuh", ("hipcub/hipcub.hpp", c.CONV_INCLUDE, c.API_BLAS)),
        ("cub/device/device_scan.cuh", ("hipcub/hipcub.hpp", c.CONV_INCLUDE, c.API_BLAS)),
        ("nvToolsExt.h", ("roctracer/roctx.h", c.CONV_INCLUDE, c.API_ROCTX)),
    ]
)

CUDA_IDENTIFIER_MAP = collections.OrderedDict(
    [
        ("__CUDACC__", ("__HIPCC__", c.CONV_DEF, c.API_RUNTIME)),
        (
            "CUDA_ERROR_INVALID_CONTEXT",
            ("hipErrorInvalidContext", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            ("hipErrorContextAlreadyCurrent", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CUDA_ERROR_ARRAY_IS_MAPPED",
            ("hipErrorArrayIsMapped", c.CONV_TYPE, c.API_DRIVER),
        ),
        ("CUDA_ERROR_ALREADY_MAPPED", ("hipErrorAlreadyMapped", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CUDA_ERROR_ALREADY_ACQUIRED",
            ("hipErrorAlreadyAcquired", c.CONV_TYPE, c.API_DRIVER),
        ),
        ("CUDA_ERROR_NOT_MAPPED", ("hipErrorNotMapped", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
            ("hipErrorNotMappedAsArray", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
            ("hipErrorNotMappedAsPointer", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
            ("hipErrorContextAlreadyInUse", c.CONV_TYPE, c.API_DRIVER),
        ),
        ("CUDA_ERROR_INVALID_SOURCE", ("hipErrorInvalidSource", c.CONV_TYPE, c.API_DRIVER)),
        ("CUDA_ERROR_FILE_NOT_FOUND", ("hipErrorFileNotFound", c.CONV_TYPE, c.API_DRIVER)),
        ("CUDA_ERROR_NOT_FOUND", ("hipErrorNotFound", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
            (
                "hipErrorLaunchIncompatibleTexturing",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            ("hipErrorPrimaryContextActive", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_CONTEXT_IS_DESTROYED",
            ("hipErrorContextIsDestroyed", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NOT_PERMITTED",
            ("hipErrorNotPermitted", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NOT_SUPPORTED",
            ("hipErrorNotSupported", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMissingConfiguration",
            ("hipErrorMissingConfiguration", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorPriorLaunchFailure",
            ("hipErrorPriorLaunchFailure", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidDeviceFunction",
            ("hipErrorInvalidDeviceFunction", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidConfiguration",
            ("hipErrorInvalidConfiguration", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidPitchValue",
            ("hipErrorInvalidPitchValue", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidSymbol",
            ("hipErrorInvalidSymbol", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidHostPointer",
            ("hipErrorInvalidHostPointer", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidDevicePointer",
            ("hipErrorInvalidDevicePointer", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaErrorInvalidTexture",
            ("hipErrorInvalidTexture", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidTextureBinding",
            ("hipErrorInvalidTextureBinding", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidChannelDescriptor",
            (
                "hipErrorInvalidChannelDescriptor",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorInvalidMemcpyDirection",
            ("hipErrorInvalidMemcpyDirection", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorAddressOfConstant",
            ("hipErrorAddressOfConstant", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTextureFetchFailed",
            ("hipErrorTextureFetchFailed", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTextureNotBound",
            ("hipErrorTextureNotBound", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSynchronizationError",
            ("hipErrorSynchronizationError", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidFilterSetting",
            ("hipErrorInvalidFilterSetting", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidNormSetting",
            ("hipErrorInvalidNormSetting", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMixedDeviceExecution",
            ("hipErrorMixedDeviceExecution", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNotYetImplemented",
            ("hipErrorNotYetImplemented", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMemoryValueTooLarge",
            ("hipErrorMemoryValueTooLarge", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInsufficientDriver",
            ("hipErrorInsufficientDriver", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSetOnActiveProcess",
            ("hipErrorSetOnActiveProcess", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidSurface",
            ("hipErrorInvalidSurface", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateVariableName",
            ("hipErrorDuplicateVariableName", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateTextureName",
            ("hipErrorDuplicateTextureName", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateSurfaceName",
            ("hipErrorDuplicateSurfaceName", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDevicesUnavailable",
            ("hipErrorDevicesUnavailable", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorIncompatibleDriverContext",
            (
                "hipErrorIncompatibleDriverContext",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorDeviceAlreadyInUse",
            ("hipErrorDeviceAlreadyInUse", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchMaxDepthExceeded",
            ("hipErrorLaunchMaxDepthExceeded", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFileScopedTex",
            ("hipErrorLaunchFileScopedTex", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFileScopedSurf",
            ("hipErrorLaunchFileScopedSurf", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSyncDepthExceeded",
            ("hipErrorSyncDepthExceeded", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchPendingCountExceeded",
            (
                "hipErrorLaunchPendingCountExceeded",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorNotPermitted",
            ("hipErrorNotPermitted", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNotSupported",
            ("hipErrorNotSupported", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorStartupFailure",
            ("hipErrorStartupFailure", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorApiFailureBase",
            ("hipErrorApiFailureBase", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("CUDA_SUCCESS", ("hipSuccess", c.CONV_TYPE, c.API_DRIVER)),
        ("cudaSuccess", ("hipSuccess", c.CONV_TYPE, c.API_RUNTIME)),
        ("CUDA_ERROR_INVALID_VALUE", ("hipErrorInvalidValue", c.CONV_TYPE, c.API_DRIVER)),
        ("cudaErrorInvalidValue", ("hipErrorInvalidValue", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "CUDA_ERROR_OUT_OF_MEMORY",
            ("hipErrorMemoryAllocation", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorMemoryAllocation",
            ("hipErrorMemoryAllocation", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "CUDA_ERROR_NOT_INITIALIZED",
            ("hipErrorNotInitialized", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorInitializationError",
            ("hipErrorInitializationError", c.CONV_TYPE, c.API_RUNTIME),
        ),
        ("CUDA_ERROR_DEINITIALIZED", ("hipErrorDeinitialized", c.CONV_TYPE, c.API_DRIVER)),
        (
            "cudaErrorCudartUnloading",
            ("hipErrorDeinitialized", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_DISABLED",
            ("hipErrorProfilerDisabled", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorProfilerDisabled",
            ("hipErrorProfilerDisabled", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
            ("hipErrorProfilerNotInitialized", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorProfilerNotInitialized",
            ("hipErrorProfilerNotInitialized", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_ALREADY_STARTED",
            ("hipErrorProfilerAlreadyStarted", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorProfilerAlreadyStarted",
            ("hipErrorProfilerAlreadyStarted", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
            ("hipErrorProfilerAlreadyStopped", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorProfilerAlreadyStopped",
            ("hipErrorProfilerAlreadyStopped", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_NO_DEVICE", ("hipErrorNoDevice", c.CONV_TYPE, c.API_DRIVER)),
        ("cudaErrorNoDevice", ("hipErrorNoDevice", c.CONV_TYPE, c.API_RUNTIME)),
        ("CUDA_ERROR_INVALID_DEVICE", ("hipErrorInvalidDevice", c.CONV_TYPE, c.API_DRIVER)),
        ("cudaErrorInvalidDevice", ("hipErrorInvalidDevice", c.CONV_TYPE, c.API_RUNTIME)),
        ("CUDA_ERROR_INVALID_IMAGE", ("hipErrorInvalidImage", c.CONV_TYPE, c.API_DRIVER)),
        (
            "cudaErrorInvalidKernelImage",
            ("hipErrorInvalidImage", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_MAP_FAILED", ("hipErrorMapFailed", c.CONV_TYPE, c.API_DRIVER)),
        (
            "cudaErrorMapBufferObjectFailed",
            ("hipErrorMapFailed", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_UNMAP_FAILED", ("hipErrorUnmapFailed", c.CONV_TYPE, c.API_DRIVER)),
        (
            "cudaErrorUnmapBufferObjectFailed",
            ("hipErrorUnmapFailed", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NO_BINARY_FOR_GPU",
            ("hipErrorNoBinaryForGpu", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorNoKernelImageForDevice",
            ("hipErrorNoBinaryForGpu", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_ECC_UNCORRECTABLE",
            ("hipErrorECCNotCorrectable", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorECCUncorrectable",
            ("hipErrorECCNotCorrectable", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_UNSUPPORTED_LIMIT",
            ("hipErrorUnsupportedLimit", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorUnsupportedLimit",
            ("hipErrorUnsupportedLimit", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
            ("hipErrorPeerAccessUnsupported", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessUnsupported",
            ("hipErrorPeerAccessUnsupported", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_PTX",
            ("hipErrorInvalidKernelFile", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorInvalidPtx",
            ("hipErrorInvalidKernelFile", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
            ("hipErrorInvalidGraphicsContext", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorInvalidGraphicsContext",
            ("hipErrorInvalidGraphicsContext", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NVLINK_UNCORRECTABLE",
            ("hipErrorNvlinkUncorrectable", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNvlinkUncorrectable",
            ("hipErrorNvlinkUncorrectable", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
            ("hipErrorSharedObjectSymbolNotFound", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorSharedObjectSymbolNotFound",
            (
                "hipErrorSharedObjectSymbolNotFound",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
            ("hipErrorSharedObjectInitFailed", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorSharedObjectInitFailed",
            ("hipErrorSharedObjectInitFailed", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_OPERATING_SYSTEM",
            ("hipErrorOperatingSystem", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorOperatingSystem",
            ("hipErrorOperatingSystem", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_HANDLE",
            ("hipErrorInvalidResourceHandle", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorInvalidResourceHandle",
            ("hipErrorInvalidResourceHandle", c.CONV_TYPE, c.API_RUNTIME),
        ),
        ("CUDA_ERROR_NOT_READY", ("hipErrorNotReady", c.CONV_TYPE, c.API_DRIVER)),
        ("cudaErrorNotReady", ("hipErrorNotReady", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            ("hipErrorIllegalAddress", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorIllegalAddress",
            ("hipErrorIllegalAddress", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            ("hipErrorLaunchOutOfResources", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorLaunchOutOfResources",
            ("hipErrorLaunchOutOfResources", c.CONV_TYPE, c.API_RUNTIME),
        ),
        ("CUDA_ERROR_LAUNCH_TIMEOUT", ("hipErrorLaunchTimeOut", c.CONV_TYPE, c.API_DRIVER)),
        (
            "cudaErrorLaunchTimeout",
            ("hipErrorLaunchTimeOut", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
            ("hipErrorPeerAccessAlreadyEnabled", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessAlreadyEnabled",
            ("hipErrorPeerAccessAlreadyEnabled", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
            ("hipErrorPeerAccessNotEnabled", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessNotEnabled",
            ("hipErrorPeerAccessNotEnabled", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "CUDA_ERROR_ASSERT",
            ("hipErrorAssert", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorAssert",
            ("hipErrorAssert", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_TOO_MANY_PEERS",
            ("hipErrorTooManyPeers", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTooManyPeers",
            ("hipErrorTooManyPeers", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
            ("hipErrorHostMemoryAlreadyRegistered", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorHostMemoryAlreadyRegistered",
            ("hipErrorHostMemoryAlreadyRegistered", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
            ("hipErrorHostMemoryNotRegistered", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "cudaErrorHostMemoryNotRegistered",
            ("hipErrorHostMemoryNotRegistered", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "CUDA_ERROR_HARDWARE_STACK_ERROR",
            ("hipErrorHardwareStackError", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorHardwareStackError",
            ("hipErrorHardwareStackError", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            ("hipErrorIllegalInstruction", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorIllegalInstruction",
            ("hipErrorIllegalInstruction", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_MISALIGNED_ADDRESS",
            ("hipErrorMisalignedAddress", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMisalignedAddress",
            ("hipErrorMisalignedAddress", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_ADDRESS_SPACE",
            ("hipErrorInvalidAddressSpace", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidAddressSpace",
            ("hipErrorInvalidAddressSpace", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_PC",
            ("hipErrorInvalidPc", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidPc",
            ("hipErrorInvalidPc", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_LAUNCH_FAILED",
            ("hipErrorLaunchFailure", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFailure",
            ("hipErrorLaunchFailure", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_UNKNOWN",
            ("hipErrorUnknown", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cudaErrorUnknown", ("hipErrorUnknown", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "CU_TR_ADDRESS_MODE_WRAP",
            ("HIP_TR_ADDRESS_MODE_WRAP", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_CLAMP",
            ("HIP_TR_ADDRESS_MODE_CLAMP", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_MIRROR",
            ("HIP_TR_ADDRESS_MODE_MIRROR", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_BORDER",
            ("HIP_TR_ADDRESS_MODE_BORDER", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_X",
            ("HIP_CUBEMAP_FACE_POSITIVE_X", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_X",
            ("HIP_CUBEMAP_FACE_NEGATIVE_X", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_Y",
            ("HIP_CUBEMAP_FACE_POSITIVE_Y", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_Y",
            ("HIP_CUBEMAP_FACE_NEGATIVE_Y", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_Z",
            ("HIP_CUBEMAP_FACE_POSITIVE_Z", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_Z",
            ("HIP_CUBEMAP_FACE_NEGATIVE_Z", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT8",
            ("HIP_AD_FORMAT_UNSIGNED_INT8", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT16",
            ("HIP_AD_FORMAT_UNSIGNED_INT16", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT32",
            ("HIP_AD_FORMAT_UNSIGNED_INT32", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT8",
            ("HIP_AD_FORMAT_SIGNED_INT8", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT16",
            ("HIP_AD_FORMAT_SIGNED_INT16", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT32",
            ("HIP_AD_FORMAT_SIGNED_INT32", c.CONV_TYPE, c.API_DRIVER),
        ),
        ("CU_AD_FORMAT_HALF", ("HIP_AD_FORMAT_HALF", c.CONV_TYPE, c.API_DRIVER)),
        ("CU_AD_FORMAT_FLOAT", ("HIP_AD_FORMAT_FLOAT", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CU_COMPUTEMODE_DEFAULT",
            ("hipComputeModeDefault", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_EXCLUSIVE",
            ("hipComputeModeExclusive", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_PROHIBITED",
            ("hipComputeModeProhibited", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_EXCLUSIVE_PROCESS",
            ("hipComputeModeExclusiveProcess", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_SET_READ_MOSTLY",
            ("hipMemAdviseSetReadMostly", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_UNSET_READ_MOSTLY",
            ("hipMemAdviseUnsetReadMostly", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
            (
                "hipMemAdviseSetPreferredLocation",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
            (
                "hipMemAdviseUnsetPreferredLocation",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_ADVISE_SET_ACCESSED_BY",
            ("hipMemAdviseSetAccessedBy", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_UNSET_ACCESSED_BY",
            ("hipMemAdviseUnsetAccessedBy", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
            ("hipMemRangeAttributeReadMostly", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",
            (
                "hipMemRangeAttributePreferredLocation",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",
            ("hipMemRangeAttributeAccessedBy", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION",
            (
                "hipMemRangeAttributeLastPrefetchLocation",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_CTX_SCHED_AUTO",
            ("HIP_CTX_SCHED_AUTO", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_SPIN",
            ("HIP_CTX_SCHED_SPIN", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_YIELD",
            ("HIP_CTX_SCHED_YIELD", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_BLOCKING_SYNC",
            ("HIP_CTX_SCHED_BLOCKING_SYNC", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_BLOCKING_SYNC",
            ("HIP_CTX_BLOCKING_SYNC", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_MASK",
            ("HIP_CTX_SCHED_MASK", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_MAP_HOST",
            ("HIP_CTX_MAP_HOST", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_LMEM_RESIZE_TO_MAX",
            ("HIP_CTX_LMEM_RESIZE_TO_MAX", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_FLAGS_MASK",
            ("HIP_CTX_FLAGS_MASK", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_LAUNCH_PARAM_BUFFER_POINTER",
            ("HIP_LAUNCH_PARAM_BUFFER_POINTER", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_LAUNCH_PARAM_BUFFER_SIZE",
            ("HIP_LAUNCH_PARAM_BUFFER_SIZE", c.CONV_TYPE, c.API_DRIVER),
        ),
        ("CU_LAUNCH_PARAM_END", ("HIP_LAUNCH_PARAM_END", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CU_IPC_HANDLE_SIZE",
            ("HIP_LAUNCH_PARAM_END", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_DEVICEMAP",
            ("HIP_MEMHOSTALLOC_DEVICEMAP", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_PORTABLE",
            ("HIP_MEMHOSTALLOC_PORTABLE", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_WRITECOMBINED",
            ("HIP_MEMHOSTALLOC_WRITECOMBINED", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_DEVICEMAP",
            ("HIP_MEMHOSTREGISTER_DEVICEMAP", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_IOMEMORY",
            ("HIP_MEMHOSTREGISTER_IOMEMORY", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_PORTABLE",
            ("HIP_MEMHOSTREGISTER_PORTABLE", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_PARAM_TR_DEFAULT",
            ("HIP_PARAM_TR_DEFAULT", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_LEGACY",
            ("HIP_STREAM_LEGACY", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_PER_THREAD",
            ("HIP_STREAM_PER_THREAD", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSA_OVERRIDE_FORMAT",
            ("HIP_TRSA_OVERRIDE_FORMAT", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSF_NORMALIZED_COORDINATES",
            ("HIP_TRSF_NORMALIZED_COORDINATES", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSF_READ_AS_INTEGER",
            ("HIP_TRSF_READ_AS_INTEGER", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CU_TRSF_SRGB", ("HIP_TRSF_SRGB", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CUDA_ARRAY3D_2DARRAY",
            ("HIP_ARRAY3D_LAYERED", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_CUBEMAP",
            ("HIP_ARRAY3D_CUBEMAP", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_DEPTH_TEXTURE",
            ("HIP_ARRAY3D_DEPTH_TEXTURE", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_LAYERED",
            ("HIP_ARRAY3D_LAYERED", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_SURFACE_LDST",
            ("HIP_ARRAY3D_SURFACE_LDST", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_TEXTURE_GATHER",
            ("HIP_ARRAY3D_TEXTURE_GATHER", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxThreadsPerBlock",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
            ("hipDeviceAttributeMaxBlockDimX", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
            ("hipDeviceAttributeMaxBlockDimY", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
            ("hipDeviceAttributeMaxBlockDimZ", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
            ("hipDeviceAttributeMaxGridDimX", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",
            ("hipDeviceAttributeMaxGridDimY", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",
            ("hipDeviceAttributeMaxGridDimZ", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
            (
                "hipDeviceAttributeMaxSharedMemoryPerBlock",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK",
            (
                "hipDeviceAttributeMaxSharedMemoryPerBlock",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",
            (
                "hipDeviceAttributeTotalConstantMemory",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_WARP_SIZE",
            ("hipDeviceAttributeWarpSize", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_PITCH",
            ("hipDeviceAttributeMaxPitch", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxRegistersPerBlock",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxRegistersPerBlock",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CLOCK_RATE",
            ("hipDeviceAttributeClockRate", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",
            (
                "hipDeviceAttributeTextureAlignment",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP",
            (
                "hipDeviceAttributeAsyncEngineCount",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
            (
                "hipDeviceAttributeMultiprocessorCount",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",
            (
                "hipDeviceAttributeKernelExecTimeout",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_INTEGRATED",
            ("hipDeviceAttributeIntegrated", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",
            (
                "hipDeviceAttributeCanMapHostMemory",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE",
            ("hipDeviceAttributeComputeMode", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture3DWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture3DHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH",
            (
                "hipDeviceAttributeMaxTexture3DDepth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",
            (
                "hipDeviceAttributeSurfaceAlignment",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",
            ("hipDeviceAttributeConcurrentKernels", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_ECC_ENABLED",
            ("hipDeviceAttributeEccEnabled", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",
            ("hipDeviceAttributePciBusId", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
            ("hipDeviceAttributePciDeviceId", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TCC_DRIVER",
            ("hipDeviceAttributeTccDriver", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE",
            (
                "hipDeviceAttributeMemoryClockRate",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",
            ("hipDeviceAttributeMemoryBusWidth", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",
            ("hipDeviceAttributeL2CacheSize", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
            ("hipDeviceAttributeMaxThreadsPerMultiProcessor", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",
            (
                "hipDeviceAttributeAsyncEngineCount",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",
            (
                "hipDeviceAttributeUnifiedAddressing",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DLayeredWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTexture1DLayeredLayers",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER",
            (
                "hipDeviceAttributeCanTex2DGather",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DGatherWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DGatherHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DWidthAlternate",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DHeightAlternate",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DDepthAlternate",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
            ("hipDeviceAttributePciDomainId", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",
            (
                "hipDeviceAttributeTexturePitchAlignment",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH",
            (
                "hipDeviceAttributeMaxTextureCubemapWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredLayers",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface1DWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface2DWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface2DHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface3DWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface3DHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH",
            (
                "hipDeviceAttributeMaxSurface3DDepth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurface1DLayeredWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurface1DLayeredLayers",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurface2DLayeredWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface2DLayeredHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurface2DLayeredLayers",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH",
            (
                "hipDeviceAttributeMaxSurfaceCubemapWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DLinearWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLinearWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLinearHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH",
            (
                "hipDeviceAttributeMaxTexture2DLinearPitch",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedHeight",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
            ("hipDeviceAttributeComputeCapabilityMajor", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
            ("hipDeviceAttributeComputeCapabilityMinor", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DMipmappedWidth",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",
            (
                "hipDeviceAttributeStreamPrioritiesSupported",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",
            (
                "hipDeviceAttributeGlobalL1CacheSupported",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED",
            (
                "hipDeviceAttributeLocalL1CacheSupported",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
            (
                "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
                c.CONV_TYPE,
                c.API_DRIVER,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",
            (
                "hipDeviceAttributeMaxRegistersPerMultiprocessor",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY",
            ("hipDeviceAttributeManagedMemory", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",
            ("hipDeviceAttributeIsMultiGpuBoard", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID",
            (
                "hipDeviceAttributeMultiGpuBoardGroupId",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
            (
                "hipDeviceAttributeHostNativeAtomicSupported",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
            (
                "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
            (
                "hipDeviceAttributePageableMemoryAccess",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
            (
                "hipDeviceAttributeConcurrentManagedAccess",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED",
            (
                "hipDeviceAttributeComputePreemptionSupported",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
            (
                "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX",
            ("hipDeviceAttributeMax", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_CONTEXT",
            ("hipPointerAttributeContext", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_MEMORY_TYPE",
            ("hipPointerAttributeMemoryType", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_DEVICE_POINTER",
            (
                "hipPointerAttributeDevicePointer",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_POINTER_ATTRIBUTE_HOST_POINTER",
            ("hipPointerAttributeHostPointer", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_P2P_TOKENS",
            ("hipPointerAttributeP2pTokens", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS",
            ("hipPointerAttributeSyncMemops", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_BUFFER_ID",
            ("hipPointerAttributeBufferId", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_IS_MANAGED",
            ("hipPointerAttributeIsManaged", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
            (
                "hipFuncAttributeMaxThreadsPerBlocks",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
            ("hipFuncAttributeSharedSizeBytes", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
            ("hipFuncAttributeConstSizeBytes", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
            ("hipFuncAttributeLocalSizeBytes", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_NUM_REGS",
            ("hipFuncAttributeNumRegs", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_PTX_VERSION",
            ("hipFuncAttributePtxVersion", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_BINARY_VERSION",
            ("hipFuncAttributeBinaryVersion", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA",
            ("hipFuncAttributeCacheModeCA", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX",
            ("hipFuncAttributeMax", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE",
            ("hipGraphicsMapFlagsNone", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY",
            ("hipGraphicsMapFlagsReadOnly", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
            ("hipGraphicsMapFlagsWriteDiscard", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_NONE",
            ("hipGraphicsRegisterFlagsNone", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY",
            (
                "hipGraphicsRegisterFlagsReadOnly",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD",
            (
                "hipGraphicsRegisterFlagsWriteDiscard",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST",
            (
                "hipGraphicsRegisterFlagsSurfaceLoadStore",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER",
            (
                "hipGraphicsRegisterFlagsTextureGather",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_OCCUPANCY_DEFAULT",
            ("hipOccupancyDefault", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE",
            (
                "hipOccupancyDisableCachingOverride",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_FUNC_CACHE_PREFER_NONE",
            ("hipFuncCachePreferNone", c.CONV_CACHE, c.API_DRIVER),
        ),
        (
            "CU_FUNC_CACHE_PREFER_SHARED",
            ("hipFuncCachePreferShared", c.CONV_CACHE, c.API_DRIVER),
        ),
        ("CU_FUNC_CACHE_PREFER_L1", ("hipFuncCachePreferL1", c.CONV_CACHE, c.API_DRIVER)),
        (
            "CU_FUNC_CACHE_PREFER_EQUAL",
            ("hipFuncCachePreferEqual", c.CONV_CACHE, c.API_DRIVER),
        ),
        (
            "CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS",
            ("hipIpcMemLazyEnablePeerAccess", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CUDA_IPC_HANDLE_SIZE", ("HIP_IPC_HANDLE_SIZE", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CU_JIT_CACHE_OPTION_NONE",
            ("hipJitCacheModeOptionNone", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_CACHE_OPTION_CG",
            ("hipJitCacheModeOptionCG", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_CACHE_OPTION_CA",
            ("hipJitCacheModeOptionCA", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_PREFER_PTX",
            ("hipJitFallbackPreferPtx", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_PREFER_BINARY",
            ("hipJitFallbackPreferBinary", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CU_JIT_MAX_REGISTERS", ("hipJitOptionMaxRegisters", c.CONV_JIT, c.API_DRIVER)),
        (
            "CU_JIT_THREADS_PER_BLOCK",
            ("hipJitOptionThreadsPerBlock", c.CONV_JIT, c.API_DRIVER),
        ),
        ("CU_JIT_WALL_TIME", ("hipJitOptionWallTime", c.CONV_JIT, c.API_DRIVER)),
        ("CU_JIT_INFO_LOG_BUFFER", ("hipJitOptionInfoLogBuffer", c.CONV_JIT, c.API_DRIVER)),
        (
            "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES",
            ("hipJitOptionInfoLogBufferSizeBytes", c.CONV_JIT, c.API_DRIVER),
        ),
        (
            "CU_JIT_ERROR_LOG_BUFFER",
            ("hipJitOptionErrorLogBuffer", c.CONV_JIT, c.API_DRIVER),
        ),
        (
            "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES",
            ("hipJitOptionErrorLogBufferSizeBytes", c.CONV_JIT, c.API_DRIVER),
        ),
        (
            "CU_JIT_OPTIMIZATION_LEVEL",
            ("hipJitOptionOptimizationLevel", c.CONV_JIT, c.API_DRIVER),
        ),
        (
            "CU_JIT_TARGET_FROM_CUCONTEXT",
            ("hipJitOptionTargetFromContext", c.CONV_JIT, c.API_DRIVER),
        ),
        ("CU_JIT_TARGET", ("hipJitOptionTarget", c.CONV_JIT, c.API_DRIVER)),
        (
            "CU_JIT_FALLBACK_STRATEGY",
            ("hipJitOptionFallbackStrategy", c.CONV_JIT, c.API_DRIVER),
        ),
        (
            "CU_JIT_GENERATE_DEBUG_INFO",
            ("hipJitOptionGenerateDebugInfo", c.CONV_JIT, c.API_DRIVER),
        ),
        ("CU_JIT_LOG_VERBOSE", ("hipJitOptionLogVerbose", c.CONV_JIT, c.API_DRIVER)),
        (
            "CU_JIT_GENERATE_LINE_INFO",
            ("hipJitOptionGenerateLineInfo", c.CONV_JIT, c.API_DRIVER),
        ),
        ("CU_JIT_CACHE_MODE", ("hipJitOptionCacheMode", c.CONV_JIT, c.API_DRIVER)),
        ("CU_JIT_NEW_SM3X_OPT", ("hipJitOptionSm3xOpt", c.CONV_JIT, c.API_DRIVER)),
        ("CU_JIT_FAST_COMPILE", ("hipJitOptionFastCompile", c.CONV_JIT, c.API_DRIVER)),
        ("CU_JIT_NUM_OPTIONS", ("hipJitOptionNumOptions", c.CONV_JIT, c.API_DRIVER)),
        (
            "CU_TARGET_COMPUTE_10",
            ("hipJitTargetCompute10", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_11",
            ("hipJitTargetCompute11", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_12",
            ("hipJitTargetCompute12", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_13",
            ("hipJitTargetCompute13", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_20",
            ("hipJitTargetCompute20", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_21",
            ("hipJitTargetCompute21", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_30",
            ("hipJitTargetCompute30", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_32",
            ("hipJitTargetCompute32", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_35",
            ("hipJitTargetCompute35", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_37",
            ("hipJitTargetCompute37", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_50",
            ("hipJitTargetCompute50", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_52",
            ("hipJitTargetCompute52", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_53",
            ("hipJitTargetCompute53", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_60",
            ("hipJitTargetCompute60", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_61",
            ("hipJitTargetCompute61", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_62",
            ("hipJitTargetCompute62", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_CUBIN",
            ("hipJitInputTypeBin", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_PTX",
            ("hipJitInputTypePtx", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_FATBINARY",
            ("hipJitInputTypeFatBinary", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_OBJECT",
            ("hipJitInputTypeObject", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_LIBRARY",
            ("hipJitInputTypeLibrary", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_NUM_INPUT_TYPES",
            ("hipJitInputTypeNumInputTypes", c.CONV_JIT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_STACK_SIZE",
            ("hipLimitStackSize", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_PRINTF_FIFO_SIZE",
            ("hipLimitPrintfFifoSize", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_MALLOC_HEAP_SIZE",
            ("hipLimitMallocHeapSize", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH",
            ("hipLimitDevRuntimeSyncDepth", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT",
            (
                "hipLimitDevRuntimePendingLaunchCount",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_LIMIT_STACK_SIZE",
            ("hipLimitStackSize", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_GLOBAL",
            ("hipMemAttachGlobal", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_HOST",
            ("hipMemAttachHost", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_SINGLE",
            ("hipMemAttachSingle", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_HOST",
            ("hipMemTypeHost", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_DEVICE",
            ("hipMemTypeDevice", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_ARRAY",
            ("hipMemTypeArray", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_UNIFIED",
            ("hipMemTypeUnified", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_ARRAY",
            ("hipResourceTypeArray", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY",
            ("hipResourceTypeMipmappedArray", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_LINEAR",
            ("hipResourceTypeLinear", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_RESOURCE_TYPE_PITCH2D",
            ("hipResourceTypePitch2D", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CU_RES_VIEW_FORMAT_NONE", ("hipResViewFormatNone", c.CONV_TEX, c.API_DRIVER)),
        (
            "CU_RES_VIEW_FORMAT_UINT_1X8",
            ("hipResViewFormatUnsignedChar1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_2X8",
            ("hipResViewFormatUnsignedChar2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_4X8",
            ("hipResViewFormatUnsignedChar4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_1X8",
            ("hipResViewFormatSignedChar1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_2X8",
            ("hipResViewFormatSignedChar2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_4X8",
            ("hipResViewFormatSignedChar4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_1X16",
            ("hipResViewFormatUnsignedShort1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_2X16",
            ("hipResViewFormatUnsignedShort2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_4X16",
            ("hipResViewFormatUnsignedShort4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_1X16",
            ("hipResViewFormatSignedShort1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_2X16",
            ("hipResViewFormatSignedShort2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_4X16",
            ("hipResViewFormatSignedShort4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_1X32",
            ("hipResViewFormatUnsignedInt1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_2X32",
            ("hipResViewFormatUnsignedInt2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UINT_4X32",
            ("hipResViewFormatUnsignedInt4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_1X32",
            ("hipResViewFormatSignedInt1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_2X32",
            ("hipResViewFormatSignedInt2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SINT_4X32",
            ("hipResViewFormatSignedInt4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_1X16",
            ("hipResViewFormatHalf1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_2X16",
            ("hipResViewFormatHalf2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_4X16",
            ("hipResViewFormatHalf4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_1X32",
            ("hipResViewFormatFloat1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_2X32",
            ("hipResViewFormatFloat2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_FLOAT_4X32",
            ("hipResViewFormatFloat4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC1",
            ("hipResViewFormatUnsignedBlockCompressed1", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC2",
            ("hipResViewFormatUnsignedBlockCompressed2", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC3",
            ("hipResViewFormatUnsignedBlockCompressed3", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC4",
            ("hipResViewFormatUnsignedBlockCompressed4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SIGNED_BC4",
            ("hipResViewFormatSignedBlockCompressed4", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC5",
            ("hipResViewFormatUnsignedBlockCompressed5", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SIGNED_BC5",
            ("hipResViewFormatSignedBlockCompressed5", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC6H",
            ("hipResViewFormatUnsignedBlockCompressed6H", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_SIGNED_BC6H",
            ("hipResViewFormatSignedBlockCompressed6H", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_RES_VIEW_FORMAT_UNSIGNED_BC7",
            ("hipResViewFormatUnsignedBlockCompressed7", c.CONV_TEX, c.API_DRIVER),
        ),
        (
            "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE",
            ("hipSharedMemBankSizeDefault", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE",
            ("hipSharedMemBankSizeFourByte", c.CONV_TYPE, c.API_DRIVER),
        ),
        (
            "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE",
            ("hipSharedMemBankSizeEightByte", c.CONV_TYPE, c.API_DRIVER),
        ),
        ("CU_STREAM_DEFAULT", ("hipStreamDefault", c.CONV_TYPE, c.API_DRIVER)),
        ("CU_STREAM_NON_BLOCKING", ("hipStreamNonBlocking", c.CONV_TYPE, c.API_DRIVER)),
        (
            "CU_STREAM_WAIT_VALUE_GEQ",
            ("hipStreamWaitValueGeq", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WAIT_VALUE_EQ",
            ("hipStreamWaitValueEq", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WAIT_VALUE_AND",
            ("hipStreamWaitValueAnd", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WAIT_VALUE_FLUSH",
            ("hipStreamWaitValueFlush", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WRITE_VALUE_DEFAULT",
            ("hipStreamWriteValueDefault", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER",
            (
                "hipStreamWriteValueNoMemoryBarrier",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_STREAM_MEM_OP_WAIT_VALUE_32",
            ("hipStreamBatchMemOpWaitValue32", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_MEM_OP_WRITE_VALUE_32",
            ("hipStreamBatchMemOpWriteValue32", c.CONV_TYPE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES",
            (
                "hipStreamBatchMemOpFlushRemoteWrites",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGetErrorName",
            ("hipGetErrorName___", c.CONV_ERROR, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGetErrorString",
            ("hipGetErrorString___", c.CONV_ERROR, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuInit", ("hipInit", c.CONV_INIT, c.API_DRIVER)),
        ("cuDriverGetVersion", ("hipDriverGetVersion", c.CONV_VERSION, c.API_DRIVER)),
        ("cuCtxCreate_v2", ("hipCtxCreate", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxDestroy_v2", ("hipCtxDestroy", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxGetApiVersion", ("hipCtxGetApiVersion", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxGetCacheConfig", ("hipCtxGetCacheConfig", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxGetCurrent", ("hipCtxGetCurrent", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxGetDevice", ("hipCtxGetDevice", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxGetFlags", ("hipCtxGetFlags", c.CONV_CONTEXT, c.API_DRIVER)),
        (
            "cuCtxGetLimit",
            ("hipCtxGetLimit", c.CONV_CONTEXT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuCtxGetSharedMemConfig",
            ("hipCtxGetSharedMemConfig", c.CONV_CONTEXT, c.API_DRIVER),
        ),
        (
            "cuCtxGetStreamPriorityRange",
            ("hipCtxGetStreamPriorityRange", c.CONV_CONTEXT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuCtxPopCurrent_v2", ("hipCtxPopCurrent", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxPushCurrent_v2", ("hipCtxPushCurrent", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxSetCacheConfig", ("hipCtxSetCacheConfig", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxSetCurrent", ("hipCtxSetCurrent", c.CONV_CONTEXT, c.API_DRIVER)),
        (
            "cuCtxSetLimit",
            ("hipCtxSetLimit", c.CONV_CONTEXT, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuCtxSetSharedMemConfig",
            ("hipCtxSetSharedMemConfig", c.CONV_CONTEXT, c.API_DRIVER),
        ),
        ("cuCtxSynchronize", ("hipCtxSynchronize", c.CONV_CONTEXT, c.API_DRIVER)),
        ("cuCtxAttach", ("hipCtxAttach", c.CONV_CONTEXT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuCtxDetach", ("hipCtxDetach", c.CONV_CONTEXT, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuCtxEnablePeerAccess", ("hipCtxEnablePeerAccess", c.CONV_PEER, c.API_DRIVER)),
        ("cuCtxDisablePeerAccess", ("hipCtxDisablePeerAccess", c.CONV_PEER, c.API_DRIVER)),
        ("cuDeviceCanAccessPeer", ("hipDeviceCanAccessPeer", c.CONV_PEER, c.API_DRIVER)),
        (
            "cuDeviceGetP2PAttribute",
            ("hipDeviceGetP2PAttribute", c.CONV_PEER, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuDevicePrimaryCtxGetState",
            ("hipDevicePrimaryCtxGetState", c.CONV_CONTEXT, c.API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxRelease",
            ("hipDevicePrimaryCtxRelease", c.CONV_CONTEXT, c.API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxReset",
            ("hipDevicePrimaryCtxReset", c.CONV_CONTEXT, c.API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxRetain",
            ("hipDevicePrimaryCtxRetain", c.CONV_CONTEXT, c.API_DRIVER),
        ),
        (
            "cuDevicePrimaryCtxSetFlags",
            ("hipDevicePrimaryCtxSetFlags", c.CONV_CONTEXT, c.API_DRIVER),
        ),
        ("cuDeviceGet", ("hipGetDevice", c.CONV_DEVICE, c.API_DRIVER)),
        ("cuDeviceGetName", ("hipDeviceGetName", c.CONV_DEVICE, c.API_DRIVER)),
        ("cuDeviceGetCount", ("hipGetDeviceCount", c.CONV_DEVICE, c.API_DRIVER)),
        ("cuDeviceGetAttribute", ("hipDeviceGetAttribute", c.CONV_DEVICE, c.API_DRIVER)),
        ("cuDeviceGetPCIBusId", ("hipDeviceGetPCIBusId", c.CONV_DEVICE, c.API_DRIVER)),
        ("cuDeviceGetByPCIBusId", ("hipDeviceGetByPCIBusId", c.CONV_DEVICE, c.API_DRIVER)),
        ("cuDeviceTotalMem_v2", ("hipDeviceTotalMem", c.CONV_DEVICE, c.API_DRIVER)),
        (
            "cuDeviceComputeCapability",
            ("hipDeviceComputeCapability", c.CONV_DEVICE, c.API_DRIVER),
        ),
        ("cuDeviceGetProperties", ("hipGetDeviceProperties", c.CONV_DEVICE, c.API_DRIVER)),
        ("cuLinkAddData", ("hipLinkAddData", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuLinkAddFile", ("hipLinkAddFile", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuLinkComplete",
            ("hipLinkComplete", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuLinkCreate", ("hipLinkCreate", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuLinkDestroy", ("hipLinkDestroy", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuModuleGetFunction", ("hipModuleGetFunction", c.CONV_MODULE, c.API_DRIVER)),
        ("cuModuleGetGlobal_v2", ("hipModuleGetGlobal", c.CONV_MODULE, c.API_DRIVER)),
        (
            "cuModuleGetSurfRef",
            ("hipModuleGetSurfRef", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuModuleGetTexRef", ("hipModuleGetTexRef", c.CONV_MODULE, c.API_DRIVER)),
        ("cuModuleLoad", ("hipModuleLoad", c.CONV_MODULE, c.API_DRIVER)),
        ("cuModuleLoadData", ("hipModuleLoadData", c.CONV_MODULE, c.API_DRIVER)),
        ("cuModuleLoadDataEx", ("hipModuleLoadDataEx", c.CONV_MODULE, c.API_DRIVER)),
        (
            "cuModuleLoadFatBinary",
            ("hipModuleLoadFatBinary", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuModuleUnload", ("hipModuleUnload", c.CONV_MODULE, c.API_DRIVER)),
        (
            "CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK",
            (
                "hipDeviceP2PAttributePerformanceRank",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED",
            (
                "hipDeviceP2PAttributeAccessSupported",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED",
            (
                "hipDeviceP2PAttributeNativeAtomicSupported",
                c.CONV_TYPE,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("CU_EVENT_DEFAULT", ("hipEventDefault", c.CONV_EVENT, c.API_DRIVER)),
        ("CU_EVENT_BLOCKING_SYNC", ("hipEventBlockingSync", c.CONV_EVENT, c.API_DRIVER)),
        ("CU_EVENT_DISABLE_TIMING", ("hipEventDisableTiming", c.CONV_EVENT, c.API_DRIVER)),
        ("CU_EVENT_INTERPROCESS", ("hipEventInterprocess", c.CONV_EVENT, c.API_DRIVER)),
        ("cuEventCreate", ("hipEventCreate", c.CONV_EVENT, c.API_DRIVER)),
        ("cuEventDestroy_v2", ("hipEventDestroy", c.CONV_EVENT, c.API_DRIVER)),
        ("cuEventElapsedTime", ("hipEventElapsedTime", c.CONV_EVENT, c.API_DRIVER)),
        ("cuEventQuery", ("hipEventQuery", c.CONV_EVENT, c.API_DRIVER)),
        ("cuEventRecord", ("hipEventRecord", c.CONV_EVENT, c.API_DRIVER)),
        ("cuEventSynchronize", ("hipEventSynchronize", c.CONV_EVENT, c.API_DRIVER)),
        (
            "cuFuncGetAttribute",
            ("hipFuncGetAttribute", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuFuncSetCacheConfig", ("hipFuncSetCacheConfig", c.CONV_MODULE, c.API_DRIVER)),
        (
            "cuFuncSetSharedMemConfig",
            ("hipFuncSetSharedMemConfig", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuLaunchKernel", ("hipModuleLaunchKernel", c.CONV_MODULE, c.API_DRIVER)),
        (
            "cuFuncSetBlockShape",
            ("hipFuncSetBlockShape", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuFuncSetSharedSize",
            ("hipFuncSetSharedSize", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuLaunch", ("hipLaunch", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuLaunchGrid", ("hipLaunchGrid", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuLaunchGridAsync",
            ("hipLaunchGridAsync", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuParamSetf", ("hipParamSetf", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuParamSeti", ("hipParamSeti", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuParamSetSize",
            ("hipParamSetSize", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuParamSetSize",
            ("hipParamSetSize", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuParamSetv", ("hipParamSetv", c.CONV_MODULE, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            (
                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                c.CONV_OCCUPANCY,
                c.API_DRIVER,
            ),
        ),
        (
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            (
                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                c.CONV_OCCUPANCY,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuOccupancyMaxPotentialBlockSize",
            ("hipModuleOccupancyMaxPotentialBlockSize", c.CONV_OCCUPANCY, c.API_DRIVER),
        ),
        (
            "cuOccupancyMaxPotentialBlockSizeWithFlags",
            (
                "hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
                c.CONV_OCCUPANCY,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cuStreamAddCallback", ("hipStreamAddCallback", c.CONV_STREAM, c.API_DRIVER)),
        (
            "cuStreamAttachMemAsync",
            ("hipStreamAttachMemAsync", c.CONV_STREAM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuStreamCreate",
            ("hipStreamCreate__", c.CONV_STREAM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuStreamCreateWithPriority",
            ("hipStreamCreateWithPriority", c.CONV_STREAM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuStreamDestroy_v2", ("hipStreamDestroy", c.CONV_STREAM, c.API_DRIVER)),
        ("cuStreamGetFlags", ("hipStreamGetFlags", c.CONV_STREAM, c.API_DRIVER)),
        (
            "cuStreamGetPriority",
            ("hipStreamGetPriority", c.CONV_STREAM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuStreamQuery", ("hipStreamQuery", c.CONV_STREAM, c.API_DRIVER)),
        ("cuStreamSynchronize", ("hipStreamSynchronize", c.CONV_STREAM, c.API_DRIVER)),
        ("cuStreamWaitEvent", ("hipStreamWaitEvent", c.CONV_STREAM, c.API_DRIVER)),
        (
            "cuStreamWaitValue32",
            ("hipStreamWaitValue32", c.CONV_STREAM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuStreamWriteValue32",
            ("hipStreamWriteValue32", c.CONV_STREAM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuStreamBatchMemOp",
            ("hipStreamBatchMemOp", c.CONV_STREAM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuArray3DCreate", ("hipArray3DCreate", c.CONV_MEM, c.API_DRIVER)),
        (
            "cuArray3DGetDescriptor",
            ("hipArray3DGetDescriptor", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuArrayCreate", ("hipArrayCreate", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuArrayDestroy", ("hipArrayDestroy", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuArrayGetDescriptor",
            ("hipArrayGetDescriptor", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuIpcCloseMemHandle",
            ("hipIpcCloseMemHandle", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuIpcGetEventHandle",
            ("hipIpcGetEventHandle", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuIpcGetMemHandle",
            ("hipIpcGetMemHandle", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuIpcOpenEventHandle",
            ("hipIpcOpenEventHandle", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuIpcOpenMemHandle",
            ("hipIpcOpenMemHandle", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemAlloc_v2", ("hipMalloc", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemAllocHost", ("hipMemAllocHost", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemAllocManaged",
            ("hipMemAllocManaged", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMemAllocPitch",
            ("hipMemAllocPitch__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemcpy", ("hipMemcpy__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuMemcpy2D", ("hipMemcpy2D__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemcpy2DAsync",
            ("hipMemcpy2DAsync__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMemcpy2DUnaligned",
            ("hipMemcpy2DUnaligned", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemcpy3D", ("hipMemcpy3D__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemcpy3DAsync",
            ("hipMemcpy3DAsync__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMemcpy3DPeer",
            ("hipMemcpy3DPeer__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMemcpy3DPeerAsync",
            ("hipMemcpy3DPeerAsync__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemcpyAsync", ("hipMemcpyAsync__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuMemcpyAtoA", ("hipMemcpyAtoA", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuMemcpyAtoD", ("hipMemcpyAtoD", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuMemcpyAtoH", ("hipMemcpyAtoH", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemcpyAtoHAsync",
            ("hipMemcpyAtoHAsync", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemcpyDtoA", ("hipMemcpyDtoA", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuMemcpyDtoD_v2", ("hipMemcpyDtoD", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemcpyDtoDAsync_v2", ("hipMemcpyDtoDAsync", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemcpyDtoH_v2", ("hipMemcpyDtoH", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemcpyDtoHAsync_v2", ("hipMemcpyDtoHAsync", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemcpyHtoA", ("hipMemcpyHtoA", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemcpyHtoAAsync",
            ("hipMemcpyHtoAAsync", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemcpyHtoD_v2", ("hipMemcpyHtoD", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemcpyHtoDAsync_v2", ("hipMemcpyHtoDAsync", c.CONV_MEM, c.API_DRIVER)),
        (
            "cuMemcpyPeerAsync",
            ("hipMemcpyPeerAsync__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemcpyPeer", ("hipMemcpyPeer__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuMemFree_v2", ("hipFree", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemFreeHost", ("hipHostFree", c.CONV_MEM, c.API_DRIVER)),
        (
            "cuMemGetAddressRange",
            ("hipMemGetAddressRange", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemGetInfo_v2", ("hipMemGetInfo", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemHostAlloc", ("hipHostMalloc", c.CONV_MEM, c.API_DRIVER)),
        (
            "cuMemHostGetDevicePointer",
            ("hipMemHostGetDevicePointer", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMemHostGetFlags",
            ("hipMemHostGetFlags", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemHostRegister_v2", ("hipHostRegister", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemHostUnregister", ("hipHostUnregister", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemsetD16_v2", ("hipMemsetD16", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemsetD16Async",
            ("hipMemsetD16Async", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemsetD2D16_v2", ("hipMemsetD2D16", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemsetD2D16Async",
            ("hipMemsetD2D16Async", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemsetD2D32_v2", ("hipMemsetD2D32", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemsetD2D32Async",
            ("hipMemsetD2D32Async", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemsetD2D8_v2", ("hipMemsetD2D8", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemsetD2D8Async",
            ("hipMemsetD2D8Async", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemsetD32_v2", ("hipMemset", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemsetD32Async", ("hipMemsetAsync", c.CONV_MEM, c.API_DRIVER)),
        ("cuMemsetD8_v2", ("hipMemsetD8", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemsetD8Async",
            ("hipMemsetD8Async", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMipmappedArrayCreate",
            ("hipMipmappedArrayCreate", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMipmappedArrayDestroy",
            ("hipMipmappedArrayDestroy", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMipmappedArrayGetLevel",
            ("hipMipmappedArrayGetLevel", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMemPrefetchAsync",
            ("hipMemPrefetchAsync__", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuMemAdvise", ("hipMemAdvise", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuMemRangeGetAttribute",
            ("hipMemRangeGetAttribute", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuMemRangeGetAttributes",
            ("hipMemRangeGetAttributes", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuPointerGetAttribute",
            ("hipPointerGetAttribute", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuPointerGetAttributes",
            ("hipPointerGetAttributes", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuPointerSetAttribute",
            ("hipPointerSetAttribute", c.CONV_MEM, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("CU_TR_FILTER_MODE_POINT", ("hipFilterModePoint", c.CONV_TEX, c.API_DRIVER)),
        (
            "CU_TR_FILTER_MODE_LINEAR",
            ("hipFilterModeLinear", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetAddress",
            ("hipTexRefGetAddress", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetAddressMode",
            ("hipTexRefGetAddressMode", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetArray",
            ("hipTexRefGetArray", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetBorderColor",
            ("hipTexRefGetBorderColor", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetFilterMode",
            ("hipTexRefGetFilterMode", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetFlags",
            ("hipTexRefGetFlags", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetFormat",
            ("hipTexRefGetFormat", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMaxAnisotropy",
            ("hipTexRefGetMaxAnisotropy", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmapFilterMode",
            ("hipTexRefGetMipmapFilterMode", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmapLevelBias",
            ("hipTexRefGetMipmapLevelBias", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmapLevelClamp",
            ("hipTexRefGetMipmapLevelClamp", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefGetMipmappedArray",
            ("hipTexRefGetMipmappedArray", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetAddress",
            ("hipTexRefSetAddress", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetAddress2D",
            ("hipTexRefSetAddress2D", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuTexRefSetAddressMode", ("hipTexRefSetAddressMode", c.CONV_TEX, c.API_DRIVER)),
        ("cuTexRefSetArray", ("hipTexRefSetArray", c.CONV_TEX, c.API_DRIVER)),
        (
            "cuTexRefSetBorderColor",
            ("hipTexRefSetBorderColor", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuTexRefSetFilterMode", ("hipTexRefSetFilterMode", c.CONV_TEX, c.API_DRIVER)),
        ("cuTexRefSetFlags", ("hipTexRefSetFlags", c.CONV_TEX, c.API_DRIVER)),
        ("cuTexRefSetFormat", ("hipTexRefSetFormat", c.CONV_TEX, c.API_DRIVER)),
        (
            "cuTexRefSetMaxAnisotropy",
            ("hipTexRefSetMaxAnisotropy", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmapFilterMode",
            ("hipTexRefSetMipmapFilterMode", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmapLevelBias",
            ("hipTexRefSetMipmapLevelBias", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmapLevelClamp",
            ("hipTexRefSetMipmapLevelClamp", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexRefSetMipmappedArray",
            ("hipTexRefSetMipmappedArray", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuTexRefCreate", ("hipTexRefCreate", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuTexRefDestroy",
            ("hipTexRefDestroy", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuSurfRefGetArray",
            ("hipSurfRefGetArray", c.CONV_SURFACE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuSurfRefSetArray",
            ("hipSurfRefSetArray", c.CONV_SURFACE, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectCreate",
            ("hipTexObjectCreate", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectDestroy",
            ("hipTexObjectDestroy", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectGetResourceDesc",
            ("hipTexObjectGetResourceDesc", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectGetResourceViewDesc",
            ("hipTexObjectGetResourceViewDesc", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuTexObjectGetTextureDesc",
            ("hipTexObjectGetTextureDesc", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuSurfObjectCreate",
            ("hipSurfObjectCreate", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuSurfObjectDestroy",
            ("hipSurfObjectDestroy", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuSurfObjectGetResourceDesc",
            ("hipSurfObjectGetResourceDesc", c.CONV_TEX, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsMapResources",
            ("hipGraphicsMapResources", c.CONV_GRAPHICS, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsResourceGetMappedMipmappedArray",
            (
                "hipGraphicsResourceGetMappedMipmappedArray",
                c.CONV_GRAPHICS,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsResourceGetMappedPointer",
            (
                "hipGraphicsResourceGetMappedPointer",
                c.CONV_GRAPHICS,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsResourceSetMapFlags",
            (
                "hipGraphicsResourceSetMapFlags",
                c.CONV_GRAPHICS,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsSubResourceGetMappedArray",
            (
                "hipGraphicsSubResourceGetMappedArray",
                c.CONV_GRAPHICS,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsUnmapResources",
            ("hipGraphicsUnmapResources", c.CONV_GRAPHICS, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsUnregisterResource",
            (
                "hipGraphicsUnregisterResource",
                c.CONV_GRAPHICS,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuProfilerInitialize",
            ("hipProfilerInitialize", c.CONV_OTHER, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuProfilerStart", ("hipProfilerStart", c.CONV_OTHER, c.API_DRIVER)),
        ("cuProfilerStop", ("hipProfilerStop", c.CONV_OTHER, c.API_DRIVER)),
        (
            "CU_GL_DEVICE_LIST_ALL",
            ("HIP_GL_DEVICE_LIST_ALL", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GL_DEVICE_LIST_CURRENT_FRAME",
            ("HIP_GL_DEVICE_LIST_CURRENT_FRAME", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GL_DEVICE_LIST_NEXT_FRAME",
            ("HIP_GL_DEVICE_LIST_NEXT_FRAME", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuGLGetDevices", ("hipGLGetDevices", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuGraphicsGLRegisterBuffer",
            ("hipGraphicsGLRegisterBuffer", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsGLRegisterImage",
            ("hipGraphicsGLRegisterImage", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        ("cuWGLGetDevice", ("hipWGLGetDevice", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "CU_GL_MAP_RESOURCE_FLAGS_NONE",
            ("HIP_GL_MAP_RESOURCE_FLAGS_NONE", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY",
                c.CONV_GL,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
                c.CONV_GL,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cuGLCtxCreate", ("hipGLCtxCreate", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        ("cuGLInit", ("hipGLInit", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED)),
        (
            "cuGLMapBufferObject",
            ("hipGLMapBufferObject", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGLMapBufferObjectAsync",
            ("hipGLMapBufferObjectAsync", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGLRegisterBufferObject",
            ("hipGLRegisterBufferObject", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGLSetBufferObjectMapFlags",
            ("hipGLSetBufferObjectMapFlags", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGLUnmapBufferObject",
            ("hipGLUnmapBufferObject", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGLUnmapBufferObjectAsync",
            ("hipGLUnmapBufferObjectAsync", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGLUnregisterBufferObject",
            ("hipGLUnregisterBufferObject", c.CONV_GL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_DEVICE_LIST_ALL",
            ("HIP_D3D9_DEVICE_LIST_ALL", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_DEVICE_LIST_CURRENT_FRAME",
            (
                "HIP_D3D9_DEVICE_LIST_CURRENT_FRAME",
                c.CONV_D3D9,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D9_DEVICE_LIST_NEXT_FRAME",
            ("HIP_D3D9_DEVICE_LIST_NEXT_FRAME", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9CtxCreate",
            ("hipD3D9CtxCreate", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9CtxCreateOnDevice",
            ("hipD3D9CtxCreateOnDevice", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9GetDevice",
            ("hipD3D9GetDevice", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9GetDevices",
            ("hipD3D9GetDevices", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9GetDirect3DDevice",
            ("hipD3D9GetDirect3DDevice", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsD3D9RegisterResource",
            ("hipGraphicsD3D9RegisterResource", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_MAPRESOURCE_FLAGS_NONE",
            ("HIP_D3D9_MAPRESOURCE_FLAGS_NONE", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_MAPRESOURCE_FLAGS_READONLY",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",
                c.CONV_D3D9,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",
                c.CONV_D3D9,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D9_REGISTER_FLAGS_NONE",
            ("HIP_D3D9_REGISTER_FLAGS_NONE", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D9_REGISTER_FLAGS_ARRAY",
            ("HIP_D3D9_REGISTER_FLAGS_ARRAY", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9MapResources",
            ("hipD3D9MapResources", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9RegisterResource",
            ("hipD3D9RegisterResource", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedArray",
            ("hipD3D9ResourceGetMappedArray", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedPitch",
            ("hipD3D9ResourceGetMappedPitch", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedPointer",
            ("hipD3D9ResourceGetMappedPointer", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetMappedSize",
            ("hipD3D9ResourceGetMappedSize", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9ResourceGetSurfaceDimensions",
            (
                "hipD3D9ResourceGetSurfaceDimensions",
                c.CONV_D3D9,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D9ResourceSetMapFlags",
            ("hipD3D9ResourceSetMapFlags", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9UnmapResources",
            ("hipD3D9UnmapResources", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D9UnregisterResource",
            ("hipD3D9UnregisterResource", c.CONV_D3D9, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D10_DEVICE_LIST_ALL",
            ("HIP_D3D10_DEVICE_LIST_ALL", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D10_DEVICE_LIST_CURRENT_FRAME",
            (
                "HIP_D3D10_DEVICE_LIST_CURRENT_FRAME",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_DEVICE_LIST_NEXT_FRAME",
            (
                "HIP_D3D10_DEVICE_LIST_NEXT_FRAME",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D10GetDevice",
            ("hipD3D10GetDevice", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10GetDevices",
            ("hipD3D10GetDevices", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsD3D10RegisterResource",
            (
                "hipGraphicsD3D10RegisterResource",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_MAPRESOURCE_FLAGS_NONE",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_NONE",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_MAPRESOURCE_FLAGS_READONLY",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D10_REGISTER_FLAGS_NONE",
            ("HIP_D3D10_REGISTER_FLAGS_NONE", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D10_REGISTER_FLAGS_ARRAY",
            ("HIP_D3D10_REGISTER_FLAGS_ARRAY", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10CtxCreate",
            ("hipD3D10CtxCreate", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10CtxCreateOnDevice",
            ("hipD3D10CtxCreateOnDevice", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10GetDirect3DDevice",
            ("hipD3D10GetDirect3DDevice", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10MapResources",
            ("hipD3D10MapResources", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10RegisterResource",
            ("hipD3D10RegisterResource", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetMappedArray",
            ("hipD3D10ResourceGetMappedArray", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetMappedPitch",
            ("hipD3D10ResourceGetMappedPitch", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetMappedPointer",
            (
                "hipD3D10ResourceGetMappedPointer",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D10ResourceGetMappedSize",
            ("hipD3D10ResourceGetMappedSize", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10ResourceGetSurfaceDimensions",
            (
                "hipD3D10ResourceGetSurfaceDimensions",
                c.CONV_D3D10,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD310ResourceSetMapFlags",
            ("hipD3D10ResourceSetMapFlags", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10UnmapResources",
            ("hipD3D10UnmapResources", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D10UnregisterResource",
            ("hipD3D10UnregisterResource", c.CONV_D3D10, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D11_DEVICE_LIST_ALL",
            ("HIP_D3D11_DEVICE_LIST_ALL", c.CONV_D3D11, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "CU_D3D11_DEVICE_LIST_CURRENT_FRAME",
            (
                "HIP_D3D11_DEVICE_LIST_CURRENT_FRAME",
                c.CONV_D3D11,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_D3D11_DEVICE_LIST_NEXT_FRAME",
            (
                "HIP_D3D11_DEVICE_LIST_NEXT_FRAME",
                c.CONV_D3D11,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D11GetDevice",
            ("hipD3D11GetDevice", c.CONV_D3D11, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D11GetDevices",
            ("hipD3D11GetDevices", c.CONV_D3D11, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsD3D11RegisterResource",
            (
                "hipGraphicsD3D11RegisterResource",
                c.CONV_D3D11,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuD3D11CtxCreate",
            ("hipD3D11CtxCreate", c.CONV_D3D11, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D11CtxCreateOnDevice",
            ("hipD3D11CtxCreateOnDevice", c.CONV_D3D11, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuD3D11GetDirect3DDevice",
            ("hipD3D11GetDirect3DDevice", c.CONV_D3D11, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsVDPAURegisterOutputSurface",
            (
                "hipGraphicsVDPAURegisterOutputSurface",
                c.CONV_VDPAU,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuGraphicsVDPAURegisterVideoSurface",
            (
                "hipGraphicsVDPAURegisterVideoSurface",
                c.CONV_VDPAU,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuVDPAUGetDevice",
            ("hipVDPAUGetDevice", c.CONV_VDPAU, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuVDPAUCtxCreate",
            ("hipVDPAUCtxCreate", c.CONV_VDPAU, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerAcquireFrame",
            ("hipEGLStreamConsumerAcquireFrame", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerConnect",
            ("hipEGLStreamConsumerConnect", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerConnectWithFlags",
            (
                "hipEGLStreamConsumerConnectWithFlags",
                c.CONV_EGL,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cuEGLStreamConsumerDisconnect",
            ("hipEGLStreamConsumerDisconnect", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamConsumerReleaseFrame",
            ("hipEGLStreamConsumerReleaseFrame", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerConnect",
            ("hipEGLStreamProducerConnect", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerDisconnect",
            ("hipEGLStreamProducerDisconnect", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerPresentFrame",
            ("hipEGLStreamProducerPresentFrame", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuEGLStreamProducerReturnFrame",
            ("hipEGLStreamProducerReturnFrame", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsEGLRegisterImage",
            ("hipGraphicsEGLRegisterImage", c.CONV_EGL, c.API_DRIVER, c.HIP_UNSUPPORTED),
        ),
        (
            "cuGraphicsResourceGetMappedEglFrame",
            (
                "hipGraphicsResourceGetMappedEglFrame",
                c.CONV_EGL,
                c.API_DRIVER,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cudaDataType_t", ("hipDataType_t", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("cudaDataType", ("hipDataType_t", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_R_16F", ("hipR16F", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_C_16F", ("hipC16F", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_R_32F", ("hipR32F", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_C_32F", ("hipC32F", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_R_64F", ("hipR64F", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_C_64F", ("hipC64F", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_R_8I", ("hipR8I", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_C_8I", ("hipC8I", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_R_8U", ("hipR8U", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_C_8U", ("hipC8U", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_R_32I", ("hipR32I", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_C_32I", ("hipC32I", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_R_32U", ("hipR32U", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("CUDA_C_32U", ("hipC32U", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        (
            "MAJOR_VERSION",
            ("hipLibraryMajorVersion", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "MINOR_VERSION",
            ("hipLibraryMinorVersion", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "PATCH_LEVEL",
            ("hipLibraryPatchVersion", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAttachGlobal",
            ("hipMemAttachGlobal", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAttachHost",
            ("hipMemAttachHost", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAttachSingle",
            ("hipMemAttachSingle", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaOccupancyDefault",
            ("hipOccupancyDefault", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaOccupancyDisableCachingOverride",
            (
                "hipOccupancyDisableCachingOverride",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cudaGetLastError", ("hipGetLastError", c.CONV_ERROR, c.API_RUNTIME)),
        ("cudaPeekAtLastError", ("hipPeekAtLastError", c.CONV_ERROR, c.API_RUNTIME)),
        ("cudaGetErrorName", ("hipGetErrorName", c.CONV_ERROR, c.API_RUNTIME)),
        ("cudaGetErrorString", ("hipGetErrorString", c.CONV_ERROR, c.API_RUNTIME)),
        ("cudaMemcpy3DParms", ("hipMemcpy3DParms", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMemcpy3DPeerParms",
            ("hipMemcpy3DPeerParms", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMemcpy", ("hipMemcpy", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpyToArray", ("hipMemcpyToArray", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpyToSymbol", ("hipMemcpyToSymbol", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpyToSymbolAsync", ("hipMemcpyToSymbolAsync", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpyAsync", ("hipMemcpyAsync", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpy2D", ("hipMemcpy2D", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpy2DAsync", ("hipMemcpy2DAsync", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpy2DToArray", ("hipMemcpy2DToArray", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMemcpy2DArrayToArray",
            ("hipMemcpy2DArrayToArray", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy2DFromArray",
            ("hipMemcpy2DFromArray", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy2DFromArrayAsync",
            ("hipMemcpy2DFromArrayAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy2DToArrayAsync",
            ("hipMemcpy2DToArrayAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMemcpy3D", ("hipMemcpy3D", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMemcpy3DAsync",
            ("hipMemcpy3DAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy3DPeer",
            ("hipMemcpy3DPeer", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpy3DPeerAsync",
            ("hipMemcpy3DPeerAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpyArrayToArray",
            ("hipMemcpyArrayToArray", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemcpyFromArrayAsync",
            ("hipMemcpyFromArrayAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMemcpyFromSymbol", ("hipMemcpyFromSymbol", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMemcpyFromSymbolAsync",
            ("hipMemcpyFromSymbolAsync", c.CONV_MEM, c.API_RUNTIME),
        ),
        ("cudaMemAdvise", ("hipMemAdvise", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        (
            "cudaMemRangeGetAttribute",
            ("hipMemRangeGetAttribute", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeGetAttributes",
            ("hipMemRangeGetAttributes", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseSetReadMostly",
            ("hipMemAdviseSetReadMostly", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseUnsetReadMostly",
            ("hipMemAdviseUnsetReadMostly", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseSetPreferredLocation",
            (
                "hipMemAdviseSetPreferredLocation",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaMemAdviseUnsetPreferredLocation",
            (
                "hipMemAdviseUnsetPreferredLocation",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaMemAdviseSetAccessedBy",
            ("hipMemAdviseSetAccessedBy", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemAdviseUnsetAccessedBy",
            ("hipMemAdviseUnsetAccessedBy", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeAttributeReadMostly",
            ("hipMemRangeAttributeReadMostly", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeAttributePreferredLocation",
            (
                "hipMemRangeAttributePreferredLocation",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaMemRangeAttributeAccessedBy",
            ("hipMemRangeAttributeAccessedBy", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemRangeAttributeLastPrefetchLocation",
            (
                "hipMemRangeAttributeLastPrefetchLocation",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cudaMemcpyHostToHost", ("hipMemcpyHostToHost", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpyHostToDevice", ("hipMemcpyHostToDevice", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpyDeviceToHost", ("hipMemcpyDeviceToHost", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMemcpyDeviceToDevice",
            ("hipMemcpyDeviceToDevice", c.CONV_MEM, c.API_RUNTIME),
        ),
        ("cudaMemcpyDefault", ("hipMemcpyDefault", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemset", ("hipMemset", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemsetAsync", ("hipMemsetAsync", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemset2D", ("hipMemset2D", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMemset2DAsync",
            ("hipMemset2DAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMemset3D", ("hipMemset3D", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        (
            "cudaMemset3DAsync",
            ("hipMemset3DAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMemGetInfo", ("hipMemGetInfo", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaArrayGetInfo",
            ("hipArrayGetInfo", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaFreeMipmappedArray",
            ("hipFreeMipmappedArray", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGetMipmappedArrayLevel",
            ("hipGetMipmappedArrayLevel", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSymbolAddress",
            ("hipGetSymbolAddress", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSymbolSize",
            ("hipGetSymbolSize", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMemPrefetchAsync",
            ("hipMemPrefetchAsync", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMallocHost", ("hipHostMalloc", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMallocArray", ("hipMallocArray", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMalloc", ("hipMalloc", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMalloc3D", ("hipMalloc3D", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMalloc3DArray", ("hipMalloc3DArray", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaMallocManaged",
            ("hipMallocManaged", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaMallocMipmappedArray",
            ("hipMallocMipmappedArray", c.CONV_MEM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaMallocPitch", ("hipMallocPitch", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaFreeHost", ("hipHostFree", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaFreeArray", ("hipFreeArray", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaFree", ("hipFree", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaHostRegister", ("hipHostRegister", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaHostUnregister", ("hipHostUnregister", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaHostAlloc", ("hipHostMalloc", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemoryTypeHost", ("hipMemoryTypeHost", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemoryTypeDevice", ("hipMemoryTypeDevice", c.CONV_MEM, c.API_RUNTIME)),
        ("make_cudaExtent", ("make_hipExtent", c.CONV_MEM, c.API_RUNTIME)),
        ("make_cudaPitchedPtr", ("make_hipPitchedPtr", c.CONV_MEM, c.API_RUNTIME)),
        ("make_cudaPos", ("make_hipPos", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaHostAllocDefault", ("hipHostMallocDefault", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaHostAllocPortable", ("hipHostMallocPortable", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaHostAllocMapped", ("hipHostMallocMapped", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaHostAllocWriteCombined",
            ("hipHostMallocWriteCombined", c.CONV_MEM, c.API_RUNTIME),
        ),
        ("cudaHostGetFlags", ("hipHostGetFlags", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaHostRegisterDefault", ("hipHostRegisterDefault", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaHostRegisterPortable",
            ("hipHostRegisterPortable", c.CONV_MEM, c.API_RUNTIME),
        ),
        ("cudaHostRegisterMapped", ("hipHostRegisterMapped", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaHostRegisterIoMemory",
            ("hipHostRegisterIoMemory", c.CONV_MEM, c.API_RUNTIME),
        ),
        # ("warpSize", ("hipWarpSize", c.CONV_SPECIAL_FUNC, c.API_RUNTIME), (HIP actually uses warpSize...)),
        ("cudaEventCreate", ("hipEventCreate", c.CONV_EVENT, c.API_RUNTIME)),
        (
            "cudaEventCreateWithFlags",
            ("hipEventCreateWithFlags", c.CONV_EVENT, c.API_RUNTIME),
        ),
        ("cudaEventDestroy", ("hipEventDestroy", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventRecord", ("hipEventRecord", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventElapsedTime", ("hipEventElapsedTime", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventSynchronize", ("hipEventSynchronize", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventQuery", ("hipEventQuery", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventDefault", ("hipEventDefault", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventBlockingSync", ("hipEventBlockingSync", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventDisableTiming", ("hipEventDisableTiming", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaEventInterprocess", ("hipEventInterprocess", c.CONV_EVENT, c.API_RUNTIME)),
        ("cudaStreamCreate", ("hipStreamCreate", c.CONV_STREAM, c.API_RUNTIME)),
        (
            "cudaStreamCreateWithFlags",
            ("hipStreamCreateWithFlags", c.CONV_STREAM, c.API_RUNTIME),
        ),
        (
            "cudaStreamCreateWithPriority",
            ("hipStreamCreateWithPriority", c.CONV_STREAM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaStreamDestroy", ("hipStreamDestroy", c.CONV_STREAM, c.API_RUNTIME)),
        ("cudaStreamWaitEvent", ("hipStreamWaitEvent", c.CONV_STREAM, c.API_RUNTIME)),
        ("cudaStreamSynchronize", ("hipStreamSynchronize", c.CONV_STREAM, c.API_RUNTIME)),
        ("cudaStreamGetFlags", ("hipStreamGetFlags", c.CONV_STREAM, c.API_RUNTIME)),
        ("cudaStreamQuery", ("hipStreamQuery", c.CONV_STREAM, c.API_RUNTIME)),
        ("cudaStreamAddCallback", ("hipStreamAddCallback", c.CONV_STREAM, c.API_RUNTIME)),
        (
            "cudaStreamAttachMemAsync",
            ("hipStreamAttachMemAsync", c.CONV_STREAM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaStreamGetPriority",
            ("hipStreamGetPriority", c.CONV_STREAM, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaStreamDefault", ("hipStreamDefault", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaStreamNonBlocking", ("hipStreamNonBlocking", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaDeviceSynchronize", ("hipDeviceSynchronize", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaDeviceReset", ("hipDeviceReset", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaSetDevice", ("hipSetDevice", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaGetDevice", ("hipGetDevice", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaGetDeviceCount", ("hipGetDeviceCount", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaChooseDevice", ("hipChooseDevice", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaThreadExit", ("hipDeviceReset", c.CONV_THREAD, c.API_RUNTIME)),
        (
            "cudaThreadGetCacheConfig",
            ("hipDeviceGetCacheConfig", c.CONV_THREAD, c.API_RUNTIME),
        ),
        (
            "cudaThreadGetLimit",
            ("hipThreadGetLimit", c.CONV_THREAD, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaThreadSetCacheConfig",
            ("hipDeviceSetCacheConfig", c.CONV_THREAD, c.API_RUNTIME),
        ),
        (
            "cudaThreadSetLimit",
            ("hipThreadSetLimit", c.CONV_THREAD, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaThreadSynchronize", ("hipDeviceSynchronize", c.CONV_THREAD, c.API_RUNTIME)),
        ("cudaDeviceGetAttribute", ("hipDeviceGetAttribute", c.CONV_DEVICE, c.API_RUNTIME)),
        (
            "cudaDevAttrMaxThreadsPerBlock",
            ("hipDeviceAttributeMaxThreadsPerBlock", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxBlockDimX",
            ("hipDeviceAttributeMaxBlockDimX", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxBlockDimY",
            ("hipDeviceAttributeMaxBlockDimY", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxBlockDimZ",
            ("hipDeviceAttributeMaxBlockDimZ", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxGridDimX",
            ("hipDeviceAttributeMaxGridDimX", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxGridDimY",
            ("hipDeviceAttributeMaxGridDimY", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxGridDimZ",
            ("hipDeviceAttributeMaxGridDimZ", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxSharedMemoryPerBlock",
            ("hipDeviceAttributeMaxSharedMemoryPerBlock", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrTotalConstantMemory",
            ("hipDeviceAttributeTotalConstantMemory", c.CONV_TYPE, c.API_RUNTIME),
        ),
        ("cudaDevAttrWarpSize", ("hipDeviceAttributeWarpSize", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "cudaDevAttrMaxPitch",
            ("hipDeviceAttributeMaxPitch", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrMaxRegistersPerBlock",
            ("hipDeviceAttributeMaxRegistersPerBlock", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrClockRate",
            ("hipDeviceAttributeClockRate", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrTextureAlignment",
            (
                "hipDeviceAttributeTextureAlignment",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrGpuOverlap",
            ("hipDeviceAttributeGpuOverlap", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrMultiProcessorCount",
            ("hipDeviceAttributeMultiprocessorCount", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrKernelExecTimeout",
            (
                "hipDeviceAttributeKernelExecTimeout",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrIntegrated",
            ("hipDeviceAttributeIntegrated", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrCanMapHostMemory",
            (
                "hipDeviceAttributeCanMapHostMemory",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrComputeMode",
            ("hipDeviceAttributeComputeMode", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxTexture1DWidth",
            (
                "hipDeviceAttributeMaxTexture1DWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DWidth",
            (
                "hipDeviceAttributeMaxTexture2DWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DHeight",
            (
                "hipDeviceAttributeMaxTexture2DHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DWidth",
            (
                "hipDeviceAttributeMaxTexture3DWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DHeight",
            (
                "hipDeviceAttributeMaxTexture3DHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DDepth",
            (
                "hipDeviceAttributeMaxTexture3DDepth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLayeredWidth",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLayeredHeight",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLayeredLayers",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrSurfaceAlignment",
            (
                "hipDeviceAttributeSurfaceAlignment",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrConcurrentKernels",
            ("hipDeviceAttributeConcurrentKernels", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrEccEnabled",
            ("hipDeviceAttributeEccEnabled", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaDevAttrPciBusId", ("hipDeviceAttributePciBusId", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "cudaDevAttrPciDeviceId",
            ("hipDeviceAttributePciDeviceId", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrTccDriver",
            ("hipDeviceAttributeTccDriver", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrMemoryClockRate",
            ("hipDeviceAttributeMemoryClockRate", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrGlobalMemoryBusWidth",
            ("hipDeviceAttributeMemoryBusWidth", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrL2CacheSize",
            ("hipDeviceAttributeL2CacheSize", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxThreadsPerMultiProcessor",
            ("hipDeviceAttributeMaxThreadsPerMultiProcessor", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrAsyncEngineCount",
            (
                "hipDeviceAttributeAsyncEngineCount",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrUnifiedAddressing",
            (
                "hipDeviceAttributeUnifiedAddressing",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture1DLayeredWidth",
            (
                "hipDeviceAttributeMaxTexture1DLayeredWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture1DLayeredLayers",
            (
                "hipDeviceAttributeMaxTexture1DLayeredLayers",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DGatherWidth",
            (
                "hipDeviceAttributeMaxTexture2DGatherWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DGatherHeight",
            (
                "hipDeviceAttributeMaxTexture2DGatherHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DWidthAlt",
            (
                "hipDeviceAttributeMaxTexture3DWidthAlternate",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DHeightAlt",
            (
                "hipDeviceAttributeMaxTexture3DHeightAlternate",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture3DDepthAlt",
            (
                "hipDeviceAttributeMaxTexture3DDepthAlternate",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrPciDomainId",
            ("hipDeviceAttributePciDomainId", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaDevAttrTexturePitchAlignment",
            (
                "hipDeviceAttributeTexturePitchAlignment",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTextureCubemapWidth",
            (
                "hipDeviceAttributeMaxTextureCubemapWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTextureCubemapLayeredWidth",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTextureCubemapLayeredLayers",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredLayers",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface1DWidth",
            (
                "hipDeviceAttributeMaxSurface1DWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DWidth",
            (
                "hipDeviceAttributeMaxSurface2DWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DHeight",
            (
                "hipDeviceAttributeMaxSurface2DHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface3DWidth",
            (
                "hipDeviceAttributeMaxSurface3DWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface3DHeight",
            (
                "hipDeviceAttributeMaxSurface3DHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface3DDepth",
            (
                "hipDeviceAttributeMaxSurface3DDepth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface1DLayeredWidth",
            (
                "hipDeviceAttributeMaxSurface1DLayeredWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface1DLayeredLayers",
            (
                "hipDeviceAttributeMaxSurface1DLayeredLayers",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DLayeredWidth",
            (
                "hipDeviceAttributeMaxSurface2DLayeredWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DLayeredHeight",
            (
                "hipDeviceAttributeMaxSurface2DLayeredHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurface2DLayeredLayers",
            (
                "hipDeviceAttributeMaxSurface2DLayeredLayers",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurfaceCubemapWidth",
            (
                "hipDeviceAttributeMaxSurfaceCubemapWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurfaceCubemapLayeredWidth",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSurfaceCubemapLayeredLayers",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture1DLinearWidth",
            (
                "hipDeviceAttributeMaxTexture1DLinearWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLinearWidth",
            (
                "hipDeviceAttributeMaxTexture2DLinearWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLinearHeight",
            (
                "hipDeviceAttributeMaxTexture2DLinearHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DLinearPitch",
            (
                "hipDeviceAttributeMaxTexture2DLinearPitch",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DMipmappedWidth",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxTexture2DMipmappedHeight",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedHeight",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrComputeCapabilityMajor",
            ("hipDeviceAttributeComputeCapabilityMajor", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrComputeCapabilityMinor",
            ("hipDeviceAttributeComputeCapabilityMinor", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMaxTexture1DMipmappedWidth",
            (
                "hipDeviceAttributeMaxTexture1DMipmappedWidth",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrStreamPrioritiesSupported",
            (
                "hipDeviceAttributeStreamPrioritiesSupported",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrGlobalL1CacheSupported",
            (
                "hipDeviceAttributeGlobalL1CacheSupported",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrLocalL1CacheSupported",
            (
                "hipDeviceAttributeLocalL1CacheSupported",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrMaxSharedMemoryPerMultiprocessor",
            (
                "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
                c.CONV_TYPE,
                c.API_RUNTIME,
            ),
        ),
        (
            "cudaDevAttrMaxRegistersPerMultiprocessor",
            (
                "hipDeviceAttributeMaxRegistersPerMultiprocessor",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrManagedMemory",
            (
                "hipDeviceAttributeManagedMemory",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrIsMultiGpuBoard",
            ("hipDeviceAttributeIsMultiGpuBoard", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDevAttrMultiGpuBoardGroupID",
            (
                "hipDeviceAttributeMultiGpuBoardGroupID",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrHostNativeAtomicSupported",
            (
                "hipDeviceAttributeHostNativeAtomicSupported",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrSingleToDoublePrecisionPerfRatio",
            (
                "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrPageableMemoryAccess",
            (
                "hipDeviceAttributePageableMemoryAccess",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrConcurrentManagedAccess",
            (
                "hipDeviceAttributeConcurrentManagedAccess",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrComputePreemptionSupported",
            (
                "hipDeviceAttributeComputePreemptionSupported",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevAttrCanUseHostPointerForRegisteredMem",
            (
                "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaPointerGetAttributes",
            ("hipPointerGetAttributes", c.CONV_MEM, c.API_RUNTIME),
        ),
        (
            "cudaHostGetDevicePointer",
            ("hipHostGetDevicePointer", c.CONV_MEM, c.API_RUNTIME),
        ),
        (
            "cudaGetDeviceProperties",
            ("hipGetDeviceProperties", c.CONV_DEVICE, c.API_RUNTIME),
        ),
        ("cudaDeviceGetPCIBusId", ("hipDeviceGetPCIBusId", c.CONV_DEVICE, c.API_RUNTIME)),
        (
            "cudaDeviceGetByPCIBusId",
            ("hipDeviceGetByPCIBusId", c.CONV_DEVICE, c.API_RUNTIME),
        ),
        (
            "cudaDeviceGetStreamPriorityRange",
            (
                "hipDeviceGetStreamPriorityRange",
                c.CONV_DEVICE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaSetValidDevices",
            ("hipSetValidDevices", c.CONV_DEVICE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaDevP2PAttrPerformanceRank",
            (
                "hipDeviceP2PAttributePerformanceRank",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevP2PAttrAccessSupported",
            (
                "hipDeviceP2PAttributeAccessSupported",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDevP2PAttrNativeAtomicSupported",
            (
                "hipDeviceP2PAttributeNativeAtomicSupported",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaDeviceGetP2PAttribute",
            ("hipDeviceGetP2PAttribute", c.CONV_DEVICE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeDefault",
            ("hipComputeModeDefault", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeExclusive",
            ("hipComputeModeExclusive", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeProhibited",
            ("hipComputeModeProhibited", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeModeExclusiveProcess",
            ("hipComputeModeExclusiveProcess", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGetDeviceFlags",
            ("hipGetDeviceFlags", c.CONV_DEVICE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaSetDeviceFlags", ("hipSetDeviceFlags", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaDeviceScheduleAuto", ("hipDeviceScheduleAuto", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaDeviceScheduleSpin", ("hipDeviceScheduleSpin", c.CONV_TYPE, c.API_RUNTIME)),
        ("cudaDeviceScheduleYield", ("hipDeviceScheduleYield", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "cudaDeviceBlockingSync",
            ("hipDeviceScheduleBlockingSync", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDeviceScheduleBlockingSync",
            ("hipDeviceScheduleBlockingSync", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDeviceScheduleMask",
            ("hipDeviceScheduleMask", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaDeviceMapHost", ("hipDeviceMapHost", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "cudaDeviceLmemResizeToMax",
            ("hipDeviceLmemResizeToMax", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaDeviceMask", ("hipDeviceMask", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        (
            "cudaDeviceSetCacheConfig",
            ("hipDeviceSetCacheConfig", c.CONV_CACHE, c.API_RUNTIME),
        ),
        (
            "cudaDeviceGetCacheConfig",
            ("hipDeviceGetCacheConfig", c.CONV_CACHE, c.API_RUNTIME),
        ),
        ("cudaFuncSetCacheConfig", ("hipFuncSetCacheConfig", c.CONV_CACHE, c.API_RUNTIME)),
        (
            "cudaFuncCachePreferNone",
            ("hipFuncCachePreferNone", c.CONV_CACHE, c.API_RUNTIME),
        ),
        (
            "cudaFuncCachePreferShared",
            ("hipFuncCachePreferShared", c.CONV_CACHE, c.API_RUNTIME),
        ),
        ("cudaFuncCachePreferL1", ("hipFuncCachePreferL1", c.CONV_CACHE, c.API_RUNTIME)),
        (
            "cudaFuncCachePreferEqual",
            ("hipFuncCachePreferEqual", c.CONV_CACHE, c.API_RUNTIME),
        ),
        (
            "cudaFuncGetAttributes",
            ("hipFuncGetAttributes", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaFuncSetSharedMemConfig",
            ("hipFuncSetSharedMemConfig", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGetParameterBuffer",
            ("hipGetParameterBuffer", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaSetDoubleForDevice",
            ("hipSetDoubleForDevice", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaSetDoubleForHost",
            ("hipSetDoubleForHost", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaConfigureCall",
            ("hipConfigureCall", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaLaunch", ("hipLaunch", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        (
            "cudaSetupArgument",
            ("hipSetupArgument", c.CONV_EXEC, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaDriverGetVersion", ("hipDriverGetVersion", c.CONV_VERSION, c.API_RUNTIME)),
        (
            "cudaRuntimeGetVersion",
            ("hipRuntimeGetVersion", c.CONV_VERSION, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSize",
            ("hipOccupancyMaxPotentialBlockSize", c.CONV_OCCUPANCY, c.API_RUNTIME),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSizeWithFlags",
            (
                "hipOccupancyMaxPotentialBlockSizeWithFlags",
                c.CONV_OCCUPANCY,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            (
                "hipOccupancyMaxActiveBlocksPerMultiprocessor",
                c.CONV_OCCUPANCY,
                c.API_RUNTIME,
            ),
        ),
        (
            "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            (
                "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                c.CONV_OCCUPANCY,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSizeVariableSMem",
            (
                "hipOccupancyMaxPotentialBlockSizeVariableSMem",
                c.CONV_OCCUPANCY,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
            (
                "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
                c.CONV_OCCUPANCY,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cudaDeviceCanAccessPeer", ("hipDeviceCanAccessPeer", c.CONV_PEER, c.API_RUNTIME)),
        (
            "cudaDeviceDisablePeerAccess",
            ("hipDeviceDisablePeerAccess", c.CONV_PEER, c.API_RUNTIME),
        ),
        (
            "cudaDeviceEnablePeerAccess",
            ("hipDeviceEnablePeerAccess", c.CONV_PEER, c.API_RUNTIME),
        ),
        ("cudaMemcpyPeerAsync", ("hipMemcpyPeerAsync", c.CONV_MEM, c.API_RUNTIME)),
        ("cudaMemcpyPeer", ("hipMemcpyPeer", c.CONV_MEM, c.API_RUNTIME)),
        (
            "cudaIpcMemLazyEnablePeerAccess",
            ("hipIpcMemLazyEnablePeerAccess", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaDeviceSetSharedMemConfig",
            ("hipDeviceSetSharedMemConfig", c.CONV_DEVICE, c.API_RUNTIME),
        ),
        (
            "cudaDeviceGetSharedMemConfig",
            ("hipDeviceGetSharedMemConfig", c.CONV_DEVICE, c.API_RUNTIME),
        ),
        (
            "cudaSharedMemBankSizeDefault",
            ("hipSharedMemBankSizeDefault", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaSharedMemBankSizeFourByte",
            ("hipSharedMemBankSizeFourByte", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaSharedMemBankSizeEightByte",
            ("hipSharedMemBankSizeEightByte", c.CONV_TYPE, c.API_RUNTIME),
        ),
        (
            "cudaLimitStackSize",
            ("hipLimitStackSize", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaLimitPrintfFifoSize",
            ("hipLimitPrintfFifoSize", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaLimitMallocHeapSize", ("hipLimitMallocHeapSize", c.CONV_TYPE, c.API_RUNTIME)),
        (
            "cudaLimitDevRuntimeSyncDepth",
            ("hipLimitDevRuntimeSyncDepth", c.CONV_TYPE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaLimitDevRuntimePendingLaunchCount",
            (
                "hipLimitDevRuntimePendingLaunchCount",
                c.CONV_TYPE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cudaDeviceGetLimit", ("hipDeviceGetLimit", c.CONV_DEVICE, c.API_RUNTIME)),
        (
            "cudaProfilerInitialize",
            ("hipProfilerInitialize", c.CONV_OTHER, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaProfilerStart", ("hipProfilerStart", c.CONV_OTHER, c.API_RUNTIME)),
        ("cudaProfilerStop", ("hipProfilerStop", c.CONV_OTHER, c.API_RUNTIME)),
        (
            "cudaKeyValuePair",
            ("hipKeyValuePair", c.CONV_OTHER, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        ("cudaCSV", ("hipCSV", c.CONV_OTHER, c.API_RUNTIME, c.HIP_UNSUPPORTED)),
        ("cudaReadModeElementType", ("hipReadModeElementType", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaReadModeNormalizedFloat",
            ("hipReadModeNormalizedFloat", c.CONV_TEX, c.API_RUNTIME),
        ),
        ("cudaFilterModePoint", ("hipFilterModePoint", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaFilterModeLinear", ("hipFilterModeLinear", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaBindTexture", ("hipBindTexture", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaUnbindTexture", ("hipUnbindTexture", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaBindTexture2D", ("hipBindTexture2D", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaBindTextureToArray", ("hipBindTextureToArray", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaBindTextureToMipmappedArray",
            ("hipBindTextureToMipmappedArray", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaGetTextureAlignmentOffset",
            ("hipGetTextureAlignmentOffset", c.CONV_TEX, c.API_RUNTIME),
        ),
        ("cudaGetTextureReference", ("hipGetTextureReference", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaChannelFormatKindSigned",
            ("hipChannelFormatKindSigned", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaChannelFormatKindUnsigned",
            ("hipChannelFormatKindUnsigned", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaChannelFormatKindFloat",
            ("hipChannelFormatKindFloat", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaChannelFormatKindNone",
            ("hipChannelFormatKindNone", c.CONV_TEX, c.API_RUNTIME),
        ),
        ("cudaCreateChannelDesc", ("hipCreateChannelDesc", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaGetChannelDesc", ("hipGetChannelDesc", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResourceTypeArray", ("hipResourceTypeArray", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaResourceTypeMipmappedArray",
            ("hipResourceTypeMipmappedArray", c.CONV_TEX, c.API_RUNTIME),
        ),
        ("cudaResourceTypeLinear", ("hipResourceTypeLinear", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResourceTypePitch2D", ("hipResourceTypePitch2D", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResViewFormatNone", ("hipResViewFormatNone", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaResViewFormatUnsignedChar1",
            ("hipResViewFormatUnsignedChar1", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedChar2",
            ("hipResViewFormatUnsignedChar2", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedChar4",
            ("hipResViewFormatUnsignedChar4", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedChar1",
            ("hipResViewFormatSignedChar1", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedChar2",
            ("hipResViewFormatSignedChar2", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedChar4",
            ("hipResViewFormatSignedChar4", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedShort1",
            ("hipResViewFormatUnsignedShort1", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedShort2",
            ("hipResViewFormatUnsignedShort2", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedShort4",
            ("hipResViewFormatUnsignedShort4", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedShort1",
            ("hipResViewFormatSignedShort1", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedShort2",
            ("hipResViewFormatSignedShort2", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedShort4",
            ("hipResViewFormatSignedShort4", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedInt1",
            ("hipResViewFormatUnsignedInt1", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedInt2",
            ("hipResViewFormatUnsignedInt2", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedInt4",
            ("hipResViewFormatUnsignedInt4", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedInt1",
            ("hipResViewFormatSignedInt1", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedInt2",
            ("hipResViewFormatSignedInt2", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedInt4",
            ("hipResViewFormatSignedInt4", c.CONV_TEX, c.API_RUNTIME),
        ),
        ("cudaResViewFormatHalf1", ("hipResViewFormatHalf1", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResViewFormatHalf2", ("hipResViewFormatHalf2", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResViewFormatHalf4", ("hipResViewFormatHalf4", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResViewFormatFloat1", ("hipResViewFormatFloat1", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResViewFormatFloat2", ("hipResViewFormatFloat2", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaResViewFormatFloat4", ("hipResViewFormatFloat4", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaResViewFormatUnsignedBlockCompressed1",
            ("hipResViewFormatUnsignedBlockCompressed1", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed2",
            ("hipResViewFormatUnsignedBlockCompressed2", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed3",
            ("hipResViewFormatUnsignedBlockCompressed3", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed4",
            ("hipResViewFormatUnsignedBlockCompressed4", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedBlockCompressed4",
            ("hipResViewFormatSignedBlockCompressed4", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed5",
            ("hipResViewFormatUnsignedBlockCompressed5", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedBlockCompressed5",
            ("hipResViewFormatSignedBlockCompressed5", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed6H",
            ("hipResViewFormatUnsignedBlockCompressed6H", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatSignedBlockCompressed6H",
            ("hipResViewFormatSignedBlockCompressed6H", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaResViewFormatUnsignedBlockCompressed7",
            ("hipResViewFormatUnsignedBlockCompressed7", c.CONV_TEX, c.API_RUNTIME),
        ),
        ("cudaAddressModeWrap", ("hipAddressModeWrap", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaAddressModeClamp", ("hipAddressModeClamp", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaAddressModeMirror", ("hipAddressModeMirror", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaAddressModeBorder", ("hipAddressModeBorder", c.CONV_TEX, c.API_RUNTIME)),
        ("cudaCreateTextureObject", ("hipCreateTextureObject", c.CONV_TEX, c.API_RUNTIME)),
        (
            "cudaDestroyTextureObject",
            ("hipDestroyTextureObject", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaGetTextureObjectResourceDesc",
            ("hipGetTextureObjectResourceDesc", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaGetTextureObjectResourceViewDesc",
            ("hipGetTextureObjectResourceViewDesc", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaGetTextureObjectTextureDesc",
            ("hipGetTextureObjectTextureDesc", c.CONV_TEX, c.API_RUNTIME),
        ),
        (
            "cudaBindSurfaceToArray",
            ("hipBindSurfaceToArray", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSurfaceReference",
            ("hipGetSurfaceReference", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaBoundaryModeZero",
            ("hipBoundaryModeZero", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaBoundaryModeClamp",
            ("hipBoundaryModeClamp", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaBoundaryModeTrap",
            ("hipBoundaryModeTrap", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaFormatModeForced",
            ("hipFormatModeForced", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaFormatModeAuto",
            ("hipFormatModeAuto", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaCreateSurfaceObject",
            ("hipCreateSurfaceObject", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaDestroySurfaceObject",
            ("hipDestroySurfaceObject", c.CONV_SURFACE, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGetSurfaceObjectResourceDesc",
            (
                "hipGetSurfaceObjectResourceDesc",
                c.CONV_SURFACE,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cudaIpcCloseMemHandle", ("hipIpcCloseMemHandle", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaIpcGetEventHandle", ("hipIpcGetEventHandle", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaIpcGetMemHandle", ("hipIpcGetMemHandle", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaIpcOpenEventHandle", ("hipIpcOpenEventHandle", c.CONV_DEVICE, c.API_RUNTIME)),
        ("cudaIpcOpenMemHandle", ("hipIpcOpenMemHandle", c.CONV_DEVICE, c.API_RUNTIME)),
        (
            "cudaGLGetDevices",
            ("hipGLGetDevices", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterBuffer",
            ("hipGraphicsGLRegisterBuffer", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterImage",
            ("hipGraphicsGLRegisterImage", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaWGLGetDevice",
            ("hipWGLGetDevice", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsMapResources",
            ("hipGraphicsMapResources", c.CONV_GRAPHICS, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsResourceGetMappedMipmappedArray",
            (
                "hipGraphicsResourceGetMappedMipmappedArray",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsResourceGetMappedPointer",
            (
                "hipGraphicsResourceGetMappedPointer",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsResourceSetMapFlags",
            (
                "hipGraphicsResourceSetMapFlags",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsSubResourceGetMappedArray",
            (
                "hipGraphicsSubResourceGetMappedArray",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsUnmapResources",
            ("hipGraphicsUnmapResources", c.CONV_GRAPHICS, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsUnregisterResource",
            (
                "hipGraphicsUnregisterResource",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFacePositiveX",
            (
                "hipGraphicsCubeFacePositiveX",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFaceNegativeX",
            (
                "hipGraphicsCubeFaceNegativeX",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFacePositiveY",
            (
                "hipGraphicsCubeFacePositiveY",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFaceNegativeY",
            (
                "hipGraphicsCubeFaceNegativeY",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFacePositiveZ",
            (
                "hipGraphicsCubeFacePositiveZ",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsCubeFaceNegativeZ",
            (
                "hipGraphicsCubeFaceNegativeZ",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsMapFlagsNone",
            ("hipGraphicsMapFlagsNone", c.CONV_GRAPHICS, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsMapFlagsReadOnly",
            (
                "hipGraphicsMapFlagsReadOnly",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsMapFlagsWriteDiscard",
            (
                "hipGraphicsMapFlagsWriteDiscard",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsNone",
            (
                "hipGraphicsRegisterFlagsNone",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsReadOnly",
            (
                "hipGraphicsRegisterFlagsReadOnly",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsWriteDiscard",
            (
                "hipGraphicsRegisterFlagsWriteDiscard",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsSurfaceLoadStore",
            (
                "hipGraphicsRegisterFlagsSurfaceLoadStore",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsRegisterFlagsTextureGather",
            (
                "hipGraphicsRegisterFlagsTextureGather",
                c.CONV_GRAPHICS,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGLDeviceListAll",
            ("HIP_GL_DEVICE_LIST_ALL", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLDeviceListCurrentFrame",
            ("HIP_GL_DEVICE_LIST_CURRENT_FRAME", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLDeviceListNextFrame",
            ("HIP_GL_DEVICE_LIST_NEXT_FRAME", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLGetDevices",
            ("hipGLGetDevices", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterBuffer",
            ("hipGraphicsGLRegisterBuffer", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsGLRegisterImage",
            ("hipGraphicsGLRegisterImage", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaWGLGetDevice",
            ("hipWGLGetDevice", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLMapFlagsNone",
            ("HIP_GL_MAP_RESOURCE_FLAGS_NONE", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLMapFlagsReadOnly",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY",
                c.CONV_GL,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGLMapFlagsWriteDiscard",
            (
                "HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
                c.CONV_GL,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGLMapBufferObject",
            ("hipGLMapBufferObject__", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLMapBufferObjectAsync",
            ("hipGLMapBufferObjectAsync__", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLRegisterBufferObject",
            ("hipGLRegisterBufferObject", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLSetBufferObjectMapFlags",
            ("hipGLSetBufferObjectMapFlags", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLSetGLDevice",
            ("hipGLSetGLDevice", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLUnmapBufferObject",
            ("hipGLUnmapBufferObject", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLUnmapBufferObjectAsync",
            ("hipGLUnmapBufferObjectAsync", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGLUnregisterBufferObject",
            ("hipGLUnregisterBufferObject", c.CONV_GL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9DeviceListAll",
            ("HIP_D3D9_DEVICE_LIST_ALL", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9DeviceListCurrentFrame",
            (
                "HIP_D3D9_DEVICE_LIST_CURRENT_FRAME",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9DeviceListNextFrame",
            (
                "HIP_D3D9_DEVICE_LIST_NEXT_FRAME",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9GetDevice",
            ("hipD3D9GetDevice", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9GetDevices",
            ("hipD3D9GetDevices", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9GetDirect3DDevice",
            ("hipD3D9GetDirect3DDevice", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9SetDirect3DDevice",
            ("hipD3D9SetDirect3DDevice", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D9RegisterResource",
            (
                "hipGraphicsD3D9RegisterResource",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9MapFlags",
            ("hipD3D9MapFlags", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9MapFlagsNone",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_NONE",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9MapFlagsReadOnly",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9MapFlagsWriteDiscard",
            (
                "HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9RegisterFlagsNone",
            ("HIP_D3D9_REGISTER_FLAGS_NONE", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9RegisterFlagsArray",
            ("HIP_D3D9_REGISTER_FLAGS_ARRAY", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9MapResources",
            ("hipD3D9MapResources", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9RegisterResource",
            ("hipD3D9RegisterResource", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetMappedArray",
            ("hipD3D9ResourceGetMappedArray", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetMappedPitch",
            ("hipD3D9ResourceGetMappedPitch", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetMappedPointer",
            (
                "hipD3D9ResourceGetMappedPointer",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9ResourceGetMappedSize",
            ("hipD3D9ResourceGetMappedSize", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9ResourceGetSurfaceDimensions",
            (
                "hipD3D9ResourceGetSurfaceDimensions",
                c.CONV_D3D9,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D9ResourceSetMapFlags",
            ("hipD3D9ResourceSetMapFlags", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9UnmapResources",
            ("hipD3D9UnmapResources", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9UnregisterResource",
            ("hipD3D9UnregisterResource", c.CONV_D3D9, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10DeviceListAll",
            ("HIP_D3D10_DEVICE_LIST_ALL", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10DeviceListCurrentFrame",
            (
                "HIP_D3D10_DEVICE_LIST_CURRENT_FRAME",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10DeviceListNextFrame",
            (
                "HIP_D3D10_DEVICE_LIST_NEXT_FRAME",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10GetDevice",
            ("hipD3D10GetDevice", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10GetDevices",
            ("hipD3D10GetDevices", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D10RegisterResource",
            (
                "hipGraphicsD3D10RegisterResource",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10MapFlagsNone",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_NONE",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10MapFlagsReadOnly",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10MapFlagsWriteDiscard",
            (
                "HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10RegisterFlagsNone",
            ("HIP_D3D10_REGISTER_FLAGS_NONE", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10RegisterFlagsArray",
            (
                "HIP_D3D10_REGISTER_FLAGS_ARRAY",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10GetDirect3DDevice",
            ("hipD3D10GetDirect3DDevice", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10MapResources",
            ("hipD3D10MapResources", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10RegisterResource",
            ("hipD3D10RegisterResource", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10ResourceGetMappedArray",
            (
                "hipD3D10ResourceGetMappedArray",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceGetMappedPitch",
            (
                "hipD3D10ResourceGetMappedPitch",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceGetMappedPointer",
            (
                "hipD3D10ResourceGetMappedPointer",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceGetMappedSize",
            ("hipD3D10ResourceGetMappedSize", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10ResourceGetSurfaceDimensions",
            (
                "hipD3D10ResourceGetSurfaceDimensions",
                c.CONV_D3D10,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D10ResourceSetMapFlags",
            ("hipD3D10ResourceSetMapFlags", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10SetDirect3DDevice",
            ("hipD3D10SetDirect3DDevice", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10UnmapResources",
            ("hipD3D10UnmapResources", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10UnregisterResource",
            ("hipD3D10UnregisterResource", c.CONV_D3D10, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11DeviceListAll",
            ("HIP_D3D11_DEVICE_LIST_ALL", c.CONV_D3D11, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11DeviceListCurrentFrame",
            (
                "HIP_D3D11_DEVICE_LIST_CURRENT_FRAME",
                c.CONV_D3D11,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D11DeviceListNextFrame",
            (
                "HIP_D3D11_DEVICE_LIST_NEXT_FRAME",
                c.CONV_D3D11,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D11GetDevice",
            ("hipD3D11GetDevice", c.CONV_D3D11, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11GetDevices",
            ("hipD3D11GetDevices", c.CONV_D3D11, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D11RegisterResource",
            (
                "hipGraphicsD3D11RegisterResource",
                c.CONV_D3D11,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaD3D11GetDevice",
            ("hipD3D11GetDevice", c.CONV_D3D11, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11GetDevices",
            ("hipD3D11GetDevices", c.CONV_D3D11, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsD3D11RegisterResource",
            (
                "hipGraphicsD3D11RegisterResource",
                c.CONV_D3D11,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsVDPAURegisterOutputSurface",
            (
                "hipGraphicsVDPAURegisterOutputSurface",
                c.CONV_VDPAU,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaGraphicsVDPAURegisterVideoSurface",
            (
                "hipGraphicsVDPAURegisterVideoSurface",
                c.CONV_VDPAU,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaVDPAUGetDevice",
            ("hipVDPAUGetDevice", c.CONV_VDPAU, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaVDPAUSetVDPAUDevice",
            ("hipVDPAUSetDevice", c.CONV_VDPAU, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamConsumerAcquireFrame",
            (
                "hipEGLStreamConsumerAcquireFrame",
                c.CONV_EGL,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamConsumerConnect",
            ("hipEGLStreamConsumerConnect", c.CONV_EGL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamConsumerConnectWithFlags",
            (
                "hipEGLStreamConsumerConnectWithFlags",
                c.CONV_EGL,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamConsumerReleaseFrame",
            (
                "hipEGLStreamConsumerReleaseFrame",
                c.CONV_EGL,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamProducerConnect",
            ("hipEGLStreamProducerConnect", c.CONV_EGL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamProducerDisconnect",
            ("hipEGLStreamProducerDisconnect", c.CONV_EGL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaEGLStreamProducerPresentFrame",
            (
                "hipEGLStreamProducerPresentFrame",
                c.CONV_EGL,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaEGLStreamProducerReturnFrame",
            ("hipEGLStreamProducerReturnFrame", c.CONV_EGL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsEGLRegisterImage",
            ("hipGraphicsEGLRegisterImage", c.CONV_EGL, c.API_RUNTIME, c.HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsResourceGetMappedEglFrame",
            (
                "hipGraphicsResourceGetMappedEglFrame",
                c.CONV_EGL,
                c.API_RUNTIME,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cublasInit", ("rocblas_init", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasShutdown",
            ("rocblas_shutdown", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasGetVersion",
            ("rocblas_get_version", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasGetError",
            ("rocblas_get_error", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasAlloc", ("rocblas_alloc", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasFree", ("rocblas_free", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasSetKernelStream",
            ("rocblas_set_kernel_stream", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasGetAtomicsMode",
            ("rocblas_get_atomics_mode", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSetAtomicsMode",
            ("rocblas_set_atomics_mode", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasGetMathMode",
            ("rocblas_get_math_mode", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSetMathMode",
            ("rocblas_set_math_mode", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("CUBLAS_OP_N", ("rocblas_operation_none", c.CONV_NUMERIC_LITERAL, c.API_BLAS)),
        (
            "CUBLAS_OP_T",
            ("rocblas_operation_transpose", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_OP_C",
            ("rocblas_operation_conjugate_transpose", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_SUCCESS",
            ("rocblas_status_success", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_NOT_INITIALIZED",
            ("rocblas_status_invalid_handle", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_ALLOC_FAILED",
            ("rocblas_status_memory_error", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_INVALID_VALUE",
            ("rocblas_status_invalid_pointer", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_MAPPING_ERROR",
            ("rocblas_status_internal_error", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_EXECUTION_FAILED",
            ("rocblas_status_internal_error", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_INTERNAL_ERROR",
            ("rocblas_status_internal_error", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_NOT_SUPPORTED",
            ("rocblas_status_not_implemented", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_STATUS_ARCH_MISMATCH",
            ("rocblas_status_not_implemented", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_FILL_MODE_LOWER",
            ("rocblas_fill_lower", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_FILL_MODE_UPPER",
            ("rocblas_fill_upper", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_DIAG_NON_UNIT",
            ("rocblas_diagonal_non_unit", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        ("CUBLAS_DIAG_UNIT", ("rocblas_diagonal_unit", c.CONV_NUMERIC_LITERAL, c.API_BLAS)),
        ("CUBLAS_SIDE_LEFT", ("rocblas_side_left", c.CONV_NUMERIC_LITERAL, c.API_BLAS)),
        ("CUBLAS_SIDE_RIGHT", ("rocblas_side_right", c.CONV_NUMERIC_LITERAL, c.API_BLAS)),
        (
            "CUBLAS_POINTER_MODE_HOST",
            ("rocblas_pointer_mode_host", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_POINTER_MODE_DEVICE",
            ("rocblas_pointer_mode_device", c.CONV_NUMERIC_LITERAL, c.API_BLAS),
        ),
        (
            "CUBLAS_ATOMICS_NOT_ALLOWED",
            (
                "rocblas_atomics_not_allowed",
                c.CONV_NUMERIC_LITERAL,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_ATOMICS_ALLOWED",
            (
                "rocblas_atomics_allowed",
                c.CONV_NUMERIC_LITERAL,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_DATA_FLOAT",
            (
                "rocblas_precision_float",
                c.CONV_NUMERIC_LITERAL,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_DATA_DOUBLE",
            (
                "rocblas_precision_double",
                c.CONV_NUMERIC_LITERAL,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUBLAS_DATA_HALF",
            ("rocblas_precision_half", c.CONV_NUMERIC_LITERAL, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "CUBLAS_DATA_INT8",
            ("rocblas_precision_int8", c.CONV_NUMERIC_LITERAL, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasCreate", ("rocblas_create_handle", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDestroy", ("rocblas_destroy_handle", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasSetVector", ("rocblas_set_vector", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasGetVector", ("rocblas_get_vector", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasSetVectorAsync",
            ("rocblas_set_vector_async", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasGetVectorAsync",
            ("rocblas_get_vector_async", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSetMatrix", ("rocblas_set_matrix", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasGetMatrix", ("rocblas_get_matrix", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasGetMatrixAsync",
            ("rocblas_get_matrix_async", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSetMatrixAsync",
            ("rocblas_set_matrix_async", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasXerbla", ("rocblas_xerbla", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSnrm2", ("rocblas_snrm2", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDnrm2", ("rocblas_dnrm2", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasScnrm2", ("rocblas_scnrm2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDznrm2", ("rocblas_dznrm2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasNrm2Ex",
            ("rocblas_nrm2_ex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSdot", ("rocblas_sdot", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasSdotBatched",
            ("rocblas_sdot_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasDdot", ("rocblas_ddot", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasDdotBatched",
            ("rocblas_ddot_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasCdotu", ("rocblas_cdotu", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCdotc", ("rocblas_cdotc", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasZdotu", ("rocblas_zdotu", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasZdotc", ("rocblas_zdotc", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasSscal", ("rocblas_sscal", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasSscalBatched",
            ("rocblas_sscal_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasDscal", ("rocblas_dscal", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasDscalBatched",
            ("rocblas_dscal_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasCscal", ("rocblas_cscal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsscal", ("rocblas_csscal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZscal", ("rocblas_zscal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZdscal", ("rocblas_zdscal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSaxpy", ("rocblas_saxpy", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasSaxpyBatched",
            ("rocblas_saxpy_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasDaxpy", ("rocblas_daxpy", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCaxpy", ("rocblas_caxpy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZaxpy", ("rocblas_zaxpy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasScopy", ("rocblas_scopy", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasScopyBatched",
            ("rocblas_scopy_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasDcopy", ("rocblas_dcopy", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasDcopyBatched",
            ("rocblas_dcopy_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasCcopy", ("rocblas_ccopy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZcopy", ("rocblas_zcopy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSswap", ("rocblas_sswap", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDswap", ("rocblas_dswap", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCswap", ("rocblas_cswap", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZswap", ("rocblas_zswap", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasIsamax", ("rocblas_isamax", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasIdamax", ("rocblas_idamax", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasIcamax", ("rocblas_icamax", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasIzamax", ("rocblas_izamax", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasIsamin", ("rocblas_isamin", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasIdamin", ("rocblas_idamin", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasIcamin", ("rocblas_icamin", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasIzamin", ("rocblas_izamin", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSasum", ("rocblas_sasum", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasSasumBatched",
            ("rocblas_sasum_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasDasum", ("rocblas_dasum", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasDasumBatched",
            ("rocblas_dasum_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasScasum", ("rocblas_scasum", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDzasum", ("rocblas_dzasum", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSrot", ("rocblas_srot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDrot", ("rocblas_drot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCrot", ("rocblas_crot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsrot", ("rocblas_csrot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZrot", ("rocblas_zrot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZdrot", ("rocblas_zdrot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSrotg", ("rocblas_srotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDrotg", ("rocblas_drotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCrotg", ("rocblas_crotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZrotg", ("rocblas_zrotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSrotm", ("rocblas_srotm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDrotm", ("rocblas_drotm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSrotmg", ("rocblas_srotmg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDrotmg", ("rocblas_drotmg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSgemv", ("rocblas_sgemv", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasSgemvBatched",
            ("rocblas_sgemv_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasDgemv", ("rocblas_dgemv", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCgemv", ("rocblas_cgemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZgemv", ("rocblas_zgemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSgbmv", ("rocblas_sgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDgbmv", ("rocblas_dgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCgbmv", ("rocblas_cgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZgbmv", ("rocblas_zgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStrmv", ("rocblas_strmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtrmv", ("rocblas_dtrmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtrmv", ("rocblas_ctrmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtrmv", ("rocblas_ztrmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStbmv", ("rocblas_stbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtbmv", ("rocblas_dtbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtbmv", ("rocblas_ctbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtbmv", ("rocblas_ztbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStpmv", ("rocblas_stpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtpmv", ("rocblas_dtpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtpmv", ("rocblas_ctpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtpmv", ("rocblas_ztpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStrsv", ("rocblas_strsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtrsv", ("rocblas_dtrsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtrsv", ("rocblas_ctrsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtrsv", ("rocblas_ztrsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStpsv", ("rocblas_stpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtpsv", ("rocblas_dtpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtpsv", ("rocblas_ctpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtpsv", ("rocblas_ztpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStbsv", ("rocblas_stbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtbsv", ("rocblas_dtbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtbsv", ("rocblas_ctbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtbsv", ("rocblas_ztbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSsymv", ("rocblas_ssymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsymv", ("rocblas_dsymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsymv", ("rocblas_csymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZsymv", ("rocblas_zsymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasChemv", ("rocblas_chemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZhemv", ("rocblas_zhemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSsbmv", ("rocblas_ssbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsbmv", ("rocblas_dsbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasChbmv", ("rocblas_chbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZhbmv", ("rocblas_zhbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSspmv", ("rocblas_sspmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDspmv", ("rocblas_dspmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasChpmv", ("rocblas_chpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZhpmv", ("rocblas_zhpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSger", ("rocblas_sger", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDger", ("rocblas_dger", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCgeru", ("rocblas_cgeru", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCgerc", ("rocblas_cgerc", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZgeru", ("rocblas_zgeru", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZgerc", ("rocblas_zgerc", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSsyr", ("rocblas_ssyr", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDsyr", ("rocblas_dsyr", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCher", ("rocblas_cher", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZher", ("rocblas_zher", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSspr", ("rocblas_sspr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDspr", ("rocblas_dspr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasChpr", ("rocblas_chpr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZhpr", ("rocblas_zhpr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSsyr2", ("rocblas_ssyr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsyr2", ("rocblas_dsyr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCher2", ("rocblas_cher2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZher2", ("rocblas_zher2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSspr2", ("rocblas_sspr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDspr2", ("rocblas_dspr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasChpr2", ("rocblas_chpr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZhpr2", ("rocblas_zhpr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasSgemmBatched",
            ("rocblas_sgemm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDgemmBatched",
            ("rocblas_dgemm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasHgemmBatched",
            ("rocblas_hgemm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSgemmStridedBatched",
            ("rocblas_sgemm_strided_batched", c.CONV_MATH_FUNC, c.API_BLAS),
        ),
        (
            "cublasDgemmStridedBatched",
            ("rocblas_dgemm_strided_batched", c.CONV_MATH_FUNC, c.API_BLAS),
        ),
        (
            "cublasHgemmStridedBatched",
            ("rocblas_hgemm_strided_batched", c.CONV_MATH_FUNC, c.API_BLAS),
        ),
        (
            "cublasCgemmBatched",
            ("rocblas_cgemm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemm3mBatched",
            ("rocblas_cgemm_3m_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemmBatched",
            ("rocblas_zgemm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemmStridedBatched",
            (
                "rocblas_cgemm_strided_batched",
                c.CONV_MATH_FUNC,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cublasCgemm3mStridedBatched",
            (
                "rocblas_cgemm_3m_strided_batched",
                c.CONV_MATH_FUNC,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cublasZgemmStridedBatched",
            (
                "rocblas_zgemm_strided_batched",
                c.CONV_MATH_FUNC,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "cublasHgemmStridedBatched",
            (
                "rocblas_hgemm_strided_batched",
                c.CONV_MATH_FUNC,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cublasSgemm", ("rocblas_sgemm", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDgemm", ("rocblas_dgemm", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCgemm", ("rocblas_cgemm", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasZgemm", ("rocblas_zgemm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasHgemm", ("rocblas_hgemm", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasSsyrk", ("rocblas_ssyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsyrk", ("rocblas_dsyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsyrk", ("rocblas_csyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZsyrk", ("rocblas_zsyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCherk", ("rocblas_cherk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZherk", ("rocblas_zherk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSsyr2k", ("rocblas_ssyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsyr2k", ("rocblas_dsyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsyr2k", ("rocblas_csyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZsyr2k", ("rocblas_zyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSsyrkx", ("rocblas_ssyrkx", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsyrkx", ("rocblas_dsyrkx", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsyrkx", ("rocblas_csyrkx", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZsyrkx", ("rocblas_zsyrkx", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCher2k", ("rocblas_cher2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZher2k", ("rocblas_zher2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCherkx", ("rocblas_cherkx", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZherkx", ("rocblas_zherkx", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSsymm", ("rocblas_ssymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsymm", ("rocblas_dsymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsymm", ("rocblas_csymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZsymm", ("rocblas_zsymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasChemm", ("rocblas_chemm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZhemm", ("rocblas_zhemm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStrsm", ("rocblas_strsm", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDtrsm", ("rocblas_dtrsm", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCtrsm", ("rocblas_ctrsm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtrsm", ("rocblas_ztrsm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasStrsmBatched",
            ("rocblas_strsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsmBatched",
            ("rocblas_dtrsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsmBatched",
            ("rocblas_ctrsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsmBatched",
            ("rocblas_ztrsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasStrmm", ("rocblas_strmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtrmm", ("rocblas_dtrmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtrmm", ("rocblas_ctrmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtrmm", ("rocblas_ztrmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSgeam", ("rocblas_sgeam", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDgeam", ("rocblas_dgeam", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasCgeam", ("rocblas_cgeam", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZgeam", ("rocblas_zgeam", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasSgetrfBatched",
            ("rocblas_sgetrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDgetrfBatched",
            ("rocblas_dgetrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgetrfBatched",
            ("rocblas_cgetrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgetrfBatched",
            ("rocblas_zgetrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSgetriBatched",
            ("rocblas_sgetri_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDgetriBatched",
            ("rocblas_dgetri_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgetriBatched",
            ("rocblas_cgetri_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgetriBatched",
            ("rocblas_zgetri_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSgetrsBatched",
            ("rocblas_sgetrs_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDgetrsBatched",
            ("rocblas_dgetrs_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgetrsBatched",
            ("rocblas_cgetrs_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgetrsBatched",
            ("rocblas_zgetrs_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStrsmBatched",
            ("rocblas_strsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsmBatched",
            ("rocblas_dtrsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsmBatched",
            ("rocblas_ctrsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsmBatched",
            ("rocblas_dtrsm_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSmatinvBatched",
            ("rocblas_smatinv_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDmatinvBatched",
            ("rocblas_dmatinv_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCmatinvBatched",
            ("rocblas_cmatinv_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZmatinvBatched",
            ("rocblas_zmatinv_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSgeqrfBatched",
            ("rocblas_sgeqrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDgeqrfBatched",
            ("rocblas_dgeqrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgeqrfBatched",
            ("rocblas_cgeqrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgeqrfBatched",
            ("rocblas_zgeqrf_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSgelsBatched",
            ("rocblas_sgels_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDgelsBatched",
            ("rocblas_dgels_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgelsBatched",
            ("rocblas_cgels_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgelsBatched",
            ("rocblas_zgels_batched", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSdgmm", ("rocblas_sdgmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDdgmm", ("rocblas_ddgmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCdgmm", ("rocblas_cdgmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZdgmm", ("rocblas_zdgmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStpttr", ("rocblas_stpttr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtpttr", ("rocblas_dtpttr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtpttr", ("rocblas_ctpttr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtpttr", ("rocblas_ztpttr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasStrttp", ("rocblas_strttp", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDtrttp", ("rocblas_dtrttp", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCtrttp", ("rocblas_ctrttp", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZtrttp", ("rocblas_ztrttp", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCreate_v2", ("rocblas_create_handle", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDestroy_v2", ("rocblas_destroy_handle", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasGetVersion_v2",
            ("rocblas_get_version", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSetStream", ("rocblas_set_stream", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasGetStream", ("rocblas_get_stream", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasSetStream_v2", ("rocblas_set_stream", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasGetStream_v2", ("rocblas_get_stream", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasGetPointerMode",
            ("rocblas_get_pointer_mode", c.CONV_MATH_FUNC, c.API_BLAS),
        ),
        (
            "cublasSetPointerMode",
            ("rocblas_set_pointer_mode", c.CONV_MATH_FUNC, c.API_BLAS),
        ),
        (
            "cublasGetPointerMode_v2",
            ("rocblas_get_pointer_mode", c.CONV_MATH_FUNC, c.API_BLAS),
        ),
        (
            "cublasSetPointerMode_v2",
            ("rocblas_set_pointer_mode", c.CONV_MATH_FUNC, c.API_BLAS),
        ),
        ("cublasSgemv_v2", ("rocblas_sgemv", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDgemv_v2", ("rocblas_dgemv", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCgemv_v2",
            ("rocblas_cgemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemv_v2",
            ("rocblas_zgemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSgbmv_v2",
            ("rocblas_sgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDgbmv_v2",
            ("rocblas_dgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgbmv_v2",
            ("rocblas_cgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgbmv_v2",
            ("rocblas_zgbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStrmv_v2",
            ("rocblas_strmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrmv_v2",
            ("rocblas_dtrmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrmv_v2",
            ("rocblas_ctrmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrmv_v2",
            ("rocblas_ztrmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStbmv_v2",
            ("rocblas_stbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtbmv_v2",
            ("rocblas_dtbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtbmv_v2",
            ("rocblas_ctbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtbmv_v2",
            ("rocblas_ztbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStpmv_v2",
            ("rocblas_stpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtpmv_v2",
            ("rocblas_dtpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtpmv_v2",
            ("rocblas_ctpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtpmv_v2",
            ("rocblas_ztpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStrsv_v2",
            ("rocblas_strsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsv_v2",
            ("rocblas_dtrsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsv_v2",
            ("rocblas_ctrsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsv_v2",
            ("rocblas_ztrsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStpsv_v2",
            ("rocblas_stpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtpsv_v2",
            ("rocblas_dtpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtpsv_v2",
            ("rocblas_ctpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtpsv_v2",
            ("rocblas_ztpsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStbsv_v2",
            ("rocblas_stbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtbsv_v2",
            ("rocblas_dtbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtbsv_v2",
            ("rocblas_ctbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtbsv_v2",
            ("rocblas_ztbsv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSsymv_v2",
            ("rocblas_ssymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDsymv_v2",
            ("rocblas_dsymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsymv_v2",
            ("rocblas_csymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZsymv_v2",
            ("rocblas_zsymv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasChemv_v2",
            ("rocblas_chemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZhemv_v2",
            ("rocblas_zhemv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSsbmv_v2",
            ("rocblas_ssbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDsbmv_v2",
            ("rocblas_dsbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasChbmv_v2",
            ("rocblas_chbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZhbmv_v2",
            ("rocblas_zhbmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSspmv_v2",
            ("rocblas_sspmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDspmv_v2",
            ("rocblas_dspmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasChpmv_v2",
            ("rocblas_chpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZhpmv_v2",
            ("rocblas_zhpmv", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSger_v2", ("rocblas_sger", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDger_v2", ("rocblas_dger", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCgeru_v2",
            ("rocblas_cgeru", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgerc_v2",
            ("rocblas_cergc", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgeru_v2",
            ("rocblas_zgeru", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgerc_v2",
            ("rocblas_zgerc", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSsyr_v2", ("rocblas_ssyr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDsyr_v2", ("rocblas_dsyr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCsyr_v2", ("rocblas_csyr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZsyr_v2", ("rocblas_zsyr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCher_v2", ("rocblas_cher", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZher_v2", ("rocblas_zher", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSspr_v2", ("rocblas_sspr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDspr_v2", ("rocblas_dspr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasChpr_v2", ("rocblas_chpr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasZhpr_v2", ("rocblas_zhpr", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasSsyr2_v2",
            ("rocblas_ssyr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDsyr2_v2",
            ("rocblas_dsyr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyr2_v2",
            ("rocblas_csyr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZsyr2_v2",
            ("rocblas_zsyr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCher2_v2",
            ("rocblas_cher2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZher2_v2",
            ("rocblas_zher2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSspr2_v2",
            ("rocblas_sspr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDspr2_v2",
            ("rocblas_dspr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasChpr2_v2",
            ("rocblas_chpr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZhpr2_v2",
            ("rocblas_zhpr2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSgemm_v2", ("rocblas_sgemm", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDgemm_v2", ("rocblas_dgemm", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCgemm_v2",
            ("rocblas_cgemm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemm3m",
            ("rocblas_cgemm_3m", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCgemm3mEx",
            ("rocblas_cgemm_3mex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemm_v2",
            ("rocblas_zgemm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZgemm3m",
            ("rocblas_zgemm_3m", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        # NB: The function rocblas_sgemmex doesn't actually exist in
        # rocblas, as of 2018-12-05
        (
            "cublasSgemmEx",
            ("rocblas_sgemmex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasGemmEx", ("rocblas_gemmex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasCgemmEx",
            ("rocblas_cgemmex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasUint8gemmBias",
            ("rocblas_uint8gemmbias", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSsyrk_v2",
            ("rocblas_ssyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDsyrk_v2",
            ("rocblas_dsyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyrk_v2",
            ("rocblas_csyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZsyrk_v2",
            ("rocblas_zsyrk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyrkEx",
            ("rocblas_csyrkex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyrk3mEx",
            ("rocblas_csyrk3mex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCherk_v2",
            ("rocblas_cherk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCherkEx",
            ("rocblas_cherkex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCherk3mEx",
            ("rocblas_cherk3mex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZherk_v2",
            ("rocblas_zherk", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSsyr2k_v2",
            ("rocblas_ssyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDsyr2k_v2",
            ("rocblas_dsyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsyr2k_v2",
            ("rocblas_csyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZsyr2k_v2",
            ("rocblas_zsyr2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCher2k_v2",
            ("rocblas_cher2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZher2k_v2",
            ("rocblas_zher2k", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSsymm_v2",
            ("rocblas_ssymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDsymm_v2",
            ("rocblas_dsymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsymm_v2",
            ("rocblas_csymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZsymm_v2",
            ("rocblas_zsymm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasChemm_v2",
            ("rocblas_chemm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZhemm_v2",
            ("rocblas_zhemm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStrsm_v2",
            ("rocblas_strsm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrsm_v2",
            ("rocblas_dtrsm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrsm_v2",
            ("rocblas_ctrsm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrsm_v2",
            ("rocblas_ztrsm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasStrmm_v2",
            ("rocblas_strmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDtrmm_v2",
            ("rocblas_dtrmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCtrmm_v2",
            ("rocblas_ctrmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZtrmm_v2",
            ("rocblas_ztrmm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSnrm2_v2", ("rocblas_snrm2", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDnrm2_v2", ("rocblas_dnrm2", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasScnrm2_v2",
            ("rocblas_scnrm2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDznrm2_v2",
            ("rocblas_dznrm2", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasDotEx", ("rocblas_dotex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDotcEx", ("rocblas_dotcex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSdot_v2", ("rocblas_sdot", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDdot_v2", ("rocblas_ddot", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCdotu_v2",
            ("rocblas_cdotu", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCdotc_v2",
            ("rocblas_cdotc", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZdotu_v2",
            ("rocblas_zdotu", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZdotc_v2",
            ("rocblas_zdotc", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasScalEx", ("rocblas_scalex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSscal_v2", ("rocblas_sscal", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDscal_v2", ("rocblas_dscal", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCscal_v2",
            ("rocblas_cscal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCsscal_v2",
            ("rocblas_csscal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZscal_v2",
            ("rocblas_zcsal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZdscal_v2",
            ("rocblas_zdscal", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasAxpyEx", ("rocblas_axpyex", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasSaxpy_v2", ("rocblas_saxpy", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDaxpy_v2", ("rocblas_daxpy", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCaxpy_v2",
            ("rocblas_caxpy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZaxpy_v2",
            ("rocblas_zaxpy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasScopy_v2", ("rocblas_scopy", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDcopy_v2", ("rocblas_dcopy", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCcopy_v2",
            ("rocblas_ccopy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZcopy_v2",
            ("rocblas_zcopy", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSswap_v2", ("rocblas_sswap", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDswap_v2", ("rocblas_dswap", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasCswap_v2",
            ("rocblas_cswap", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZswap_v2",
            ("rocblas_zswap", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasIsamax_v2", ("rocblas_isamax", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasIdamax_v2", ("rocblas_idamax", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasIcamax_v2",
            ("rocblas_icamax", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasIzamax_v2",
            ("rocblas_izamax", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasIsamin_v2", ("rocblas_isamin", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasIdamin_v2", ("rocblas_idamin", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasIcamin_v2",
            ("rocblas_icamin", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasIzamin_v2",
            ("rocblas_izamin", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSasum_v2", ("rocblas_sasum", c.CONV_MATH_FUNC, c.API_BLAS)),
        ("cublasDasum_v2", ("rocblas_dasum", c.CONV_MATH_FUNC, c.API_BLAS)),
        (
            "cublasScasum_v2",
            ("rocblas_scasum", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDzasum_v2",
            ("rocblas_dzasum", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasSrot_v2", ("rocblas_srot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasDrot_v2", ("rocblas_drot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        ("cublasCrot_v2", ("rocblas_crot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasCsrot_v2",
            ("rocblas_csrot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        ("cublasZrot_v2", ("rocblas_zrot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED)),
        (
            "cublasZdrot_v2",
            ("rocblas_zdrot", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSrotg_v2",
            ("rocblas_srotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDrotg_v2",
            ("rocblas_drotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasCrotg_v2",
            ("rocblas_crotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasZrotg_v2",
            ("rocblas_zrotg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSrotm_v2",
            ("rocblas_srotm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDrotm_v2",
            ("rocblas_drotm", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasSrotmg_v2",
            ("rocblas_srotmg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "cublasDrotmg_v2",
            ("rocblas_drotmg", c.CONV_MATH_FUNC, c.API_BLAS, c.HIP_UNSUPPORTED),
        ),
        (
            "CURAND_STATUS_SUCCESS",
            ("HIPRAND_STATUS_SUCCESS", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_VERSION_MISMATCH",
            ("HIPRAND_STATUS_VERSION_MISMATCH", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_NOT_INITIALIZED",
            ("HIPRAND_STATUS_NOT_INITIALIZED", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_ALLOCATION_FAILED",
            ("HIPRAND_STATUS_ALLOCATION_FAILED", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_TYPE_ERROR",
            ("HIPRAND_STATUS_TYPE_ERROR", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_OUT_OF_RANGE",
            ("HIPRAND_STATUS_OUT_OF_RANGE", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
            ("HIPRAND_STATUS_LENGTH_NOT_MULTIPLE", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
            (
                "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
            ),
        ),
        (
            "CURAND_STATUS_LAUNCH_FAILURE",
            ("HIPRAND_STATUS_LAUNCH_FAILURE", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_PREEXISTING_FAILURE",
            ("HIPRAND_STATUS_PREEXISTING_FAILURE", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_INITIALIZATION_FAILED",
            ("HIPRAND_STATUS_INITIALIZATION_FAILED", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_ARCH_MISMATCH",
            ("HIPRAND_STATUS_ARCH_MISMATCH", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_STATUS_INTERNAL_ERROR",
            ("HIPRAND_STATUS_INTERNAL_ERROR", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        ("CURAND_RNG_TEST", ("HIPRAND_RNG_TEST", c.CONV_NUMERIC_LITERAL, c.API_RAND)),
        (
            "mtgp32dc_params_fast_11213",
            ("mtgp32dc_params_fast_11213", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_DEFAULT",
            ("HIPRAND_RNG_PSEUDO_DEFAULT", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_XORWOW",
            ("HIPRAND_RNG_PSEUDO_XORWOW", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_MRG32K3A",
            ("HIPRAND_RNG_PSEUDO_MRG32K3A", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_MTGP32",
            ("HIPRAND_RNG_PSEUDO_MTGP32", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_MT19937",
            ("HIPRAND_RNG_PSEUDO_MT19937", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_PSEUDO_PHILOX4_32_10",
            ("HIPRAND_RNG_PSEUDO_PHILOX4_32_10", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_DEFAULT",
            ("HIPRAND_RNG_QUASI_DEFAULT", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SOBOL32",
            ("HIPRAND_RNG_QUASI_SOBOL32", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",
            ("HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SOBOL64",
            ("HIPRAND_RNG_QUASI_SOBOL64", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",
            ("HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64", c.CONV_NUMERIC_LITERAL, c.API_RAND),
        ),
        (
            "curand_ORDERING_PSEUDO_BEST",
            (
                "HIPRAND_ORDERING_PSEUDO_BEST",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_ORDERING_PSEUDO_DEFAULT",
            (
                "HIPRAND_ORDERING_PSEUDO_DEFAULT",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_ORDERING_PSEUDO_SEEDED",
            (
                "HIPRAND_ORDERING_PSEUDO_SEEDED",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_ORDERING_QUASI_DEFAULT",
            (
                "HIPRAND_ORDERING_QUASI_DEFAULT",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_DIRECTION_VECTORS_32_JOEKUO6",
            (
                "HIPRAND_DIRECTION_VECTORS_32_JOEKUO6",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
            (
                "HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_DIRECTION_VECTORS_64_JOEKUO6",
            (
                "HIPRAND_DIRECTION_VECTORS_64_JOEKUO6",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
            (
                "HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
                c.CONV_NUMERIC_LITERAL,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_CHOOSE_BEST",
            ("HIPRAND_CHOOSE_BEST", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_ITR",
            ("HIPRAND_ITR", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_KNUTH",
            ("HIPRAND_KNUTH", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_HITR",
            ("HIPRAND_HITR", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("curand_M1", ("HIPRAND_M1", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED)),
        ("curand_M2", ("HIPRAND_M2", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED)),
        (
            "curand_BINARY_SEARCH",
            ("HIPRAND_BINARY_SEARCH", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_DISCRETE_GAUSS",
            ("HIPRAND_DISCRETE_GAUSS", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_REJECTION",
            ("HIPRAND_REJECTION", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_DEVICE_API",
            ("HIPRAND_DEVICE_API", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_FAST_REJECTION",
            ("HIPRAND_FAST_REJECTION", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_3RD",
            ("HIPRAND_3RD", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_DEFINITION",
            ("HIPRAND_DEFINITION", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_POISSON",
            ("HIPRAND_POISSON", c.CONV_NUMERIC_LITERAL, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("curandCreateGenerator", ("hiprandCreateGenerator", c.CONV_MATH_FUNC, c.API_RAND)),
        (
            "curandCreateGeneratorHost",
            ("hiprandCreateGeneratorHost", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandCreatePoissonDistribution",
            ("hiprandCreatePoissonDistribution", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandDestroyDistribution",
            ("hiprandDestroyDistribution", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandDestroyGenerator",
            ("hiprandDestroyGenerator", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        ("curandGenerate", ("hiprandGenerate", c.CONV_MATH_FUNC, c.API_RAND)),
        (
            "curandGenerateLogNormal",
            ("hiprandGenerateLogNormal", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandGenerateLogNormalDouble",
            ("hiprandGenerateLogNormalDouble", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandGenerateLongLong",
            ("hiprandGenerateLongLong", c.CONV_MATH_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("curandGenerateNormal", ("hiprandGenerateNormal", c.CONV_MATH_FUNC, c.API_RAND)),
        (
            "curandGenerateNormalDouble",
            ("hiprandGenerateNormalDouble", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        ("curandGeneratePoisson", ("hiprandGeneratePoisson", c.CONV_MATH_FUNC, c.API_RAND)),
        ("curandGenerateSeeds", ("hiprandGenerateSeeds", c.CONV_MATH_FUNC, c.API_RAND)),
        ("curandGenerateUniform", ("hiprandGenerateUniform", c.CONV_MATH_FUNC, c.API_RAND)),
        (
            "curandGenerateUniformDouble",
            ("hiprandGenerateUniformDouble", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandGetDirectionVectors32",
            ("hiprandGetDirectionVectors32", c.CONV_MATH_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandGetDirectionVectors64",
            ("hiprandGetDirectionVectors64", c.CONV_MATH_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandGetProperty",
            ("hiprandGetProperty", c.CONV_MATH_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandGetScrambleConstants32",
            (
                "hiprandGetScrambleConstants32",
                c.CONV_MATH_FUNC,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curandGetScrambleConstants64",
            (
                "hiprandGetScrambleConstants64",
                c.CONV_MATH_FUNC,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("curandGetVersion", ("hiprandGetVersion", c.CONV_MATH_FUNC, c.API_RAND)),
        (
            "curandSetGeneratorOffset",
            ("hiprandSetGeneratorOffset", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandSetGeneratorOrdering",
            ("hiprandSetGeneratorOrdering", c.CONV_MATH_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curandSetPseudoRandomGeneratorSeed",
            ("hiprandSetPseudoRandomGeneratorSeed", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        (
            "curandSetQuasiRandomGeneratorDimensions",
            ("hiprandSetQuasiRandomGeneratorDimensions", c.CONV_MATH_FUNC, c.API_RAND),
        ),
        ("curandSetStream", ("hiprandSetStream", c.CONV_MATH_FUNC, c.API_RAND)),
        ("curand", ("hiprand", c.CONV_DEVICE_FUNC, c.API_RAND)),
        ("curand4", ("hiprand4", c.CONV_DEVICE_FUNC, c.API_RAND)),
        ("curand_init", ("hiprand_init", c.CONV_DEVICE_FUNC, c.API_RAND)),
        ("curand_log_normal", ("hiprand_log_normal", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_log_normal_double",
            ("hiprand_log_normal_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        ("curand_log_normal2", ("hiprand_log_normal2", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_log_normal2_double",
            ("hiprand_log_normal2_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        ("curand_log_normal4", ("hiprand_log_normal4", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_log_normal4_double",
            ("hiprand_log_normal4_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        (
            "curand_mtgp32_single",
            ("hiprand_mtgp32_single", c.CONV_DEVICE_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        (
            "curand_mtgp32_single_specific",
            (
                "hiprand_mtgp32_single_specific",
                c.CONV_DEVICE_FUNC,
                c.API_RAND,
                c.HIP_UNSUPPORTED,
            ),
        ),
        (
            "curand_mtgp32_specific",
            ("hiprand_mtgp32_specific", c.CONV_DEVICE_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("curand_normal", ("hiprand_normal", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curandMakeMTGP32Constants",
            ("hiprandMakeMTGP32Constants", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        (
            "curandMakeMTGP32KernelState",
            ("hiprandMakeMTGP32KernelState", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        ("curand_normal_double", ("hiprand_normal_double", c.CONV_DEVICE_FUNC, c.API_RAND)),
        ("curand_normal2", ("hiprand_normal2", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_normal2_double",
            ("hiprand_normal2_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        ("curand_normal4", ("hiprand_normal4", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_normal4_double",
            ("hiprand_normal4_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        ("curand_uniform", ("hiprand_uniform", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_uniform_double",
            ("hiprand_uniform_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        (
            "curand_uniform2_double",
            ("hiprand_uniform2_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        ("curand_uniform4", ("hiprand_uniform4", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_uniform4_double",
            ("hiprand_uniform4_double", c.CONV_DEVICE_FUNC, c.API_RAND),
        ),
        ("curand_discrete", ("hiprand_discrete", c.CONV_DEVICE_FUNC, c.API_RAND)),
        ("curand_discrete4", ("hiprand_discrete4", c.CONV_DEVICE_FUNC, c.API_RAND)),
        ("curand_poisson", ("hiprand_poisson", c.CONV_DEVICE_FUNC, c.API_RAND)),
        ("curand_poisson4", ("hiprand_poisson4", c.CONV_DEVICE_FUNC, c.API_RAND)),
        (
            "curand_Philox4x32_10",
            ("hiprand_Philox4x32_10", c.CONV_DEVICE_FUNC, c.API_RAND, c.HIP_UNSUPPORTED),
        ),
        ("mtgp32_kernel_params", ("mtgp32_kernel_params_t", c.CONV_MATH_FUNC, c.API_RAND)),
        ("CUFFT_FORWARD", ("HIPFFT_FORWARD", c.CONV_NUMERIC_LITERAL, c.API_BLAS)),
        ("CUFFT_INVERSE", ("HIPFFT_BACKWARD", c.CONV_NUMERIC_LITERAL, c.API_BLAS)),
        (
            "CUFFT_COMPATIBILITY_DEFAULT",
            (
                "HIPFFT_COMPATIBILITY_DEFAULT",
                c.CONV_NUMERIC_LITERAL,
                c.API_BLAS,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cuComplex", ("rocblas_float_complex", c.CONV_TYPE, c.API_BLAS)),
        ("cuDoubleComplex", ("rocblas_double_complex", c.CONV_TYPE, c.API_BLAS)),
        ("cufftResult_t", ("hipfftResult_t", c.CONV_TYPE, c.API_FFT)),
        ("cufftResult", ("hipfftResult", c.CONV_TYPE, c.API_FFT)),
        ("CUFFT_SUCCESS", ("HIPFFT_SUCCESS", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_INVALID_PLAN", ("HIPFFT_INVALID_PLAN", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_ALLOC_FAILED", ("HIPFFT_ALLOC_FAILED", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_INVALID_TYPE", ("HIPFFT_INVALID_TYPE", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        (
            "CUFFT_INVALID_VALUE",
            ("HIPFFT_INVALID_VALUE", c.CONV_NUMERIC_LITERAL, c.API_FFT),
        ),
        (
            "CUFFT_INTERNAL_ERROR",
            ("HIPFFT_INTERNAL_ERROR", c.CONV_NUMERIC_LITERAL, c.API_FFT),
        ),
        ("CUFFT_EXEC_FAILED", ("HIPFFT_EXEC_FAILED", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_SETUP_FAILED", ("HIPFFT_SETUP_FAILED", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_INVALID_SIZE", ("HIPFFT_INVALID_SIZE", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        (
            "CUFFT_UNALIGNED_DATA",
            ("HIPFFT_UNALIGNED_DATA", c.CONV_NUMERIC_LITERAL, c.API_FFT),
        ),
        (
            "CUFFT_INCOMPLETE_PARAMETER_LIST",
            ("HIPFFT_INCOMPLETE_PARAMETER_LIST", c.CONV_NUMERIC_LITERAL, c.API_FFT),
        ),
        (
            "CUFFT_INVALID_DEVICE",
            ("HIPFFT_INVALID_DEVICE", c.CONV_NUMERIC_LITERAL, c.API_FFT),
        ),
        ("CUFFT_PARSE_ERROR", ("HIPFFT_PARSE_ERROR", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_NO_WORKSPACE", ("HIPFFT_NO_WORKSPACE", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        (
            "CUFFT_NOT_IMPLEMENTED",
            ("HIPFFT_NOT_IMPLEMENTED", c.CONV_NUMERIC_LITERAL, c.API_FFT),
        ),
        (
            "CUFFT_LICENSE_ERROR",
            ("HIPFFT_LICENSE_ERROR", c.CONV_NUMERIC_LITERAL, c.API_FFT, c.HIP_UNSUPPORTED),
        ),
        (
            "CUFFT_NOT_SUPPORTED",
            ("HIPFFT_NOT_SUPPORTED", c.CONV_NUMERIC_LITERAL, c.API_FFT),
        ),
        ("cufftType_t", ("hipfftType_t", c.CONV_TYPE, c.API_FFT)),
        ("cufftType", ("hipfftType", c.CONV_TYPE, c.API_FFT)),
        ("CUFFT_R2C", ("HIPFFT_R2C", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_C2R", ("HIPFFT_C2R", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_C2C", ("HIPFFT_C2C", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_D2Z", ("HIPFFT_D2Z", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_Z2D", ("HIPFFT_Z2D", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        ("CUFFT_Z2Z", ("HIPFFT_Z2Z", c.CONV_NUMERIC_LITERAL, c.API_FFT)),
        (
            "cufftCompatibility_t",
            ("hipfftCompatibility_t", c.CONV_TYPE, c.API_FFT, c.HIP_UNSUPPORTED),
        ),
        (
            "cufftCompatibility",
            ("hipfftCompatibility", c.CONV_TYPE, c.API_FFT, c.HIP_UNSUPPORTED),
        ),
        (
            "CUFFT_COMPATIBILITY_FFTW_PADDING",
            (
                "HIPFFT_COMPATIBILITY_FFTW_PADDING",
                c.CONV_NUMERIC_LITERAL,
                c.API_FFT,
                c.HIP_UNSUPPORTED,
            ),
        ),
        ("cufftReal", ("hipfftReal", c.CONV_TYPE, c.API_FFT)),
        ("cufftDoubleReal", ("hipfftDoubleReal", c.CONV_TYPE, c.API_FFT)),
        ("cufftComplex", ("hipfftComplex", c.CONV_TYPE, c.API_FFT)),
        ("cufftDoubleComplex", ("hipfftDoubleComplex", c.CONV_TYPE, c.API_FFT)),
        ("cufftHandle", ("hipfftHandle", c.CONV_TYPE, c.API_FFT)),
        ("cufftPlan1d", ("hipfftPlan1d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftPlan2d", ("hipfftPlan2d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftPlan3d", ("hipfftPlan3d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftPlanMany", ("hipfftPlanMany", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftMakePlan1d", ("hipfftMakePlan1d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftMakePlan2d", ("hipfftMakePlan2d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftMakePlan3d", ("hipfftMakePlan3d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftMakePlanMany", ("hipfftMakePlanMany", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftMakePlanMany64", ("hipfftMakePlanMany64", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftGetSizeMany64", ("hipfftGetSizeMany64", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftEstimate1d", ("hipfftEstimate1d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftEstimate2d", ("hipfftEstimate2d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftEstimate3d", ("hipfftEstimate3d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftEstimateMany", ("hipfftEstimateMany", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftCreate", ("hipfftCreate", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftGetSize1d", ("hipfftGetSize1d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftGetSize2d", ("hipfftGetSize2d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftGetSize3d", ("hipfftGetSize3d", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftGetSizeMany", ("hipfftGetSizeMany", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftGetSize", ("hipfftGetSize", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftSetWorkArea", ("hipfftSetWorkArea", c.CONV_MATH_FUNC, c.API_FFT)),
        (
            "cufftSetAutoAllocation",
            ("hipfftSetAutoAllocation", c.CONV_MATH_FUNC, c.API_FFT),
        ),
        ("cufftExecC2C", ("hipfftExecC2C", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftExecR2C", ("hipfftExecR2C", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftExecC2R", ("hipfftExecC2R", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftExecZ2Z", ("hipfftExecZ2Z", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftExecD2Z", ("hipfftExecD2Z", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftExecZ2D", ("hipfftExecZ2D", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftSetStream", ("hipfftSetStream", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftDestroy", ("hipfftDestroy", c.CONV_MATH_FUNC, c.API_FFT)),
        ("cufftGetVersion", ("hipfftGetVersion", c.CONV_MATH_FUNC, c.API_FFT)),
        (
            "cufftGetProperty",
            ("hipfftGetProperty", c.CONV_MATH_FUNC, c.API_FFT, c.HIP_UNSUPPORTED),
        ),
        ("nvrtcResult", ("hiprtcResult", c.CONV_TYPE, c.API_RTC)),
        ("NVRTC_SUCCESS", ("HIPRTC_SUCCESS", c.CONV_TYPE, c.API_RTC)),
        (
            "NVRTC_ERROR_OUT_OF_MEMORY",
            ("HIPRTC_ERROR_OUT_OF_MEMORY", c.CONV_TYPE, c.API_RTC),
        ),
        (
            "NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
            ("HIPRTC_ERROR_PROGRAM_CREATION_FAILURE", c.CONV_TYPE, c.API_RTC),
        ),
        (
            "NVRTC_ERROR_INVALID_INPUT",
            ("HIPRTC_ERROR_INVALID_INPUT", c.CONV_TYPE, c.API_RTC),
        ),
        (
            "NVRTC_ERROR_INVALID_PROGRAM",
            ("HIPRTC_ERROR_INVALID_PROGRAM", c.CONV_TYPE, c.API_RTC),
        ),
        ("NVRTC_ERROR_COMPILATION", ("HIPRTC_ERROR_COMPILATION", c.CONV_TYPE, c.API_RTC)),
        (
            "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
            ("HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE", c.CONV_TYPE, c.API_RTC),
        ),
        (
            "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
            ("HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION", c.CONV_TYPE, c.API_RTC),
        ),
        (
            "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
            ("HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID", c.CONV_TYPE, c.API_RTC),
        ),
        (
            "NVRTC_ERROR_INTERNAL_ERROR",
            ("HIPRTC_ERROR_INTERNAL_ERROR", c.CONV_TYPE, c.API_RTC),
        ),
        ("nvrtcGetErrorString", ("hiprtcGetErrorString", c.CONV_JIT, c.API_RTC)),
        ("nvrtcVersion", ("hiprtcVersion", c.CONV_JIT, c.API_RTC)),
        ("nvrtcProgram", ("hiprtcProgram", c.CONV_TYPE, c.API_RTC)),
        ("nvrtcAddNameExpression", ("hiprtcAddNameExpression", c.CONV_JIT, c.API_RTC)),
        ("nvrtcCompileProgram", ("hiprtcCompileProgram", c.CONV_JIT, c.API_RTC)),
        ("nvrtcCreateProgram", ("hiprtcCreateProgram", c.CONV_JIT, c.API_RTC)),
        ("nvrtcDestroyProgram", ("hiprtcDestroyProgram", c.CONV_JIT, c.API_RTC)),
        ("nvrtcGetLoweredName", ("hiprtcGetLoweredName", c.CONV_JIT, c.API_RTC)),
        ("nvrtcGetProgramLog", ("hiprtcGetProgramLog", c.CONV_JIT, c.API_RTC)),
        ("nvrtcGetProgramLogSize", ("hiprtcGetProgramLogSize", c.CONV_JIT, c.API_RTC)),
        ("nvrtcGetPTX", ("hiprtcGetCode", c.CONV_JIT, c.API_RTC)),
        ("nvrtcGetPTXSize", ("hiprtcGetCodeSize", c.CONV_JIT, c.API_RTC)),
        ("thrust::cuda", ("thrust::hip", c.CONV_MATH_FUNC, c.API_BLAS)),
        # The caffe2 directory does a string match; pytorch does a word-boundary match.
        # Patterns such as 'cub::' will not match for pytorch.
        # We list all current uses of cub symbols for this reason.
        ("cub::", ("hipcub::", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::ArgMax", ("hipcub::ArgMax", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::ArgMin", ("hipcub::ArgMin", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::BLOCK_REDUCE_WARP_REDUCTIONS", ("hipcub::BLOCK_REDUCE_WARP_REDUCTIONS", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::BlockReduce", ("hipcub::BlockReduce", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::CachingDeviceAllocator", ("hipcub::CachingDeviceAllocator", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::CountingInputIterator", ("hipcub::CountingInputIterator", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::DeviceRadixSort", ("hipcub::DeviceRadixSort", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::DeviceReduce", ("hipcub::DeviceReduce", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::DeviceScan", ("hipcub::DeviceScan", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::DeviceSegmentedRadixSort", ("hipcub::DeviceSegmentedRadixSort", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::DeviceSelect", ("hipcub::DeviceSelect", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::KeyValuePair", ("hipcub::KeyValuePair", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::Max", ("hipcub::Max", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::Min", ("hipcub::Min", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::Sum", ("hipcub::Sum", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::TransformInputIterator", ("hipcub::TransformInputIterator", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("cub::WarpReduce", ("hipcub::WarpReduce", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("namespace cub", ("namespace hipcub", c.CONV_SPECIAL_FUNC, c.API_RUNTIME)),
        ("nvtxMark", ("roctxMark", c.CONV_OTHER, c.API_ROCTX)),
        ("nvtxMarkA", ("roctxMarkA", c.CONV_OTHER, c.API_ROCTX)),
        ("nvtxRangePushA", ("roctxRangePushA", c.CONV_OTHER, c.API_ROCTX)),
        ("nvtxRangePop", ("roctxRangePop", c.CONV_OTHER, c.API_ROCTX)),
    ]
)

CUDA_SPARSE_MAP = collections.OrderedDict(
    [
        ("cusparseStatus_t", ("hipsparseStatus_t", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseHandle_t", ("hipsparseHandle_t", c.CONV_MATH_FUNC, c.API_SPARSE)),
        (
            "CUSPARSE_POINTER_MODE_HOST",
            ("HIPSPARSE_POINTER_MODE_HOST", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        ("cusparseOperation_t", ("hipsparseOperation_t", c.CONV_TYPE, c.API_SPARSE)),
        (
            "cusparseCreateMatDescr",
            ("hipsparseCreateMatDescr", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        ("cusparseCreate", ("hipsparseCreate", c.CONV_MATH_FUNC, c.API_SPARSE)),
        (
            "cusparseDestroyMatDescr",
            ("hipsparseDestroyMatDescr", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        ("cusparseDestroy", ("hipsparseDestroy", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseXcoo2csr", ("hipsparseXcoo2csr", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseMatDescr_t", ("hipsparseMatDescr_t", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseScsrmm2", ("hipsparseScsrmm2", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseDcsrmm2", ("hipsparseDcsrmm2", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseScsrmm", ("hipsparseScsrmm", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseDcsrmm", ("hipsparseDcsrmm", c.CONV_MATH_FUNC, c.API_SPARSE)),
        (
            "cusparseXcsrsort_bufferSizeExt",
            ("hipsparseXcsrsort_bufferSizeExt", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        ("cusparseCreateCsrgemm2Info", ("hipsparseCreateCsrgemm2Info", c.CONV_MATH_FUNC, c.API_SPARSE)),
        (
            "cusparseDestroyCsrgemm2Info", 
            ("hipsparseDestroyCsrgemm2Info", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        ("cusparseXcsrgemm2Nnz", ("hipsparseXcsrgemm2Nnz", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseDcsrgemm2_bufferSizeExt", ("hipsparseDcsrgemm2_bufferSizeExt", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseScsrgemm2_bufferSizeExt", ("hipsparseScsrgemm2_bufferSizeExt", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseDcsrgemm2", ("hipsparseDcsrgemm2", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseScsrgemm2", ("hipsparseScsrgemm2", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseSetPointerMode", ("hipsparseSetPointerMode", c.CONV_MATH_FUNC, c.API_SPARSE)),
        ("cusparseXcsrsort", ("hipsparseXcsrsort", c.CONV_MATH_FUNC, c.API_SPARSE)),
        (
            "cusparseXcoosort_bufferSizeExt",
            ("hipsparseXcoosort_bufferSizeExt", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        (
            "cusparseXcoosortByRow",
            ("hipsparseXcoosortByRow", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        ("cusparseSetStream", ("hipsparseSetStream", c.CONV_MATH_FUNC, c.API_SPARSE)),
        (
            "cusparseCreateIdentityPermutation",
            ("hipsparseCreateIdentityPermutation", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        (
            "cusparseSetMatIndexBase",
            ("hipsparseSetMatIndexBase", c.CONV_MATH_FUNC, c.API_SPARSE),
        ),
        ("cusparseSetMatType", ("hipsparseSetMatType", c.CONV_MATH_FUNC, c.API_SPARSE)),
        (
            "CUSPARSE_STATUS_SUCCESS",
            ("HIPSPARSE_STATUS_SUCCESS", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_NOT_INITIALIZED",
            ("HIPSPARSE_STATUS_NOT_INITIALIZED", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_ALLOC_FAILED",
            ("HIPSPARSE_STATUS_ALLOC_FAILED", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_INVALID_VALUE",
            ("HIPSPARSE_STATUS_INVALID_VALUE", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_MAPPING_ERROR",
            ("HIPSPARSE_STATUS_MAPPING_ERROR", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_EXECUTION_FAILED",
            ("HIPSPARSE_STATUS_EXECUTION_FAILED", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_INTERNAL_ERROR",
            ("HIPSPARSE_STATUS_INTERNAL_ERROR", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
            (
                "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
                c.CONV_NUMERIC_LITERAL,
                c.API_SPARSE,
            ),
        ),
        (
            "CUSPARSE_STATUS_ARCH_MISMATCH",
            ("HIPSPARSE_STATUS_ARCH_MISMATCH", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_STATUS_ZERO_PIVOT",
            ("HIPSPARSE_STATUS_ZERO_PIVOT", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_OPERATION_TRANSPOSE",
            ("HIPSPARSE_OPERATION_TRANSPOSE", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_OPERATION_NON_TRANSPOSE",
            ("HIPSPARSE_OPERATION_NON_TRANSPOSE", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE",
            (
                "HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE",
                c.CONV_NUMERIC_LITERAL,
                c.API_SPARSE,
            ),
        ),
        (
            "CUSPARSE_INDEX_BASE_ZERO",
            ("HIPSPARSE_INDEX_BASE_ZERO", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_INDEX_BASE_ONE",
            ("HIPSPARSE_INDEX_BASE_ONE", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
        (
            "CUSPARSE_MATRIX_TYPE_GENERAL",
            ("HIPSPARSE_MATRIX_TYPE_GENERAL", c.CONV_NUMERIC_LITERAL, c.API_SPARSE),
        ),
    ]
)

PYTORCH_SPECIFIC_MAPPINGS = collections.OrderedDict(
    [
        ("USE_CUDA", ("USE_ROCM", c.API_PYTORCH)),
        ("CUDA_VERSION", ("HIP_VERSION", c.API_PYTORCH)),
        ("cudaHostAllocator", ("hipHostAllocator", c.API_PYTORCH)),
        ("cudaDeviceAllocator", ("hipDeviceAllocator", c.API_PYTORCH)),
        ("define MAX_NUM_BLOCKS 200", ("define MAX_NUM_BLOCKS 64", c.API_PYTORCH)),
        ("cuda::CUDAGuard", ("hip::HIPGuardMasqueradingAsCUDA", c.API_PYTORCH)),
        ("CUDAGuard", ("HIPGuardMasqueradingAsCUDA", c.API_PYTORCH)),
        (
            "cuda::OptionalCUDAGuard",
            ("hip::OptionalHIPGuardMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        ("OptionalCUDAGuard", ("OptionalHIPGuardMasqueradingAsCUDA", c.API_PYTORCH)),
        (
            "cuda::CUDAStreamGuard",
            ("hip::HIPStreamGuardMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        ("CUDAStreamGuard", ("HIPStreamGuardMasqueradingAsCUDA", c.API_PYTORCH)),
        (
            "cuda::OptionalCUDAStreamGuard",
            ("hip::OptionalHIPStreamGuardMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        (
            "OptionalCUDAStreamGuard",
            ("OptionalHIPStreamGuardMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        # Only get needs to be transformed this way; all the other ones can go
        # straight to the normal versions hip::HIPCachingAllocator
        (
            "cuda::CUDACachingAllocator::get",
            ("hip::HIPCachingAllocatorMasqueradingAsCUDA::get", c.API_PYTORCH),
        ),
        (
            "CUDACachingAllocator::get",
            ("HIPCachingAllocatorMasqueradingAsCUDA::get", c.API_PYTORCH),
        ),
        (
            "cuda::CUDACachingAllocator::recordStream",
            (
                "hip::HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA",
                c.API_PYTORCH,
            ),
        ),
        (
            "CUDACachingAllocator::recordStream",
            (
                "HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA",
                c.API_PYTORCH,
            ),
        ),
        ("cuda::CUDAStream", ("hip::HIPStreamMasqueradingAsCUDA", c.API_PYTORCH)),
        ("CUDAStream", ("HIPStreamMasqueradingAsCUDA", c.API_PYTORCH)),
        (
            "cuda::getStreamFromPool",
            ("hip::getStreamFromPoolMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        ("getStreamFromPool", ("getStreamFromPoolMasqueradingAsCUDA", c.API_PYTORCH)),
        (
            "cuda::getDefaultCUDAStream",
            ("hip::getDefaultHIPStreamMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        (
            "getDefaultCUDAStream",
            ("getDefaultHIPStreamMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        (
            "cuda::getCurrentCUDAStream",
            ("hip::getCurrentHIPStreamMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        (
            "getCurrentCUDAStream",
            ("getCurrentHIPStreamMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        (
            "cuda::setCurrentCUDAStream",
            ("hip::setCurrentHIPStreamMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        (
            "setCurrentCUDAStream",
            ("setCurrentHIPStreamMasqueradingAsCUDA", c.API_PYTORCH),
        ),
        # TODO: Undo this special-case; see the header for motivation behind this
        # hack.  It's VERY important this is only applied to PyTorch HIPify.
        (
            "c10/cuda/CUDAGuard.h",
            ("ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h", c.API_PYTORCH),
        ),
        (
            "c10/cuda/CUDACachingAllocator.h",
            ("ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h", c.API_PYTORCH),
        ),
        (
            "c10/cuda/CUDAStream.h",
            ("ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h", c.API_PYTORCH),
        ),
        ("gloo/cuda.h", ("gloo/hip.h", c.API_PYTORCH)),
        (
            "gloo/cuda_allreduce_halving_doubling.h",
            ("gloo/hip_allreduce_halving_doubling.h", c.API_PYTORCH),
        ),
        (
            "gloo/cuda_allreduce_halving_doubling_pipelined.h",
            ("gloo/hip_allreduce_halving_doubling_pipelined.h", c.API_PYTORCH),
        ),
        ("gloo/cuda_allreduce_ring.h", ("gloo/hip_allreduce_ring.h", c.API_PYTORCH)),
        (
            "gloo/cuda_broadcast_one_to_all.h",
            ("gloo/hip_broadcast_one_to_all.h", c.API_PYTORCH),
        ),
        (
            "gloo::CudaAllreduceHalvingDoublingPipelined",
            ("gloo::HipAllreduceHalvingDoublingPipelined", c.API_PYTORCH),
        ),
        ("gloo::CudaBroadcastOneToAll", ("gloo::HipBroadcastOneToAll", c.API_PYTORCH)),
        ("gloo::CudaHostWorkspace", ("gloo::HipHostWorkspace", c.API_PYTORCH)),
        ("gloo::CudaDeviceWorkspace", ("gloo::HipDeviceWorkspace", c.API_PYTORCH)),
        ("CUDNN_RNN_RELU", ("miopenRNNRELU", c.API_PYTORCH)),
        ("CUDNN_RNN_TANH", ("miopenRNNTANH", c.API_PYTORCH)),
        ("CUDNN_LSTM", ("miopenLSTM", c.API_PYTORCH)),
        ("CUDNN_GRU", ("miopenGRU", c.API_PYTORCH)),
        ("cudnnRNNMode_t", ("miopenRNNMode_t", c.API_PYTORCH)),
    ]
)

CAFFE2_SPECIFIC_MAPPINGS = collections.OrderedDict(
    [
        ("cuda_stream", ("hip_stream", c.API_CAFFE2)),
        # if the header is a native hip folder (under hip directory),
        # there is no need to add a hip path to it; the trie in hipify script
        # takes this mapping order to forbid further replacement
        ("/hip/", ("/hip/", c.API_CAFFE2)),
        ("/context_gpu", ("/hip/context_gpu", c.API_CAFFE2)),
        ("/common_gpu", ("/hip/common_gpu", c.API_CAFFE2)),
        ("/cuda_nccl_gpu", ("/hip/hip_nccl_gpu", c.API_CAFFE2)),
        ("/mixed_utils", ("/hip/mixed_utils", c.API_CAFFE2)),
        ("/operator_fallback_gpu", ("/hip/operator_fallback_gpu", c.API_CAFFE2)),
        (
            "/spatial_batch_norm_op_impl",
            ("/hip/spatial_batch_norm_op_impl", c.API_CAFFE2),
        ),
        (
            "/recurrent_network_executor_gpu",
            ("/hip/recurrent_network_executor_gpu", c.API_CAFFE2),
        ),
        (
            "/generate_proposals_op_util_nms_gpu",
            ("/hip/generate_proposals_op_util_nms_gpu", c.API_CAFFE2),
        ),
        ("/max_pool_with_index_gpu", ("/hip/max_pool_with_index_gpu", c.API_CAFFE2)),
        ("/THCCachingAllocator_gpu", ("/hip/THCCachingAllocator_gpu", c.API_CAFFE2)),
        ("/top_k_heap_selection", ("/hip/top_k_heap_selection", c.API_CAFFE2)),
        ("/top_k_radix_selection", ("/hip/top_k_radix_selection", c.API_CAFFE2)),
        ("/GpuDefs", ("/hip/GpuDefs", c.API_CAFFE2)),
        ("/GpuScanUtils", ("/hip/GpuScanUtils", c.API_CAFFE2)),
        ("/GpuBitonicSort", ("/hip/GpuBitonicSort", c.API_CAFFE2)),
        ("/math/reduce.cuh", ("/math/hip/reduce.cuh", c.API_CAFFE2)),
        ("/sgd/adagrad_fused_op_gpu.cuh", ("/sgd/hip/adagrad_fused_op_gpu.cuh", c.API_CAFFE2)),
        ("/operators/segment_reduction_op_gpu.cuh", ("/operators/hip/segment_reduction_op_gpu.cuh", c.API_CAFFE2)),
        ("/gather_op.cuh", ("/hip/gather_op.cuh", c.API_CAFFE2)),
        ("caffe2/core/common_cudnn.h", ("caffe2/core/hip/common_miopen.h", c.API_CAFFE2)),
        ("REGISTER_CUDA_OPERATOR", ("REGISTER_HIP_OPERATOR", c.API_CAFFE2)),
        ("CUDA_1D_KERNEL_LOOP", ("HIP_1D_KERNEL_LOOP", c.API_CAFFE2)),
        ("CUDAContext", ("HIPContext", c.API_CAFFE2)),
        ("CAFFE_CUDA_NUM_THREADS", ("CAFFE_HIP_NUM_THREADS", c.API_CAFFE2)),
        ("HasCudaGPU", ("HasHipGPU", c.API_CAFFE2)),
        ("__expf", ("expf", c.API_CAFFE2)),
        ("CUBLAS_ENFORCE", ("ROCBLAS_ENFORCE", c.API_CAFFE2)),
        ("CUBLAS_CHECK", ("ROCBLAS_CHECK", c.API_CAFFE2)),
        ("cublas_handle", ("rocblashandle", c.API_CAFFE2)),
        ("CURAND_ENFORCE", ("HIPRAND_ENFORCE", c.API_CAFFE2)),
        ("CURAND_CHECK", ("HIPRAND_CHECK", c.API_CAFFE2)),
        ("curandGenerateUniform", ("hiprandGenerateUniform", c.API_CAFFE2)),
        ("curand_generator", ("hiprand_generator", c.API_CAFFE2)),
        ("CaffeCudaGetDevice", ("CaffeHipGetDevice", c.API_CAFFE2)),
        # do not rename CUDA_KERNEL_ASSERT, lazyInitCUDA in caffe2 sources
        # the ordered dict guarantees this pattern will match first, before "CUDA"
        ("CUDA_KERNEL_ASSERT", ("CUDA_KERNEL_ASSERT", c.API_CAFFE2)),
        ("lazyInitCUDA", ("lazyInitCUDA", c.API_CAFFE2)),
        ("CUDA", ("HIP", c.API_CAFFE2)),
        ("Cuda", ("Hip", c.API_CAFFE2)),
        ("cuda_", ("hip_", c.API_CAFFE2)),
        ("_cuda", ("_hip", c.API_CAFFE2)),
        ("CUDNN", ("MIOPEN", c.API_CAFFE2)),
        ("CuDNN", ("MIOPEN", c.API_CAFFE2)),
        ("cudnn", ("miopen", c.API_CAFFE2)),
        ("namespace cuda", ("namespace hip", c.API_CAFFE2)),
        ("cuda::CUDAGuard", ("hip::HIPGuard", c.API_CAFFE2)),
        ("cuda::OptionalCUDAGuard", ("hip::OptionalHIPGuard", c.API_CAFFE2)),
        ("cuda::CUDAStreamGuard", ("hip::HIPStreamGuard", c.API_CAFFE2)),
        ("cuda::OptionalCUDAStreamGuard", ("hip::OptionalHIPStreamGuard", c.API_CAFFE2)),
        ("c10/cuda/CUDAGuard.h", ("c10/hip/HIPGuard.h", c.API_CAFFE2)),
        ("gloo/cuda", ("gloo/hip", c.API_CAFFE2)),
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
# put it as c.API_CAFFE2
C10_MAPPINGS = collections.OrderedDict(
    [
        ("cuda::compat::", ("hip::compat::", c.API_C10)),
        ("c10/cuda/CUDAException.h", ("c10/hip/HIPException.h", c.API_C10)),
        ("c10/cuda/CUDAMacros.h", ("c10/hip/HIPMacros.h", c.API_C10)),
        ("c10/cuda/CUDAMathCompat.h", ("c10/hip/HIPMathCompat.h", c.API_C10)),
        ("c10/cuda/CUDAFunctions.h", ("c10/hip/HIPFunctions.h", c.API_C10)),
        ("c10/cuda/CUDAStream.h", ("c10/hip/HIPStream.h", c.API_C10)),
        ("c10/cuda/CUDACachingAllocator.h", ("c10/hip/HIPCachingAllocator.h", c.API_C10)),
        ("c10/cuda/impl/CUDATest.h", ("c10/hip/impl/HIPTest.h", c.API_C10)),
        ("c10/cuda/impl/CUDAGuardImpl.h", ("c10/hip/impl/HIPGuardImpl.h", c.API_C10)),
        (
            "c10/cuda/impl/cuda_cmake_macros.h",
            ("c10/hip/impl/hip_cmake_macros.h", c.API_C10),
        ),
        ("C10_CUDA_CHECK", ("C10_HIP_CHECK", c.API_C10)),
        ("C10_CUDA_CHECK_WARN", ("C10_HIP_CHECK_WARN", c.API_C10)),
        ("c10::cuda", ("c10::hip", c.API_C10)),
        ("cuda::CUDAStream", ("hip::HIPStream", c.API_C10)),
        ("CUDAStream", ("HIPStream", c.API_C10)),
        # This substitution is not permissible, because there's another copy of this
        # function in torch/cuda.h
        # ("cuda::device_count", ("hip::device_count", c.API_C10)),
        ("cuda::current_device", ("hip::current_device", c.API_C10)),
        ("cuda::set_device", ("hip::set_device", c.API_C10)),
        ("cuda::device_synchronize", ("hip::device_synchronize", c.API_C10)),
        ("cuda::getStreamFromPool", ("hip::getStreamFromPool", c.API_C10)),
        ("getStreamFromPool", ("getStreamFromPool", c.API_C10)),
        ("cuda::getDefaultCUDAStream", ("hip::getDefaultHIPStream", c.API_C10)),
        ("getDefaultCUDAStream", ("getDefaultHIPStream", c.API_C10)),
        ("cuda::getCurrentCUDAStream", ("hip::getCurrentHIPStream", c.API_C10)),
        ("getCurrentCUDAStream", ("getCurrentHIPStream", c.API_C10)),
        ("cuda::setCurrentCUDAStream", ("hip::setCurrentHIPStream", c.API_C10)),
        ("setCurrentCUDAStream", ("setCurrentHIPStream", c.API_C10)),
        ("cuda::CUDACachingAllocator", ("hip::HIPCachingAllocator", c.API_C10)),
        ("CUDACachingAllocator", ("HIPCachingAllocator", c.API_C10)),
        ("C10_CUDA_KERNEL_LAUNCH_CHECK", ("C10_HIP_KERNEL_LAUNCH_CHECK", c.API_C10))
    ]
)

# NB: C10 mappings are more specific than Caffe2 mappings, so run them
# first
CUDA_TO_HIP_MAPPINGS = [
    CUDA_IDENTIFIER_MAP,
    CUDA_TYPE_NAME_MAP,
    CUDA_INCLUDE_MAP,
    CUDA_SPARSE_MAP,
    C10_MAPPINGS,
    PYTORCH_SPECIFIC_MAPPINGS,
    CAFFE2_SPECIFIC_MAPPINGS,
]
