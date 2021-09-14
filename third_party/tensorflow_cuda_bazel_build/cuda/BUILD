licenses([
    "restricted",
    "reciprocal",
    "notice",
])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_nvcc",
    values = {
        "define": "using_cuda_nvcc=true",
    },
)

config_setting(
    name = "using_clang",
    values = {
        "define": "using_cuda_clang=true",
    },
)

# Equivalent to using_clang && -c opt.
config_setting(
    name = "using_clang_opt",
    values = {
        "define": "using_cuda_clang=true",
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
)

cc_library(
    name = "cuda_headers",
    hdrs = [
        ":cuda-include",
        ":cudnn-include",
    ],
    includes = [
        ".",
        "include",
    ],
)

cc_library(
    name = "cudnn_headers",
    hdrs = [
        ":cudnn-include",
    ],
    includes = [
        ".",
        "include",
    ],
)

cc_library(
    name = "cudart_static",
    linkopts = [
        "-L/usr/local/cuda/lib64",
    ],
)

cc_library(
    name = "cuda_driver",
    linkopts = ["-lcuda"],
    deps = [":linker_search_path"],
)

# Provides the RPATH for Nvidia-less sytems to be able to run binaries linked to libcuda.
cc_library(
    name = "driver_stub_runtime",
    linkopts = [
        "-Wl,-rpath,/usr/local/cuda/lib64/stubs",
    ],
    deps = [":cuda_driver"],
)

cc_library(
    name = "linker_search_path",
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-L/usr/local/cuda/lib64/stubs",
        "-Wl,-rpath-link,/usr/local/cuda/lib64",
        "-Wl,-rpath-link,/usr/local/cuda/lib64/stubs",
    ],
)

[
    cc_library(
        name = libname,
        linkopts = ["-l" + libname] + (["-lgomp"] if (libname == "cusolver") else []),
        linkstatic = True,
        deps = [":linker_search_path"],
    )
    for libname in [
        "cublas",
        "cudart",
        "cudnn",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nvrtc",
        "nvToolsExt",
    ]
]

cc_library(
    name = "cuda",
    deps = [
        ":cublas",
        ":cuda_headers",
        ":cudart",
        ":cudnn",
        ":cufft",
        ":curand",
        ":nvToolsExt",
    ],
)

# NVIDIA Performance Primitives (http://docs.nvidia.com/cuda/npp/modules.html))
# used by OpenCV
cc_library(
    name = "nppi",
    linkopts = [
        "-lnppc",
        "-lnppial",
        "-lnppicom",
        "-lnppidei",
        "-lnppif",
        "-lnppig",
        "-lnppim",
        "-lnppist",
        "-lnppitc",
        "-lnpps",
    ],
    linkstatic = True,
    deps = [":linker_search_path"],
)

# NVIDIA Management Library
cc_library(
    name = "nvml",
    linkopts = [
        "-lnvidia-ml",
        "-Wl,-rpath,/usr/lib/nvidia-410",
        "-Wl,-rpath,/usr/lib/nvidia-390",
        "-Wl,-rpath,/usr/lib/nvidia-387",
        "-Wl,-rpath,/usr/lib/nvidia-384",
    ],
    deps = [":linker_search_path"],
)

cc_library(
    name = "cupti_headers",
    hdrs = [
        ":cuda-extras",
    ],
    includes = [
        ".",
        "extras/CUPTI/include/",
    ],
)

# cupti .so exposed at linktime
cc_library(
    name = "cupti_link",
    linkopts = [
        "-L/usr/local/cuda/extras/CUPTI/lib64",
        "-lcupti",
    ],
)

cc_library(
    name = "libdevice_root",
    data = [":cuda-nvvm"],
)

CUDA_INCLUDES_FILES = [
    "include/builtin_types.h",
    "include/channel_descriptor.h",
    "include/CL/cl_egl.h",
    "include/CL/cl_ext.h",
    "include/CL/cl_gl_ext.h",
    "include/CL/cl_gl.h",
    "include/CL/cl.h",
    "include/CL/cl.hpp",
    "include/CL/cl_platform.h",
    "include/CL/opencl.h",
    "include/common_functions.h",
    "include/cooperative_groups.h",
    "include/cooperative_groups_helpers.h",
    "include/crt/common_functions.h",
    "include/crt/device_double_functions.h",
    "include/crt/device_double_functions.hpp",
    "include/crt/device_functions.h",
    "include/crt/device_functions.hpp",
    "include/crt/func_macro.h",
    "include/crt/host_config.h",
    "include/crt/host_defines.h",
    "include/crt/host_runtime.h",
    "include/crt/math_functions.h",
    "include/crt/math_functions.hpp",
    "include/crt/mma.h",
    "include/crt/mma.hpp",
    "include/crt/nvfunctional",
    "include/crt/sm_70_rt.h",
    "include/crt/sm_70_rt.hpp",
    "include/crt/storage_class.h",
    # TODO: figure out why on a CI machine with CUDA 10.2 it's not present
    # "include/cublas_api.h",
    # "include/cublas.h",
    # "include/cublas_v2.h",
    # "include/cublasXt.h",
    "include/cuComplex.h",
    "include/cuda_device_runtime_api.h",
    "include/cudaEGL.h",
    "include/cuda_egl_interop.h",
    "include/cuda_fp16.h",
    "include/cuda_fp16.hpp",
    "include/cudaGL.h",
    "include/cuda_gl_interop.h",
    "include/cuda.h",
    "include/cudalibxt.h",
    "include/cuda_occupancy.h",
    "include/cuda_profiler_api.h",
    "include/cudaProfiler.h",
    "include/cudart_platform.h",
    "include/cuda_runtime_api.h",
    "include/cuda_runtime.h",
    "include/cuda_surface_types.h",
    "include/cuda_texture_types.h",
    "include/cudaVDPAU.h",
    "include/cuda_vdpau_interop.h",
    "include/cufft.h",
    "include/cufftw.h",
    "include/cufftXt.h",
    "include/curand_discrete2.h",
    "include/curand_discrete.h",
    "include/curand_globals.h",
    "include/curand.h",
    "include/curand_kernel.h",
    "include/curand_lognormal.h",
    "include/curand_mrg32k3a.h",
    "include/curand_mtgp32dc_p_11213.h",
    "include/curand_mtgp32.h",
    "include/curand_mtgp32_host.h",
    "include/curand_mtgp32_kernel.h",
    "include/curand_normal.h",
    "include/curand_normal_static.h",
    "include/curand_philox4x32_x.h",
    "include/curand_poisson.h",
    "include/curand_precalc.h",
    "include/curand_uniform.h",
    "include/cusolver_common.h",
    "include/cusolverDn.h",
    "include/cusolverRf.h",
    "include/cusolverSp.h",
    "include/cusolverSp_LOWLEVEL_PREVIEW.h",
    "include/cusparse.h",
    "include/cusparse_v2.h",
    "include/device_atomic_functions.h",
    "include/device_atomic_functions.hpp",
    "include/device_double_functions.h",
    "include/device_functions.h",
    "include/device_launch_parameters.h",
    "include/device_types.h",
    "include/driver_functions.h",
    "include/driver_types.h",
    "include/fatBinaryCtl.h",
    "include/fatbinary.h",
    "include/host_config.h",
    "include/host_defines.h",
    "include/library_types.h",
    "include/math_constants.h",
    "include/math_functions.h",
    "include/mma.h",
    "include/nppcore.h",
    "include/nppdefs.h",
    "include/npp.h",
    "include/nppi_arithmetic_and_logical_operations.h",
    "include/nppi_color_conversion.h",
    "include/nppi_compression_functions.h",
    "include/nppi_computer_vision.h",
    "include/nppi_data_exchange_and_initialization.h",
    "include/nppi_filtering_functions.h",
    "include/nppi_geometry_transforms.h",
    "include/nppi.h",
    "include/nppi_linear_transforms.h",
    "include/nppi_morphological_operations.h",
    "include/nppi_statistics_functions.h",
    "include/nppi_support_functions.h",
    "include/nppi_threshold_and_compare_operations.h",
    "include/npps_arithmetic_and_logical_operations.h",
    "include/npps_conversion_functions.h",
    "include/npps_filtering_functions.h",
    "include/npps.h",
    "include/npps_initialization.h",
    "include/npps_statistics_functions.h",
    "include/npps_support_functions.h",
    # Note: CUDA 10.0 only
    # "include/nppversion.h",
    # TODO: figure out why on a CI machine with CUDA 10.2 it's not present
    # "include/nvblas.h",
    "include/nvfunctional",
    "include/nvgraph.h",
    "include/nvjpeg.h",
    "include/nvml.h",
    "include/nvrtc.h",
    "include/nvToolsExtCuda.h",
    "include/nvToolsExtCudaRt.h",
    "include/nvToolsExt.h",
    "include/nvToolsExtMeta.h",
    "include/nvToolsExtSync.h",
    "include/nvtx3/nvToolsExtCuda.h",
    "include/nvtx3/nvToolsExtCudaRt.h",
    "include/nvtx3/nvToolsExt.h",
    "include/nvtx3/nvToolsExtOpenCL.h",
    "include/nvtx3/nvToolsExtSync.h",
    "include/nvtx3/nvtxDetail/nvtxImplCore.h",
    "include/nvtx3/nvtxDetail/nvtxImplCudaRt_v3.h",
    "include/nvtx3/nvtxDetail/nvtxImplCuda_v3.h",
    "include/nvtx3/nvtxDetail/nvtxImpl.h",
    "include/nvtx3/nvtxDetail/nvtxImplOpenCL_v3.h",
    "include/nvtx3/nvtxDetail/nvtxImplSync_v3.h",
    "include/nvtx3/nvtxDetail/nvtxInitDecls.h",
    "include/nvtx3/nvtxDetail/nvtxInitDefs.h",
    "include/nvtx3/nvtxDetail/nvtxInit.h",
    "include/nvtx3/nvtxDetail/nvtxLinkOnce.h",
    "include/nvtx3/nvtxDetail/nvtxTypes.h",
    "include/sm_20_atomic_functions.h",
    "include/sm_20_atomic_functions.hpp",
    "include/sm_20_intrinsics.h",
    "include/sm_20_intrinsics.hpp",
    "include/sm_30_intrinsics.h",
    "include/sm_30_intrinsics.hpp",
    "include/sm_32_atomic_functions.h",
    "include/sm_32_atomic_functions.hpp",
    "include/sm_32_intrinsics.h",
    "include/sm_32_intrinsics.hpp",
    "include/sm_35_atomic_functions.h",
    "include/sm_35_intrinsics.h",
    "include/sm_60_atomic_functions.h",
    "include/sm_60_atomic_functions.hpp",
    "include/sm_61_intrinsics.h",
    "include/sm_61_intrinsics.hpp",
    # CUDA 10.0 only
    # "include/sobol_direction_vectors.h",
    "include/surface_functions.h",
    "include/surface_functions.hpp",
    "include/surface_indirect_functions.h",
    "include/surface_indirect_functions.hpp",
    "include/surface_types.h",
    "include/texture_fetch_functions.h",
    "include/texture_fetch_functions.hpp",
    "include/texture_indirect_functions.h",
    "include/texture_indirect_functions.hpp",
    "include/texture_types.h",
    "include/vector_functions.h",
    "include/vector_functions.hpp",
    "include/vector_types.h",
]

genrule(
    name = "cuda-include",
    outs = CUDA_INCLUDES_FILES,
    cmd = " && ".join([
        "ln -s /usr/local/cuda/{relpath} $(@D)/{relpath}".format(relpath = p)
        for p in CUDA_INCLUDES_FILES
    ]),
    local = True,
    tags = ["no-cache"],
)

CUDA_NVVM_FILES = [
    "nvvm/bin/cicc",
    "nvvm/include/nvvm.h",
    "nvvm/lib64/libnvvm.so",
    "nvvm/lib64/libnvvm.so.3",
    "nvvm/lib64/libnvvm.so.3.3.0",
    "nvvm/libdevice/libdevice.10.bc",
]

genrule(
    name = "cuda-nvvm",
    outs = CUDA_NVVM_FILES,
    cmd = " && ".join([
        "ln -s /usr/local/cuda/{relpath} $(@D)/{relpath}".format(relpath = p)
        for p in CUDA_NVVM_FILES
    ]),
    local = True,
    tags = ["no-cache"],
)

CUDA_EXTRAS_FILES = [
    "extras/CUPTI/include/cuda_stdint.h",
    "extras/CUPTI/include/cupti.h",
    "extras/CUPTI/include/cupti_activity.h",
    "extras/CUPTI/include/cupti_callbacks.h",
    "extras/CUPTI/include/cupti_driver_cbid.h",
    "extras/CUPTI/include/cupti_events.h",
    "extras/CUPTI/include/cupti_metrics.h",
    "extras/CUPTI/include/cupti_nvtx_cbid.h",
    "extras/CUPTI/include/cupti_result.h",
    "extras/CUPTI/include/cupti_runtime_cbid.h",
    "extras/CUPTI/include/cupti_version.h",
    "extras/CUPTI/include/generated_cuda_gl_interop_meta.h",
    "extras/CUPTI/include/generated_cuda_meta.h",
    "extras/CUPTI/include/generated_cuda_runtime_api_meta.h",
    "extras/CUPTI/include/generated_cuda_vdpau_interop_meta.h",
    "extras/CUPTI/include/generated_cudaGL_meta.h",
    "extras/CUPTI/include/generated_cudaVDPAU_meta.h",
    "extras/CUPTI/include/generated_nvtx_meta.h",
    "extras/CUPTI/include/GL/gl.h",
    "extras/CUPTI/include/GL/glew.h",
    "extras/CUPTI/include/GL/glext.h",
    "extras/CUPTI/include/GL/glu.h",
    "extras/CUPTI/include/GL/glut.h",
    "extras/CUPTI/include/GL/glx.h",
    "extras/CUPTI/include/GL/glxext.h",
    "extras/CUPTI/include/GL/wglew.h",
    "extras/CUPTI/include/GL/wglext.h",
    "extras/CUPTI/include/openacc/cupti_openacc.h",
]

genrule(
    name = "cuda-extras",
    outs = CUDA_EXTRAS_FILES,
    cmd = " && ".join([
        "ln -s /usr/local/cuda/{relpath} $(@D)/{relpath}".format(relpath = p)
        for p in CUDA_EXTRAS_FILES
    ]),
    local = True,
    tags = ["no-cache"],
)

genrule(
    name = "cudnn-include",
    outs = [
        "include/cudnn.h",
    ],
    cmd = """
        ln -s /usr/include/cudnn.h $(@D)/cudnn.h""",
    local = True,
    tags = ["no-cache"],
)

