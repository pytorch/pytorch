# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# WARNING: the contents of this file must be BOTH valid Starlark (so Buck can
# load() the lists below) AND valid Python (so the CMake build can read them by
# exec()-ing this file; see get_filelist() in CMakeLists.txt). That means no
# load() directives and no glob() calls here -- neither is valid Python. Keep
# every source list an explicit list of string literals. Headers for the Buck
# targets are globbed in BUCK itself, which is Starlark-only; only the curated
# public-header install surface lives here.

# Source lists are decomposed into fine-grained "atoms" so both build systems
# draw from one source of truth: the Buck targets in BUCK compose the atoms they
# need, and the coarse per-backend accessors below (which CMake calls) compose
# the same atoms. A file appears in exactly one atom, so a rename on GitHub is a
# single edit here that flows to every build.

# --- Core (backend-independent) source atoms ---
THREAD_UTIL_SRCS = [
    "src/ThreadUtil.cpp",
]

LIBKINETO_API_SRCS = [
    "src/libkineto_api.cpp",
]

LOGGER_SRCS = [
    "src/ILoggerObserver.cpp",
    "src/Logger.cpp",
]

LOGGING_API_SRCS = [
    "src/LoggingAPI.cpp",
]

CONFIG_SRCS = [
    "src/AbstractConfig.cpp",
    "src/Config.cpp",
]

ACTIVITY_TYPE_SRCS = [
    "src/ActivityType.cpp",
]

CONFIG_LOADER_SRCS = [
    "src/ConfigLoader.cpp",
    "src/DaemonConfigLoader.cpp",
]

CONFIG_CLIENT_SRCS = [
    "src/IpcFabricConfigClient.cpp",
]

DEVICE_FUNCTIONS_SRCS = [
    "src/DeviceProperties.cpp",
    "src/DeviceUtil.cpp",
]

BASE_LOGGER_SRCS = [
    "src/GenericTraceActivity.cpp",
]

DEMANGLE_SRCS = [
    "src/Demangle.cpp",
]

APPROXIMATE_CLOCK_SRCS = [
    "src/ApproximateClock.cpp",
]

KERNEL_REGISTRY_SRCS = [
    "src/KernelRegistry.cpp",
]

OUTPUT_JSON_SRCS = [
    "src/output_json.cpp",
]

ACTIVITY_PROFILER_CORE_SRCS = [
    "src/ActivityProfilerController.cpp",
    "src/ActivityProfilerProxy.cpp",
    "src/GenericActivityProfiler.cpp",
    "src/AsyncActivityProfilerHandler.cpp",
    "src/SyncActivityProfilerHandler.cpp",
]

INIT_SRCS = [
    "src/init.cpp",
]

# --- CUPTI (CUDA) source atoms ---
CUPTI_API_SRCS = [
    "src/CuptiActivityApi.cpp",
    "src/CuptiCallbackApi.cpp",
    "src/CuptiCbidRegistry.cpp",
    "src/cupti_strings.cpp",
]

CUPTI_ACTIVITY_PROFILER_SRCS = [
    "src/CuptiActivityProfiler.cpp",
    "src/CuptiTimestamp.cpp",
]

CUPTI_PM_SAMPLING_API_SRCS = [
    "src/CuptiPMSamplingApi.cpp",
]

CUPTI_PM_SAMPLING_PROFILER_SRCS = [
    "src/CuptiPMSamplingController.cpp",
    "src/CuptiPMSamplingProfiler.cpp",
]

WEAK_SYMBOLS_SRCS = [
    "src/WeakSymbols.cpp",
]

# --- ROCm source atoms ---
ROCM_API_SRCS = [
    "src/RocLogger.cpp",
    "src/RocprofActivityApi.cpp",
    "src/RocprofLogger.cpp",
]

ROCM_ACTIVITY_PROFILER_SRCS = [
    "src/RocmActivityProfiler.cpp",
]

# --- XPU (xpupti) source atoms ---
XPUPTI_SRCS = [
    "src/plugin/xpupti/XpuptiActivityApi.cpp",
    "src/plugin/xpupti/XpuptiActivityHandlers.cpp",
    "src/plugin/xpupti/XpuptiActivityProfiler.cpp",
    "src/plugin/xpupti/XpuptiActivityProfilerSession.cpp",
    "src/plugin/xpupti/XpuptiProfilerMacros.cpp",
    "src/plugin/xpupti/XpuptiScopeProfilerApi.cpp",
    "src/plugin/xpupti/XpuptiScopeProfilerConfig.cpp",
    "src/plugin/xpupti/XpuptiScopeProfilerHandlers.cpp",
    "src/plugin/xpupti/XpuptiScopeProfilerSession.cpp",
]

# --- Coarse per-backend accessors (composed from the atoms above) ---
# CMake calls these; Buck targets compose the atoms directly.

def get_libkineto_api_srcs():
    return THREAD_UTIL_SRCS + LIBKINETO_API_SRCS

def get_libkineto_cpu_only_srcs(with_api = True):
    return (
        CONFIG_SRCS
        + ACTIVITY_TYPE_SRCS
        + CONFIG_LOADER_SRCS
        + CONFIG_CLIENT_SRCS
        + LOGGER_SRCS
        + LOGGING_API_SRCS
        + DEMANGLE_SRCS
        + DEVICE_FUNCTIONS_SRCS
        + BASE_LOGGER_SRCS
        + APPROXIMATE_CLOCK_SRCS
        + ACTIVITY_PROFILER_CORE_SRCS
        + INIT_SRCS
        + OUTPUT_JSON_SRCS
        + (get_libkineto_api_srcs() if with_api else [])
    )

def get_libkineto_cupti_srcs(with_api = True):
    return CUPTI_API_SRCS + CUPTI_ACTIVITY_PROFILER_SRCS + KERNEL_REGISTRY_SRCS + WEAK_SYMBOLS_SRCS + get_libkineto_cpu_only_srcs(with_api)

def get_libkineto_cupti_pm_sampling_srcs():
    return CUPTI_PM_SAMPLING_API_SRCS + CUPTI_PM_SAMPLING_PROFILER_SRCS

def get_libkineto_rocprofiler_srcs(with_api = True):
    return ROCM_ACTIVITY_PROFILER_SRCS + ROCM_API_SRCS + get_libkineto_cpu_only_srcs(with_api)

def get_libkineto_xpupti_srcs(with_api = True):
    return XPUPTI_SRCS + get_libkineto_cpu_only_srcs(with_api)

def get_libkineto_public_headers():
    return [
        "include/AbstractConfig.h",
        "include/ActivityProfilerInterface.h",
        "include/ActivityTraceInterface.h",
        "include/ActivityType.h",
        "include/Config.h",
        "include/ClientInterface.h",
        "include/GenericTraceActivity.h",
        "include/IActivityProfiler.h",
        "include/ILoggerObserver.h",
        "include/ITraceActivity.h",
        "include/LoggingAPI.h",
        "include/MetadataFieldCatalog.h",
        "include/TraceSpan.h",
        "include/ThreadUtil.h",
        "include/TypedMetadata.h",
        "include/TypedMetadataJson.h",
        "include/libkineto.h",
        "include/time_since_epoch.h",
        "include/output_base.h",
    ]

# kineto code should be updated to not have to
# suppress these warnings.
KINETO_COMPILER_FLAGS = [
    "-fexceptions",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-private-field",
]
