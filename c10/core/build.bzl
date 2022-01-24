def define_targets(rules):
    rules.cc_library(
        name = "CPUAllocator",
        srcs = ["CPUAllocator.cpp"],
        hdrs = ["CPUAllocator.h"],
        # This library defines a flag, The use of alwayslink keeps it
        # from being stripped.
        alwayslink = True,
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":alignment",
            ":base",
            "//c10/mobile:CPUCachingAllocator",
            "//c10/mobile:CPUProfilingAllocator",
            "//c10/util:base",
        ],
    )

    rules.cc_library(
        name = "ScalarType",
        hdrs = ["ScalarType.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            "//c10/util:min",
            "//c10/util:types",
        ],
    )

    rules.cc_library(
        name = "alignment",
        hdrs = ["alignment.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cc_library(
        name = "alloc_cpu",
        srcs = ["impl/alloc_cpu.cpp"],
        hdrs = ["impl/alloc_cpu.h"],
        # This library defines flags, The use of alwayslink keeps them
        # from being stripped.
        alwayslink = True,
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":alignment",
            "//c10/macros",
            "//c10/util:base",
        ],
    )

    rules.cc_library(
        name = "base",
        srcs = [
            "GeneratorImpl.cpp",
            "Scalar.cpp",
            "thread_pool.cpp",
        ],
        hdrs = [
            "CompileTimeFunctionPointer.h",
            "DefaultTensorOptions.h",
            "DeviceArray.h",
            "DeviceGuard.h",
            "Event.h",
            "GeneratorImpl.h",
            "OptionalRef.h",
            "QEngine.h",
            "QScheme.h",
            "Scalar.h",
            "StreamGuard.h",
            "alignment.h",
            "impl/FakeGuardImpl.h",
            "impl/InlineDeviceGuard.h",
            "impl/InlineEvent.h",
            "impl/InlineStreamGuard.h",
            "thread_pool.h",
        ],
        # This library uses flags and registration. Do not let the
        # linker remove them.
        alwayslink = True,
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":ScalarType",
            ":min",
            "//c10/macros",
            "//c10/util:TypeCast",
            "//c10/util:base",
            "//c10/util:intrusive_ptr",
            "//c10/util:python_stub",
            "//c10/util:typeid",
        ],
    )

    rules.cc_library(
        name = "min",
        srcs = [
            "Allocator.cpp",
            "AutogradState.cpp",
            "CopyBytes.cpp",
            "DefaultDtype.cpp",
            "Device.cpp",
            "DeviceType.cpp",
            "DispatchKey.cpp",
            "DispatchKeySet.cpp",
            "GradMode.cpp",
            "InferenceMode.cpp",
            "Storage.cpp",
            "StorageImpl.cpp",
            "Stream.cpp",
            "TensorImpl.cpp",
            "TensorOptions.cpp",
            "UndefinedTensorImpl.cpp",
            "impl/DeviceGuardImplInterface.cpp",
            "impl/LocalDispatchKeySet.cpp",
            "impl/SizesAndStrides.cpp",
        ],
        hdrs = [
            "Allocator.h", # util/threadlocaldebuginfo
            "AutogradState.h",
            "Backend.h",
            "CopyBytes.h",
            "DefaultDtype.h",
            "Device.h",
            "DeviceType.h",
            "DispatchKey.h",
            "DispatchKeySet.h",
            "GradMode.h",
            "InferenceMode.h",
            "Layout.h",
            "MemoryFormat.h",
            "ScalarTypeToTypeMeta.h",
            "Storage.h",
            "StorageImpl.h", # util/intrusive_ptr
            "Stream.h",
            "TensorImpl.h",
            "TensorOptions.h",
            "UndefinedTensorImpl.h",
            "WrapDimMinimal.h",
            "impl/DeviceGuardImplInterface.h",
            "impl/LocalDispatchKeySet.h",
            "impl/SizesAndStrides.h",
            "impl/VirtualGuardImpl.h",
        ],
        # This library uses flags and registration. Do not let the
        # linker remove them.
        alwayslink = True,
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":ScalarType",
            "//c10/macros",
            "//c10/util:ThreadLocalDebugInfo",
            "//c10/util:UniqueVoidPtr",
            "//c10/util:accumulate",
            "//c10/util:intrusive_ptr",
            "//c10/util:llvmMathExtras",
            "//c10/util:python_stub",
            "//c10/util:typeid",
        ],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
                "alignment.h",
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )
