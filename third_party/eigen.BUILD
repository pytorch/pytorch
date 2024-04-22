# This is BUILD file is derived from https://github.com/tensorflow/tensorflow/blob/master/third_party/eigen.BUILD

# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.

load("@rules_cc//cc:defs.bzl", "cc_library")

licenses([
    # Note: Eigen is an MPL2 library that includes GPL v3 and LGPL v2.1+ code.
    #       We've taken special care to not reference any restricted code.
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["COPYING.MPL2"])

# License-restricted (i.e. not reciprocal or notice) files inside Eigen/...
EIGEN_RESTRICTED_FILES = [
    "Eigen/src/OrderingMethods/Amd.h",
    "Eigen/src/SparseCholesky/**",
]

# Notable transitive dependencies of restricted files inside Eigen/...
EIGEN_RESTRICTED_DEPS = [
    "Eigen/Eigen",
    "Eigen/IterativeLinearSolvers",
    "Eigen/MetisSupport",
    "Eigen/Sparse",
    "Eigen/SparseCholesky",
    "Eigen/SparseLU",
]

EIGEN_FILES = [
    "Eigen/**",
    "unsupported/Eigen/CXX11/**",
    "unsupported/Eigen/FFT",
    "unsupported/Eigen/KroneckerProduct",
    "unsupported/Eigen/src/FFT/**",
    "unsupported/Eigen/src/KroneckerProduct/**",
    "unsupported/Eigen/MatrixFunctions",
    "unsupported/Eigen/SpecialFunctions",
    "unsupported/Eigen/Splines",
    "unsupported/Eigen/src/MatrixFunctions/**",
    "unsupported/Eigen/src/SpecialFunctions/**",
    "unsupported/Eigen/src/Splines/**",
    "unsupported/Eigen/NonLinearOptimization",
    "unsupported/Eigen/NumericalDiff",
    "unsupported/Eigen/src/**",
    "unsupported/Eigen/Polynomials",
]

# List of files picked up by glob but actually part of another target.
EIGEN_EXCLUDE_FILES = ["Eigen/src/Core/arch/AVX/PacketMathGoogleTest.cc"]

# Disallowed eigen modules/files in rNA:
# * Using the custom STL and memory support, it is not needed and should
#   not be used with c++17.
# * We will only support the EulerAnglesZYX provided by //atg/geometry so
#   just don't allow people to access the unsupported eigen module.
EIGEN_DISALLOW_FILES = [
    "Eigen/StlSupport/*.h",
    "unsupported/Eigen/EulerAngles",
    "unsupported/Eigen/src/EulerAngles/**",
]

# Files known to be under MPL2 license.
EIGEN_MPL2_HEADER_FILES = glob(
    EIGEN_FILES,
    exclude = EIGEN_EXCLUDE_FILES +
              EIGEN_RESTRICTED_FILES +
              EIGEN_DISALLOW_FILES +
              EIGEN_RESTRICTED_DEPS + [
        # Guarantees any file missed by excludes above will not compile.
        "Eigen/src/Core/util/NonMPL2.h",
        "Eigen/**/CMakeLists.txt",
    ],
)

cc_library(
    name = "eigen",
    hdrs = EIGEN_MPL2_HEADER_FILES,
    defines = [
        # This define (mostly) guarantees we don't link any problematic
        # code. We use it, but we do not rely on it, as evidenced above.
        "EIGEN_MPL2_ONLY",
        "EIGEN_MAX_ALIGN_BYTES=64",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
