#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

#include "ATen/cuda/detail/TensorInfo.cuh"

/*
Tests related to tensor indexing and applying operations. 
*/
#ifndef _WIN32

CATCH_TEST_CASE("2D Contiguous", "Collapses a 2D contiguous tensor to 1D contiguous") {
    int sizes[] = {4, 4};
    int strides[] = {4, 1};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
    ti.collapseDims();
    CATCH_REQUIRE(ti.dims == 1);
    CATCH_REQUIRE(ti.sizes[0] == (4 * 4));
}

CATCH_TEST_CASE("3D Contiguous", "Collapses a 3D contiguous tensor to a 1D contiguous") {
    int sizes[] = {6, 3, 7};
    int strides[] = {3 * 7, 7, 1};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
    ti.collapseDims();
    CATCH_REQUIRE(ti.dims == 1);
    CATCH_REQUIRE(ti.sizes[0] == (6 * 3 * 7));
}

CATCH_TEST_CASE("3D Partial Collapse", "Collapses a 3D noncontiguous tensor to a 2D tensor") {
    int sizes[] = {4, 3, 2};
    int strides[] = {3 * 3, 3, 1};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
    ti.collapseDims();
    CATCH_REQUIRE(ti.dims == 2);
    CATCH_REQUIRE(ti.sizes[0] == (4 * 3));
    CATCH_REQUIRE(ti.sizes[1] == 2);
}

CATCH_TEST_CASE("2D Strided Collapse", "Collapses a 2D skip contiguous tensor to a 1D skip contiguous tensor") {
    int sizes[] = {3, 2};
    int strides[] = {2 * 2, 2};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
    ti.collapseDims();
    CATCH_REQUIRE(ti.dims == 1);
    CATCH_REQUIRE(ti.sizes[0] == (3 * 2));
    CATCH_REQUIRE(ti.strides[0] == 2);
}

CATCH_TEST_CASE("4D Partial Strided Collapse", "Collapses a 4D tensor to a 2D tensor"){
    int sizes[] = {3, 6, 5, 2};
    int strides[] = {6 * 22, 22, 2 * 2, 2};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
    ti.collapseDims();
    CATCH_REQUIRE(ti.dims == 2);
    CATCH_REQUIRE(ti.sizes[0] == (3 * 6));
    CATCH_REQUIRE(ti.strides[0] == 22);
    CATCH_REQUIRE(ti.sizes[1] == (5 * 2));
    CATCH_REQUIRE(ti.strides[1] == 2);
}

CATCH_TEST_CASE("Collapsing Zeros and Ones", "Collapses a 5D tensor to a 1D tensor") {
    int sizes[] = {1, 10, 1, 5, 4};
    int strides[] = {4, 0, 16, 0, 1};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 5, sizes, strides};
    ti.collapseDims();
    CATCH_REQUIRE(ti.dims == 2);
    CATCH_REQUIRE(ti.sizes[0] == (10 * 5));
    CATCH_REQUIRE(ti.strides[0] == 0);
    CATCH_REQUIRE(ti.sizes[1] == 4);
    CATCH_REQUIRE(ti.strides[1] == 1);
}

CATCH_TEST_CASE("Collapsing to a Point Tensor", "Collapses a 3D tensor to a point tensor") {
    int sizes[] = {1, 1, 1};
    int strides[] = {17, 12, 3};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
    CATCH_REQUIRE(ti.collapseDims() == 0);
    CATCH_REQUIRE(ti.dims == 1);
    CATCH_REQUIRE(ti.sizes[0] == 1);
    CATCH_REQUIRE(ti.strides[0] == 1);
}

CATCH_TEST_CASE("Excluding in a 4D Contiguous", "Collapses a 4D tensor to a 3D tensor") {
    int sizes[] = {3, 6, 5, 2};
    int strides[] = {6 * 22, 22, 2 * 2, 2};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
    CATCH_REQUIRE(ti.collapseDims(1) == 1);
    CATCH_REQUIRE(ti.dims == 3);
    CATCH_REQUIRE(ti.sizes[0] == 3);
    CATCH_REQUIRE(ti.strides[0] == (6 * 22));
    CATCH_REQUIRE(ti.sizes[1] == 6);
    CATCH_REQUIRE(ti.strides[1] == 22);
    CATCH_REQUIRE(ti.sizes[2] == (5 * 2));
    CATCH_REQUIRE(ti.strides[2] == 2);
}

CATCH_TEST_CASE("Roving Exclusion", "Collapses a 4D tensor to a 3D tensor") {
    int sizes[] = {3, 6, 5, 2};
    int strides[] = {6 * 22, 22, 2 * 2, 2};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
    CATCH_REQUIRE(ti.collapseDims(2) == 1);
    CATCH_REQUIRE(ti.dims == 3);
    CATCH_REQUIRE(ti.sizes[0] == (3 * 6));
    CATCH_REQUIRE(ti.strides[0] == 22);
    CATCH_REQUIRE(ti.sizes[1] == 5);
    CATCH_REQUIRE(ti.strides[1] == 4);
    CATCH_REQUIRE(ti.sizes[2] == 2);
    CATCH_REQUIRE(ti.strides[2] == 2);
}

CATCH_TEST_CASE("Invalid Exclusion", "Attempts to exclude a nonexisting dimension") {
    int sizes[] = {1, 1, 1};
    int strides[] = {17, 12, 3};
    ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
    _CATCH_REQUIRE_THROWS(ti.collapseDims(5));
} 

#endif
