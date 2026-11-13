#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <type_traits>

namespace at::hip::detail::ck_tile_group_gemm {

// True if a ck_tile pipeline problem has separate A/B compute types, i.e. the CK being compiled
// against has the 13-parameter UniversalGemmPipelineProblem (rocm-libraries#6978). Older CK has a
// single ComputeDataType_ and 12 parameters.
template <typename Problem, typename = void>
struct has_split_compute_types : std::false_type {};

template <typename Problem>
struct has_split_compute_types<Problem, std::void_t<typename Problem::BComputeDataType>>
    : std::true_type {};

template <typename Problem>
inline constexpr bool has_split_compute_types_v = has_split_compute_types<Problem>::value;

using ck_tile::index_t;
using Scheduler = ck_tile::GemmPipelineScheduler;
using PassThrough = ck_tile::element_wise::PassThrough;

// One overload per UniversalGemmPipelineProblem parameter list, selected by has_split_compute_types.
// CK's template is only named through the template template parameter Problem, so the overload whose
// parameter list does not match the CK in use is discarded instead of failing to compile.
template <template <typename, typename, typename, typename, typename, Scheduler,
                    typename, typename, typename, typename, bool, index_t, index_t> class Problem,
          typename T, typename Acc, typename Shape, typename Traits, bool Fixed, index_t Vec>
auto make_pipeline_problem(std::true_type) -> Problem<T, T, Acc, Shape, Traits, Scheduler::Intrawave,
    PassThrough, PassThrough, T, T, Fixed, Vec, Vec>;

template <template <typename, typename, typename, typename, typename, Scheduler,
                    typename, typename, typename, bool, index_t, index_t> class Problem,
          typename T, typename Acc, typename Shape, typename Traits, bool Fixed, index_t Vec>
auto make_pipeline_problem(std::false_type) -> Problem<T, T, Acc, Shape, Traits, Scheduler::Intrawave,
    PassThrough, PassThrough, T, Fixed, Vec, Vec>;

// UniversalGemmPipelineProblem for data type T with the given vector-size settings, on either CK version.
template <typename T, typename Acc, typename Shape, typename Traits, bool Fixed, index_t Vec>
using PipelineProblem = decltype(make_pipeline_problem<ck_tile::UniversalGemmPipelineProblem,
                                                       T, Acc, Shape, Traits, Fixed, Vec>(
    std::bool_constant<has_split_compute_types_v<
        ck_tile::UniversalGemmPipelineProblem<T, T, Acc, Shape, Traits>>>{}));

} // namespace at::hip::detail::ck_tile_group_gemm
