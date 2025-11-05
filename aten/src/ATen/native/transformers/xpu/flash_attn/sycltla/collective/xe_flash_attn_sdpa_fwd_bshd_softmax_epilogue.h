#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include <sycl/sycl.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace flash_attention {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <bool CausalMask_, class DispatchPolicy, class... Args>
class FlashPrefillSoftmaxEpilogue {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>,
                "Could not find an epilogue specialization.");
};

template <bool CausalMask_, class Element_>
class FlashPrefillSoftmaxEpilogue<CausalMask_, epilogue::IntelXeXMX16,
                                  Element_> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = epilogue::IntelXeXMX16;
  using Element = Element_;

  static constexpr bool CausalMask = CausalMask_;

  using GmemTiledCopyOut = void;

  // Host side epilogue arguments
  struct Arguments {
    Element const scale;
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  static constexpr Params to_underlying_arguments(Arguments const &args) {
    constexpr double kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
    Element val = args.scale * static_cast<Element>(kLog2e);
    return Params{val};
  }

  template <class ProblemShape> static size_t get_workspace_size() { return 0; }

  template <class ProblemShape> static cutlass::Status initialize_workspace() {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement() {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FlashPrefillSoftmaxEpilogue(Params const &params_) : params(params_) {}

  template <int Vec, int FragsM, int FragsN, class FragAcc, class FragMax,
            class FragSum>
  CUTLASS_DEVICE void scale_exp_log2(FragAcc &frag_s, FragMax const &max,
                                     FragSum &sum) {
    auto g = compat::get_nd_item<1>().get_sub_group();
    const auto max_scale = max * params.scale;
    CUTLASS_PRAGMA_UNROLL
    for (int indx = 0; indx < Vec * FragsM; indx++) {
      const auto max_scale_bcast = group_broadcast(g, max_scale, indx);
      CUTLASS_PRAGMA_UNROLL
      for (int z = 0; z < FragsN; z++) {
        auto base_indx = indx + (z * Vec * FragsM);
        Element eq = frag_s(base_indx) - max_scale_bcast;
        frag_s(base_indx) = sycl::native::exp2(eq);
        sum(indx) += frag_s(base_indx);
      }
    }
  }

  template <int Vec, int FragsM, int FragsN, class FragSrc, class FragMax>
  CUTLASS_DEVICE void reduce_max(FragSrc &src, FragMax &max) {
    auto g = compat::get_nd_item<1>().get_sub_group();
    CUTLASS_PRAGMA_UNROLL
    for (int indx = 0; indx < Vec * FragsM; indx++) {
      auto maxptr = group_broadcast(g, max, indx);
      CUTLASS_PRAGMA_UNROLL
      for (int z = 0; z < FragsN; z++) {
        auto base_indx = indx + (z * Vec * FragsM);
        maxptr = sycl::max(maxptr, src(base_indx));
        src(base_indx) *= params.scale;
      }
      maxptr = reduce_over_group(g, maxptr, sycl::maximum<>());
      if (indx == g.get_local_id()[0]) {
        max = maxptr;
      }
    }
  }

  template <class FragAcc, class FragMax, class FragSum, class FragOut>
  CUTLASS_DEVICE void operator()(bool is_first, FragAcc &frag_s, FragMax &max,
                                 FragSum &sum, FragOut &out) {
    auto max_prev = max;
    using FragAccLayout = typename FragAcc::layout_type;
    using FragOutLayout = typename FragOut::layout_type;
    constexpr int Vec = get<0>(FragAccLayout{}.shape());
    constexpr int FragsM = get<1>(FragAccLayout{}.shape());
    constexpr int FragsNAcc = get<2>(FragAccLayout{}.shape());
    constexpr int FragsNOut = size(select<2,3>(FragOutLayout{}.shape()));
    reduce_max<Vec, FragsM, FragsNAcc>(frag_s, max);
    static_assert(Vec * FragsM % 8 == 0,
                  " No. of attention rows per subgroup should be >= 1 MMA Atom "
                  "worth of rows.");
    if (!is_first) {
      auto g = compat::get_nd_item<1>().get_sub_group();
      Element max_scale{max * params.scale};
      Element exp_scale{
          sycl::native::exp2(max_prev * params.scale - max_scale)};
      CUTLASS_PRAGMA_UNROLL
      for (int indx = 0; indx < Vec * FragsM; indx++) { // 16 rows in total
        auto max_scale_bcast = group_broadcast(g, max_scale, indx);
        auto exp_scale_bcast = group_broadcast(g, exp_scale, indx);
        sum(indx) *= exp_scale_bcast;
        CUTLASS_PRAGMA_UNROLL
        for (int z = 0; z < FragsNAcc; z++) {
          auto base_indx = indx + (z * Vec * FragsM);
          frag_s(base_indx) =
              sycl::native::exp2((frag_s(base_indx) - max_scale_bcast));
          sum(indx) += frag_s(base_indx);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int z = 0; z < FragsNOut; z++) {
          auto base_indx = indx + (z * Vec * FragsM); // z * 16 rows
          out(base_indx) *= exp_scale_bcast;
        }
      }
    } else {
      scale_exp_log2<Vec, FragsM, FragsNAcc>(frag_s, max, sum);
    }
  }
  Params params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace flash_attention
} // namespace cutlass
