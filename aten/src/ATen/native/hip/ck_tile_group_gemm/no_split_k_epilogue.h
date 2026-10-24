#pragma once

#include <ck_tile/core.hpp>

#include <utility>

namespace at::hip::detail::ck_tile_group_gemm {

    // GroupedGemmKernel instantiates a split-K (k_batch > 1) atomic_add epilogue next to the plain store one.
    // We always run k_batch == 1, so compile the atomic_add path out: 16-bit atomics reject a C vector width of 1.
    template <typename Epilogue>
    struct NoSplitKEpilogue : Epilogue {
        template <typename OutWindow, typename... Args>
        CK_TILE_DEVICE auto operator()(OutWindow& out_window, Args&&... args) {
            if constexpr (OutWindow::BottomTensorView::DstInMemOp == ck_tile::memory_operation_enum::set) {
                return Epilogue::operator()(out_window, std::forward<Args>(args)...);
            }
        }
    };

} // namespace at::hip::detail::ck_tile_group_gemm
