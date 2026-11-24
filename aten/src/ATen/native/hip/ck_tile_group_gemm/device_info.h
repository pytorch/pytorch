#pragma once

#include <optional>
#include <string_view>

namespace at::hip::detail::ck_tile_group_gemm {

    // Datasheet peak dense matrix throughput (TFLOPS) and DRAM bandwidth (GB/s) of each GPU model.
    enum class GpuModel { MI300A, MI300X, MI355X };

    template <GpuModel Model>
    struct DeviceInfo;

    // Source: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf
    template <>
    struct DeviceInfo<GpuModel::MI355X> {
        static constexpr double fp32_tflops = 157.3, fp16_tflops = 2516.6, bf16_tflops = 2516.6, dram_bw_gbs = 8000.0;
    };

    // Source: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    template <>
    struct DeviceInfo<GpuModel::MI300X> {
        static constexpr double fp32_tflops = 163.4, fp16_tflops = 1307.4, bf16_tflops = 1307.4, dram_bw_gbs = 5300.0;
    };

    // Source: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf
    template <>
    struct DeviceInfo<GpuModel::MI300A> {
        static constexpr double fp32_tflops = 122.6, fp16_tflops = 980.6, bf16_tflops = 980.6, dram_bw_gbs = 5300.0;
    };

    // Calls f(DeviceInfo<Model>{}) for the GPU with architecture `gcn_arch` (hipDeviceProp_t::gcnArchName, e.g.
    // "gfx950:sramecc+:xnack-") and `cus` compute units and returns its result, or std::nullopt for another
    // architecture. gfx950 (MI350X, MI355X) uses MI355X; gfx942 uses MI300A at 228 CUs and MI300X otherwise.
    template <typename F>
    auto with_device_info(std::string_view gcn_arch, int cus, F&& f) -> std::optional<decltype(f(DeviceInfo<GpuModel::MI355X>{}))> {
        const auto arch = gcn_arch.substr(0, gcn_arch.find(':'));
        if (arch == "gfx950") return f(DeviceInfo<GpuModel::MI355X>{});
        if (arch == "gfx942") return cus == 228 ? f(DeviceInfo<GpuModel::MI300A>{}) : f(DeviceInfo<GpuModel::MI300X>{});
        return std::nullopt;
    }

} // namespace at::hip::detail::ck_tile_group_gemm
