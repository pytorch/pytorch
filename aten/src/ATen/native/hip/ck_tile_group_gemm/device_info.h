#pragma once

#include <optional>
#include <string_view>

namespace at::hip::detail::ck_tile_group_gemm {

    // Datasheet peak dense matrix throughput (TFLOPS) and DRAM bandwidth (GB/s) of each GPU model.
    enum class GpuModel { MI300A, MI300X, MI350X, MI355X };

    template <GpuModel Model>
    struct DeviceInfo;

    template <>
    struct DeviceInfo<GpuModel::MI355X> {
        static constexpr double fp32_tflops = 157.3, fp16_tflops = 2516.6, bf16_tflops = 2516.6, dram_bw_gbs = 8000.0;
    };

    // Source: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf
    template <>
    struct DeviceInfo<GpuModel::MI350X> {
        static constexpr double fp32_tflops = 144.2, fp16_tflops = 2309.6, bf16_tflops = 2309.6, dram_bw_gbs = 8000.0;
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

    // Calls f(DeviceInfo<Model>{}) for the GPU named `name` (hipDeviceProp_t::name, e.g. "AMD Instinct MI355X") and
    // returns its result, or std::nullopt for a GPU without an entry.
    template <typename F>
    auto with_device_info(std::string_view name, F&& f) -> std::optional<decltype(f(DeviceInfo<GpuModel::MI355X>{}))> {
        const auto is = [&](std::string_view model) { return name.find(model) != std::string_view::npos; };
        if (is("MI355X")) return f(DeviceInfo<GpuModel::MI355X>{});
        if (is("MI350X")) return f(DeviceInfo<GpuModel::MI350X>{});
        if (is("MI300X")) return f(DeviceInfo<GpuModel::MI300X>{});
        if (is("MI300A")) return f(DeviceInfo<GpuModel::MI300A>{});
        return std::nullopt;
    }

} // namespace at::hip::detail::ck_tile_group_gemm
