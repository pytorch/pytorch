#pragma once

#include <ATen/core/CachingHostAllocator.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <include/openreg.h>

namespace c10::openreg {
struct OpenRegHostAllocator final : at::HostAllocator {
  OpenRegHostAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override;

  at::DeleterFnPtr raw_deleter() const override;

  void copy_data(void* dest, const void* src, std::size_t count) const final;

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override;

  void empty_cache() override;

  at::HostStats get_stats() override;

  void reset_accumulated_stats() override;

  void reset_peak_stats() override;
};

} // namespace c10::openreg
