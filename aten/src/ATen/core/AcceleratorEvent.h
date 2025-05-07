#pragma once

#include <optional>
#include <ATen/Device.h>

namespace at {

struct AcceleratorEvent {
    virtual ~AcceleratorEvent() = default;

    virtual bool query() const = 0;
    virtual double elapsed_time(const AcceleratorEvent& other) const = 0;
};

} // namespace at