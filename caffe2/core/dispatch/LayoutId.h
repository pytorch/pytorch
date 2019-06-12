#pragma once

#include "c10/util/IdWrapper.h"

namespace c10 {

class LayoutId final : public at::IdWrapper<LayoutId, uint8_t> {
public:
    constexpr explicit LayoutId(underlying_type id): IdWrapper(id) {}

    constexpr uint8_t value() const {
        return underlyingId();
    }

    // Don't use this default constructor!
    // Unfortunately, a default constructor needs to be defined because of https://reviews.llvm.org/D41223
    constexpr LayoutId(): IdWrapper(0) {}
};

}

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::LayoutId)
