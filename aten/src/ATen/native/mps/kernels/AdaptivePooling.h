#pragma once

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

struct AdaptiveAvgPool2DParams {
  long B;
  long C;
  long input_height;
  long input_width;
  long output_height;
  long output_width;
  long input_strides[4];
  long output_strides[4];

#ifndef __METAL_VERSION__
  constexpr bool bin_bounds_fit(uint64_t max_index) const {
    const auto fits = [max_index](long input_size, long output_size) {
      if (input_size <= 0 || output_size <= 0) {
        return false;
      }
      const auto input = static_cast<uint64_t>(input_size);
      const auto output = static_cast<uint64_t>(output_size);
      const auto largest = input > output ? input : output;
      // Bound the addition before -1 in both forward and backward bin ends.
      return largest <= max_index && input <= (max_index - largest) / output;
    };
    return fits(input_height, output_height) && fits(input_width, output_width);
  }
#endif
};
