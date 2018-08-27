#include "ATen/ATen.h"

namespace at { namespace native {

Tensor constant_pad_nd(const Tensor& self, IntList pad, Scalar value) {
    auto input_sizes = self.sizes();
    auto l_inp = self.dim();

    auto l_pad = pad.size() / 2;
    auto l_diff = l_inp - l_pad;
    AT_CHECK(l_inp >= l_pad, "Padding length too large");

    std::vector<int64_t> new_shape;

    for (int i = 0; i < l_diff; i ++) {
        new_shape.push_back(input_sizes[i]);
    }

    for (int i = 0; i < l_pad; i++) {
        auto pad_idx = pad.size() - ((i + 1) * 2);
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        AT_CHECK(new_dim > 0, "input is too small");
        new_shape.push_back(new_dim);
    }

    auto output = at::empty(new_shape, self.options());
    output.fill_(value);

    auto c_input = self;
    for (int i = l_diff; i < l_inp; i++) {
        auto pad_idx = pad.size() - (i - l_diff + 1) * 2;
        if (pad[pad_idx] < 0) {
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
        }
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        }
    }

    auto c_output = output;
    for (int i = l_diff; i < l_inp; i++) {
        auto pad_idx = pad.size() - (i - l_diff + 1) * 2;
        if (pad[pad_idx] > 0) {
            c_output = c_output.narrow(i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
        }
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.narrow(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    c_output.copy_(c_input);
    return output;
}

}}  // namespace at::native