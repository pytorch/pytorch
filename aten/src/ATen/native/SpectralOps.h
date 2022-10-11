#pragma once

#include <string>
#include <stdexcept>
#include <sstream>

namespace at { namespace native {

Tensor fft_ifft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<c10::string_view> norm);

Tensor& fft_ifft_out(const Tensor& self, c10::optional<int64_t> n,
                     int64_t dim, c10::optional<c10::string_view> norm,
                     Tensor& out);

Tensor fft_fft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<c10::string_view> norm);

}} // at::native
