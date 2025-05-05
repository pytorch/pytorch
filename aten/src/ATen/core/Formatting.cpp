#include <ATen/core/Formatting.h>
#include <c10/util/irange.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>

namespace c10 {
std::ostream& operator<<(std::ostream& out, Backend b) {
  return out << toString(b);
}

std::ostream& operator<<(std::ostream& out, const Scalar& s) {
  if (s.isFloatingPoint()) {
    return out << s.toDouble();
  }
  if (s.isComplex()) {
    return out << s.toComplexDouble();
  }
  if (s.isBoolean()) {
    return out << (s.toBool() ? "true" : "false");
  }
  if (s.isSymInt()) {
    return out << (s.toSymInt());
  }
  if (s.isSymFloat()) {
    return out << (s.toSymFloat());
  }
  if (s.isIntegral(false)) {
    return out << s.toLong();
  }
  throw std::logic_error("Unknown type in Scalar");
}

std::string toString(const Scalar& s) {
  return fmt::format("{}", s);
}
} // namespace c10

namespace at {

// Format guard to preserve formatting options
class FormatGuard {
 public:
  explicit FormatGuard(std::ostream& out) : out(out) {
    saved.copyfmt(out);
  }
  ~FormatGuard() {
    out.copyfmt(saved);
  }
  FormatGuard(const FormatGuard&) = delete;
  FormatGuard(FormatGuard&&) = delete;
  FormatGuard& operator=(const FormatGuard&) = delete;
  FormatGuard& operator=(FormatGuard&&) = delete;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::ostream& out;
  std::ios saved{nullptr};
};

std::ostream& operator<<(std::ostream& out, const DeprecatedTypeProperties& t) {
  return out << t.toString();
}

// Determine formatting options for tensor printing
static std::tuple<double, int> __printFormat(const Tensor& self) {
  auto size = self.numel();
  if (size == 0) {
    return std::make_tuple(1.0, 0);
  }

  bool intMode = true;
  auto self_p = self.const_data_ptr<double>();
  for (const auto i : c10::irange(size)) {
    auto z = self_p[i];
    if (std::isfinite(z)) {
      if (z != std::ceil(z)) {
        intMode = false;
        break;
      }
    }
  }

  int64_t offset = 0;
  while (offset < size && !std::isfinite(self_p[offset])) {
    offset++;
  }

  double expMin = 1;
  double expMax = 1;
  if (offset != size) {
    expMin = fabs(self_p[offset]);
    expMax = fabs(self_p[offset]);
    for (const auto i : c10::irange(offset, size)) {
      double z = fabs(self_p[i]);
      if (std::isfinite(z)) {
        if (z < expMin) {
          expMin = z;
        }
        if (self_p[i] > expMax) {
          expMax = z;
        }
      }
    }

    expMin = (expMin != 0) ? std::floor(std::log10(expMin)) + 1 : 1;
    expMax = (expMax != 0) ? std::floor(std::log10(expMax)) + 1 : 1;
  }

  double scale = 1;
  int sz = 11;

  if (intMode) {
    sz = (expMax > 9) ? 11 : static_cast<int>(expMax) + 1;
  } else {
    if (expMax - expMin > 4) {
      sz = 11;
      if (std::fabs(expMax) > 99 || std::fabs(expMin) > 99) {
        sz++;
      }
    } else {
      if (expMax > 5 || expMax < 0) {
        sz = 7;
        scale = std::pow(10, expMax - 1);
      } else {
        sz = (expMax == 0) ? 7 : static_cast<int>(expMax) + 6;
      }
    }
  }

  return std::make_tuple(scale, sz);
}

// Print indentation
static std::string __printIndent(int64_t indent) {
  return std::string(indent, ' ');
}

// Print scale information
static std::string printScale(double scale) {
  return fmt::format("{} *\n", scale);
}

// Print a matrix
static void __printMatrix(
    std::ostream& stream,
    const Tensor& self,
    int64_t linesize,
    int64_t indent) {
  auto [scale, sz] = __printFormat(self);
  std::string indent_str = __printIndent(indent);

  fmt::print(stream, "{}", indent_str);
  int64_t nColumnPerLine = (linesize - indent) / (sz + 1);
  int64_t firstColumn = 0;
  int64_t lastColumn = -1;

  while (firstColumn < self.size(1)) {
    lastColumn = std::min(firstColumn + nColumnPerLine - 1, self.size(1) - 1);

    if (nColumnPerLine < self.size(1)) {
      if (firstColumn != 0) {
        fmt::print(stream, "\n");
      }
      fmt::print(stream, "Columns {} to {}", firstColumn + 1, lastColumn + 1);
      fmt::print(stream, "{}", indent_str);
    }

    if (scale != 1) {
      fmt::print(stream, "{}", printScale(scale));
      fmt::print(stream, "{}", indent_str);
    }

    for (const auto l : c10::irange(self.size(0))) {
      Tensor row = self.select(0, l);
      const double* row_ptr = row.const_data_ptr<double>();

      for (const auto c : c10::irange(firstColumn, lastColumn + 1)) {
        // Print with width but only add space if not the last column
        if (c < lastColumn) {
          fmt::print(stream, "{:>{}} ", row_ptr[c] / scale, sz);
        } else {
          fmt::print(stream, "{:>{}}", row_ptr[c] / scale, sz);
          fmt::print(stream, "\n");
          if (l != self.size(0) - 1) {
            fmt::print(
                stream, "{}", scale != 1 ? indent_str + " " : indent_str);
          }
        }
      }
    }

    firstColumn = lastColumn + 1;
  }
}

// Print a tensor with more than 2 dimensions
static void __printTensor(
    std::ostream& stream,
    Tensor& self,
    int64_t linesize) {
  std::vector<int64_t> counter(self.ndimension() - 2);
  bool start = true;
  bool finished = false;

  counter[0] = -1;
  for (const auto i : c10::irange(1, counter.size())) {
    counter[i] = 0;
  }

  while (true) {
    for (int64_t i = 0; i < self.ndimension() - 2; i++) {
      counter[i]++;
      if (counter[i] >= self.size(i)) {
        if (i == self.ndimension() - 3) {
          finished = true;
          break;
        }
        counter[i] = 0;
      } else {
        break;
      }
    }

    if (finished) {
      break;
    }

    if (start) {
      start = false;
    } else {
      fmt::print(stream, "\n");
    }

    fmt::print(stream, "(");
    Tensor tensor = self;
    for (const auto i : c10::irange(self.ndimension() - 2)) {
      tensor = tensor.select(0, counter[i]);
      fmt::print(stream, "{},", counter[i] + 1);
    }
    fmt::print(stream, ".,.) = \n");
    __printMatrix(stream, tensor, linesize, 1);
  }
}

void print(const Tensor& t, int64_t linesize) {
  print(std::cout, t, linesize);
}

std::ostream& print(
    std::ostream& stream,
    const Tensor& tensor_,
    int64_t linesize) {
  FormatGuard guard(stream);

  if (!tensor_.defined()) {
    fmt::print(stream, "[ Tensor (undefined) ]");
    return stream;
  }

  if (tensor_.is_sparse()) {
    fmt::print(stream, "[ {}{{}}]\n", tensor_.toString());
    fmt::print(stream, "indices:\n");
    print(stream, tensor_._indices(), linesize);
    fmt::print(stream, "\nvalues:\n");
    print(stream, tensor_._values(), linesize);
    fmt::print(stream, "\nsize:\n{}\n]", tensor_.sizes());
    return stream;
  }

  Tensor tensor;
  if (tensor_.is_quantized()) {
    tensor = tensor_.dequantize().to(kCPU, kDouble).contiguous();
  } else if (tensor_.is_mkldnn()) {
    fmt::print(stream, "MKLDNN Tensor: ");
    tensor = tensor_.to_dense().to(kCPU, kDouble).contiguous();
  } else if (tensor_.is_mps()) {
    // MPS does not support double tensors, so first copy then convert
    tensor = tensor_.to(kCPU).to(kDouble).contiguous();
  } else {
    tensor = tensor_.to(kCPU, kDouble).contiguous();
  }

  if (tensor.ndimension() == 0) {
    fmt::print(stream, "{}\n", tensor.const_data_ptr<double>()[0]);
    fmt::print(stream, "[ {}{{}}", tensor_.toString());
  } else if (tensor.ndimension() == 1) {
    if (tensor.numel() > 0) {
      auto [scale, sz] = __printFormat(tensor);
      if (scale != 1) {
        fmt::print(stream, "{}", printScale(scale));
      }

      const double* tensor_p = tensor.const_data_ptr<double>();
      for (const auto i : c10::irange(tensor.size(0))) {
        // Match the original format without trailing space
        fmt::print(stream, "{:>{}}\n", tensor_p[i] / scale, sz);
      }
    }
    fmt::print(stream, "[ {}{{{}}}", tensor_.toString(), tensor.size(0));
  } else if (tensor.ndimension() == 2) {
    if (tensor.numel() > 0) {
      __printMatrix(stream, tensor, linesize, 0);
    }
    fmt::print(
        stream,
        "[ {}{{{},{}}}",
        tensor_.toString(),
        tensor.size(0),
        tensor.size(1));
  } else {
    if (tensor.numel() > 0) {
      __printTensor(stream, tensor, linesize);
    }

    // Build the size string using fmt::format_to for efficiency
    fmt::memory_buffer sizes;
    fmt::format_to(std::back_inserter(sizes), "{}", tensor.size(0));
    for (const auto i : c10::irange(1, tensor.ndimension())) {
      fmt::format_to(std::back_inserter(sizes), ",{}", tensor.size(i));
    }

    fmt::print(stream, "[ {}{{{}}}", tensor_.toString(), sizes);
  }

  if (tensor_.is_quantized()) {
    fmt::print(stream, ", qscheme: {}", toString(tensor_.qscheme()));

    if (tensor_.qscheme() == c10::kPerTensorAffine) {
      fmt::print(stream, ", scale: {}", tensor_.q_scale());
      fmt::print(stream, ", zero_point: {}", tensor_.q_zero_point());
    } else if (
        tensor_.qscheme() == c10::kPerChannelAffine ||
        tensor_.qscheme() == c10::kPerChannelAffineFloatQParams) {
      fmt::print(stream, ", scales: ");
      Tensor scales = tensor_.q_per_channel_scales();
      print(stream, scales, linesize);
      fmt::print(stream, ", zero_points: ");
      Tensor zero_points = tensor_.q_per_channel_zero_points();
      print(stream, zero_points, linesize);
      fmt::print(stream, ", axis: {}", tensor_.q_per_channel_axis());
    }
  }

  // Proxy check for if autograd was built
  if (tensor.getIntrusivePtr()->autograd_meta()) {
    auto& fw_grad = tensor._fw_grad(/* level */ 0);
    if (fw_grad.defined()) {
      fmt::print(stream, ", tangent:\n");
      print(stream, fw_grad, linesize);
    }
  }

  fmt::print(stream, " ]");
  return stream;
}

} // namespace at
