#include <ATen/core/Formatting.h>
#include <c10/util/irange.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>

namespace c10 {
std::ostream& operator<<(std::ostream & out, Backend b) {
  return out << toString(b);
}

std::ostream& operator<<(std::ostream & out, const Scalar& s) {
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
  std::stringstream out;
  out << s;
  return out.str();
}
}
namespace at {

//not all C++ compilers have default float so we define our own here
inline std::ios_base& defaultfloat(std::ios_base& __base) {
  __base.unsetf(std::ios_base::floatfield);
  return __base;
}
//saves/restores number formatting inside scope
struct FormatGuard {
  FormatGuard(std::ostream & out)
  : out(out), saved(nullptr) {
    saved.copyfmt(out);
  }
  ~FormatGuard() {
    out.copyfmt(saved);
  }
private:
  std::ostream & out;
  std::ios saved;
};

std::ostream& operator<<(std::ostream & out, const DeprecatedTypeProperties& t) {
  return out << t.toString();
}

static std::tuple<double, int64_t> __printFormat(std::ostream& stream, const Tensor& self) {
  auto size = self.numel();
  if(size == 0) {
    return std::make_tuple(1., 0);
  }
  bool intMode = true;
  auto self_p = self.data_ptr<double>();
  for (const auto i : c10::irange(size)) {
    auto z = self_p[i];
    if(std::isfinite(z)) {
      if(z != std::ceil(z)) {
        intMode = false;
        break;
      }
    }
  }
  int64_t offset = 0;
  while(!std::isfinite(self_p[offset])) {
    offset = offset + 1;
    if(offset == size) {
      break;
    }
  }
  double expMin = 1;
  double expMax = 1;
  if(offset != size) {
    expMin = fabs(self_p[offset]);
    expMax = fabs(self_p[offset]);
    for (const auto i : c10::irange(offset, size)) {
      double z = fabs(self_p[i]);
      if(std::isfinite(z)) {
        if(z < expMin) {
          expMin = z;
        }
        if(self_p[i] > expMax) {
          expMax = z;
        }
      }
    }
    if(expMin != 0) {
      expMin = std::floor(std::log10(expMin)) + 1;
    } else {
      expMin = 1;
    }
    if(expMax != 0) {
      expMax = std::floor(std::log10(expMax)) + 1;
    } else {
      expMax = 1;
    }
  }
  double scale = 1;
  int64_t sz = 11;
  if(intMode) {
    if(expMax > 9) {
      sz = 11;
      stream << std::scientific << std::setprecision(4);
    } else {
      sz = expMax + 1;
      stream << defaultfloat;
    }
  } else {
    if(expMax-expMin > 4) {
      sz = 11;
      if(std::fabs(expMax) > 99 || std::fabs(expMin) > 99) {
        sz = sz + 1;
      }
      stream << std::scientific << std::setprecision(4);
    } else {
      if(expMax > 5 || expMax < 0) {
        sz = 7;
        scale = std::pow(10, expMax-1);
        stream << std::fixed << std::setprecision(4);
      } else {
        if(expMax == 0) {
          sz = 7;
        } else {
          sz = expMax+6;
        }
        stream << std::fixed << std::setprecision(4);
      }
    }
  }
  return std::make_tuple(scale, sz);
}

static void __printIndent(std::ostream &stream, int64_t indent)
{
  for (C10_UNUSED const auto i : c10::irange(indent)) {
    stream << " ";
  }
}

static void printScale(std::ostream & stream, double scale) {
  FormatGuard guard(stream);
  stream << defaultfloat << scale << " *" << std::endl;
}
static void __printMatrix(std::ostream& stream, const Tensor& self, int64_t linesize, int64_t indent)
{
  double scale = 0.0;
  int64_t sz = 0;
  std::tie(scale, sz) = __printFormat(stream, self);

  __printIndent(stream, indent);
  int64_t nColumnPerLine = (linesize-indent)/(sz+1);
  int64_t firstColumn = 0;
  int64_t lastColumn = -1;
  while(firstColumn < self.size(1)) {
    if(firstColumn + nColumnPerLine <= self.size(1)) {
      lastColumn = firstColumn + nColumnPerLine - 1;
    } else {
      lastColumn = self.size(1) - 1;
    }
    if(nColumnPerLine < self.size(1)) {
      if(firstColumn != 0) {
        stream << std::endl;
      }
      stream << "Columns " << firstColumn+1 << " to " << lastColumn+1;
      __printIndent(stream, indent);
    }
    if(scale != 1) {
      printScale(stream,scale);
      __printIndent(stream, indent);
    }
    for (const auto l : c10::irange(self.size(0))) {
      Tensor row = self.select(0,l);
      double *row_ptr = row.data_ptr<double>();
      for (const auto c : c10::irange(firstColumn, lastColumn+1)) {
        stream << std::setw(sz) << row_ptr[c]/scale;
        if(c == lastColumn) {
          stream << std::endl;
          if(l != self.size(0)-1) {
            if(scale != 1) {
              __printIndent(stream, indent);
              stream << " ";
            } else {
              __printIndent(stream, indent);
            }
          }
        } else {
          stream << " ";
        }
      }
    }
    firstColumn = lastColumn + 1;
  }
}

static void __printTensor(std::ostream& stream, Tensor& self, int64_t linesize)
{
  std::vector<int64_t> counter(self.ndimension()-2);
  bool start = true;
  bool finished = false;
  counter[0] = -1;
  for (const auto i : c10::irange(1, counter.size())) {
    counter[i] = 0;
  }
  while(true) {
    for(int64_t i = 0; self.ndimension()-2; i++) {
      counter[i] = counter[i] + 1;
      if(counter[i] >= self.size(i)) {
        if(i == self.ndimension()-3) {
          finished = true;
          break;
        }
        counter[i] = 0;
      } else {
        break;
      }
    }
    if(finished) {
      break;
    }
    if(start) {
      start = false;
    } else {
      stream << std::endl;
    }
    stream << "(";
    Tensor tensor = self;
    for (const auto i : c10::irange(self.ndimension()-2)) {
      tensor = tensor.select(0, counter[i]);
      stream << counter[i]+1 << ",";
    }
    stream << ".,.) = " << std::endl;
    __printMatrix(stream, tensor, linesize, 1);
  }
}

void print(const Tensor & t, int64_t linesize) {
  print(std::cout,t,linesize);
}
std::ostream& print(std::ostream& stream, const Tensor & tensor_, int64_t linesize) {
  FormatGuard guard(stream);
  if(!tensor_.defined()) {
    stream << "[ Tensor (undefined) ]";
  } else if (tensor_.is_sparse()) {
    stream << "[ " << tensor_.toString() << "{}\n";
    stream << "indices:\n" << tensor_._indices() << "\n";
    stream << "values:\n" << tensor_._values() << "\n";
    stream << "size:\n" << tensor_.sizes() << "\n";
    stream << "]";
  } else {
    Tensor tensor;
    if (tensor_.is_quantized()) {
      tensor = tensor_.dequantize().to(kCPU, kDouble).contiguous();
    } else if (tensor_.is_mkldnn()) {
      stream << "MKLDNN Tensor: ";
      tensor = tensor_.to_dense().to(kCPU, kDouble).contiguous();
    } else if (tensor_.is_mps()) {
      // MPS does not support double tensors, so first copy then convert
      tensor = tensor_.to(kCPU).to(kDouble).contiguous();
    } else {
      tensor = tensor_.to(kCPU, kDouble).contiguous();
    }
    if(tensor.ndimension() == 0) {
      stream << defaultfloat << tensor.data_ptr<double>()[0] << std::endl;
      stream << "[ " << tensor_.toString() << "{}";
    } else if(tensor.ndimension() == 1) {
      if (tensor.numel() > 0) {
        double scale = 0.0;
        int64_t sz = 0;
        std::tie(scale, sz) =  __printFormat(stream, tensor);
        if(scale != 1) {
          printScale(stream, scale);
        }
        double* tensor_p = tensor.data_ptr<double>();
        for (const auto i : c10::irange(tensor.size(0))) {
          stream << std::setw(sz) << tensor_p[i]/scale << std::endl;
        }
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "}";
    } else if(tensor.ndimension() == 2) {
      if (tensor.numel() > 0) {
        __printMatrix(stream, tensor, linesize, 0);
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "," <<  tensor.size(1) << "}";
    } else {
      if (tensor.numel() > 0) {
        __printTensor(stream, tensor, linesize);
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0);
      for (const auto i : c10::irange(1, tensor.ndimension())) {
        stream << "," << tensor.size(i);
      }
      stream << "}";
    }
    if (tensor_.is_quantized()) {
      stream << ", qscheme: " << toString(tensor_.qscheme());
      if (tensor_.qscheme() == c10::kPerTensorAffine) {
        stream << ", scale: " << tensor_.q_scale();
        stream << ", zero_point: " << tensor_.q_zero_point();
      } else if (tensor_.qscheme() == c10::kPerChannelAffine ||
          tensor_.qscheme() == c10::kPerChannelAffineFloatQParams) {
        stream << ", scales: ";
        Tensor scales = tensor_.q_per_channel_scales();
        print(stream, scales, linesize);
        stream << ", zero_points: ";
        Tensor zero_points = tensor_.q_per_channel_zero_points();
        print(stream, zero_points, linesize);
        stream << ", axis: " << tensor_.q_per_channel_axis();
      }
    }

    // Proxy check for if autograd was built
    if (tensor.getIntrusivePtr()->autograd_meta()) {
      auto& fw_grad = tensor._fw_grad(/* level */ 0);
      if (fw_grad.defined()) {
        stream << ", tangent:" << std::endl << fw_grad;
      }
    }
    stream << " ]";
  }
  return stream;
}

}
