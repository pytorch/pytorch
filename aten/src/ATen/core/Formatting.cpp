#include <ATen/core/Formatting.h>

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

struct Formatter {
  Formatter(const Tensor& self) 
  : sz_(1), intMode_(true), sciMode_(false) {
    std::stringstream ss;

    Tensor mask = (self.ne(at::Scalar(0))).__and__(self.isfinite());
    Tensor nonzeroFiniteVals = self.masked_select(mask);

    if (nonzeroFiniteVals.numel() == 0){
      return;
    }

    Tensor nonzeroFiniteAbs = nonzeroFiniteVals.abs();
    double nonZeroFiniteMin = nonzeroFiniteAbs.min().item<double>();
    double nonZeroFiniteMax = nonzeroFiniteAbs.max().item<double>();

    auto size = nonzeroFiniteVals.numel();
    auto nzf_p = nonzeroFiniteVals.data_ptr<double>();

    for (int64_t i = 0; i < size; i++) {
      auto z = nzf_p[i];
      if (z != std::ceil(z)) {
        intMode_ = false;
        break;
      }
    }

    if (intMode_) {
      if ((nonZeroFiniteMax / nonZeroFiniteMin) > 1000. ||
          nonZeroFiniteMax > 1.e8) {
        sciMode_ = true;
        ss << std::scientific << std::setprecision(4);
      } else {
        ss << std::defaultfloat;
      }
    } else {
      if ((nonZeroFiniteMax / nonZeroFiniteMin) > 1000. ||
          nonZeroFiniteMax > 1.e8 || nonZeroFiniteMin < 1.e-4) {
        sciMode_ = true;
        ss << std::scientific << std::setprecision(4);
      } else {
        ss << std::fixed << std::setprecision(4);
      }
    }

    for (int64_t i = 0; i < nzfValsSize; i++) {
      ss << nzf_p[i];
      sz_ = std::max(sz_, (int64_t) ss.str().length());
      ss.str("");
    }
  }

  std::string format(double value, bool isFloatingDtype) const {
    std::stringstream ss;

    if (isFloatingDtype) {
      if (sciMode_) {
        ss << std::scientific << std::setprecision(4);
        ss << std::setw(sz_) << value;
      } else if (intMode_) {
        ss << defaultfloat;
        if (std::isfinite(value)) {
          ss << std::setw(sz_) << value << ".";
        } else {
          ss << std::setw(sz_) << value ;
        }
      } else {
        ss << std::fixed << std::setprecision(4);
        ss << std::setw(sz_) << value;
      }
    } else {
      ss << defaultfloat;
      ss << std::setw(sz_) << value;
    }
    return ss.str();
  }

private:
  int64_t sz_;
  bool intMode_;
  bool sciMode_;
};

std::ostream& operator<<(std::ostream & out, const DeprecatedTypeProperties& t) {
  return out << t.toString();
}

std::string __vectorString( const Tensor& tensor_, const Formatter& formatter, bool isFloatingDtype) {
  std::stringstream ss;
  if (tensor_.numel() > 0) {
    double* tensor_p = tensor_.data_ptr<double>();
    ss << "[";
    for (int64_t i = 0; i < tensor_.size(0); i++) {
      ss << formatter.format(tensor_p[i], isFloatingDtype);
      if (i < tensor_.size(0) - 1) {
        ss << ", ";
      }
    }
    ss << "]";
  }
  return ss.str();
}

std::string __tensorString(const Tensor & tensor_, const Formatter& formatter, int64_t indent, bool isFloatingDtype){
  std::stringstream ss ;
  std::vector<std::string> slices;
  int64_t dim = tensor_.dim();

  if (dim == 0) {
    ss << defaultfloat << tensor_.data_ptr<double>()[0] << std::endl;
    ss << "[ " << tensor_.toString() << "{}";
    return ss.str();
  } else if (dim == 1) {
    return __vectorString(tensor_, formatter, isFloatingDtype);
  } else {
    for ( int64_t i = 0; i < tensor_.size(0); i++){
      slices.push_back(__tensorString(tensor_.select(0, i), formatter, indent+1, isFloatingDtype));
    }
  }
  ss << "[";
  for (unsigned long i = 0; i < slices.size(); i++){
    if (i < slices.size() - 1){
      ss << slices[i] << "," ;
      for (int64_t d = 0; d < dim-1; d++){
        ss << std::endl;
      }
      for(int64_t i = 0; i < indent+1; i++) {
        ss << " ";
      }
    } else {
      ss << slices[i];
    }
  }
  ss << "]";
  return ss.str();
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
    bool isFloatingDtype =  tensor_.is_floating_point();
    Tensor tensor;    
    if (tensor_.is_quantized()) {
      tensor = tensor_.dequantize().to(kCPU, kDouble).contiguous();
    } else if (tensor_.is_mkldnn()) {
      stream << "MKLDNN Tensor: ";
      tensor = tensor_.to_dense().to(kCPU, kDouble).contiguous();
    } else {
      tensor = tensor_.to(kCPU, kDouble).contiguous();
    }

    std::string prefix = "tensor(";
    int64_t indent = prefix.length();
    stream << prefix;
    Formatter formatter(tensor);
    std::string tensorStr = __tensorString(tensor, formatter, indent, isFloatingDtype);
    stream << tensorStr << ")" << std::endl;

    stream << "[ " << tensor_.toString() << "{" << tensor.size(0);
    for(int64_t i = 1; i < tensor.ndimension(); i++) {
      stream << "," << tensor.size(i);
    }
    stream << "}";

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
