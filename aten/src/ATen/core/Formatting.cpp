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
  Formatter(const Tensor& tensor_, bool isFloatingDType) 
  : maxWidth_(1), intMode_(true), sciMode_(false), isFloatingDtype_(isFloatingDType) {

    //stream used to find maxWidth later
    std::stringstream sstream;

    if (!isFloatingDtype_){
      auto value_p = tensor_.data_ptr<double>();
      for (int64_t i = 0; i < tensor_.numel(); i++) {
        sstream << value_p[i];
        maxWidth_ = std::max(maxWidth_, (int64_t)sstream.str().length());
        sstream.str("");
      }
      return;
    }
          
    Tensor mask = (tensor_.ne(at::Scalar(0))).__and__(tensor_.isfinite());
    Tensor nonzeroFiniteVals = tensor_.masked_select(mask);

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
        sstream << std::scientific << std::setprecision(4);
      } else {
        sstream << std::defaultfloat;
      }
    } else {
      if ((nonZeroFiniteMax / nonZeroFiniteMin) > 1000. ||
          nonZeroFiniteMax > 1.e8 || nonZeroFiniteMin < 1.e-4) {
        sciMode_ = true;
        sstream << std::scientific << std::setprecision(4);
      } else {
        sstream << std::fixed << std::setprecision(4);
      }
    }

    // Iterate once over the elements to set maxWidth_
    for (int64_t i = 0; i < size; i++) {
      sstream << nzf_p[i];
      maxWidth_ = std::max(maxWidth_, (int64_t) sstream.str().length());
      sstream.str("");
    }
  }

  std::string format(double value) const {
    std::stringstream sstream;

    if (isFloatingDtype_) {
      if (sciMode_) {
        sstream << std::scientific << std::setprecision(4);
        sstream << std::setw(maxWidth_) << value;
      } else if (intMode_) {
        sstream << defaultfloat;
        if (std::isfinite(value)) {
          sstream << std::setw(maxWidth_) << value << ".";
        } else {
          sstream << std::setw(maxWidth_) << value;
        }
      } else {
        sstream << std::fixed << std::setprecision(4);
        sstream << std::setw(maxWidth_) << value;
      }
    } else {
      sstream << std::setw(maxWidth_) << value;
    }
    return sstream.str();
  }

private:
  int64_t maxWidth_;
  bool intMode_;
  bool sciMode_;
  bool isFloatingDtype_;
};

std::ostream& operator<<(std::ostream & out, const DeprecatedTypeProperties& t) {
  return out << t.toString();
}

std::string __vectorString( const Tensor& tensor_, const Formatter& formatter) {
  std::stringstream ss;
  if (tensor_.numel() > 0) {
    double* tensor_p = tensor_.data_ptr<double>();
    ss << "[";
    for (int64_t i = 0; i < tensor_.size(0); i++) {
      ss << formatter.format(tensor_p[i]);
      if (i < tensor_.size(0) - 1) {
        ss << ", ";
      }
    }
    ss << "]";
  }
  return ss.str();
}

std::string __scalarString( const Tensor& tensor_, const Formatter& formatter) {
  std::stringstream ss;
  if (tensor_.numel() > 0) {
    double* tensor_p = tensor_.data_ptr<double>();
    ss << formatter.format(tensor_p[0]);
  }
  return ss.str();
}

// Recursive function to generate the print output. Relies on the formatter.
std::string __tensorString(const Tensor & tensor_, const Formatter& formatter, int64_t indent){
  std::stringstream ss ;
  std::vector<std::string> slices;
  int64_t dim = tensor_.dim();

  if (dim == 0) {
    return __scalarString(tensor_, formatter);
  } else if (dim == 1) {
    return __vectorString(tensor_, formatter);
  } else {
    for ( int64_t i = 0; i < tensor_.size(0); i++){
      slices.push_back(__tensorString(tensor_.select(0, i), formatter, indent+1));
    }
  }
  ss << "[";
  for (unsigned long i = 0; i < slices.size(); i++){
    if (i < slices.size() - 1){
      ss << slices[i] << "," ;
      for (int64_t d = 0; d < dim-1; d++){
        ss << std::endl;
      }
      for(int64_t j = 0; j < indent+1; j++) {
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
    std::string tensorStr;
    if (tensor.numel() == 0) {
        tensorStr = "[]";
    } else{
        bool isFloatingDtype =  tensor_.is_floating_point();
        Formatter formatter(tensor, isFloatingDtype);
        tensorStr = __tensorString(tensor, formatter, indent);
    }
    // Print the formatted values
    stream << prefix << tensorStr << ")" << std::endl;

    // Print the tensor type and dimensions
    stream << "[ " << tensor_.toString() << "{";
    if (tensor.ndimension() != 0) {
      stream << tensor.size(0);
      for (int64_t i = 1; i < tensor.ndimension(); i++) {
        stream << "," << tensor.size(i);
      }
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
