#include "ATen/Formatting.h"
#include "ATen/Tensor.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"

#include <cmath>
#include <iostream>
#include <iomanip>


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

std::ostream& operator<<(std::ostream & out, IntList list) {
  int i = 0;
  out << "[";
  for(auto e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

std::ostream& operator<<(std::ostream & out, Backend b) {
  return out << toString(b);
}

std::ostream& operator<<(std::ostream & out, ScalarType t) {
  return out << toString(t);
}

std::ostream& operator<<(std::ostream & out, const Type& t) {
  return out << t.toString();
}

static std::tuple<double, int64_t> __printFormat(std::ostream& stream, const Tensor& self) {
  auto size = self.numel();
  if(size == 0) {
    return std::make_tuple(1., 0);
  }
  bool intMode = true;
  auto self_p = self.data<double>();
  for(int64_t i = 0; i < size; i++) {
    auto z = self_p[i];
    if(std::isfinite(z)) {
      if(z != ceil(z)) {
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
  double expMin;
  double expMax;
  if(offset == size) {
    expMin = 1;
    expMax = 1;
  } else {
    expMin = fabs(self_p[offset]);
    expMax = fabs(self_p[offset]);
    for(int64_t i = offset; i < size; i++) {
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
      expMin = floor(log10(expMin)) + 1;
    } else {
      expMin = 1;
    }
    if(expMax != 0) {
      expMax = floor(log10(expMax)) + 1;
    } else {
      expMax = 1;
    }
  }
  double scale = 1;
  int64_t sz;
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
      if(fabs(expMax) > 99 || fabs(expMin) > 99) {
        sz = sz + 1;
      }
      stream << std::scientific << std::setprecision(4);
    } else {
      if(expMax > 5 || expMax < 0) {
        sz = 7;
        scale = pow(10, expMax-1);
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
  for(int64_t i = 0; i < indent; i++) {
    stream << " ";
  }
}

static void printScale(std::ostream & stream, double scale) {
  FormatGuard guard(stream);
  stream << defaultfloat << scale << " *" << std::endl;
}
static void __printMatrix(std::ostream& stream, const Tensor& self, int64_t linesize, int64_t indent)
{
  double scale;
  int64_t sz;
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
    for(int64_t l = 0; l < self.size(0); l++) {
      Tensor row = self.select(0,l);
      double *row_ptr = row.data<double>();
      for(int64_t c = firstColumn; c < lastColumn+1; c++) {
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

void __printTensor(std::ostream& stream, Tensor& self, int64_t linesize)
{
  std::vector<int64_t> counter(self.ndimension()-2);
  bool start = true;
  bool finished = false;
  counter[0] = -1;
  for(size_t i = 1; i < counter.size(); i++)
    counter[i] = 0;
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
    for(int64_t i=0; i < self.ndimension()-2; i++) {
      tensor = tensor.select(0, counter[i]);
      stream << counter[i]+1 << ",";
    }
    stream << ".,.) = " << std::endl;
    __printMatrix(stream, tensor, linesize, 1);
  }
}

std::ostream& print(std::ostream& stream, const Tensor & tensor_, int64_t linesize) {
  FormatGuard guard(stream);
  if(!tensor_.defined()) {
    stream << "[ Tensor (undefined) ]";
  } else {
    Type& cpudouble = tensor_.type().toBackend(kCPU).toScalarType(kDouble);
    Tensor tensor = tensor_.toType(cpudouble).contiguous();
    if(tensor.ndimension() == 0) {
      stream << defaultfloat << tensor.data<double>()[0] << std::endl;
      stream << "[ " << tensor_.pImpl->toString() << "{} ]";
    } else if(tensor.ndimension() == 1) {
      if (tensor.numel() == 0) {
        stream << "[ Tensor (empty) ]";
      }
      else {
        double scale;
        int64_t sz;
        std::tie(scale, sz) =  __printFormat(stream, tensor);
        if(scale != 1) {
          printScale(stream, scale);
        }
        double* tensor_p = tensor.data<double>();
        for(int64_t i = 0; i < tensor.size(0); i++) {
          stream << std::setw(sz) << tensor_p[i]/scale << std::endl;
        }
        stream << "[ " << tensor_.pImpl->toString() << "{" << tensor.size(0) << "} ]";
      }
    } else if(tensor.ndimension() == 2) {
      __printMatrix(stream, tensor, linesize, 0);
      stream << "[ " << tensor_.pImpl->toString() << "{" << tensor.size(0) << "," <<  tensor.size(1) << "} ]";
    } else {
        __printTensor(stream, tensor, linesize);
        stream << "[ " << tensor_.pImpl->toString() << "{" << tensor.size(0);
        for(int64_t i = 1; i < tensor.ndimension(); i++) {
          stream << "," << tensor.size(i);
        }
        stream << "} ]";
    }
  }
  return stream;
}

}
