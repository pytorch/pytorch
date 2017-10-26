#include "MSEMeter.h"
#include <cassert>
#include <math.h>

using namespace at;

MSEMeter::MSEMeter() {
   reset();
}

void MSEMeter::reset() {
   n_ = 0;
   val_ = .0;
}

void MSEMeter::add(Tensor& output, Tensor& target) {
   //assert(isSameSizeAs(output, output
  Tensor t = output.sub(target);
  Tensor result = t.mul(t).contiguous().toType(CPU(kDouble));
  double * data = result.data<double>();
  for(uint64_t n = 0; n < numel(result); ++n) {
    n_++;
    val_ += ( (1. / ((double)n_ - 1.) * val_) +
              ((1. /  (double)n_) * data[n]));
  }
}

void MSEMeter::value(Tensor& val) {
  //TODO: 0-dim
  val.resize_({1}).fill_(val_);
}
